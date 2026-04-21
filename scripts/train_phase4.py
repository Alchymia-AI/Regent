"""
Phase 4: DPO alignment.

Direct Preference Optimization. Trains the model to prefer chosen responses
over rejected ones, using a frozen reference copy of the model.

Data format (one JSON object per line):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
    PYTHONPATH=. python3 scripts/train_phase4.py \
        --config configs/regent_7b.yaml \
        --train-data data/phase4/train.jsonl \
        --tokenizer data/tokenizer/regent.model \
        --base-checkpoint checkpoints/verification/step_5000.safetensors
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mu
import numpy as np

from regent_model.layers.model import RegentModel, RegentConfig
from regent_model.utils.config import TrainConfig
from regent_model.utils.tokenizer import RegentTokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class PreferencePairIterator:
    """
    Reads preference pair JSONL. Each line has prompt, chosen, and rejected.
    Tokenizes both (prompt + chosen) and (prompt + rejected) for DPO.
    """

    def __init__(self, path: str, tokenizer: RegentTokenizer, max_seq_len: int, batch_size: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.samples: list[tuple[list[int], list[int]]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                chosen = obj.get("chosen", "")
                rejected = obj.get("rejected", "")

                chosen_text = f"<user>{prompt}<assistant>{chosen}"
                rejected_text = f"<user>{prompt}<assistant>{rejected}"

                chosen_ids = tokenizer.encode(chosen_text, add_bos=True, add_eos=True)
                rejected_ids = tokenizer.encode(rejected_text, add_bos=True, add_eos=True)

                if len(chosen_ids) > 2 and len(rejected_ids) > 2:
                    self.samples.append((chosen_ids, rejected_ids))

        self.n_batches = len(self.samples) // batch_size
        self._order = list(range(len(self.samples)))
        self._pos = 0
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def shuffle(self):
        random.shuffle(self._order)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._pos >= self.n_batches:
            self._epoch += 1
            self._pos = 0
            self.shuffle()

        start = self._pos * self.batch_size
        end = start + self.batch_size
        indices = self._order[start:end]
        self._pos += 1

        chosen_batch = []
        rejected_batch = []
        for idx in indices:
            c_ids, r_ids = self.samples[idx]
            c_ids = c_ids[: self.max_seq_len + 1]
            r_ids = r_ids[: self.max_seq_len + 1]
            c_ids = c_ids + [0] * (self.max_seq_len + 1 - len(c_ids))
            r_ids = r_ids + [0] * (self.max_seq_len + 1 - len(r_ids))
            chosen_batch.append(c_ids)
            rejected_batch.append(r_ids)

        chosen_np = np.array(chosen_batch, dtype=np.int32)
        rejected_np = np.array(rejected_batch, dtype=np.int32)

        return {
            "chosen_input": mx.array(chosen_np[:, :-1]),
            "chosen_labels": mx.array(chosen_np[:, 1:]),
            "rejected_input": mx.array(rejected_np[:, :-1]),
            "rejected_labels": mx.array(rejected_np[:, 1:]),
        }


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def _log_probs(model: RegentModel, input_ids: mx.array, labels: mx.array) -> mx.array:
    """Compute per-sequence log probability under the model."""
    output = model(input_ids=input_ids, use_chunked=True)
    logits = output["logits"]  # (B, T, V)

    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    labels_flat = labels.reshape(B * T)

    mask = (labels_flat > 0).astype(mx.float32)
    safe_labels = mx.where(labels_flat > 0, labels_flat, mx.zeros_like(labels_flat))

    per_token_loss = nn.losses.cross_entropy(logits_flat, safe_labels, reduction="none")
    per_token_logp = -per_token_loss  # log prob = -cross_entropy

    # Sum log probs per sequence
    per_token_logp = per_token_logp.reshape(B, T) * mask.reshape(B, T)
    seq_logp = per_token_logp.sum(axis=-1)  # (B,)

    return seq_logp


def compute_dpo_loss(
    model: RegentModel,
    ref_model: RegentModel,
    batch: dict,
    beta: float = 0.1,
) -> mx.array:
    """
    DPO loss: -log(sigmoid(beta * (log_pi(chosen)/log_pi(rejected) - log_ref(chosen)/log_ref(rejected))))
    """
    # Policy log probs
    pi_chosen = _log_probs(model, batch["chosen_input"], batch["chosen_labels"])
    pi_rejected = _log_probs(model, batch["rejected_input"], batch["rejected_labels"])

    # Reference log probs (no grad)
    ref_chosen = mx.stop_gradient(_log_probs(ref_model, batch["chosen_input"], batch["chosen_labels"]))
    ref_rejected = mx.stop_gradient(_log_probs(ref_model, batch["rejected_input"], batch["rejected_labels"]))

    # DPO
    pi_diff = pi_chosen - pi_rejected
    ref_diff = ref_chosen - ref_rejected
    logits = beta * (pi_diff - ref_diff)

    loss = -nn.activations.log_sigmoid(logits).mean()
    return loss


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model: RegentModel, step: int, loss: float, ckpt_dir: Path):
    ckpt_path = ckpt_dir / f"step_{step}.safetensors"
    model.save_weights(str(ckpt_path))
    state_path = ckpt_dir / f"step_{step}_state.json"
    with open(state_path, "w") as f:
        json.dump({"step": step, "loss": loss}, f)
    print(f"  Checkpoint saved: {ckpt_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    config_path: str,
    train_data: str,
    val_data: str | None,
    tokenizer_path: str,
    base_checkpoint: str | None,
    checkpoint_dir: str = "checkpoints",
    beta: float = 0.1,
):
    model_cfg = RegentConfig.from_yaml(config_path)
    train_cfg = TrainConfig.from_yaml(config_path)

    max_steps = train_cfg.phase4_steps if hasattr(train_cfg, "phase4_steps") and train_cfg.phase4_steps else 3000

    print("=" * 60)
    print("Regent Model — Phase 4: DPO Alignment")
    print("=" * 60)

    # Policy model (will be trained)
    model = RegentModel(model_cfg)
    if base_checkpoint:
        weights = mx.load(base_checkpoint)
        model.load_weights(list(weights.items()))
        print(f"Base checkpoint loaded: {base_checkpoint}")

    # Reference model (frozen copy)
    ref_model = RegentModel(model_cfg)
    if base_checkpoint:
        ref_model.load_weights(list(mx.load(base_checkpoint).items()))
    ref_model.freeze()

    params = model.count_parameters()
    print(f"Parameters: {params['total_millions']}M")
    print(f"DPO beta: {beta}")

    tokenizer = RegentTokenizer(tokenizer_path)

    train_iter = PreferencePairIterator(train_data, tokenizer, train_cfg.max_seq_len, train_cfg.batch_size)
    print(f"Train pairs: {len(train_iter.samples):,}")

    # Lower LR for alignment
    lr = train_cfg.lr * 0.05
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=train_cfg.weight_decay)

    # Wrap loss to match nn.value_and_grad signature
    def loss_fn(model, batch):
        return compute_dpo_loss(model, ref_model, batch, beta=beta)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    ckpt_dir = Path(checkpoint_dir) / "alignment"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"LR: {lr}")
    print(f"Max steps: {max_steps:,}")
    print("=" * 60)

    train_iter.shuffle()
    step = 0
    running_loss = 0.0
    step_start = time.time()

    log_path = ckpt_dir / "train_log.jsonl"
    log_file = open(log_path, "a")

    while step < max_steps:
        batch = next(train_iter)
        loss, grads = loss_and_grad_fn(model, batch)

        # Clip
        grad_norm_sq = sum(
            (v * v).sum().item()
            for _, v in mu.tree_flatten(grads)
            if isinstance(v, mx.array)
        )
        grad_norm = math.sqrt(grad_norm_sq)
        if grad_norm > train_cfg.grad_clip:
            scale = train_cfg.grad_clip / (grad_norm + 1e-8)
            grads = mu.tree_map(lambda g: g * scale, grads)

        model.update(optimizer.apply_gradients(grads, model))
        mx.eval(model.parameters(), optimizer.state)

        running_loss += loss.item()
        step += 1

        if step % 10 == 0:
            avg = running_loss / 10
            elapsed = time.time() - step_start
            print(f"  step {step:>6d} | loss {avg:.4f} | gnorm {grad_norm:.3f} | {elapsed:.1f}s")
            log_file.write(json.dumps({"step": step, "loss": round(avg, 4), "grad_norm": round(grad_norm, 3)}) + "\n")
            log_file.flush()
            running_loss = 0.0
            step_start = time.time()

        if step % 500 == 0:
            save_checkpoint(model, step, loss.item(), ckpt_dir)

    save_checkpoint(model, step, loss.item(), ckpt_dir)
    log_file.close()
    print(f"\nPhase 4 complete. {step} steps. Checkpoints in: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Regent — Phase 4: DPO alignment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    train(
        config_path=args.config,
        train_data=args.train_data,
        val_data=args.val_data,
        tokenizer_path=args.tokenizer,
        base_checkpoint=args.base_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
