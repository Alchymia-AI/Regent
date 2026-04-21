"""
Phase 3: Verification head training.

Freezes the backbone and trains only the verification head on labeled
grounding pairs. The head learns to output high scores for grounded claims
and low scores for fabricated ones.

Data format (one JSON object per line):
    {"text": "Aspirin reduces fever.", "label": 1, "source": "pharmacology_node_id"}
    {"text": "Aspirin cures diabetes.", "label": 0, "source": null}

label 1 = grounded (traceable to a knowledge source)
label 0 = fabricated or contradicted

Usage:
    PYTHONPATH=. python3 scripts/train_phase3.py \
        --config configs/regent_7b.yaml \
        --train-data data/phase3/train.jsonl \
        --tokenizer data/tokenizer/regent.model \
        --base-checkpoint checkpoints/identity/step_5000.safetensors
"""

import argparse
import json
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

class GroundingPairIterator:
    """
    Reads grounding pair JSONL. Each sample is tokenized text with a binary label.
    The model runs a forward pass and the verification head output is compared
    to the label via binary cross-entropy.
    """

    def __init__(self, path: str, tokenizer: RegentTokenizer, max_seq_len: int, batch_size: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.samples: list[tuple[list[int], float]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                label = float(obj.get("label", 0))
                ids = tokenizer.encode(text, add_bos=True, add_eos=False)
                if len(ids) > 1:
                    self.samples.append((ids, label))

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

        batch_ids = []
        batch_labels = []
        for idx in indices:
            ids, label = self.samples[idx]
            ids = ids[: self.max_seq_len]
            ids = ids + [0] * (self.max_seq_len - len(ids))
            batch_ids.append(ids)
            batch_labels.append(label)

        input_ids = mx.array(np.array(batch_ids, dtype=np.int32))
        labels = mx.array(np.array(batch_labels, dtype=np.float32))

        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Loss: binary cross-entropy on mean grounding score vs label
# ---------------------------------------------------------------------------

def compute_ver_loss(model: RegentModel, batch: dict) -> mx.array:
    output = model(input_ids=batch["input_ids"], use_chunked=False)

    grounding = output.get("grounding")
    if grounding is None:
        return mx.array(0.0)

    # grounding: (B, T) — take mean across non-padding positions
    mask = (batch["input_ids"] > 0).astype(mx.float32)
    masked_grounding = grounding * mask
    mean_score = masked_grounding.sum(axis=-1) / mx.maximum(mask.sum(axis=-1), mx.array(1.0))

    # Binary cross-entropy: label is 0 or 1
    labels = batch["labels"]  # (B,)
    eps = 1e-7
    mean_score = mx.clip(mean_score, eps, 1.0 - eps)
    bce = -(labels * mx.log(mean_score) + (1.0 - labels) * mx.log(1.0 - mean_score))

    return bce.mean()


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
):
    model_cfg = RegentConfig.from_yaml(config_path)
    train_cfg = TrainConfig.from_yaml(config_path)

    max_steps = train_cfg.phase3_steps if hasattr(train_cfg, "phase3_steps") and train_cfg.phase3_steps else 5000

    print("=" * 60)
    print("Regent Model — Phase 3: Verification Head Training")
    print("=" * 60)

    model = RegentModel(model_cfg)
    if base_checkpoint:
        weights = mx.load(base_checkpoint)
        model.load_weights(list(weights.items()))
        print(f"Base checkpoint loaded: {base_checkpoint}")

    # Freeze everything except the verification head
    model.freeze()
    model.ver_head.unfreeze()

    trainable = sum(v.size for _, v in mu.tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in mu.tree_flatten(model.parameters()))
    print(f"Total parameters: {total:,}")
    print(f"Trainable (ver_head only): {trainable:,} ({100*trainable/total:.3f}%)")

    tokenizer = RegentTokenizer(tokenizer_path)

    train_iter = GroundingPairIterator(train_data, tokenizer, train_cfg.max_seq_len, train_cfg.batch_size)
    print(f"Train samples: {len(train_iter.samples):,}")

    val_iter = None
    if val_data:
        val_iter = GroundingPairIterator(val_data, tokenizer, train_cfg.max_seq_len, train_cfg.batch_size)
        print(f"Val samples: {len(val_iter.samples):,}")

    # Higher LR for the small head
    lr = 1e-3
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad_fn = nn.value_and_grad(model, compute_ver_loss)

    ckpt_dir = Path(checkpoint_dir) / "verification"
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

        model.update(optimizer.apply_gradients(grads, model))
        mx.eval(model.parameters(), optimizer.state)

        running_loss += loss.item()
        step += 1

        if step % 10 == 0:
            avg = running_loss / 10
            elapsed = time.time() - step_start
            print(f"  step {step:>6d} | loss {avg:.4f} | {elapsed:.1f}s")
            log_file.write(json.dumps({"step": step, "loss": round(avg, 4)}) + "\n")
            log_file.flush()
            running_loss = 0.0
            step_start = time.time()

        if step % 500 == 0:
            save_checkpoint(model, step, loss.item(), ckpt_dir)

    save_checkpoint(model, step, loss.item(), ckpt_dir)
    log_file.close()
    print(f"\nPhase 3 complete. {step} steps. Checkpoints in: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Regent — Phase 3: Verification head")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    train(
        config_path=args.config,
        train_data=args.train_data,
        val_data=args.val_data,
        tokenizer_path=args.tokenizer,
        base_checkpoint=args.base_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
