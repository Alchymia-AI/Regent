"""
Phase 2: Identity fine-tuning.

Trains on conversational data to teach the model output format, tone,
and domain behavior. Uses the same base architecture and loss function
as Phase 1 but reads JSONL conversational data instead of packed tokens.

Data format (one JSON object per line):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    PYTHONPATH=. python3 scripts/train_phase2.py \
        --config configs/regent_7b.yaml \
        --train-data data/phase2/train.jsonl \
        --val-data data/phase2/val.jsonl \
        --tokenizer data/tokenizer/regent.model \
        --base-checkpoint checkpoints/base/step_50000.safetensors
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
# LR schedule
# ---------------------------------------------------------------------------

def cosine_schedule(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class ConversationIterator:
    """
    Reads JSONL conversational data, tokenizes, and yields batches.

    Each line is a JSON object with a "messages" field containing a list
    of role/content dicts. Tokenized as:
        <user>content<assistant>content[EOS]

    Sequences longer than max_seq_len are truncated. Shorter ones are padded.
    """

    def __init__(self, path: str, tokenizer: RegentTokenizer, max_seq_len: int, batch_size: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        # Load and tokenize all conversations
        self.samples: list[list[int]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids = self._tokenize_conversation(obj)
                if len(ids) > 2:  # skip empty
                    self.samples.append(ids)

        self.n_batches = len(self.samples) // batch_size
        self._order = list(range(len(self.samples)))
        self._pos = 0
        self._epoch = 0

    def _tokenize_conversation(self, obj: dict) -> list[int]:
        messages = obj.get("messages", [])
        if not messages:
            # Legacy format: single text field
            text = obj.get("text", "")
            return self.tokenizer.encode(text, add_bos=True, add_eos=True)

        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<{role}>{content}")
        text = "".join(parts)
        return self.tokenizer.encode(text, add_bos=True, add_eos=True)

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

        # Pad to max_seq_len
        batch = []
        for idx in indices:
            ids = self.samples[idx][: self.max_seq_len + 1]
            # Pad with 0 (PAD)
            ids = ids + [0] * (self.max_seq_len + 1 - len(ids))
            batch.append(ids)

        batch_np = np.array(batch, dtype=np.int32)
        input_ids = mx.array(batch_np[:, :-1])
        labels = mx.array(batch_np[:, 1:])

        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(model: RegentModel, batch: dict) -> mx.array:
    output = model(input_ids=batch["input_ids"], use_chunked=True)
    logits = output["logits"]
    labels = batch["labels"]

    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    labels_flat = labels.reshape(B * T)

    mask = (labels_flat > 0).astype(mx.float32)
    safe_labels = mx.where(labels_flat > 0, labels_flat, mx.zeros_like(labels_flat))

    per_token_loss = nn.losses.cross_entropy(logits_flat, safe_labels, reduction="none")
    masked_loss = per_token_loss * mask
    n_valid = mx.maximum(mask.sum(), mx.array(1.0))

    return masked_loss.sum() / n_valid


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

    # Phase 2 uses lower LR than Phase 1
    lr = train_cfg.lr * 0.1
    min_lr = train_cfg.min_lr * 0.1
    max_steps = train_cfg.phase2_steps if hasattr(train_cfg, "phase2_steps") and train_cfg.phase2_steps else train_cfg.max_steps // 10

    print("=" * 60)
    print("Regent Model — Phase 2: Identity Fine-tuning")
    print("=" * 60)

    # Build and load base weights
    model = RegentModel(model_cfg)
    if base_checkpoint:
        weights = mx.load(base_checkpoint)
        model.load_weights(list(weights.items()))
        print(f"Base checkpoint loaded: {base_checkpoint}")

    params = model.count_parameters()
    print(f"Parameters: {params['total_millions']}M")

    tokenizer = RegentTokenizer(tokenizer_path)

    train_iter = ConversationIterator(train_data, tokenizer, train_cfg.max_seq_len, train_cfg.batch_size)
    print(f"Train samples: {len(train_iter.samples):,}")
    print(f"Batches/epoch: {train_iter.n_batches:,}")

    val_iter = None
    if val_data:
        val_iter = ConversationIterator(val_data, tokenizer, train_cfg.max_seq_len, train_cfg.batch_size)
        print(f"Val samples: {len(val_iter.samples):,}")

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=train_cfg.weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

    ckpt_dir = Path(checkpoint_dir) / "identity"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLR: {lr} → {min_lr}")
    print(f"Max steps: {max_steps:,}")
    print("=" * 60)

    train_iter.shuffle()
    step = 0
    accum_loss = 0.0
    accum_grads = None
    accum_count = 0
    step_start = time.time()

    log_path = ckpt_dir / "train_log.jsonl"
    log_file = open(log_path, "a")

    while step < max_steps:
        batch = next(train_iter)
        loss, grads = loss_and_grad_fn(model, batch)

        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = mu.tree_map(lambda a, b: a + b, accum_grads, grads)

        accum_loss += loss.item()
        accum_count += 1

        if accum_count >= train_cfg.gradient_accumulation:
            avg_grads = mu.tree_map(lambda g: g / train_cfg.gradient_accumulation, accum_grads)

            # Clip
            grad_norm_sq = sum(
                (v * v).sum().item()
                for _, v in mu.tree_flatten(avg_grads)
                if isinstance(v, mx.array)
            )
            grad_norm = math.sqrt(grad_norm_sq)
            if grad_norm > train_cfg.grad_clip:
                scale = train_cfg.grad_clip / (grad_norm + 1e-8)
                avg_grads = mu.tree_map(lambda g: g * scale, avg_grads)

            current_lr = cosine_schedule(step, min(100, max_steps // 10), max_steps, lr, min_lr)
            optimizer.learning_rate = mx.array(current_lr)

            model.update(optimizer.apply_gradients(avg_grads, model))
            mx.eval(model.parameters(), optimizer.state)

            avg_loss = accum_loss / accum_count
            step += 1

            if step % 10 == 0:
                elapsed = time.time() - step_start
                ppl = math.exp(min(avg_loss, 20.0))
                print(f"  step {step:>6d} | loss {avg_loss:.4f} | ppl {ppl:.2f} | lr {current_lr:.2e} | gnorm {grad_norm:.3f}")
                log_file.write(json.dumps({"step": step, "loss": round(avg_loss, 4), "ppl": round(ppl, 2)}) + "\n")
                log_file.flush()
                step_start = time.time()

            if step % 500 == 0:
                save_checkpoint(model, step, avg_loss, ckpt_dir)

            accum_loss = 0.0
            accum_count = 0
            accum_grads = None

    save_checkpoint(model, step, avg_loss, ckpt_dir)
    log_file.close()
    print(f"\nPhase 2 complete. {step} steps. Checkpoints in: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Regent — Phase 2: Identity fine-tuning")
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
