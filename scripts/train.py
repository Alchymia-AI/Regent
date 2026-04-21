"""
Regent model training script — Phase 1 (base pre-training).

Usage:
    # Full pipeline:
    python scripts/train_tokenizer.py --input data/raw/ --vocab-size 32768
    python scripts/prepare_data.py --input data/raw/train.txt --tokenizer data/tokenizer/regent.model --output data/processed/train.npy
    python scripts/prepare_data.py --input data/raw/val.txt --tokenizer data/tokenizer/regent.model --output data/processed/val.npy
    PYTHONPATH=. python scripts/train.py --config configs/regent_370m.yaml --train-data data/processed/train.npy --val-data data/processed/val.npy

    # Resume from checkpoint:
    PYTHONPATH=. python scripts/train.py --config configs/regent_370m.yaml --train-data data/processed/train.npy --resume checkpoints/base/step_5000.safetensors
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mu
import numpy as np

from regent_model.layers.model import RegentModel, RegentConfig
from regent_model.utils.config import TrainConfig


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def cosine_schedule(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Cosine learning rate with linear warmup."""
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Data iterator
# ---------------------------------------------------------------------------

class PackedTokenIterator:
    """
    Iterates over a packed .npy token array, yielding (input_ids, labels) batches.

    The token array is a flat 1D array of token IDs. We slice it into
    non-overlapping windows of seq_len, then batch them.

    On each epoch, we reshuffle the sequence order (not the tokens within
    a sequence — that would destroy language structure).
    """

    def __init__(self, token_file: str, seq_len: int, batch_size: int):
        self.tokens = np.load(token_file).astype(np.int32)
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Number of complete sequences (each seq_len + 1 for the label shift)
        self.n_sequences = len(self.tokens) // (seq_len + 1)
        # Trim to exact fit
        usable = self.n_sequences * (seq_len + 1)
        self.tokens = self.tokens[:usable]

        # Reshape into (n_sequences, seq_len + 1)
        self.sequences = self.tokens.reshape(self.n_sequences, seq_len + 1)
        self.n_batches = self.n_sequences // batch_size

        self._order = np.arange(self.n_sequences)
        self._pos = 0
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def tokens_per_epoch(self) -> int:
        return self.n_batches * self.batch_size * self.seq_len

    def shuffle(self):
        np.random.shuffle(self._order)

    def reset(self):
        self._pos = 0

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._pos >= self.n_batches:
            # Epoch finished — reshuffle and reset
            self._epoch += 1
            self._pos = 0
            self.shuffle()

        start = self._pos * self.batch_size
        end = start + self.batch_size
        indices = self._order[start:end]

        batch_np = self.sequences[indices]  # (batch_size, seq_len + 1)

        input_ids = mx.array(batch_np[:, :-1])  # (batch_size, seq_len)
        labels = mx.array(batch_np[:, 1:])       # (batch_size, seq_len)

        self._pos += 1

        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_lm_loss(model: RegentModel, batch: dict) -> mx.array:
    """Cross-entropy loss for next-token prediction."""
    output = model(
        input_ids=batch["input_ids"],
        use_chunked=True,
    )

    logits = output["logits"]  # (B, T, V)
    labels = batch["labels"]   # (B, T)

    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    labels_flat = labels.reshape(B * T)

    # Mask padding tokens (labels == -100 or labels == 0 for PAD)
    mask = (labels_flat > 0).astype(mx.float32)
    safe_labels = mx.where(labels_flat > 0, labels_flat, mx.zeros_like(labels_flat))

    per_token_loss = nn.losses.cross_entropy(logits_flat, safe_labels, reduction="none")
    masked_loss = per_token_loss * mask
    n_valid = mx.maximum(mask.sum(), mx.array(1.0))

    return masked_loss.sum() / n_valid


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model: RegentModel, optimizer, step: int, loss: float, ckpt_dir: Path):
    """Save model weights and training state."""
    ckpt_path = ckpt_dir / f"step_{step}.safetensors"
    model.save_weights(str(ckpt_path))

    # Save training state (step, loss) as JSON alongside
    state_path = ckpt_dir / f"step_{step}_state.json"
    state = {"step": step, "loss": loss}
    with open(state_path, "w") as f:
        json.dump(state, f)

    print(f"  Checkpoint saved: {ckpt_path}")


def load_checkpoint(model: RegentModel, path: str) -> int:
    """Load model weights. Returns the step number if available."""
    weights = mx.load(path)
    model.load_weights(list(weights.items()))

    # Try to load training state
    state_path = Path(path).with_name(Path(path).stem + "_state.json")
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        return state.get("step", 0)
    return 0


# ---------------------------------------------------------------------------

def run_validation(model: RegentModel, val_iter: PackedTokenIterator, max_batches: int = 50) -> dict:
    """Run validation and return metrics."""
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in val_iter:
        if n_batches >= max_batches:
            break

        output = model(input_ids=batch["input_ids"], use_chunked=True)
        logits = output["logits"]
        labels = batch["labels"]

        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        labels_flat = labels.reshape(B * T)

        mask = (labels_flat > 0).astype(mx.float32)
        safe_labels = mx.where(labels_flat > 0, labels_flat, mx.zeros_like(labels_flat))

        per_token_loss = nn.losses.cross_entropy(logits_flat, safe_labels, reduction="none")
        batch_loss = (per_token_loss * mask).sum()
        batch_tokens = mask.sum()

        mx.eval(batch_loss, batch_tokens)

        total_loss += batch_loss.item()
        total_tokens += batch_tokens.item()
        n_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))

    return {"val_loss": round(avg_loss, 4), "val_ppl": round(ppl, 2), "val_batches": n_batches}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    config_path: str,
    train_data: str,
    val_data: str | None = None,
    checkpoint_dir: str = "checkpoints",
    resume_from: str | None = None,
    log_interval: int = 10,
    save_interval: int = 1000,
    val_interval: int = 500,
    max_val_batches: int = 50,
):
    model_cfg = RegentConfig.from_yaml(config_path)
    train_cfg = TrainConfig.from_yaml(config_path)

    print("=" * 60)
    print("Regent Model — Phase 1: Base Pre-training")
    print("=" * 60)
    print(f"Model: d_model={model_cfg.d_model}, n_layer={model_cfg.n_layer}")
    print(f"SSM: d_state={model_cfg.ssm_d_state}, n_heads={model_cfg.ssm_n_heads}")
    print(f"Attention layers: {model_cfg.attn_layers}")

    model = RegentModel(model_cfg)
    params = model.count_parameters()
    print(f"Parameters: {params['total_millions']}M")

    start_step = 0
    if resume_from:
        print(f"Resuming from {resume_from}")
        start_step = load_checkpoint(model, resume_from)
        print(f"  Resuming at step {start_step}")

    print(f"\nTrain data: {train_data}")
    train_iter = PackedTokenIterator(train_data, train_cfg.max_seq_len, train_cfg.batch_size)
    print(f"  Sequences: {train_iter.n_sequences:,}")
    print(f"  Batches/epoch: {train_iter.n_batches:,}")
    print(f"  Tokens/epoch: {train_iter.tokens_per_epoch:,}")

    val_iter = None
    if val_data:
        print(f"Val data: {val_data}")
        val_iter = PackedTokenIterator(val_data, train_cfg.max_seq_len, train_cfg.batch_size)
        print(f"  Sequences: {val_iter.n_sequences:,}")

    optimizer = optim.AdamW(learning_rate=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    loss_and_grad_fn = nn.value_and_grad(model, compute_lm_loss)

    ckpt_dir = Path(checkpoint_dir) / "base"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eff_batch = train_cfg.batch_size * train_cfg.gradient_accumulation
    max_steps = train_cfg.max_steps

    print(f"\nTraining config:")
    print(f"  Batch size: {train_cfg.batch_size} x {train_cfg.gradient_accumulation} accum = {eff_batch} effective")
    print(f"  Sequence length: {train_cfg.max_seq_len}")
    print(f"  LR: {train_cfg.lr} → {train_cfg.min_lr} (cosine, {train_cfg.warmup_steps} warmup)")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Grad clip: {train_cfg.grad_clip}")
    print(f"  Log every: {log_interval} steps")
    print(f"  Save every: {save_interval} steps")
    if val_iter:
        print(f"  Val every: {val_interval} steps")
    print("=" * 60)

    train_iter.shuffle()

    step = start_step
    accum_loss = 0.0
    accum_grads = None
    accum_count = 0
    best_val_loss = float("inf")
    step_start = time.time()

    log_path = ckpt_dir / "train_log.jsonl"
    log_file = open(log_path, "a")

    print(f"\nStarting training from step {step}...\n")

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
            avg_grads = mu.tree_map(
                lambda g: g / train_cfg.gradient_accumulation, accum_grads
            )

            grad_norm_sq = sum(
                (v * v).sum().item()
                for _, v in mu.tree_flatten(avg_grads)
                if isinstance(v, mx.array)
            )
            grad_norm = math.sqrt(grad_norm_sq)

            if grad_norm > train_cfg.grad_clip:
                scale = train_cfg.grad_clip / (grad_norm + 1e-8)
                avg_grads = mu.tree_map(lambda g: g * scale, avg_grads)

            lr = cosine_schedule(step, train_cfg.warmup_steps, max_steps, train_cfg.lr, train_cfg.min_lr)
            optimizer.learning_rate = mx.array(lr)

            model.update(optimizer.apply_gradients(avg_grads, model))
            mx.eval(model.parameters(), optimizer.state)

            avg_loss = accum_loss / accum_count

            step += 1

            if step % log_interval == 0:
                elapsed = time.time() - step_start
                tokens_per_sec = (eff_batch * train_cfg.max_seq_len * log_interval) / max(elapsed, 1e-6)
                ppl = math.exp(min(avg_loss, 20.0))

                log_entry = {
                    "step": step,
                    "loss": round(avg_loss, 4),
                    "ppl": round(ppl, 2),
                    "lr": round(lr, 8),
                    "grad_norm": round(grad_norm, 4),
                    "tok/s": round(tokens_per_sec),
                    "epoch": train_iter.epoch,
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()

                print(
                    f"  step {step:>6d} | loss {avg_loss:.4f} | ppl {ppl:>8.2f} | "
                    f"lr {lr:.2e} | gnorm {grad_norm:.3f} | "
                    f"{tokens_per_sec:,.0f} tok/s | epoch {train_iter.epoch}"
                )
                step_start = time.time()

            if val_iter and step % val_interval == 0:
                val_metrics = run_validation(model, val_iter, max_val_batches)
                print(f"  --- VAL step {step}: loss={val_metrics['val_loss']}, ppl={val_metrics['val_ppl']} ---")

                log_file.write(json.dumps({"step": step, **val_metrics}) + "\n")
                log_file.flush()

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    save_checkpoint(model, optimizer, step, val_metrics["val_loss"], ckpt_dir)
                    print(f"  New best val loss: {best_val_loss}")

                # Reset val iterator position
                val_iter.reset()

            if step % save_interval == 0:
                save_checkpoint(model, optimizer, step, avg_loss, ckpt_dir)

            accum_loss = 0.0
            accum_count = 0
            accum_grads = None

    save_checkpoint(model, optimizer, step, avg_loss, ckpt_dir)
    log_file.close()

    print("\n" + "=" * 60)
    print(f"Training complete. {step} steps.")
    print(f"Checkpoints in: {ckpt_dir}")
    print(f"Log: {log_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train Regent model — Phase 1")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--train-data", required=True, help="Training data (.npy packed tokens)")
    parser.add_argument("--val-data", default=None, help="Validation data (.npy packed tokens)")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint (.safetensors)")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--val-interval", type=int, default=500)
    args = parser.parse_args()

    train(
        config_path=args.config,
        train_data=args.train_data,
        val_data=args.val_data,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
    )


if __name__ == "__main__":
    main()
