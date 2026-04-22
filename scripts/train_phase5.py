"""
Phase 5: Adaptive gate calibration.

Trains the gate modules to learn when attention is needed (long-range
dependency) vs when Mamba alone is sufficient (local flow).

Strategy:
    1. Freeze the entire model except gate parameters.
    2. Run forward passes on varied-length sequences.
    3. The loss combines:
       - Standard LM loss (quality must not degrade)
       - Sparsity penalty (encourage the gate to stay closed when possible)
       - Consistency loss (gate should be stable, not flickering)

The gate learns from the signal: if closing the gate (skipping attention)
doesn't hurt the LM loss, keep it closed. If it hurts, open it.

Data format: Same as Phase 1 — packed token .npy files. The gate learns
from the model's own behavior, not from external labels.

Usage:
    PYTHONPATH=. python3 scripts/train_phase5.py \
        --config configs/regent_7b.yaml \
        --train-data data/processed/train.npy \
        --tokenizer data/tokenizer/regent.model \
        --base-checkpoint checkpoints/alignment/step_3000.safetensors

Requires: adaptive_gate: true in the model config.
"""

import argparse
import json
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
# Data iterator (reused from Phase 1)
# ---------------------------------------------------------------------------

class PackedTokenIterator:
    def __init__(self, token_file: str, seq_len: int, batch_size: int):
        self.tokens = np.load(token_file).astype(np.int32)
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.n_sequences = len(self.tokens) // (seq_len + 1)
        usable = self.n_sequences * (seq_len + 1)
        self.tokens = self.tokens[:usable]
        self.sequences = self.tokens.reshape(self.n_sequences, seq_len + 1)
        self.n_batches = self.n_sequences // batch_size

        self._order = np.arange(self.n_sequences)
        self._pos = 0
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def shuffle(self):
        np.random.shuffle(self._order)

    def reset(self):
        self._pos = 0

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
        batch_np = self.sequences[indices]

        self._pos += 1
        return {
            "input_ids": mx.array(batch_np[:, :-1]),
            "labels": mx.array(batch_np[:, 1:]),
        }


# ---------------------------------------------------------------------------
# Loss: LM quality + gate sparsity + gate consistency
# ---------------------------------------------------------------------------

def compute_gate_loss(
    model: RegentModel,
    batch: dict,
    sparsity_weight: float = 0.01,
    consistency_weight: float = 0.005,
) -> mx.array:
    """
    Combined loss:
        L = L_lm + sparsity_weight * L_sparsity + consistency_weight * L_consistency

    L_lm: standard cross-entropy (quality must not degrade)
    L_sparsity: mean gate activation (penalize unnecessary attention usage)
    L_consistency: variance of gate over sequence (penalize flickering)
    """
    output = model(input_ids=batch["input_ids"], use_chunked=True)
    logits = output["logits"]
    labels = batch["labels"]

    # LM loss
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    labels_flat = labels.reshape(B * T)
    mask = (labels_flat > 0).astype(mx.float32)
    safe_labels = mx.where(labels_flat > 0, labels_flat, mx.zeros_like(labels_flat))
    per_token_loss = nn.losses.cross_entropy(logits_flat, safe_labels, reduction="none")
    lm_loss = (per_token_loss * mask).sum() / mx.maximum(mask.sum(), mx.array(1.0))

    # Collect gate activations from adaptive layers
    gate_activations = []
    for layer in model.layers:
        if hasattr(layer, 'adaptive') and layer.adaptive:
            # Run gate on the pre-normed input to get current gate values
            h = layer.pre_norm(output["hidden"])
            gate_val = layer.gate(h)  # (batch, seq_len, 1)
            gate_activations.append(gate_val)

    if not gate_activations:
        return lm_loss

    # Stack all gate activations: (n_gates, batch, seq_len, 1)
    gates = mx.concatenate(gate_activations, axis=-1)  # (batch, seq_len, n_gates)

    # Sparsity: penalize mean gate value (encourage gates to stay closed)
    sparsity_loss = gates.mean()

    # Consistency: penalize high variance along sequence dimension (flickering)
    gate_mean = gates.mean(axis=1, keepdims=True)
    consistency_loss = ((gates - gate_mean) ** 2).mean()

    total = lm_loss + sparsity_weight * sparsity_loss + consistency_weight * consistency_loss
    return total


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
    sparsity_weight: float = 0.01,
    consistency_weight: float = 0.005,
):
    model_cfg = RegentConfig.from_yaml(config_path)
    train_cfg = TrainConfig.from_yaml(config_path)

    if not model_cfg.adaptive_gate:
        print("ERROR: adaptive_gate is not enabled in the config. Set adaptive_gate: true.")
        return

    max_steps = train_cfg.phase5_steps if hasattr(train_cfg, "phase5_steps") and train_cfg.phase5_steps else 5000

    print("=" * 60)
    print("Regent Model — Phase 5: Adaptive Gate Calibration")
    print("=" * 60)

    model = RegentModel(model_cfg)
    if base_checkpoint:
        weights = mx.load(base_checkpoint)
        model.load_weights(list(weights.items()))
        print(f"Base checkpoint loaded: {base_checkpoint}")

    # Freeze everything except gate parameters
    model.freeze()
    for layer in model.layers:
        if hasattr(layer, 'adaptive') and layer.adaptive:
            layer.gate.unfreeze()

    trainable = sum(v.size for _, v in mu.tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in mu.tree_flatten(model.parameters()))
    print(f"Total parameters: {total:,}")
    print(f"Trainable (gates only): {trainable:,} ({100*trainable/total:.4f}%)")
    print(f"Sparsity weight: {sparsity_weight}")
    print(f"Consistency weight: {consistency_weight}")

    train_iter = PackedTokenIterator(train_data, train_cfg.max_seq_len, train_cfg.batch_size)
    print(f"Train sequences: {train_iter.n_sequences:,}")
    print(f"Batches/epoch: {train_iter.n_batches:,}")

    # Higher LR for the small gate modules
    lr = 5e-4
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, batch):
        return compute_gate_loss(model, batch, sparsity_weight, consistency_weight)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    ckpt_dir = Path(checkpoint_dir) / "adaptive_gate"
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

            # Measure gate openness
            gate_means = []
            with mx.no_grad():
                dummy = model(input_ids=batch["input_ids"], use_chunked=True)
                for layer in model.layers:
                    if hasattr(layer, 'adaptive') and layer.adaptive:
                        h = layer.pre_norm(dummy["hidden"])
                        g = layer.gate(h).mean().item()
                        gate_means.append(g)

            gate_avg = sum(gate_means) / len(gate_means) if gate_means else 0.0

            print(f"  step {step:>6d} | loss {avg:.4f} | gate_open {gate_avg:.3f} | {elapsed:.1f}s")
            log_file.write(json.dumps({
                "step": step,
                "loss": round(avg, 4),
                "gate_openness": round(gate_avg, 4),
            }) + "\n")
            log_file.flush()
            running_loss = 0.0
            step_start = time.time()

        if step % 500 == 0:
            save_checkpoint(model, step, loss.item(), ckpt_dir)

    save_checkpoint(model, step, loss.item(), ckpt_dir)
    log_file.close()

    # Print final gate statistics
    print(f"\nPhase 5 complete. {step} steps.")
    print(f"Checkpoints in: {ckpt_dir}")
    print(f"\nFinal gate openness per layer:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'adaptive') and layer.adaptive:
            with mx.no_grad():
                test_input = mx.array(np.random.randint(10, model_cfg.vocab_size, (1, 128)).astype(np.int32))
                out = model(input_ids=test_input, use_chunked=True)
                h = layer.pre_norm(out["hidden"])
                g = layer.gate(h)
                print(f"  Layer {i}: mean={g.mean().item():.3f}, min={g.min().item():.3f}, max={g.max().item():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train Regent — Phase 5: Adaptive gate calibration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--consistency-weight", type=float, default=0.005)
    args = parser.parse_args()

    train(
        config_path=args.config,
        train_data=args.train_data,
        val_data=args.val_data,
        tokenizer_path=args.tokenizer,
        base_checkpoint=args.base_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        sparsity_weight=args.sparsity_weight,
        consistency_weight=args.consistency_weight,
    )


if __name__ == "__main__":
    main()
