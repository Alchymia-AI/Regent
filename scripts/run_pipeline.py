"""
Automated training pipeline — scrape data and run all 4 phases sequentially.

This is the single entry point to go from raw data to a trained Regent model.

Stages:
    1. SCRAPE   — collect corpus from configured sources
    2. TOKENIZE — train BPE tokenizer on corpus
    3. PREPARE  — pack tokens into .npy arrays
    4. PHASE 1  — base pre-training
    5. PHASE 2  — Regent identity fine-tuning
    6. PHASE 3  — verification head training
    7. PHASE 4  — DPO alignment

Each stage checks if its output already exists and skips if so (idempotent).
To force re-run a stage, delete its output or use --force-stage.

Usage:
    # Full pipeline with default config
    PYTHONPATH=. python3 scripts/run_pipeline.py --config configs/regent_370m.yaml

    # Full pipeline with scraping config
    PYTHONPATH=. python3 scripts/run_pipeline.py --config configs/regent_370m.yaml --scrape-config pipeline.yaml

    # Start from a specific stage
    PYTHONPATH=. python3 scripts/run_pipeline.py --config configs/regent_370m.yaml --start-stage 4

    # Test pipeline (tiny model, synthetic data)
    PYTHONPATH=. python3 scripts/run_pipeline.py --config configs/regent_test.yaml --synthetic
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PYTHON = sys.executable


def run_cmd(cmd: list[str], description: str, env: dict | None = None) -> bool:
    """Run a command, printing output in real-time. Returns True on success."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = "."
    if env:
        full_env.update(env)

    start = time.time()
    result = subprocess.run(cmd, env=full_env)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✓ {description} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ✗ {description} FAILED (exit code {result.returncode})")
        return False


def check_output_exists(path: str) -> bool:
    """Check if an output file/directory exists and is non-empty."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size > 0
    if p.is_dir():
        return any(p.iterdir())
    return False


def run_pipeline(
    config: str,
    scrape_config: str | None = None,
    synthetic: bool = False,
    start_stage: int = 1,
    checkpoint_dir: str = "checkpoints",
    vocab_size: int = 32768,
    phase1_steps: int | None = None,
    phase2_steps: int | None = None,
    phase3_steps: int | None = None,
    phase4_steps: int | None = None,
):
    pipeline_start = time.time()
    stages_run = 0
    stages_skipped = 0
    stages_failed = 0

    print("=" * 60)
    print("  REGENT MODEL — AUTOMATED TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Config: {config}")
    print(f"  Synthetic data: {synthetic}")
    print(f"  Start stage: {start_stage}")
    print(f"  Checkpoint dir: {checkpoint_dir}")

    # Determine paths
    tokenizer_path = "data/tokenizer/regent.model"
    train_tokens = "data/processed/train.npy"
    val_tokens = "data/processed/val.npy"
    phase2_train = "data/phase2/train.jsonl"
    phase2_val = "data/phase2/val.jsonl"
    phase3_train = "data/phase3/train.jsonl"
    phase3_val = "data/phase3/val.jsonl"
    phase4_train = "data/phase4/train.jsonl"
    phase4_val = "data/phase4/val.jsonl"

    # Determine vocab size from config
    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)
    vocab_size = cfg["model"].get("vocab_size", vocab_size)
    seq_len = cfg["training"].get("max_seq_len", 2048)

    # -----------------------------------------------------------------------
    # STAGE 1: SCRAPE / GENERATE DATA
    # -----------------------------------------------------------------------
    if start_stage <= 1:
        if synthetic:
            # Generate synthetic data for all phases
            if not check_output_exists("data/raw/train.txt"):
                ok = run_cmd(
                    [PYTHON, "scripts/generate_test_corpus.py"],
                    "Stage 1a: Generate synthetic base corpus",
                )
                if not ok:
                    stages_failed += 1
                else:
                    stages_run += 1
            else:
                print("\n  → Stage 1a: Synthetic corpus exists, skipping")
                stages_skipped += 1

            if not check_output_exists(phase2_train):
                ok = run_cmd(
                    [PYTHON, "scripts/generate_phase2_data.py",
                     "--output", phase2_train, "--count", "5000"],
                    "Stage 1b: Generate Phase 2 data",
                )
                run_cmd(
                    [PYTHON, "scripts/generate_phase2_data.py",
                     "--output", phase2_val, "--count", "500", "--seed", "99"],
                    "Stage 1b: Generate Phase 2 val data",
                )
                stages_run += 1
            else:
                stages_skipped += 1

            if not check_output_exists(phase3_train):
                run_cmd(
                    [PYTHON, "scripts/generate_phase3_data.py",
                     "--output", phase3_train, "--count", "5000"],
                    "Stage 1c: Generate Phase 3 data",
                )
                run_cmd(
                    [PYTHON, "scripts/generate_phase3_data.py",
                     "--output", phase3_val, "--count", "500", "--seed", "99"],
                    "Stage 1c: Generate Phase 3 val data",
                )
                stages_run += 1
            else:
                stages_skipped += 1

            if not check_output_exists(phase4_train):
                run_cmd(
                    [PYTHON, "scripts/generate_phase4_data.py",
                     "--output", phase4_train, "--count", "3000"],
                    "Stage 1d: Generate Phase 4 data",
                )
                run_cmd(
                    [PYTHON, "scripts/generate_phase4_data.py",
                     "--output", phase4_val, "--count", "300", "--seed", "99"],
                    "Stage 1d: Generate Phase 4 val data",
                )
                stages_run += 1
            else:
                stages_skipped += 1

        elif scrape_config:
            if not check_output_exists("data/raw/train.txt"):
                ok = run_cmd(
                    [PYTHON, "scripts/scrape_corpus.py", "--config", scrape_config],
                    "Stage 1: Scrape corpus",
                )
                if not ok:
                    stages_failed += 1
                    print("\n  PIPELINE STOPPED: scraping failed")
                    return
                stages_run += 1
            else:
                print("\n  → Stage 1: Corpus exists, skipping")
                stages_skipped += 1
        else:
            if not check_output_exists("data/raw/train.txt"):
                print("\n  ERROR: No training data found at data/raw/train.txt")
                print("  Either provide --scrape-config, --synthetic, or place data manually.")
                return
            print("\n  → Stage 1: Using existing corpus")
            stages_skipped += 1

    # -----------------------------------------------------------------------
    # STAGE 2: TRAIN TOKENIZER
    # -----------------------------------------------------------------------
    if start_stage <= 2:
        if not check_output_exists(tokenizer_path):
            ok = run_cmd(
                [PYTHON, "scripts/train_tokenizer.py",
                 "--input", "data/raw/train.txt",
                 "--output", "data/tokenizer",
                 "--vocab-size", str(vocab_size)],
                "Stage 2: Train tokenizer",
            )
            if not ok:
                stages_failed += 1
                print("\n  PIPELINE STOPPED: tokenizer training failed")
                return
            stages_run += 1
        else:
            print("\n  → Stage 2: Tokenizer exists, skipping")
            stages_skipped += 1

    # -----------------------------------------------------------------------
    # STAGE 3: PREPARE TOKEN ARRAYS
    # -----------------------------------------------------------------------
    if start_stage <= 3:
        if not check_output_exists(train_tokens):
            ok = run_cmd(
                [PYTHON, "scripts/prepare_data.py",
                 "--input", "data/raw/train.txt",
                 "--tokenizer", tokenizer_path,
                 "--output", train_tokens,
                 "--seq-len", str(seq_len)],
                "Stage 3a: Prepare training tokens",
            )
            if not ok:
                stages_failed += 1
                print("\n  PIPELINE STOPPED: data preparation failed")
                return
            stages_run += 1
        else:
            stages_skipped += 1

        if check_output_exists("data/raw/val.txt") and not check_output_exists(val_tokens):
            run_cmd(
                [PYTHON, "scripts/prepare_data.py",
                 "--input", "data/raw/val.txt",
                 "--tokenizer", tokenizer_path,
                 "--output", val_tokens,
                 "--seq-len", str(seq_len)],
                "Stage 3b: Prepare validation tokens",
            )
            stages_run += 1
        else:
            stages_skipped += 1

    # -----------------------------------------------------------------------
    # STAGE 4: PHASE 1 — BASE PRE-TRAINING
    # -----------------------------------------------------------------------
    phase1_ckpt = None
    if start_stage <= 4:
        # Find latest Phase 1 checkpoint
        phase1_dir = Path(checkpoint_dir) / "base"
        phase1_ckpt = _find_latest_checkpoint(phase1_dir)

        if phase1_ckpt is None:
            cmd = [
                PYTHON, "scripts/train.py",
                "--config", config,
                "--train-data", train_tokens,
            ]
            if check_output_exists(val_tokens):
                cmd.extend(["--val-data", val_tokens])

            ok = run_cmd(cmd, "Stage 4: Phase 1 — Base pre-training")
            if not ok:
                stages_failed += 1
                print("\n  PIPELINE STOPPED: Phase 1 failed")
                return
            stages_run += 1
            phase1_ckpt = _find_latest_checkpoint(phase1_dir)
        else:
            print(f"\n  → Stage 4: Phase 1 checkpoint exists: {phase1_ckpt}")
            stages_skipped += 1

    # -----------------------------------------------------------------------
    # STAGE 5: PHASE 2 — IDENTITY FINE-TUNING
    # -----------------------------------------------------------------------
    phase2_ckpt = None
    if start_stage <= 5 and check_output_exists(phase2_train):
        phase2_dir = Path(checkpoint_dir) / "identity"
        phase2_ckpt = _find_latest_checkpoint(phase2_dir)

        if phase2_ckpt is None:
            cmd = [
                PYTHON, "scripts/train_phase2.py",
                "--config", config,
                "--train-data", phase2_train,
                "--tokenizer", tokenizer_path,
            ]
            if check_output_exists(phase2_val):
                cmd.extend(["--val-data", phase2_val])
            if phase1_ckpt:
                cmd.extend(["--base-checkpoint", phase1_ckpt])

            ok = run_cmd(cmd, "Stage 5: Phase 2 — Identity fine-tuning")
            if not ok:
                stages_failed += 1
                print("\n  PIPELINE STOPPED: Phase 2 failed")
                return
            stages_run += 1
            phase2_ckpt = _find_latest_checkpoint(phase2_dir)
        else:
            print(f"\n  → Stage 5: Phase 2 checkpoint exists: {phase2_ckpt}")
            stages_skipped += 1
    elif start_stage <= 5:
        print("\n  → Stage 5: No Phase 2 data, skipping")
        phase2_ckpt = phase1_ckpt
        stages_skipped += 1

    # -----------------------------------------------------------------------
    # STAGE 6: PHASE 3 — VERIFICATION HEAD
    # -----------------------------------------------------------------------
    phase3_ckpt = None
    if start_stage <= 6 and check_output_exists(phase3_train):
        phase3_dir = Path(checkpoint_dir) / "verification"
        phase3_ckpt = _find_latest_checkpoint(phase3_dir)

        if phase3_ckpt is None:
            base_ckpt = phase2_ckpt or phase1_ckpt
            cmd = [
                PYTHON, "scripts/train_phase3.py",
                "--config", config,
                "--train-data", phase3_train,
                "--tokenizer", tokenizer_path,
            ]
            if check_output_exists(phase3_val):
                cmd.extend(["--val-data", phase3_val])
            if base_ckpt:
                cmd.extend(["--base-checkpoint", base_ckpt])

            ok = run_cmd(cmd, "Stage 6: Phase 3 — Verification head")
            if not ok:
                stages_failed += 1
                print("\n  PIPELINE STOPPED: Phase 3 failed")
                return
            stages_run += 1
            phase3_ckpt = _find_latest_checkpoint(phase3_dir)
        else:
            print(f"\n  → Stage 6: Phase 3 checkpoint exists: {phase3_ckpt}")
            stages_skipped += 1
    elif start_stage <= 6:
        print("\n  → Stage 6: No Phase 3 data, skipping")
        phase3_ckpt = phase2_ckpt or phase1_ckpt
        stages_skipped += 1

    # -----------------------------------------------------------------------
    # STAGE 7: PHASE 4 — DPO ALIGNMENT
    # -----------------------------------------------------------------------
    if start_stage <= 7 and check_output_exists(phase4_train):
        phase4_dir = Path(checkpoint_dir) / "alignment"
        phase4_ckpt = _find_latest_checkpoint(phase4_dir)

        if phase4_ckpt is None:
            base_ckpt = phase3_ckpt or phase2_ckpt or phase1_ckpt
            cmd = [
                PYTHON, "scripts/train_phase4.py",
                "--config", config,
                "--train-data", phase4_train,
                "--tokenizer", tokenizer_path,
            ]
            if check_output_exists(phase4_val):
                cmd.extend(["--val-data", phase4_val])
            if base_ckpt:
                cmd.extend(["--base-checkpoint", base_ckpt])

            ok = run_cmd(cmd, "Stage 7: Phase 4 — DPO alignment")
            if not ok:
                stages_failed += 1
            else:
                stages_run += 1
        else:
            print(f"\n  → Stage 7: Phase 4 checkpoint exists: {phase4_ckpt}")
            stages_skipped += 1
    elif start_stage <= 7:
        print("\n  → Stage 7: No Phase 4 data, skipping")
        stages_skipped += 1

    # -----------------------------------------------------------------------
    # STAGE 8: PHASE 5 — ADAPTIVE GATE CALIBRATION
    # -----------------------------------------------------------------------
    if start_stage <= 8:
        # Only runs if adaptive_gate is enabled in the config
        # Uses the same training data as Phase 1 (packed tokens)
        phase5_dir = Path(checkpoint_dir) / "adaptive_gate"
        phase5_ckpt = _find_latest_checkpoint(phase5_dir)

        if phase5_ckpt is None:
            base_ckpt = (
                _find_latest_checkpoint(Path(checkpoint_dir) / "alignment")
                or phase3_ckpt or phase2_ckpt or phase1_ckpt
            )
            train_npy = "data/processed/train.npy"

            if check_output_exists(train_npy) and base_ckpt:
                cmd = [
                    PYTHON, "scripts/train_phase5.py",
                    "--config", config,
                    "--train-data", train_npy,
                    "--tokenizer", tokenizer_path,
                    "--base-checkpoint", base_ckpt,
                ]
                ok = run_cmd(cmd, "Stage 8: Phase 5 — Adaptive gate calibration")
                if not ok:
                    stages_failed += 1
                else:
                    stages_run += 1
            else:
                print("\n  → Stage 8: No training data or base checkpoint, skipping")
                stages_skipped += 1
        else:
            print(f"\n  → Stage 8: Phase 5 checkpoint exists: {phase5_ckpt}")
            stages_skipped += 1
    elif start_stage > 8:
        stages_skipped += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Stages run: {stages_run}")
    print(f"  Stages skipped: {stages_skipped}")
    print(f"  Stages failed: {stages_failed}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")

    # Find final checkpoint
    final = (
        _find_latest_checkpoint(Path(checkpoint_dir) / "adaptive_gate")
        or _find_latest_checkpoint(Path(checkpoint_dir) / "alignment")
        or _find_latest_checkpoint(Path(checkpoint_dir) / "verification")
        or _find_latest_checkpoint(Path(checkpoint_dir) / "identity")
        or _find_latest_checkpoint(Path(checkpoint_dir) / "base")
    )
    if final:
        print(f"\n  Final model: {final}")
        print(f"  Serve with:")
        print(f"    PYTHONPATH=. python3 -m serve.server --config {config} --model {final}")
    print("=" * 60)


def _find_latest_checkpoint(ckpt_dir: Path) -> str | None:
    """Find the latest .safetensors checkpoint in a directory."""
    if not ckpt_dir.exists():
        return None

    checkpoints = sorted(ckpt_dir.glob("step_*.safetensors"))
    if not checkpoints:
        return None

    # Sort by step number
    def step_num(p):
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    checkpoints.sort(key=step_num)
    return str(checkpoints[-1])


def main():
    parser = argparse.ArgumentParser(description="Regent Model — Automated Training Pipeline")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--scrape-config", default=None, help="Scraping pipeline YAML")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for all phases")
    parser.add_argument("--start-stage", type=int, default=1, choices=range(1, 9),
                        help="Start from stage N (1=scrape, 4=phase1, 5=phase2, 6=phase3, 7=phase4, 8=phase5-gate)")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    run_pipeline(
        config=args.config,
        scrape_config=args.scrape_config,
        synthetic=args.synthetic,
        start_stage=args.start_stage,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
