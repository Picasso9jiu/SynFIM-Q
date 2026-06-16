import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def quote_cmd(cmd):
    return " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd)


def newest_checkpoint(root: Path, pattern: str, start_time: float) -> Optional[Path]:
    candidates = []
    for path in root.glob(f"checkpoints/quant_result/*/{pattern}"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= start_time - 2:
            candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def run_step(name: str, cmd, cwd: Path, log_path: Path, env: dict) -> int:
    print(f"\n[{name}]")
    print(quote_cmd(cmd))
    print(f"Log: {log_path}")
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(quote_cmd(cmd) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
        return process.wait()


def build_common_args(args):
    return [
        args.python,
        "test_quant.py",
        "--model",
        args.model,
        "--config",
        args.config,
        "--dataset",
        args.dataset,
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run 3bit experiment 2 and 3 sequentially from the same Fisher-Calib checkpoint. "
            "Experiment 2 uses fixed k/p, experiment 3 enables adaptive k/p candidate selection."
        )
    )
    parser.add_argument("--dataset", default="D:/AI/IaS-ViT-main/dataset/imagenet")
    parser.add_argument("--config", default="./configs/3bit/best.py")
    parser.add_argument("--model", default="deit_tiny")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--calib-checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--optim-size", type=int, default=1024)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--p1", type=float, default=1.0)
    parser.add_argument("--p2", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    run_tag = datetime.now().strftime("%Y%m%d_%H%M_%S_3bit_exp23_sequence")
    log_dir = root / "checkpoints" / "sequence_runs" / run_tag
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    checkpoint_pattern = f"{args.model}_w3_a3_calibsize_128_fisher_diag.pth"
    calib_checkpoint = Path(args.calib_checkpoint).resolve() if args.calib_checkpoint else None

    if calib_checkpoint is None:
        calib_cmd = build_common_args(args) + [
            "--calibrate",
            "--w_bit",
            "3",
            "--a_bit",
            "3",
            "--calib-metric",
            "fisher_diag",
            "--val-batch-size",
            str(args.val_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
        ]
        if args.dry_run:
            print("[calibrate]")
            print(quote_cmd(calib_cmd))
            calib_checkpoint = root / "checkpoints/quant_result/<timestamp>" / checkpoint_pattern
        else:
            start_time = time.time()
            code = run_step("1/3 generate shared 3bit Fisher-Calib checkpoint", calib_cmd, root, log_dir / "calibrate.log", env)
            if code != 0:
                raise SystemExit(f"Calibration failed with exit code {code}. See {log_dir / 'calibrate.log'}")
            calib_checkpoint = newest_checkpoint(root, checkpoint_pattern, start_time)
            if calib_checkpoint is None:
                raise SystemExit("Calibration finished, but the expected 3bit Fisher-Calib checkpoint was not found.")
    else:
        if not calib_checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {calib_checkpoint}")

    print(f"\nShared checkpoint: {calib_checkpoint}")

    fixed_b_cmd = build_common_args(args) + [
        "--load-calibrate-checkpoint",
        str(calib_checkpoint),
        "--optimize",
        "--w_bit",
        "3",
        "--a_bit",
        "3",
        "--calib-metric",
        "fisher_diag",
        "--optim-metric",
        "fisher_dplr",
        "--optim-size",
        str(args.optim_size),
        "--val-batch-size",
        str(args.val_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--k",
        str(args.k),
        "--p1",
        str(args.p1),
        "--p2",
        str(args.p2),
        "--no-adaptive-k",
        "--no-adaptive-p",
        "--no-adaptive-candidate-select",
        "--logit-guard",
        "--no-logit-bias-correction",
    ]

    adaptive_d_cmd = build_common_args(args) + [
        "--load-calibrate-checkpoint",
        str(calib_checkpoint),
        "--optimize",
        "--w_bit",
        "3",
        "--a_bit",
        "3",
        "--calib-metric",
        "fisher_diag",
        "--optim-metric",
        "fisher_dplr",
        "--optim-size",
        str(args.optim_size),
        "--val-batch-size",
        str(args.val_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--k",
        str(args.k),
        "--p1",
        str(args.p1),
        "--p2",
        str(args.p2),
        "--adaptive-k",
        "--adaptive-p",
        "--adaptive-candidate-select",
        "--adaptive-3bit-select-profile",
        "safe_plus",
        "--logit-guard",
        "--no-logit-bias-correction",
    ]

    print(f"Sequence logs: {log_dir}")
    if args.dry_run:
        print("[experiment 2]")
        print(quote_cmd(fixed_b_cmd))
        print("[experiment 3]")
        print(quote_cmd(adaptive_d_cmd))
        return

    code = run_step("2/3 experiment 2: Fisher-Calib + fixed k/p", fixed_b_cmd, root, log_dir / "experiment_2_fixed_kp.log", env)
    if code != 0:
        raise SystemExit(f"Experiment 2 failed with exit code {code}. See {log_dir / 'experiment_2_fixed_kp.log'}")

    code = run_step("3/3 experiment 3: Fisher-Calib + adaptive k/p", adaptive_d_cmd, root, log_dir / "experiment_3_adaptive_kp.log", env)
    if code != 0:
        raise SystemExit(f"Experiment 3 failed with exit code {code}. See {log_dir / 'experiment_3_adaptive_kp.log'}")

    print("\nExperiment 2 and 3 finished successfully.")


if __name__ == "__main__":
    main()
