"""CIFAR-10 + MobileNetV2: float training, PTQ, and (staged) QAT with optional KD.

    python train.py float --epochs 200
    python train.py ptq --ckpt runs/float/best.pt --w-bits 8 --a-bits 8 --per-channel
    python train.py qat --ckpt runs/float/best.pt --w-bits 4 --a-bits 8 --schedule staged --kd

torch is imported inside the run functions so --help works without it.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--name", help="run directory name")
    p.add_argument("--subset", type=int, default=0, help="training subset size for smoke tests")
    p.add_argument("--eval-subset", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after-epoch", type=int, default=0)


def add_quant(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", type=Path, required=True, help="float checkpoint to start from")
    p.add_argument("--w-bits", type=int, default=4)
    p.add_argument("--a-bits", type=int, default=8)
    p.add_argument("--weight-quantizer", choices=["uniform", "apot"], default="uniform")
    p.add_argument("--rounding", choices=["nearest", "floor"], default="nearest", help="activation rounding")
    p.add_argument("--per-channel", action="store_true")
    p.add_argument("--clip", choices=["minmax", "mse"], default="mse")
    p.add_argument("--clip-candidates", type=int, default=100)


def run_float(args: argparse.Namespace) -> None:
    from quant_pitfalls.training import train
    train(args)


def run_ptq(args: argparse.Namespace) -> None:
    from quant_pitfalls.training import ptq
    ptq(args)


def run_qat(args: argparse.Namespace) -> None:
    from quant_pitfalls.training import train
    train(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("float", help="train the float baseline")
    add_common(p)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--amp", choices=["off", "bf16"], default="bf16")
    p.set_defaults(optimizer="sgd", master_dtype="fp32", schedule="staged", weight_quant_from=0.05,
                   act_quant_from=0.1, freeze_bn_from=0.5, freeze_observer_from=0.75, track_updates=False,
                   track_diagnostics=False, stop_after_epoch=0)
    p.set_defaults(run=run_float)

    p = sub.add_parser("ptq", help="post-training quantization")
    add_common(p)
    add_quant(p)
    p.add_argument("--calib-batches", type=int, default=32)
    p.set_defaults(run=run_ptq)

    p = sub.add_parser("qat", help="quantization-aware training")
    add_common(p)
    add_quant(p)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--calib-batches", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--schedule", choices=["staged", "one-shot"], default="staged")
    p.add_argument("--master-dtype", choices=["fp32", "fp16"], default="fp32", help="fp16 reproduces pitfall 6")
    p.add_argument("--kd", action="store_true", help="distill from the float checkpoint")
    p.add_argument("--kd-lambda", type=float, default=0.5)
    p.add_argument("--kd-tau", type=float, default=4.0)
    p.add_argument("--weight-decay", type=float, default=0)
    p.add_argument("--amp", choices=["off"], default="off")
    p.add_argument("--weight-quant-from", type=float, default=0.05)
    p.add_argument("--act-quant-from", type=float, default=0.10)
    p.add_argument("--freeze-bn-from", type=float, default=0.50)
    p.add_argument("--freeze-observer-from", type=float, default=0.75)
    p.add_argument("--track-updates", action="store_true")
    p.add_argument("--track-diagnostics", action="store_true", help="record first 1000 training steps")
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--fold-order", choices=["fold", "raw"], default="fold")
    p.set_defaults(run=run_qat)

    return parser


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.subset < 0 or args.eval_subset < 0 or args.max_steps < 0:
        parser.error("batch size must be positive; workers and smoke limits must be nonnegative")
    if hasattr(args, "epochs") and (args.epochs <= 0 or args.lr <= 0):
        parser.error("epochs and learning rate must be positive")
    if args.stage == "qat" and (not 0 <= args.kd_lambda <= 1 or args.kd_tau <= 0):
        parser.error("KD weight must be in [0, 1] and temperature positive")
    args.run(args)


if __name__ == "__main__":
    main()
