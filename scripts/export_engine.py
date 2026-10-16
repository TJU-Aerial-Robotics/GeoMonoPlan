#!/usr/bin/env python3
"""Convert the release PT model into encoder/planner TensorRT engines locally."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--weights")
    parser.add_argument("--engine-dir")
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    parser.add_argument("--onnx-only", action="store_true", help="Only export ONNX; permits --device cpu")
    parser.add_argument("--verify-input", type=Path, help="Also verify a real rgb/obs/pose NPZ after the standard moving-pair checks")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--require-candidate-match", action="store_true", help="Require PT/engine candidate equality; default records differences as in the original FP16 workflow")
    parser.add_argument("--benchmark", type=int, default=100)
    args = parser.parse_args()
    if args.workspace_gib <= 0 or args.benchmark < 1 or args.atol < 0 or args.rtol < 0:
        parser.error("workspace-gib and benchmark must be positive; tolerances must be nonnegative")
    from planner.settings import load_settings
    from planner.runtime.export import convert
    settings = load_settings(args.config)
    for key in ("weights", "engine_dir", "device"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    convert(settings, args)


if __name__ == "__main__":
    main()
