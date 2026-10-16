#!/usr/bin/env python3
"""Run ROS planning, or infer one preprocessed NPZ without ROS."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pt", "engine"), default="pt")
    parser.add_argument("--config", help="YAML path; relative paths use the project root")
    parser.add_argument("--weights", help="Override the release PT checkpoint")
    parser.add_argument("--engine-dir", help="Override the directory containing both engines and manifest")
    parser.add_argument("--device", help="Torch device, e.g. cuda:0 or cpu (PT only)")
    parser.add_argument("--input", type=Path, help="Offline NPZ with rgb, obs and pose; otherwise run ROS")
    parser.add_argument("--output", type=Path, default=Path("prediction.npz"), help="Offline result path")
    args, remaining = parser.parse_known_args()
    if any(":=" not in value for value in remaining):
        parser.error("Unknown arguments: {}".format(" ".join(remaining)))

    from planner.settings import load_settings
    from planner.runtime.policy import Policy
    settings = load_settings(args.config)
    for key in ("weights", "engine_dir", "device"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    if args.input is not None:
        import numpy as np
        from planner.processing.inputs import load_inputs
        policy = Policy(settings, args.backend)
        print("Planning desired speed: {:.2f} m/s".format(policy.config["velocity"]))
        endstate, score = policy(*load_inputs(args.input))
        if not np.isfinite(endstate).all() or not np.isfinite(score).all():
            raise RuntimeError("Inference returned nonfinite outputs")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as output:
            np.savez_compressed(output, endstate=endstate, score=score)
        print("Saved {} (candidate {})".format(args.output, int(score.argmax())))
    else:
        from planner.ros.node import PlannerNode
        PlannerNode(settings, args.backend).run()


if __name__ == "__main__":
    main()
