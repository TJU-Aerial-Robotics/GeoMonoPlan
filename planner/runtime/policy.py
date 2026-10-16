"""Common two-frame cache and output contract for PT and TensorRT."""
import hashlib
import json
from threading import RLock
import torch

from planner.settings import resolve_path
from planner.processing.scaling import PlanningSpeed
from .feature_cache import RGBFeatureCache


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Policy:
    def __init__(self, settings, backend="pt"):
        self.device = torch.device(settings["device"])
        self.lock = RLock()
        if backend == "pt":
            from planner.models.network import load_model
            self.model = load_model(resolve_path(settings["weights"]), self.device)
            self.config = self.model.config
            self.encode = self.model.encode_frame
            self.plan = self.model.plan_from_features
        elif backend == "engine":
            from .trt_runner import Engine
            directory = resolve_path(settings["engine_dir"])
            if not (directory / "manifest.json").is_file():
                raise FileNotFoundError("No engine manifest in {}. Run scripts/export_engine.py first.".format(directory))
            with (directory / "manifest.json").open() as stream:
                manifest = json.load(stream)
            if manifest.get("format_version") != 1:
                raise ValueError("Unsupported engine manifest")
            for name, expected_hash in manifest["engine_sha256"].items():
                if sha256(directory / name) != expected_hash:
                    raise ValueError("Engine file does not match its manifest: {}".format(name))
            self.config = manifest["model_config"]
            self.encoder = Engine(directory / "encoder.engine", ("rgb",), ("features",), self.device)
            self.planner = Engine(directory / "planner.engine", ("features", "obs", "pose"), ("endstate", "score"), self.device)
            self.encode = lambda rgb: self.encoder.run(rgb=rgb)[0]
            self.plan = lambda features, obs, pose: self.planner.run(features=features, obs=obs, pose=pose)
        else:
            raise ValueError("backend must be pt or engine")
        self.feature_cache = RGBFeatureCache(self.encode, max_frames=2)
        # Model config describes the fixed normalization embedded in PT/engine.
        # Runtime config exposes the requested physical scales to trajectory code.
        self.model_config = self.config
        self.speed = PlanningSpeed(self.model_config, settings.get("velocity"))
        self.config = dict(
            self.model_config,
            velocity=self.speed.velocity,
            acceleration=self.speed.acceleration,
            trajectory_duration=self.speed.duration,
        )

    def clear(self):
        with self.lock:
            self.feature_cache.clear()

    @torch.inference_mode()
    def __call__(self, rgb, obs, pose, frame_ids=None, return_tensors=False):
        rgb, obs, pose = [torch.as_tensor(value, device=self.device, dtype=torch.float32) for value in (rgb, obs, pose)]
        expected = (1, 2, 3, self.config["image_height"], self.config["image_width"])
        if tuple(rgb.shape) != expected or tuple(obs.shape) != (1, 9) or tuple(pose.shape) != (1, 7):
            raise ValueError("Expected rgb={}, obs=(1,9), pose=(1,7)".format(expected))
        if frame_ids is not None and len(frame_ids) != 2:
            raise ValueError("Supply two immutable frame IDs, or omit IDs to disable caching")
        with self.lock:
            features = self.feature_cache.encode_sequence(rgb, frame_ids)
            model_obs = self.speed.model_observation(obs)
            endstate, score = self.plan(features, model_obs, pose)
            endstate = self.speed.physical_endstate(endstate)
            if return_tensors:
                return endstate, score
            return endstate.float().cpu().numpy(), score.float().cpu().numpy()
