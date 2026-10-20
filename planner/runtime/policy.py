"""Common two-frame cache and output contract for PT and TensorRT."""
from threading import RLock
import torch

from planner.settings import resolve_path
from planner.processing.scaling import PlanningSpeed
from .feature_cache import RGBFeatureCache


# Physical normalization embedded in the released YOPO_100/epoch140 model.
ENGINE_SCALES = {"velocity": 6.0, "acceleration": 6.0, "trajectory_duration": 10.0 / 6.0}


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
            self.encoder = Engine(directory / "encoder.engine", ("rgb",), ("features",), self.device)
            self.planner = Engine(directory / "planner.engine", ("features", "obs", "pose"), ("endstate", "score"), self.device)
            rgb_shape = tuple(self.encoder.engine.get_tensor_shape("rgb"))
            features_shape = tuple(self.encoder.engine.get_tensor_shape("features"))
            score_shape = tuple(self.planner.engine.get_tensor_shape("score"))
            if (len(rgb_shape) != 4 or rgb_shape[:2] != (1, 3)
                    or len(features_shape) != 4 or features_shape[0] != 1
                    or len(score_shape) != 3 or score_shape[0] != 1
                    or any(size <= 0 for size in rgb_shape + features_shape + score_shape)):
                raise ValueError("Expected fixed batch-one RGB encoder and candidate-grid planner")
            expected_shapes = {
                "features": (1, 2, *features_shape[1:]), "obs": (1, 9), "pose": (1, 7),
                "endstate": (1, 9, *score_shape[1:]),
            }
            for name, expected_shape in expected_shapes.items():
                if tuple(self.planner.engine.get_tensor_shape(name)) != expected_shape:
                    raise ValueError("Incompatible planner binding: {}".format(name))
            self.config = dict(ENGINE_SCALES, image_height=rgb_shape[2], image_width=rgb_shape[3],
                               vertical_num=score_shape[1], horizon_num=score_shape[2])
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
