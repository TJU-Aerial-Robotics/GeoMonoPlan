"""Runtime speed scaling matching the original motion-primitive planner.

The network/engine keeps its exported normalization. Rescaling physical inputs
and decoded outputs implements the requested speed without rebuilding engines.
"""
import math


class PlanningSpeed:
    def __init__(self, model_config, velocity=None):
        base_velocity = float(model_config["velocity"])
        base_acceleration = float(model_config["acceleration"])
        base_duration = float(model_config["trajectory_duration"])
        self.velocity = base_velocity if velocity is None else float(velocity)
        values = (base_velocity, base_acceleration, base_duration, self.velocity)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Planning velocity, acceleration and duration must be finite and positive")
        self.ratio = self.velocity / base_velocity
        self.acceleration = base_acceleration * self.ratio ** 2
        self.duration = base_duration / self.ratio

    def model_observation(self, obs):
        if self.ratio == 1.0:
            return obs
        scaled = obs.clone()
        scaled[:, :3] /= self.ratio
        scaled[:, 3:6] /= self.ratio ** 2
        return scaled

    def physical_endstate(self, endstate):
        if self.ratio == 1.0:
            return endstate
        scaled = endstate.clone()
        scaled[:, 3:6] *= self.ratio
        scaled[:, 6:9] *= self.ratio ** 2
        return scaled
