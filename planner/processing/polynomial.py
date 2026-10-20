"""Trajectory interpolation and heading adapted from YOPO.

MIT License

Copyright (c) 2024, TJU-Aerial-Robotics
Tianjin University, China

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


class Poly5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """ 5-th order polynomial at each Axis """
        State_Mat = np.array([pos0, vel0, acc0, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1 / 2, 0, 0, 0],
                             [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
                             [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
                             [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]])
        self.A = np.dot(Coef_inv, State_Mat)

    def get_snap(self, t):
        """Return the scalar jerk at time t."""
        return 24 * self.A[4] + 120 * self.A[5] * t

    def get_jerk(self, t):
        """Return the scalar jerk at time t."""
        return 6 * self.A[3] + 24 * self.A[4] * t + 60 * self.A[5] * t * t

    def get_acceleration(self, t):
        """Return the scalar acceleration at time t."""
        return 2 * self.A[2] + 6 * self.A[3] * t + 12 * self.A[4] * t * t + 20 * self.A[5] * t * t * t

    def get_velocity(self, t):
        """Return the scalar velocity at time t."""
        return self.A[1] + 2 * self.A[2] * t + 3 * self.A[3] * t * t + 4 * self.A[4] * t * t * t + \
            5 * self.A[5] * t * t * t * t

    def get_position(self, t):
        """Return the scalar position at time t."""
        return self.A[0] + self.A[1] * t + self.A[2] * t * t + self.A[3] * t * t * t + self.A[4] * t * t * t * t + \
            self.A[5] * t * t * t * t * t


class Polys5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """ multiple 5-th order polynomials at each Axis (only used for visualization of multiple trajectories) """
        N = len(pos1)
        State_Mat = np.array([[pos0] * N, [vel0] * N, [acc0] * N, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1 / 2, 0, 0, 0],
                             [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
                             [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
                             [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]])
        self.A = np.dot(Coef_inv, State_Mat)

    def get_position(self, t):
        """Return the position array at time t."""
        t = np.atleast_1d(t)
        result = (self.A[0][:, np.newaxis] + self.A[1][:, np.newaxis] * t + self.A[2][:, np.newaxis] * t ** 2 +
                  self.A[3][:, np.newaxis] * t ** 3 + self.A[4][:, np.newaxis] * t ** 4 + self.A[5][:, np.newaxis] * t ** 5)
        return result.flatten()

def wrap_to_pi(angle):
    """Wrap the angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_yaw(vel_dir, goal_dir, last_yaw, dt, max_yaw_rate=0.5):
    # Normalize velocity and goal directions
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-5)
    goal_dist = np.linalg.norm(goal_dir)
    goal_dir = goal_dir / (goal_dist + 1e-5)

    # Goal yaw and weighting
    goal_yaw = np.arctan2(goal_dir[1], goal_dir[0])
    delta_yaw = wrap_to_pi(goal_yaw - last_yaw)
    weight = 6 * abs(delta_yaw) / np.pi  # weight ∈ [0,6]; equal weight at 30°, goal weight increases as delta_yaw grows

    # Desired direction and yaw
    dir_des = vel_dir + weight * goal_dir
    yaw_desired = np.arctan2(dir_des[1], dir_des[0]) if goal_dist > 0.5 else last_yaw

    # Yaw difference and limit
    yaw_diff = wrap_to_pi(yaw_desired - last_yaw)
    max_yaw_change = max_yaw_rate * np.pi * dt
    yaw_change = np.clip(yaw_diff, -max_yaw_change, max_yaw_change)

    # Updated yaw and yaw rate
    yaw = wrap_to_pi(last_yaw + yaw_change)
    yawdot = yaw_change / dt

    return yaw, yawdot
