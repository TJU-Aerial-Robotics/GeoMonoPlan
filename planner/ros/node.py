"""ROS adapter preserving the original RGB planner callback/control sequence."""
from threading import Lock
import time
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Header
try:
    from quadrotor_msgs.msg import PositionCommand
except ImportError as exc:
    raise ImportError("Build/source the controller workspace providing quadrotor_msgs first") from exc

from planner.processing.polynomial import Poly5Solver, calculate_yaw
from planner.runtime.policy import Policy


class PlannerNode:
    def __init__(self, settings, backend):
        rospy.init_node("geomonoplan", anonymous=False)
        self.settings = settings["ros"]
        self.policy = Policy(settings, backend)
        self.device = self.policy.device
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.height = self.policy.config["image_height"]
        self.width = self.policy.config["image_width"]
        self.timesteps = 2
        self.traj_num = self.policy.config["vertical_num"] * self.policy.config["horizon_num"]
        self.traj_time = self.policy.config["trajectory_duration"]
        self.goal = np.array(self.settings["goal"])
        self.auto_arrival_hold = bool(self.settings.get("auto_arrival_hold", True))
        self.arrival_tolerance = float(self.settings["arrival_tolerance_m"])
        self.plan_from_reference = self.settings["plan_from_reference"]
        self.verbose = bool(self.settings.get("verbose", False))
        self.visualize = self.settings["visualize"]
        self.validation_mode = bool(self.settings.get("validation_mode", False))
        self.flight_enabled = not self.validation_mode
        self.Rotation_bc = R.from_euler(
            "ZYX", [0, self.settings["camera_pitch_deg"], 0], degrees=True
        ).as_matrix()
        self.bridge = CvBridge()
        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        # Rolling trajectory sample step, independent of publication rate.
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = self.desire_vel = self.desire_acc = None
        self.optimal_poly_x = self.optimal_poly_y = self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
        self.rgb_lock = Lock()
        self.rgb_buffer = []
        self.rgb_frame_ids = []
        self.rgb_frame_counter = 0
        self.pose_buffer = []
        self.Rotation_wc = np.eye(3, dtype=np.float32)
        self.warm_up()

        self.ctrl_pub = rospy.Publisher(self.settings["command_topic"], PositionCommand, queue_size=1)
        self.path_pub = rospy.Publisher(self.settings["trajectory_topic"], Path, queue_size=1)
        self.subscribers = [
            rospy.Subscriber(self.settings["odom_topic"], Odometry, self.callback_odometry,
                             queue_size=1, tcp_nodelay=True),
            rospy.Subscriber(self.settings["image_topic"], Image, self.callback_rgb,
                             queue_size=1, tcp_nodelay=True, buff_size=1 << 20),
            rospy.Subscriber(self.settings["goal_topic"], PoseStamped, self.callback_set_goal, queue_size=1),
            rospy.Subscriber(self.settings.get("reset_pose_topic", "/sim/reset_pose"), PoseStamped,
                             self.callback_reset_pose, queue_size=1),
        ]
        if self.validation_mode:
            self.subscribers.append(rospy.Subscriber(
                self.settings["enable_topic"], Bool, self.callback_flight_enable, queue_size=1
            ))
            rospy.loginfo("Validation mode: waiting for %s", self.settings["enable_topic"])
        rospy.sleep(1.0)
        plan_period = 1.0 / float(self.settings["plan_hz"])
        self.timer = rospy.Timer(rospy.Duration(plan_period), self.control_pub)
        rospy.loginfo("GeoMonoPlan ready: backend=%s, image=%s", backend, self.settings["image_topic"])
        rospy.loginfo("Planning desired speed: %.2f m/s", self.policy.config["velocity"])

    def callback_reset_pose(self, data):
        """Invalidate the previous trajectory and rebase a cached arrival hold."""
        if data.header.frame_id not in ("", self.settings["world_frame"]):
            return
        p = data.pose.position
        q = data.pose.orientation
        position = np.array([p.x, p.y, p.z], dtype=float)
        quaternion = np.array([q.x, q.y, q.z, q.w], dtype=float)
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            return
        if np.linalg.norm(quaternion) < 1e-9:
            return
        yaw = float(R.from_quat(quaternion).as_euler("ZYX")[0])
        hold = PositionCommand()
        hold.header.frame_id = self.settings["world_frame"]
        hold.header.stamp = rospy.Time.now()
        hold.trajectory_flag = hold.TRAJECTORY_STATUS_EMPTY
        hold.position.x, hold.position.y, hold.position.z = position
        hold.yaw = yaw
        with self.rgb_lock:
            with self.lock:
                # Preserve stopped/arrived status. A reset must not restart flight.
                self.ctrl_time = None
                self.desire_init = False
                self.desire_pos = position.copy()
                self.desire_vel = np.zeros(3)
                self.desire_acc = np.zeros(3)
                self.last_yaw = yaw
                self.last_control_msg = hold
                self.optimal_poly_x = self.optimal_poly_y = self.optimal_poly_z = None
            self.rgb_buffer.clear()
            self.pose_buffer.clear()
            self.rgb_frame_ids.clear()
            self.policy.clear()
        rospy.loginfo("Reset planner reference to [%.3f, %.3f, %.3f]", *position)

    def callback_set_goal(self, data):
        self.goal = np.asarray([
            data.pose.position.x,
            data.pose.position.y,
            1.5,
        ])
        self.arrive = False
        print(
            f"New Goal: ({data.pose.position.x:.1f}, "
            f"{data.pose.position.y:.1f}, {data.pose.position.z:.1f})"
        )

    def callback_flight_enable(self, data):
        with self.rgb_lock:
            return self._callback_flight_enable_locked(data)

    def _callback_flight_enable_locked(self, data):
        enabled = bool(data.data)
        if enabled == self.flight_enabled:
            return
        self.flight_enabled = enabled
        self.arrive = False
        self.ctrl_time = None
        self.rgb_buffer.clear()
        self.pose_buffer.clear()
        self.rgb_frame_ids.clear()
        self.policy.clear()
        if enabled:
            self.desire_init = False
        rospy.loginfo("Planning %s by %s", "enabled" if enabled else "disabled", self.settings["enable_topic"])

    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            # Use reference acceleration, including during real-world flight.
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        if (
            self.auto_arrival_hold
            and np.linalg.norm(pos - self.goal) < self.arrival_tolerance
            and not self.arrive
        ):
            print("Arrive!")
            self.arrive = True
            hover_msg = PositionCommand()
            hover_msg.header.stamp = rospy.Time.now()
            hover_msg.header.frame_id = data.header.frame_id
            hover_msg.trajectory_flag = hover_msg.TRAJECTORY_STATUS_EMPTY
            hover_msg.position.x = float(pos[0])
            hover_msg.position.y = float(pos[1])
            hover_msg.position.z = float(pos[2])
            hover_msg.velocity.x = 0.0
            hover_msg.velocity.y = 0.0
            hover_msg.velocity.z = 0.0
            hover_msg.acceleration.x = 0.0
            hover_msg.acceleration.y = 0.0
            hover_msg.acceleration.z = 0.0
            hover_msg.yaw = self.last_yaw
            hover_msg.yaw_dot = 0.0
            with self.lock:
                self.desire_pos = pos.copy()
                self.desire_vel = np.zeros(3)
                self.desire_acc = np.zeros(3)
                self.last_control_msg = hover_msg
            # Immediately replace the last forward trajectory command instead
            # of waiting for the next control timer tick.
            self.ctrl_pub.publish(hover_msg)

    def process_odom(self):
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel if self.plan_from_reference else np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_b = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_b = np.dot(Rotation_cw, acc_w)
        # goal_dir
        goal_w = self.goal - self.desire_pos
        goal_b = np.dot(Rotation_cw, goal_w)

        return np.concatenate((vel_b, acc_b, goal_b), axis=0).astype(np.float32)

    def build_pose_from_odom(self):
        quat_xyzw = np.array([
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
            self.odom.pose.pose.orientation.w,
        ], dtype=np.float32)
        pos_w = np.array([
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ], dtype=np.float32)
        return np.concatenate((quat_xyzw, pos_w), axis=0)

    @staticmethod
    def build_relative_pose(prev_pose, cur_pose):
        q_rel = (R.from_quat(cur_pose[0:4]).inv() * R.from_quat(prev_pose[0:4])).as_quat()
        t_rel = R.from_quat(cur_pose[0:4]).inv().apply(prev_pose[4:7] - cur_pose[4:7])
        return np.concatenate((q_rel, t_rel), axis=0).astype(np.float32)

    def process_rgb(self, data):
        rgb_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        if rgb_image.shape[0] != self.height or rgb_image.shape[1] != self.width:
            rgb_image = cv2.resize(
                rgb_image, (self.width, self.height), interpolation=cv2.INTER_AREA
            )

        # Match the normalization used during training.
        rgb_image = rgb_image.astype(np.float32) / 255.0
        return rgb_image.astype(np.float32)

    @torch.inference_mode()
    def callback_rgb(self, data):
        with self.rgb_lock:
            return self._callback_rgb_locked(data)

    def _callback_rgb_locked(self, data):
        if not self.flight_enabled or not self.odom_init or self.arrive:
            return
        time0 = time.time()
        rgb = self.process_rgb(data)
        pose = self.build_pose_from_odom()
        self.rgb_frame_counter += 1
        self.rgb_frame_ids.append(self.rgb_frame_counter)
        self.rgb_buffer.append(rgb)
        self.pose_buffer.append(pose)
        if len(self.rgb_buffer) > self.timesteps:
            self.rgb_buffer.pop(0)
            self.rgb_frame_ids.pop(0)
            self.pose_buffer.pop(0)
        if len(self.rgb_buffer) < self.timesteps:
            self.rgb_buffer.insert(0, rgb.copy())
            self.rgb_frame_ids.insert(0, self.rgb_frame_counter)
            self.pose_buffer.insert(0, pose.copy())
        time1 = time.time()
        rgb_input = np.stack(self.rgb_buffer, axis=0).transpose(0, 3, 1, 2)[None, ...]
        pose_input = self.build_relative_pose(self.pose_buffer[0], self.pose_buffer[-1])[None, :]
        obs_input = self.process_odom()[None, :]
        rgb_input = torch.from_numpy(rgb_input).to(self.device, non_blocking=True)
        pose_input = torch.from_numpy(pose_input).to(self.device, non_blocking=True)
        obs_input = torch.from_numpy(obs_input).to(self.device, non_blocking=True)
        time2 = time.time()
        policy_outputs = self.forward_policy(rgb_input, obs_input, pose_input, frame_ids=tuple(self.rgb_frame_ids))
        endstate_pred, selection_logits = policy_outputs[:2]
        endstate_pred = endstate_pred[0]
        selection_logits = selection_logits[0]
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        time3 = time.time()
        start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        rgb_score, rgb_endstate_w, rgb_action_id = self.select_action_from_outputs(endstate_pred, selection_logits)
        with self.lock:
            self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], rgb_endstate_w[rgb_action_id, 0, 0] + start_pos[0], rgb_endstate_w[rgb_action_id, 0, 1], rgb_endstate_w[rgb_action_id, 0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], rgb_endstate_w[rgb_action_id, 1, 0] + start_pos[1], rgb_endstate_w[rgb_action_id, 1, 1], rgb_endstate_w[rgb_action_id, 1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], rgb_endstate_w[rgb_action_id, 2, 0] + start_pos[2], rgb_endstate_w[rgb_action_id, 2, 1], rgb_endstate_w[rgb_action_id, 2, 2], self.traj_time)
            self.ctrl_time = 0.0
        time4 = time.time()
        self.visualize_trajectory(rgb_score, rgb_endstate_w)
        time5 = time.time()
        self.print_time(time0, time1, time2, time3, time4, time5)

    def control_pub(self, _timer):
        if not self.flight_enabled:
            return
        if self.arrive and self.last_control_msg is not None:
            with self.lock:
                self.last_control_msg.header.stamp = rospy.Time.now()
                self.ctrl_pub.publish(self.last_control_msg)
            return
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return

        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety and publish frequency
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

    def select_action_from_outputs(self, endstate_pred, selection_logits, rotation_wc=None):
        endstate = endstate_pred.reshape(9, self.traj_num).T.detach().cpu().numpy()
        selection_score = selection_logits.reshape(self.traj_num).detach().cpu().numpy()

        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)
        rotation_wc = self.Rotation_wc if rotation_wc is None else rotation_wc
        endstate_w = np.matmul(rotation_wc.astype(endstate_c.dtype, copy=False), endstate_c)
        score = selection_score
        action_id = int(np.argmax(selection_score))
        if not self.visualize:
            endstate_w = endstate_w[action_id:action_id + 1]
            score = score[action_id:action_id + 1]
            action_id = 0
        return score, endstate_w, action_id

    def visualize_trajectory(self, pred_score, pred_endstate):
        if not self.visualize or self.path_pub.get_num_connections() == 0:
            return

        # Preserve the original sampling: 20 points, excluding Tf.
        dt = self.traj_time / 20.0
        t_values = np.arange(0, self.traj_time, dt)
        points = np.stack((
            self.optimal_poly_x.get_position(t_values),
            self.optimal_poly_y.get_position(t_values),
            self.optimal_poly_z.get_position(t_values),
        ), axis=-1).astype(np.float32)
        path = Path()
        path.header = Header()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.settings["world_frame"]
        for x, y, z in points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

    def print_time(self, time0, time1, time2, time3, time4, time5):
        if not self.verbose:
            return
        boundaries = (time0, time1, time2, time3, time4, time5)
        stages = [(end - start) * 1000 for start, end in zip(boundaries, boundaries[1:])]
        rospy.loginfo_throttle(
            1.0,
            "Processing latency (ms): image=%.2f, input=%.2f, inference=%.2f, "
            "trajectory=%.2f, visualization=%.2f, total=%.2f",
            *(stages + [(time5 - time0) * 1000])
        )

    def warm_up(self):
        rgb = torch.zeros((1, self.timesteps, 3, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        pose = torch.zeros((1, 7), dtype=torch.float32, device=self.device)
        pose[:, 3] = 1.0
        _ = self.forward_policy(rgb, obs, pose)

    @torch.inference_mode()
    def forward_policy(self, rgb, obs, pose, frame_ids=None):
        return self.policy(rgb, obs, pose, frame_ids=frame_ids, return_tensors=True)

    def run(self):
        rospy.spin()
