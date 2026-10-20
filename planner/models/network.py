"""Inference model only: frame encoder, temporal fusion and planning outputs."""
import copy
import torch
from torch import nn
from scipy.spatial.transform import Rotation

from .backbone import ImageEncoder
from .layers import (
    PositionalEncoding, PrimitiveEncoder, TrajectoryAlignedPlanningAdapter,
    PlanningFeatureBridge,
    CustomTransformerEncoderLayer, CustomTransformerEncoder,
    CustomTransformerDecoderLayer, CustomTransformerDecoder,
)


class PlanningTransform(nn.Module):
    """Original primitive construction, observation normalization and decoding."""

    def __init__(self, config):
        super().__init__()
        self.goal_length = 2.0 * config["radio_range"]
        self.vel_max = config["velocity"]
        self.acc_max = config["acceleration"]
        self.horizon_num = config["horizon_num"]
        self.vertical_num = config["vertical_num"]
        self.horizon_fov = config["horizon_camera_fov"]
        self.vertical_fov = config["vertical_camera_fov"]
        self.horizon_anchor_fov = config["horizon_anchor_fov"]
        self.vertical_anchor_fov = config["vertical_anchor_fov"]
        self.radio_range = config["radio_range"]
        direction_diff = 0 if self.horizon_num == 1 else (self.horizon_fov / 180.0 * torch.pi) / self.horizon_num
        altitude_diff = 0 if self.vertical_num == 1 else (self.vertical_fov / 180.0 * torch.pi) / self.vertical_num
        angles = []
        rotations = []
        for i in range(self.vertical_num):
            for j in range(self.horizon_num):
                alpha = torch.tensor(-direction_diff * (self.horizon_num - 1) / 2 + j * direction_diff)
                beta = torch.tensor(-altitude_diff * (self.vertical_num - 1) / 2 + i * altitude_diff)
                angles.append(torch.tensor([alpha, beta]))
                rotations.append(torch.tensor(Rotation.from_euler("ZYX", [alpha, -beta, 0.0], degrees=False).as_matrix()))
        self.register_buffer("angles", torch.stack(angles).to(dtype=torch.float32), persistent=False)
        self.register_buffer("rotations", torch.stack(rotations).to(dtype=torch.float32), persistent=False)
        self.yaw_diff = 0.5 * self.horizon_anchor_fov / 180.0 * torch.pi
        self.pitch_diff = 0.5 * self.vertical_anchor_fov / 180.0 * torch.pi

    def getAngleLattice(self):
        return self.angles[:, 0], self.angles[:, 1]

    def getRotation(self):
        return self.rotations

    def normalize(self, obs):
        return self.normalize_obs(obs.clone())

    def normalize_obs(self, vel_acc_goal):
        vel_acc_goal[:, 0:3] = vel_acc_goal[:, 0:3] / self.vel_max
        vel_acc_goal[:, 3:6] = vel_acc_goal[:, 3:6] / self.acc_max

        # Clamp the goal direction to unit length
        goal_norm = vel_acc_goal[:, 6:9].norm(dim=1, keepdim=True)
        vel_acc_goal[:, 6:9] = vel_acc_goal[:, 6:9] / goal_norm.clamp(min=self.goal_length)
        return vel_acc_goal

    def decode(self, endstate_pred: torch.Tensor) -> torch.Tensor:
        """
            Transform the predicted state to the body frame (Original prediction → Primitive frame → Body frame).
            endstate_pred: [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h]
            :return [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h] in body frame
        """
        B, V, H = endstate_pred.shape[0], endstate_pred.shape[2], endstate_pred.shape[3]

        # [B, 9, 3, 5] -> [B, 3, 5, 9] -> [B, 15, 9]
        endstate_pred = endstate_pred.permute(0, 2, 3, 1).reshape(B, V * H, 9)
        dtype = endstate_pred.dtype
        device = endstate_pred.device

        # Flip lattice angles and rotations to match the reversed grid order.
        yaw, pitch = self.getAngleLattice()  # [15]
        yaw = yaw.flip(0).to(device=device, dtype=dtype)[None, :].expand(B, -1)  # [B, 15]
        pitch = pitch.flip(0).to(device=device, dtype=dtype)[None, :].expand(B, -1)  # [B, 15]
        Rbp = self.getRotation().flip(0)  # [15, 3, 3]
        Rbp = Rbp.to(device=device, dtype=dtype)[None, :, :, :].expand(B, -1, -1, -1)  # [B, 15, 3, 3]

        yaw_diff = torch.as_tensor(self.yaw_diff, device=device, dtype=dtype)
        pitch_diff = torch.as_tensor(self.pitch_diff, device=device, dtype=dtype)
        radio_range = torch.as_tensor(self.radio_range, device=device, dtype=dtype)
        vel_max = torch.as_tensor(self.vel_max, device=device, dtype=dtype)
        acc_max = torch.as_tensor(self.acc_max, device=device, dtype=dtype)

        delta_yaw = endstate_pred[:, :, 0] * yaw_diff  # [B, 15]
        delta_pitch = endstate_pred[:, :, 1] * pitch_diff
        radio = (endstate_pred[:, :, 2] + endstate_pred.new_tensor(1.0)) * radio_range

        cos_pitch = torch.cos(pitch + delta_pitch)
        endstate_x = cos_pitch * torch.cos(yaw + delta_yaw) * radio
        endstate_y = cos_pitch * torch.sin(yaw + delta_yaw) * radio
        endstate_z = torch.sin(pitch + delta_pitch) * radio
        endstate_p = torch.stack([endstate_x, endstate_y, endstate_z], dim=-1)  # [B, 15, 3]

        # vel / acc
        endstate_vp = endstate_pred[:, :, 3:6] * vel_max  # [B, 15, 3]
        endstate_ap = endstate_pred[:, :, 6:9] * acc_max  # [B, 15, 3]

        # Rotate velocity and acceleration into the body frame.
        endstate_vb = torch.matmul(Rbp, endstate_vp.unsqueeze(-1)).squeeze(-1)  # [B, 15, 3]
        endstate_ab = torch.matmul(Rbp, endstate_ap.unsqueeze(-1)).squeeze(-1)

        endstate = torch.cat([endstate_p, endstate_vb, endstate_ab], dim=-1)  # [B, 15, 9]

        endstate = endstate.permute(0, 2, 1).reshape(B, 9, V, H)  # [B, 9, 3, 5]
        return endstate



class GeoMonoPlan(nn.Module):
    """All inputs are BGR [0,1], metric observations, and relative xyzw pose."""

    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        width = config["feature_channels"]
        hidden = config["hidden_channels"]
        rows, columns = config["vertical_num"], config["horizon_num"]
        self.rgb_backbone = ImageEncoder(config)
        self.positional_encoding = PositionalEncoding(hidden) if width == hidden else None
        self.rgb_positional_encoding = PositionalEncoding(width) if width != hidden else None
        self.pose_embed = nn.Linear(7, width)
        self.frame_encoder = CustomTransformerEncoder(CustomTransformerEncoderLayer(width, 4, 2 * width), config["frame_layers"])
        self.fusion_encoder = CustomTransformerEncoder(CustomTransformerEncoderLayer(2 * width, 4, 2 * width), config["fusion_layers"])
        self.fusion_head = nn.Conv1d(2 * width, width, 1)
        self.r2d_plan_conv = PlanningFeatureBridge(
            width, hidden, config["feature_stride"]
        )
        self.state_backbone = PrimitiveEncoder(hidden, rows * columns)
        self.planning_adapter = TrajectoryAlignedPlanningAdapter(
            hidden, 4, rows, columns, config["vertical_camera_fov"], config["horizon_camera_fov"],
            config["local_window"], tuple(config["global_size"]),
        )
        self.plan_header = CustomTransformerDecoder(CustomTransformerDecoderLayer(hidden, 4, 2 * hidden), config["decoder_layers"])
        self.traj_header = nn.Sequential(nn.Linear(hidden, 10))
        self.transform = PlanningTransform(config)

    def encode_frame(self, image):
        return self.rgb_backbone(image)

    def plan_from_features(self, features, obs, pose):
        obs = self.transform.normalize(obs)
        batch, frames, width, height, columns = features.shape
        embedding = self.pose_embed(pose.to(dtype=features.dtype))
        tokens = features.flatten(3).permute(0, 1, 3, 2)
        tokens = (tokens + embedding[:, None, None, :]).reshape(batch * frames, height * columns, width)
        position = self.rgb_positional_encoding if self.rgb_positional_encoding is not None else self.positional_encoding
        frame_tokens = self.frame_encoder(position(tokens))
        fusion = frame_tokens.reshape(batch, frames, height * columns, width).permute(0, 2, 1, 3).reshape(batch, height * columns, frames * width)
        fusion = self.fusion_encoder(fusion)
        detail_map = self.fusion_head(fusion.permute(0, 2, 1)).reshape(batch, width, height, columns)
        detail_map = self.r2d_plan_conv(detail_map)
        queries = self.state_backbone(obs)
        queries, memory, _ = self.planning_adapter(detail_map, queries)
        output = self.traj_header(self.plan_header(queries, memory)).permute(0, 2, 1)
        output = output.reshape(batch, 10, self.config["vertical_num"], self.config["horizon_num"])
        # Keep all four slice axes, including full spatial slices, exactly as
        # the original exporter. Shorthand indexing changes ScatterND's ONNX
        # index construction even though eager PyTorch returns the same values.
        output[:, 0:9, :, :] = output[:, 0:9, :, :].tanh()
        return self.transform.decode(output[:, 0:9, :, :]), output[:, 9, :, :]

    def forward(self, images, obs, pose):
        batch, frames, channels, height, width = images.shape
        encoded = self.encode_frame(images.reshape(batch * frames, channels, height, width))
        features = encoded.reshape(batch, frames, self.config["feature_channels"], *encoded.shape[-2:])
        return self.plan_from_features(features, obs, pose)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected a GeoMonoPlan release checkpoint (format_version=1)")
    model = GeoMonoPlan(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()
