"""Inference layers adapted from YOPO.

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
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


class PlanningFeatureBridge(nn.Sequential):
    """Downsample visual features to the stride-32 planning map.

    Keep numeric Sequential children so existing r2d_plan_conv.0/2/4
    checkpoint keys remain compatible.
    """

    def __init__(self, width, hidden, feature_stride):
        if feature_stride not in (8, 16):
            raise ValueError("feature_stride must be 8 or 16")
        super().__init__(
            nn.Conv2d(width, width, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                width, width, 3,
                stride=32 // (2 * feature_stride), padding=1, bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, hidden, 3, padding=1, bias=False),
        )


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=3000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        pe = self.positional_encoding.repeat(x.size(0), 1, 1)
        return x + pe[:, 0:x.shape[1], :]

class PrimitiveEncoder(nn.Module):

    def __init__(self, d_model, primitive_num, primitive_dim=9, state_dim=9):
        super(PrimitiveEncoder, self).__init__()
        self.primitive_num = primitive_num
        self.primitive_token = nn.Parameter(torch.rand(1, primitive_num, primitive_dim))
        self.linear1 = nn.Linear(primitive_dim + state_dim, d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, state):
        state = state.unsqueeze(1).repeat(1, self.primitive_num, 1)
        primitive_token = self.primitive_token.repeat(state.size(0), 1, 1)
        x = torch.cat([primitive_token, state], dim=2)
        primitive_embedding = self.linear2(self.relu(self.linear1(x)))
        return primitive_embedding

class TrajectoryAlignedPlanningAdapter(nn.Module):

    def __init__(self, d_model, nhead, vertical_num, horizon_num, vertical_fov_deg, horizon_fov_deg, local_window=5, global_size=(2, 3)):
        super().__init__()
        if local_window <= 0 or local_window % 2 == 0:
            raise ValueError('planning_local_window must be a positive odd number')
        if min(global_size) <= 0:
            raise ValueError('planning global grid dimensions must be positive')
        self.vertical_num = int(vertical_num)
        self.horizon_num = int(horizon_num)
        self.query_num = self.vertical_num * self.horizon_num
        self.vertical_fov = math.radians(float(vertical_fov_deg))
        self.horizon_fov = math.radians(float(horizon_fov_deg))
        self.local_window = int(local_window)
        self.global_size = tuple((int(value) for value in global_size))
        self.angular_embedding = nn.Sequential(nn.Linear(4, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))
        self.local_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.local_norm = nn.LayerNorm(d_model)
        self.global_fusion = nn.Conv2d(2 * d_model, d_model, kernel_size=1, bias=False)
        self.global_norm = nn.LayerNorm(d_model)
        pitch = self._axis_centers(self.vertical_num, self.vertical_fov, True)
        yaw = self._axis_centers(self.horizon_num, self.horizon_fov, True)
        query_pitch, query_yaw = torch.meshgrid(pitch, yaw, indexing='ij')
        self.register_buffer('query_angles', torch.stack((query_yaw.reshape(-1), query_pitch.reshape(-1)), dim=-1), persistent=False)

    @staticmethod
    def _axis_centers(count, field_of_view, reverse):
        centers = (torch.arange(count, dtype=torch.float32) - (count - 1) / 2.0) * (field_of_view / count)
        return centers.flip(0) if reverse else centers

    @staticmethod
    def _angle_features(angles):
        yaw, pitch = angles.unbind(dim=-1)
        return torch.stack((torch.sin(yaw), torch.cos(yaw), torch.sin(pitch), torch.cos(pitch)), dim=-1)

    def _grid_angles(self, height, width, device):
        pitch = self._axis_centers(height, self.vertical_fov, True).to(device)
        yaw = self._axis_centers(width, self.horizon_fov, True).to(device)
        grid_pitch, grid_yaw = torch.meshgrid(pitch, yaw, indexing='ij')
        return torch.stack((grid_yaw.reshape(-1), grid_pitch.reshape(-1)), dim=-1)

    def _query_center_indices(self, height, width, device):
        angles = self.query_angles.to(device)
        yaw, pitch = angles.unbind(dim=-1)
        columns = torch.round((0.5 - yaw / self.horizon_fov) * width - 0.5).long().clamp(0, width - 1)
        rows = torch.round((0.5 - pitch / self.vertical_fov) * height - 0.5).long().clamp(0, height - 1)
        return rows * width + columns

    def _local_indices(self, height, width, device):
        centers = self._query_center_indices(height, width, device)
        center_rows = centers // width
        center_columns = centers % width
        offsets = torch.arange(-(self.local_window // 2), self.local_window // 2 + 1, device=device)
        row_offsets, column_offsets = torch.meshgrid(offsets, offsets, indexing='ij')
        rows = (center_rows[:, None] + row_offsets.reshape(1, -1)).clamp(0, height - 1)
        columns = (center_columns[:, None] + column_offsets.reshape(1, -1)).clamp(0, width - 1)
        return rows * width + columns

    def _position_embedding(self, angles, dtype):
        features = self._angle_features(angles).to(dtype=dtype)
        return self.angular_embedding(features)

    def forward(self, detail_map, trajectory_queries):
        if detail_map.dim() != 4:
            raise ValueError('detail_map must have shape [B, C, H, W]')
        if trajectory_queries.dim() != 3:
            raise ValueError('trajectory_queries must have shape [B, N, C]')
        batch_size, channels, height, width = detail_map.shape
        if trajectory_queries.shape[:2] != (batch_size, self.query_num):
            raise ValueError(f'Expected {self.query_num} trajectory queries, got {trajectory_queries.shape[1]}')
        detail_tokens = detail_map.flatten(2).permute(0, 2, 1)
        grid_angles = self._grid_angles(height, width, detail_map.device)
        grid_position = self._position_embedding(grid_angles, detail_map.dtype).transpose(0, 1).reshape(1, channels, height, width)
        positioned_tokens = (detail_map + grid_position).flatten(2).permute(0, 2, 1)
        local_indices = self._local_indices(height, width, detail_map.device).reshape(-1)
        local_memory = positioned_tokens.index_select(1, local_indices).reshape(batch_size * self.query_num, self.local_window * self.local_window, channels)
        query_position = self._position_embedding(self.query_angles.to(detail_map.device), trajectory_queries.dtype).unsqueeze(0)
        positioned_queries = trajectory_queries + query_position
        local_query = positioned_queries.reshape(batch_size * self.query_num, 1, channels)
        local_output, _ = self.local_attention(local_query, local_memory, local_memory, need_weights=False)
        aligned_queries = self.local_norm(positioned_queries + local_output.reshape(batch_size, self.query_num, channels))
        global_coarse = F.interpolate(detail_map, size=self.global_size, mode='bilinear', align_corners=False)
        global_mean = detail_map.mean(dim=(2, 3), keepdim=True).expand(-1, -1, self.global_size[0], self.global_size[1])
        global_map = self.global_fusion(torch.cat((global_coarse, global_mean), dim=1))
        global_tokens = global_map.flatten(2).permute(0, 2, 1)
        global_angles = self._grid_angles(self.global_size[0], self.global_size[1], detail_map.device)
        global_tokens = self.global_norm(global_tokens + self._position_embedding(global_angles, global_tokens.dtype).unsqueeze(0))
        return (aligned_queries, global_tokens, detail_tokens)

class CustomTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        self_attn_output, _ = self.self_attention(tgt, tgt, tgt, need_weights=False)
        tgt = self.norm0(tgt + self_attn_output)
        cross_attn_output, _ = self.cross_attention(tgt, memory, memory, need_weights=False)
        tgt = self.norm1(tgt + cross_attn_output)
        ff_output = self.linear2(self.relu(self.linear1(tgt)))
        tgt = self.norm2(tgt + ff_output)
        return tgt

class CustomTransformerDecoder(nn.Module):

    def __init__(self, layer, num_layers):
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt

class CustomTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, src):
        self_attn_output, _ = self.self_attention(src, src, src, need_weights=False)
        src = self.norm0(src + self_attn_output)
        ff_output = self.linear2(self.relu(self.linear1(src)))
        src = self.norm1(src + ff_output)
        return src

class CustomTransformerEncoder(nn.Module):

    def __init__(self, layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src
