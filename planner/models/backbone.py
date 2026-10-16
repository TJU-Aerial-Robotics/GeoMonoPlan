"""Residual block adapted from torchvision.

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch
import math
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Optional, Callable

def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: Optional[nn.Module]=None, groups: int=1, base_width: int=64, dilation: int=1, norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out



class ResidualConvAdapter(nn.Module):

    def __init__(self, channels, bottleneck=32, residual_scale=1.0):
        super().__init__()
        if bottleneck <= 0:
            raise ValueError('Adapter bottleneck must be positive')
        residual_scale = float(residual_scale)
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError('Adapter residual_scale must be finite and nonnegative')
        self.register_buffer('residual_scale', torch.tensor(residual_scale))
        self.down = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.spatial = nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck, bias=False)
        self.up = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.activation = nn.SiLU()
        nn.init.zeros_(self.up.weight)

    def forward(self, feature):
        correction = self.activation(self.down(feature))
        correction = self.activation(self.spatial(correction))
        return feature + self.residual_scale * self.up(correction)


class DepthFeatureExtractor(nn.Module):

    def __init__(self, output_dim=128, channels=32, output_stride=8):
        super().__init__()
        if channels <= 0 or output_dim <= 0:
            raise ValueError('Depth feature channel counts must be positive')
        if output_stride not in (8, 16):
            raise ValueError('Depth feature output_stride must be 8 or 16')
        stem_channels = max(1, channels // 2)
        self.stem = nn.Sequential(nn.Conv2d(1, stem_channels, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(stem_channels), nn.ReLU())
        shortcut = nn.Sequential(nn.Conv2d(stem_channels, channels, 1, stride=2, bias=False), nn.BatchNorm2d(channels))
        self.refine = BasicBlock(stem_channels, channels, stride=2, downsample=shortcut)
        if output_stride == 16:
            coarse_channels = 2 * channels
            coarse_shortcut = nn.Sequential(nn.Conv2d(channels, coarse_channels, 1, stride=2, bias=False), nn.BatchNorm2d(coarse_channels))
            self.coarse_refine = BasicBlock(channels, coarse_channels, stride=2, downsample=coarse_shortcut)
        else:
            coarse_channels = channels
            self.coarse_refine = nn.Identity()
        self.projection = nn.Conv2d(coarse_channels, output_dim, 1, bias=False)

    def forward(self, depth):
        feature = self.stem(depth)
        feature = self.refine(feature)
        feature = self.coarse_refine(feature)
        return self.projection(feature)


class ImageEncoder(nn.Module):
    """Build YOLO26-Depth; all parameters come from geomonoplan.pt."""

    def __init__(self, config):
        super().__init__()
        import copy
        from ultralytics.nn.tasks import DepthModel

        architecture = copy.deepcopy(config["encoder_architecture"])
        model = DepthModel(architecture, ch=3, verbose=False).float()
        layers = list(model.model)
        backbone_count = len(architecture["backbone"])
        self.backbone = nn.ModuleList(layers[:backbone_count])
        self.neck = nn.ModuleList(layers[backbone_count:-1])
        self.depth_head = layers[-1]
        if type(self.depth_head).__name__ != "Depth":
            raise ValueError("The release requires a depth-capable Ultralytics version")
        self.depth_head.export = False
        self.feature_indices = tuple(self.depth_head.f)
        self.input_size = tuple(config["encoder_input_size"])
        self.output_stride = config["feature_stride"]
        self.adapters = nn.ModuleDict()
        for index in config["adapter_layers"]:
            self.adapters[str(index)] = ResidualConvAdapter(
                layers[index].cv2.conv.out_channels,
                config["adapter_bottleneck"], config["adapter_scale"],
            )
        self.depth_features = DepthFeatureExtractor(
            config["feature_channels"], config["depth_feature_channels"], self.output_stride,
        )
        self.register_buffer("depth_normalization_m", torch.tensor(float(config["depth_normalization_m"])))

    def _run_layer(self, layer, current, outputs):
        source = layer.f
        if source != -1:
            current = (
                outputs[source] if isinstance(source, int)
                else [current if index == -1 else outputs[index] for index in source]
            )
        current = layer(current)
        adapter_key = str(layer.i)
        if adapter_key in self.adapters:
            current = self.adapters[adapter_key](current)
        # All later sequential and skip/concat routes use the adapted output.
        outputs.append(current)
        return current


    def forward_native_depth(self, images):
        """Return calibrated depth in meters at the head's native H/4 x W/4."""
        # Dataset/ROS inputs are BGR floats in [0,1]; YOLO expects RGB.
        current = F.interpolate(
            images[:, [2, 1, 0]], size=self.input_size,
            mode="bilinear", align_corners=False,
        )
        outputs = []
        # Frozen weights naturally avoid building a graph before the first
        # trainable adapter. Later frozen layers must propagate adapter gradients.
        for layer in self.backbone:
            current = self._run_layer(layer, current, outputs)
        for layer in self.neck:
            current = self._run_layer(layer, current, outputs)

        # No no_grad()/detach(): the frozen decoder must pass gradients back to
        # the adapters. Its original calibration buffers are retained as well.
        depth = self.depth_head([outputs[index] for index in self.feature_indices])
        if isinstance(depth, dict):
            # An optionally unfrozen head returns uncalibrated depth in train
            # mode. Apply the same calibration used by its eval forward path.
            depth = depth["depth"].pow(self.depth_head.cal_a) * self.depth_head.cal_b.exp()
        return depth


    def forward(self, images):
        # Align the lightweight depth features to the requested effective token
        # stride with only a small final resize (18x32 -> 17x30 by default).
        # Keep the pretrained YOLO decoder and calibration unchanged.
        stride = self.output_stride
        output_size = (
            (images.shape[-2] + stride - 1) // stride,
            (images.shape[-1] + stride - 1) // stride,
        )
        depth = self.forward_native_depth(images) / self.depth_normalization_m
        features = self.depth_features(depth)
        return F.interpolate(
            features, size=output_size, mode="bilinear", align_corners=False
        )
