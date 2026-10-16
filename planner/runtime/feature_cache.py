"""Small inference-only cache keyed by immutable RGB frame identifiers."""

from collections import OrderedDict
from threading import RLock

import torch


class RGBFeatureCache:
    """Reuse pre-pose features; callers supply fresh IDs for fresh image content.

    Clear after loading new weights or resetting a rollout. IDs must identify
    the same preprocessed image, not a tensor allocation or a pose. Omitting IDs
    deliberately computes uncached features (for warmup and stateless callers).
    """

    def __init__(self, encode_frame, max_frames=2):
        if max_frames < 1:
            raise ValueError("RGB cache capacity must be positive")
        self.encode_frame = encode_frame
        self.max_frames = max_frames
        self._entries = OrderedDict()
        self._signature = None
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._signature = None
            self.hits = self.misses = 0

    @torch.inference_mode()
    def encode_sequence(self, rgb, frame_ids=None):
        owner = getattr(self.encode_frame, "__self__", None)
        if isinstance(owner, torch.nn.Module) and owner.training:
            raise RuntimeError("Cached RGB encoding requires model.eval()")
        if rgb.dim() != 5 or rgb.shape[0] != 1:
            raise ValueError("Streaming RGB cache expects [1,T,3,H,W]")
        frames = rgb.shape[1]
        if frame_ids is not None:
            frame_ids = tuple(frame_ids)
            if len(frame_ids) != frames or frames > self.max_frames:
                raise ValueError("Frame IDs must match the sequence and fit the cache capacity")
        signature = (tuple(rgb.shape[2:]), rgb.dtype, rgb.device)
        with self._lock:
            if signature != self._signature:
                self.clear()
                self._signature = signature
            features = []
            for index in range(frames):
                key = None if frame_ids is None else frame_ids[index]
                if frame_ids is not None and key in self._entries:
                    feature, event = self._entries[key]
                    self._entries.move_to_end(key)
                    self.hits += 1
                else:
                    feature = self.encode_frame(rgb[:, index].contiguous()).detach()
                    if feature.dim() != 4 or feature.shape[0] != 1:
                        raise ValueError("RGB encoder must return [1,C,H,W] features")
                    event = None
                    if feature.is_cuda:
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(feature.device))
                    if frame_ids is not None:
                        self._entries[key] = (feature, event)
                        self.misses += 1
                        while len(self._entries) > self.max_frames:
                            self._entries.popitem(last=False)
                if feature.is_cuda:
                    stream = torch.cuda.current_stream(feature.device)
                    if event is not None:
                        stream.wait_event(event)
                    feature.record_stream(stream)
                features.append(feature)
            return torch.stack(features, dim=1)
