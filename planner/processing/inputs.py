"""Image and pose conventions shared by online planning and offline verification."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def preprocess_bgr(image, height, width):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a uint8 HWC BGR image")
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray((image.astype(np.float32) / 255).transpose(2, 0, 1))


def relative_pose(previous, current):
    """Previous pose in the current body frame, as [qx,qy,qz,qw,tx,ty,tz]."""
    previous, current = np.asarray(previous), np.asarray(current)
    rotation = Rotation.from_quat(current[:4]).inv()
    quaternion = (rotation * Rotation.from_quat(previous[:4])).as_quat()
    translation = rotation.apply(previous[4:7] - current[4:7])
    return np.concatenate((quaternion, translation)).astype(np.float32)


def load_inputs(path):
    with np.load(path, allow_pickle=False) as values:
        return tuple(np.asarray(values[name], dtype=np.float32) for name in ("rgb", "obs", "pose"))
