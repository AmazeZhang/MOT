"""Pure helpers for D3 privileged detection-coordinate diagnostics."""

from __future__ import annotations

import numpy as np


def replace_detection_xyxy(
    detections: np.ndarray,
    detection_index: int,
    replacement_xyxy: np.ndarray,
) -> np.ndarray:
    """Copy a detection matrix and replace one XYXY box, preserving all else."""
    array = np.asarray(detections)
    replacement = np.asarray(replacement_xyxy, dtype=array.dtype)
    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError("detections must have shape [N,5]")
    if not 0 <= detection_index < len(array):
        raise IndexError("detection index out of range")
    if replacement.shape != (4,) or not np.all(np.isfinite(replacement)):
        raise ValueError("replacement XYXY must contain four finite values")
    if replacement[2] <= replacement[0] or replacement[3] <= replacement[1]:
        raise ValueError("replacement XYXY must have positive area")
    corrected = array.copy()
    corrected[detection_index, :4] = replacement
    return corrected
