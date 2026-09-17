"""Shared image and text input contracts; never silently fabricate features."""
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError('Missing or unreadable image: {}'.format(path))
    return image.astype(np.float32)


def load_text(path, feature_format, height, width):
    path = Path(path)
    try:
        array = np.load(str(path), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError('Cannot load text feature {}: {}'.format(path, exc)) from exc
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if feature_format == 'legacy_image':
        if array.shape == (height, width):
            array = np.repeat(array[None], 3, axis=0)
        expected = (3, height, width)
    else:
        expected = (64, height // 4, width // 4)
    if array.shape != expected or not np.isfinite(array).all():
        raise ValueError('{}: expected finite {} {}, got {}'.format(
            path, feature_format, expected, array.shape))
    return np.ascontiguousarray(array, dtype=np.float32)
