from typing import Tuple

import numpy as np


def center_crop(target: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    assert len(target.shape) == 4
    assert np.all(np.array(target.shape[1:3]) > np.array(image_size))
    left_edge = (target.shape[1] - image_size[0]) // 2
    right_edge = (target.shape[1] - image_size[0]) // 2 + int(
        (target.shape[1] - image_size[0]) % 2 != 0
    )
    top_edge = (target.shape[2] - image_size[1]) // 2
    bottom_edge = (target.shape[2] - image_size[1]) // 2 + int(
        (target.shape[2] - image_size[1]) % 2 != 0
    )
    return target[:, left_edge:-right_edge, top_edge:-bottom_edge]
