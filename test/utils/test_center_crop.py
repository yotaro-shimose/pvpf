from typing import Tuple

import numpy as np
import pytest
from pvpf.utils.center_crop import center_crop


@pytest.mark.parametrize(
    "feature_shape, image_size",
    [((100, 301, 201, 17), (100, 100)), ((100, 300, 200, 17), (100, 100))],
)
def test_image_size(
    feature_shape: Tuple[int, int, int, int], image_size: Tuple[int, int]
):
    target = np.random.normal(np.zeros(feature_shape), np.ones(feature_shape))
    cropped = center_crop(target, image_size)
    expected_shape = (feature_shape[0],) + image_size + (feature_shape[-1],)
    assert cropped.shape == expected_shape
