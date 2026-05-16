import numpy as np
import pytest

from features import FEATURE_NAMES, featurize_coords


def _sample_coords(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((21, 3))


def test_feature_count_matches_names():
    out = featurize_coords(_sample_coords())
    assert out.shape == (288,)
    assert len(FEATURE_NAMES) == 288


def test_no_nans_for_random_inputs():
    for s in range(10):
        out = featurize_coords(_sample_coords(s))
        assert np.isfinite(out).all(), f"NaN/Inf at seed {s}"


def test_distances_are_rotation_invariant():
    coords = _sample_coords(1)
    theta = np.deg2rad(37.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = coords.copy()
    rotated[:, :2] = coords[:, :2] @ R.T

    a = featurize_coords(coords)
    b = featurize_coords(rotated)

    # 63 raw coords differ; 210 distances + 15 angles must match
    np.testing.assert_allclose(a[63:], b[63:], atol=1e-8)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        featurize_coords(np.zeros((20, 3)))
