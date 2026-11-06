"""Tests for the input transforms (``hasar.data.transforms``).

Every test uses tiny synthetic arrays built from ``np.arange`` so the geometry of a crop or a
flip/rotation can be verified exactly. No data files and no network are needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from hasar.data.transforms import augment, random_crop, random_flip_rot

H, W = 6, 8


def base_grid() -> np.ndarray:
    """Return an ``H x W`` array whose values identify each pixel."""
    return np.arange(H * W, dtype=np.int64).reshape(H, W)


def square_grid() -> np.ndarray:
    """Return an ``H x H`` array whose values identify each pixel."""
    return np.arange(H * H, dtype=np.int64).reshape(H, H)


def assert_fresh(out: np.ndarray, *inputs: np.ndarray) -> None:
    """Assert ``out`` is a new C-contiguous array that shares no memory with ``inputs``."""
    assert isinstance(out, np.ndarray)
    assert out.flags["C_CONTIGUOUS"]
    for inp in inputs:
        assert not np.shares_memory(out, inp)


# --------------------------------------------------------------------------- random_crop


def test_random_crop_same_window_for_all_arrays() -> None:
    base = base_grid()
    pre = np.stack([base, base + 1000], axis=-1)
    post = base + 2000
    label = base + 3000

    out_pre, out_post, out_label = random_crop([pre, post, label], size=4, rng=np.random.default_rng(0))

    assert out_pre.shape == (4, 4, 2)
    assert out_post.shape == (4, 4)
    assert out_label.shape == (4, 4)
    # The same window was cut from every array: the offsets between the arrays survive.
    np.testing.assert_array_equal(out_pre[..., 0], out_label - 3000)
    np.testing.assert_array_equal(out_pre[..., 1], out_label - 3000 + 1000)
    np.testing.assert_array_equal(out_post, out_label - 3000 + 2000)
    # The crop is a contiguous block of the original grid.
    flat = out_label.reshape(-1)
    np.testing.assert_array_equal(np.diff(flat[:4]), np.ones(3))
    assert flat[4] - flat[3] == W - 3


def test_random_crop_returns_new_arrays() -> None:
    base = base_grid()
    pre = np.stack([base, base + 1000], axis=-1)
    label = base.copy()

    out_pre, out_label = random_crop([pre, label], size=4, rng=np.random.default_rng(1))

    assert_fresh(out_pre, pre)
    assert_fresh(out_label, label)
    assert out_pre.dtype == pre.dtype
    assert out_label.dtype == label.dtype


def test_random_crop_full_size_returns_full_array() -> None:
    base = square_grid()
    label = base.copy()

    (out,) = random_crop([label], size=H, rng=np.random.default_rng(2))

    np.testing.assert_array_equal(out, label)
    assert_fresh(out, label)


def test_random_crop_rejects_size_larger_than_height() -> None:
    with pytest.raises(ValueError):
        random_crop([base_grid()], size=H + 1, rng=np.random.default_rng(3))


def test_random_crop_rejects_size_larger_than_width() -> None:
    with pytest.raises(ValueError):
        random_crop([base_grid()], size=W + 1, rng=np.random.default_rng(4))


def test_random_crop_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        random_crop([base_grid(), base_grid()[: H - 1]], size=4, rng=np.random.default_rng(5))

    with pytest.raises(ValueError):
        random_crop([base_grid(), base_grid()[:, : W - 1]], size=4, rng=np.random.default_rng(6))


# ---------------------------------------------------------------------- random_flip_rot


def test_random_flip_rot_disabled_is_identity() -> None:
    base = base_grid()
    pre = np.stack([base, base + 1000], axis=-1)
    label = base + 2000

    out_pre, out_label = random_flip_rot([pre, label], np.random.default_rng(0), flips=False, rot90=False)

    np.testing.assert_array_equal(out_pre, pre)
    np.testing.assert_array_equal(out_label, label)
    assert_fresh(out_pre, pre)
    assert_fresh(out_label, label)


def test_random_flip_rot_applies_identical_ops_to_every_array() -> None:
    base = square_grid()
    pre = np.stack([base, base + 1000, base + 2000], axis=-1)
    label = base + 3000

    for seed in range(20):
        out_pre, out_label = random_flip_rot([pre, label], np.random.default_rng(seed))

        assert out_pre.shape == pre.shape
        assert out_label.shape == label.shape
        # Channel axis untouched: the per-channel offsets are preserved.
        np.testing.assert_array_equal(out_pre[..., 1], out_pre[..., 0] + 1000)
        np.testing.assert_array_equal(out_pre[..., 2], out_pre[..., 0] + 2000)
        # Same geometry for the label as for the first channel of pre.
        np.testing.assert_array_equal(out_label, out_pre[..., 0] + 3000)
        assert_fresh(out_pre, pre)
        assert_fresh(out_label, label)


def test_random_flip_rot_covers_all_dihedral_results() -> None:
    arr = np.array([[0, 1], [2, 3]], dtype=np.int64)
    rng = np.random.default_rng(0)

    seen = set()
    for _ in range(200):
        (out,) = random_flip_rot([arr], rng)
        assert out.shape == arr.shape
        seen.add(tuple(out.reshape(-1).tolist()))

    assert len(seen) == 8


def test_random_flip_rot_uses_both_flip_axes() -> None:
    arr = np.array([[0, 1], [2, 3]], dtype=np.int64)
    rng = np.random.default_rng(0)

    seen = set()
    for _ in range(200):
        (out,) = random_flip_rot([arr], rng, rot90=False)
        seen.add(tuple(out.reshape(-1).tolist()))

    assert seen == {
        (0, 1, 2, 3),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 1, 0),
    }


# ----------------------------------------------------------------------------- augment


def test_augment_returns_tuple_with_extras() -> None:
    base = base_grid()
    pre = np.stack([base, base + 1000], axis=-1)
    post = base + 2000
    label = base + 3000
    valid = base % 2 == 0

    out = augment(pre, post, label, np.random.default_rng(0), crop=4, extra=(valid,))

    assert isinstance(out, tuple)
    assert len(out) == 4
    out_pre, out_post, out_label, out_valid = out
    assert out_pre.shape == (4, 4, 2)
    assert out_post.shape == (4, 4)
    assert out_label.shape == (4, 4)
    assert out_valid.shape == (4, 4)
    assert out_valid.dtype == np.bool_
    # Same geometry for the extra mask as for the label.
    np.testing.assert_array_equal(out_valid, out_label % 2 == 0)
    np.testing.assert_array_equal(out_pre[..., 0], out_label - 3000)
    np.testing.assert_array_equal(out_post, out_label - 3000 + 2000)
    assert_fresh(out_pre, pre)
    assert_fresh(out_post, post)
    assert_fresh(out_label, label)
    assert_fresh(out_valid, valid)


def test_augment_without_extras_returns_three_arrays() -> None:
    base = base_grid()
    pre = np.stack([base, base + 1000], axis=-1)
    post = base + 2000
    label = base + 3000

    out = augment(pre, post, label, np.random.default_rng(0), crop=4)

    assert isinstance(out, tuple)
    assert len(out) == 3


def test_augment_is_deterministic_for_same_seed() -> None:
    base = base_grid()
    pre = np.stack([base, base + 1000], axis=-1)
    post = base + 2000
    label = base + 3000
    valid = base % 2 == 0

    first = augment(pre, post, label, np.random.default_rng(7), crop=4, extra=(valid,))
    second = augment(pre, post, label, np.random.default_rng(7), crop=4, extra=(valid,))

    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_augment_keeps_label_consistent_with_inputs() -> None:
    base = base_grid()
    pre = np.stack([base, base + 1000], axis=-1)
    post = base.copy()
    label = base.copy()

    for seed in range(10):
        out_pre, out_post, out_label = augment(pre, post, label, np.random.default_rng(seed), crop=4)

        np.testing.assert_array_equal(out_label, out_pre[..., 0])
        np.testing.assert_array_equal(out_label, out_post)


def test_augment_identity_when_disabled_and_full_crop() -> None:
    base = square_grid()
    pre = np.stack([base, base + 1000], axis=-1)
    post = base + 2000
    label = base + 3000

    out_pre, out_post, out_label = augment(pre, post, label, np.random.default_rng(0), crop=H, flips=False, rot90=False)

    np.testing.assert_array_equal(out_pre, pre)
    np.testing.assert_array_equal(out_post, post)
    np.testing.assert_array_equal(out_label, label)
