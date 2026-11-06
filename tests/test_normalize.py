"""Tests for optical/SAR normalization (``hasar.data.normalize``).

All arrays are tiny and synthetic; nothing here touches the dataset or the network.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hasar.data.normalize import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SarStats,
    compute_sar_stats,
    normalize_optical,
    sar_to_input,
)

from hasar.guard import TurkeyLockError


def test_imagenet_constants() -> None:
    assert IMAGENET_MEAN == (123.675, 116.28, 103.53)
    assert IMAGENET_STD == (58.395, 57.12, 57.375)


def test_normalize_optical_constant_image() -> None:
    pre = np.full((2, 3, 3), 100, dtype=np.uint8)

    out = normalize_optical(pre)

    assert out.shape == (3, 2, 3)
    assert out.dtype == np.float32
    for channel in range(3):
        expected = (100 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert np.allclose(out[channel], expected)


def test_normalize_optical_per_channel_values() -> None:
    pre = np.zeros((1, 1, 3), dtype=np.uint8)
    pre[0, 0] = (10, 20, 30)

    out = normalize_optical(pre)

    assert out.shape == (3, 1, 1)
    assert out.dtype == np.float32
    for channel, value in enumerate((10, 20, 30)):
        expected = (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert np.allclose(out[channel, 0, 0], expected)


def test_sar_stats_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(SarStats)
    stats = SarStats(mean=1.5, std=2.5)
    assert stats.mean == 1.5
    assert stats.std == 2.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.mean = 0.0  # type: ignore[misc]


def test_sar_stats_to_dict_round_trip() -> None:
    stats = SarStats(mean=-3.25, std=7.5)

    payload = stats.to_dict()
    restored = SarStats.from_dict(payload)

    assert restored == stats
    assert restored.mean == pytest.approx(-3.25)
    assert restored.std == pytest.approx(7.5)


def test_sar_stats_rejects_non_positive_std() -> None:
    with pytest.raises(ValueError):
        SarStats(mean=0.0, std=0.0)
    with pytest.raises(ValueError):
        SarStats(mean=0.0, std=-1.0)


def test_sar_to_input_official_constant_image() -> None:
    post = np.full((2, 3), 100, dtype=np.uint8)

    out = sar_to_input(post, mode="official")

    assert out.shape == (3, 2, 3)
    assert out.dtype == np.float32
    expected0 = (100 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert np.allclose(out[0], expected0)
    for channel in range(3):
        expected = (100 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert np.allclose(out[channel], expected)


def test_sar_to_input_train_stats() -> None:
    post = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    stats = SarStats(mean=25.0, std=5.0)

    out = sar_to_input(post, mode="train_stats", stats=stats)

    assert out.shape == (1, 2, 2)
    assert out.dtype == np.float32
    assert np.allclose(out[0], (post - 25.0) / 5.0)


def test_sar_to_input_train_stats_requires_stats() -> None:
    post = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        sar_to_input(post, mode="train_stats")


def test_sar_to_input_unknown_mode() -> None:
    post = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        sar_to_input(post, mode="nope")


def _tiles() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "tile_a": (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[True, True], [False, True]]),
        ),
        "tile_b": (
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            np.array([[True, False], [True, True]]),
        ),
        "tile_c": (
            np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
            np.array([[False, True], [True, True]]),
        ),
    }


def test_compute_sar_stats_matches_numpy() -> None:
    tiles = _tiles()

    def load(name: str) -> tuple[np.ndarray, np.ndarray]:
        return tiles[name]

    stats = compute_sar_stats(list(tiles), load)

    valid = np.concatenate([tiles[name][0][tiles[name][1]] for name in tiles]).astype(np.float64)
    assert isinstance(stats, SarStats)
    assert stats.mean == pytest.approx(float(np.mean(valid)))
    assert stats.std == pytest.approx(float(np.std(valid)))


def test_compute_sar_stats_rejects_turkey_without_loading() -> None:
    calls: list[str] = []

    def load(name: str) -> tuple[np.ndarray, np.ndarray]:
        calls.append(name)
        return np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=bool)

    with pytest.raises(TurkeyLockError):
        compute_sar_stats(["tile_a", "turkey-earthquake_00000001"], load)

    assert calls == []


def test_compute_sar_stats_rejects_empty_names() -> None:
    def load(name: str) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError("load must not be called")

    with pytest.raises(ValueError):
        compute_sar_stats([], load)


def test_compute_sar_stats_rejects_zero_valid_pixels() -> None:
    def load(name: str) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=bool)

    with pytest.raises(ValueError):
        compute_sar_stats(["tile_a"], load)
