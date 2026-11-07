"""Tests for the BRIGHT dataset (``hasar.data.dataset``).

Every test writes tiny synthetic 32x32 tiles into ``tmp_path`` in the layout of
:func:`hasar.data.io.tile_paths`; nothing touches the network or the real data set. The label has
values 0..3 and a 4x4 nodata corner where both modalities are empty and the label is 2.

The module under test does not exist yet: the import at the top is the expected red.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from hasar.data.dataset import BrightDataset
from hasar.data.io import tile_paths
from hasar.data.normalize import IMAGENET_MEAN, IMAGENET_STD, SarStats
from hasar.guard import TurkeyAccess, TurkeyLockError

SIZE = 32
NODATA = 4
TURKEY_NAME = "turkey-earthquake_00000001"
TRAIN_NAME = "noto-earthquake_00000003"

MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32)
STD = np.asarray(IMAGENET_STD, dtype=np.float32)


def make_tile(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a synthetic ``(pre, post, label)`` tile with a nodata corner.

    On valid pixels optical channel 0 is ``label * 10`` and SAR is ``label * 10 + 100``, so the
    alignment of inputs and label can be checked after augmentation. The top-left ``4x4`` corner is
    nodata: all optical channels and SAR are zero and the label is 2.
    """
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 4, size=(SIZE, SIZE), dtype=np.uint8)

    pre = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    pre[..., 0] = label * 10
    pre[..., 1] = 1
    pre[..., 2] = 2
    post = (label * 10 + 100).astype(np.uint8)

    pre[:NODATA, :NODATA] = 0
    post[:NODATA, :NODATA] = 0
    label[:NODATA, :NODATA] = 2
    return pre, post, label


def write_tile(root: Path, name: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write a synthetic tile under ``root`` and return its raw arrays."""
    pre, post, label = make_tile(seed)
    paths = tile_paths(root, name)
    for path, array in ((paths.pre, pre), (paths.post, post), (paths.target, label)):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, array)
    return pre, post, label


def nodata_of(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Return ``True`` where both modalities are empty."""
    return np.all(pre == 0, axis=-1) & (post == 0)


def find_nodata_key(ds: BrightDataset, limit: int = 200) -> tuple[int, int]:
    """Return a train key whose crop contains at least one nodata pixel."""
    for sample_no in range(limit):
        key = (0, sample_no)
        if not ds[key]["valid"].all():
            return key
    raise AssertionError("hiçbir kırpma nodata pikseli içermedi")


def expected_loc(label: np.ndarray) -> np.ndarray:
    """Return the expected localization target: ``(label > 0)``, values >= 4 kept as-is."""
    return np.where(label >= 4, label, (label > 0).astype(np.int64)).astype(np.int64)


# --------------------------------------------------------------------------- length


def test_len_matches_names(tmp_path: Path) -> None:
    names = ["noto-earthquake_00000001", "noto-earthquake_00000002", "noto-earthquake_00000003"]
    for name in names:
        write_tile(tmp_path, name)

    ds = BrightDataset(tmp_path, names, mode="eval")

    assert len(ds) == len(names)


# --------------------------------------------------------------------------- eval


def test_eval_returns_full_tile_dict(tmp_path: Path) -> None:
    pre, post, label = write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="eval")

    out = ds[0]

    assert isinstance(out, dict)
    assert out["name"] == TRAIN_NAME
    assert out["pre"].dtype == np.float32
    assert out["pre"].shape == (3, SIZE, SIZE)
    assert out["post"].dtype == np.float32
    assert out["post"].shape == (3, SIZE, SIZE)
    assert out["clf"].dtype == np.int64
    assert out["clf"].shape == (SIZE, SIZE)
    assert out["loc"].dtype == np.int64
    assert out["loc"].shape == (SIZE, SIZE)
    assert out["valid"].dtype == np.bool_
    assert out["valid"].shape == (SIZE, SIZE)


def test_eval_clf_is_raw_label_including_nodata(tmp_path: Path) -> None:
    pre, post, label = write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="eval")

    out = ds[0]

    np.testing.assert_array_equal(out["clf"], label.astype(np.int64))
    # The nodata corner keeps its label (official eval path).
    np.testing.assert_array_equal(out["clf"][:NODATA, :NODATA], np.full((NODATA, NODATA), 2))


def test_eval_loc_and_valid(tmp_path: Path) -> None:
    pre, post, label = write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="eval")

    out = ds[0]

    np.testing.assert_array_equal(out["loc"], expected_loc(label))
    np.testing.assert_array_equal(out["valid"], ~nodata_of(pre, post))
    assert not out["valid"][:NODATA, :NODATA].any()


def test_eval_is_deterministic(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="eval")

    first = ds[0]
    second = ds[0]

    for key in ("pre", "post", "clf", "loc", "valid"):
        np.testing.assert_array_equal(first[key], second[key])


def test_eval_train_stats_sar_is_single_channel(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)
    stats = SarStats(mean=50.0, std=10.0)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="eval", sar_mode="train_stats", sar_stats=stats)

    out = ds[0]

    assert out["post"].dtype == np.float32
    assert out["post"].shape == (1, SIZE, SIZE)


# --------------------------------------------------------------------------- train


def test_train_crop_size_and_key(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="train", crop=16)

    out = ds[(0, 0)]

    assert out["pre"].shape == (3, 16, 16)
    assert out["post"].shape == (3, 16, 16)
    assert out["clf"].shape == (16, 16)
    assert out["loc"].shape == (16, 16)
    assert out["valid"].shape == (16, 16)


def test_train_nodata_ignore_sets_ignore_index(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="train", crop=16, nodata_ignore=True)

    out = ds[find_nodata_key(ds)]

    nodata = ~out["valid"]
    assert nodata.any()
    np.testing.assert_array_equal(out["clf"][nodata], np.full(nodata.sum(), 255))
    np.testing.assert_array_equal(out["loc"][nodata], np.full(nodata.sum(), 255))


def test_train_without_nodata_ignore_keeps_label(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="train", crop=16, nodata_ignore=False)

    out = ds[find_nodata_key(ds)]

    nodata = ~out["valid"]
    assert nodata.any()
    # Nodata pixels keep their raw label (2) instead of the ignore index.
    np.testing.assert_array_equal(out["clf"][nodata], np.full(nodata.sum(), 2))


def test_train_same_key_is_identical(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="train", crop=16)

    first = ds[(0, 3)]
    second = ds[(0, 3)]

    for key in ("pre", "post", "clf", "loc", "valid"):
        np.testing.assert_array_equal(first[key], second[key])


def test_train_different_sample_no_changes_crop(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="train", crop=16)

    crops = [ds[(0, sample_no)]["clf"] for sample_no in range(5)]

    # Different sample numbers draw (almost surely) different crops; not all may be identical.
    assert any(not np.array_equal(crops[0], other) for other in crops[1:])


def test_train_augmentation_keeps_inputs_and_label_aligned(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="train", crop=16, nodata_ignore=True)

    out = ds[(0, 0)]

    valid = out["valid"]
    assert valid.any()
    clf = out["clf"][valid]

    # Optical channel 0 was label * 10 on valid pixels; undo the ImageNet normalization.
    optical = out["pre"][0][valid] * STD[0] + MEAN[0]
    np.testing.assert_allclose(optical, clf.astype(np.float32) * 10, atol=1e-3)

    # SAR was label * 10 + 100 on valid pixels; official mode copies it to three channels, each
    # normalized with its own ImageNet statistics.
    sar = out["post"][0][valid] * STD[0] + MEAN[0]
    np.testing.assert_allclose(sar, clf.astype(np.float32) * 10 + 100, atol=1e-3)
    for channel in range(1, 3):
        band = out["post"][channel][valid] * STD[channel] + MEAN[channel]
        np.testing.assert_allclose(band, clf.astype(np.float32) * 10 + 100, atol=1e-3)

    # The valid mask stays aligned with the optical channel 1 (1 on valid, 0 on nodata).
    expected_ch1 = np.where(valid, (1.0 - MEAN[1]) / STD[1], (0.0 - MEAN[1]) / STD[1]).astype(np.float32)
    np.testing.assert_allclose(out["pre"][1], expected_ch1, atol=1e-3)


# --------------------------------------------------------------------------- key types


def test_train_rejects_int_key(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="train", crop=16)

    with pytest.raises(TypeError):
        ds[0]


def test_eval_rejects_tuple_key(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)
    ds = BrightDataset(tmp_path, [TRAIN_NAME], mode="eval")

    with pytest.raises(TypeError):
        ds[(0, 0)]


# --------------------------------------------------------------------------- Turkey gate


def test_turkey_locked_without_token(tmp_path: Path) -> None:
    write_tile(tmp_path, TURKEY_NAME)

    with pytest.raises(TurkeyLockError):
        BrightDataset(tmp_path, [TURKEY_NAME], mode="eval")


def test_turkey_never_trains_even_with_token(tmp_path: Path) -> None:
    write_tile(tmp_path, TURKEY_NAME)
    access = TurkeyAccess(tag="protokol-kilidi", commit="abc")

    with pytest.raises(TurkeyLockError):
        BrightDataset(tmp_path, [TURKEY_NAME], mode="train", crop=16, access=access)


def test_turkey_eval_with_token_reads(tmp_path: Path) -> None:
    write_tile(tmp_path, TURKEY_NAME)
    access = TurkeyAccess(tag="protokol-kilidi", commit="abc")

    ds = BrightDataset(tmp_path, [TURKEY_NAME], mode="eval", access=access)
    out = ds[0]

    assert out["name"] == TURKEY_NAME
    assert out["pre"].shape == (3, SIZE, SIZE)


# --------------------------------------------------------------------------- validation


def test_invalid_mode_raises(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)

    with pytest.raises(ValueError):
        BrightDataset(tmp_path, [TRAIN_NAME], mode="bogus")


def test_train_non_positive_crop_raises(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)

    with pytest.raises(ValueError):
        BrightDataset(tmp_path, [TRAIN_NAME], mode="train", crop=0)


def test_train_stats_without_stats_raises(tmp_path: Path) -> None:
    write_tile(tmp_path, TRAIN_NAME)

    with pytest.raises(ValueError):
        BrightDataset(tmp_path, [TRAIN_NAME], mode="eval", sar_mode="train_stats")
