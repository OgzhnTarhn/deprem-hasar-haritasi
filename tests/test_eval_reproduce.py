"""Tests for the Phase 3 reproduction runner (``hasar.eval.reproduce``).

Everything runs offline on the CPU with tiny synthetic 32x32 tiles and a fake model defined here;
no real BRIGHT data, no checkpoint and no network. The fake access function stands in for
:func:`hasar.guard.require_reproduction_access`, so the Turkey gate is exercised without touching
git.

The module under test does not exist yet: the import at the top is the expected red.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

torch = pytest.importorskip("torch")

from hasar.data.io import tile_paths  # noqa: E402
from hasar.eval.metrics import (  # noqa: E402
    TOTAL_KEY,
    building_miou,
    iou_per_class,
    miou,
)
from hasar.eval.reproduce import reproduce  # noqa: E402
from hasar.guard import REPRODUCTION_CHECKPOINTS, TurkeyAccess, TurkeyLockError  # noqa: E402
from hasar.models.bright_baselines import BASELINES  # noqa: E402
from hasar.results import read_result  # noqa: E402

SIZE = 32
NUM_CLF = 4
NUM_LOC = 2
KEY = "unet"
TURKEY_NAMES = ["turkey-earthquake_00000001", "turkey-earthquake_00000002"]
VAL_NAMES = ["noto-earthquake_00000001", "noto-earthquake_00000002"]
ACCESS = TurkeyAccess(tag="faz3-yeniden-uretim", commit="abc123")
TOLERANCE_PCT = 0.3


class FakeModel(torch.nn.Module):
    """1x1 convolution on the concatenated pre/post stack; flip-equivariant by construction."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.clf = torch.nn.Conv2d(6, NUM_CLF, kernel_size=1)
        self.loc = torch.nn.Conv2d(6, NUM_LOC, kernel_size=1)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        features = torch.cat([pre, post], dim=1)
        return {"clf": self.clf(features), "loc": self.loc(features)}


class AsymmetricModel(FakeModel):
    """Adds a horizontal ramp to the input, so a horizontal flip is not a symmetry."""

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        ramp = torch.linspace(0.0, 1.0, pre.shape[-1]).view(1, 1, 1, -1)
        return super().forward(pre + ramp, post)


def make_tile(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a synthetic ``(pre, post, label)`` tile with a nodata corner."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 4, size=(SIZE, SIZE), dtype=np.uint8)

    pre = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    pre[..., 0] = label * 10
    pre[..., 1] = 1
    pre[..., 2] = 2
    post = (label * 10 + 100).astype(np.uint8)

    pre[:4, :4] = 0
    post[:4, :4] = 0
    label[:4, :4] = 2
    return pre, post, label


def write_tile(root: Path, name: str, seed: int = 0) -> None:
    """Write a synthetic tile under ``root`` in the layout of :func:`hasar.data.io.tile_paths`."""
    pre, post, label = make_tile(seed)
    paths = tile_paths(root, name)
    for path, array in ((paths.pre, pre), (paths.post, post), (paths.target, label)):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, array)


def write_tiles(root: Path, names: list[str]) -> None:
    for seed, name in enumerate(names):
        write_tile(root, name, seed=seed)


def fake_model_fn(key: str) -> FakeModel:
    """Return a tiny fake model; the key is ignored but recorded by the caller."""
    assert key in BASELINES
    return FakeModel()


def fake_access_fn(repo_root: Path | str, checkpoint_name: str) -> TurkeyAccess:
    """Stand in for :func:`hasar.guard.require_reproduction_access`."""
    return ACCESS


def expected_values(cm: np.ndarray) -> dict[str, float]:
    """Return the percent metrics that must be derived from a confusion matrix."""
    ious = iou_per_class(cm)
    return {
        "miou_pct": miou(cm) * 100,
        "iou_background_pct": float(ious[0]) * 100,
        "iou_intact_pct": float(ious[1]) * 100,
        "iou_damaged_pct": float(ious[2]) * 100,
        "iou_destroyed_pct": float(ious[3]) * 100,
        "building_miou_pct": building_miou(cm) * 100,
    }


def load_total(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.array(data[TOTAL_KEY], dtype=np.int64)


# --------------------------------------------------------------------------- test split


def test_test_split_writes_npz_and_json(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_tiles(data_root, TURKEY_NAMES)

    record = reproduce(
        KEY,
        data_root,
        out_dir,
        split="test",
        names=TURKEY_NAMES,
        checkpoint_path=None,
        repo_root=tmp_path,
        access_fn=fake_access_fn,
        model_fn=fake_model_fn,
    )

    npz_path = out_dir / f"{KEY}_test_matrices.npz"
    json_path = out_dir / f"{KEY}_test.json"
    assert npz_path.is_file()
    assert json_path.is_file()

    with np.load(npz_path) as data:
        assert TOTAL_KEY in data.files
        for name in TURKEY_NAMES:
            assert name in data.files

    assert read_result(json_path) == record


def test_test_split_record_fields(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_tiles(data_root, TURKEY_NAMES)

    record = reproduce(
        KEY,
        data_root,
        out_dir,
        split="test",
        names=TURKEY_NAMES,
        checkpoint_path=None,
        repo_root=tmp_path,
        access_fn=fake_access_fn,
        model_fn=fake_model_fn,
    )

    assert record.metric_tag == "[ZS-EVENT mIoU, repro]"
    assert record.model == BASELINES[KEY].registry_name
    assert record.split == "zero_shot/test"
    assert record.commit == ACCESS.commit
    assert record.config_hash == "none"

    assert "checkpoint" in record.extra
    assert record.extra["tiles"] == len(TURKEY_NAMES)
    assert record.extra["tta"] is False
    assert record.extra["tolerance_pct"] == TOLERANCE_PCT
    assert isinstance(record.extra["within_tolerance"], bool)


def test_test_split_values_match_saved_matrices(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_tiles(data_root, TURKEY_NAMES)

    record = reproduce(
        KEY,
        data_root,
        out_dir,
        split="test",
        names=TURKEY_NAMES,
        checkpoint_path=None,
        repo_root=tmp_path,
        access_fn=fake_access_fn,
        model_fn=fake_model_fn,
    )

    cm = load_total(out_dir / f"{KEY}_test_matrices.npz")
    expected = expected_values(cm)
    for name, value in expected.items():
        assert record.values[name] == pytest.approx(value, abs=1e-6)

    published = BASELINES[KEY].published_turkey
    assert record.values["published_pct"] == pytest.approx(published)
    assert record.values["diff_pct"] == pytest.approx(record.values["miou_pct"] - published)
    assert record.extra["within_tolerance"] == (abs(record.values["diff_pct"]) <= TOLERANCE_PCT)


def test_test_split_requests_access_with_baseline_checkpoint(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_tiles(data_root, TURKEY_NAMES)
    calls: list[tuple[Path | str, str]] = []

    def access_fn(repo_root: Path | str, checkpoint_name: str) -> TurkeyAccess:
        calls.append((repo_root, checkpoint_name))
        return ACCESS

    reproduce(
        KEY,
        data_root,
        out_dir,
        split="test",
        names=TURKEY_NAMES,
        checkpoint_path=None,
        repo_root=tmp_path,
        access_fn=access_fn,
        model_fn=fake_model_fn,
    )

    assert len(calls) == 1
    assert Path(calls[0][1]).name in REPRODUCTION_CHECKPOINTS


def test_test_split_tta_is_recorded_and_changes_matrices(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_tiles(data_root, TURKEY_NAMES)

    plain = reproduce(
        KEY,
        data_root,
        tmp_path / "plain",
        split="test",
        names=TURKEY_NAMES,
        checkpoint_path=None,
        repo_root=tmp_path,
        access_fn=fake_access_fn,
        model_fn=lambda key: AsymmetricModel(),
    )
    tta = reproduce(
        KEY,
        data_root,
        tmp_path / "tta",
        split="test",
        names=TURKEY_NAMES,
        checkpoint_path=None,
        repo_root=tmp_path,
        tta=True,
        access_fn=fake_access_fn,
        model_fn=lambda key: AsymmetricModel(),
    )

    assert plain.extra["tta"] is False
    assert tta.extra["tta"] is True
    plain_cm = load_total(tmp_path / "plain" / f"{KEY}_test_matrices.npz")
    tta_cm = load_total(tmp_path / "tta" / f"{KEY}_test_matrices.npz")
    assert not np.array_equal(plain_cm, tta_cm)


# --------------------------------------------------------------------------- val split


def test_val_split_does_not_request_access(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_tiles(data_root, VAL_NAMES)

    def access_fn(repo_root: Path | str, checkpoint_name: str) -> TurkeyAccess:
        raise AssertionError("val bölmesi Türkiye erişimi istememeli")

    record = reproduce(
        KEY,
        data_root,
        out_dir,
        split="val",
        names=VAL_NAMES,
        checkpoint_path=None,
        access_fn=access_fn,
        model_fn=fake_model_fn,
    )

    assert (out_dir / f"{KEY}_val_matrices.npz").is_file()
    assert (out_dir / f"{KEY}_val.json").is_file()
    assert record.metric_tag == "[VAL13 mIoU, repro]"
    assert record.split == "zero_shot/val"
    assert isinstance(record.commit, str) and record.commit
    assert record.config_hash == "none"
    assert record.extra["tiles"] == len(VAL_NAMES)
    assert record.extra["tta"] is False


def test_val_split_values_match_saved_matrices(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_tiles(data_root, VAL_NAMES)

    record = reproduce(
        KEY,
        data_root,
        out_dir,
        split="val",
        names=VAL_NAMES,
        checkpoint_path=None,
        access_fn=fake_access_fn,
        model_fn=fake_model_fn,
    )

    cm = load_total(out_dir / f"{KEY}_val_matrices.npz")
    expected = expected_values(cm)
    for name, value in expected.items():
        assert record.values[name] == pytest.approx(value, abs=1e-6)

    # The published Turkey number and its difference belong to the test split only.
    assert "published_pct" not in record.values
    assert "diff_pct" not in record.values
    assert "tolerance_pct" not in record.extra
    assert "within_tolerance" not in record.extra


def test_val_split_rejects_turkey_name(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_tiles(data_root, TURKEY_NAMES[:1])

    with pytest.raises(TurkeyLockError):
        reproduce(
            KEY,
            data_root,
            tmp_path / "out",
            split="val",
            names=TURKEY_NAMES[:1],
            checkpoint_path=None,
            access_fn=fake_access_fn,
            model_fn=fake_model_fn,
        )
