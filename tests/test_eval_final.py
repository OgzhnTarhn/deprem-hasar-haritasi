"""Tests for the single Turkey evaluation (``hasar.eval.final``).

Protocol sections 3, 6, 7, 8, 10 and 11: the ``final`` step reads the Turkey tiles exactly once,
behind the protocol lock, and writes the headline record, the pre-declared secondary rows, the
per-tile confusion matrices, the building-level F1 and the calibration record. Everything here runs
offline on the CPU with tiny synthetic 32x32 GeoTIFF tiles and a fake model defined in this file; no
real BRIGHT data, no checkpoint and no network. The fake access function stands in for
:func:`hasar.guard.require_turkey_access`, so the Turkey gate is exercised without touching git.

The module under test does not exist yet: the import at the top is the expected red.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

torch = pytest.importorskip("torch")

from hasar.data.dataset import BrightDataset  # noqa: E402
from hasar.data.io import tile_paths  # noqa: E402
from hasar.eval.bootstrap import anchor_verdict, block_bootstrap  # noqa: E402
from hasar.eval.final import run_final  # noqa: E402
from hasar.eval.metrics import (  # noqa: E402
    TOTAL_KEY,
    building_miou,
    iou_per_class,
    miou,
)
from hasar.eval.predict import evaluate, predict  # noqa: E402
from hasar.eval.rows import ROWS  # noqa: E402
from hasar.guard import TurkeyAccess, TurkeyLockError  # noqa: E402
from hasar.results import read_result  # noqa: E402

SIZE = 32
NUM_CLF = 4
NUM_LOC = 2
NAMES = ["turkey-earthquake_00000001", "turkey-earthquake_00000002", "turkey-earthquake_00000003"]
ACCESS = TurkeyAccess(tag="protokol-kilidi", commit="abc")
MODEL_NAME = "ana"
CONFIG_HASH = "cfg123"
N_BOOT = 200
ANCHOR = 41.28
SQUARE = [[2, 2, 10, 2, 10, 10, 2, 10]]


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


class CountingModel(FakeModel):
    """Counts every forward pass, so a test can prove the model was not run."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.calls = 0

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        self.calls += 1
        return super().forward(pre, post)


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


def annotation(seg: str, *, ann_id: int = 1, category_id: int = 1) -> dict:
    """Return a minimal COCO-like annotation dict."""
    return {
        "id": ann_id,
        "image_id": 1,
        "category_id": category_id,
        "segmentation": seg,
        "area": 64.0,
        "bbox": [2.0, 2.0, 8.0, 8.0],
        "iscrowd": 0,
    }


def write_instances(directory: Path, name: str) -> None:
    """Write a tiny CVPRW26-style ``<name>_instance_damage.json`` file."""
    payload = {
        "images": [{"id": 1, "file_name": f"{name}.png"}],
        "annotations": [annotation(json.dumps(SQUARE), ann_id=1, category_id=2)],
        "categories": [],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}_instance_damage.json").write_text(json.dumps(payload), encoding="utf-8")


def fake_access_fn(repo_root: Path | str) -> TurkeyAccess:
    """Stand in for :func:`hasar.guard.require_turkey_access`."""
    return ACCESS


def run(tmp_path: Path, model: torch.nn.Module | None = None, **kwargs) -> dict:
    """Write the synthetic tiles and call :func:`run_final` with the test defaults."""
    data_root = tmp_path / "data"
    write_tiles(data_root, NAMES)
    params: dict = {
        "model_name": MODEL_NAME,
        "config_hash": CONFIG_HASH,
        "names": NAMES,
        "repo_root": tmp_path,
        "access_fn": fake_access_fn,
        "n_boot": N_BOOT,
    }
    params.update(kwargs)
    return run_final(
        FakeModel() if model is None else model,
        data_root,
        params.pop("out_dir", tmp_path / "out"),
        params.pop("cache_dir", tmp_path / "cache"),
        **params,
    )


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


def load_tile_matrices(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.array(data[name], dtype=np.int64) for name in data.files if name != TOTAL_KEY}


# --------------------------------------------------------------------------- records and files


def test_writes_headline_and_secondary_records(tmp_path: Path) -> None:
    records = run(tmp_path)
    out_dir = tmp_path / "out"

    assert set(records) == set(ROWS)
    assert (out_dir / "turkiye.json").is_file()
    for row in ROWS:
        assert (out_dir / f"turkiye_kare_matrisleri_{row}.npz").is_file(), row
        if row != "headline":
            assert (out_dir / f"turkiye_{row}.json").is_file(), row

    assert read_result(out_dir / "turkiye.json") == records["headline"]
    for row in ROWS:
        if row != "headline":
            assert read_result(out_dir / f"turkiye_{row}.json") == records[row]


def test_metric_tags(tmp_path: Path) -> None:
    records = run(tmp_path)

    assert records["headline"].metric_tag == "[ZS-EVENT mIoU]"
    for row in ROWS:
        if row != "headline":
            assert records[row].metric_tag == f"[ZS-EVENT mIoU, {row}]"


def test_headline_record_provenance(tmp_path: Path) -> None:
    records = run(tmp_path)
    record = records["headline"]

    assert record.model == MODEL_NAME
    assert record.config_hash == CONFIG_HASH
    assert record.commit == ACCESS.commit
    assert record.extra["anchor"] == pytest.approx(ANCHOR)
    assert record.extra["tiles"] == len(NAMES)
    assert record.extra["n_boot"] == N_BOOT
    assert record.extra["seed"] == 0
    # The access token's tag and commit are recorded somewhere in ``extra``.
    serialized = json.dumps(record.extra, default=str)
    assert ACCESS.tag in serialized
    assert ACCESS.commit in serialized


def test_headline_values_match_saved_matrices(tmp_path: Path) -> None:
    records = run(tmp_path)

    cm = load_total(tmp_path / "out" / "turkiye_kare_matrisleri_headline.npz")
    for name, value in expected_values(cm).items():
        assert records["headline"].values[name] == pytest.approx(value, abs=1e-6)


def test_secondary_row_values_match_their_matrices(tmp_path: Path) -> None:
    records = run(tmp_path)

    for row in ROWS:
        if row == "headline":
            continue
        cm = load_total(tmp_path / "out" / f"turkiye_kare_matrisleri_{row}.npz")
        assert records[row].values["miou_pct"] == pytest.approx(miou(cm) * 100, abs=1e-6), row


def test_headline_equals_evaluate(tmp_path: Path) -> None:
    model = FakeModel()
    records = run(tmp_path, model=model)

    dataset = BrightDataset(tmp_path / "data", NAMES, mode="eval", sar_mode="official", access=ACCESS)
    accumulator = evaluate(model, dataset)

    assert records["headline"].values["miou_pct"] == pytest.approx(miou(accumulator.total) * 100, abs=1e-6)


# --------------------------------------------------------------------------- access and lock


def test_access_is_requested_before_reading_tiles(tmp_path: Path) -> None:
    calls: list[Path | str] = []

    def access_fn(repo_root: Path | str) -> TurkeyAccess:
        calls.append(repo_root)
        raise TurkeyLockError("kilit yok")

    with pytest.raises(TurkeyLockError):
        run(tmp_path, access_fn=access_fn)

    assert len(calls) == 1
    assert not (tmp_path / "out" / "turkiye.json").exists()
    cache_dir = tmp_path / "cache"
    assert not cache_dir.exists() or not list(cache_dir.glob("*.npz"))


def test_existing_result_raises_without_model_call(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "turkiye.json").write_text("{}", encoding="utf-8")
    model = CountingModel()

    with pytest.raises(TurkeyLockError):
        run(tmp_path, model=model, out_dir=out_dir)

    assert model.calls == 0


# --------------------------------------------------------------------------- single-inference cache


def test_cache_files_hold_float16_probabilities(tmp_path: Path) -> None:
    run(tmp_path)

    for name in NAMES:
        path = tmp_path / "cache" / f"{name}.npz"
        assert path.is_file(), name
        with np.load(path) as data:
            assert set(data.files) == {"clf", "loc", "clf_tta", "loc_tta", "valid"}
            assert data["clf"].dtype == np.float16
            assert data["loc"].dtype == np.float16
            assert data["clf_tta"].dtype == np.float16
            assert data["loc_tta"].dtype == np.float16
            assert data["valid"].dtype == np.bool_
            assert data["clf"].shape == (NUM_CLF, SIZE, SIZE)
            assert data["loc"].shape == (NUM_LOC, SIZE, SIZE)
            assert data["clf_tta"].shape == (NUM_CLF, SIZE, SIZE)
            assert data["loc_tta"].shape == (NUM_LOC, SIZE, SIZE)
            assert data["valid"].shape == (SIZE, SIZE)


def test_cache_holds_plain_and_tta_probabilities(tmp_path: Path) -> None:
    model = AsymmetricModel()
    run(tmp_path, model=model)

    item = BrightDataset(tmp_path / "data", NAMES, mode="eval", sar_mode="official", access=ACCESS)[0]
    pre = torch.from_numpy(np.asarray(item["pre"])).unsqueeze(0)
    post = torch.from_numpy(np.asarray(item["post"])).unsqueeze(0)
    plain_clf, plain_loc = predict(model, pre, post, tta=False)
    tta_clf, tta_loc = predict(model, pre, post, tta=True)

    with np.load(tmp_path / "cache" / f"{NAMES[0]}.npz") as data:
        assert np.allclose(data["clf"].astype(np.float32), plain_clf[0].numpy(), atol=1e-3)
        assert np.allclose(data["loc"].astype(np.float32), plain_loc[0].numpy(), atol=1e-3)
        assert np.allclose(data["clf_tta"].astype(np.float32), tta_clf[0].numpy(), atol=1e-3)
        assert np.allclose(data["loc_tta"].astype(np.float32), tta_loc[0].numpy(), atol=1e-3)


def test_existing_cache_is_reused(tmp_path: Path) -> None:
    run(tmp_path)
    model = CountingModel()

    run(tmp_path, model=model, out_dir=tmp_path / "out2")

    assert model.calls == 0


def test_cached_tile_does_not_call_model(tmp_path: Path) -> None:
    run(tmp_path)
    (tmp_path / "cache" / f"{NAMES[0]}.npz").unlink()
    model = CountingModel()

    run(tmp_path, model=model, out_dir=tmp_path / "out2")

    # One plain pass plus the four flip views of the single uncached tile.
    assert model.calls == 5


# --------------------------------------------------------------------------- bootstrap


def test_blocks_none_uses_one_block_per_tile(tmp_path: Path) -> None:
    records = run(tmp_path)

    tiles = load_tile_matrices(tmp_path / "out" / "turkiye_kare_matrisleri_headline.npz")
    expected = block_bootstrap(tiles, {name: ("all", name) for name in NAMES}, n=N_BOOT, seed=0)

    assert records["headline"].values["ci_low_pct"] == pytest.approx(expected.low * 100, abs=1e-6)
    assert records["headline"].values["ci_high_pct"] == pytest.approx(expected.high * 100, abs=1e-6)


def test_verdict_comes_from_anchor_verdict(tmp_path: Path) -> None:
    records = run(tmp_path)
    record = records["headline"]

    expected = anchor_verdict(
        record.values["miou_pct"],
        record.values["ci_low_pct"],
        record.values["ci_high_pct"],
        anchor=ANCHOR,
    )
    assert record.extra["verdict"] == expected


# --------------------------------------------------------------------------- building F1


def test_instances_write_building_f1(tmp_path: Path) -> None:
    instances_dir = tmp_path / "instances"
    write_instances(instances_dir, NAMES[0])

    run(tmp_path, instances_dir=instances_dir)

    path = tmp_path / "out" / "turkiye_bina_f1.json"
    assert path.is_file()
    record = read_result(path)
    assert record.metric_tag == "[ZS-EVENT bina-makro-F1]"
    for key in ("macro_f1_pct", "f1_intact_pct", "f1_damaged_pct", "f1_destroyed_pct"):
        assert key in record.values, key
    assert record.extra["tiles_with_instances"] == 1


def test_building_tiles_mismatch_raises_before_writing(tmp_path: Path) -> None:
    instances_dir = tmp_path / "instances"
    write_instances(instances_dir, NAMES[0])
    listing = tmp_path / "bina_kareleri.json"
    listing.write_text(json.dumps({"tiles": [NAMES[1]]}), encoding="utf-8")

    with pytest.raises(ValueError):
        run(tmp_path, instances_dir=instances_dir, building_tiles=listing)

    assert not (tmp_path / "out" / "turkiye.json").exists()
    cache_dir = tmp_path / "cache"
    assert not cache_dir.exists() or not list(cache_dir.glob("*.npz"))


def test_building_tiles_matching_list_runs(tmp_path: Path) -> None:
    instances_dir = tmp_path / "instances"
    write_instances(instances_dir, NAMES[0])
    listing = tmp_path / "bina_kareleri.json"
    listing.write_text(json.dumps({"tiles": [NAMES[0]]}), encoding="utf-8")

    records = run(tmp_path, instances_dir=instances_dir, building_tiles=listing)

    assert (tmp_path / "out" / "turkiye_bina_f1.json").is_file()
    assert set(records) == set(ROWS)


# --------------------------------------------------------------------------- calibration


def test_temperature_writes_calibration(tmp_path: Path) -> None:
    records = run(tmp_path, temperature=1.5)

    path = tmp_path / "out" / "turkiye_kalibrasyon.json"
    assert path.is_file()
    record = read_result(path)
    assert record.metric_tag == "[ZS-EVENT mIoU, kalibrasyon]"
    assert record.values["temperature"] == pytest.approx(1.5)
    assert "ece_pct" in record.values
    # The headline mIoU is unchanged, so the tag stays a valid metric record.
    assert record.values["miou_pct"] == pytest.approx(records["headline"].values["miou_pct"], abs=1e-6)


def test_crash_after_inference_leaves_no_headline_and_resumes(tmp_path: Path, monkeypatch) -> None:
    import hasar.eval.final as final_module

    def boom(*args, **kwargs):
        raise RuntimeError("bina F1 çöktü")

    monkeypatch.setattr(final_module, "building_f1", boom)
    instances = tmp_path / "instances"
    for name in NAMES:
        write_instances(instances, name)
    with pytest.raises(RuntimeError):
        run(tmp_path, instances_dir=instances, building_tiles=None)
    # The lock file is written last, so the crashed run can be repeated from the cache.
    assert not (tmp_path / "out" / "turkiye.json").exists()
    assert sorted(p.stem for p in (tmp_path / "cache").glob("*.npz")) == sorted(NAMES)

    monkeypatch.undo()
    model = CountingModel()
    records = run(tmp_path, model=model, instances_dir=instances, building_tiles=None)
    assert model.calls == 0
    assert (tmp_path / "out" / "turkiye.json").is_file()
    assert "headline" in records
