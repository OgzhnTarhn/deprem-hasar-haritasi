"""Tests for the cache-based analysis runner (``hasar.analysis.runner``).

After the single Turkey evaluation (``final`` step) every analysis reads the per-tile cache only;
the model is never rerun. The runner refuses to start unless the headline record
``<final_dir>/turkiye.json`` exists, builds the building records from the cached flip-TTA
probabilities and the instance polygons, and writes the Y2 analyst-budget record, the
risk-coverage curve and the error gallery.

Everything here is tiny synthetic numpy data: two 32x32 Turkey tiles written as GeoTIFFs, hand-made
``.npz`` cache files and CVPRW26-style instance JSONs. No torch, no real BRIGHT data, no network.

The module under test does not exist yet: the import at the top is the expected red.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from hasar.analysis.runner import run_cache_analyses
from hasar.data.io import tile_paths
from hasar.guard import TurkeyAccess, TurkeyLockError
from hasar.results import ResultRecord, read_result, write_result

SIZE = 32
NUM_CLF = 4
NUM_LOC = 2
NAMES = ["turkey-earthquake_00000001", "turkey-earthquake_00000002"]
ACCESS = TurkeyAccess(tag="protokol-kilidi", commit="abc")
BUDGET_PCTS = (5, 10, 20)
RANKINGS = ("belirsizlik", "hasar-olasiligi", "rastgele")
#: Two non-overlapping squares, one per building instance.
SQUARES = [
    [[2, 2, 10, 2, 10, 10, 2, 10]],
    [[14, 14, 22, 14, 22, 22, 14, 22]],
]


# --------------------------------------------------------------------------- synthetic tiles


def make_label(seed: int) -> np.ndarray:
    """Return a 32x32 damage label with two building squares (classes 1-3)."""
    label = np.zeros((SIZE, SIZE), dtype=np.uint8)
    label[2:11, 2:11] = 2 if seed == 0 else 1
    label[14:23, 14:23] = 3 if seed == 0 else 2
    return label


def write_tile(root: Path, name: str, seed: int = 0) -> None:
    """Write a synthetic tile under ``root`` in the layout of :func:`hasar.data.io.tile_paths`."""
    label = make_label(seed)
    pre = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    pre[..., 0] = label * 10
    pre[..., 1] = 1
    pre[..., 2] = 2
    post = (label * 10 + 100).astype(np.uint8)

    paths = tile_paths(root, name)
    for path, array in ((paths.pre, pre), (paths.post, post), (paths.target, label)):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, array)


def make_cache(seed: int) -> dict[str, np.ndarray]:
    """Return the cached arrays of one tile: perfect on tile 0, all-intact on the others."""
    label = make_label(seed)
    pred = label if seed == 0 else np.ones_like(label)

    clf = np.zeros((NUM_CLF, SIZE, SIZE), dtype=np.float32)
    for cls in range(NUM_CLF):
        clf[cls] = (pred == cls).astype(np.float32)

    loc = np.zeros((NUM_LOC, SIZE, SIZE), dtype=np.float32)
    loc[0] = (pred == 0).astype(np.float32)
    loc[1] = (pred > 0).astype(np.float32)

    return {
        "clf": clf.astype(np.float16),
        "loc": loc.astype(np.float16),
        "clf_tta": clf.astype(np.float16),
        "loc_tta": loc.astype(np.float16),
        "valid": np.ones((SIZE, SIZE), dtype=np.bool_),
    }


def annotation(seg: str, ann_id: int) -> dict:
    """Return a minimal COCO-like annotation dict."""
    return {
        "id": ann_id,
        "image_id": 1,
        "category_id": 1,
        "segmentation": seg,
        "area": 81.0,
        "bbox": [2.0, 2.0, 9.0, 9.0],
        "iscrowd": 0,
    }


def write_instances(directory: Path, name: str) -> None:
    """Write a tiny CVPRW26-style ``<name>_instance_damage.json`` file with two buildings."""
    payload = {
        "images": [{"id": 1, "file_name": f"{name}.png"}],
        "annotations": [annotation(json.dumps(SQUARES[0]), 1), annotation(json.dumps(SQUARES[1]), 2)],
        "categories": [],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}_instance_damage.json").write_text(json.dumps(payload), encoding="utf-8")


def headline_record() -> ResultRecord:
    """Return a valid headline record standing in for the single Turkey evaluation."""
    return ResultRecord(
        metric_tag="[ZS-EVENT mIoU]",
        values={"miou_pct": 41.28},
        model="ana",
        split="zero_shot/test",
        commit=ACCESS.commit,
        config_hash="cfg123",
    )


def setup_tree(tmp_path: Path, names: list[str] | None = None) -> dict[str, Path]:
    """Write the tiles, cache, instances and the fake headline record; return the paths."""
    names = NAMES if names is None else names
    tree = {
        "data_root": tmp_path / "data",
        "cache_dir": tmp_path / "cache",
        "instances_dir": tmp_path / "instances",
        "final_dir": tmp_path / "final",
        "out_dir": tmp_path / "out",
    }
    tree["cache_dir"].mkdir(parents=True)
    for seed, name in enumerate(names):
        write_tile(tree["data_root"], name, seed=seed)
        write_instances(tree["instances_dir"], name)
        np.savez(tree["cache_dir"] / f"{name}.npz", **make_cache(seed))
    tree["final_dir"].mkdir(parents=True)
    write_result(tree["final_dir"] / "turkiye.json", headline_record())
    return tree


def run(tree: dict[str, Path], **kwargs) -> dict[str, Path]:
    """Call :func:`run_cache_analyses` with the test defaults."""
    return run_cache_analyses(
        tree["cache_dir"],
        tree["data_root"],
        tree["instances_dir"],
        tree["out_dir"],
        final_dir=tree["final_dir"],
        access=ACCESS,
        **kwargs,
    )


# --------------------------------------------------------------------------- outputs


def test_writes_every_output(tmp_path: Path) -> None:
    tree = setup_tree(tmp_path)

    result = run(tree)

    out_dir = tree["out_dir"]
    assert (out_dir / "y2_butce.json").is_file()
    assert (out_dir / "y2_risk_kapsam.json").is_file()
    assert (out_dir / "galeri" / "index.json").is_file()

    assert isinstance(result, dict)
    assert all(isinstance(path, Path) for path in result.values())
    assert all(path.exists() for path in result.values())
    returned = {path.resolve() for path in result.values()}
    for path in (out_dir / "y2_butce.json", out_dir / "y2_risk_kapsam.json", out_dir / "galeri" / "index.json"):
        assert path.resolve() in returned, path


def test_budget_record_round_trips(tmp_path: Path) -> None:
    tree = setup_tree(tmp_path)

    run(tree)

    record = read_result(tree["out_dir"] / "y2_butce.json")
    assert record.metric_tag == "[ZS-EVENT bina-makro-F1, analist-butce]"
    assert record.commit == ACCESS.commit
    assert record.values
    assert all(np.isfinite(value) for value in record.values.values())


def test_budget_values_are_flattened(tmp_path: Path) -> None:
    tree = setup_tree(tmp_path)

    run(tree)

    values = read_result(tree["out_dir"] / "y2_butce.json").values
    for ranking in RANKINGS:
        for pct in BUDGET_PCTS:
            key = f"{ranking}_{pct}_macro_f1_pct"
            assert key in values, key
            assert 0.0 <= values[key] <= 100.0, key

    assert "aurc" in values
    assert 0.0 <= values["aurc"] <= 1.0
    # No DPM layer in the cache, so the DPM ranking must be skipped.
    assert not any(key.startswith("dpm_") for key in values)


def test_risk_coverage_json(tmp_path: Path) -> None:
    tree = setup_tree(tmp_path)

    run(tree)

    payload = json.loads((tree["out_dir"] / "y2_risk_kapsam.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    coverage = payload["coverage"]
    risk = payload["risk"]
    assert isinstance(coverage, list) and isinstance(risk, list)
    assert len(coverage) == len(risk) == 4  # two buildings per tile
    assert coverage[-1] == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in coverage)
    assert all(0.0 <= value <= 1.0 for value in risk)


def test_gallery_writes_worst_tiles(tmp_path: Path) -> None:
    tree = setup_tree(tmp_path)

    run(tree)

    gallery = tree["out_dir"] / "galeri"
    images = sorted(gallery.glob("*.png"))
    assert len(images) == len(NAMES)

    index = json.loads((gallery / "index.json").read_text(encoding="utf-8"))
    serialized = json.dumps(index)
    for name in NAMES:
        assert name in serialized, name


def test_gallery_k_limits_tiles(tmp_path: Path) -> None:
    tree = setup_tree(tmp_path)

    run(tree, gallery_k=1)

    assert len(list((tree["out_dir"] / "galeri").glob("*.png"))) == 1


def test_cache_without_instances_is_skipped(tmp_path: Path) -> None:
    tree = setup_tree(tmp_path)
    orphan = "turkey-earthquake_00000003"
    write_tile(tree["data_root"], orphan, seed=2)
    np.savez(tree["cache_dir"] / f"{orphan}.npz", **make_cache(2))

    run(tree)

    assert len(list((tree["out_dir"] / "galeri").glob("*.png"))) == len(NAMES)


# --------------------------------------------------------------------------- lock


def test_missing_headline_raises(tmp_path: Path) -> None:
    tree = setup_tree(tmp_path)
    (tree["final_dir"] / "turkiye.json").unlink()

    with pytest.raises(TurkeyLockError):
        run(tree)

    assert not (tree["out_dir"] / "y2_butce.json").exists()
    assert not (tree["out_dir"] / "y2_risk_kapsam.json").exists()
