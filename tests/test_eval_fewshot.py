"""Tests for the Türkiye few-shot evaluation and comparison module (``hasar.eval.fewshot_eval``).

Design sections 5.5, 9.4, 9.7, 9.8 and 9.9: all tests run offline on CPU with tiny synthetic
32x32 GeoTIFF tiles, fake models and synthetic confusion-matrix archives under ``tmp_path``.
No real BRIGHT data, no checkpoint and no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml

torch = pytest.importorskip("torch")

import hasar.results as results_mod  # noqa: E402
from hasar.config import Config, run_id  # noqa: E402
from hasar.data.io import tile_paths  # noqa: E402
from hasar.eval.fewshot_eval import (  # noqa: E402
    BUILDING_FILE,
    BUILDING_TAG,
    CLASS_DIFF_TAG,
    DIFF_TAG,
    DONE_FILE,
    FS_BASE_TAGS,
    MATRICES_FILE,
    SPLIT,
    TAG,
    TEST_FILE,
    ZS_TAG,
    compare_fewshot,
    load_tile_matrices,
    require_runs_done,
    run_fewshot_test,
    slice_matrices,
)
from hasar.eval.metrics import TOTAL_KEY, ConfusionAccumulator, iou_per_class, miou  # noqa: E402
from hasar.fewshot.split import _list_sha256  # noqa: E402
from hasar.guard import FewshotAccess, TurkeyAccess, TurkeyLockError  # noqa: E402
from hasar.results import read_result  # noqa: E402

SIZE = 32
NUM_CLF = 4
NUM_LOC = 2
TEST_NAMES = [
    "turkey-earthquake_00000001",
    "turkey-earthquake_00000002",
    "turkey-earthquake_00000003",
]
EXTRA_TILE = "turkey-earthquake_00000099"
BLOCKS = {
    "turkey-earthquake_00000001": ("sehir_a", "1_1"),
    "turkey-earthquake_00000002": ("sehir_a", "1_2"),
    "turkey-earthquake_00000003": ("sehir_b", "2_1"),
    "turkey-earthquake_00000099": ("sehir_b", "2_2"),
}
N_BOOT = 200
SQUARE = [[2, 2, 10, 2, 10, 10, 2, 10]]


@pytest.fixture(autouse=True)
def _register_fewshot_base_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Temporarily add few-shot base tags until ``hasar.results.BASE_TAGS`` includes them."""
    missing = tuple(tag for tag in FS_BASE_TAGS if tag not in results_mod.BASE_TAGS)
    if missing:
        monkeypatch.setattr(results_mod, "BASE_TAGS", results_mod.BASE_TAGS + missing)


class FakeModel(torch.nn.Module):
    """Tiny 1x1 conv model matching the ``{"clf", "loc"}`` contract."""

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


class CountingModel(FakeModel):
    """Counts every forward call so tests can prove zero inference on early failure."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.calls = 0

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        self.calls += 1
        return super().forward(pre, post)


def make_tile(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a synthetic ``(pre, post, label)`` tile."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 4, size=(SIZE, SIZE), dtype=np.uint8)
    pre = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    pre[..., 0] = label * 10
    pre[..., 1] = 1
    pre[..., 2] = 2
    post = (label * 10 + 100).astype(np.uint8)
    return pre, post, label


def write_tiles(root: Path, names: list[str]) -> None:
    for seed, name in enumerate(names):
        pre, post, label = make_tile(seed)
        paths = tile_paths(root, name)
        for path, array in ((paths.pre, pre), (paths.post, post), (paths.target, label)):
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(path, array)


def write_instances(directory: Path, name: str) -> None:
    payload = {
        "images": [{"id": 1, "file_name": f"{name}.png"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "segmentation": json.dumps(SQUARE),
                "area": 64.0,
                "bbox": [2.0, 2.0, 8.0, 8.0],
                "iscrowd": 0,
            }
        ],
        "categories": [],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}_instance_damage.json").write_text(json.dumps(payload), encoding="utf-8")


def make_test_access(names: list[str] = TEST_NAMES, purpose: str = "test") -> FewshotAccess:
    return FewshotAccess(
        tag="fewshot-kilidi",
        commit="deadbeef1234",
        purpose=purpose,
        allowed=frozenset(names),
    )


def make_archive(path: Path, per_tile: dict[str, np.ndarray]) -> Path:
    acc = ConfusionAccumulator()
    for name, cm in per_tile.items():
        mat = np.asarray(cm, dtype=np.int64)
        acc.per_tile[name] = mat
        acc.total += mat
    acc.save_npz(path)
    return path


# --------------------------------------------------------------------------- require_runs_done


def test_require_runs_done_refuses_when_one_lacks_done(tmp_path: Path) -> None:
    cfg_a = Config(experiment="fewshot_a")
    cfg_b = Config(experiment="fewshot_b")
    runs_dir = tmp_path / "runs"

    dir_a = runs_dir / run_id(cfg_a)
    dir_a.mkdir(parents=True)
    (dir_a / "DONE").write_text("ok", encoding="utf-8")
    (dir_a / "best.pt").write_bytes(b"ckpt")

    dir_b = runs_dir / run_id(cfg_b)
    dir_b.mkdir(parents=True)
    (dir_b / "best.pt").write_bytes(b"ckpt")

    with pytest.raises(RuntimeError, match="b"):
        require_runs_done(runs_dir, {"a": cfg_a, "b": cfg_b})


def test_require_runs_done_refuses_when_best_pt_missing(tmp_path: Path) -> None:
    cfg_a = Config(experiment="fewshot_a")
    cfg_b = Config(experiment="fewshot_b")
    runs_dir = tmp_path / "runs"

    for cfg, with_ckpt in ((cfg_a, True), (cfg_b, False)):
        run_dir = runs_dir / run_id(cfg)
        run_dir.mkdir(parents=True)
        (run_dir / "DONE").write_text("ok", encoding="utf-8")
        if with_ckpt:
            (run_dir / "best.pt").write_bytes(b"ckpt")

    with pytest.raises(RuntimeError, match="best.pt"):
        require_runs_done(runs_dir, {"a": cfg_a, "b": cfg_b})


def test_require_runs_done_returns_paths_for_config_and_yaml(tmp_path: Path) -> None:
    cfg_a = Config(experiment="fewshot_a")
    yaml_b = tmp_path / "b.yaml"
    yaml_b.write_text(yaml.safe_dump({"experiment": "fewshot_b"}), encoding="utf-8")
    cfg_b = Config(experiment="fewshot_b")

    runs_dir = tmp_path / "runs"
    for cfg in (cfg_a, cfg_b):
        run_dir = runs_dir / run_id(cfg)
        run_dir.mkdir(parents=True)
        (run_dir / "DONE").write_text("ok", encoding="utf-8")
        (run_dir / "best.pt").write_bytes(b"ckpt")

    got = require_runs_done(runs_dir, {"a": cfg_a, "b": yaml_b})
    assert got == {
        "a": runs_dir / run_id(cfg_a) / "best.pt",
        "b": runs_dir / run_id(cfg_b) / "best.pt",
    }


# --------------------------------------------------------------------------- run_fewshot_test guards


def test_run_fewshot_test_refuses_adapt_purpose_and_plain_access(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_tiles(data_root, TEST_NAMES)
    instances_dir = tmp_path / "instances"
    write_instances(instances_dir, TEST_NAMES[0])
    sha = _list_sha256(sorted(TEST_NAMES))

    model = CountingModel()
    with pytest.raises(TurkeyLockError):
        run_fewshot_test(
            model,
            data_root,
            tmp_path / "out1",
            run_id="fewshot_a-123",
            split_sha256=sha,
            model_name="fewshot-a",
            config_hash="cfg_a",
            access=make_test_access(purpose="adapt"),
            blocks=BLOCKS,
            instances_dir=instances_dir,
            building_tiles=[TEST_NAMES[0]],
            n_boot=N_BOOT,
        )
    assert model.calls == 0

    plain_access = TurkeyAccess(tag="protokol-kilidi", commit="abc")
    with pytest.raises(TypeError):
        run_fewshot_test(
            model,
            data_root,
            tmp_path / "out2",
            run_id="fewshot_a-123",
            split_sha256=sha,
            model_name="fewshot-a",
            config_hash="cfg_a",
            access=plain_access,  # type: ignore[arg-type]
            blocks=BLOCKS,
            instances_dir=instances_dir,
            building_tiles=[TEST_NAMES[0]],
            n_boot=N_BOOT,
        )
    assert model.calls == 0


def test_run_fewshot_test_refuses_when_done_exists(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_tiles(data_root, TEST_NAMES)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / DONE_FILE).write_text("{}", encoding="utf-8")
    model = CountingModel()

    with pytest.raises(RuntimeError):
        run_fewshot_test(
            model,
            data_root,
            out_dir,
            run_id="fewshot_a-123",
            split_sha256=_list_sha256(sorted(TEST_NAMES)),
            model_name="fewshot-a",
            config_hash="cfg_a",
            access=make_test_access(),
            blocks=BLOCKS,
            instances_dir=tmp_path / "instances",
            building_tiles=[],
            n_boot=N_BOOT,
        )
    assert model.calls == 0


def test_run_fewshot_test_refuses_split_sha256_mismatch(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_tiles(data_root, TEST_NAMES)
    model = CountingModel()

    with pytest.raises(ValueError):
        run_fewshot_test(
            model,
            data_root,
            tmp_path / "out",
            run_id="fewshot_a-123",
            split_sha256="0" * 64,
            model_name="fewshot-a",
            config_hash="cfg_a",
            access=make_test_access(),
            blocks=BLOCKS,
            instances_dir=tmp_path / "instances",
            building_tiles=[],
            n_boot=N_BOOT,
        )
    assert model.calls == 0


def test_run_fewshot_test_refuses_missing_instance_in_intersection(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_tiles(data_root, TEST_NAMES)
    instances_dir = tmp_path / "instances"
    write_instances(instances_dir, TEST_NAMES[0])
    model = CountingModel()

    # TEST_NAMES[1] is in test AND in building_tiles, but its instance file is missing.
    with pytest.raises(FileNotFoundError):
        run_fewshot_test(
            model,
            data_root,
            tmp_path / "out",
            run_id="fewshot_a-123",
            split_sha256=_list_sha256(sorted(TEST_NAMES)),
            model_name="fewshot-a",
            config_hash="cfg_a",
            access=make_test_access(),
            blocks=BLOCKS,
            instances_dir=instances_dir,
            building_tiles=[TEST_NAMES[0], TEST_NAMES[1]],
            n_boot=N_BOOT,
        )
    assert model.calls == 0


def test_run_fewshot_test_ignores_building_tile_outside_test(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_tiles(data_root, TEST_NAMES)
    instances_dir = tmp_path / "instances"
    write_instances(instances_dir, TEST_NAMES[0])
    listing = tmp_path / "bina_kareleri.json"
    # EXTRA_TILE is in building_tiles but NOT in test; its missing instance file must not fail.
    listing.write_text(json.dumps({"tiles": [TEST_NAMES[0], EXTRA_TILE]}), encoding="utf-8")

    records = run_fewshot_test(
        FakeModel(),
        data_root,
        tmp_path / "out",
        run_id="fewshot_a-123",
        split_sha256=_list_sha256(sorted(TEST_NAMES)),
        model_name="fewshot-a",
        config_hash="cfg_a",
        access=make_test_access(),
        blocks=BLOCKS,
        instances_dir=instances_dir,
        building_tiles=listing,
        n_boot=N_BOOT,
    )
    assert "bina_f1" in records
    assert records["bina_f1"].extra["tiles_with_instances"] == 1


# --------------------------------------------------------------------------- run_fewshot_test happy path


def test_run_fewshot_test_happy_path(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    # Write an extra Turkey tile on disk to verify only access.allowed tiles are evaluated.
    write_tiles(data_root, [*TEST_NAMES, EXTRA_TILE])
    instances_dir = tmp_path / "instances"
    write_instances(instances_dir, TEST_NAMES[0])
    write_instances(instances_dir, TEST_NAMES[2])

    out_dir = tmp_path / "out"
    sha = _list_sha256(sorted(TEST_NAMES))
    access = make_test_access()
    model = CountingModel()

    records = run_fewshot_test(
        model,
        data_root,
        out_dir,
        run_id="fewshot_a-123",
        split_sha256=sha,
        model_name="fewshot-a",
        config_hash="cfg_a",
        access=access,
        blocks=BLOCKS,
        instances_dir=instances_dir,
        building_tiles=[TEST_NAMES[0], TEST_NAMES[2], EXTRA_TILE],
        n_boot=N_BOOT,
        seed=7,
    )

    assert model.calls == len(TEST_NAMES)

    npz_path = out_dir / MATRICES_FILE
    assert npz_path.is_file()
    with np.load(npz_path) as data:
        assert set(data.files) == {TOTAL_KEY, *TEST_NAMES}
        total_cm = np.array(data[TOTAL_KEY], dtype=np.int64)

    test_record = read_result(out_dir / TEST_FILE)
    assert test_record == records["test"]
    assert test_record.metric_tag == TAG == "[FS-EVENT mIoU]"
    assert test_record.split == SPLIT == "fewshot/test"
    assert test_record.model == "fewshot-a"
    assert test_record.commit == access.commit
    assert test_record.config_hash == "cfg_a"
    assert test_record.seed == 7
    expected_keys = {
        "miou_pct",
        "iou_background_pct",
        "iou_intact_pct",
        "iou_damaged_pct",
        "iou_destroyed_pct",
        "ci_low_pct",
        "ci_high_pct",
    }
    assert set(test_record.values) == expected_keys
    assert test_record.values["miou_pct"] == pytest.approx(miou(total_cm) * 100.0, abs=1e-6)
    assert test_record.extra["run_id"] == "fewshot_a-123"
    assert test_record.extra["split_sha256"] == sha
    assert test_record.extra["access"] == {"tag": access.tag, "commit": access.commit}
    assert test_record.extra["tiles"] == len(TEST_NAMES)
    assert test_record.extra["n_boot"] == N_BOOT
    assert test_record.extra["seed"] == 7
    assert "anchor" not in test_record.extra

    bina_record = read_result(out_dir / BUILDING_FILE)
    assert bina_record == records["bina_f1"]
    assert bina_record.metric_tag == BUILDING_TAG == "[FS-EVENT bina-makro-F1]"
    assert bina_record.extra["tiles_with_instances"] == 2
    for key in ("macro_f1_pct", "f1_intact_pct", "f1_damaged_pct", "f1_destroyed_pct", "buildings", "missed"):
        assert key in bina_record.values

    done_payload = json.loads((out_dir / DONE_FILE).read_text(encoding="utf-8"))
    assert done_payload["run_id"] == "fewshot_a-123"
    assert done_payload["split_sha256"] == sha
    assert done_payload["tiles"] == len(TEST_NAMES)


def test_run_fewshot_test_skips_building_file_when_intersection_empty(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_tiles(data_root, TEST_NAMES)
    out_dir = tmp_path / "out"

    records = run_fewshot_test(
        FakeModel(),
        data_root,
        out_dir,
        run_id="fewshot_a-123",
        split_sha256=_list_sha256(sorted(TEST_NAMES)),
        model_name="fewshot-a",
        config_hash="cfg_a",
        access=make_test_access(),
        blocks=BLOCKS,
        instances_dir=tmp_path / "instances",
        building_tiles=[EXTRA_TILE],
        n_boot=N_BOOT,
    )
    assert "bina_f1" not in records
    assert not (out_dir / BUILDING_FILE).exists()
    assert (out_dir / DONE_FILE).is_file()


# --------------------------------------------------------------------------- slice_matrices & load_tile_matrices


def test_slice_matrices_and_missing_key(tmp_path: Path) -> None:
    cm1 = np.diag([10, 5, 3, 2]).astype(np.int64)
    cm2 = np.diag([8, 4, 2, 1]).astype(np.int64)
    cm3 = np.diag([6, 3, 1, 1]).astype(np.int64)
    archive = make_archive(
        tmp_path / "mats.npz",
        {TEST_NAMES[0]: cm1, TEST_NAMES[1]: cm2, EXTRA_TILE: cm3},
    )

    sliced = slice_matrices(archive, [TEST_NAMES[0], TEST_NAMES[1]])
    assert set(sliced) == {TEST_NAMES[0], TEST_NAMES[1]}
    assert np.array_equal(sliced[TEST_NAMES[0]], cm1)
    assert sliced[TEST_NAMES[0]].dtype == np.int64

    loaded = load_tile_matrices(archive)
    assert set(loaded) == {TEST_NAMES[0], TEST_NAMES[1], EXTRA_TILE}
    assert TOTAL_KEY not in loaded

    with pytest.raises(KeyError, match=TEST_NAMES[2]):
        slice_matrices(archive, [TEST_NAMES[0], TEST_NAMES[2]])


# --------------------------------------------------------------------------- compare_fewshot


def test_compare_fewshot_direction_tags_and_classes(tmp_path: Path) -> None:
    # Build synthetic confusion matrices where:
    #   B (highest) > A (middle) > zs-ana (lowest) on mIoU, damaged IoU and destroyed IoU.
    cm_zs = np.array(
        [
            [80, 5, 5, 5],
            [5, 40, 10, 5],
            [5, 10, 10, 10],
            [5, 5, 10, 10],
        ],
        dtype=np.int64,
    )
    cm_a = np.array(
        [
            [85, 4, 3, 3],
            [4, 50, 4, 2],
            [3, 5, 25, 2],
            [3, 2, 5, 20],
        ],
        dtype=np.int64,
    )
    cm_b = np.array(
        [
            [90, 2, 2, 1],
            [2, 54, 2, 2],
            [1, 2, 30, 2],
            [1, 1, 2, 26],
        ],
        dtype=np.int64,
    )

    a_npz = make_archive(tmp_path / "a.npz", {name: cm_a for name in TEST_NAMES})
    b_npz = make_archive(tmp_path / "b.npz", {name: cm_b for name in TEST_NAMES})
    # zs_npz contains EXTRA_TILE too, simulating the full 1114-tile Turkey archive.
    zs_npz = make_archive(
        tmp_path / "zs.npz",
        {**{name: cm_zs for name in TEST_NAMES}, EXTRA_TILE: cm_b},
    )

    out_dir = tmp_path / "kiyas"
    records = compare_fewshot(
        a_npz,
        b_npz,
        zs_npz,
        TEST_NAMES,
        BLOCKS,
        out_dir,
        commit="deadbeef1234",
        n_boot=N_BOOT,
        seed=0,
    )

    expected_stems = {"b_vs_a", "a_vs_zs-ana", "b_vs_zs-ana", "b_vs_a_siniflar", "zs_ana"}
    assert set(records) == expected_stems
    for stem in expected_stems:
        assert read_result(out_dir / f"{stem}.json") == records[stem]

    # b_vs_a: B - A > 0
    b_vs_a = records["b_vs_a"]
    assert b_vs_a.metric_tag == DIFF_TAG == "[FS-EVENT mIoU, eslestirilmis-fark]"
    assert b_vs_a.split == SPLIT
    assert b_vs_a.values["diff_pct"] > 0
    assert b_vs_a.extra["direction"] == "fewshot-b - fewshot-a"

    # a_vs_zs-ana: A - zs-ana > 0, carries design 9.7 note
    a_vs_zs = records["a_vs_zs-ana"]
    assert a_vs_zs.metric_tag == DIFF_TAG
    assert a_vs_zs.values["diff_pct"] > 0
    assert a_vs_zs.extra["note"] == "Türkiye uyarlama etiketleri + ek 4000 adım eğitim"

    # b_vs_zs-ana: B - zs-ana > 0
    b_vs_zs = records["b_vs_zs-ana"]
    assert b_vs_zs.metric_tag == DIFF_TAG
    assert b_vs_zs.values["diff_pct"] > a_vs_zs.values["diff_pct"]

    # Check sign flips when A is worse than zs-ana
    out_dir_rev = tmp_path / "kiyas_rev"
    rev_records = compare_fewshot(
        zs_npz_small := make_archive(tmp_path / "a_weak.npz", {name: cm_zs for name in TEST_NAMES}),
        b_npz,
        make_archive(tmp_path / "zs_strong.npz", {name: cm_a for name in TEST_NAMES}),
        TEST_NAMES,
        BLOCKS,
        out_dir_rev,
        commit="deadbeef1234",
        n_boot=N_BOOT,
        seed=0,
    )
    assert zs_npz_small.is_file()
    assert rev_records["a_vs_zs-ana"].values["diff_pct"] < 0

    # b_vs_a_siniflar: per-class damaged (2) and destroyed (3) IoU diff
    sinif = records["b_vs_a_siniflar"]
    assert sinif.metric_tag == CLASS_DIFF_TAG == "[FS-EVENT IoU, eslestirilmis-fark]"
    for prefix in ("damaged", "destroyed"):
        for suffix in ("a_pct", "b_pct", "diff_pct", "low_pct", "high_pct"):
            assert f"{prefix}_{suffix}" in sinif.values
    assert sinif.values["damaged_diff_pct"] > 0
    assert sinif.values["destroyed_diff_pct"] > 0
    assert sinif.extra["verdict_damaged"] in {"eşit", "farklı"}
    assert sinif.extra["verdict_destroyed"] in {"eşit", "farklı"}

    # zs_ana: sliced zero-shot record on test tiles only
    zs_rec = records["zs_ana"]
    assert zs_rec.metric_tag == ZS_TAG == "[FS-EVENT mIoU, zs-ana]"
    total_zs = cm_zs * len(TEST_NAMES)
    ious_zs = iou_per_class(total_zs)
    assert zs_rec.values["miou_pct"] == pytest.approx(miou(total_zs) * 100.0, abs=1e-6)
    assert zs_rec.values["iou_damaged_pct"] == pytest.approx(float(ious_zs[2]) * 100.0, abs=1e-6)
    assert zs_rec.values["iou_destroyed_pct"] == pytest.approx(float(ious_zs[3]) * 100.0, abs=1e-6)


def test_compare_fewshot_raises_when_a_archive_tile_set_differs(tmp_path: Path) -> None:
    cm = np.diag([10, 5, 3, 2]).astype(np.int64)
    a_npz = make_archive(tmp_path / "a.npz", {TEST_NAMES[0]: cm, TEST_NAMES[1]: cm})
    b_npz = make_archive(tmp_path / "b.npz", {name: cm for name in TEST_NAMES})
    zs_npz = make_archive(tmp_path / "zs.npz", {name: cm for name in TEST_NAMES})

    with pytest.raises(ValueError):
        compare_fewshot(
            a_npz,
            b_npz,
            zs_npz,
            TEST_NAMES,
            BLOCKS,
            tmp_path / "kiyas",
            commit="deadbeef1234",
            n_boot=N_BOOT,
        )


def test_require_runs_done_needs_two_distinct_arms(tmp_path: Path) -> None:
    from hasar.eval.fewshot_eval import require_runs_done

    with pytest.raises(ValueError):
        require_runs_done(tmp_path, {"a": Config()})
    with pytest.raises(ValueError):
        require_runs_done(tmp_path, {"a": Config(), "b": Config()})


def test_compare_fewshot_rejects_duplicate_test_names(tmp_path: Path) -> None:
    from hasar.eval.fewshot_eval import compare_fewshot

    name = "turkey-earthquake_00000001"
    with pytest.raises(ValueError):
        compare_fewshot(
            tmp_path / "a.npz",
            tmp_path / "b.npz",
            tmp_path / "zs.npz",
            [name, name],
            {name: ("Şehir", "0")},
            tmp_path / "out",
            commit="abc",
        )
