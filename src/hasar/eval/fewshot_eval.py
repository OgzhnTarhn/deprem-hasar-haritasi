"""Turkey few-shot test evaluation and paired comparisons (design sections 5.5, 9.4, 9.7-9.9).

Protocol 1.4 (§15)
------------------
This module implements the dedicated few-shot test evaluation on the locked Turkey ``test`` split
without reusing :func:`hasar.eval.final.run_final` (design §9.8):

* **Both arms first (§9.4):** :func:`require_runs_done` verifies that every configured arm has both
  ``DONE`` and ``best.pt`` before any Turkey test evaluation starts.
* **Strict pre-inference checks (§9.8, §9.9):** :func:`run_fewshot_test` validates the
  :class:`hasar.guard.FewshotAccess` token (``purpose="test"``), refuses to run twice when
  ``test_done.json`` exists, checks the ``split_sha256`` digest of ``sorted(access.allowed)``,
  verifies that every test tile has a bootstrap block, and verifies that every instance file in
  ``building_tiles ∩ test`` exists on disk—all before the model is called or any file is written.
* **Headline-only float32 path (§9.8):** predictions come from ``argmax(clf)`` on float32
  probabilities with ``tta=False``, no building mask, no temperature scaling, no 41.28 zero-shot
  anchor, and no cache files on the metric path (optional ``pred_dir`` writes per-tile
  predictions for the demo only; metrics are never computed from them).
* **Paired comparisons (§5.5, §9.7):** :func:`compare_fewshot` computes ``B - A`` (primary),
  ``A - zs-ana`` (annotated as adaptation labels + extra 4000 training steps), ``B - zs-ana``,
  per-class damaged/destroyed IoU differences for ``B - A``, and the sliced ``zs-ana`` record on
  the exact same test tiles and city-stratified 2 km blocks.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from hasar.config import Config, load_config
from hasar.config import run_id as config_run_id
from hasar.data.dataset import BrightDataset
from hasar.eval.bootstrap import (
    DEFAULT_N,
    DEFAULT_SEED,
    block_bootstrap,
    equal_verdict,
    paired_block_bootstrap,
    turkey_blocks,
)
from hasar.eval.building import building_classes, building_f1, load_instances, rasterize_instances
from hasar.eval.compare import compare_models
from hasar.eval.metrics import NUM_CLASSES, TOTAL_KEY, ConfusionAccumulator, iou_per_class, miou
from hasar.eval.predict import predict
from hasar.fewshot.split import _list_sha256
from hasar.guard import FewshotAccess, TurkeyLockError
from hasar.results import ResultRecord, write_result

#: Base metric tags required by this module in :data:`hasar.results.BASE_TAGS`.
FS_BASE_TAGS: tuple[str, ...] = (
    "FS-EVENT mIoU",
    "FS-EVENT bina-makro-F1",
    "FS-EVENT IoU",
)

SPLIT = "fewshot/test"
DONE_FILE = "test_done.json"
MATRICES_FILE = "turkiye_fewshot_kare_matrisleri.npz"
TEST_FILE = "test.json"
BUILDING_FILE = "test_bina_f1.json"

TAG = "[FS-EVENT mIoU]"
BUILDING_TAG = "[FS-EVENT bina-makro-F1]"
DIFF_TAG = "[FS-EVENT mIoU, eslestirilmis-fark]"
CLASS_DIFF_TAG = "[FS-EVENT IoU, eslestirilmis-fark]"
ZS_TAG = "[FS-EVENT mIoU, zs-ana]"

_INSTANCE_SUFFIX = "_instance_damage.json"
_A_VS_ZS_NOTE = "Türkiye uyarlama etiketleri + ek 4000 adım eğitim"


def _write_json_atomic(path: Path, payload: dict[str, object]) -> Path:
    """Write ``payload`` as indented UTF-8 JSON to ``path`` atomically via temp file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
    return path


def _write_pred_npz_atomic(path: Path, pred: np.ndarray, entropy: np.ndarray, bina: np.ndarray) -> Path:
    """Write ``pred``, ``entropy`` and ``bina`` arrays to ``path`` atomically via temp file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            np.savez_compressed(handle, pred=pred, entropy=entropy, bina=bina)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
    return path


def _resolve_blocks(
    blocks: Mapping[str, tuple[str, str]] | str | Path,
    names: Sequence[str],
) -> dict[str, tuple[str, str]]:
    """Load or copy ``blocks`` and verify that every tile in ``names`` has a block entry."""
    block_map = turkey_blocks(blocks) if isinstance(blocks, (str, Path)) else dict(blocks)
    missing = [name for name in names if name not in block_map]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Bootstrap blok haritasında eksik kareler var ({len(missing)} adet, ilk örnekler: {preview}).")
    return block_map


def _resolve_building_tile_list(building_tiles: Path | str | Sequence[str]) -> list[str]:
    """Read tile names from a JSON file with a ``tiles`` key or accept an explicit sequence."""
    if isinstance(building_tiles, (str, Path)):
        payload = json.loads(Path(building_tiles).read_text(encoding="utf-8"))
        return [str(name) for name in payload["tiles"]]
    return [str(name) for name in building_tiles]


def _miou_values(cm: np.ndarray, ci_low: float, ci_high: float) -> dict[str, float]:
    """Build the percent mIoU + per-class IoU + bootstrap CI dictionary."""
    ious = iou_per_class(cm)
    return {
        "miou_pct": miou(cm) * 100.0,
        "iou_background_pct": float(ious[0]) * 100.0,
        "iou_intact_pct": float(ious[1]) * 100.0,
        "iou_damaged_pct": float(ious[2]) * 100.0,
        "iou_destroyed_pct": float(ious[3]) * 100.0,
        "ci_low_pct": ci_low * 100.0,
        "ci_high_pct": ci_high * 100.0,
    }


def require_runs_done(
    runs_dir: Path | str,
    configs: Mapping[str, Config | str | Path],
) -> dict[str, Path]:
    """Verify every arm in ``configs`` has finished training and return ``{name: best.pt}``.

    Checks all configured arms before raising so every missing ``DONE`` or ``best.pt`` is
    reported at once (design §9.4: Turkey few-shot test may start only after both arms finish).
    """
    base = Path(runs_dir)
    checkpoints: dict[str, Path] = {}
    missing: list[str] = []

    loaded = {
        name: load_config(cfg_or_path) if isinstance(cfg_or_path, (str, Path)) else cfg_or_path
        for name, cfg_or_path in configs.items()
    }
    run_ids = [config_run_id(cfg) for cfg in loaded.values()]
    if len(loaded) < 2 or len(set(run_ids)) != len(run_ids):
        raise ValueError("Few-shot testi için en az iki farklı kol (ayrı run_id) gereklidir (tasarım §9.4).")

    for name, cfg in loaded.items():
        run_dir = base / config_run_id(cfg)
        done_path = run_dir / "DONE"
        best_path = run_dir / "best.pt"
        lacks: list[str] = []
        if not done_path.is_file():
            lacks.append("DONE")
        if not best_path.is_file():
            lacks.append("best.pt")
        if lacks:
            missing.append(f"{name} ({run_dir}: eksik {', '.join(lacks)})")
        else:
            checkpoints[name] = best_path

    if missing:
        details = "; ".join(missing)
        raise RuntimeError(
            "Türkiye few-shot testi yalnız her iki kolun eğitimi tamamlandıktan sonra başlatılabilir "
            f"(tasarım §9.4). Eksik koşular: {details}."
        )
    return checkpoints


def run_fewshot_test(
    model: torch.nn.Module,
    data_root: Path | str,
    out_dir: Path | str,
    *,
    run_id: str,
    split_sha256: str,
    model_name: str,
    config_hash: str,
    access: FewshotAccess,
    blocks: Mapping[str, tuple[str, str]] | str | Path,
    instances_dir: Path | str,
    building_tiles: Path | str | Sequence[str],
    device: str = "cpu",
    n_boot: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    sar_mode: str = "official",
    pred_dir: Path | str | None = None,
) -> dict[str, ResultRecord]:
    """Run the single Turkey few-shot test evaluation on ``access.allowed`` tiles.

    All guard, lock, split digest, block-map and instance-file checks run before the model is
    called or any output file is written. If the intersection of ``building_tiles`` and the test
    split is empty, ``test_bina_f1.json`` is skipped and omitted from the returned mapping. When
    ``pred_dir`` is given, per-tile ``pred``, ``entropy`` and ``bina`` ``uint8`` arrays are saved
    as compressed ``.npz`` files for the demo without affecting any metric.
    """
    if not isinstance(access, FewshotAccess):
        raise TypeError(f"Few-shot testi için FewshotAccess gereklidir, alınan: {type(access).__name__}.")
    if access.purpose != "test":
        raise TurkeyLockError(f"Few-shot testi yalnız purpose='test' erişimiyle çalışır, alınan: {access.purpose!r}.")

    out_path = Path(out_dir)
    if (out_path / DONE_FILE).exists():
        raise RuntimeError(f"Few-shot testi zaten çalıştırıldı ({out_path / DONE_FILE}): ikinci kez çalıştırılamaz.")

    names = sorted(access.allowed)
    if not names:
        raise ValueError("Few-shot test kare listesi boş olamaz.")

    actual_sha = _list_sha256(names)
    if actual_sha != split_sha256:
        raise ValueError(
            f"Few-shot test bölmesi sha256 özeti uyuşmuyor: beklenen {split_sha256}, hesaplanan {actual_sha}."
        )

    block_map = _resolve_blocks(blocks, names)

    if instances_dir is None:
        raise ValueError("Bina F1 doğrulaması için instances_dir zorunludur.")
    instances_path = Path(instances_dir)
    tile_list = _resolve_building_tile_list(building_tiles)
    inst_names = sorted(set(tile_list) & set(names))
    missing_instances = [name for name in inst_names if not (instances_path / f"{name}{_INSTANCE_SUFFIX}").is_file()]
    if missing_instances:
        preview = ", ".join(missing_instances[:5])
        raise FileNotFoundError(
            f"Bina F1 örnek dosyaları eksik ({len(missing_instances)} adet, ilk örnekler: {preview}; "
            f"klasör: {instances_path}). Few-shot testi başlatılmadı."
        )

    pred_path = Path(pred_dir) if pred_dir is not None else None
    if pred_path is not None:
        pred_path.mkdir(parents=True, exist_ok=True)

    dataset = BrightDataset(data_root, names, mode="eval", sar_mode=sar_mode, access=access)
    model.to(device)
    model.eval()

    accumulator = ConfusionAccumulator()
    inst_set = set(inst_names)
    true_classes: list[np.ndarray] = []
    pred_classes: list[np.ndarray] = []

    for item in dataset:
        name = str(item["name"])
        label = np.asarray(item["clf"])
        pre = torch.from_numpy(np.asarray(item["pre"])).unsqueeze(0).to(device)
        post = torch.from_numpy(np.asarray(item["post"])).unsqueeze(0).to(device)

        clf_prob, loc_prob = predict(model, pre, post, tta=False)
        clf = clf_prob[0].float().cpu().numpy()
        pred = clf.argmax(axis=0)
        accumulator.add(label, pred, name)

        if name in inst_set:
            instance_file = instances_path / f"{name}{_INSTANCE_SUFFIX}"
            instance_map = rasterize_instances(load_instances(instance_file), label.shape)
            true_cls, pred_cls = building_classes(label, instance_map, pred)
            true_classes.append(true_cls)
            pred_classes.append(pred_cls)

        if pred_path is not None:
            clipped_p = np.clip(clf, 1e-12, 1.0)
            norm_entropy = -np.sum(clipped_p * np.log(clipped_p), axis=0) / np.log(NUM_CLASSES)
            entropy_u8 = np.rint(np.clip(norm_entropy, 0.0, 1.0) * 255.0).astype(np.uint8)
            bina_prob = loc_prob[0, 1].float().cpu().numpy()
            bina_u8 = np.clip(np.rint(bina_prob * 255.0), 0, 255).astype(np.uint8)
            _write_pred_npz_atomic(
                pred_path / f"{name}.npz",
                pred.astype(np.uint8),
                entropy_u8,
                bina_u8,
            )

    out_path.mkdir(parents=True, exist_ok=True)
    accumulator.save_npz(out_path / MATRICES_FILE)

    boot = block_bootstrap(accumulator.per_tile, block_map, n=n_boot, seed=seed)
    base_extra: dict[str, object] = {
        "tiles": len(names),
        "n_boot": n_boot,
        "seed": seed,
        "run_id": run_id,
        "split_sha256": split_sha256,
        "access": {"tag": access.tag, "commit": access.commit},
    }

    test_record = ResultRecord(
        metric_tag=TAG,
        values=_miou_values(accumulator.total, boot.low, boot.high),
        model=model_name,
        split=SPLIT,
        commit=access.commit,
        config_hash=config_hash,
        seed=seed,
        extra=dict(base_extra),
    )
    write_result(out_path / TEST_FILE, test_record)

    records: dict[str, ResultRecord] = {"test": test_record}

    if inst_names:
        metrics = building_f1(np.concatenate(true_classes), np.concatenate(pred_classes))
        building_values = {
            "macro_f1_pct": metrics["macro_f1"] * 100.0,
            "f1_intact_pct": metrics["f1_intact"] * 100.0,
            "f1_damaged_pct": metrics["f1_damaged"] * 100.0,
            "f1_destroyed_pct": metrics["f1_destroyed"] * 100.0,
            "buildings": float(metrics["buildings"]),
            "missed": float(metrics["missed"]),
        }
        building_extra = dict(base_extra)
        building_extra["tiles_with_instances"] = len(inst_names)
        building_record = ResultRecord(
            metric_tag=BUILDING_TAG,
            values=building_values,
            model=model_name,
            split=SPLIT,
            commit=access.commit,
            config_hash=config_hash,
            seed=seed,
            extra=building_extra,
        )
        write_result(out_path / BUILDING_FILE, building_record)
        records["bina_f1"] = building_record

    # Written last so a crash earlier leaves no completion marker behind.
    _write_json_atomic(
        out_path / DONE_FILE,
        {
            "run_id": run_id,
            "split_sha256": split_sha256,
            "model": model_name,
            "tiles": len(names),
            "access": {"tag": access.tag, "commit": access.commit},
        },
    )
    return records


def load_tile_matrices(npz_path: Path | str) -> dict[str, np.ndarray]:
    """Load all per-tile ``int64`` confusion matrices from ``npz_path``, ignoring ``__total__``."""
    with np.load(npz_path) as data:
        return {key: np.array(data[key], dtype=np.int64) for key in data.files if key != TOTAL_KEY}


def slice_matrices(npz_path: Path | str, names: Iterable[str]) -> dict[str, np.ndarray]:
    """Return ``{name: int64 matrix}`` for ``names`` from ``npz_path``, ignoring ``__total__``."""
    requested = list(names)
    with np.load(npz_path) as data:
        available = {key for key in data.files if key != TOTAL_KEY}
        missing = [name for name in requested if name not in available]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"Arşivde eksik kareler var ({npz_path}; {len(missing)} adet, ilk örnekler: {preview}).")
        return {name: np.array(data[name], dtype=np.int64) for name in requested}


def _check_exact_tile_set(label: str, tiles: Mapping[str, np.ndarray], expected: set[str]) -> None:
    actual = set(tiles)
    if actual == expected:
        return
    missing = sorted(expected - actual)[:5]
    extra = sorted(actual - expected)[:5]
    raise ValueError(
        f"'{label}' arşivindeki kare kümesi test listesiyle birebir uyuşmuyor (eksik: {missing}, fazla: {extra})."
    )


def _class_iou_metric(class_id: int):
    def _metric(cm: np.ndarray) -> float:
        return float(iou_per_class(cm)[class_id])

    return _metric


def compare_fewshot(
    a_npz: Path | str,
    b_npz: Path | str,
    zs_npz: Path | str,
    test_names: Iterable[str],
    blocks: Mapping[str, tuple[str, str]] | str | Path,
    out_dir: Path | str,
    *,
    commit: str,
    config_hash: str = "fewshot-kiyas",
    names: tuple[str, str, str] = ("fewshot-a", "fewshot-b", "zs-ana"),
    n_boot: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> dict[str, ResultRecord]:
    """Compute and write all few-shot paired comparisons and the sliced ``zs-ana`` record.

    File naming convention: ``X_vs_Y.json`` reports the paired difference ``X - Y``.
    """
    test = sorted(test_names)
    if len(set(test)) != len(test):
        raise ValueError("Test kare listesinde yinelenen ad var.")
    if not test:
        raise ValueError("Kıyas için en az bir test karesi gereklidir.")
    test_set = set(test)
    name_a, name_b, name_zs = names

    tiles_a = load_tile_matrices(a_npz)
    tiles_b = load_tile_matrices(b_npz)
    _check_exact_tile_set(name_a, tiles_a, test_set)
    _check_exact_tile_set(name_b, tiles_b, test_set)
    tiles_zs = slice_matrices(zs_npz, test)

    block_map = _resolve_blocks(blocks, test)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: dict[str, ResultRecord] = {}

    comparisons: tuple[tuple[str, str, dict[str, np.ndarray], str, dict[str, np.ndarray], str | None], ...] = (
        ("b_vs_a", name_b, tiles_b, name_a, tiles_a, None),
        ("a_vs_zs-ana", name_a, tiles_a, name_zs, tiles_zs, _A_VS_ZS_NOTE),
        ("b_vs_zs-ana", name_b, tiles_b, name_zs, tiles_zs, None),
    )
    for stem, name_x, tiles_x, name_y, tiles_y, note in comparisons:
        res = compare_models(per_tile_a=tiles_y, per_tile_b=tiles_x, blocks=block_map, n=n_boot, seed=seed)
        values = {k: float(v) for k, v in res.items() if isinstance(v, (int, float))}
        extra: dict[str, object] = {
            "verdict": res["verdict"],
            "model_a": name_y,
            "model_b": name_x,
            "direction": f"{name_x} - {name_y}",
            "tiles": len(test),
            "n_boot": n_boot,
            "seed": seed,
        }
        if note is not None:
            extra["note"] = note
        rec = ResultRecord(
            metric_tag=DIFF_TAG,
            values=values,
            model=name_x,
            split=SPLIT,
            commit=commit,
            config_hash=config_hash,
            seed=seed,
            extra=extra,
        )
        write_result(out_path / f"{stem}.json", rec)
        records[stem] = rec

    total_a = sum((tiles_a[n] for n in test), np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))
    total_b = sum((tiles_b[n] for n in test), np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))
    ious_a = iou_per_class(total_a)
    ious_b = iou_per_class(total_b)

    class_values: dict[str, float] = {}
    class_extra: dict[str, object] = {
        "model_a": name_a,
        "model_b": name_b,
        "direction": f"{name_b} - {name_a}",
        "tiles": len(test),
        "n_boot": n_boot,
        "seed": seed,
    }
    for class_id, class_label in ((2, "damaged"), (3, "destroyed")):
        boot_cls = paired_block_bootstrap(
            tiles_a,
            tiles_b,
            block_map,
            metric=_class_iou_metric(class_id),
            n=n_boot,
            seed=seed,
        )
        diff_pct = boot_cls.estimate * 100.0
        low_pct = boot_cls.low * 100.0
        high_pct = boot_cls.high * 100.0
        class_values[f"{class_label}_a_pct"] = float(ious_a[class_id]) * 100.0
        class_values[f"{class_label}_b_pct"] = float(ious_b[class_id]) * 100.0
        class_values[f"{class_label}_diff_pct"] = diff_pct
        class_values[f"{class_label}_low_pct"] = low_pct
        class_values[f"{class_label}_high_pct"] = high_pct
        class_extra[f"verdict_{class_label}"] = equal_verdict(diff_pct, low_pct, high_pct)

    class_rec = ResultRecord(
        metric_tag=CLASS_DIFF_TAG,
        values=class_values,
        model=name_b,
        split=SPLIT,
        commit=commit,
        config_hash=config_hash,
        seed=seed,
        extra=class_extra,
    )
    write_result(out_path / "b_vs_a_siniflar.json", class_rec)
    records["b_vs_a_siniflar"] = class_rec

    total_zs = sum((tiles_zs[n] for n in test), np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))
    boot_zs = block_bootstrap(tiles_zs, block_map, n=n_boot, seed=seed)
    zs_rec = ResultRecord(
        metric_tag=ZS_TAG,
        values=_miou_values(total_zs, boot_zs.low, boot_zs.high),
        model=name_zs,
        split=SPLIT,
        commit=commit,
        config_hash=config_hash,
        seed=seed,
        extra={
            "tiles": len(test),
            "n_boot": n_boot,
            "seed": seed,
            "source": Path(zs_npz).name,
        },
    )
    write_result(out_path / "zs_ana.json", zs_rec)
    records["zs_ana"] = zs_rec

    return records
