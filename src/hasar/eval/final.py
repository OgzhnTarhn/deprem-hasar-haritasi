"""The single Turkey evaluation (``final`` step, protocol sections 3, 6, 7, 8, 10 and 11).

One inference, many analyses
----------------------------
This module is the only code path that reads Turkey imagery or labels (protocol section 11). It
runs the model over the Turkey test tiles exactly once, behind the protocol lock, and stores every
tile's ``clf`` and ``loc`` probabilities plus the flip-TTA probabilities as ``float16`` in a
per-tile cache. The headline record, the pre-declared secondary rows, the bootstrap intervals, the
building-level F1 and the calibration record are all computed from that cache, so no analysis ever
runs the model again.

Refuses to run twice
--------------------
The headline record ``turkiye.json`` is the proof that the single Turkey evaluation already
happened; it is written after every other record. When it exists the run stops with
:class:`hasar.guard.TurkeyLockError` before the model is touched, so a second evaluation cannot
silently overwrite the locked result.

The building-level F1 (protocol section 7) is computed on the CVPRW26 instance polygons of the
1,001 tiles fixed in ``results/arastirma/turkiye_bina_kareleri.json``. The instance folder must
match that list exactly; a mismatch is rejected before anything is written.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from hasar.data.dataset import BrightDataset
from hasar.data.events import load_standard_lists, make_zero_shot_split
from hasar.eval.bootstrap import (
    ANCHOR_MIOU,
    DEFAULT_N,
    DEFAULT_SEED,
    anchor_verdict,
    block_bootstrap,
    turkey_blocks,
)
from hasar.eval.building import building_classes, building_f1, load_instances, rasterize_instances
from hasar.eval.calibration import EceAccumulator, apply_temperature
from hasar.eval.metrics import ConfusionAccumulator, building_miou, iou_per_class, miou
from hasar.eval.predict import predict
from hasar.eval.rows import ROWS, row_confusions
from hasar.guard import TurkeyAccess, TurkeyLockError, require_turkey_access
from hasar.results import ResultRecord, write_result

AccessFn = Callable[[Path | str], TurkeyAccess]

#: The headline record; its presence means the single Turkey evaluation already ran.
HEADLINE_FILE = "turkiye.json"
#: The fixed building-tile list of protocol section 7, relative to the repository root.
DEFAULT_BUILDING_TILES = Path("results") / "arastirma" / "turkiye_bina_kareleri.json"
#: The split name written into every record.
SPLIT = "zero_shot/test"

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTANCE_SUFFIX = "_instance_damage.json"


def _metric_values(cm: np.ndarray) -> dict[str, float]:
    """Return the percent metrics derived from one event-level confusion matrix."""
    ious = iou_per_class(cm)
    return {
        "miou_pct": miou(cm) * 100,
        "iou_background_pct": float(ious[0]) * 100,
        "iou_intact_pct": float(ious[1]) * 100,
        "iou_damaged_pct": float(ious[2]) * 100,
        "iou_destroyed_pct": float(ious[3]) * 100,
        "building_miou_pct": building_miou(cm) * 100,
    }


def _save_cache(
    path: Path,
    clf: np.ndarray,
    loc: np.ndarray,
    clf_tta: np.ndarray,
    loc_tta: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Write one tile's probabilities to ``path`` atomically as ``float16``.

    The arrays go to a ``.tmp`` file first and are moved into place with :func:`os.replace`, so a
    crash never leaves a half-written cache file behind.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(
            handle,
            clf=clf.astype(np.float16),
            loc=loc.astype(np.float16),
            clf_tta=clf_tta.astype(np.float16),
            loc_tta=loc_tta.astype(np.float16),
            valid=np.asarray(valid, dtype=np.bool_),
        )
    os.replace(tmp, path)


def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the cached ``(clf, loc, clf_tta, loc_tta)`` probabilities, or ``None`` if absent."""
    if not path.is_file():
        return None
    with np.load(path) as data:
        return (
            np.asarray(data["clf"], dtype=np.float32),
            np.asarray(data["loc"], dtype=np.float32),
            np.asarray(data["clf_tta"], dtype=np.float32),
            np.asarray(data["loc_tta"], dtype=np.float32),
        )


def _resolve_building_tiles(building_tiles: Path | str, repo_root: Path) -> Path:
    """Resolve a building-tile list path; a relative path is taken from ``repo_root``."""
    path = Path(building_tiles)
    return path if path.is_absolute() else repo_root / path


def _check_building_tiles(instances_dir: Path, listing: Path) -> None:
    """Raise ``ValueError`` unless the instance folder matches the fixed tile list exactly."""
    expected = set(json.loads(listing.read_text(encoding="utf-8"))["tiles"])
    found = {path.name[: -len(_INSTANCE_SUFFIX)] for path in instances_dir.glob(f"*{_INSTANCE_SUFFIX}")}
    if found == expected:
        return
    missing = sorted(expected - found)[:5]
    extra = sorted(found - expected)[:5]
    raise ValueError(
        f"Bina örnek klasörü sabit listeyle uyuşmuyor ({instances_dir}): "
        f"eksik {missing}, fazla {extra}. Türkiye değerlendirmesi başlatılmadı."
    )


def _blocks(
    blocks: Mapping[str, tuple[str, str]] | str | Path | None,
    names: Sequence[str],
) -> dict[str, tuple[str, str]]:
    """Return the bootstrap block map; ``None`` uses one block per tile."""
    if blocks is None:
        return {name: ("all", name) for name in names}
    if isinstance(blocks, (str, Path)):
        return turkey_blocks(blocks)
    return dict(blocks)


def run_final(
    model: torch.nn.Module,
    data_root: Path | str,
    out_dir: Path | str,
    cache_dir: Path | str,
    model_name: str,
    config_hash: str,
    names: Sequence[str] | None = None,
    repo_root: Path | str | None = None,
    access_fn: AccessFn | None = None,
    n_boot: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    blocks: Mapping[str, tuple[str, str]] | str | Path | None = None,
    instances_dir: Path | str | None = None,
    building_tiles: Path | str | None = DEFAULT_BUILDING_TILES,
    temperature: float | None = None,
    sar_mode: str = "official",
    device: str = "cpu",
) -> dict[str, ResultRecord]:
    """Run the single Turkey evaluation and write every record of protocol sections 3-11.

    ``names`` defaults to the Turkey test tiles of the zero-shot split built from the
    ``standard_ML`` lists. Turkey access is requested through ``access_fn`` (default
    :func:`hasar.guard.require_turkey_access`) before any tile is read; the access tag and commit
    are written into every record. The per-tile probabilities are cached under ``cache_dir`` and
    reused on a later call, so the model runs at most once per tile.

    ``blocks`` is the bootstrap block map (``None``: one block per tile); a path is read with
    :func:`hasar.eval.bootstrap.turkey_blocks`. When ``instances_dir`` is given, the building-level
    F1 is computed on the instance files and, if ``building_tiles`` names an existing list, the
    folder must match it exactly. ``temperature`` adds the calibration record.

    Returns the records keyed by row name (``"headline"`` plus :data:`hasar.eval.rows.ROWS`).
    """
    repo_root = Path(repo_root) if repo_root is not None else _DEFAULT_REPO_ROOT
    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir)

    access_fn = access_fn if access_fn is not None else require_turkey_access
    access = access_fn(repo_root)

    if (out_dir / HEADLINE_FILE).exists():
        raise TurkeyLockError(
            f"Türkiye değerlendirmesi zaten yapıldı ({out_dir / HEADLINE_FILE}): "
            "ikinci kez çalıştırılamaz (protokol §1, §11)."
        )

    if names is None:
        names = make_zero_shot_split(*load_standard_lists()).test
    names = list(names)

    instances_path = Path(instances_dir) if instances_dir is not None else None
    if instances_path is not None and building_tiles is not None:
        listing = _resolve_building_tiles(building_tiles, repo_root)
        if listing.is_file():
            _check_building_tiles(instances_path, listing)

    dataset = BrightDataset(data_root, names, mode="eval", sar_mode=sar_mode, access=access)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()

    accumulators = {row: ConfusionAccumulator() for row in ROWS}
    ece = EceAccumulator() if temperature is not None else None
    true_classes: list[np.ndarray] = []
    pred_classes: list[np.ndarray] = []
    tiles_with_instances = 0

    for item in dataset:
        name = str(item["name"])
        label = np.asarray(item["clf"])
        valid = np.asarray(item["valid"], dtype=np.bool_)

        cached = _load_cache(cache_dir / f"{name}.npz")
        if cached is None:
            pre = torch.from_numpy(np.asarray(item["pre"])).unsqueeze(0).to(device)
            post = torch.from_numpy(np.asarray(item["post"])).unsqueeze(0).to(device)
            clf_prob, loc_prob = predict(model, pre, post, tta=False)
            clf_tta_prob, loc_tta_prob = predict(model, pre, post, tta=True)
            clf = clf_prob[0].float().cpu().numpy()
            loc = loc_prob[0].float().cpu().numpy()
            clf_tta = clf_tta_prob[0].float().cpu().numpy()
            loc_tta = loc_tta_prob[0].float().cpu().numpy()
            _save_cache(cache_dir / f"{name}.npz", clf, loc, clf_tta, loc_tta, valid)
        else:
            clf, loc, clf_tta, loc_tta = cached

        confusions = row_confusions(label, clf, loc, clf_tta, valid)
        for row in ROWS:
            accumulator = accumulators[row]
            accumulator.total += confusions[row]
            accumulator.per_tile[name] = confusions[row]

        if ece is not None:
            ece.add(apply_temperature(clf, temperature).reshape(clf.shape[0], -1), label.reshape(-1))

        if instances_path is not None:
            instance_file = instances_path / f"{name}{_INSTANCE_SUFFIX}"
            if instance_file.is_file():
                instance_map = rasterize_instances(load_instances(instance_file), label.shape)
                pred = np.asarray(clf).argmax(axis=0)
                true_cls, pred_cls = building_classes(label, instance_map, pred)
                true_classes.append(true_cls)
                pred_classes.append(pred_cls)
                tiles_with_instances += 1

    block_map = _blocks(blocks, names)
    base_extra: dict[str, object] = {
        "access": {"tag": access.tag, "commit": access.commit},
        "tiles": len(names),
        "n_boot": n_boot,
        "seed": seed,
    }

    records: dict[str, ResultRecord] = {}
    for row in ROWS:
        accumulator = accumulators[row]
        accumulator.save_npz(out_dir / f"turkiye_kare_matrisleri_{row}.npz")

        values = _metric_values(accumulator.total)
        bootstrap = block_bootstrap(accumulator.per_tile, block_map, n=n_boot, seed=seed)
        values["ci_low_pct"] = bootstrap.low * 100
        values["ci_high_pct"] = bootstrap.high * 100

        extra = dict(base_extra)
        if row == "headline":
            metric_tag = "[ZS-EVENT mIoU]"
            extra["anchor"] = ANCHOR_MIOU
            extra["verdict"] = anchor_verdict(
                values["miou_pct"],
                values["ci_low_pct"],
                values["ci_high_pct"],
                anchor=ANCHOR_MIOU,
            )
            path = out_dir / HEADLINE_FILE
        else:
            metric_tag = f"[ZS-EVENT mIoU, {row}]"
            extra["row"] = row
            path = out_dir / f"turkiye_{row}.json"

        record = ResultRecord(
            metric_tag=metric_tag,
            values=values,
            model=model_name,
            split=SPLIT,
            commit=access.commit,
            config_hash=config_hash,
            seed=seed,
            extra=extra,
        )
        # The headline is the lock: it is written last, so a crash in any later record leaves no
        # ``turkiye.json`` behind and the run can resume from the cache.
        if row != "headline":
            write_result(path, record)
        records[row] = record

    if instances_path is not None and tiles_with_instances:
        metrics = building_f1(np.concatenate(true_classes), np.concatenate(pred_classes))
        building_values = {
            "macro_f1_pct": metrics["macro_f1"] * 100,
            "f1_intact_pct": metrics["f1_intact"] * 100,
            "f1_damaged_pct": metrics["f1_damaged"] * 100,
            "f1_destroyed_pct": metrics["f1_destroyed"] * 100,
            "buildings": float(metrics["buildings"]),
            "missed": float(metrics["missed"]),
        }
        building_extra = dict(base_extra)
        building_extra["tiles_with_instances"] = tiles_with_instances
        write_result(
            out_dir / "turkiye_bina_f1.json",
            ResultRecord(
                metric_tag="[ZS-EVENT bina-makro-F1]",
                values=building_values,
                model=model_name,
                split=SPLIT,
                commit=access.commit,
                config_hash=config_hash,
                seed=seed,
                extra=building_extra,
            ),
        )

    if ece is not None:
        calibration_values = {
            "temperature": float(temperature),
            "ece_pct": ece.value() * 100,
            "miou_pct": records["headline"].values["miou_pct"],
        }
        calibration_extra = dict(base_extra)
        calibration_extra["bins"] = ece.bins
        write_result(
            out_dir / "turkiye_kalibrasyon.json",
            ResultRecord(
                metric_tag="[ZS-EVENT mIoU, kalibrasyon]",
                values=calibration_values,
                model=model_name,
                split=SPLIT,
                commit=access.commit,
                config_hash=config_hash,
                seed=seed,
                extra=calibration_extra,
            ),
        )

    write_result(out_dir / HEADLINE_FILE, records["headline"])
    return records
