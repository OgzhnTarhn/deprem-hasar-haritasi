"""Phase 7 cache-based analysis runner (Y2 analyst budget, risk-coverage, SAR error gallery).

Every analysis here reads the per-tile cache written by the single Turkey evaluation
(:func:`hasar.eval.final.run_final`) and never reruns the model. The runner refuses to start unless
the headline record ``<final_dir>/turkiye.json`` exists, so it can only run after the one and only
``final`` step.

The Y1 reference comparison (HOT/OSM destroyed-building layer and the ARIA damage proxy map) needs
external layers that are not part of the cache; it is run with the scripts under ``scripts/analiz/``
and is deliberately not part of this module.

The building records come from the cached flip-TTA probabilities and the CVPRW26 instance polygons;
the per-tile confusion matrices used by the gallery come from the headline argmax of the cached
``clf`` probabilities against the label. The gallery panels read the pre-disaster optical and
post-disaster SAR tiles through :func:`hasar.data.io.read_tile`, which enforces the Turkey guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hasar.analysis.gallery import panel, rank_tiles
from hasar.analysis.uncertainty import BuildingRecord, analyst_budget, building_records, risk_coverage
from hasar.data.io import read_label, read_tile
from hasar.eval.building import load_instances, rasterize_instances
from hasar.eval.metrics import confusion_matrix
from hasar.guard import TurkeyAccess, TurkeyLockError
from hasar.results import ResultRecord, read_result, write_result

#: The headline record; its presence means the single Turkey evaluation already ran.
HEADLINE_FILE = "turkiye.json"
#: Suffix of the CVPRW26 instance files.
INSTANCE_SUFFIX = "_instance_damage.json"

#: Output names under ``out_dir``.
BUDGET_FILE = "y2_butce.json"
RISK_COVERAGE_FILE = "y2_risk_kapsam.json"
GALLERY_DIR = "galeri"
GALLERY_INDEX = "index.json"

#: Metric tag of the analyst-budget record (rule 4: the type is always named).
BUDGET_TAG = "[ZS-EVENT bina-makro-F1, analist-butce]"

#: Default number of worst tiles in the gallery.
DEFAULT_GALLERY_K = 10


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    """Return every array of one ``.npz`` cache file as a plain numpy array."""
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _budget_values(records: list[BuildingRecord]) -> dict[str, float]:
    """Flatten the analyst-budget table into ``<ranking>_<pct>_macro_f1_pct`` keys plus ``aurc``."""
    values: dict[str, float] = {}
    for ranking, budgets in analyst_budget(records).items():
        for budget, metrics in budgets.items():
            pct = round(float(budget) * 100)
            values[f"{ranking}_{pct}_macro_f1_pct"] = metrics["macro_f1"] * 100
    _coverage, _risk, aurc = risk_coverage(records)
    values["aurc"] = aurc
    values["buildings"] = float(len(records))
    return values


def _write_risk_coverage(path: Path, records: list[BuildingRecord]) -> Path:
    """Write the risk-coverage curve of the uncertainty ranking as plain JSON."""
    coverage, risk, aurc = risk_coverage(records)
    payload = {
        "coverage": coverage.tolist(),
        "risk": risk.tolist(),
        "aurc": aurc,
        "buildings": len(records),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_gallery(
    gallery_dir: Path,
    per_tile: dict[str, np.ndarray],
    cache_dir: Path,
    data_root: Path,
    access: TurkeyAccess | None,
    k: int,
) -> Path:
    """Write the worst ``k`` tile panels and the ``index.json`` naming them.

    Only the ``k`` chosen tiles are read again (imagery through the Turkey guard, prediction from
    the cache): keeping every tile's imagery in memory would need ~13 GB for 1,001 tiles.
    """
    gallery_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, score in rank_tiles(per_tile, key="building_miou", worst=True, k=k):
        pre, post, label = read_tile(data_root, name, access)
        pred = _load_cache(cache_dir / f"{name}.npz")["clf"].astype(np.float32).argmax(axis=0)
        image_path = gallery_dir / f"{name}.png"
        panel(pre, post, label, pred).save(image_path)
        entries.append({"tile": name, "building_miou": score, "image": image_path.name})

    index_path = gallery_dir / GALLERY_INDEX
    index_path.write_text(
        json.dumps({"tiles": entries}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return index_path


def run_cache_analyses(
    cache_dir: Path | str,
    data_root: Path | str,
    instances_dir: Path | str,
    out_dir: Path | str,
    final_dir: Path | str | None = None,
    access: TurkeyAccess | None = None,
    gallery_k: int = DEFAULT_GALLERY_K,
) -> dict[str, Path]:
    """Run the Phase 7 cache analyses and return the written paths.

    ``cache_dir`` holds one ``<name>.npz`` per tile (``clf``, ``loc``, ``clf_tta``, ``loc_tta``,
    ``valid``); ``instances_dir`` holds the CVPRW26 ``<name>_instance_damage.json`` files. A cache
    file without an instance file is skipped. ``final_dir`` defaults to ``out_dir`` and must contain
    the headline ``turkiye.json``; otherwise :class:`hasar.guard.TurkeyLockError` is raised before
    anything is written.

    Writes the analyst-budget record (``y2_butce.json``), the risk-coverage curve
    (``y2_risk_kapsam.json``) and the error gallery (``galeri/``). Returns the three paths keyed
    ``butce``, ``risk_kapsam`` and ``galeri``.
    """
    cache_dir = Path(cache_dir)
    data_root = Path(data_root)
    instances_dir = Path(instances_dir)
    out_dir = Path(out_dir)
    final_dir = Path(final_dir) if final_dir is not None else out_dir

    headline_path = final_dir / HEADLINE_FILE
    if not headline_path.is_file():
        raise TurkeyLockError(
            f"Başlık kaydı yok ({headline_path}): analizler tek Türkiye değerlendirmesinden sonra çalışır."
        )
    headline = read_result(headline_path)

    per_tile: dict[str, np.ndarray] = {}
    records: list[BuildingRecord] = []

    for cache_path in sorted(cache_dir.glob("*.npz")):
        name = cache_path.stem
        instance_file = instances_dir / f"{name}{INSTANCE_SUFFIX}"
        if not instance_file.is_file():
            continue

        label = read_label(data_root, name, access)
        cache = _load_cache(cache_path)
        clf = cache["clf"].astype(np.float32)
        clf_tta = cache["clf_tta"].astype(np.float32)

        instance_map = rasterize_instances(load_instances(instance_file), label.shape)
        pred = clf.argmax(axis=0)

        per_tile[name] = confusion_matrix(label, pred)
        records.extend(building_records(name, label, instance_map, clf_tta))

    budget_path = write_result(
        out_dir / BUDGET_FILE,
        ResultRecord(
            metric_tag=BUDGET_TAG,
            values=_budget_values(records),
            model=headline.model,
            split=headline.split,
            commit=headline.commit,
            config_hash=headline.config_hash,
            seed=headline.seed,
            extra={"tiles": len(per_tile), "buildings": len(records)},
        ),
    )
    risk_path = _write_risk_coverage(out_dir / RISK_COVERAGE_FILE, records)
    gallery_path = _write_gallery(out_dir / GALLERY_DIR, per_tile, cache_dir, data_root, access, gallery_k)

    return {"butce": budget_path, "risk_kapsam": risk_path, "galeri": gallery_path}
