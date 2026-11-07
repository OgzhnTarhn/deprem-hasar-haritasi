"""Download and extract the BRIGHT data set from Hugging Face.

The data set lives in the ``Kullervo/BRIGHT`` repository as three zip archives, one per
modality::

    pre-event.zip    <root>/pre-event/<name>_pre_disaster.tif
    post-event.zip   <root>/post-event/<name>_post_disaster.tif
    target.zip       <root>/target/<name>_building_damage.tif

Every archive carries a single top-level folder (``pre-event/``, ``post-event/`` or
``target/``), so extracting all three into the same root yields the layout read by
:mod:`hasar.data.io`.

Resumability
------------
Colab sessions are disconnected often, so extraction is guarded by a marker file
``.<archive-stem>.done`` written into the destination root only after a successful
extract. :func:`prepare_bright` checks the marker before downloading, which makes a
rerun after a disconnect free.

Safety
------
Member names are validated before anything is written: absolute paths, Windows drive
letters and ``..`` parts are rejected with :class:`ValueError`, so a malicious archive
cannot escape the destination directory.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

from hasar.data.io import tile_paths

REPO_ID = "Kullervo/BRIGHT"
REPO_TYPE = "dataset"
ARCHIVES = ("pre-event.zip", "post-event.zip", "target.zip")
INSTANCE_ARCHIVE = "cvprw26_trainval_instance_labels.zip"
INSTANCE_SUFFIX = "_instance_damage.json"

Downloader = Callable[..., str]

_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")


def fetch(
    filename: str,
    cache_dir: Path | str,
    *,
    downloader: Downloader | None = None,
) -> Path:
    """Download ``filename`` from the BRIGHT repository into ``cache_dir``.

    ``huggingface_hub`` is imported lazily so that the rest of the package works without
    it; tests inject their own ``downloader`` with the same keyword arguments.
    """
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = downloader(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        local_dir=str(cache_dir),
    )
    return Path(downloaded)


def _is_safe_member(name: str) -> bool:
    """Return ``True`` when the archive member stays inside the destination directory."""
    if not name or name.startswith(("/", "\\")):
        return False
    if _DRIVE_LETTER.match(name):
        return False
    parts = name.replace("\\", "/").split("/")
    return ".." not in parts


def _write_marker(marker: Path) -> None:
    """Write ``marker`` atomically: a temp file first, then :func:`os.replace`."""
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text("ok\n", encoding="utf-8")
    os.replace(tmp, marker)


def extract_once(zip_path: Path | str, dest: Path | str) -> bool:
    """Extract ``zip_path`` into ``dest`` unless the marker says it is already done.

    Returns ``True`` when the archive was extracted and ``False`` when the marker was
    already present. Every member name is validated before the first byte is written;
    the marker is written only after a successful extract.
    """
    zip_path = Path(zip_path)
    dest = Path(dest)
    marker = dest / f".{zip_path.stem}.done"
    if marker.is_file():
        return False

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for name in names:
            if not _is_safe_member(name):
                raise ValueError(f"Güvenli olmayan arşiv üyesi: {name!r} ({zip_path})")
        dest.mkdir(parents=True, exist_ok=True)
        archive.extractall(dest)

    _write_marker(marker)
    return True


def prepare_bright(
    root: Path | str,
    cache_dir: Path | str,
    *,
    downloader: Downloader | None = None,
    archives: Iterable[str] = ARCHIVES,
) -> Path:
    """Download and extract every BRIGHT archive into ``root`` and return ``root``.

    Archives whose marker already exists are skipped before any download, so a rerun
    after a Colab disconnect costs nothing. Raises :class:`FileNotFoundError` when an
    archive does not contain its expected top-level folder.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for archive in archives:
        marker = root / f".{Path(archive).stem}.done"
        if marker.is_file():
            continue
        zip_path = fetch(archive, cache_dir, downloader=downloader)
        extract_once(zip_path, root)
        expected = root / Path(archive).stem
        if not expected.is_dir():
            marker.unlink(missing_ok=True)
            raise FileNotFoundError(f"Arşivde beklenen klasör yok: {expected} ({archive})")

    return root


def prepare_instances(
    dest: Path | str,
    cache_dir: Path | str,
    tiles_file: Path | str,
    *,
    downloader: Downloader | None = None,
) -> Path:
    """Download the instance-label archive and extract the listed tiles flat into ``dest``.

    Protocol section 7: building-level F1 is computed on the 1,001 Turkey tiles fixed by
    file names before the lock; only file names are used to select members, contents are
    read later by the final step only.

    ``tiles_file`` is a JSON file with a ``tiles`` key holding the tile names. Only members
    whose base name is ``<tile>_instance_damage.json`` for a listed tile are written, and
    they are written directly under ``dest`` (the archive's nested folders are dropped).
    Every member name is validated before anything is written. A ``.instances.done`` marker
    makes a rerun free; it is written only after a successful extract. Raises
    :class:`ValueError` when a listed tile is missing from the archive.
    """
    dest = Path(dest)
    marker = dest / ".instances.done"
    if marker.is_file():
        return dest

    tiles = json.loads(Path(tiles_file).read_text(encoding="utf-8"))["tiles"]
    wanted = set(tiles)

    zip_path = fetch(INSTANCE_ARCHIVE, cache_dir, downloader=downloader)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for name in names:
            if not _is_safe_member(name):
                raise ValueError(f"Güvenli olmayan arşiv üyesi: {name!r} ({zip_path})")

        selected: dict[str, str] = {}
        for name in names:
            base = name.replace("\\", "/").rsplit("/", 1)[-1]
            if not base.endswith(INSTANCE_SUFFIX):
                continue
            tile = base[: -len(INSTANCE_SUFFIX)]
            if tile in wanted:
                selected[tile] = name

        missing = sorted(wanted - set(selected))
        if missing:
            shown = ", ".join(missing[:5])
            raise ValueError(f"Örnek etiketi eksik kareler: {shown} ({len(missing)} kare)")

        dest.mkdir(parents=True, exist_ok=True)
        for tile, name in selected.items():
            (dest / f"{tile}{INSTANCE_SUFFIX}").write_bytes(archive.read(name))

    _write_marker(marker)
    return dest


def missing_tiles(root: Path | str, names: Iterable[str]) -> list[str]:
    """Return the sorted tile names whose three files are not all present under ``root``.

    Only file existence is checked; no file is opened, so this is safe for Turkey names
    and serves as the integrity check of the download step.
    """
    missing: list[str] = []
    for name in names:
        paths = tile_paths(root, name)
        if not (paths.pre.is_file() and paths.post.is_file() and paths.target.is_file()):
            missing.append(name)
    return sorted(missing)
