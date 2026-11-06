"""Tile reader for the BRIGHT data set.

Layout
------
Every tile is stored as three files under a common root, named after the tile::

    <root>/pre-event/<name>_pre_disaster.tif      H x W x 3  (optical, uint8)
    <root>/post-event/<name>_post_disaster.tif    H x W      (SAR, uint8)
    <root>/target/<name>_building_damage.tif      H x W      (label, uint8)

Turkey guard
------------
Project rule 1 allows a single evaluation on Turkey, in the ``final`` step only. Every reader in
this module therefore asks :func:`hasar.guard.is_turkey_tile` first and raises
:class:`hasar.guard.TurkeyLockError` unless a :class:`hasar.guard.TurkeyAccess` token is passed.
The check runs before any path is opened, so a locked Turkey tile never touches the disk.

Nodata
------
A pixel is nodata when both modalities are empty: all optical channels are zero *and* the SAR
value is zero. BRIGHT tiles carry ``GDAL_NODATA=0``; some SAR frames are entirely empty and must
be ignored by the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from hasar.guard import TurkeyAccess, TurkeyLockError, is_turkey_tile

PRE_DIR = "pre-event"
POST_DIR = "post-event"
TARGET_DIR = "target"

PRE_SUFFIX = "_pre_disaster.tif"
POST_SUFFIX = "_post_disaster.tif"
TARGET_SUFFIX = "_building_damage.tif"


@dataclass(frozen=True)
class TilePaths:
    """Paths of the three files that make up one tile."""

    pre: Path
    post: Path
    target: Path


@dataclass(frozen=True)
class GeoMeta:
    """Georeferencing header of a tile: size, origin, pixel size and EPSG code."""

    width: int
    height: int
    origin_x: float
    origin_y: float
    pixel_x: float
    pixel_y: float
    epsg: int | None

    def bounds(self) -> tuple[float, float, float, float]:
        """Return ``(left, bottom, right, top)`` in the tile's CRS."""
        left = self.origin_x
        top = self.origin_y
        right = left + self.width * self.pixel_x
        bottom = top - self.height * self.pixel_y
        return (left, bottom, right, top)


def tile_paths(root: Path | str, name: str) -> TilePaths:
    """Return the three file paths of tile ``name`` under ``root``."""
    root = Path(root)
    return TilePaths(
        pre=root / PRE_DIR / f"{name}{PRE_SUFFIX}",
        post=root / POST_DIR / f"{name}{POST_SUFFIX}",
        target=root / TARGET_DIR / f"{name}{TARGET_SUFFIX}",
    )


def _check_turkey(name: str, access: TurkeyAccess | None) -> None:
    """Raise :class:`TurkeyLockError` for a Turkey tile without an access token."""
    if is_turkey_tile(name) and access is None:
        raise TurkeyLockError(f"'{name}' Türkiye test olayına ait: protokol kilidi olmadan Türkiye verisi okunamaz.")


def _read_array(path: Path) -> np.ndarray:
    """Read a single-page TIFF as a numpy array."""
    if not path.is_file():
        raise FileNotFoundError(f"Kare dosyası bulunamadı: {path}")
    return np.asarray(tifffile.imread(path))


def _read_pre(path: Path) -> np.ndarray:
    """Read the optical pre-event image and require three channels."""
    pre = _read_array(path)
    if pre.ndim != 3 or pre.shape[2] != 3:
        raise ValueError(f"Optik görüntü H x W x 3 olmalı, bulunan: {pre.shape} ({path})")
    return pre


def _read_post(path: Path) -> np.ndarray:
    """Read the SAR post-event image and squeeze a single channel if present."""
    post = _read_array(path)
    if post.ndim == 3 and post.shape[2] == 1:
        post = post[:, :, 0]
    if post.ndim != 2:
        raise ValueError(f"SAR görüntüsü H x W olmalı, bulunan: {post.shape} ({path})")
    return post


def _read_target(path: Path) -> np.ndarray:
    """Read the damage label and require a 2-D array."""
    target = _read_array(path)
    if target.ndim == 3 and target.shape[2] == 1:
        target = target[:, :, 0]
    if target.ndim != 2:
        raise ValueError(f"Etiket H x W olmalı, bulunan: {target.shape} ({path})")
    return target


def read_tile(
    root: Path | str,
    name: str,
    access: TurkeyAccess | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one tile and return ``(pre, post, label)`` as uint8 arrays.

    ``pre`` is ``H x W x 3``, ``post`` and ``label`` are ``H x W``. The Turkey gate runs before
    any file is opened.
    """
    _check_turkey(name, access)
    paths = tile_paths(root, name)
    pre = _read_pre(paths.pre)
    post = _read_post(paths.post)
    label = _read_target(paths.target)

    if pre.shape[:2] != post.shape or pre.shape[:2] != label.shape:
        raise ValueError(
            f"Kare boyutları uyuşmuyor: pre {pre.shape[:2]}, post {post.shape}, etiket {label.shape} ({name})"
        )
    return pre, post, label


def read_label(
    root: Path | str,
    name: str,
    access: TurkeyAccess | None = None,
) -> np.ndarray:
    """Read only the damage label of one tile; the Turkey gate runs before any disk access."""
    _check_turkey(name, access)
    return _read_target(tile_paths(root, name).target)


def nodata_mask(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Return ``True`` where both modalities are empty (all optical zero and SAR zero)."""
    optical_empty = np.all(pre == 0, axis=-1)
    sar_empty = post == 0
    return np.asarray(optical_empty & sar_empty, dtype=np.bool_)


def read_geo_meta(path: Path | str) -> GeoMeta:
    """Read the GeoTIFF header of ``path`` without decoding pixel data.

    Raises :class:`ValueError` when the file carries no georeferencing tags.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        meta = tif.geotiff_metadata
        height, width = tif.pages[0].shape[:2]

    if not meta or "ModelPixelScale" not in meta or "ModelTiepoint" not in meta:
        raise ValueError(f"Jeo-etiketler bulunamadı: {path}")

    scale = meta["ModelPixelScale"]
    tiepoint = meta["ModelTiepoint"]
    pixel_x = float(scale[0])
    pixel_y = float(scale[1])
    origin_x = float(tiepoint[3]) - float(tiepoint[0]) * pixel_x
    origin_y = float(tiepoint[4]) + float(tiepoint[1]) * pixel_y

    epsg: int | None = None
    if "ProjectedCSTypeGeoKey" in meta:
        epsg = int(meta["ProjectedCSTypeGeoKey"])
    elif "GeographicTypeGeoKey" in meta:
        epsg = int(meta["GeographicTypeGeoKey"])

    return GeoMeta(
        width=int(width),
        height=int(height),
        origin_x=origin_x,
        origin_y=origin_y,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        epsg=epsg,
    )
