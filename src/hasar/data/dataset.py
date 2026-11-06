"""BRIGHT tile dataset for training and evaluation.

One item is one tile (eval) or one augmented crop of a tile (train). The pipeline is:

1. :func:`hasar.data.io.read_tile` reads ``(pre, post, label)``; the Turkey gate runs first.
2. ``valid = ~io.nodata_mask(pre, post)`` is computed on the full tile.
3. Train only: when ``nodata_ignore`` is set, the label is set to ``ignore_index`` where the tile is
   nodata, then :func:`hasar.data.transforms.augment` applies the same crop, flips and rotation to
   ``pre``, ``post``, ``label`` and ``valid``. The generator is ``sampler.rng_for(seed, sample_no)``,
   so a key fully determines the sample.
4. ``pre`` is normalized with the ImageNet statistics and ``post`` with the configured SAR mode.
5. ``clf`` is the integer label and ``loc`` is the localization target: ``1`` for damage classes
   ``1..3``, ``0`` for background and the raw value for anything else (ignore index or
   out-of-range).

Project rule 1 (Turkey gate): every name is checked with :func:`hasar.guard.is_turkey_tile` at
construction. A Turkey tile needs a :class:`hasar.guard.TurkeyAccess` token and is never allowed in
``mode="train"``. The eval path leaves the label untouched (official evaluation path: no nodata
handling); nodata ignoring is a train-only rule.

The module imports only numpy. A torch ``DataLoader`` can still consume it through ``__getitem__``
and ``__len__``; collation is the trainer's job.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hasar.data import io, normalize, sampler, transforms
from hasar.guard import TurkeyAccess, TurkeyLockError, is_turkey_tile

MODES: tuple[str, ...] = ("train", "eval")

#: Valid damage labels are ``0..NUM_DAMAGE_CLASSES-1``; classes 1..3 map to building (loc 1).
NUM_DAMAGE_CLASSES = 4


class BrightDataset:
    """Map-style dataset over BRIGHT tiles.

    Parameters
    ----------
    root:
        Data set root holding the ``pre-event``, ``post-event`` and ``target`` folders.
    names:
        Tile names; every name is validated against the Turkey gate at construction.
    mode:
        ``"train"`` (tuple key ``(index, sample_no)``, crop and augmentation) or ``"eval"``
        (integer key, full tile, no augmentation).
    crop:
        Crop size for training; required and positive in ``mode="train"``.
    flips, rot90:
        Enable the paired horizontal/vertical flips and the 90 degree rotation in training.
    nodata_ignore:
        Train only: set the label to ``ignore_index`` where the tile is nodata.
    sar_mode:
        SAR normalization mode, one of :data:`hasar.data.normalize.SAR_MODES`.
    sar_stats:
        Statistics for ``sar_mode="train_stats"``.
    seed:
        Run seed; part of the augmentation generator key.
    access:
        Turkey access token; required to read a Turkey tile and never valid in training.
    ignore_index:
        Label value used for ignored pixels.
    """

    def __init__(
        self,
        root: Path | str,
        names: Sequence[str],
        mode: str = "train",
        crop: int | None = 640,
        flips: bool = True,
        rot90: bool = True,
        nodata_ignore: bool = True,
        sar_mode: str = "official",
        sar_stats: normalize.SarStats | None = None,
        seed: int = 42,
        access: TurkeyAccess | None = None,
        ignore_index: int = 255,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"bilinmeyen mod: {mode!r} (izinli: {MODES})")
        if mode == "train" and (crop is None or crop <= 0):
            raise ValueError(f"mode='train' için pozitif kırpma boyutu gerekli, alınan: {crop!r}")
        if sar_mode not in normalize.SAR_MODES:
            raise ValueError(f"bilinmeyen SAR normalizasyon modu: {sar_mode!r} (izinli: {normalize.SAR_MODES})")
        if sar_mode == "train_stats" and sar_stats is None:
            raise ValueError("sar_mode='train_stats' için SarStats gerekli")

        self.root = root
        self.names = list(names)
        self.mode = mode
        self.crop = crop
        self.flips = flips
        self.rot90 = rot90
        self.nodata_ignore = nodata_ignore
        self.sar_mode = sar_mode
        self.sar_stats = sar_stats
        self.seed = seed
        self.access = access
        self.ignore_index = ignore_index

        for name in self.names:
            if not is_turkey_tile(name):
                continue
            if mode == "train":
                raise TurkeyLockError(f"'{name}' Türkiye karosu: Türkiye eğitimde kullanılamaz (proje kuralı 1).")
            if access is None:
                raise TurkeyLockError(
                    f"'{name}' Türkiye test olayına ait: protokol kilidi olmadan Türkiye verisi okunamaz."
                )

    def __len__(self) -> int:
        """Return the number of tiles."""
        return len(self.names)

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, object]:
        """Return one sample; train keys are ``(index, sample_no)``, eval keys are integers."""
        if self.mode == "train":
            if not isinstance(key, tuple):
                raise TypeError(f"mode='train' anahtarı (index, sample_no) olmalı, alınan: {key!r}")
            index, sample_no = key
            return self._train_item(int(index), int(sample_no))

        if isinstance(key, tuple):
            raise TypeError(f"mode='eval' anahtarı tamsayı olmalı, alınan: {key!r}")
        return self._eval_item(int(key))

    def _read(self, index: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read tile ``index`` and return ``(name, pre, post, label, valid)``."""
        name = self.names[index]
        pre, post, label = io.read_tile(self.root, name, self.access)
        valid = ~io.nodata_mask(pre, post)
        return name, pre, post, label, valid

    def _targets(self, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(clf, loc)`` from an integer label."""
        clf = label.astype(np.int64)
        loc = np.where((clf > 0) & (clf < NUM_DAMAGE_CLASSES), 1, clf).astype(np.int64)
        return clf, loc

    def _eval_item(self, index: int) -> dict[str, object]:
        """Return the full tile; the label is left untouched (official evaluation path)."""
        name, pre, post, label, valid = self._read(index)
        clf, loc = self._targets(label)
        return {
            "name": name,
            "pre": normalize.normalize_optical(pre),
            "post": normalize.sar_to_input(post, self.sar_mode, self.sar_stats),
            "clf": clf,
            "loc": loc,
            "valid": valid,
        }

    def _train_item(self, index: int, sample_no: int) -> dict[str, object]:
        """Return one augmented crop; nodata pixels are ignored when ``nodata_ignore`` is set."""
        name, pre, post, label, valid = self._read(index)

        label = label.astype(np.int64)
        if self.nodata_ignore:
            label = np.where(valid, label, self.ignore_index)

        rng = sampler.rng_for(self.seed, sample_no)
        pre, post, label, valid = transforms.augment(
            pre,
            post,
            label,
            rng,
            crop=self.crop,
            flips=self.flips,
            rot90=self.rot90,
            extra=(valid,),
        )

        clf, loc = self._targets(label)
        return {
            "name": name,
            "pre": normalize.normalize_optical(pre),
            "post": normalize.sar_to_input(post, self.sar_mode, self.sar_stats),
            "clf": clf,
            "loc": loc,
            "valid": np.asarray(valid, dtype=np.bool_),
        }
