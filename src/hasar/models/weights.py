"""Downloader for the pretrained initial weights of the main model.

The only allowed initial weight is the ImageNet VMamba-Tiny encoder
``vssm_tiny_0230_ckpt_epoch_262.pth`` from Zenodo record 15479555 (the latest version of record
14037769, CC BY 4.0). The same record also holds
``ckpt_ChangeMamba_bright_standard_ML_split.pth``, which saw Turkey during training and is
forbidden as an initial weight by project rule 2; :data:`FORBIDDEN_SUBSTRINGS` rejects it.

The module is stdlib only (no torch), so the CLI can download weights without importing the model
stack. Downloads stream into ``<name>.part`` in 1 MiB chunks, are verified against the published
md5 and only then moved into place with :func:`os.replace`.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

#: Zenodo record holding the ImageNet VMamba-Tiny encoder (latest version of 14037769).
ZENODO_RECORD = "15479555"

_CHUNK_SIZE = 1 << 20


@dataclass(frozen=True)
class Weight:
    """A downloadable weight file: its Zenodo URL and the published md5 checksum."""

    url: str
    md5: str


#: Allowed pretrained weights, keyed by file name.
PRETRAINED: dict[str, Weight] = {
    "vssm_tiny_0230_ckpt_epoch_262.pth": Weight(
        url="https://zenodo.org/records/15479555/files/vssm_tiny_0230_ckpt_epoch_262.pth?download=1",
        md5="d64653bba8f6e5c0d6f4ac6275e1be61",
    ),
}

#: File-name substrings that mark a weight as forbidden (it saw Turkey during training).
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = ("standard_ML",)


def _urlopen(url: str):
    """Open ``url`` for reading; a thin wrapper so tests can replace it without network access."""
    return urllib.request.urlopen(url)


def file_md5(path: str | Path) -> str:
    """Return the hex md5 digest of the file at ``path``."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pretrained(out_dir: str | Path, name: str = "vssm_tiny_0230_ckpt_epoch_262.pth") -> Path:
    """Download pretrained weight ``name`` into ``out_dir`` and return its path.

    An existing file with the expected md5 is reused without touching the network. Otherwise the
    download streams into ``<name>.part`` and is moved into place with :func:`os.replace` only once
    the md5 matches; a mismatch raises :class:`RuntimeError` and leaves neither file behind.

    Raises :class:`ValueError` for an unknown name or a name matching
    :data:`FORBIDDEN_SUBSTRINGS` (project rule 2: no weight that saw Turkey).
    """
    if any(marker in name for marker in FORBIDDEN_SUBSTRINGS):
        raise ValueError(f"{name}: Türkiye'yi görmüş ağırlık başlangıç ağırlığı olamaz (kural 2).")
    if name not in PRETRAINED:
        raise ValueError(f"{name}: bilinmeyen ağırlık adı; izin verilenler: {sorted(PRETRAINED)}")

    entry = PRETRAINED[name]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final = out / name
    if final.exists() and file_md5(final) == entry.md5:
        return final

    part = out / f"{name}.part"
    with _urlopen(entry.url) as response, part.open("wb") as handle:
        while chunk := response.read(_CHUNK_SIZE):
            handle.write(chunk)
    if file_md5(part) != entry.md5:
        part.unlink()
        raise RuntimeError(f"{name}: indirilen dosyanın md5'i Zenodo ile uyuşmuyor (eksik veya bozuk indirme).")
    os.replace(part, final)
    return final
