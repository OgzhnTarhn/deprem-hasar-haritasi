"""Tests for the pretrained weight downloader (``hasar.models.weights``) and the ``agirlik`` CLI.

Everything is data-free and offline: the module's ``_urlopen`` wrapper is monkeypatched to return an
:class:`io.BytesIO` with fake bytes, and the expected md5 of the entry is patched to the md5 of those
bytes. No torch is needed.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from hasar.cli import main

NAME = "vssm_tiny_0230_ckpt_epoch_262.pth"
URL = "https://zenodo.org/records/15479555/files/vssm_tiny_0230_ckpt_epoch_262.pth?download=1"
MD5 = "d64653bba8f6e5c0d6f4ac6275e1be61"
PAYLOAD = b"sahte agirlik dosyasi\n"


# --- helpers ---------------------------------------------------------------


@pytest.fixture()
def weights():
    """Import ``hasar.models.weights`` lazily so a missing module fails the test, not collection."""
    try:
        from hasar.models import weights as module
    except ImportError as exc:  # pragma: no cover - only while the module does not exist yet
        pytest.fail(f"hasar.models.weights bulunamadı: {exc}")
    return module


def fake_urlopen(payload: bytes, calls: list[str] | None = None):
    """Return a ``_urlopen`` replacement that serves ``payload`` and records the requested URLs."""

    def _fake(url: str) -> io.BytesIO:
        if calls is not None:
            calls.append(url)
        return io.BytesIO(payload)

    return _fake


def no_network(url: str):  # pragma: no cover - called only if the test is wrong
    raise AssertionError(f"ağa çıkılmamalı: {url}")


def patch_md5(monkeypatch: pytest.MonkeyPatch, weights, md5: str) -> None:
    """Replace the ``PRETRAINED`` entry with one carrying ``md5`` but the real url."""
    entry = weights.PRETRAINED[NAME]
    monkeypatch.setitem(weights.PRETRAINED, NAME, SimpleNamespace(url=entry.url, md5=md5))


# --- PRETRAINED ------------------------------------------------------------


def test_pretrained_has_exactly_one_entry(weights) -> None:
    assert set(weights.PRETRAINED) == {NAME}


def test_pretrained_entry_url_and_md5(weights) -> None:
    entry = weights.PRETRAINED[NAME]
    assert entry.url == URL
    assert entry.md5 == MD5


def test_forbidden_substrings_contains_standard_ml(weights) -> None:
    assert "standard_ML" in weights.FORBIDDEN_SUBSTRINGS


# --- download_pretrained: reddedilen adlar ---------------------------------


def test_download_unknown_name_raises_value_error(weights, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        weights.download_pretrained(tmp_path, "bilinmeyen.pth")


def test_download_forbidden_name_raises_value_error(weights, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden = "ckpt_standard_ML_split.pth"
    monkeypatch.setitem(weights.PRETRAINED, forbidden, SimpleNamespace(url=URL, md5=MD5))
    monkeypatch.setattr(weights, "_urlopen", no_network)

    with pytest.raises(ValueError):
        weights.download_pretrained(tmp_path, forbidden)

    assert not (tmp_path / forbidden).exists()


# --- download_pretrained: indirme ------------------------------------------


def test_download_streams_verifies_and_moves_into_place(
    weights, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    patch_md5(monkeypatch, weights, hashlib.md5(PAYLOAD).hexdigest())
    monkeypatch.setattr(weights, "_urlopen", fake_urlopen(PAYLOAD, calls))

    path = weights.download_pretrained(tmp_path)

    assert path == tmp_path / NAME
    assert path.read_bytes() == PAYLOAD
    assert not (tmp_path / f"{NAME}.part").exists()
    assert calls == [URL]


def test_download_creates_out_dir(weights, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "nested" / "agirliklar"
    patch_md5(monkeypatch, weights, hashlib.md5(PAYLOAD).hexdigest())
    monkeypatch.setattr(weights, "_urlopen", fake_urlopen(PAYLOAD))

    path = weights.download_pretrained(out_dir)

    assert path == out_dir / NAME
    assert path.read_bytes() == PAYLOAD


def test_download_md5_mismatch_raises_and_cleans_up(weights, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_md5(monkeypatch, weights, "0" * 32)
    monkeypatch.setattr(weights, "_urlopen", fake_urlopen(PAYLOAD))

    with pytest.raises(RuntimeError):
        weights.download_pretrained(tmp_path)

    assert not (tmp_path / NAME).exists()
    assert not (tmp_path / f"{NAME}.part").exists()


# --- download_pretrained: önbellek -----------------------------------------


def test_download_reuses_valid_existing_file(weights, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / NAME).write_bytes(PAYLOAD)
    patch_md5(monkeypatch, weights, hashlib.md5(PAYLOAD).hexdigest())
    monkeypatch.setattr(weights, "_urlopen", no_network)

    path = weights.download_pretrained(tmp_path)

    assert path == tmp_path / NAME
    assert path.read_bytes() == PAYLOAD


def test_download_redownloads_wrong_existing_file(weights, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / NAME).write_bytes(b"bozuk")
    calls: list[str] = []
    patch_md5(monkeypatch, weights, hashlib.md5(PAYLOAD).hexdigest())
    monkeypatch.setattr(weights, "_urlopen", fake_urlopen(PAYLOAD, calls))

    path = weights.download_pretrained(tmp_path)

    assert path == tmp_path / NAME
    assert path.read_bytes() == PAYLOAD
    assert calls == [URL]


# --- cli: agirlik ----------------------------------------------------------


def test_agirlik_calls_download_pretrained(weights, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, str | None]] = []

    def fake_download(out_dir, name=NAME) -> Path:
        calls.append((Path(out_dir), name))
        return Path(out_dir) / name

    # Patch both the source module and the CLI module, so the test holds whether the CLI
    # calls ``hasar.models.weights.download_pretrained`` or imported the name directly.
    monkeypatch.setattr(weights, "download_pretrained", fake_download)
    monkeypatch.setattr("hasar.cli.download_pretrained", fake_download, raising=False)

    code = main(["agirlik", "--out", str(tmp_path)])

    assert code == 0
    assert len(calls) == 1
    assert calls[0][0] == tmp_path
