"""Tests for the BRIGHT archive download/extract helpers (``hasar.data.download``).

No test touches the network: every test injects a fake downloader that records its
keyword arguments and writes a small file into ``local_dir``. Zip archives are built
in memory or in ``tmp_path`` with :mod:`zipfile`.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from hasar.data.download import (
    ARCHIVES,
    REPO_ID,
    REPO_TYPE,
    extract_once,
    fetch,
    missing_tiles,
    prepare_bright,
)

# --- helpers ---------------------------------------------------------------


def zip_bytes(members: dict[str, bytes]) -> bytes:
    """Build an in-memory zip archive from ``{member_name: content}``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def write_zip(path: Path, members: dict[str, bytes]) -> Path:
    """Write a zip archive to ``path`` and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zip_bytes(members))
    return path


class FakeDownloader:
    """Stand-in for ``huggingface_hub.hf_hub_download`` that never hits the network."""

    def __init__(self, payloads: dict[str, bytes] | None = None) -> None:
        self.payloads = payloads or {}
        self.calls: list[dict[str, str]] = []

    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        local_dir: str,
    ) -> str:
        self.calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "local_dir": local_dir,
            }
        )
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.payloads.get(filename, b"payload"))
        return str(target)


BRIGHT_PAYLOADS: dict[str, bytes] = {
    "pre-event.zip": zip_bytes({"pre-event/a_00000001_pre_disaster.tif": b"pre"}),
    "post-event.zip": zip_bytes({"post-event/a_00000001_post_disaster.tif": b"post"}),
    "target.zip": zip_bytes({"target/a_00000001_building_damage.tif": b"target"}),
}


# --- constants -------------------------------------------------------------


def test_repo_constants() -> None:
    assert REPO_ID == "Kullervo/BRIGHT"
    assert REPO_TYPE == "dataset"


def test_archives_constant() -> None:
    assert ARCHIVES == ("pre-event.zip", "post-event.zip", "target.zip")


# --- fetch -----------------------------------------------------------------


def test_fetch_calls_downloader_with_expected_kwargs(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    downloader = FakeDownloader()

    fetch("pre-event.zip", cache_dir, downloader=downloader)

    assert downloader.calls == [
        {
            "repo_id": REPO_ID,
            "filename": "pre-event.zip",
            "repo_type": REPO_TYPE,
            "local_dir": str(cache_dir),
        }
    ]


def test_fetch_returns_path_of_downloaded_file(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    downloader = FakeDownloader()

    result = fetch("target.zip", cache_dir, downloader=downloader)

    assert isinstance(result, Path)
    assert result == cache_dir / "target.zip"
    assert result.is_file()


# --- extract_once ----------------------------------------------------------


def test_extract_once_extracts_members_and_writes_marker(tmp_path: Path) -> None:
    zip_path = write_zip(
        tmp_path / "pre-event.zip",
        {"pre-event/a_00000001_pre_disaster.tif": b"pre"},
    )
    dest = tmp_path / "out"

    extracted = extract_once(zip_path, dest)

    assert extracted is True
    assert (dest / "pre-event" / "a_00000001_pre_disaster.tif").read_bytes() == b"pre"
    assert (dest / ".pre-event.done").is_file()


def test_extract_once_second_call_does_not_reextract(tmp_path: Path) -> None:
    zip_path = write_zip(
        tmp_path / "pre-event.zip",
        {"pre-event/a_00000001_pre_disaster.tif": b"pre"},
    )
    dest = tmp_path / "out"
    assert extract_once(zip_path, dest) is True

    extracted_file = dest / "pre-event" / "a_00000001_pre_disaster.tif"
    extracted_file.unlink()

    assert extract_once(zip_path, dest) is False
    assert not extracted_file.exists()


@pytest.mark.parametrize("member", ["../evil.txt", "/abs/evil.txt"])
def test_extract_once_rejects_unsafe_member(tmp_path: Path, member: str) -> None:
    zip_path = write_zip(tmp_path / "pre-event.zip", {member: b"evil"})
    dest = tmp_path / "out"

    with pytest.raises(ValueError):
        extract_once(zip_path, dest)

    assert not (dest / ".pre-event.done").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_extract_once_corrupt_zip_raises_bad_zip_file(tmp_path: Path) -> None:
    zip_path = tmp_path / "pre-event.zip"
    zip_path.write_bytes(b"this is not a zip archive")
    dest = tmp_path / "out"

    with pytest.raises(zipfile.BadZipFile):
        extract_once(zip_path, dest)

    assert not (dest / ".pre-event.done").exists()


# --- prepare_bright --------------------------------------------------------


def test_prepare_bright_fetches_and_extracts_every_archive(tmp_path: Path) -> None:
    root = tmp_path / "bright"
    cache_dir = tmp_path / "cache"
    downloader = FakeDownloader(BRIGHT_PAYLOADS)

    result = prepare_bright(root, cache_dir, downloader=downloader)

    assert isinstance(result, Path)
    assert result == root
    assert [call["filename"] for call in downloader.calls] == list(ARCHIVES)
    assert (root / "pre-event" / "a_00000001_pre_disaster.tif").is_file()
    assert (root / "post-event" / "a_00000001_post_disaster.tif").is_file()
    assert (root / "target" / "a_00000001_building_damage.tif").is_file()
    for archive in ARCHIVES:
        assert (root / f".{Path(archive).stem}.done").is_file()


def test_prepare_bright_missing_directory_raises(tmp_path: Path) -> None:
    root = tmp_path / "bright"
    cache_dir = tmp_path / "cache"
    payloads = dict(BRIGHT_PAYLOADS)
    payloads["target.zip"] = zip_bytes({"a_00000001_building_damage.tif": b"target"})
    downloader = FakeDownloader(payloads)

    with pytest.raises(FileNotFoundError):
        prepare_bright(root, cache_dir, downloader=downloader)


def test_prepare_bright_second_call_skips_downloader(tmp_path: Path) -> None:
    root = tmp_path / "bright"
    cache_dir = tmp_path / "cache"
    downloader = FakeDownloader(BRIGHT_PAYLOADS)

    prepare_bright(root, cache_dir, downloader=downloader)
    calls_after_first = len(downloader.calls)
    prepare_bright(root, cache_dir, downloader=downloader)

    assert calls_after_first == len(ARCHIVES)
    assert len(downloader.calls) == calls_after_first
    assert (root / "pre-event" / "a_00000001_pre_disaster.tif").is_file()


# --- missing_tiles ---------------------------------------------------------


def make_tile(root: Path, name: str, *, pre: bool = True, post: bool = True, target: bool = True) -> None:
    """Create the requested files of one tile under ``root``."""
    files = {
        root / "pre-event" / f"{name}_pre_disaster.tif": pre,
        root / "post-event" / f"{name}_post_disaster.tif": post,
        root / "target" / f"{name}_building_damage.tif": target,
    }
    for path, present in files.items():
        if present:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"tile")


def test_missing_tiles_returns_empty_when_all_present(tmp_path: Path) -> None:
    make_tile(tmp_path, "a_00000001")
    make_tile(tmp_path, "b_00000002")

    assert missing_tiles(tmp_path, ["a_00000001", "b_00000002"]) == []


def test_missing_tiles_reports_each_missing_file_kind(tmp_path: Path) -> None:
    make_tile(tmp_path, "a_00000001")
    make_tile(tmp_path, "b_00000002", pre=False)
    make_tile(tmp_path, "c_00000003", post=False)
    make_tile(tmp_path, "d_00000004", target=False)

    result = missing_tiles(
        tmp_path,
        ["a_00000001", "b_00000002", "c_00000003", "d_00000004"],
    )

    assert result == ["b_00000002", "c_00000003", "d_00000004"]


def test_missing_tiles_is_sorted(tmp_path: Path) -> None:
    make_tile(tmp_path, "z_00000009", pre=False)
    make_tile(tmp_path, "m_00000005", target=False)
    make_tile(tmp_path, "a_00000001", post=False)

    result = missing_tiles(
        tmp_path,
        ["z_00000009", "m_00000005", "a_00000001"],
    )

    assert result == ["a_00000001", "m_00000005", "z_00000009"]
