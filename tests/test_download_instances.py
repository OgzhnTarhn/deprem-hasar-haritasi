"""Tests for the instance-label download helper (``hasar.data.download.prepare_instances``).

No test touches the network: every test injects a fake downloader that records its keyword
arguments and writes a small zip archive into ``local_dir``. The archives are built in memory
with :mod:`zipfile`, and the tile list is a tiny JSON file written by the test.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from hasar.cli import main
from hasar.data.download import (
    INSTANCE_ARCHIVE,
    INSTANCE_SUFFIX,
    REPO_ID,
    REPO_TYPE,
    prepare_instances,
)

# --- helpers ---------------------------------------------------------------


def zip_bytes(members: dict[str, bytes]) -> bytes:
    """Build an in-memory zip archive from ``{member_name: content}``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


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


def write_tiles(path: Path, tiles: list[str]) -> Path:
    """Write a tile-list JSON file with a ``tiles`` key and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tiles": tiles}), encoding="utf-8")
    return path


def instance_payload(members: dict[str, bytes]) -> dict[str, bytes]:
    """Wrap ``members`` in the instance archive payload keyed by its filename."""
    return {INSTANCE_ARCHIVE: zip_bytes(members)}


# --- constants -------------------------------------------------------------


def test_instance_constants() -> None:
    assert INSTANCE_ARCHIVE == "cvprw26_trainval_instance_labels.zip"
    assert INSTANCE_SUFFIX == "_instance_damage.json"


# --- prepare_instances -----------------------------------------------------


def test_prepare_instances_extracts_only_listed_tiles_flat(tmp_path: Path) -> None:
    dest = tmp_path / "instances"
    cache_dir = tmp_path / "cache"
    tiles_file = write_tiles(tmp_path / "tiles.json", ["a_00000001", "b_00000002"])
    downloader = FakeDownloader(
        instance_payload(
            {
                "a/b/a_00000001_instance_damage.json": b'{"a": 1}',
                "a/b/b_00000002_instance_damage.json": b'{"b": 2}',
                "a/b/c_00000003_instance_damage.json": b'{"c": 3}',
                "a/b/notes.txt": b"not an instance file",
            }
        )
    )

    result = prepare_instances(dest, cache_dir, tiles_file, downloader=downloader)

    assert isinstance(result, Path)
    assert result == dest
    assert sorted(path.name for path in dest.iterdir() if path.is_file()) == [
        ".instances.done",
        "a_00000001_instance_damage.json",
        "b_00000002_instance_damage.json",
    ]
    assert (dest / "a_00000001_instance_damage.json").read_bytes() == b'{"a": 1}'
    assert (dest / "b_00000002_instance_damage.json").read_bytes() == b'{"b": 2}'
    assert not (dest / "c_00000003_instance_damage.json").exists()
    assert not (dest / "notes.txt").exists()
    assert not (dest / "a").exists()
    assert (dest / ".instances.done").is_file()


def test_prepare_instances_fetches_instance_archive(tmp_path: Path) -> None:
    dest = tmp_path / "instances"
    cache_dir = tmp_path / "cache"
    tiles_file = write_tiles(tmp_path / "tiles.json", ["a_00000001"])
    downloader = FakeDownloader(instance_payload({"a_00000001_instance_damage.json": b"{}"}))

    prepare_instances(dest, cache_dir, tiles_file, downloader=downloader)

    assert downloader.calls == [
        {
            "repo_id": REPO_ID,
            "filename": INSTANCE_ARCHIVE,
            "repo_type": REPO_TYPE,
            "local_dir": str(cache_dir),
        }
    ]


def test_prepare_instances_second_call_skips_downloader(tmp_path: Path) -> None:
    dest = tmp_path / "instances"
    cache_dir = tmp_path / "cache"
    tiles_file = write_tiles(tmp_path / "tiles.json", ["a_00000001"])
    downloader = FakeDownloader(instance_payload({"a_00000001_instance_damage.json": b"{}"}))

    prepare_instances(dest, cache_dir, tiles_file, downloader=downloader)
    calls_after_first = len(downloader.calls)
    result = prepare_instances(dest, cache_dir, tiles_file, downloader=downloader)

    assert calls_after_first == 1
    assert len(downloader.calls) == calls_after_first
    assert result == dest
    assert (dest / "a_00000001_instance_damage.json").is_file()


def test_prepare_instances_missing_tile_raises_and_writes_no_marker(tmp_path: Path) -> None:
    dest = tmp_path / "instances"
    cache_dir = tmp_path / "cache"
    tiles_file = write_tiles(tmp_path / "tiles.json", ["a_00000001", "b_00000002"])
    downloader = FakeDownloader(instance_payload({"a_00000001_instance_damage.json": b"{}"}))

    with pytest.raises(ValueError) as excinfo:
        prepare_instances(dest, cache_dir, tiles_file, downloader=downloader)

    assert "b_00000002" in str(excinfo.value)
    assert not (dest / ".instances.done").exists()


def test_prepare_instances_names_at_most_five_missing_tiles(tmp_path: Path) -> None:
    dest = tmp_path / "instances"
    cache_dir = tmp_path / "cache"
    missing = [f"t_0000000{i}" for i in range(7)]
    tiles_file = write_tiles(tmp_path / "tiles.json", missing)
    downloader = FakeDownloader(instance_payload({}))

    with pytest.raises(ValueError) as excinfo:
        prepare_instances(dest, cache_dir, tiles_file, downloader=downloader)

    message = str(excinfo.value)
    assert sum(name in message for name in missing) <= 5
    assert not (dest / ".instances.done").exists()


def test_prepare_instances_rejects_unsafe_member(tmp_path: Path) -> None:
    dest = tmp_path / "instances"
    cache_dir = tmp_path / "cache"
    tiles_file = write_tiles(tmp_path / "tiles.json", ["x"])
    downloader = FakeDownloader(instance_payload({"../x_instance_damage.json": b"evil"}))

    with pytest.raises(ValueError):
        prepare_instances(dest, cache_dir, tiles_file, downloader=downloader)

    assert not (dest / ".instances.done").exists()
    assert not (tmp_path / "x_instance_damage.json").exists()


# --- cli: ornekler ---------------------------------------------------------


def test_ornekler_calls_prepare_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, Path, Path]] = []

    def fake_prepare_instances(dest, cache_dir, tiles_file, **kwargs) -> Path:
        calls.append((Path(dest), Path(cache_dir), Path(tiles_file)))
        return Path(dest)

    # Patch both the source module and the CLI module, so the test holds whether the CLI
    # calls ``hasar.data.download.prepare_instances`` or imported the name directly.
    monkeypatch.setattr("hasar.data.download.prepare_instances", fake_prepare_instances)
    monkeypatch.setattr("hasar.cli.prepare_instances", fake_prepare_instances, raising=False)

    dest = tmp_path / "instances"
    cache = tmp_path / "cache"
    tiles = write_tiles(tmp_path / "tiles.json", ["a_00000001"])

    code = main(["ornekler", "--dest", str(dest), "--cache", str(cache), "--tiles", str(tiles)])

    assert code == 0
    assert calls == [(dest, cache, tiles)]
