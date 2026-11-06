"""Tests for the Turkey access guard (``hasar.guard``).

Every test builds a throwaway git repository in ``tmp_path`` so nothing depends on the
developer's global git configuration or on the project repository itself.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest
from hasar.guard import (
    TEST_EVENT,
    TurkeyAccess,
    TurkeyLockError,
    is_turkey_tile,
    require_turkey_access,
)

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]


def git(repo: Path, *args: str) -> str:
    """Run git inside ``repo`` with an explicit identity and return stripped stdout."""
    result = subprocess.run(
        [*GIT, *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def init_repo(path: Path) -> Path:
    """Create a repository with a single commit and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    (path / "a.txt").write_text("a\n", encoding="utf-8")
    git(path, "add", "a.txt")
    git(path, "commit", "-q", "-m", "first")
    return path


def head_sha(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD")


def test_test_event_constant() -> None:
    assert TEST_EVENT == "turkey-earthquake"


@pytest.mark.parametrize(
    "name",
    [
        "turkey-earthquake_00000001",
        "turkey-earthquake_00000001_pre_disaster",
        "turkey-earthquake_00000001_post_disaster",
    ],
)
def test_is_turkey_tile_true(name: str) -> None:
    assert is_turkey_tile(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "la_palma-volcano_00000001",
        "noto-earthquake_00000003",
        "turkey-earthquake",
        "turkey-earthquakeX_00000001",
        "",
    ],
)
def test_is_turkey_tile_false(name: str) -> None:
    assert is_turkey_tile(name) is False


def test_turkey_lock_error_is_runtime_error() -> None:
    assert issubclass(TurkeyLockError, RuntimeError)


def test_turkey_access_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(TurkeyAccess)
    assert [field.name for field in dataclasses.fields(TurkeyAccess)] == ["tag", "commit"]

    access = TurkeyAccess(tag="protokol-kilidi", commit="a" * 40)
    assert access.tag == "protokol-kilidi"
    assert access.commit == "a" * 40

    with pytest.raises(dataclasses.FrozenInstanceError):
        access.tag = "other"  # type: ignore[misc]


def test_require_turkey_access_lightweight_tag(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    git(repo, "tag", "protokol-kilidi")

    access = require_turkey_access(repo)

    assert isinstance(access, TurkeyAccess)
    assert access.tag == "protokol-kilidi"
    assert access.commit == head_sha(repo)
    assert len(access.commit) == 40


def test_require_turkey_access_annotated_tag(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    git(repo, "tag", "-a", "protokol-kilidi", "-m", "protocol lock")

    access = require_turkey_access(repo)

    assert access.tag == "protokol-kilidi"
    assert access.commit == head_sha(repo)


def test_require_turkey_access_custom_tag(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    git(repo, "tag", "varyant-kilidi")

    access = require_turkey_access(repo, tag="varyant-kilidi")

    assert access.tag == "varyant-kilidi"
    assert access.commit == head_sha(repo)


def test_require_turkey_access_rejects_non_repo(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(TurkeyLockError) as excinfo:
        require_turkey_access(plain)

    assert "protokol-kilidi" in str(excinfo.value)


def test_require_turkey_access_rejects_missing_tag(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")

    with pytest.raises(TurkeyLockError) as excinfo:
        require_turkey_access(repo)

    assert "protokol-kilidi" in str(excinfo.value)


def test_require_turkey_access_rejects_commit_after_tag(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    git(repo, "tag", "protokol-kilidi")
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    git(repo, "add", "b.txt")
    git(repo, "commit", "-q", "-m", "second")

    with pytest.raises(TurkeyLockError) as excinfo:
        require_turkey_access(repo)

    assert "protokol-kilidi" in str(excinfo.value)


def test_require_turkey_access_rejects_modified_tracked_file(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    git(repo, "tag", "protokol-kilidi")
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(TurkeyLockError) as excinfo:
        require_turkey_access(repo)

    assert "protokol-kilidi" in str(excinfo.value)


def test_require_turkey_access_rejects_untracked_file(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    git(repo, "tag", "protokol-kilidi")
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")

    with pytest.raises(TurkeyLockError) as excinfo:
        require_turkey_access(repo)

    assert "protokol-kilidi" in str(excinfo.value)
