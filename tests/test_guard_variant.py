"""Tests for the final-access guard (``hasar.guard.require_final_access``).

Rule 1 and the Phase 6 order (KN1 -> ``varyant-kilidi`` -> the main model's single Turkey run)
mean the ``final`` step needs both locks: the protocol tag (``protokol-kilidi``) and the variant
tag (``varyant-kilidi``), each an ancestor of ``HEAD``, on a clean working tree.

Every test builds a throwaway git repository in ``tmp_path`` so nothing depends on the
developer's global git configuration or on the project repository itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hasar.guard import (
    VARIANT_TAG,
    TurkeyAccess,
    TurkeyLockError,
    require_final_access,
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


def test_variant_tag_constant() -> None:
    assert VARIANT_TAG == "varyant-kilidi"


def test_require_final_access_rejects_no_tags(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")

    with pytest.raises(TurkeyLockError):
        require_final_access(repo)


def test_require_final_access_rejects_only_protocol_tag(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    git(repo, "tag", "protokol-kilidi")

    with pytest.raises(TurkeyLockError) as excinfo:
        require_final_access(repo)

    assert "varyant-kilidi" in str(excinfo.value)


def test_require_final_access_accepts_both_tags_on_head(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    git(repo, "tag", "protokol-kilidi")
    git(repo, "tag", "varyant-kilidi")

    access = require_final_access(repo)

    assert isinstance(access, TurkeyAccess)
    assert access.tag == "varyant-kilidi"
    assert access.commit == head_sha(repo)
    assert len(access.commit) == 40


def test_require_final_access_rejects_variant_tag_not_ancestor(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    git(repo, "tag", "protokol-kilidi")

    git(repo, "checkout", "-q", "-b", "side")
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    git(repo, "add", "b.txt")
    git(repo, "commit", "-q", "-m", "side")
    git(repo, "tag", "varyant-kilidi")
    git(repo, "checkout", "-q", "-")

    with pytest.raises(TurkeyLockError) as excinfo:
        require_final_access(repo)

    assert "varyant-kilidi" in str(excinfo.value)


def test_require_final_access_rejects_dirty_tree(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    git(repo, "tag", "protokol-kilidi")
    git(repo, "tag", "varyant-kilidi")
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(TurkeyLockError):
        require_final_access(repo)
