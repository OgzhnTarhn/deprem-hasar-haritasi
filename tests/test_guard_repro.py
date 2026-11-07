"""Tests for the Phase 3 reproduction access guard (``hasar.guard``).

Project rule 1 allows one exception for Turkey: reproducing published zero-shot
checkpoints in Phase 3. :func:`hasar.guard.require_reproduction_access` grants that
access for the known Zenodo checkpoints only, without requiring the protocol tag.

Every test builds a throwaway git repository in ``tmp_path`` so nothing depends on
the developer's global git configuration or on the project repository itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hasar.guard import (
    REPRODUCTION_CHECKPOINTS,
    TurkeyAccess,
    TurkeyLockError,
    require_reproduction_access,
)

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]

REPRO_TAG = "faz3-yeniden-uretim"

ALLOWED = [
    "ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth",
    "ckpt_SiamCRNN_cross_event_zeroshot_turkey-earthquake.pth",
    "ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth",
    "ckpt_DeepLabV3Plus_cross_event_zeroshot_turkey-earthquake.pth",
]


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


def test_reproduction_checkpoints_exact_set() -> None:
    assert isinstance(REPRODUCTION_CHECKPOINTS, frozenset)
    assert REPRODUCTION_CHECKPOINTS == frozenset(ALLOWED)


@pytest.mark.parametrize("checkpoint", ALLOWED)
def test_require_reproduction_access_allowed(tmp_path: Path, checkpoint: str) -> None:
    repo = init_repo(tmp_path / "repo")

    access = require_reproduction_access(repo, checkpoint)

    assert isinstance(access, TurkeyAccess)
    assert access.tag == REPRO_TAG
    assert access.commit == head_sha(repo)
    assert len(access.commit) == 40


def test_require_reproduction_access_accepts_full_path(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    checkpoint = "cache/ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth"

    access = require_reproduction_access(repo, checkpoint)

    assert access.tag == REPRO_TAG
    assert access.commit == head_sha(repo)


def test_require_reproduction_access_does_not_need_protocol_tag(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")

    # No ``protokol-kilidi`` tag exists; reproduction access must still be granted.
    access = require_reproduction_access(repo, "ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth")

    assert access.tag == REPRO_TAG


@pytest.mark.parametrize(
    "checkpoint",
    [
        "ckpt_UNet_standard_ML_split.pth",
        "ckpt_UNet_cross_event_zeroshot_noto-earthquake.pth",
        "ana.pt",
    ],
)
def test_require_reproduction_access_rejects_unknown_checkpoint(tmp_path: Path, checkpoint: str) -> None:
    repo = init_repo(tmp_path / "repo")

    with pytest.raises(TurkeyLockError):
        require_reproduction_access(repo, checkpoint)


def test_require_reproduction_access_rejects_modified_tracked_file(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(TurkeyLockError):
        require_reproduction_access(repo, "ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth")


def test_require_reproduction_access_rejects_untracked_file(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")

    with pytest.raises(TurkeyLockError):
        require_reproduction_access(repo, "ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth")


def test_require_reproduction_access_rejects_repo_without_head(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    repo.mkdir()
    git(repo, "init", "-q")

    with pytest.raises(TurkeyLockError):
        require_reproduction_access(repo, "ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth")
