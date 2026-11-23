"""Tests for the few-shot access guard (``hasar.guard.require_fewshot_access``).

Every test builds a throwaway git repository in ``tmp_path`` with an explicit git identity
and a synthetic split JSON; nothing touches real BRIGHT data or the project repository.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from hasar.data.dataset import BrightDataset
from hasar.data.io import read_label, read_tile
from hasar.fewshot.split import write_split
from hasar.guard import (
    FEWSHOT_LOCKED_FILES,
    FEWSHOT_PURPOSES,
    FEWSHOT_TAG,
    FewshotAccess,
    TurkeyAccess,
    TurkeyLockError,
    require_fewshot_access,
)

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]

ADAPT_TILES = ["turkey-earthquake_00000001", "turkey-earthquake_00000002"]
BUFFER_TILES = ["turkey-earthquake_00000003"]
TEST_TILES = ["turkey-earthquake_00000004", "turkey-earthquake_00000005"]
NON_TURKEY = "noto-earthquake_00000003"


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


def head_sha(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD")


def _sha(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _synthetic_split() -> dict[str, object]:
    return {
        "kind": "fewshot-split",
        "params": {"fraction": 0.2, "seed": 2023, "buffer_km": 1.0, "source": "synthetic.json"},
        "counts": {"adapt": len(ADAPT_TILES), "buffer": len(BUFFER_TILES), "test": len(TEST_TILES), "total": 5},
        "per_city": {"sehir_a": {"adapt": len(ADAPT_TILES), "buffer": len(BUFFER_TILES), "test": len(TEST_TILES)}},
        "adapt": list(ADAPT_TILES),
        "buffer": list(BUFFER_TILES),
        "test": list(TEST_TILES),
        "sha256": {
            "adapt": _sha(ADAPT_TILES),
            "buffer": _sha(BUFFER_TILES),
            "test": _sha(TEST_TILES),
        },
    }


def _init_fewshot_repo(path: Path, *, tag: bool = True, split: dict[str, object] | None = None) -> Path:
    """Create a repo containing all locked few-shot files, optionally tagged with ``fewshot-kilidi``."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")

    for rel, content in (
        ("docs/protokol.md", "protokol 1.4\n"),
        ("configs/fewshot_a.yaml", "loss: plain\n"),
        ("configs/fewshot_b.yaml", "loss: weighted\n"),
    ):
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    write_split(path / "results/fewshot/bolme.json", split if split is not None else _synthetic_split())

    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "fewshot kilidi")
    if tag:
        git(path, "tag", FEWSHOT_TAG)
    return path


def test_constants() -> None:
    assert FEWSHOT_TAG == "fewshot-kilidi"
    assert FEWSHOT_PURPOSES == ("adapt", "test")
    assert FEWSHOT_LOCKED_FILES == (
        "docs/protokol.md",
        "configs/fewshot_a.yaml",
        "configs/fewshot_b.yaml",
        "results/fewshot/bolme.json",
    )


def test_invalid_purpose_raises_value_error(tmp_path: Path) -> None:
    # Checked first, even before touching git.
    with pytest.raises(ValueError):
        require_fewshot_access(tmp_path / "nonexistent", "buffer")


def test_missing_tag_raises(tmp_path: Path) -> None:
    repo = _init_fewshot_repo(tmp_path / "repo", tag=False)

    with pytest.raises(TurkeyLockError, match=FEWSHOT_TAG):
        require_fewshot_access(repo, "adapt")


def test_tag_not_ancestor_raises(tmp_path: Path) -> None:
    repo = _init_fewshot_repo(tmp_path / "repo", tag=False)

    git(repo, "checkout", "-q", "-b", "side")
    (repo / "side.txt").write_text("side\n", encoding="utf-8")
    git(repo, "add", "side.txt")
    git(repo, "commit", "-q", "-m", "side")
    git(repo, "tag", FEWSHOT_TAG)
    git(repo, "checkout", "-q", "-")

    with pytest.raises(TurkeyLockError, match=FEWSHOT_TAG):
        require_fewshot_access(repo, "adapt")


def test_dirty_tree_modified_file_raises(tmp_path: Path) -> None:
    repo = _init_fewshot_repo(tmp_path / "repo")
    (repo / "docs/protokol.md").write_text("modified\n", encoding="utf-8")

    with pytest.raises(TurkeyLockError):
        require_fewshot_access(repo, "adapt")


def test_dirty_tree_untracked_file_raises(tmp_path: Path) -> None:
    repo = _init_fewshot_repo(tmp_path / "repo")
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")

    with pytest.raises(TurkeyLockError):
        require_fewshot_access(repo, "test")


def test_locked_file_changed_after_tag_raises(tmp_path: Path) -> None:
    repo = _init_fewshot_repo(tmp_path / "repo")
    (repo / "configs/fewshot_a.yaml").write_text("loss: tampered\n", encoding="utf-8")
    git(repo, "add", "configs/fewshot_a.yaml")
    git(repo, "commit", "-q", "-m", "change locked config")

    with pytest.raises(TurkeyLockError):
        require_fewshot_access(repo, "adapt")


def test_non_locked_code_file_committed_after_tag_is_allowed(tmp_path: Path) -> None:
    repo = _init_fewshot_repo(tmp_path / "repo")
    code = repo / "src/helper.py"
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text("VALUE = 1\n", encoding="utf-8")
    git(repo, "add", "src/helper.py")
    git(repo, "commit", "-q", "-m", "code fix after tag")

    access = require_fewshot_access(repo, "adapt")
    assert access.commit == head_sha(repo)
    assert access.allowed == frozenset(ADAPT_TILES)


def test_missing_split_in_tagged_commit_raises(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "docs").mkdir()
    (repo / "docs/protokol.md").write_text("p\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "no split")
    git(repo, "tag", FEWSHOT_TAG)

    with pytest.raises(TurkeyLockError):
        require_fewshot_access(repo, "adapt")



def test_locked_file_missing_in_tag_and_head_raises(tmp_path: Path) -> None:
    repo = _init_fewshot_repo(tmp_path / "repo", tag=False)
    git(repo, "rm", "-q", "configs/fewshot_b.yaml")
    git(repo, "commit", "-q", "-m", "drop config b")
    git(repo, "tag", FEWSHOT_TAG)

    with pytest.raises(TurkeyLockError):
        require_fewshot_access(repo, "adapt")

def test_sha_mismatch_committed_before_tag_raises(tmp_path: Path) -> None:
    tampered = _synthetic_split()
    tampered["sha256"] = dict(tampered["sha256"], adapt="0" * 64)  # type: ignore[arg-type]
    repo = _init_fewshot_repo(tmp_path / "repo", split=tampered)

    # Diff check passes because the tampered file was committed before the tag,
    # but load_split catches the SHA mismatch and require_fewshot_access wraps it as TurkeyLockError.
    with pytest.raises(TurkeyLockError, match="sha256"):
        require_fewshot_access(repo, "adapt")


def test_success_for_adapt_and_test(tmp_path: Path) -> None:
    repo = _init_fewshot_repo(tmp_path / "repo")
    expected_commit = head_sha(repo)

    adapt_token = require_fewshot_access(repo, "adapt")
    test_token = require_fewshot_access(repo, "test")

    assert isinstance(adapt_token, FewshotAccess)
    assert isinstance(adapt_token, TurkeyAccess)
    assert adapt_token.tag == FEWSHOT_TAG
    assert adapt_token.commit == expected_commit
    assert adapt_token.purpose == "adapt"
    assert adapt_token.allowed == frozenset(ADAPT_TILES)

    assert isinstance(test_token, FewshotAccess)
    assert test_token.tag == FEWSHOT_TAG
    assert test_token.commit == expected_commit
    assert test_token.purpose == "test"
    assert test_token.allowed == frozenset(TEST_TILES)


def test_purpose_isolation_io_and_dataset(tmp_path: Path) -> None:
    repo = _init_fewshot_repo(tmp_path / "repo")
    adapt_token = require_fewshot_access(repo, "adapt")
    test_token = require_fewshot_access(repo, "test")
    missing_root = tmp_path / "does-not-exist"

    # io.read_tile and io.read_label reject disallowed Turkey tiles before touching disk.
    with pytest.raises(TurkeyLockError, match="adapt"):
        read_tile(missing_root, TEST_TILES[0], access=adapt_token)
    with pytest.raises(TurkeyLockError, match="adapt"):
        read_label(missing_root, TEST_TILES[0], access=adapt_token)
    with pytest.raises(TurkeyLockError, match="test"):
        read_tile(missing_root, ADAPT_TILES[0], access=test_token)
    with pytest.raises(TurkeyLockError, match="test"):
        read_label(missing_root, ADAPT_TILES[0], access=test_token)

    # Dataset construction enforces purpose and allow-list.
    with pytest.raises(TurkeyLockError):
        BrightDataset(missing_root, [TEST_TILES[0]], mode="train", crop=16, access=adapt_token)
    with pytest.raises(TurkeyLockError):
        BrightDataset(missing_root, [ADAPT_TILES[0]], mode="train", crop=16, access=test_token)
    with pytest.raises(TurkeyLockError):
        BrightDataset(missing_root, [ADAPT_TILES[0]], mode="eval", access=test_token)

    # Allowed constructions succeed without reading from disk.
    train_ds = BrightDataset(missing_root, [NON_TURKEY, *ADAPT_TILES], mode="train", crop=16, access=adapt_token)
    eval_ds = BrightDataset(missing_root, list(TEST_TILES), mode="eval", access=test_token)
    assert len(train_ds) == 1 + len(ADAPT_TILES)
    assert len(eval_ds) == len(TEST_TILES)
