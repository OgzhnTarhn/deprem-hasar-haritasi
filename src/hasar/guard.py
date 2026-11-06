"""Turkey access guard: the single gate for Turkey test tiles.

Project rule 1 and protocol sections 1 and 11 allow exactly one evaluation on Turkey, in the
``final`` step, and only when the protocol tag (``protokol-kilidi``) points at a clean working
tree. This module is the only place that decides whether such access is granted: the data layer
will require a :class:`TurkeyAccess` token before it yields any Turkey tile, so no other code
path can read Turkey imagery or labels.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

TEST_EVENT = "turkey-earthquake"
PROTOCOL_TAG = "protokol-kilidi"

_TILE_SUFFIXES = ("_pre_disaster", "_post_disaster")


class TurkeyLockError(RuntimeError):
    """Raised when Turkey access is requested without a valid protocol lock."""


@dataclass(frozen=True)
class TurkeyAccess:
    """Proof that the protocol tag was verified; carries the locked commit SHA."""

    tag: str
    commit: str


def is_turkey_tile(name: str) -> bool:
    """Return ``True`` if ``name`` is a tile of the Turkey test event."""
    stem = name
    for suffix in _TILE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return False
    return parts[0] == TEST_EVENT


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
    )


def require_turkey_access(repo_root: Path | str, tag: str = PROTOCOL_TAG) -> TurkeyAccess:
    """Verify the protocol lock and return a :class:`TurkeyAccess` token.

    The tag must exist, must point at ``HEAD`` and the working tree must be clean (untracked
    files count as dirty). Any failure raises :class:`TurkeyLockError`.
    """
    repo_root = Path(repo_root)

    tagged = _git(repo_root, "rev-parse", "--verify", f"{tag}^{{commit}}")
    if tagged.returncode != 0:
        raise TurkeyLockError(f"'{tag}' etiketi bulunamadı ({repo_root}): Türkiye erişimi kapalı.")
    tag_commit = tagged.stdout.strip()

    head = _git(repo_root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise TurkeyLockError(f"'{tag}' etiketi doğrulanamadı ({repo_root}): HEAD okunamadı.")
    head_commit = head.stdout.strip()

    if tag_commit != head_commit:
        raise TurkeyLockError(
            f"'{tag}' etiketi HEAD'de değil ({tag_commit[:12]} != {head_commit[:12]}): Türkiye erişimi kapalı."
        )

    status = _git(repo_root, "status", "--porcelain")
    if status.returncode != 0:
        raise TurkeyLockError(f"'{tag}' etiketi doğrulanamadı ({repo_root}): çalışma ağacı okunamadı.")
    if status.stdout.strip():
        raise TurkeyLockError(
            f"Çalışma ağacı temiz değil ({repo_root}): '{tag}' kilidi Türkiye erişimine izin vermiyor."
        )

    return TurkeyAccess(tag=tag, commit=tag_commit)
