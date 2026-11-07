"""Turkey access guard: the single gate for Turkey test tiles.

Project rule 1 and protocol sections 1 and 11 allow exactly one evaluation on Turkey, in the
``final`` step, and only when the protocol tag (``protokol-kilidi``) points at a clean working
tree. This module is the only place that decides whether such access is granted: the data layer
will require a :class:`TurkeyAccess` token before it yields any Turkey tile, so no other code
path can read Turkey imagery or labels.

The only other path to Turkey is rule 1's first exception: reproducing the published zero-shot
BRIGHT checkpoints in Phase 3. That path does no training and its results are tagged ``repro``,
so it does not need the protocol tag; it does need a clean working tree, so the commit that
produced the numbers is recorded. :func:`require_reproduction_access` grants it for the known
checkpoint names only.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

TEST_EVENT = "turkey-earthquake"
PROTOCOL_TAG = "protokol-kilidi"
REPRODUCTION_TAG = "faz3-yeniden-uretim"

REPRODUCTION_CHECKPOINTS = frozenset(
    {
        "ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth",
        "ckpt_SiamCRNN_cross_event_zeroshot_turkey-earthquake.pth",
        "ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth",
        "ckpt_DeepLabV3Plus_cross_event_zeroshot_turkey-earthquake.pth",
    }
)

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


def _require_clean_tree(repo_root: Path, tag: str) -> None:
    """Raise :class:`TurkeyLockError` unless the working tree is clean."""
    status = _git(repo_root, "status", "--porcelain")
    if status.returncode != 0:
        raise TurkeyLockError(f"'{tag}' etiketi doğrulanamadı ({repo_root}): çalışma ağacı okunamadı.")
    if status.stdout.strip():
        raise TurkeyLockError(
            f"Çalışma ağacı temiz değil ({repo_root}): '{tag}' kilidi Türkiye erişimine izin vermiyor."
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

    _require_clean_tree(repo_root, tag)

    return TurkeyAccess(tag=tag, commit=tag_commit)


def require_reproduction_access(repo_root: Path | str, checkpoint: str) -> TurkeyAccess:
    """Verify a Phase 3 reproduction run and return a :class:`TurkeyAccess` token.

    Only the published zero-shot BRIGHT checkpoints in :data:`REPRODUCTION_CHECKPOINTS` are
    accepted (a full path is fine, only the file name is compared). The protocol tag is not
    required, but ``HEAD`` must be readable and the working tree must be clean (untracked files
    count as dirty) so the commit behind the numbers is recorded. Any failure raises
    :class:`TurkeyLockError`.
    """
    repo_root = Path(repo_root)

    name = Path(checkpoint).name
    if name not in REPRODUCTION_CHECKPOINTS:
        raise TurkeyLockError(
            f"'{checkpoint}' yeniden üretim için izinli bir checkpoint değil: Türkiye erişimi kapalı."
        )

    head = _git(repo_root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise TurkeyLockError(
            f"'{REPRODUCTION_TAG}' doğrulanamadı ({repo_root}): HEAD okunamadı, Türkiye erişimi kapalı."
        )
    head_commit = head.stdout.strip()

    _require_clean_tree(repo_root, REPRODUCTION_TAG)

    return TurkeyAccess(tag=REPRODUCTION_TAG, commit=head_commit)
