"""Event and split helpers for the BRIGHT tile lists.

Two traps live here.

**Event names.** A tile name looks like ``la_palma-volcano_00000001_post_disaster.tif``: the event
name, an underscore, the tile number and an optional suffix. To get the event name you must first
strip the suffix (``_pre_disaster``, ``_post_disaster`` or ``_building_damage``) and only then cut
the tile number with ``rsplit('_', 1)[0]``. ``split('_')[0]`` is wrong: it turns
``la_palma-volcano`` into ``la`` because event names may contain underscores.

**Zero-shot rule.** The headline metric is ``[ZS-EVENT mIoU]`` on the Turkey earthquake, which is
never seen during training. BRIGHT's ``cross_event/zero_shot`` protocol builds the split from the
``standard_ML`` lists as follows: train = standard train + standard test, val = standard val, and
Turkey is excluded from both. Every Turkey tile, wherever it appears in the lists, goes to test.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TEST_EVENT = "turkey-earthquake"
TILE_SUFFIXES = ("_pre_disaster", "_post_disaster", "_building_damage")

DEFAULT_LIST_DIR = Path(__file__).resolve().parents[3] / "third_party" / "bright" / "splitname" / "standard_ML"


class SplitLeakError(RuntimeError):
    """Raised when a split would leak the test event or repeat a tile."""


@dataclass(frozen=True)
class Split:
    """Tile names of the three split parts, in list order."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def counts(self) -> dict[str, int]:
        """Return the number of tiles per part."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def tile_stem(name: str) -> str:
    """Return the tile name without its file extension and without one known suffix."""
    stem = Path(name).name
    lowered = stem.lower()
    for extension in (".tif", ".tiff"):
        if lowered.endswith(extension):
            stem = stem[: -len(extension)]
            break
    for suffix in TILE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def event_of(name: str) -> str:
    """Return the event name of a tile name.

    The suffix is stripped first, then the tile number is cut with ``rsplit('_', 1)[0]`` so that
    underscored event names such as ``la_palma-volcano`` survive intact.
    """
    parts = tile_stem(name).rsplit("_", 1)
    if len(parts) != 2 or not parts[0] or not parts[1].isdigit():
        raise ValueError(f"Olay adı çıkarılamadı (kare numarası yok): {name!r}")
    return parts[0]


def read_list(path: Path | str) -> list[str]:
    """Read a split list file, dropping blank lines and surrounding whitespace."""
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def list_digest(path: Path | str) -> str:
    """Return the sha256 of a list file's names, one per line with LF endings.

    Hashing the parsed names instead of the raw bytes keeps the digest identical on Windows
    (CRLF checkout) and Linux/Colab (LF).
    """
    text = "".join(f"{name}\n" for name in read_list(path))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_zero_shot_split(
    train_list: list[str] | tuple[str, ...],
    val_list: list[str] | tuple[str, ...],
    test_list: list[str] | tuple[str, ...],
    test_event: str = TEST_EVENT,
) -> Split:
    """Build the zero-shot split: train = train + test, val = val, test = every test-event tile."""
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    # Train = standard train + standard test; val = standard val; every test-event tile to test.
    for names, target in ((train_list, train), (val_list, val), (test_list, train)):
        for name in names:
            if event_of(name) == test_event:
                test.append(name)
            else:
                target.append(name)

    split = Split(train=tuple(train), val=tuple(val), test=tuple(test))
    check_no_leak(split, test_event=test_event)
    return split


def check_no_leak(split: Split, test_event: str = TEST_EVENT) -> None:
    """Raise :class:`SplitLeakError` if the split leaks the test event or repeats a tile."""
    seen: set[str] = set()
    for part in ("train", "val", "test"):
        names = getattr(split, part)
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise SplitLeakError(f"'{part}' bölümünde tekrar eden kare var: {duplicates[:5]}")
        repeated = sorted(seen.intersection(names))
        if repeated:
            raise SplitLeakError(f"Kare birden fazla bölümde: {repeated[:5]}")
        seen.update(names)

    for part in ("train", "val"):
        leaked = sorted(name for name in getattr(split, part) if event_of(name) == test_event)
        if leaked:
            raise SplitLeakError(f"'{test_event}' kareleri '{part}' bölümünde: {leaked[:5]} (sızıntı).")

    foreign = sorted(name for name in split.test if event_of(name) != test_event)
    if foreign:
        raise SplitLeakError(f"Test bölümünde '{test_event}' dışında kare var: {foreign[:5]} (sızıntı).")


def load_standard_lists(
    list_dir: Path | str = DEFAULT_LIST_DIR,
) -> tuple[list[str], list[str], list[str]]:
    """Read the ``standard_ML`` train/val/test lists and return them in that order."""
    list_dir = Path(list_dir)
    return (
        read_list(list_dir / "train_set.txt"),
        read_list(list_dir / "val_set.txt"),
        read_list(list_dir / "test_set.txt"),
    )


def split_summary(split: Split, test_event: str = TEST_EVENT) -> dict[str, object]:
    """Return a JSON-friendly summary; per-tile test names are deliberately left out."""
    events: dict[str, dict[str, int]] = {}
    for part in ("train", "val"):
        counts = Counter(event_of(name) for name in getattr(split, part))
        events[part] = {event: counts[event] for event in sorted(counts)}

    return {
        "counts": split.counts(),
        "events": events,
        "test_event": test_event,
        "test_count": len(split.test),
    }
