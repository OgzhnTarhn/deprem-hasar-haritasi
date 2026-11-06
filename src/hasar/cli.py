"""Command line interface of the ``hasar`` package.

Two subcommands are wired here:

``indir``
    Download and extract the BRIGHT archives into a data root (resumable).
``veri-ozet``
    Write a JSON summary of the zero-shot split, optionally with per-class label pixel
    counts for the train and val parts.

The entry point is :func:`main`, which returns an exit code instead of calling
:func:`sys.exit` so that tests can call it directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hasar.data.download import prepare_bright
from hasar.data.events import DEFAULT_LIST_DIR, list_digest, load_standard_lists, make_zero_shot_split
from hasar.data.summary import build_split_report


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the ``indir`` and ``veri-ozet`` subcommands."""
    parser = argparse.ArgumentParser(prog="hasar", description="Deprem hasar haritası araçları.")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="komut")

    download = subparsers.add_parser("indir", help="BRIGHT veri setini indir ve aç.")
    download.add_argument("--root", type=Path, required=True, help="Verinin açılacağı kök klasör.")
    download.add_argument("--cache", type=Path, required=True, help="İndirilen arşivlerin önbellek klasörü.")

    summary = subparsers.add_parser("veri-ozet", help="Bölme özetini JSON olarak yaz.")
    summary.add_argument("--out", type=Path, required=True, help="Yazılacak JSON dosyası.")
    summary.add_argument("--root", type=Path, default=None, help="Etiketlerin okunacağı veri kökü (isteğe bağlı).")
    summary.add_argument(
        "--list-dir",
        type=Path,
        default=DEFAULT_LIST_DIR,
        help="standard_ML bölme listelerinin klasörü.",
    )

    return parser


def _run_indir(args: argparse.Namespace) -> int:
    """Download and extract the BRIGHT archives."""
    prepare_bright(args.root, args.cache)
    return 0


def _run_veri_ozet(args: argparse.Namespace) -> int:
    """Write the split summary as UTF-8 JSON with a trailing newline."""
    split = make_zero_shot_split(*load_standard_lists(args.list_dir))
    report = build_split_report(split, root=args.root)
    report["kind"] = "data-summary"
    report["lists"] = {
        name: list_digest(Path(args.list_dir) / name) for name in ("train_set.txt", "val_set.txt", "test_set.txt")
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the selected subcommand; return its exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "indir":
        return _run_indir(args)
    return _run_veri_ozet(args)


if __name__ == "__main__":
    raise SystemExit(main())
