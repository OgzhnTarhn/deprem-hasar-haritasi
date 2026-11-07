"""Command line interface of the ``hasar`` package.

Three subcommands are wired here:

``indir``
    Download and extract the BRIGHT archives into a data root (resumable).
``veri-ozet``
    Write a JSON summary of the zero-shot split, optionally with per-class label pixel
    counts for the train and val parts.
``yeniden-uret``
    Re-run a published BRIGHT zero-shot baseline through :mod:`hasar.eval.reproduce`.

The entry point is :func:`main`, which returns an exit code instead of calling
:func:`sys.exit` so that tests can call it directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from hasar.data.download import prepare_bright
from hasar.data.events import DEFAULT_LIST_DIR, list_digest, load_standard_lists, make_zero_shot_split
from hasar.data.summary import build_split_report

#: Published checkpoint file of each baseline (kept here so parsing needs no torch).
_REPRO_CHECKPOINTS: dict[str, str] = {
    "unet": "ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth",
    "siamcrnn": "ckpt_SiamCRNN_cross_event_zeroshot_turkey-earthquake.pth",
    "damageformer": "ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth",
    "deeplabv3plus": "ckpt_DeepLabV3Plus_cross_event_zeroshot_turkey-earthquake.pth",
}


def _load_reproduce() -> SimpleNamespace:
    """Import the reproduction runner lazily and return it with its checkpoint downloader.

    ``hasar.eval.reproduce`` pulls in torch through the model wrappers, so it is imported only when
    the ``yeniden-uret`` subcommand actually runs.
    """
    from hasar.eval import reproduce as module
    from hasar.models.bright_baselines import download_checkpoint

    return SimpleNamespace(reproduce=module.reproduce, download_checkpoint=download_checkpoint)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the ``indir``, ``veri-ozet`` and ``yeniden-uret`` subcommands."""
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

    reproduce = subparsers.add_parser("yeniden-uret", help="Yayımlanmış bir zero-shot taban modelini yeniden çalıştır.")
    reproduce.add_argument("--model", required=True, choices=sorted(_REPRO_CHECKPOINTS), help="Taban model anahtarı.")
    reproduce.add_argument("--data", type=Path, required=True, help="BRIGHT veri kökü.")
    reproduce.add_argument("--out", type=Path, required=True, help="Sonuçların yazılacağı klasör.")
    reproduce.add_argument("--split", default="test", choices=("test", "val"), help="Değerlendirilecek bölme.")
    reproduce.add_argument("--checkpoint", type=Path, default=None, help="Hazır checkpoint yolu (indirilmez).")
    reproduce.add_argument("--cache", type=Path, default=None, help="Checkpoint önbellek klasörü.")
    reproduce.add_argument("--device", default="cpu", help="Çalıştırma aygıtı.")
    reproduce.add_argument("--tta", action="store_true", help="Test zamanı artırımı uygula.")

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


def _run_yeniden_uret(args: argparse.Namespace) -> int:
    """Re-run one published baseline and print a one-line Turkish summary."""
    runner = _load_reproduce()
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        cache = Path(args.cache) if args.cache is not None else Path(args.data) / "checkpoints"
        checkpoint_path = runner.download_checkpoint(_REPRO_CHECKPOINTS[args.model], cache)

    record = runner.reproduce(
        args.model,
        args.data,
        args.out,
        split=args.split,
        checkpoint_path=checkpoint_path,
        device=args.device,
        tta=args.tta,
    )
    miou_pct = record.values["miou_pct"]
    print(f"{args.model} {args.split}: {miou_pct:.2f} {record.metric_tag}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the selected subcommand; return its exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "indir":
        return _run_indir(args)
    if args.command == "yeniden-uret":
        return _run_yeniden_uret(args)
    return _run_veri_ozet(args)


if __name__ == "__main__":
    raise SystemExit(main())
