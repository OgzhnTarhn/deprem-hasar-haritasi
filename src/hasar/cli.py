"""Command line interface of the ``hasar`` package.

Eleven subcommands are wired here:

``indir``
    Download and extract the BRIGHT archives into a data root (resumable).
``ornekler``
    Download the instance-label archive and extract the listed tiles flat into a folder
    through :mod:`hasar.data.download`.
``veri-ozet``
    Write a JSON summary of the zero-shot split, optionally with per-class label pixel
    counts for the train and val parts.
``yeniden-uret``
    Re-run a published BRIGHT zero-shot baseline through :mod:`hasar.eval.reproduce`.
``agirlik``
    Download the allowed ImageNet VMamba-Tiny initial weight through
    :mod:`hasar.models.weights`.
``egit``
    Run the single full training through :mod:`hasar.train.run`.
``on-kontrol``
    Run the pre-training gates of design section 8 through :mod:`hasar.train.run`.
``sicaklik``
    Fit the single ``clf`` temperature on the 13-event validation split through
    :mod:`hasar.eval.final_run`.
``final``
    Run the single Turkey evaluation through :mod:`hasar.eval.final_run`.
``readme``
    Regenerate the README result table from the JSON files under ``results/`` through
    :mod:`hasar.report`, or check with ``--check`` that it is current.
``demo``
    Build the static Faz 9 demo site from the Faz 6 cache through :mod:`hasar.demo`; the model
    is never rerun.

The entry point is :func:`main`, which returns an exit code instead of calling
:func:`sys.exit` so that tests can call it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from hasar.data.download import INSTANCE_SUFFIX, prepare_bright, prepare_instances
from hasar.data.events import DEFAULT_LIST_DIR, list_digest, load_standard_lists, make_zero_shot_split
from hasar.data.summary import build_split_report
from hasar.report import check_readme, update_readme

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


def _load_training() -> SimpleNamespace:
    """Import the training runner lazily and return its two entry points.

    ``hasar.train.run`` imports torch at module top, so it is imported only when the ``egit`` or
    ``on-kontrol`` subcommand actually runs.
    """
    from hasar.train import run as module

    return SimpleNamespace(run_training=module.run_training, run_preflight=module.run_preflight)


def _load_final() -> SimpleNamespace:
    """Import the Turkey evaluation runner lazily and return its two entry points.

    ``hasar.eval.final_run`` imports torch at module top, so it is imported only when the ``final``
    or ``sicaklik`` subcommand actually runs.
    """
    from hasar.eval import final_run as module

    return SimpleNamespace(run_final_cli=module.run_final_cli, fit_temperature_cli=module.fit_temperature_cli)


def _parse_gates(raw: str) -> list[int]:
    """Parse a comma separated gate list such as ``"2,3,4"`` into ``[2, 3, 4]``."""
    return [int(part) for part in raw.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with every subcommand."""
    parser = argparse.ArgumentParser(prog="hasar", description="Deprem hasar haritası araçları.")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="komut")

    download = subparsers.add_parser("indir", help="BRIGHT veri setini indir ve aç.")
    download.add_argument("--root", type=Path, required=True, help="Verinin açılacağı kök klasör.")
    download.add_argument("--cache", type=Path, required=True, help="İndirilen arşivlerin önbellek klasörü.")

    instances = subparsers.add_parser("ornekler", help="Bina örnek etiketlerini indir ve aç.")
    instances.add_argument("--dest", type=Path, required=True, help="Etiketlerin açılacağı klasör.")
    instances.add_argument("--cache", type=Path, required=True, help="İndirilen arşivin önbellek klasörü.")
    instances.add_argument(
        "--tiles",
        type=Path,
        default=Path("results") / "arastirma" / "turkiye_bina_kareleri.json",
        help="Kare adlarını taşıyan JSON dosyası (varsayılan: protokol §7'nin sabit listesi).",
    )

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

    weights = subparsers.add_parser("agirlik", help="İzin verilen başlangıç ağırlığını indir.")
    weights.add_argument(
        "--out",
        type=Path,
        default=Path("data/checkpoints"),
        help="Ağırlığın indirileceği klasör.",
    )

    train = subparsers.add_parser("egit", help="Tek tam eğitimi çalıştır.")
    train.add_argument("--config", type=Path, required=True, help="Yapılandırma YAML dosyası.")
    train.add_argument("--data", type=Path, required=True, help="BRIGHT veri kökü.")
    train.add_argument("--out", type=Path, required=True, help="Çıktıların yazılacağı klasör.")
    train.add_argument("--device", default="cuda", help="Çalıştırma aygıtı.")
    train.add_argument("--pretrained", type=Path, default=None, help="Başlangıç ağırlığı (isteğe bağlı).")
    train.add_argument("--commit", default=None, help="Kayda yazılacak commit SHA'sı (varsayılan: git HEAD).")
    train.add_argument("--steps", type=int, default=None, help="Kısa prova için adım sayısını değiştir.")

    preflight = subparsers.add_parser("on-kontrol", help="Eğitim öncesi kapıları çalıştır.")
    preflight.add_argument("--config", type=Path, required=True, help="Yapılandırma YAML dosyası.")
    preflight.add_argument("--data", type=Path, required=True, help="BRIGHT veri kökü.")
    preflight.add_argument("--out", type=Path, required=True, help="Çıktıların yazılacağı klasör.")
    preflight.add_argument("--device", default="cuda", help="Çalıştırma aygıtı.")
    preflight.add_argument("--pretrained", type=Path, default=None, help="Başlangıç ağırlığı (isteğe bağlı).")
    preflight.add_argument("--gates", default=None, help="Virgülle ayrılmış kapı numaraları (varsayılan: hepsi).")
    preflight.add_argument("--repro-dir", type=Path, default=None, help="Yeniden üretim kayıtlarının klasörü.")

    sicaklik = subparsers.add_parser("sicaklik", help="13 olay doğrulamasında sıcaklığı seç.")
    sicaklik.add_argument("--config", type=Path, required=True, help="Yapılandırma YAML dosyası.")
    sicaklik.add_argument("--data", type=Path, required=True, help="BRIGHT veri kökü.")
    sicaklik.add_argument("--runs", type=Path, required=True, help="Eğitim çıktılarının kök klasörü.")
    sicaklik.add_argument("--device", default="cuda", help="Çalıştırma aygıtı.")
    sicaklik.add_argument("--out", type=Path, default=Path("results") / "ana", help="Sonuçların yazılacağı klasör.")

    final = subparsers.add_parser("final", help="Tek Türkiye değerlendirmesini çalıştır.")
    final.add_argument("--config", type=Path, required=True, help="Yapılandırma YAML dosyası.")
    final.add_argument("--data", type=Path, required=True, help="BRIGHT veri kökü.")
    final.add_argument("--runs", type=Path, required=True, help="Eğitim çıktılarının kök klasörü.")
    final.add_argument("--device", default="cuda", help="Çalıştırma aygıtı.")
    final.add_argument("--out", type=Path, default=Path("results") / "ana", help="Sonuçların yazılacağı klasör.")
    final.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Kare olasılık önbelleğinin kökü; model başına <kök>/<run_id> kullanılır (varsayılan: data/final_cache).",
    )
    final.add_argument("--instances", type=Path, default=None, help="Bina örnek (instance) klasörü.")
    final.add_argument(
        "--blocks",
        type=Path,
        default=Path("results") / "arastirma" / "turkiye_bloklar.json",
        help="Bootstrap blok haritası JSON dosyası.",
    )

    readme = subparsers.add_parser("readme", help="README sonuç tablosunu results/ dosyalarından üret.")
    readme.add_argument("--readme", type=Path, default=Path("README.md"), help="Güncellenecek README dosyası.")
    readme.add_argument("--results", type=Path, default=Path("results"), help="Sonuç JSON dosyalarının kökü.")
    readme.add_argument("--check", action="store_true", help="Yazmadan yalnız güncel mi diye denetle.")

    demo = subparsers.add_parser("demo", help="Faz 6 önbelleğinden statik demo sitesi üret.")
    demo.add_argument("--cache", type=Path, required=True, help="Faz 6 kare olasılık önbelleği.")
    demo.add_argument("--data", type=Path, required=True, help="BRIGHT veri kökü.")
    demo.add_argument("--out", type=Path, required=True, help="Sitenin yazılacağı klasör.")

    return parser


def _run_indir(args: argparse.Namespace) -> int:
    """Download and extract the BRIGHT archives."""
    prepare_bright(args.root, args.cache)
    return 0


def _run_ornekler(args: argparse.Namespace) -> int:
    """Download and extract the instance labels and print the destination and file count."""
    dest = prepare_instances(args.dest, args.cache, args.tiles)
    dest = Path(dest)
    count = (
        sum(1 for path in dest.iterdir() if path.is_file() and path.name.endswith(INSTANCE_SUFFIX))
        if dest.is_dir()
        else 0
    )
    print(f"Örnek etiketleri: {dest} ({count} dosya)")
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


def _run_agirlik(args: argparse.Namespace) -> int:
    """Download the allowed pretrained weight and print its path."""
    from hasar.models import weights

    path = weights.download_pretrained(args.out)
    print(path)
    return 0


def _run_egit(args: argparse.Namespace) -> int:
    """Run the single full training and return its exit code."""
    runner = _load_training()
    return runner.run_training(
        config=args.config,
        data=args.data,
        out=args.out,
        device=args.device,
        pretrained=args.pretrained,
        commit=args.commit,
        steps=args.steps,
    )


def _run_on_kontrol(args: argparse.Namespace) -> int:
    """Run the pre-training gates and return 0 only if every selected gate passed."""
    runner = _load_training()
    gates = _parse_gates(args.gates) if args.gates is not None else None
    return runner.run_preflight(
        config=args.config,
        data=args.data,
        out=args.out,
        device=args.device,
        pretrained=args.pretrained,
        gates=gates,
        repro_dir=args.repro_dir,
    )


def _run_sicaklik(args: argparse.Namespace) -> int:
    """Fit the temperature on the validation split and return its exit code."""
    runner = _load_final()
    return runner.fit_temperature_cli(
        config=args.config,
        data=args.data,
        runs=args.runs,
        device=args.device,
        out=args.out,
    )


def _run_final(args: argparse.Namespace) -> int:
    """Run the single Turkey evaluation and return its exit code."""
    runner = _load_final()
    return runner.run_final_cli(
        config=args.config,
        data=args.data,
        runs=args.runs,
        device=args.device,
        out=args.out,
        cache=args.cache,
        instances=args.instances,
        blocks=args.blocks,
    )


def _run_readme(args: argparse.Namespace) -> int:
    """Regenerate the README result table, or check it with ``--check`` (exit 1 when stale)."""
    if args.check:
        return 0 if check_readme(args.readme, args.results) else 1
    update_readme(args.readme, args.results)
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    """Build the static demo site from the Faz 6 cache and print the page path."""
    from hasar import demo
    from hasar.guard import require_turkey_access

    access = require_turkey_access(Path.cwd())

    to_lonlat = None
    epsg = demo.first_epsg(args.cache, args.data)
    if epsg is not None and epsg != demo.WGS84:
        try:
            from pyproj import Transformer
        except ImportError:
            print(
                "Kareler EPSG 4326 konumunda değil ve pyproj kurulu değil: "
                "`uv sync --group arastirma` ile arastirma grubunu kurun.",
                file=sys.stderr,
            )
            return 1
        to_lonlat = Transformer.from_crs(epsg, demo.WGS84, always_xy=True).transform

    index = demo.build_demo(args.cache, args.data, args.out, access=access, to_lonlat=to_lonlat)
    print(index)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the selected subcommand; return its exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "indir":
        return _run_indir(args)
    if args.command == "ornekler":
        return _run_ornekler(args)
    if args.command == "yeniden-uret":
        return _run_yeniden_uret(args)
    if args.command == "agirlik":
        return _run_agirlik(args)
    if args.command == "egit":
        return _run_egit(args)
    if args.command == "on-kontrol":
        return _run_on_kontrol(args)
    if args.command == "sicaklik":
        return _run_sicaklik(args)
    if args.command == "final":
        return _run_final(args)
    if args.command == "readme":
        return _run_readme(args)
    if args.command == "demo":
        return _run_demo(args)
    return _run_veri_ozet(args)


if __name__ == "__main__":
    raise SystemExit(main())
