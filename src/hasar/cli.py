"""Command line interface of the ``hasar`` package.

Seventeen subcommands are wired here:

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
    Download the allowed ImageNet VMamba-Tiny initial weight, or the DamageFormer zero-shot
    checkpoint selected by ``--config``.
``egit``
    Run the single full training through :mod:`hasar.train.run`.
``on-kontrol``
    Run the pre-training gates of design section 8 through :mod:`hasar.train.run`.
``sicaklik``
    Fit the single ``clf`` temperature on the 13-event validation split through
    :mod:`hasar.eval.final_run`.
``final``
    Run the single Turkey evaluation through :mod:`hasar.eval.final_run`.
``kiyas``
    Compare saved per-tile confusion matrices through :mod:`hasar.eval.compare`.
``sha256``
    Print a file's SHA-256 digest.
``fewshot-bolme``
    Write or check the canonical coordinate-only few-shot split.
``fewshot-test``
    Evaluate one few-shot arm after all arms have completed training.
``fewshot-kiyas``
    Compare the few-shot arms and the zero-shot main model on the same test tiles.
``readme``
    Regenerate the README result tables from the JSON files under ``results/`` through
    :mod:`hasar.report`, or check with ``--check`` that it is current.
``sonuc-al``
    Import downloaded Colab results, compare the models and update the README table.
``demo``
    Build the static Faz 9 demo site from the Faz 6 cache through :mod:`hasar.demo`; the model
    is never rerun.

The entry point is :func:`main`, which returns an exit code instead of calling
:func:`sys.exit` so that tests can call it directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from hasar.data.download import INSTANCE_SUFFIX, prepare_bright, prepare_instances
from hasar.data.events import DEFAULT_LIST_DIR, list_digest, load_standard_lists, make_zero_shot_split
from hasar.data.summary import build_split_report
from hasar.guard import FEWSHOT_SPLIT_FILE
from hasar.report import check_readme, update_readme

_TURKEY_BLOCKS = Path("results/arastirma/turkiye_bloklar.json")
_BUILDING_TILES = Path("results/arastirma/turkiye_bina_kareleri.json")
_FEWSHOT_RESULTS = Path(FEWSHOT_SPLIT_FILE).parent
_FEWSHOT_MATRICES = "turkiye_fewshot_kare_matrisleri.npz"

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


def _load_fewshot() -> SimpleNamespace:
    """Load the few-shot evaluation entry points only when needed."""
    from hasar.config import config_hash, load_config, run_id
    from hasar.eval.bootstrap import turkey_blocks
    from hasar.eval.fewshot_eval import compare_fewshot, require_runs_done, run_fewshot_test
    from hasar.fewshot.split import load_split
    from hasar.guard import require_fewshot_access

    def load_model(cfg, runs: Path, device: str):
        from hasar.eval.final_run import _load_model

        return _load_model(cfg, runs, device)

    return SimpleNamespace(
        require_runs_done=require_runs_done,
        run_fewshot_test=run_fewshot_test,
        compare_fewshot=compare_fewshot,
        require_fewshot_access=require_fewshot_access,
        load_split=load_split,
        load_model=load_model,
        load_config=load_config,
        run_id=run_id,
        config_hash=config_hash,
        turkey_blocks=turkey_blocks,
    )


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
    weights.add_argument("--config", type=Path, default=None, help="Yapılandırma YAML dosyası.")
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
    train.add_argument(
        "--preflight",
        type=Path,
        default=None,
        help="Ön kontrol kaydı (varsayılan: <out>/preflight.json); train.clip boşsa eşik buradan okunur.",
    )

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

    compare = subparsers.add_parser("kiyas", help="Kayıtlı kare matrislerinden iki modeli kıyasla.")
    compare.add_argument("--a", type=Path, required=True, help="A modelinin matris NPZ dosyası.")
    compare.add_argument("--b", type=Path, required=True, help="B modelinin matris NPZ dosyası.")
    compare.add_argument("--name-a", required=True, help="A modelinin adı.")
    compare.add_argument("--name-b", required=True, help="B modelinin adı.")
    compare.add_argument("--out", type=Path, required=True, help="Yazılacak kıyas JSON dosyası.")
    compare.add_argument(
        "--blocks", type=Path, default=None, help="Bootstrap blok haritası JSON dosyası (isteğe bağlı)."
    )

    sha256 = subparsers.add_parser("sha256", help="Dosyanın SHA-256 özetini yaz.")
    sha256.add_argument("path", type=Path, metavar="DOSYA", help="Özeti hesaplanacak dosya.")

    fewshot_split = subparsers.add_parser("fewshot-bolme", help="Few-shot bölmesini üret veya denetle.")
    fewshot_split.add_argument("--blocks", type=Path, default=_TURKEY_BLOCKS, help="Türkiye blok haritası dosyası.")
    fewshot_split.add_argument("--out", type=Path, default=Path(FEWSHOT_SPLIT_FILE), help="Bölme JSON dosyası.")
    fewshot_split.add_argument("--check", action="store_true", help="Bölme dosyasını yazmadan güncelliğini denetle.")

    fewshot_test = subparsers.add_parser("fewshot-test", help="Eğitimler bitince bir kolu test bölmesinde değerlendir.")
    fewshot_test.add_argument("--config", type=Path, required=True, help="Test edilecek kolun YAML dosyası.")
    fewshot_test.add_argument("--arms", type=Path, nargs="+", required=True, help="Tüm kolların YAML dosyaları.")
    fewshot_test.add_argument("--data", type=Path, required=True, help="BRIGHT veri kökü.")
    fewshot_test.add_argument("--runs", type=Path, required=True, help="Eğitim çıktılarının kök klasörü.")
    fewshot_test.add_argument("--out", type=Path, required=True, help="Sonuçların yazılacağı klasör.")
    fewshot_test.add_argument("--instances", type=Path, required=True, help="Bina örnek etiketlerinin klasörü.")
    fewshot_test.add_argument("--building-tiles", type=Path, default=_BUILDING_TILES, help="Bina örnekli kare listesi.")
    fewshot_test.add_argument("--blocks", type=Path, default=_TURKEY_BLOCKS, help="Türkiye blok haritası dosyası.")
    fewshot_test.add_argument("--device", default="cuda", help="Çalıştırma aygıtı.")

    fewshot_compare = subparsers.add_parser(
        "fewshot-kiyas",
        help="Kayıtlı matrisleri few-shot test bölmesinde kıyasla.",
    )
    fewshot_compare.add_argument(
        "--a",
        type=Path,
        default=_FEWSHOT_RESULTS / "a" / _FEWSHOT_MATRICES,
        help="A kolunun matris NPZ dosyası.",
    )
    fewshot_compare.add_argument(
        "--b",
        type=Path,
        default=_FEWSHOT_RESULTS / "b" / _FEWSHOT_MATRICES,
        help="B kolunun matris NPZ dosyası.",
    )
    fewshot_compare.add_argument(
        "--zs",
        type=Path,
        default=Path("results/ana/turkiye_kare_matrisleri_headline.npz"),
        help="Zero-shot ana modelin matris NPZ dosyası.",
    )
    fewshot_compare.add_argument("--split", type=Path, default=Path(FEWSHOT_SPLIT_FILE), help="Bölme JSON dosyası.")
    fewshot_compare.add_argument("--blocks", type=Path, default=_TURKEY_BLOCKS, help="Türkiye blok haritası dosyası.")
    fewshot_compare.add_argument(
        "--out",
        type=Path,
        default=_FEWSHOT_RESULTS / "kiyas",
        help="Kıyasların yazılacağı klasör.",
    )

    readme = subparsers.add_parser("readme", help="README sonuç tablolarını results/ dosyalarından üret.")
    readme.add_argument("--readme", type=Path, default=Path("README.md"), help="Güncellenecek README dosyası.")
    readme.add_argument("--results", type=Path, default=Path("results"), help="Sonuç JSON dosyalarının kökü.")
    readme.add_argument("--check", action="store_true", help="Yazmadan yalnız güncel mi diye denetle.")

    import_results = subparsers.add_parser("sonuc-al", help="Colab sonuçlarını al, kıyasla ve README'yi güncelle.")
    import_results.add_argument("--kaynak", type=Path, required=True, help="İndirilen sonuclar/ klasörü.")
    import_results.add_argument("--results", type=Path, default=Path("results"), help="Sonuçların hedef kökü.")
    import_results.add_argument("--readme", type=Path, default=Path("README.md"), help="Güncellenecek README dosyası.")
    import_results.add_argument(
        "--blocks",
        type=Path,
        default=Path("results") / "arastirma" / "turkiye_bloklar.json",
        help="Varsa kullanılacak bootstrap blok haritası JSON dosyası.",
    )

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

    cfg = None
    if args.config is not None:
        from hasar.config import load_config

        cfg = load_config(args.config)
    if cfg is not None and cfg.model.init == "bright_zero_shot":
        from hasar.models import bright_baselines

        path = bright_baselines.download_checkpoint(bright_baselines.BASELINES["damageformer"].checkpoint, args.out)
    else:
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
        preflight=args.preflight,
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


def _run_kiyas(args: argparse.Namespace) -> int:
    """Compare saved matrices and print the paired difference, interval and verdict."""
    from hasar.eval.bootstrap import turkey_blocks
    from hasar.eval.compare import compare_models, compare_record
    from hasar.eval.metrics import ConfusionAccumulator
    from hasar.results import write_result

    per_tile_a = ConfusionAccumulator.load_npz(args.a).per_tile
    per_tile_b = ConfusionAccumulator.load_npz(args.b).per_tile
    names_a, names_b = set(per_tile_a), set(per_tile_b)
    if names_a != names_b:
        print(
            f"Kare kümeleri farklı: yalnız A'da {len(names_a - names_b)} kare, "
            f"yalnız B'de {len(names_b - names_a)} kare.",
            file=sys.stderr,
        )
        return 1

    from hasar.train.run import _git_commit

    blocks = turkey_blocks(args.blocks) if args.blocks is not None else None
    result = compare_models(per_tile_a, per_tile_b, blocks)
    record = compare_record(result, args.name_a, args.name_b, _git_commit(), "none")
    write_result(args.out, record)
    print(
        f"{args.name_b} - {args.name_a}: {result['diff_pct']:.2f} puan {record.metric_tag}; "
        f"%95 aralık [{result['low_pct']:.2f}, {result['high_pct']:.2f}]; {result['verdict']}"
    )
    return 0


def _run_sha256(args: argparse.Namespace) -> int:
    digest = hashlib.sha256()
    try:
        with args.path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        print(f"Dosya okunamadı ({args.path}): {error}", file=sys.stderr)
        return 1
    print(digest.hexdigest())
    return 0


def _run_fewshot_split(args: argparse.Namespace) -> int:
    from hasar.fewshot.split import make_fewshot_split, write_split

    try:
        split = make_fewshot_split(args.blocks)
        if args.check:
            with TemporaryDirectory() as directory:
                expected = Path(directory) / "bolme.json"
                write_split(expected, split)
                if not args.out.is_file() or args.out.read_bytes() != expected.read_bytes():
                    print(f"Few-shot bölmesi eksik veya güncel değil: {args.out}", file=sys.stderr)
                    return 1
            print(f"Few-shot bölmesi güncel: {args.out}")
        else:
            write_split(args.out, split)
            counts = split["counts"]
            print(
                f"Few-shot bölmesi yazıldı: {args.out} "
                f"(adapt={counts['adapt']}, buffer={counts['buffer']}, test={counts['test']})"
            )
    except (OSError, ValueError, KeyError) as error:
        print(f"Few-shot bölmesi hazırlanamadı: {error}", file=sys.stderr)
        return 1
    return 0


def _run_fewshot_test(args: argparse.Namespace) -> int:
    if args.config.resolve() not in {arm.resolve() for arm in args.arms}:
        print("--config dosyası --arms listesinde bulunmalıdır.", file=sys.stderr)
        return 1

    runner = _load_fewshot()
    # Only the lock and preparation steps are reported as one-line errors; a failure inside the
    # evaluation itself (e.g. CUDA out of memory) keeps its full traceback for debugging.
    try:
        runner.require_runs_done(args.runs, {arm.stem: arm for arm in args.arms})
        repo_root = Path.cwd()
        access = runner.require_fewshot_access(repo_root, "test")
        split = runner.load_split(repo_root / FEWSHOT_SPLIT_FILE)
        cfg = runner.load_config(args.config)
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    except (OSError, KeyError) as error:
        print(f"Few-shot testi çalıştırılamadı: {error}", file=sys.stderr)
        return 1

    model = runner.load_model(cfg, args.runs, args.device)
    records = runner.run_fewshot_test(
        model,
        args.data,
        args.out,
        run_id=runner.run_id(cfg),
        split_sha256=split["sha256"]["test"],
        model_name=args.config.stem,
        config_hash=runner.config_hash(cfg),
        access=access,
        blocks=runner.turkey_blocks(args.blocks),
        instances_dir=args.instances,
        building_tiles=args.building_tiles,
        device=args.device,
    )

    record = records["test"]
    print(f"Türkiye few-shot {record.values['miou_pct']:.2f} {record.metric_tag}")
    if "bina_f1" in records:
        record = records["bina_f1"]
        print(f"Türkiye few-shot bina F1 {record.values['macro_f1_pct']:.2f} {record.metric_tag}")
    return 0


def _git_head() -> str:
    """Return the current commit without loading training code."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head.stdout.strip() or "unknown"


def _run_fewshot_compare(args: argparse.Namespace) -> int:
    runner = _load_fewshot()
    try:
        split = runner.load_split(args.split)
        records = runner.compare_fewshot(
            args.a,
            args.b,
            args.zs,
            test_names=split["test"],
            blocks=args.blocks,
            out_dir=args.out,
            commit=_git_head(),
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"Few-shot kıyası yapılamadı: {error}", file=sys.stderr)
        return 1

    for key in ("b_vs_a", "a_vs_zs-ana", "b_vs_zs-ana"):
        record = records[key]
        values, extra = record.values, record.extra
        note = f" ({extra['note']})" if "note" in extra else ""
        print(
            f"{extra['direction']}: {values['diff_pct']:.2f} puan {record.metric_tag}; "
            f"%95 aralık [{values['low_pct']:.2f}, {values['high_pct']:.2f}]; {extra['verdict']}{note}"
        )

    record = records["b_vs_a_siniflar"]
    values, extra = record.values, record.extra
    for key, label in (("damaged", "hasarlı"), ("destroyed", "yıkık")):
        print(
            f"{label}: {values[f'{key}_diff_pct']:.2f} puan {record.metric_tag}; "
            f"%95 aralık [{values[f'{key}_low_pct']:.2f}, {values[f'{key}_high_pct']:.2f}]; {extra[f'verdict_{key}']}"
        )

    record = records["zs_ana"]
    print(f"zs-ana (few-shot test bölmesi): {record.values['miou_pct']:.2f} {record.metric_tag}")
    return 0


def _run_readme(args: argparse.Namespace) -> int:
    """Regenerate the README result table, or check it with ``--check`` (exit 1 when stale)."""
    if args.check:
        return 0 if check_readme(args.readme, args.results) else 1
    update_readme(args.readme, args.results)
    return 0


def _run_sonuc_al(args: argparse.Namespace) -> int:
    """Import both final runs after checking headlines, then compare and refresh the table."""
    from hasar.results import read_result

    names = ("ana", "varyant")
    for name in names:
        source = args.kaynak / name / "turkiye.json"
        destination = args.results / name / "turkiye.json"
        if not source.is_file():
            print(f"Sonuç dosyası bulunamadı: {source}", file=sys.stderr)
            return 1
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            print(f"Mevcut headline sonucu farklı; üzerine yazılamaz: {destination}", file=sys.stderr)
            return 1

    for name in names:
        destination = args.results / name
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted((args.kaynak / name).iterdir()):
            if source.is_file() and source.suffix in (".json", ".npz"):
                target = destination / source.name
                if not target.exists() or target.read_bytes() != source.read_bytes():
                    shutil.copy2(source, target)

    matrices = {name: args.results / name / "turkiye_kare_matrisleri_headline.npz" for name in names}
    matrices["referans"] = args.results / "yeniden-uretim" / "damageformer_test_matrices.npz"
    blocks = args.blocks if args.blocks.is_file() else None
    for name_a, name_b in (("ana", "varyant"), ("referans", "ana"), ("referans", "varyant")):
        code = _run_kiyas(
            argparse.Namespace(
                a=matrices[name_a],
                b=matrices[name_b],
                name_a=name_a,
                name_b=name_b,
                out=args.results / "kiyas" / f"{name_b}_vs_{name_a}.json",
                blocks=blocks,
            )
        )
        if code:
            return code

    update_readme(args.readme, args.results)
    for name in names:
        record = read_result(args.results / name / "turkiye.json")
        print(f"{name}: {record.values['miou_pct']:.2f} [ZS-EVENT mIoU]")
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
    if args.command == "kiyas":
        return _run_kiyas(args)
    if args.command == "sha256":
        return _run_sha256(args)
    if args.command == "fewshot-bolme":
        return _run_fewshot_split(args)
    if args.command == "fewshot-test":
        return _run_fewshot_test(args)
    if args.command == "fewshot-kiyas":
        return _run_fewshot_compare(args)
    if args.command == "readme":
        return _run_readme(args)
    if args.command == "sonuc-al":
        return _run_sonuc_al(args)
    if args.command == "demo":
        return _run_demo(args)
    return _run_veri_ozet(args)


if __name__ == "__main__":
    raise SystemExit(main())
