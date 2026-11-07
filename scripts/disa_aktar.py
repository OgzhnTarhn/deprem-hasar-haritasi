"""Deployment export trial: torch-export and ONNX on the project model (Faz 4, feeds Faz 9).

Builds a model from the registry, tries every export format with :func:`hasar.export.try_export`,
measures the CPU forward latency and writes a plain research JSON (``kind: research``, not a metric
record) to ``results/arastirma/disa_aktarim_<model>.json``, one file per model.

Usage:
    uv run python scripts/disa_aktar.py --model changemamba_mmbda_tiny --size 256 --device cpu
    uv run python scripts/disa_aktar.py --checkpoint data/checkpoints/best.pt --out data/export
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hasar.export import latency_ms, try_export  # noqa: E402
from hasar.models import changemamba, registry  # noqa: E402

#: Folder of the research reports (project rule 5: numbers come from a file).
REPORT_DIR = REPO / "results" / "arastirma"


def build(model_name: str, checkpoint: Path | None, device: str) -> torch.nn.Module:
    """Build ``model_name`` from the registry and load ``checkpoint`` (key ``"model"``) if given."""
    model = registry.build(model_name)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    return model.to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Modeli torch-export ve ONNX'e aktarmayı dener.")
    parser.add_argument("--model", default="changemamba_mmbda_tiny", help="kayıtlı model adı")
    parser.add_argument("--checkpoint", type=Path, default=None, help="best.pt ('model' anahtarlı)")
    parser.add_argument("--out", type=Path, default=Path("data/export"), help="dışa aktarım klasörü")
    parser.add_argument("--size", type=int, default=256, help="örnek girdi kenarı (piksel)")
    parser.add_argument("--device", default="cpu", help="model cihazı")
    args = parser.parse_args()

    model = build(args.model, args.checkpoint, args.device)
    results = try_export(model, args.out, size=args.size)
    for entry in results.values():
        # Only the file name: a local absolute path does not belong in a committed report.
        if entry.get("path"):
            entry["path"] = Path(entry["path"]).name
    latency = latency_ms(model, size=args.size)

    payload = {
        "kind": "research",
        "question": "Model dağıtım için hangi biçimde dışa aktarılabiliyor ve CPU gecikmesi ne?",
        "model": args.model,
        "checkpoint": args.checkpoint.name if args.checkpoint is not None else None,
        "size": args.size,
        "device": args.device,
        "torch_version": torch.__version__,
        "scan_backend": changemamba.scan_backend(),
        "latency_ms": latency,
        "formats": results,
        "note": "Faz 4 ONNX denemesi; Faz 9 dışa aktarım kararına girdi. Metrik kaydı değildir.",
    }
    report = REPORT_DIR / f"disa_aktarim_{args.model}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: entry["ok"] for name, entry in results.items()}, ensure_ascii=False))
    print(f"latency_ms={latency:.2f} -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
