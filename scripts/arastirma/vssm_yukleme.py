"""ImageNet VMamba-Tiny ağırlığının resmî ChangeMamba kodlayıcısına ne kadar yüklendiğini ölçer.

Kullanım: uv run python scripts/arastirma/vssm_yukleme.py data/checkpoints/vssm_tiny_0230_ckpt_epoch_262.pth
Çıktı: results/arastirma/vssm_yukleme.json (kind: research). Paket ve testler bu betiği kullanmaz.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "third_party" / "changemamba"))

import torch  # noqa: E402
from changedetection.checkpoints import load_encoder_pretrained_weights  # noqa: E402
from changedetection.models.Mamba_backbone import Backbone_VSSM  # noqa: E402

# Resmî get_vssm_kwargs + vssm_tiny_224_0229flex.yaml (third_party/changemamba/SOURCE.md).
VSSM_KWARGS = dict(
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    depths=[2, 2, 4, 2],
    dims=96,
    ssm_d_state=1,
    ssm_ratio=2.0,
    ssm_rank_ratio=2.0,
    ssm_dt_rank="auto",
    ssm_act_layer="silu",
    ssm_conv=3,
    ssm_conv_bias=False,
    ssm_drop_rate=0.0,
    ssm_init="v0",
    forward_type="v3noz",
    mlp_ratio=4.0,
    mlp_act_layer="gelu",
    mlp_drop_rate=0.0,
    drop_path_rate=0.2,
    patch_norm=True,
    norm_layer="ln",
    downsample_version="v3",
    patchembed_version="v2",
    gmlp=False,
    use_checkpoint=False,
)


def _count(value: object) -> int:
    return value if isinstance(value, int) else len(value)  # type: ignore[arg-type]


def main(path: str, out: str = "results/arastirma/vssm_yukleme.json") -> None:
    warnings.filterwarnings("ignore")
    encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=None, **VSSM_KWARGS)
    info = load_encoder_pretrained_weights(encoder, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    stage3 = {int(key.split(".")[3]) for key in state if key.startswith("layers.2.blocks.")}
    payload = {
        "kind": "research",
        "question": "ImageNet vssm_tiny_0230 ağırlığı resmî 0229flex ChangeMamba kodlayıcısına ne kadar yükleniyor?",
        "source": "Zenodo 15479555 vssm_tiny_0230_ckpt_epoch_262.pth; "
        "third_party/changemamba checkpoints.load_encoder_pretrained_weights",
        "checkpoint_md5": hashlib.md5(Path(path).read_bytes()).hexdigest(),
        "encoder_depths": VSSM_KWARGS["depths"],
        "checkpoint_stage3_blocks": len(stage3),
        "loaded": _count(info["loaded_keys"]),
        "checkpoint_tensors": len(state),
        "missing": sorted(info["missing_keys"]) if not isinstance(info["missing_keys"], int) else info["missing_keys"],
        "unexpected_count": _count(info["unexpected_keys"]),
        "mismatched_count": _count(info["mismatched_keys"]),
        "note": "Ağırlığın 3. aşamasında checkpoint_stage3_blocks kadar blok var, 0229flex yapılandırması 4 blok "
        "kullanır; ilk 4 blok yüklenir, fazlası (unexpected) kullanılmaz. Resmî BRIGHT eğitimi de aynı yüklemeyle "
        "çalışır ('olduğu gibi' ilkesi). Eksik olanlar yalnız yeni çıkış normları (outnorm0-3).",
    }
    Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("loaded", "checkpoint_tensors", "unexpected_count")}))


if __name__ == "__main__":
    main(*sys.argv[1:])
