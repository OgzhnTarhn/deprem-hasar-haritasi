# Kaynak

- Depo: https://github.com/ChenHongruixuan/ChangeMamba
- Commit: 9ce9cec13f9ea14bc0ad91f071577ec9b3a97983
- Lisans: Apache-2.0 (`LICENSE`)

## Alınan dosyalar

Yalnız ChangeMambaMMBDA modelini kurmak ve resmî kaybı yeniden üretmek için gereken dosyalar alındı. Yol yapısı
kaynaktakiyle aynıdır (`changedetection/...`), böylece dosyalardaki mutlak `changedetection.` içe aktarmaları
değişmeden çalışır. Sarmalayıcı (`src/hasar/models/changemamba.py`) bu klasörü `sys.path`'e ekler.

| Bizdeki yol | Ne işe yarar |
|---|---|
| `changedetection/models/ChangeMambaMMBDA.py` | Ana model: iki VMamba kodlayıcı, bina ve hasar kod çözücüleri |
| `changedetection/models/ChangeDecoder_BRIGHT.py`, `SemanticDecoder.py`, `decoder_factory.py` | Kod çözücüler (dört aşamada `cat` füzyonu) |
| `changedetection/models/builders.py`, `model_utils.py`, `Mamba_backbone.py` | Kurucular, norm/aktivasyon tabloları, omurga |
| `changedetection/models/vmamba.py` | VSSM bloğu ve selective scan (yamalı, `PATCHES.md`) |
| `changedetection/models/csm_triton.py` | Triton tarama çekirdekleri (v3noz yolunda kullanılmaz) |
| `changedetection/checkpoints.py`, `logging_utils.py` | Ön eğitim ağırlığı yükleme ve rapor |
| `changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml` | Resmî BRIGHT eğitiminin omurga yapılandırması |
| `changedetection/utils_func/lovasz_loss.py` | Resmî Lovász-softmax kaybı |

## Alınmayanlar

- `kernels/selective_scan/` (CUDA kaynakları): Colab defteri aynı commit'i klonlayıp çekirdeği orada derler.
  Çekirdek yoksa yama 3'teki saf PyTorch taraması devreye girer (yavaş ama aynı matematik).
- Eğitim betikleri (`script/`, `tasks/`, `engine/`, `datasets/`): eğitim döngüsü `src/hasar/train/` altında yeniden
  yazıldı. Resmî ayarlar ve kayıp ağırlıkları oradan okunarak `configs/` ve `src/hasar/train/losses.py`'ye taşındı:
  `CE(loc) + CE(clf) + 0.5 * Lovasz(loc) + 0.75 * Lovasz(clf)` (`tasks/bright.py`, `train_step`).

## Yapılandırma (resmî `get_vssm_kwargs` + 0229flex)

`patch_size=4, in_chans=3, num_classes=1000, depths=[2,2,4,2], dims=96, ssm_d_state=1, ssm_ratio=2.0,
ssm_rank_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
ssm_init="v0", forward_type="v3noz", mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, drop_path_rate=0.2,
patch_norm=True, norm_layer="ln", downsample_version="v3", patchembed_version="v2", gmlp=False,
use_checkpoint=False`; model `ChangeMambaMMBDA(output_building=2, output_damage=4, pretrained=<vssm_tiny_0230_ckpt_epoch_262.pth>)`.
