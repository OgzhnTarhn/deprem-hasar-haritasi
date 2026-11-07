# Kaynak

- Depo: https://github.com/ChenHongruixuan/BRIGHT
- Commit: 59269142f3a3550320513e362692732f46486985 (6 Kasım 2026'da `master`)
- Lisans: Apache-2.0 (`LICENSE`)

## Alınan dosyalar

| Bizdeki yol | Kaynaktaki yol |
|---|---|
| `splitname/standard_ML/train_set.txt` | `bda_benchmark/dataset/splitname/standard_ML/train_set.txt` |
| `splitname/standard_ML/val_set.txt` | `bda_benchmark/dataset/splitname/standard_ML/val_set.txt` |
| `splitname/standard_ML/test_set.txt` | `bda_benchmark/dataset/splitname/standard_ML/test_set.txt` |

Zero-shot bölme bu üç listeden ADR-0001'e göre üretilir.

## Faz 3: hazır modellerin kodu

Zenodo 15349462'deki zero-shot checkpoint'lerini yeniden üretmek (Faz 3, kural 1'in birinci istisnası) için aynı
commit'ten alınan model dosyaları:

| Bizdeki yol | Kaynaktaki yol |
|---|---|
| `models/UNet.py` | `bda_benchmark/model/UNet.py` |
| `models/SiamCRNN.py` | `bda_benchmark/model/SiamCRNN.py` |
| `models/DamageFormer.py` | `bda_benchmark/model/DamageFormer.py` |
| `models/DeepLabV3Plus.py` | `bda_benchmark/model/DeepLabV3Plus.py` |
| `models/metrics_reference.py` | `bda_benchmark/util_func/metrics.py` (yalnız başvuru; `hasar.eval.metrics` buna karşı sınanır) |

Bu dosyalar `hasar.models.bright_baselines` sarmalayıcıları üzerinden yol ile yüklenir; paket olarak kurulmaz.
