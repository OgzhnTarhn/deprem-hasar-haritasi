# Yamalar

Yalnız `changedetection/models/vmamba.py` değişti. Her yama dosyada `[hasar yama N]` yorumuyla işaretli.

| No | Değişiklik | Neden | Sonuca etkisi |
|---|---|---|---|
| 1 | `fvcore` içe aktarması `try/except` içinde | fvcore yalnız FLOP sayımı için; bağımlılığı azaltır | Yok |
| 2 | `csm_triton` içe aktarması `try/except` içinde; yoksa `None` | Windows ve CPU'da triton yok. Bizim `v3noz` yolumuz saf PyTorch `CrossScan`/`CrossMerge` kullanır | Yok |
| 3 | Saf PyTorch `selective_scan_torch` ve `SelectiveScanTorch`; `cross_selective_scan` içinde `SelectiveScanOflex` yerine, çekirdek kurulu değilse, girdi CPU'daysa veya `HASAR_SCAN_BACKEND=torch` ise kullanılır | CPU testleri ve çekirdek derlenemezse yedek yol | Matematik aynı; çekirdekle sayısal eşdeğerlik Colab'da ön kontrol kapısında ölçülür |

Yama 3 fp32 hesaplar ve autocast'i kendi içinde kapatır (çekirdek de fp32 çıktı verir, `oflex=True`).
Uzunluk boyunca Python döngüsüyle yürüdüğü için 512×512 eğitimde çok yavaştır; tam eğitim CUDA çekirdeğiyle yapılır.
