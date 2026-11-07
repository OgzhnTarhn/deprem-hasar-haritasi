# Bulgular

Olumlu ve olumsuz bütün bulgular burada toplanır (kural 6). Belgede elle yazılmış sonuç sayısı yoktur (kural 5):
her bulgu, sayısını taşıyan `results/` dosyasını ve metrik etiketini gösterir. Türkiye sonuçları yalnız `final`
adımından sonra eklenir.

| Durum | Anlamı |
|---|---|
| ✅ | Ölçüldü, dosyası commit'li |
| ⏳ | Kod hazır, ölçüm bekliyor (GPU veya `final` sonrası) |

## Faz 3: Yeniden üretim

- ✅ **Hat doğru.** SiamCRNN, UNet ve DamageFormer'ın Türkiye zero-shot checkpoint'leri yayımlanan
  `[ZS-EVENT mIoU]` değerlerine ±0,3 içinde (`results/yeniden-uretim/<model>_test.json`, `extra.within_tolerance`).
  Veri okuma, ön işleme ve olay düzeyi matris yolu resmî değerlendiriciyle aynı.
- ✅ **Olumsuz bulgu, DeepLabV3+.** Zenodo 15349462'deki "zero-shot" DeepLabV3+ checkpoint'i yayımlanan değerin
  çok üstünde çıktı (`results/yeniden-uretim/deeplabv3plus_test.json`). En olası açıklama ağırlığın Türkiye'yi
  görmüş olması; kural 2 gereği topluluk adaylarından ve bütün analizlerden çıkarıldı (protokol §6,
  `docs/durum.md` Faz 3).
- ✅ 13 olay doğrulaması (`results/yeniden-uretim/<model>_val.json`, `[VAL13 mIoU, repro]`): B2 topluluk adayı
  için başlangıç noktası.

## Faz 4: Ön kontrol

- ✅ Yerel GPU'da kapı 1–4, 6, 7, 10, 11 geçti (yedek modelle; `docs/durum.md` Faz 4).
- ✅ **CUDA belirsizliği.** Kesintisiz iki koşu bile parametrelerde ~1e-4, BatchNorm istatistiklerinde birkaç yüzde
  farklı. Devam testi bu yüzden `atol` 1e-3 ve kayıplarda %1 ile ölçülür; BatchNorm yalnız raporlanır
  (protokol §14, sürüm 1.0).
- ⏳ Kapı 5, 8, 9 Colab A100'de.

## Faz 5–6: Ana model

- ⏳ 13 olay doğrulaması: `results/ana/val13.json` (`[VAL13 mIoU]`, EMA ve son adım ham ağırlık).
- ⏳ Sıcaklık: `results/ana/sicaklik.json` (yalnız 13 olay).
- ⏳ Türkiye başlığı: `results/ana/turkiye.json` (`[ZS-EVENT mIoU]`, %95 blok bootstrap aralığı, 41,28 ile §9
  hükmü). İkincil satırlar `results/ana/turkiye_<satır>.json`, bina düzeyi `turkiye_bina_f1.json`, kalibrasyon
  `turkiye_kalibrasyon.json`.

## Faz 7: Analizler (Türkiye önbelleğinden)

| Analiz | Kod | Çıktı | Durum |
|---|---|---|---|
| Y2 bina belirsizliği, risk-kapsam (AURC), analist bütçesi | `hasar.analysis.uncertainty`, `runner` | `results/analiz/y2_butce.json`, `y2_risk_kapsam.json` | ⏳ |
| Y1 referans anlaşmazlığı (BRIGHT, HOT/OSM, ARIA DPM) | `hasar.analysis.agreement` | `results/analiz/y1_*.json` | ⏳ dış katmanlar indirilecek |
| Y3 transdüktif SAR (ayrı tablo, kural 1'in ikinci istisnası) | `hasar.analysis.transductive` | `[ZS-EVENT mIoU, transduktif-y3]` | ⏳ yeniden çıkarım gerekir |
| SAR hata galerisi, Capella/Umbra kırılımı | `hasar.analysis.gallery` | `results/analiz/galeri/` | ⏳ |

## Faz 8: Varyant

- ⏳ İki model yan yana; fark eşleştirilmiş blok bootstrap ile, "eşit" kuralına göre (`hasar.eval.compare`).
