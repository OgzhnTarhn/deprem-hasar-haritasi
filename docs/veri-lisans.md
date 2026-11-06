# Veri ve Lisans Tablosu

**Sürüm:** 1 (Faz 0, 6 Kasım 2026). Projede kullanılan ya da kullanılması planlanan her veri, model ağırlığı ve
kodun kaynağı, lisansı ve kullanım biçimi. "Doğrulanacak" yazan satırlar, ilgili faz başlamadan önce kaynağın kendi
sayfasından teyit edilir. README'deki atıf tablosu bu dosyadan üretilir.

## 1. Eğitim ve test verisi

| Kaynak | İçerik | Lisans | Kullanım | Durum |
|---|---|---|---|---|
| BRIGHT, Hugging Face `Kullervo/BRIGHT` | 14 olay, 4.246 çift, optik + SAR + etiket | Veri kartı: CC BY-SA 4.0 | Eğitim (13 olay), test (Türkiye) | ✅ Kart okundu (araştırma raporu) |
| BRIGHT SAR alt kaynağı | Capella Open Data, Umbra Open Data | CC BY 4.0 (Umbra'da değiştirilmiş CC BY 4.0) | BRIGHT içinde | Doğrulanacak (Faz 2): torchgeo eklentisi bu atfı veriyor |
| BRIGHT optik alt kaynağı | Çoğunlukla Maxar Open Data | CC BY-NC 4.0 (Maxar) | BRIGHT içinde | Doğrulanacak (Faz 2): BRIGHT kartındaki CC BY-SA ile Maxar'ın NC şartı arasındaki ilişki not edilecek |
| Ukrayna, Myanmar, Meksika optiği | HF zip'inde var; BRIGHT README'si bu olayların optiğinin dağıtılamadığını yazıyor | Belirsiz | Eğitim | ⚠️ Karar: HF kopyası kullanılır, README'de not düşülür (tasarım §12). Bu olaylar çıkarılırsa hasarlı piksellerin yaklaşık %63'ü kaybolur ve 41,28 kıyası bozulur |
| BRIGHT bölme listeleri | `bda_benchmark/dataset/splitname/standard_ML/*.txt` | Apache-2.0 (depo) | Bölme (ADR-0001) | ✅ Faz 0'da okundu; commit SHA'sı Faz 2'de sabitlenir |

## 2. Model ağırlıkları

| Kaynak | İçerik | Lisans | Kullanım | Durum |
|---|---|---|---|---|
| Zenodo 14037769 | ImageNet `vssm_tiny` | CC BY 4.0 | Ana model başlangıcı | Doğrulanacak (Faz 4) |
| Zenodo 14037769 | `ckpt_ChangeMamba_bright_standard_ML_split.pth` | CC BY 4.0 | **Kullanılmaz** (Türkiye'yi gördü, ADR-0003) | – |
| Zenodo 15349462 | Türkiye zero-shot checkpoint'leri: UNet, SiamCRNN, SiamAttnUNet, DeepLabV3+, DamageFormer | CC BY 4.0 | Yeniden üretim (Faz 3); KN1'de kabul edilirse topluluk ve güven aralığı analizi | ✅ |
| Swin-T ImageNet ağırlıkları | DamageFormer yedeği için | timm / Microsoft (MIT) | Yalnız yedeğe geçilirse | Doğrulanacak (Faz 4) |
| MaRS, DINOv3 | SAR ve optik temel modeller | Belirsiz / kullanım kısıtlı | **Kullanılmaz** (ADR-0003) | – |

## 3. Kod

| Kaynak | Lisans | Kullanım |
|---|---|---|
| BRIGHT deposu (`ChenHongruixuan/BRIGHT`) | Apache-2.0 | DamageFormer, bölme mantığı, metrik formülü; `third_party/bright/` |
| ChangeMamba deposu (`ChenHongruixuan/ChangeMamba`) | Apache-2.0 | ChangeMamba-MMBDA, `selective_scan` çekirdeği; `third_party/changemamba/` |

Üçüncü taraf kod alındığı commit'le, lisans dosyasıyla ve `PATCHES.md` değişiklik listesiyle saklanır (ADR-0006).

## 4. Analiz katmanları (Faz 7)

| Kaynak | İçerik | Lisans | Kullanım | Durum |
|---|---|---|---|---|
| HOT/OSM yıkık binalar (HDX `hotosm_tur_destroyed_buildings`) | Gönüllülerin hava görüntüsünden işaretlediği yıkık binalar | ODbL (OpenStreetMap) | Y1 ikinci referans | Doğrulanacak (Faz 7) |
| ARIA DPM, Türkiye 2023 | Sentinel-1 koheransından hasar vekil haritası, 30 m | NASA verisi; "modified Copernicus Sentinel data" atfı | Y1 üçüncü referans, Y2 tabanı | Doğrulanacak (Faz 7) |
| USGS ShakeMap, 6 Şubat 2023 | Yer ivmesi (PGA) ızgarası | ABD kamu malı | N3, A2 (KN1'de seçilirse) | Doğrulanacak |
| Sentinel-1 SLC / HyP3 koheransı | 14 olay için öncesi ve sonrası | Copernicus açık veri; atıf gerekli | A1 (KN1'de seçilirse) | Fizibilite Faz 2 |

## 5. Demo (Faz 9)

| Kaynak | Lisans | Not |
|---|---|---|
| BRIGHT Türkiye kareleri | §1 | Demo yalnız bunlarla kesin yapılabilir |
| Maxar Open Data, `Kahramanmaras-turkey-earthquake-23` | CC BY-NC 4.0 | BRIGHT dışı bölgeler (DM2) için yalnız öncesi optik; ticari olmayan kullanım |
| Capella / Umbra açık verisinde Türkiye | – | 6 Kasım 2026 taramasında bulunamadı |

## 6. Sonuç dosyaları ve depo

- Depo kodu: Faz 1'de lisans seçilir (öneri: Apache-2.0, üçüncü taraf kodla uyumlu).
- Model çıktıları ve demo görselleri BRIGHT'tan türediği için CC BY-SA 4.0 ve Maxar'ın NC şartına tabidir; README'de
  belirtilir.
- Veri, checkpoint ve büyük çıktılar git'e girmez (`.gitignore`).
