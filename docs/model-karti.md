# Model kartı: ChangeMamba-MMBDA Tiny (BRIGHT zero-shot, Türkiye testi)

Bu kart modelin ne için yapıldığını, neyle eğitildiğini ve nerede güvenilmemesi gerektiğini anlatır. Kartta elle
yazılmış sonuç sayısı yoktur (kural 5); bütün sayılar README'deki `hasar readme` tablosundan ve aşağıda adı geçen
`results/` dosyalarından okunur.

## Model

| | |
|---|---|
| Mimari | ChangeMamba-MMBDA Tiny: iki ayrı VMamba kodlayıcı (optik ve SAR), bina ve hasar kod çözücüleri |
| Kaynak kod | `third_party/changemamba/` (ChenHongruixuan/ChangeMamba @ 9ce9cec, Apache-2.0), sarmalayıcı `src/hasar/models/changemamba.py` |
| Başlangıç ağırlığı | Yalnız ImageNet `vssm_tiny_0230_ckpt_epoch_262.pth` (Zenodo 15479555, CC BY 4.0); 222 tensörün 200'ü yüklenir (`results/arastirma/vssm_yukleme.json`) |
| Kullanılmayan ağırlıklar | `*_standard_ML_split.pth` (Türkiye'yi gördü), MaRS, DINOv3 (ADR-0003) |
| Yedek | Swin-T DamageFormer (çekirdek derlenmezse; `damageformer_swin_t`) |
| Girdi | Önce: optik RGB, ImageNet normalizasyonu. Sonra: SAR tek bant, 3 kanala kopyalanmış, ImageNet normalizasyonu (ADR-0004) |
| Çıktı | `loc`: 2 sınıf (arka plan, bina); `clf`: 4 sınıf (arka plan, sağlam, hasarlı, yıkık); piksel düzeyi |

## Eğitim

- Veri: BRIGHT `cross_event/zero_shot` bölmesi, 13 olay, 2.795 eğitim / 337 doğrulama karesi; Türkiye'nin 1.114
  karesi eğitimde, doğrulamada ve hiçbir ayarda kullanılmadı (kural 1).
- Reçete: `configs/ana.yaml` (AdamW 1e-4, 40.000 adım × 16, 640 kırpma, bf16, EMA 0,999, CE + Lovász).
- Kayıt noktası: 13 olayın doğrulamasında EMA mIoU'su en yüksek olan (`results/ana/val13.json`). Bu seçimin zayıf
  olduğu biliniyor (doğrulama-test sıralama korelasyonu ~0,35).

## Değerlendirme

- Başlık: Türkiye'de bir kez, TTA'sız, `argmax(clf)`, olay boyunca biriktirilen 4×4 matrisle 4 sınıf mIoU
  `[ZS-EVENT mIoU]` (`results/ana/turkiye.json`), şehre göre katmanlı 2 km blok bootstrap ile %95 aralık.
- İkincil satırlar (önceden ilan): flip TTA, bina maskeli, bina oylaması, nodata yok sayılmış
  (`results/ana/turkiye_<satır>.json`); bina düzeyi makro F1 1.001 karede (`results/ana/turkiye_bina_f1.json`).
- Kıyas: yalnız başlık sayısı ChangeMamba'nın yayımlanmış 41,28 `[ZS-EVENT mIoU]` değeriyle; protokol §9'daki
  "eşit" kuralı.

## Amaçlanan kullanım

- Deprem sonrası ilk saatlerde, bulut ve gece koşullarından etkilenmeyen SAR ile **kaba hasar önceliklendirmesi**:
  analistin önce nereye bakacağını sıralamak (Y2 analist bütçesi analizi).
- Araştırma: farklı olaylar arasında genelleme, referans etiketler arasındaki anlaşmazlık (Y1).

## Kullanılmaması gereken yerler

- **Bina bazında kesin karar** (yıkım, tahliye, sigorta ödemesi): piksel mIoU'su sınırlı; hasarlı sınıfı en zayıf
  sınıf olarak bilinir.
- Eğitimde olmayan sensör veya çözünürlükler (BRIGHT SAR'ı Capella ve Umbra, ~0,3–1 m); Sentinel-1 gibi 10 m
  verilerde çalışması beklenmez.
- Etiket kaynağı BRIGHT'ın optik yorumudur; HOT/OSM ve radar tabanlı DPM ile anlaşmazlıklar Y1'de raporlanır.
  "Model yanlış" demek hangi referansa göre olduğunu söylemeyi gerektirir.

## Bilinen sınırlar ve riskler

- SAR geometrisi: yüksek binalarda bindirme (layover) ve gölge, hasar izini optik ayak izinden kaydırır
  (`docs/sar-temelleri.md`).
- Eğitimde yalnız üç deprem var (Haiti, Fas, Noto); geri kalan olaylar patlama, yanardağ, yangın, kasırga, sel ve
  savaş.
- Kalibrasyon: sıcaklık 13 olayda seçilir, Türkiye'de yalnız ölçülür (`results/ana/turkiye_kalibrasyon.json`).

## Lisans

Kod Apache-2.0 üçüncü taraf kod içerir. Model çıktıları BRIGHT'tan (CC BY-SA 4.0) ve Maxar Open Data'dan
(CC BY-NC 4.0) türediği için bu şartlara tabidir; ayrıntı `docs/veri-lisans.md`.
