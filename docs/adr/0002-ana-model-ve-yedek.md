# ADR-0002: Ana model ChangeMamba-MMBDA Tiny, yedek DamageFormer

- **Durum:** Kabul
- **Tarih:** 6 Kasım 2026 (geriye dönük kayıt)
- **İlgili:** tasarım §5, rehber §6

## Bağlam

**Kanıt:** Aynı zero-shot protokolünde Türkiye sonuçları `[ZS-EVENT mIoU]`: ChangeMamba 41,28, ChangeOS 38,45,
SiamCRNN 38,32, UNet 37,82, DeepLabV3+ 37,28, DamageFormer 37,20, SiamAttnUNet 36,09. ChangeMamba dört depremin
dördünde en iyi. 6 Kasım 2026 taramasında 41,28'i geçen yayımlanmış bir yöntem bulunamadı.

**Kanıt (risk):** ChangeMamba'nın `selective_scan` CUDA çekirdeği derlenmeli. `setup.py` koşulsuz olarak
`compute_70` ekliyor; CUDA 13 Volta için çevrimdışı derlemeyi kaldırdı; Colab imajı `torch 2.14.1+cu130`.
ChangeMamba yazarları Türkiye zero-shot checkpoint'i yayımlamadı. DamageFormer'ın Türkiye zero-shot checkpoint'i
Zenodo 15349462'de var ve özel çekirdek istemiyor.

**Kanıt (resmî reçete, `changedetection/sh_file/train_MMBDA_BRIGHT.sh`):** crop 512, batch 8, `max_iters 80000`.
Güncel ChangeMamba kodunda `max_iters` adım sayısıdır (`expand_data_list`: örnek = adım × batch); yani 640.000
örnek. Tasarımdaki 40.000 adım × 16 batch da 640.000 örnek eder.

## Seçenekler

| Seçenek | Çıpa | Risk |
|---|---|---|
| ChangeMamba-MMBDA Tiny | 41,28 | Çekirdek derlemesi |
| Swin-T DamageFormer | 37,20 | Düşük |
| Yarışma çözümleri (DFC25, CVPRW26) | Kıyaslanamaz | Transdüktif veya 3–10 tam koşu |
| Temel modeller (DINOv3 vb.) | Yok | Lisans, Türkiye riski, kanıt yok |

## Karar

Ana model ChangeMamba-MMBDA Tiny, mimari ve reçete değiştirilmeden. Çekirdek Colab'da en fazla 1 saatte derlenip
20 adımlık ileri-geri testten geçmezse ana eğitim otomatik olarak Swin-T DamageFormer ile yapılır; hedef o zaman
"37,20'yi açıkça geçmek, 41,28'e yaklaşmak" olur.

## Sonuçlar

- Model sözleşmesi (ADR-0006) yedeğe geçişi tek satırlık yapılandırma değişikliğine indirir.
- ONNX dışa aktarımı ChangeMamba ile şüpheli; Faz 4'te denenir (risk R4).
- Resmî betik eğitim sırasında test olayını da değerlendiriyor (`eval_interval = 500`, bütün `eval_loaders`);
  bizim eğitim kodumuzda test olayına erişim yolu yoktur.
