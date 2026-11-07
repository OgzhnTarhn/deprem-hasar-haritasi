# Durum

Her faz sonunda güncellenir: ne bitti, ne ölçüldü, sıradaki faz ne.

| Faz | Durum | Not |
|---|---|---|
| Araştırma ve tasarım | Bitti (6 Kasım 2026) | `reports/` (üç rapor), `docs/tasarim.md` sürüm 3, `docs/fazlar.md` sürüm 2, `docs/fikir-havuzu.md` |
| 0. Belgeler ve protokol taslağı | Bitti (6 Kasım 2026) | `docs/adr/` (0001–0006), `docs/protokol.md` 0.1, `docs/veri-lisans.md`, `docs/riskler.md`, `docs/sar-temelleri.md` |
| 1. Depo iskeleti | Bitti (6 Kasım 2026) | `src/hasar/` paketi, `configs/ana.yaml`, sonuç şeması, Türkiye guard'ı, model sözleşmesi, CI; 125 kısa test (~3 sn) |
| 2. Veri katmanı | Bitti (6 Kasım 2026) | `src/hasar/data/` (8 modül), `hasar` komutu, `results/veri/bolme.json`, `results/arastirma/` (5 araştırma); 288 kısa test (~3,5 sn) |
| 3. Metrik ve yeniden üretim | Bitti (7 Kasım 2026) | `src/hasar/eval/` (metrik, bootstrap, bina F1, çıkarım, yeniden üretim), `hasar yeniden-uret`; 3 checkpoint ±0,3 içinde (yerel RTX 4060), DeepLabV3+ olumsuz bulgu |
| 4. Model, eğitim motoru, ön kontrol | Kod bitti (7 Kasım 2026); Colab adımları kullanıcıda | `models/changemamba.py`, `train/` (kayıp, zamanlayıcı, EMA, kayıt, döngü), `preflight/gates.py` (11 kapı), `hasar agirlik/egit/on-kontrol`, protokol 1.0; kapılar 1–4, 6, 7, 10, 11 yerel GPU'da geçti |
| 5–9 | Bekliyor | |

## Faz 4'te kurulanlar ve ölçülenler

- **Model:** `ChangeMambaWrapper` resmî 0229flex yapılandırmasıyla; tek bantlı SAR'ı kendisi 3 kanala kopyalar.
  ImageNet VMamba-Tiny ağırlığı 222 tensörün 200'ünü yükler. Yedek: torchvision ImageNet Swin-T ile DamageFormer.
- **Tarama yolu:** derlenmiş `selective_scan_cuda_oflex` yoksa saf PyTorch taraması çalışır, ama çok yavaş
  (yerel RTX 4060'ta 2×256 kare, bf16: adım başına ~25 sn, 3,8 GB). Bu yüzden çekirdek derlenmezse eğitim
  otomatik olarak DamageFormer yedeğine geçer (`train/run.py`).
- **Eğitim motoru:** ısınma + kosinüs, EMA 0,999, atomik tam durum kaydı (model, EMA, optimizer, zamanlayıcı, adım,
  örnek sayacı, bütün RNG'ler), `DONE` ile tekrar çalıştırmada atlama, farklı yapılandırmayla devamı reddetme,
  yalnız 13 olayda doğrulama, son adımda ham ağırlık doğrulaması.
- **Ön kontrol (yerel GPU duman testi, DamageFormer yedeğiyle):** kapı 1 (3 zorunlu model ±0,3; DeepLabV3+ ek kayıt
  olarak raporlanır), 2, 3, 4, 6, 7, 10, 11 geçti. Duman testi sentetik testlerin kaçırdığı hataları buldu: kapı 4–6'da
  cihaz uyuşmazlığı, kapı 10'da tohumlanmamış model kurulumu ve GPU belirlenimsizliği. İki kesintisiz koşu bile
  GPU'da ayrışıyor (parametreler ~1e-4, BatchNorm istatistikleri yüzde düzeyinde); kapı 10 buna göre ölçüt kullanır.
- **Colab'da kalanlar:** kapı 5, 8, 9 (A100 ve ChangeMamba çekirdeği gerekir), `train.clip`, `protokol-kilidi`.
  Adımlar: `docs/colab.md`.

## Faz 3'te kurulanlar ve ölçülenler

- **Metrik (`hasar.eval.metrics`):** olay boyunca biriktirilen 4×4 matris, sınıf IoU'su (payda 1e-7), mIoU, bina
  mIoU'su, piksel doğruluğu; kare başına matrisler `npz` olarak saklanır, bootstrap onları okur.
- **Bootstrap (`hasar.eval.bootstrap`):** şehre göre katmanlı 2 km blok bootstrap (10.000 tekrar), eşleştirilmiş fark,
  "eşit" kuralı (protokol §9) ve 41,28 çıpası yorumu.
- **Bina düzeyi (`hasar.eval.building`):** CVPRW26 örnek etiketlerinden çoğunluk oyu, makro F1; bina maskeli ve oylama
  satırları protokol §6'daki gibi `argmax(clf[1:]) + 1`.
- **Çıkarım (`hasar.eval.predict`):** kareler tek tek okunur, olasılıklar `float16` önbelleğe yazılır (bir çıkarım,
  çok analiz).
- **Yeniden üretim (`hasar.eval.reproduce`, `hasar yeniden-uret`):** Zenodo 15349462 checkpoint'leri md5 denetimiyle
  indirilir; Türkiye'ye `guard.require_reproduction_access` (yalnız bilinen 4 checkpoint adı, temiz çalışma ağacı)
  ile erişilir; sonuç `[ZS-EVENT mIoU, repro]` etiketli.
- **Sonuç (`results/yeniden-uretim/*_test.json`, commit 81e1b7e, yerel RTX 4060, model başına ~4 dk):** SiamCRNN,
  UNet ve DamageFormer yayımlanan değerlere ±0,3 içinde (fark 0,01'in altında) → **çıkış ölçütü sağlandı**, hat
  resmî değerlendiriciyle birebir.
- **Olumsuz bulgu, DeepLabV3+:** `ckpt_DeepLabV3Plus_cross_event_zeroshot_turkey-earthquake.pth` Türkiye'de
  yayımlanan değerin çok üstünde sonuç veriyor (`deeplabv3plus_test.json`, `within_tolerance: false`); hasarlı ve
  yıkık sınıf IoU'ları bir zero-shot modelden beklenmeyecek kadar yüksek. Aynı hattan geçen öteki üç model birebir
  tuttuğu için hata hatta değil, checkpoint'te: dosya büyük olasılıkla Türkiye'yi görmüş bir eğitimden
  (yanlış adlandırılmış). Zenodo md5'i `standard_ML` ve `oneshot_turkey` dosyalarından farklı. **Karar:** bu
  checkpoint kural 2'nin ruhuyla hiçbir topluluk (B2) veya analizde kullanılmaz; yalnız bu bulgu olarak raporlanır.
- **13 olay doğrulaması (B2 adayı için):** `results/yeniden-uretim/*_val.json` `[VAL13 mIoU, repro]`.

## Faz 2'de kurulanlar ve ölçülenler

- **Veri katmanı:** `events` (olay adı, zero-shot bölme, sızıntı denetimi, liste özeti), `io` (GeoTIFF okuma, nodata
  maskesi, başlık meta verisi), `transforms` (eşleşik kırpma/çevirme/rot90), `sampler` (`seed + örnek sayacı` ile
  tam devam edebilen örnekleyici; uzunluk `adım × batch`), `normalize` (resmî SAR biçimi ve A3 için akışlı eğitim
  istatistiği), `dataset` (`BrightDataset`), `download` (HF'ten indirme, `.done` işaretli güvenli açma),
  `summary` ve `hasar veri-ozet` / `hasar indir` komutları.
- **Türkiye kapısı veri katmanında:** `io.read_tile`, `read_label`, `BrightDataset` ve `compute_sar_stats` Türkiye
  adını `TurkeyAccess` olmadan diske dokunmadan reddeder; eğitim modunda Türkiye hiçbir koşulda kabul edilmez.
- **Bölme (`results/veri/bolme.json`):** 2.795 / 337 / 1.114 kare. Arka plan piksel payı eğitimde %90,76, doğrulamada
  %90,34 → kapı 7'nin beklenen değeri ≈ 0,226.
- **SAR ölçeği:** doğrusal genlik değil, log/dB benzeri; olay ortalamaları 41–104 (`sar_olcek.json`).
- **Bina örnekleri:** CVPRW26 örnek etiketleri eğitimde 10 olayda, doğrulamada hiç, Türkiye'de 1.001/1.114 karede.
  Bağlı bileşen / örnek = 0,61 (binaların ~%39'u birleşiyor).
- **Türkiye blokları (yalnız koordinat):** 7 şehir, 2 km ızgarada 64 blok.
- **A1 fizibilitesi:** 14/14 olayda Sentinel-1 öncesi-öncesi ve öncesi-sonrası koherans çifti var; ilk sonrası çekim
  0–8 gün.
- **Araştırma betikleri** `scripts/arastirma/` altında; paket ve testler kullanmaz, ek bağımlılıkları `arastirma`
  grubunda (`uv sync --group arastirma`).

## Faz 1'de kurulanlar

- **Paket:** `uv` projesi (Python ≥3.10), torch ve timm `train` ekstrasında ve sabitlenmemiş (Colab'ın kurulu
  sürümleri). Alt paketler: `data`, `models`, `train`, `eval`, `analysis`, `preflight`.
- **Yapılandırma:** `hasar.config` dondurulmuş dataclass'lar; `configs/ana.yaml` varsayılanların tamamını açık
  yazar. `config_hash` sıralı JSON'un sha256'sının ilk 12 hanesi; `run_id = <deney>-<hash>`.
- **Sonuç şeması:** `hasar.results` metrik etiketi olmayan ya da tanınmayan etiketli kaydı yazmaz (kural 4–5).
- **Guard:** `hasar.guard.require_turkey_access` yalnız HEAD `protokol-kilidi` etiketindeyse ve çalışma ağacı
  temizse `TurkeyAccess` verir (kural 1).
- **Model sözleşmesi:** her model `{"loc": B×2×H×W, "clf": B×4×H×W}` döndürür; `registry` adla kurar.
- **Üçüncü taraf:** BRIGHT `standard_ML` bölme listeleri sabit commit'ten (`third_party/bright/SOURCE.md`).
- **Not:** `.gitignore`'daki `data/` kuralı `src/hasar/data`'yı da yutuyordu; kök klasöre bağlandı (`/data/`).

## Faz 0'da ölçülen ve öğrenilenler

- **Zero-shot bölme (BRIGHT kodu okundu):** eğitim = standart eğitim + standart test listesi, doğrulama = standart
  doğrulama (Türkiye hariç). Kare sayıları **2.795 / 337 / 1.114** (eski tahmin 2.141 eğitimdi).
- **Başlık metriği yolu (BRIGHT ve ChangeMamba kodu):** tam 1024 kare, `argmax(clf)`, maskesiz, olay boyunca
  biriktirilen 4×4 matris, arka plan dahil 4 sınıf ortalaması, IoU paydasında 1e-7; nodata için ayrı işlem yok.
- **SAR girdisi:** Resmî kod SAR'ı 3 kanala kopyalayıp ImageNet istatistikleriyle normalize ediyor. Tasarımdaki
  "eğitim kümesi istatistiği" bundan sapıyordu → ADR-0004 (öneri: resmî biçim, eğitim kümesi istatistiği A3'e).
- **ChangeMamba resmî reçetesi:** crop 512, batch 8, 80.000 adım (= 640.000 örnek; bizim 40.000 × 16 ile aynı).
- **Kapı 7:** "Hep arka plan" mIoU'su sabit 0,21 değil, arka plan piksel oranı / 4.
- **Geometri:** 5 katlı bir binanın radar bindirmesi 0,35 m pikselde yaklaşık 60 piksel; SAR izi optik ayak izinden
  kayık olabilir (`docs/sar-temelleri.md` §3).
- Eğitimde yalnız üç deprem var (Haiti, Fas, Noto); öteki olaylar patlama, yanardağ, yangın, kasırga, sel, savaş.

## Açık kararlar

- **ADR-0004 (SAR girdisi):** Önerilen seçenek C, kullanıcı onayı gelene kadar varsayılan olarak uygulanır; kod
  iki biçimi de yapılandırmayla destekler.
- **Protokol §9 "eşit" kuralı:** Öneri bekliyor (Faz 4 öncesi).
- **Protokol §6–8 (Faz 2 önerileri):** bina düzeyi metrik için CVPRW26 örnek etiketleri (1.001 kare), oylama birimi
  olarak tahmin edilen bina bileşenleri, şehre göre katmanlı 2 km blok bootstrap. Onay Faz 4 öncesi.
- Ana eğitimin yapılacağı ay için Colab Pro mu Pro+ mı (Faz 5'ten önce).
- Fikir havuzundaki adaylar (KN1, Faz 6). N1 için kural 1 istisnasının genişletilmesi (KN1).

## Ölçülmemiş tahminler

- A100 adım hızı, tepe VRAM, CU yakımı, Hugging Face indirme süresi: Faz 3–4'te ölçülecek.
