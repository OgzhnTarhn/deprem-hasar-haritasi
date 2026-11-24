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
| 5. Ana eğitim | Bitti (22 Kasım 2026) | Hazır DamageFormer kontrol fine-tune'u, `configs/ana.yaml`; 4000 adım, kapı 8 projeksiyonu 3,69 CU |
| 6. Türkiye final | Bitti (22 Kasım 2026); ana ve varyant birer kez | Sonuçlar `results/ana/`, `results/varyant/`, `results/kiyas/`. Kod: `eval/rows.py`, `eval/calibration.py`, `eval/final.py`, `eval/final_run.py`, `hasar sicaklik/final` |
| 7. Analizler | Kod bitti (8 Kasım 2026); ölçüm `final` sonrası | `analysis/` (uncertainty Y2, agreement Y1, transductive Y3, gallery, runner); `docs/bulgular.md`. Y1 dış katmanları ve Y3 yeniden çıkarımı kullanıcıda |
| 8. Varyant | Bitti (22 Kasım 2026) | `guard.require_final_access` (iki kilit), `eval/compare.py` (eşleştirilmiş kıyas, topluluk); A4 seçildi (KN1, 21 Kasım 2026); aynı defterde ana sonrası çalışır |
| 9. Demo ve ürünleştirme | Kod bitti (8 Kasım 2026); demo `final` sonrası | `hasar readme` (README tablosu üretildi), `hasar demo` (Leaflet, önbellekten), `hasar ornekler`, `export.py` + `scripts/disa_aktar.py` (yedek model torch-export ve ONNX'e aktarılıyor, `results/arastirma/disa_aktarim_*.json`) |
| 10. Few-shot ek deneyi | Kod bitti (23 Kasım 2026); Colab adımları kullanıcıda | Protokol 1.4 §15, ADR-0008; `fewshot/` (bölme, erişim), karışık örnekleyici, `configs/fewshot_{a,b}.yaml`, `eval/fewshot_eval.py`, `hasar fewshot-bolme/fewshot-test/fewshot-kiyas`, `notebooks/fewshot.ipynb`; 921 kısa test |

## Sonuçlar (22 Kasım 2026)

İki fine-tune da Colab'da 4000 adım bitti ve Türkiye'de birer kez değerlendirildi (`protokol-kilidi`, `varyant-kilidi`).
Sayılar şu dosyalardan: `results/{ana,varyant}/turkiye.json`, `turkiye_bina_f1.json`, `val13.json`,
`results/kiyas/*.json`, `results/yeniden-uretim/damageformer_test.json`. Aralıklar %95, eşleştirilmiş blok bootstrap.

| Model | Türkiye | Sağlam IoU | Hasarlı IoU | Yıkık IoU | 13 olay doğrulama |
|---|---|---|---|---|---|
| Referans: hazır DamageFormer | 37,20 `[ZS-EVENT mIoU, repro]` | 54,58 | 3,31 | 0,37 | 74,48 `[VAL13 mIoU, repro]` |
| Ana: kontrol fine-tune'u | 40,44 [39,54; 41,41] `[ZS-EVENT mIoU]` | 67,85 | 2,97 | 0,29 | 74,55 `[VAL13 mIoU, ema]` |
| Varyant: + SAR artırımı | 40,35 [39,43; 41,35] `[ZS-EVENT mIoU]` | 67,14 | 3,15 | 0,51 | 74,45 `[VAL13 mIoU, ema]` |

| Kıyas `[ZS-EVENT mIoU, eslestirilmis-fark]` | Fark | %95 aralık | Karar |
|---|---|---|---|
| ana − referans | +3,24 | [2,27; 4,20] | farklı |
| varyant − referans | +3,15 | [2,15; 4,17] | farklı |
| varyant − ana (birincil) | −0,09 | [−0,19; 0,04] | eşit |

Yorum:

- **Fine-tune işe yaradı.** Kısa kontrol fine-tune'u Türkiye'yi görmeden hazır modeli +3,24 puan geçti; aralık 0'ı içermiyor.
- **Kazancın hepsi sağlam sınıfından.** Sağlam IoU 54,58 → 67,85. Hasarlı ve yıkık IoU %3'ün altında kaldı;
  model Türkiye'de hasarı hâlâ neredeyse hiç ayıramıyor. Asıl açık problem bu.
- **SAR artırımı başlık metriğini değiştirmedi (olumsuz sonuç).** Varyant − ana "eşit". Varyantta hasarlı ve
  yıkık sınıflarda küçük artış var (IoU 2,97 → 3,15 ve 0,29 → 0,51; bina F1 `[ZS-EVENT bina-makro-F1]`
  hasarlı 3,81 → 4,42, yıkık 0,61 → 1,04), ama bu farklar için anlamlılık testi yapılmadı; bulgu değil, gözlem.
- **Bağlam:** ChangeMamba'nın yayımlanmış 41,28 `[ZS-EVENT mIoU]` değeri ananın %95 aralığının içinde.
  Eşleştirilmiş kıyas yapılamadığı için yalnız bağlamdır.
- 13 olay doğrulaması üç modelde de ~74,5; dağılım içi ölçüm farkı göstermiyor, Türkiye'deki farkı öngörmedi.

Sıradaki: isteğe bağlı Faz 7 analizleri (önbellek Drive'da `final_cache/`), demo (iki model + metrik paneli).

## Few-shot ek deneyi (23 Kasım 2026)

Zero-shot sonuçları kesinleşti; bu deney onların yerine geçmez, ayrı tabloda `[FS-EVENT mIoU]` etiketiyle raporlanır
ve 41,28 ile kıyaslanmaz. Soru: Türkiye etiketlerinin küçük bir kısmı ve hasar sınıflarına ağırlık, hasarlı/yıkık
sınıflarındaki açığı kapatır mı?

- **Bölme** (`results/fewshot/bolme.json`): 7 şehirde 2 km bloklar, her şehirden karelerin ~%20'si uyarlamaya;
  0,5 km tampon. Uyarlama 303, tampon 170, test 641 kare. Bölme yalnız koordinattan üretildi, etiket görülmedi.
- **Kollar:** ikisi de ana koşunun `best.pt`'sinden (sha256 kilitli), 4000 adım, lr 2e-5, son adım EMA; her adımda
  yarı Türkiye uyarlama, yarı 13 olay. Tek fark kayıp ağırlığı: A düz, B hasar ağırlıklı (sqrt ters frekans, 13 olay).
- **Kıyaslar:** B − A (birincil), A ve B − zero-shot ana, hasarlı ve yıkık için sınıf bazlı B − A; eşleştirilmiş blok bootstrap.
- **Kilit:** `fewshot-kilidi` etiketi; eğitim yalnız uyarlama, değerlendirme yalnız test karelerini okur; test model başına bir kez.

Sıradaki (kullanıcı): defterle sha256'yı al → YAML'lara yazılsın → `fewshot-kilidi` etiketi → Run All → sonuçlar `results/fewshot/`.

## Önceki plan: kontrollü fine-tune (21 Kasım 2026)

Kullanıcı [ADR-0007](adr/0007-ana-model-hazir-damageformer-fine-tune.md) kararını onayladı.
Colab L4 kapı 8'de ChangeMamba projeksiyonu 164,6 CU > 75 CU çıktı. Ana artık hazır DamageFormer
zero-shot kontrol fine-tune'u; varyant A4 SAR radyometrik artırımı. KN1 bu seçimle kapandı.
Hazır referans 37,20 `[ZS-EVENT mIoU, repro]`; ChangeMamba 41,28 `[ZS-EVENT mIoU]` yalnız bağlamdır.

Kod ve defter iki koşuyu destekliyor. Eğitim başlamadı; yeni eğitim ve `final` için Türkiye'ye dokunulmadı.
Faz 3'teki izinli yeniden üretim ayrı kayıttır. Fine-tune başına ≤ 15 CU hedefi ölçülmedi; kapı 8 ölçer.
Sıradaki iş: kullanıcı protokol 1.3'ü içeren commit'e `protokol-kilidi`ni taşır ve aynı commit'e
`varyant-kilidi` koyar. Defter `origin/main` kullandığından değişiklikler oraya alınmış olmalı.
Tek “Run All” önce ana, sonra varyant için ön kontrol → fine-tune → `sicaklik` → `final` çalıştırır.
Kayıt noktası son adımın EMA'sı; birincil kıyas `hasar kiyas` ile varyant − ana.
Ayrıntı: [colab.md](colab.md), [protokol.md](protokol.md).

## Faz 4'te kurulanlar ve ölçülenler (önceki akışın kaydı)

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
- **Colab'a uyum (12–13 Kasım 2026):** ilk Colab denemelerinde çıkan sorunlara göre düzeltildi. Kapı 5 bf16'da
  koşuyor; ön kontrolde çöken kapı kaydediliyor ve sonuç her kapıdan sonra diske yazılıyor (yarım kayıt
  kaybolmuyor). Colab A100 yerine L4 verirse defter onu da kabul ediyor, kapı 8 maliyeti çalışan karta göre
  hesaplıyor. Bellek yetmezse eğitim döngüsü mikro partiyi yarıya indirip gradyan biriktiriyor, etkin batch
  değişmiyor (`train.micro_batch`). ChangeMamba çekirdeği CUDA 13 için yamayla derleniyor, tekerlek Drive'da
  saklanıyor.
- **Tek "Run All" (18 Kasım 2026, protokol 1.2):** `train.clip` YAML'da boş kalıyor; `hasar egit` eşiği aynı
  yapılandırmanın geçmiş ön kontrolünün 9. kapısından okuyor. Bu yüzden `protokol-kilidi` ön kontrolden önce atıldı.
  Defterde doldurulacak alan yok: aşamayı git etiketlerinden seçiyor (kilit yoksa yalnız ön kontrol; kilit varsa
  ön kontrol + eğitim; `varyant-kilidi` varsa Türkiye `final` de). Ön kontrol kaydı
  `kosular/onkontrol/<NAME>/preflight.json`; `scripts/defter_kontrol.py` defterin yapısını denetliyor.

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

## Kilitler ve sıradaki işler

- SAR `official` biçimde kalır; protokolün “eşit”, bina ve blok bootstrap kuralları sürer.
- KN1 varyant kararı kapandı: A4 seçildi, topluluk 1.3 kapsamı dışında. Diğer fikirler ayrı karar ister.
- Zero-shot kilitleri (`protokol-kilidi`, `varyant-kilidi`) kapandı; iki final bitti. Sıradaki kilit `fewshot-kilidi`.
- Fine-tune başına ≤ 15 CU hedefi ve aylık 100 CU toplam hedefi ölçüm bekliyor.

## Ölçülmemiş tahminler

- Yeni fine-tune reçetesinin Colab hızı, tepe VRAM ve CU projeksiyonu: her koşunun 8. kapısında ölçülecek.
