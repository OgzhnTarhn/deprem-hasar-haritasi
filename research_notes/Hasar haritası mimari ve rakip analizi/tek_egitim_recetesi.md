# BRIGHT üzerinde tek seferlik eğitim reçetesi ve araç zinciri (Colab Pro, A100) — Kasım 2026

Not: "kendi ölçümüm" yazan değerler, 5 Kasım 2026'da Hugging Face'teki `Kullervo/BRIGHT` zip dosyalarını HTTP range isteğiyle (diske yazmadan, yalnız bellekte) okuyup TIFF etiketlerini ve piksel değerlerini sayarak elde edildi. `target.zip` tamamen okundu (4246 kare), `pre-event.zip` ve `post-event.zip` için örnekleme yapıldı. Kod Colab'da çalıştırılmadı; Colab'a dair hiçbir süre veya bellek değeri bu oturumda ölçülmedi.

## 1. GPU saati harcamadan önce hataları yakalayan ön kontroller

### Takeaway
Karpathy ve Tuning Playbook'un işe yarayan maddeleri, dokuz "geçti/kaldı" kapısından oluşan yaklaşık 1–1,5 saatlik (tahmin; 5–8 CU) bir ön kontrol hücresine çevrilebilir. En çok para kurtaran kapılar: `max_iters` birim denetimi, tam eğitim hattıyla 200 adımlık hız ve bellek ölçümü, gerçek kill sonrası devam testi, ve değerlendirme sıklığının düşürülmesi.

### Cited Findings
**Karpathy, "A Recipe for Training Neural Networks"**
- Veriyi elle incele, aykırıları sırala; modele girmeden hemen önceki tensörü görselleştir; tohumu sabitle; başta augmentation'ı kapat — [Karpathy](https://karpathy.github.io/2019/04/25/recipe/)
- Başlangıç kaybını beklenen değerle karşılaştır (softmax için `-log(1/n_classes)`), son katman biası'nı sınıf dengesizliğine göre başlat, sıfır girdi ile girdiden bağımsız taban çizgisi al — [Karpathy](https://karpathy.github.io/2019/04/25/recipe/)
- Küçük bir tek batch'i sıfıra yakın kayba kadar ezberlet; sabit bir batch üzerindeki tahminleri eğitim boyunca izle; çıktının yalnız beklenen girdilere bağlı olduğunu gradyanla doğrula — [Karpathy](https://karpathy.github.io/2019/04/25/recipe/)
- En yakın makaledeki en basit mimariyi kopyala, karmaşıklığı tek tek ekle; önceden eğitilmiş ağ kullan; eğitimi gereğinden uzun tut — [Karpathy](https://karpathy.github.io/2019/04/25/recipe/)

**Google Deep Learning Tuning Playbook**
- Batch boyutu doğrulama skoru için ayarlanmaz; donanımın izin verdiği en büyüğü seçilir, batch büyüdükçe adım süresi sabit kalmıyorsa önce darboğaz çözülür; gradient accumulation hız kazandırmaz — [Playbook](https://github.com/google-research/tuning_playbook)
- Adam'da 10'dan az deneme bütçesinde yalnız öğrenme hızı ayarlanır — [Playbook](https://github.com/google-research/tuning_playbook)
- `max_train_steps` çalışma içinde ayarlanmaz; sabit adım sayısı çalıştır, en iyi N checkpoint'i tut, sonunda seç (erken durdurma yerine). Değerlendirme zaman aralığıyla değil adım aralığıyla yapılır — [Playbook](https://github.com/google-research/tuning_playbook)
- Doğrusal veya kosinüs azalma varsayılandır — [Playbook](https://github.com/google-research/tuning_playbook)
- Kararsızlık testi: en iyi LR'nin 2 katında yaklaşık 500 adım, her adımda değerlendirme. Warmup uzunluğu `max_train_steps`'in en çok %10'u, %5'i aşarsa toplam adım artırılır — [Playbook](https://github.com/google-research/tuning_playbook)
- Gradyan kırpma: normu logla, eşiği 90. yüzdelikten başlat; güncellemelerin %50'sinden fazlası kırpılıyorsa LR düşür — [Playbook](https://github.com/google-research/tuning_playbook)
- Eğitim kaybının yükselmesi çoğunlukla hat hatasına işaret eder; geç eğitim varyansı için Polyak ortalaması önerilir — [Playbook](https://github.com/google-research/tuning_playbook)

### Inferences
Önerilen ön kontrol kapıları (eşikler benim önerim; tek deneyle kalibre edilmedi):

| # | Kapı | Geçme ölçütü | Yakaladığı hata |
|---|---|---|---|
| 0 | Ortam | GPU adında `A100`; VRAM yaklaşık 40 GB (80 GB görünürse High-RAM açık, yakım daha yüksek); bf16 destekli; boş disk ≥ 60 GB; `nproc` ≥ 8; `pip freeze` farkı raporlanır | Yanlış runtime, sessiz kütüphane güncellemesi |
| 1 | Birim | `len(loader) == hedef_iter`; ETA yazdırılır | `max_iters` tuzağı (bkz. Bölüm 2) |
| 2 | Veri bütünlüğü | Her id için 3 dosya; 1024×1024; pre 3 bant, post 1 bant, uint8; etiket yalnız {0,1,2,3}; bölünmeler arası kesişim yok; sınıf frekansı referansla ±%1 | Eksik veya bozuk dosya, sızıntı, etiket kodlaması |
| 3 | Görsel | Her olaydan 2, toplam 28 kare (pre/post/etiket bindirme) Drive'a PNG; eğitimden hemen önceki tensörlerle | Yanlış hizalama, normalizasyon hatası, flip eşleşmemesi |
| 4 | Başlangıç kaybı | clf CE ≈ ln 4 = 1,386; loc CE ≈ ln 2 = 0,693 (±%30) | Yanlış kayıp veya sınıf eşleme |
| 5 | Tek batch ezber | Augmentation kapalı, tüm sınıfları içeren 8 kare, 300–500 adım; eğitim mIoU > 0,90; AMP açık ve kapalı ayrı | Modelin öğrenemediği hatalar, bf16 kaynaklı NaN |
| 6 | Bağımlılık | SAR sıfırlanınca hasar çıktısı değişir; iki encoder'da da gradyan ≠ 0; sıfır girdi ile mIoU taban altında | Kopuk dal, ağırlığı yüklenmemiş encoder |
| 7 | Hız ve bellek | Gerçek hatla 200 adım (ilk 20 atılır) ve tam 1024 değerlendirme; tepe VRAM ≤ %85; veri bekleme payı < %15; `saat × CU/saat ≤ bakiye × 0,8` | OOM, darboğaz, bakiyenin bitmesi |
| 8 | LR kararlılığı | 2×LR (2e-4) ile 500 adım; NaN yok; p90 gradyan normu kaydedilir ve kırpma eşiği olur | Geç ortaya çıkan NaN |
| 9 | Devam (resume) | Adım 60'ta `os.kill(SIGKILL)`, yeniden "Run All"; adım sayacı, LR, optimizer durumu, EMA, RNG ve JSONL (yinelenen satır yok) tutarlı; yazma sırasında kill sonrası son sağlam checkpoint yüklenir | Colab kopmasında boşa giden saatler |
| 10 | Metrik | Kusursuz tahmin → mIoU 1,0; hep arka plan → yaklaşık 0,21 (std eğitim frekansından tahmin: 0,86/4); küçük bir karışıklık matrisi elle hesaplanır | Yanlış model seçimi |
| 11 | Prova | 2000 adım + değerlendirme (yaklaşık 30 dk, 3 CU; tahmin): kayıp düşüyor, mIoU sabit-tahmin tabanını geçiyor, hasar sınıfı IoU > 0 | Yalnız uzun koşuda görülen sorunlar |

- Kapı 10 için güçlü bir ek: BRIGHT'ın Zenodo checkpoint'i ile (`records/15349462`) resmi UNet'i test bölümünde çalıştırıp yayımlanmış 64,94 mIoU'ya yaklaşmak; veri ve metrik hattını eğitimsiz doğrular. Checkpoint'in güncel kodla yüklendiğini doğrulamadım.
- Koşu sırasında kendini durdurma: kayıp NaN ise hemen dur, son checkpoint'i sakla, `runtime.unassign()` çağır; aksi halde bozuk bir koşu CU yakmaya devam eder.

### Gaps
- Eşikler (mIoU > 0,90 gibi) bu veri için deneyle kalibre edilmedi.
- Playbook'ta NaN izolasyonu için ayrıntılı bir yordam bulamadım; yalnız kırpma, warmup ve LR düşürme önerileri var.

## 2. BRIGHT verisine özgü gerçekler

### Takeaway
Dosyalar sıkıştırılmamış, georeferanslı GeoTIFF; SAR tek bant uint8; etiketler {0,1,2,3} ve 255 yok. BRIGHT resmi kodunda `max_iters` örnek sayısıdır (800000 / 16 = 50000 iterasyon, doğrulandı), ama ChangeMamba'nın güncel kodunda iterasyon sayısıdır; aynı parametre adı 16 kat hataya yol açabilir. Hasarlı piksellerin yaklaşık %63'ü Ukrayna ve Meksika'dadır, yani split seçimi hasar sınıfını doğrudan belirler.

### Cited Findings
**`max_iters`**
- `bda_benchmark` veri kümesi listeyi `int(np.ceil(max_iters / len))` kez çoğaltıp `[0:max_iters]` ile kesiyor; yani `max_iters` **örnek** sayar — [make_data_loader.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/dataset/make_data_loader.py). Döngü `len(train_data_loader)` kadar döner — [train_UNet.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/script/standard_ML/train_UNet.py)
- Resmi komut: `--train_batch_size 16 --crop_size 640 --max_iters 800000 --learning_rate 1e-4` — [train_UNet.sh](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/sh_file/standard_ML_split/train_UNet.sh). Makale 50.000 iterasyon ve batch 16 diyor; böylece 800000/16 = 50000 doğrulanıyor — [ESSD](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.html)
- **Tuzak:** ChangeMamba'nın güncel kodunda `num_samples = max_iters * batch_size`, yani `max_iters` **iterasyon** sayar — [common.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/datasets/common.py). Depodaki BRIGHT betiği `batch 8, crop 512, max_iters 80000` kullanır (640.000 örnek) — [train_MMBDA_BRIGHT.sh](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/sh_file/train_MMBDA_BRIGHT.sh)

**Dosya biçimi (kendi ölçümüm, [Kullervo/BRIGHT](https://huggingface.co/datasets/Kullervo/BRIGHT))**

| Dosya | Örnek ad | Bant | Tip | Sıkıştırma | Ortalama boyut | Toplam açık | Zip |
|---|---|---|---|---|---|---|---|
| Optik | `pre-event/turkey-earthquake_00000001_pre_disaster.tif` | 3 (RGB) | uint8 | yok | 3,15 MB | 13,37 GB | 9,90 GB |
| SAR | `post-event/..._post_disaster.tif` | 1 | uint8 | yok | 1,05 MB | 4,48 GB | 3,29 GB |
| Etiket | `target/..._building_damage.tif` | 1 | uint8 | yok | 1,05 MB | 4,46 GB | 0,048 GB |

- Georeferans **var**: `ModelPixelScale`, `ModelTiepoint`, `GeoKeyDirectory`. Türkiye: `WGS 84 / UTM zone 37N`, 0,35 m. La Palma: UTM 28N, 0,30 m. Hawaii: `WGS 84 / Pseudo-Mercator`, 0,377 — kendi ölçümüm; [TorchGeo](https://mytorchgeo.readthedocs.io/en/latest/_modules/torchgeo/datasets/bright.html) de GeoTIFF ve 1024×1024 diyor. Yani tek bir CRS yok; olaya göre değişiyor.
- Pre ve post için `GDAL_NODATA = 0`, etikette nodata etiketi yok — kendi ölçümüm.
- Aynı üçlünün piksel ölçeği birebir eşit değil: Hawaii'de etiket 0,37685997, post 0,37686827 — kendi ölçümüm.
- 8-bit: optik reflektansa çevrilip 8-bit'e ölçeklenmiş; SAR tek bant VV veya HH genlik, arazi düzeltmeli, Capella ve Umbra; çözünürlük 0,3–1 m — [ESSD](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.html)

**Olay adı kodlaması**
- Biçim `<olay>_<8 hane>_<sonek>.tif`, örn. `turkey-earthquake_00000001_pre_disaster.tif`, klasörler `pre-event`, `post-event`, `target` — [BRIGHT README](https://github.com/ChenHongruixuan/BRIGHT/blob/master/README.md); [split listesi](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/dataset/splitname/standard_ML/train_set.txt)
- Split listelerinde sonek yok (`bata-explosion_00000014`). `la_palma-volcano` adında alt çizgi var; `split('_')[0]` "la" döndürür, `rsplit('_',1)[0]` doğru sonuç verir — kendi ölçümüm, [split listesi](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/dataset/splitname/standard_ML/train_set.txt)

**Etiket kodlaması**
- 0 arka plan, 1 sağlam, 2 hasarlı, 3 yıkılmış — [TorchGeo](https://mytorchgeo.readthedocs.io/en/latest/_modules/torchgeo/datasets/bright.html); [GRSS DFC25](https://www.grss-ieee.org/community/technical-committees/2025-ieee-grss-data-fusion-contest/). 4246 etiket karesinin tamamında başka değer yok (0 piksel) — kendi ölçümüm.
- Lokalizasyon etiketi kodda türetilir: 2 ve 3 → 1 — [make_data_loader.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/dataset/make_data_loader.py). 255 yalnız kırpma dolgusunda kullanılır; 640'lık kırpmada 1024 karede dolgu olmaz — [imutils.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/dataset/imutils.py)

**Olay bazında etiket istatistiği (kendi ölçümüm, [target.zip](https://huggingface.co/datasets/Kullervo/BRIGHT))**

| Olay | Kare | Binasız kare | Hasarlı kare | Arka plan % | Bina içinde sağlam / hasarlı / yıkılmış % | Bina Mpx |
|---|---|---|---|---|---|---|
| turkey-earthquake | 1114 | 8 | 466 | 72,2 | 93,0 / 4,1 / 2,9 | 324,9 |
| la_palma-volcano | 933 | 259 | 161 | 91,8 | 73,3 / 0,3 / 26,4 | 80,5 |
| morocco-earthquake | 567 | 345 | 25 | 97,9 | 98,0 / 1,7 / 0,3 | 12,2 |
| ukraine-conflict | 513 | 62 | 367 | 87,4 | 67,1 / 31,7 / 1,1 | 67,9 |
| mexico-hurricane | 212 | 35 | 166 | 88,1 | 14,2 / 78,7 / 7,1 | 26,5 |
| beirut-explosion | 133 | 0 | 18 | 68,6 | 95,2 / 3,5 / 1,3 | 43,8 |
| myanmar-hurricane | 126 | 46 | 46 | 94,8 | 90,5 / 8,9 / 0,6 | 6,9 |
| libya-flood | 124 | 13 | 42 | 81,3 | 64,1 / 30,2 / 5,7 | 24,4 |
| congo-volcano | 123 | 18 | 19 | 93,5 | 89,0 / 0,6 / 10,4 | 8,4 |
| bata-explosion | 107 | 15 | 39 | 94,4 | 91,2 / 5,4 / 3,4 | 6,2 |
| noto-earthquake | 79 | 32 | 40 | 94,1 | 85,6 / 4,3 / 10,1 | 4,9 |
| marshall-wildfire | 77 | 3 | 26 | 93,3 | 92,6 / 0,7 / 6,7 | 5,4 |
| haiti-earthquake | 73 | 3 | 29 | 87,0 | 98,4 / 1,5 / 0,1 | 9,9 |
| hawaii-wildfire | 65 | 0 | 53 | 88,5 | 33,6 / 10,2 / 56,2 | 7,9 |

- Genel: arka plan:bina = 6,07:1; bina içinde %82,69 / %10,68 / %6,64 — kendi ölçümüm. Makale yaklaşık 7:1 ve %82,8 / %10,7 / %6,5 diyor — [ESSD](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.html). Arka plan oranındaki 6,07 ile 7 farkını açıklayamadım.
- Eğitim piksel frekansı [bg, sağlam, hasarlı, yıkılmış]: `standard_ML` [0,8595, 0,1165, 0,0152, 0,0088]; `dfc25` [0,8488, 0,1340, 0,0067, 0,0105] — kendi ölçümüm.

**Split listeleri (kendi sayımım, [splitname](https://github.com/ChenHongruixuan/BRIGHT/tree/master/bda_benchmark/dataset/splitname))**

| Liste | Train | Val | Test | Not |
|---|---|---|---|---|
| `standard_ML` | 2918 | 450 | 878 | 14 olay; toplam 4246; olay içi rastgele 7:1:2 |
| `dfc25` | 2229 | 349 (`val_set`) + 661 (`holdout_set`) | 156 | Test: yalnız noto + marshall; Ukrayna, Meksika, Myanmar yok |
| `cvprw26` | 2332 | 697 | 366 (holdout) | [train_set_extra](https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/data/splits/train_set_extra.txt) 3183 kare ile 14 olayı içerir |

- Türkiye `standard_ML`'de 777 / 113 / 224 — kendi sayımım.
- DFC25 komutu `holdout_set`'i `val`, `val_set`'i `test` olarak geçirir; adlar yanıltıcıdır — [README_DFC25](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/README_DFC25.md)
- Yeniden dağıtım çelişkisi: README, Ukrayna, Myanmar ve Meksika optik verisinin dağıtılamadığını ve Google Earth + QGIS ile hazırlanacağını söylüyor — [README](https://github.com/ChenHongruixuan/BRIGHT/blob/master/README.md), [tutorial](https://github.com/ChenHongruixuan/BRIGHT/blob/master/tutorial.md). Ancak HF'deki `pre-event.zip` 4246 kare içeriyor ve bu olaylardan örneklediğim 12 kare görüntü benzeri veri taşıyor (ortalama 0,03–0,42; 9 tanesinde sıfır oranı < %1) — kendi ölçümüm. Görsel olarak incelemedim; bunların resmi yöntemle üretilip üretilmediğini bilmiyorum.
- HF kart lisansı `cc-by-sa-4.0` — [HF README](https://huggingface.co/datasets/Kullervo/BRIGHT/raw/main/README.md)

**Nodata**
- Bazı karelerin büyük kısmı boş: `morocco-earthquake` örneğinde post sıfır oranı 1,00 (tamamen boş), `mexico-hurricane` pre 0,82 ve post 0,82, `bata-explosion` pre 0,82 ve post 0,85, `congo-volcano` pre yaklaşık 0,6 — kendi ölçümüm (olay başına 2–4 örnek; oranlar temsili değildir).

**Resmi hat (dönüşümler ve kayıp)**
- Eğitim: `random_crop` (640) → `random_fliplr` → `random_flipud` → `random_rot`; sonra ImageNet normalizasyonu (0–255 ölçeğinde mean [123,675, 116,28, 103,53], std [58,395, 57,12, 57,375]). SAR 3 kez kopyalanır ve **aynı** ImageNet değerleriyle normalize edilir — [make_data_loader.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/dataset/make_data_loader.py), [imutils.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/dataset/imutils.py)
- `random_crop`, en fazla 10 denemede tek sınıfın payı %75'in altında olan kırpma arar; `random_rot` k∈{1,2,3} seçer, hiç özdeşlik yoktur — [imutils.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/dataset/imutils.py)
- SAR normalizasyonunda üç farklı kaynak var: ImageNet ([imutils.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/dataset/imutils.py)); 0,222836 / 0,179825 ([CVPRW26 config](https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/config/disaster.yaml)); 0,225 / 0,151 ([torchange](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/metrics/bright.py))
- `bda_benchmark` eğitim betiğinde AMP, LR planı, EMA, gradyan kırpma ve tohum yok; her 500 iterasyonda hem val hem test değerlendiriliyor (eval batch 4, test `num_workers=1`); `--resume` yalnız model ağırlığını yükler, optimizer ve adım yüklenmez — [train_UNet.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/script/standard_ML/train_UNet.py)
- `train_UNet.py`'de `F.cross_entropy(output_clf, labels_clf)` çağrısında `ignore_index` yok; ChangeMamba'da var — [train_UNet.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/script/standard_ML/train_UNet.py), [tasks/bright.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/tasks/bright.py)
- ChangeMamba'nın resume'u adımları atlarken veri yükleyiciyi yine de döndürür (`if iteration <= self.start_iteration: continue`) — [engine/base.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/engine/base.py)

### Inferences
- Ukrayna ve Meksika, hasarlı piksellerin yaklaşık %63'ünü taşır (21,5 + 20,9 = 42,4 Mpx / 67,2 Mpx). `dfc25` split'inde hasarlı sınıf frekansının %0,67'ye düşmesi bu yüzdendir. Hasar sınıfı önemliyse bu üç olayın optik verisinin durumu **eğitimden önce çözülmesi gereken kararların başındadır**.
- Türkiye tek başına bina piksellerinin yaklaşık %52'sidir (324,9 / 629,4 Mpx). Rastgele batch'lerde model Türkiye'ye yaslanır; bu proje için bu avantaj olabilir ama olay bazlı metrik zorunludur. Morocco'da 345 / 567 binasız kare, örnek bütçesinin önemli bir kısmını arka plana harcatır.
- Bütünlük denetiminde transform eşitliği birebir değil toleranslı olmalıdır (ör. piksel ölçeği 1e-3, tiepoint yarım piksel); aksi halde Hawaii gibi kareler yanlış alarm verir.
- 22,3 GB açık veri + 13,2 GB zip, yaklaşık 36 GB geçici disk ister. Dosyalar sıkıştırılmamış olduğu için okuma ucuzdur; `tifffile` veya `.npy` memmap ile yalnız kırpma bölgesi okunabilir.
- Boş kareler ImageNet normalizasyonunda sıfırdan farklı sabit bir değere dönüşür; bu nedenle maskeyi `pre==0 ve post==0` ile çıkarıp kayıpta yok saymak düşük riskli ve mantıklı bir eklemedir. Ancak resmi hatta yoktur; benchmark'a etkisi ölçülmedi.
- SAR için eğitim kümesinden hesaplanmış tek bir mean/std kullanılmalıdır; 0–1 ölçeğinde yaklaşık 0,22 / 0,15–0,18 iki bağımsız kaynakla tutarlı. Resmi benchmark'ın ImageNet değerleriyle SAR normalize etmesi çalışıyor ama bir yanlış okuma kaynağıdır (SAR ortalaması yaklaşık −1,2'ye oturur).

### Gaps
- Zenodo API yanıt vermedi; kayıt düzeyinde lisans ve dosya listesini doğrulayamadım. Bireysel kare lisansları (Maxar, NAIP vb.) incelenmedi.
- Morocco'daki tamamen boş SAR karesinin kaç tane olduğu sayılmadı; yalnız örnekleme yapıldı. Tam sayım ön kontrolde yapılmalı.
- `standard_ML`'de komşu karelerin val ve test'e düşmesinin (uzamsal özilinti) etkisi ölçülmedi.

## 3. Literatür ve kazanan ayarlardan varsayılanlar

### Takeaway
Aramasız çalışması en muhtemel reçete, BRIGHT resmi tarifidir: AdamW, LR 1e-4, WD 5e-3, kırpma 640, batch 16, 4 terimli CE + Lovász kaybı, sınıf ağırlığı yok, ayrı ağırlıklı çift encoder. Swin-Tiny tabanlı resmi DamageFormer (67,09 mIoU), Mamba tabanlı ChangeMamba'nın (67,63) yalnız 0,54 puan altındadır; bu yüzden derlenmiş çekirdek riskine girmek için tek koşuluk bir gerekçe yoktur. LR planı, bf16, EMA düşük riskli ama BRIGHT üzerinde kanıtsız eklemelerdir.

### Cited Findings
**Karşılaştırma tablosu**

| Kaynak | Encoder | Girdi | Batch | Optimizer, LR | Plan | Süre | Kayıp | Sonuç |
|---|---|---|---|---|---|---|---|---|
| [BRIGHT resmi](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/sh_file/standard_ML_split/train_UNet.sh) ([ESSD](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.html)) | UNet, DamageFormer, ChangeMamba vb. | 640 kırpma | 16 | AdamW 1e-4, WD 5e-3 | sabit | 50k iter | CE + Lovász | std mIoU: ChangeMamba 67,63; DamageFormer 67,09; UNet 64,94 |
| [ChangeMamba BRIGHT betiği](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/sh_file/train_MMBDA_BRIGHT.sh) | VMamba-Tiny | 512 | 8 | AdamW 1e-4, WD 5e-3 | sabit ([base.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/engine/base.py)) | 80k iter | `CE_loc + CE_clf + 0,5 Lov_loc + 0,75 Lov_clf`, ağırlık [1,1,1,1] ([bright.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/tasks/bright.py)) | — |
| [DFC25 1.](https://arxiv.org/html/2505.04941v1) (makale) | PVT-v2-b2/b3/b4 | 1024 tam | belirtilmemiş | AdamW 1e-4 | kosinüs | 100 epoch | CE + KL (2:1) | test mIoU 54,28; sağlam 67,3, hasarlı 9,7, yıkılmış 42,1 |
| [DFC25 1.](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping/blob/main/ChangeDetection/train.py) (kod) | PVT-v2-b2 | 1024 | 2 | Adam 1e-4 | poly | 40 epoch | CE | **makaleyle çelişiyor** |
| [DFC25 2.](https://github.com/zxqyiyang/DF2025/blob/master/changedetection/script/train_MambaBDA.py) | VMamba türevi | varsayılan 512 | varsayılan 4 | AdamW 1e-4, WD 5e-3 | yok | varsayılan 240k (eski anlam) | `CE_loc + CE_clf + 0,5 Lov_loc + 0,75 Lov_clf` | 0,5081 ([GRSS](https://www.grss-ieee.org/community/technical-committees/winners-of-the-2025-ieee-grss-data-fusion-contest-all-weather-land-cover-and-building-damage-mapping/)) |
| [CVPRW26 baseline](https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/config/disaster.yaml) (örnek segmentasyon) | Mask R-CNN | — | 16 | SGD 0,02 | adım | 100 epoch, AMP açık | — | test mAP 0,021 |
| [CVPRW26 1.](https://arxiv.org/html/2607.22746) (gpt_lh) | çift akış, Mask2Former | 1024 | 4 ve 2 | AdamW 1e-4 | belirtilmemiş | 100 epoch | belirtilmemiş | test mAP yaklaşık 0,18; RTX 4090 |
| [CVPRW26 2.](https://arxiv.org/html/2607.22746) (oooo) | çift ResNet-50-FPN | belirtilmemiş | belirtilmemiş | belirtilmemiş | belirtilmemiş | belirtilmemiş | belirtilmemiş | geç füzyon 0,264 (val) |

**Mimari kanıtı**
- DamageFormer iki ayrı `swin_t` kullanır (`encoder_1`, `encoder_2`), kayıp `CE_loc + CE_clf + 0,75 Lov_clf + 0,5 Lov_loc` — [DamageFormer.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/model/DamageFormer.py), [train_DamageFormer.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/script/standard_ML/train_DamageFormer.py). ChangeMamba da ayrı encoder kullanır — [ChangeMambaMMBDA.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/models/ChangeMambaMMBDA.py)
- CVPRW26: SAR-tek 0,002, optik-tek 0,093, erken füzyon 0,120, geç çift omurga 0,264 (val mAP). 1. takımda erken füzyon 0,0542, çift aşama 0,1095 (test) — [arXiv 2607.22746](https://arxiv.org/html/2607.22746)

**Genelleme**
- ChangeMamba std 67,63 → olay düzeyi ort. 51,39 → sıfır-atış 39,38 mIoU; DamageFormer olay düzeyi 52,26 — [ESSD](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.html)
- CVPRW26: en iyi holdout 0,513 → en iyi test 0,182; baseline 0,184 → 0,021 — [arXiv 2607.22746](https://arxiv.org/html/2607.22746)

**TTA ve çıkarım**
- Resmi çıkarım betiğinde `apply_tta_inference` var (yatay çevirme içeriyor) — [infer_DamageFormer.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/script/standard_ML/infer_DamageFormer.py); doğrulama sırasında tam 1024 kare kullanılıyor, kayan pencere yok — [train_UNet.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/script/standard_ML/train_UNet.py)
- DFC25 1. takım: yatay çevirme TTA'sı; son aşamada bağlı bileşen tabanlı birleştirme ve bina-güdümlü sözde etiket — [arXiv 2505.04941](https://arxiv.org/html/2505.04941v1), [repo](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping)
- CVPRW26 1. takım: çoklu görünüm TTA, 3 model topluluğu, Mask NMS, sahne eşikleri, sözde etiket — [arXiv 2607.22746](https://arxiv.org/html/2607.22746)

**Bulunamayanlar**
- Okuduğum DFC25 kodlarında (`train.py` ve `train_MambaBDA.py`) AMP, EMA ve gradyan kırpma yok; DFC25 1. makalesinde de geçmiyor — [arXiv 2505.04941](https://arxiv.org/html/2505.04941v1). CVPRW26 makalesi EMA ve AMP için rapor vermiyor — [arXiv 2607.22746](https://arxiv.org/html/2607.22746).

### Inferences
**Dondurulmuş tek koşu yapılandırması (önerim)**

```yaml
data:   {split: standard_ML, crop: 640, workers: 8, pin_memory: true, persistent_workers: true}
aug:    {flips: true, rot90: true, photometric: false}   # resmi set, fazlası yok
norm:   {optical: imagenet_0_1, sar: [train_mean, train_std]}  # yaklaşık 0,22 / 0,15-0,18
model:  {encoder: swin_tiny, pretrained: imagenet, shared_weights: false, decoder: fpn_256, heads: [loc2, clf4]}
optim:  {name: adamw, lr: 1.0e-4, wd: 5.0e-3, betas: [0.9, 0.999]}
sched:  {warmup_steps: 1500, type: cosine, min_lr: 1.0e-6}
train:  {batch: 16, iters: 40000, amp: bf16, ema: 0.999, clip: <ön kontrolde p90>, drop_last: true, seed: 42}
loss:   {ce_loc: 1.0, ce_clf: 1.0, lov_loc: 0.5, lov_clf: 0.75, class_weights: [1, 1, 1, 1], ignore_index: 255}
eval:   {every: 2000, split: val, windows: full_1024, select: val_miou, tta: false}
final:  {test_once: true, tta: [hflip, vflip, rot180], per_event_metrics: true}
ckpt:   {every: 1000, keep_last: 2, keep_best: 1, atomic: true, sinks: [local, drive, hf_hub]}
```

| Ayar | Karar | Dayanak | Risk |
|---|---|---|---|
| Kırpma 640, batch 16 | Sabit | Resmi | Düşük; Playbook'a göre batch değişirse LR yeniden sorulur |
| AdamW 1e-4, WD 5e-3 | Sabit | Resmi + DFC25 2. + ChangeMamba | Düşük. 4 ayrı kaynakta aynı |
| Kayıp katsayıları | Sabit | Resmi + DFC25 2. | Düşük |
| Sınıf ağırlığı | Yok | Resmi ağırlık [1,1,1,1]; Lovász dengesizliği zaten hedefler | Orta: hasarlı sınıf zayıf kalabilir (DFC25 kazananı 9,7). Ağırlık eklemek kanıtsız bir değişikliktir |
| Ayrı ağırlıklı encoder | Sabit | DamageFormer, ChangeMamba, CVPRW26 | Düşük |
| Warmup + kosinüs | Ekleme | Playbook; DFC25 1. makalesi | Düşük. Resmi sabit LR'dir, dolayısıyla BRIGHT üzerinde tam kıyas yok |
| bf16 AMP | Ekleme | Hız; DFC25'te yok, CVPRW26 baseline AMP kullanıyor | Düşük. Lovász'ta sıralama ve kayıp hesabı `autocast` dışında fp32 yapılmalı |
| EMA 0,999 | Ekleme | Playbook Polyak önerisi; BRIGHT kanıtı yok | Düşük. Hem ham hem EMA kaydedilir, doğrulamada ikisi de ölçülür |
| TTA | Yalnız son test | DFC25 1., resmi betik | Düşük. Seçimde kullanılmaz; 4× çıkarım süresi |
| Sözde etiket, topluluk | Yok | Kazananlarda var ama hedef olay görüntüsü ve çoklu koşu gerektirir | Tek koşu hedefine aykırı |
| 40k vs 50k iterasyon | 40k | ChangeMamba 640k örnek = 40k × 16; resmi 800k | Orta. Bütçe Pro'da 50k için dar (Bölüm 4). Daha çok bütçe varsa 50k |
| 1024 tam kare eğitim | Yok | DFC25 1. ve CVPRW26 1. bunu batch 2–4 ile yapar | Örnek/adım düşer; resmi 640 ile kıyas kaybolur |

- **Kayan pencere:** Evrişimli model ve Swin (torchvision) için tam 1024 çıkarım resmi yoldur; pencere şart değil. Yalnız `strict_img_size=True` kalan timm Swin'de pencere (640, adım 384 → kare başına 4 görünüm) gerekir.
- **Beklenti çıpası:** Resmi tarif + Swin-T std split'te yaklaşık 65–67 mIoU vermeli. 60'ın belirgin altında kalmak hat hatası işaretidir. Bu aralık görülmeyen olaydaki performansı göstermez; onun için 40–55 mIoU daha gerçekçi çıpadır (zero-shot 39,38, olay düzeyi 51–52, DFC25 kazananı 54,28).
- **Split kararı sahibe ait:** A) `standard_ML` (67,63 çıpasıyla kıyas, Ukrayna/Meksika optiği gerekli); B) `dfc25` (54,28 çıpası, gerçekten görülmeyen 2 olay, ama hasarlı sınıfın büyük kısmı yok); C) `train_set_extra` (en çok hasar verisi, doğrudan segmentasyon çıpası yok). Tek koşu için önerim A: çalıştığını kanıtlayan tek ölçüt benchmark kıyasıdır. Olay bazlı metrik (özellikle `turkey-earthquake`) her durumda zorunludur.

### Gaps
- CVPRW26 kazanan repoları ([BightSolution](https://github.com/huang-he99/BightSolution), [Damage-Aware](https://github.com/whf-68/Damage-Aware-SAR-Optical-Query-Learning-framework)) README'leri main ve master dalında 404 verdi, GitHub API limiti de doldu. Konfigürasyon dosyaları okunmadı, yalnız makale özeti kullanıldı.
- DFC25 ve CVPRW26 genel sonuç raporlarının tam metnini okumadım (arXiv 2607.22746 için HTML özeti, PDF boyut sınırına takıldı).
- Yayımlanmış BRIGHT sonuçlarının TTA ile mi üretildiğini doğrulayamadım.
- 2. takım DF2025'in gerçek komut satırı değerleri (sh dosyası) bulunamadı; tablodaki değerler parser varsayılanıdır.
- ConvNeXt-Tiny için BRIGHT üzerinde yayımlanmış bir sonuç bulamadım.
- EMA ve bf16'nın bu veri üzerindeki etkisine dair kanıt yok.

## 4. Colab Pro pratikleri

### Takeaway
En yetkili kaynaklara göre Pro'da oturum "genel olarak en çok 12 saat" ve tarayıcı kapalı devam (background execution) yalnız Pro+'tadır; "Pro'da da 24 saat" diyen üçüncü taraf yazılar FAQ ile çelişir. Google A100 yakım hızını yayımlamıyor; en güncel ölçüm 5,40 CU/saat (40 GB), 7,52 CU/saat (80 GB). 40k iterasyon, tahminen 9–14 saat ve 50–75 CU'dur; Pro'nun 100 CU'suna sığar ama büyük olasılıkla en az bir resume gerektirir ve tarayıcı açık kalmalıdır.

### Cited Findings
**Plan ve süre (birincil kaynak)**
- FAQ: "In general, notebooks can run for at most 12 hours"; CU biterse backend sonlandırılabilir; Pro+ yeterli CU ile 24 saate kadar kesintisiz çalıştırır; boşta kalma zaman aşımı yalnız kod çalışması bitince geçerlidir; süreler ve sınırlar sabit yayımlanmaz — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- Pricing sayfası kodu: Pro 100 CU/ay; Pro+ "ek 500, toplam 600 CU/ay"; CU'lar 90 günde dolar; Pro+'ta GPU için "priority access"; background execution yalnız Pro+: tarayıcı kapanınca 24 saate kadar devam; hücre çıktısı Drive'a tamamlanınca kaydedilir, boşta kalınca VM silinir — [Colab signup](https://colab.research.google.com/signup), [signup_binary.js](https://ssl.gstatic.com/colaboratory-static/common/7e1dce3b55c08f3ec82658ff4af82b00/signup_binary.js)
- Türkiye fiyatı (bu oturumda görüntülenen): Pro ₺165,60; Pro+ ₺826,80; Pay As You Go 100 CU ₺165,60 ve 500 CU ₺826,80 — [Colab signup](https://colab.research.google.com/signup)
- Pricing bileşeninde görünür hızlandırıcı listesi: T4, A100, G4, H100, L4, V5E1, V6E1 — [Colab signup](https://colab.research.google.com/signup)
- 22 Eylül 2026: Google AI abonelerine öncelikli erişim; Ultra'da kesintisiz background execution ve Premium GPU; mevcut Colab abonelerinde değişiklik yok, birikebilir — [Google Developers Blog](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/)
- Disk: Drive büyütmek VM diskini büyütmez, Pro büyütür; sayı verilmez — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- Drive: çok sayıda küçük okuma I/O hatası ve kota tetikler; arşivi VM'e kopyalayıp açmak önerilir; kota için bir gün beklemek gerekebilir — [Colab FAQ](https://research.google.com/colaboratory/faq.html)

**Yakım hızı (çelişkili kaynaklar)**

| Kaynak | A100 40 GB | A100 80 GB | Tarih | Güven |
|---|---|---|---|---|
| Google | yayımlanmıyor | — | — | Yetkili kaynak değer vermiyor — [FAQ](https://research.google.com/colaboratory/faq.html) |
| [McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/) | 5,40 | 7,52 | Mart 2026 | **En güncel ve şeffaf; "kaba tahmin" diye etiketli** |
| [mydealz](https://www.mydealz.de/deals/google-ai-pro-200-google-colab-compute-units-wert-21eur-monatlich-inklusive-auch-im-5eurmonat-promo-abo80gb-nvidia-a100-96gb-g4-gpu-2857205) | — | yaklaşık 6,8 | 2026 | Düşük, indirim gönderisi |
| [varlog](https://varlog-info.pages.dev/colab-credit-consumption-comparison-2024/) | 10,59 | — | Ekim 2024 | Eski; o dönem için geçerli olabilir |

- Sonuç: en yetkili değer, kendi oturumunda Colab "Resources" panelindeki orandır; ön kontrol bu değeri elle kaydetmelidir.

**VRAM ve makine**
- A100 40 GB ve 80 GB iki varyant, High-RAM ile seçiliyor; H100 için oran ölçülemedi — [McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/). Google bu eşlemeyi doğrulamıyor.
- A100: 12 vCPU, 84 GB RAM, 236 GB disk, 40 GB GPU belleği (Ekim 2024) — [varlog](https://varlog-info.pages.dev/colab-credit-consumption-comparison-2024/). 80 GB için 167 GB RAM iddiası — [mydealz](https://www.mydealz.de/deals/google-ai-pro-200-google-colab-compute-units-wert-21eur-monatlich-inklusive-auch-im-5eurmonat-promo-abo80gb-nvidia-a100-96gb-g4-gpu-2857205)

**Görüntü sürümleri**
- Ubuntu 24.04.5, Python 3.13.18 — [os-info](https://github.com/googlecolab/backend-info/blob/main/os-info.txt)
- `torch 2.14.1+cu130`, `torchvision 0.29.1+cu130`, `timm 1.0.31`, `rasterio 1.5.2`, `tifffile 2026.10.15`, `albumentations 2.0.8`, `numpy 2.4.6`, `huggingface_hub 1.33.0`, `hf-xet 1.8.0`, `wandb 0.28.1`, `tensorboard 2.21.0`, `triton 3.8.0`, `opencv-python 5.0.0.93` — [pip-freeze.gpu](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt). Dosya başlığı bunun canlı imajla birebir aynı olmayabileceğini söyler. `tmux` kurulu — [apt-list](https://github.com/googlecolab/backend-info/blob/main/apt-list.txt)

**Colab CLI**
- Resmi `google-colab-cli`: `colab new --gpu A100`, `colab run`, `colab exec`, `colab install` (uv, yoksa pip), etkinlik tabanlı canlılık; **yalnız Linux ve macOS** — [README](https://github.com/googlecolab/google-colab-cli/blob/main/README.md)

**Hugging Face indirme ve yükleme**
- `hf_transfer` terk edildi ve yok sayılıyor; `hf_xet` varsayılan; hız ayarı `HF_XET_HIGH_PERFORMANCE`; ağ diskten çok hızlıysa `HF_XET_CHUNK_CACHE_SIZE_BYTES=0` yardımcı olabilir — [HF docs](https://huggingface.co/docs/huggingface_hub/hf_transfer), [migration](https://sources.debian.org/src/huggingface-hub/1.2.2-2/docs/source/en/concepts/migration.md)

**Hız tahmini girdileri (ölçüm)**
- RTX 3090, AMP, batch 256, 224 px: `convnext_tiny` 819 img/s, `swin_tiny_patch4_window7_224` 649 img/s; `convnext_tiny_384` batch 96, 384 px'te 282 img/s — [timm benchmark, pt1.12](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-train-amp-nhwc-pt112-cu113-rtx3090.csv)

### Inferences
**İterasyon/saniye tahmini (TAHMİN; A100'de ölçülmedi)**

1. timm verisinden piksel hızı: ConvNeXt-T 819 × 224² = 41,1 Mpx/s; 384'te 282 × 384² = 41,6 Mpx/s. Yani eğitim hızı piksel başına doğrusal kalıyor. Swin-T 649 × 224² = 32,6 Mpx/s.
2. Bir adım: 16 çift × 2 dal × 640² = 13,1 Mpx. Yalnız encoder: ConvNeXt 0,32 s, Swin 0,40 s.
3. Hafif decoder (FPN 256 / U-Net) +%30–100 varsayımı: ConvNeXt 0,42–0,64 s, Swin 0,52–0,80 s. UperNet gibi ağır bir decoder bu tahmini geçersiz kılar.
4. Lovász: 6,55 M piksel üzerinde 6 sıralama (4 + 2 sınıf), +0,1–0,3 s varsayımı.
5. Toplam: yaklaşık 0,5–1,1 s/adım → **ConvNeXt-T yaklaşık 1,1–1,9 it/s, Swin-T yaklaşık 0,9–1,6 it/s**. A100 ile RTX 3090 eşdeğer varsayıldı (±%20, kaynaksız); benchmark PyTorch 1.12 ile alınmış. Planlamada **0,8–1,2 it/s** kullan.

| Bütçe | 1,2 it/s | 0,8 it/s | CU @5,40 | CU @7,52 |
|---|---|---|---|---|
| 40k iter | 9,3 sa | 13,9 sa | 50–75 | 70–105 |
| 50k iter | 11,6 sa | 17,4 sa | 63–94 | 87–131 |
| Ön kontrol + veri + son test (ek) | yaklaşık 1,5–2 sa | | 8–11 | 11–15 |

- Veri yükleme: bir örnek 5,25 MB, 16 çift/s ≈ 84 MB/s; sıkıştırılmamış TIFF ve yerel disk için rahat (tahmin). 8 worker ile darboğaz GPU'da olmalı. Bu tahminin geçerliliği kapı 7'de ölçülür.
- VRAM: timm 3090 ölçümü 24 GB kartta 256 × 224² = 12,8 Mpx/batch taşıyor; bizim encoder yükü 13,1 Mpx ile aynı büyüklükte. Decoder eklendiğinde 20–30 GB (tahmin); 40 GB yeterli, 80 GB'a gerek yok. **High-RAM anahtarını açma:** McCormick'e göre yakımı %39 artırır.
- Değerlendirme: resmi "her 500 adımda val + test" 50k'de 100 × 1328 kare demektir; saatler sürer (tahmin 3–5 sa) ve test'i model seçimine sızdırma riski yaratır. Her 2000 adımda yalnız val (450 kare, tahmin ~1–2 dk) yeterli; test bir kez, en sonda.
- **Pro mu Pro+ mı:** Sahibi "gözetimsiz" istiyorsa FAQ ve pricing bundle'a göre bunu yalnız Pro+ (₺826,80) resmen sağlar: 24 saat, tarayıcı kapalı, 600 CU. Pro (₺165,60) tarayıcı açık bir bilgisayar ister ve 12 saat beklentisiyle en az bir resume planlanmalıdır. 100 CU'luk bütçe, 40 GB + 40k için yeterli ama 50k'de veya 80 GB'de yetmeyebilir; tamponu Pay As You Go 100 CU ile (₺165,60) eklemek mümkün. Ayrıca A100 Pro'da "subject to availability"; ön kontrol A100 alamazsa kendi kendine durmalı.
- **A100 çıkmazsa yedek:** L4 (22,5 GB, 1,71 CU/saat, [McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)) için ChangeMamba'nın yayımlanmış ayarı crop 512, batch 8, 80k iter bir emsaldir. Bu otomatik olmamalı; sahip onaylamalı.
- **Canlılık:** Kod koşarken boşta sayılmadığı için eğitim hücrede foreground çalışmalı. JS tıklayıcıları gibi yöntemlere gerek yok ve bunlara güvenmek FAQ'ın öncelik mantığına aykırıdır. `tmux`/`nohup` ile arka plana atıp hücreyi bitirmek, hücre bittiğinde VM'i boşta saydırabilir (bu çalışma biçimi doğrulanmadı).
- **Colab CLI:** `colab new --gpu A100` + `colab exec` tarayıcısız bir yol vaat ediyor ama kullanıcı Windows'ta, WSL gerekir. Pro'da background execution sınırını aşıp aşmadığı belli değil; ana yol olarak önermiyorum, 1 CU ile denenebilir.
- **Checkpoint hedefi:** Önce yerel disk, sonra atomik kopya. Drive: kurulumu sıfır, ama kota ve I/O hatası riski ve 15 GB ücretsiz sınırı var; HF Hub özel repo: API ile yükleme, kota Drive'dan bağımsız, sürümlü. Önerim: **her 1000 adımda Drive'a son 2 + en iyi; her 5000 adımda ve en iyide HF Hub'a** — tek kopma noktası kalmaz. Yükleme eğitim thread'ini bloklamamalı. Boyut: yaklaşık 60–70 M parametre × (model + EMA + 2 Adam momenti) ≈ 1–1,2 GB/tam checkpoint (tahmin). `torch.save` önce `*.tmp`, sonra `os.replace`.
- **Son hücre:** `google.colab.runtime.unassign()` CU yakımını durdurur (bellekten; Colab belgesinde doğrulamadım). `try/finally` ile her çıkışta çağırılmalı ama Drive ve Hub kopyası bittikten sonra.
- **Kopma durumu:** "Run All" tekrar çalıştırıldığında hücreler idempotent olmalı (veri var mı → atla; `latest.pt` var mı → devam). Kopmadan sonra yerel disk gider; veri indirme süresi her resume'a eklenir.

### Gaps
- Colab'dan HF'ye gerçek indirme hızı için güvenilir bir ölçüm bulamadım; 13,2 GB için süre söyleyemiyorum, ön kontrol ölçmeli. HF Hub hız/kota limitlerini de kontrol etmedim.
- A100'ün Colab Pro'da ne kadar sık verildiği ölçülemedi. Üçüncü taraf yazılar çelişiyor; Google yalnız "subject to availability" diyor.
- Pro'nun gerçek kesilme süresi yayımlanmıyor. Pro için tarayıcı açıkken 12 saati aşan veya altında kesen rapor toplamadım.
- Google AI Pro'da 200 CU iddiası doğrulanmadı — [explainx](https://explainx.ai/blog/google-ai-pro-colab-a100-h100-200-compute-units-claimed-vs-verified-2026). Kullanıcının hesabının planı bilinmiyor.
- Yaklaşık 2 katlık 5,40 ile 10,59 farkının fiyat düşüşünden mi, ölçüm yönteminden mi geldiğini ayıramadım. Aramalarda "~13 CU/saat" ve "5,38 (UI'da görüldü)" değerleri de geçti ama birincil kaynağa bağlayamadım.
- A100 40 GB ile RTX 3090 eğitim hızı oranı için bu oturumda kaynak alınmadı.

## 5. Araç zinciri ve notebook yapısı

### Takeaway
Colab'ın hazır PyTorch ve timm'ini kullan; yeniden kurma ve venv açma. Kütüphane olarak yalnız `timm` (veya `torchvision` Swin) + küçük kendi paketin; TorchGeo, torchange ve SMP çalışma zamanına girmesin, referans olsun. ChangeMamba çekirdeğinin Colab'da kutudan çıktığı gibi derlenmesi olası görünmüyor (CUDA 13 ve eski pin'ler); çalıştırarak doğrulanmadı ve tek koşu için önerilmez.

### Cited Findings
**Sürümler (PyPI, 5 Kasım 2026)**
- [torch](https://pypi.org/project/torch/) 2.15.0 (Colab 2.14.1); [timm](https://pypi.org/project/timm/) 1.0.31; [segmentation-models-pytorch](https://pypi.org/project/segmentation-models-pytorch/) 0.6.0; [torchgeo](https://pypi.org/project/torchgeo/) 0.10.1 (Python ≥3.12); [rasterio](https://pypi.org/project/rasterio/) 1.5.2; [albumentations](https://pypi.org/project/albumentations/) 2.0.8 (son sürüm Mayıs 2025); [albumentationsx](https://pypi.org/project/albumentationsx/) 2.6.0; [kornia](https://pypi.org/project/kornia/) 0.9.0; [hydra-core](https://pypi.org/project/hydra-core/) 1.3.8; [wandb](https://pypi.org/project/wandb/) 0.31.0 (Colab 0.28.1); [uv](https://pypi.org/project/uv/) 0.13.7; [ruff](https://pypi.org/project/ruff/) 0.17.3; [pytest](https://pypi.org/project/pytest/) 9.2.0; [nbstripout](https://pypi.org/project/nbstripout/) 0.9.1; [huggingface-hub](https://pypi.org/project/huggingface-hub/) 2.4.0 (Colab 1.33.0); [torchange](https://pypi.org/project/torchange/) 0.0.4 (Ocak 2026)

**Kütüphane ayrıntıları**
- SMP'nin timm köprüsü NHWC çıktısını (Swin) NCHW'ye çevirir; `output_stride` bazı transformer'larda desteklenmez — [timm_universal.py](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/encoders/timm_universal.py)
- timm Swin varsayılanı `strict_img_size=True` ve `img_size=224`; `dynamic_img_size=True` her forward'da maskeleri kurar; giriş pencereye bölünmüyorsa pad edilir — [swin_transformer.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py)
- DamageFormer, `torchvision.models.swin_t` kullanır — [DamageFormer.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/model/DamageFormer.py)
- torchange: BRIGHT'ı `EVER-Z/torchange_bright` HF veri kümesi üzerinden sunar, `std_split`, `oneshot_split`, `umcd_split`, `event` ayarları var; metrikler sınıf bazlı IoU ve mIoU; 4 kanal normalizasyon ([0,485, 0,456, 0,406, 0,225], [0,229, 0,224, 0,225, 0,151]) — [data/bright.py](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/data/bright.py), [metrics/bright.py](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/metrics/bright.py). Python ≥3.11 ve `ever-beta>=0.6.1` ister; 0.6.0 sürümünün import edilemediği `pyproject` içinde not edilmiş — [pyproject.toml](https://github.com/Z-Zheng/pytorch-change-models/blob/main/pyproject.toml). Eğitim betiği `ever` trainer'ına bağlıdır — [bisup_train_bright.py](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/training/bisup_train_bright.py)
- TorchGeo `BRIGHTDFC2025`: yalnız DFC25 kurgusu; `test` split'inde maske yok; `train_setlevel`, `holdout_setlevel`, `val_setlevel` eşlemesi; `NonGeoDataset` — [TorchGeo](https://mytorchgeo.readthedocs.io/en/latest/_modules/torchgeo/datasets/bright.html)

**ChangeMamba `selective_scan` çekirdeği**
- Kurulum: `cd kernels/selective_scan && pip install .` — [README](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/README.md). `setup.py`, `torch` içe aktarır, `CUDAExtension` kullanır, `nvcc -V` çalıştırır, CUDA ≥11.6 ister ve `-gencode arch=compute_70,code=sm_70` bayrağını koşulsuz ekler — [setup.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/kernels/selective_scan/setup.py)
- BRIGHT yapılandırması `SSM_FORWARDTYPE: "v3noz"` (`noz` soneki soyulur, `v3` kalır) → `SelectiveScanOflex` → derlenmiş `selective_scan_cuda_oflex` gerekir; içe aktarma hatası yalnız sessizce geçilir, çağrıda patlar — [vssm_tiny_224_0229flex.yaml](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml), [vmamba.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/models/vmamba.py). Saf PyTorch `selective_scan_easy` yalnız çekirdek README'sinde bir örnek olarak duruyor — [kernels README](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/kernels/selective_scan/README.md)
- `requirements.txt` `timm==0.4.12`'yi sabitler — [requirements.txt](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/requirements.txt)
- CUDA 13.0, Maxwell, Pascal ve Volta için çevrimdışı derleme desteğini kaldırdı — [NVIDIA CUDA Features Archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html). A100 Ampere mimarisindedir — [McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/); Ampere'nin `sm_80` karşılığı bellekten yazıldı.
- Colab'da `nvidia-cuda-nvcc==13.0.88` pip paketi var, apt listesinde `cuda-toolkit` görünmüyor — [pip-freeze.gpu](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt), [apt-list](https://github.com/googlecolab/backend-info/blob/main/apt-list.txt)

### Inferences
**Karşılaştırma ve öneri**

| Bileşen | Öneri | Gerekçe |
|---|---|---|
| PyTorch | Colab'ın `2.14.1+cu130`'u; yeniden kurma | CUDA'lı wheel'i yeniden indirmek disk ve zaman harcar; sürüm kayması ön kontrolde raporlanır |
| `torch.compile` | Kapalı | Tek koşuda hata yüzeyi eklemeye değmez |
| Encoder | **Swin-T (`torchvision.swin_t`)** varsayılan; ConvNeXt-T yedek | Swin-T'nin BRIGHT'ta çıpası var (67,09); DamageFormer 640 ile eğitilip 1024 ile değerlendirildiği için torchvision sürümü esnek girişi fiilen gösteriyor. ConvNeXt-T en basit olandır ve yaklaşık %25 hızlıdır ama çıpası yok |
| timm | Evet, yedek encoder için | Hazır kurulu ve güncel |
| SMP | Hayır (isteğe bağlı) | Modeller tek girişli; çift encoder için yine kendi sarmalayıcı gerekir. 120–150 satır kendi FPN decoder + test daha şeffaf |
| torchange | Referans, çalışma zamanına girmesin | `ever` trainer'a bağımlılık, beta durum; HF veri kümesini doğrulayamadım |
| TorchGeo | Referans | Yalnız DFC25, Colab'a ek kurulum |
| Okuma | `tifffile` eğitimde, `rasterio` yalnız çıktı georeferansında | Dosyalar sıkıştırılmamış; rasterio transform/CRS doğru dışa aktarım için gerekli. Çıktı GeoTIFF'te girdi profili kopyalanır |
| Augmentation | **30 satır NumPy** (crop, flip, rot90) | Resmi set bu kadar; CPU darboğaz değil. Kornia GPU'su gereksiz; albumentations 2.0.8 bakımda görünmüyor, AlbumentationsX lisansını (AGPL olduğunu hatırlıyorum, doğrulamadım) kontrol etmek gerekir |
| Yapılandırma | Tek YAML + `dataclass` | Hydra'nın gücü çoklu koşu ve geçersiz kılmalardadır; tek koşuda cwd değişimi ve config kompozisyonu risk ekler. Config hash'i run_id'nin parçası olur |
| Takip | **JSONL (Drive) gerçek kaynak + W&B isteğe bağlı** | JSONL anahtar gerektirmez, resume'da tekilleştirilir. W&B telefondan izleme sağlar; hata verirse eğitim durmamalı (`mode="offline"`, sabit `id`, `resume="allow"`, anahtar Colab Secrets'ta). TensorBoard ek değer katmaz |
| Paket kurulumu | `pip install -e .` veya `uv pip install --system`; venv yok | venv, CUDA torch'u yeniden indirir. CLI, uv yoksa pip'e düşer. `uv`'nin Colab'da hazır olup olmadığını doğrulamadım |
| Kalite | `ruff`, `pytest`, `nbstripout --install` | Çıktısız notebook diff'i temiz kalır |
| CI | GitHub Actions: Python 3.13, CPU torch, veri yok | Testler sentetik 64×64 kare üretir; dosya adı çözümleme (`la_palma`), dönüşüm eşleşmesi (crop/flip/rot aynı anda üç diziye), kayıp, metrik, checkpoint round-trip, 30 adım ezber |

**ChangeMamba çekirdeği Colab'da derlenebilir mi?** Çalıştırarak doğrulamadım. Kanıttan tahminim: **yamayla mümkün olabilir, kutudan çıktığı gibi büyük olasılıkla başarısız.**

1. `nvcc`: Apt'te yok; pip'teki `nvidia-cuda-nvcc`, `nvcc` ikilisi veriyorsa `CUDA_HOME` el ile gösterilmeli. Bunun gerçekten derleme yapabildiğini doğrulamadım.
2. `compute_70` bayrağı: CUDA 13 `nvcc` ile hata vermesi beklenir; `setup.py`'den çıkarılıp yalnız `compute_80` bırakılmalı (A100 için yeterli).
3. `setup.py` `import torch` yapıyor; izole derleme ortamında torch yok. `pip install --no-build-isolation .` gerekir (çıkarım).
4. `timm==0.4.12` Python 3.13 / torch 2.14'te çalışmaz gibi görünüyor (eski `torch._six` kullanımını hatırlıyorum, doğrulamadım). Pin kaldırılmalı; `timm.models.layers` shim'i için timm 1.x denenmeli.
5. C++ ve torch başlıkları 2.14 ile uyumlu mu bilinmiyor; `triton 3.8` ile `csm_triton` da test edilmedi.
6. Her oturumda yeniden derleme gerekir (wheel Drive'a alınabilir, ama Colab imajı değişince ABI kırılabilir).

Getirisi 0,54 mIoU (67,63 vs 67,09; tek koşu farkı, gürültüden ayrılmıyor) iken bu riski tek koşu reçetesine almak mantıklı değil. Mamba istenirse ayrı bir 1 CU'luk derleme denemesi yapılmalı.

**DamageFormer'ı doğrudan kopyalarken dikkat:** `swin_t(weights=Swin_T_Weights)` enum sınıfını geçiriyor; güncel torchvision'da `Swin_T_Weights.IMAGENET1K_V1` yazılması gerekebilir (bellekten, doğrulamadım). Ayrıca `torch.load` varsayılanı `weights_only=True`; tam checkpoint için açıkça `False` geçilmeli (bellekten).

**İnce notebook yapısı (Run All ile yukarıdan aşağı, gözetimsiz)**

```text
repo/
  pyproject.toml            # torch burada sabitlenmez; Colab'ınki kullanılır
  configs/oneshot.yaml      # tek gerçek kaynak
  src/bright_oneshot/{data,transforms,model,losses,metrics,engine,ckpt,preflight,export}.py
  tests/                    # sentetik, veri gerektirmez, GitHub Actions
  notebooks/train.ipynb     # 7 hücre, mantık içermez
```

| Hücre | İş | Idempotent davranış |
|---|---|---|
| 1 | Drive bağla, `git clone --branch <tag>`, `pip install -e .`, secret'ları oku | Commit SHA kaydedilir; kirli ağaç reddedilir |
| 2 | Ortam kapısı: A100, VRAM, disk, CU girişi | Uymazsa `raise` |
| 3 | Veri indir (`hf download`, 3 zip), `/content`'e aç, bütünlük kapısı | `.done` varsa atlanır; süre loglanır |
| 4 | Ön kontrol kapıları (5–11) | Sonuç `preflight.json`; config hash'i aynı ve başarılıysa devamda atlanır |
| 5 | `python -m bright_oneshot.train --config ...` (subprocess + stdout akışı, Drive'a tee) | `latest.pt` varsa otomatik devam; bitişte `DONE` |
| 6 | Son değerlendirme: EMA vs ham, test bir kez, TTA, olay bazlı tablo, GeoTIFF örnek çıktıları | `DONE` yoksa çalışmaz; sonuçlar Hub'a |
| 7 | `runtime.unassign()` | Tüm kopyalar doğrulandıktan sonra |

- Durumu hep Drive'daki `runs/<run_id>/` tutar: `config.yaml`, `git_sha`, `pip_freeze.txt`, `metrics.jsonl`, `latest.pt`, `best.pt`, `preflight.json`.
- Resume için kaydedilecekler: model, EMA, optimizer, scheduler, `step`, en iyi skor, tüm RNG durumları. Veri tarafında "dataset'i örnek sayısıyla şişirme" kalıbı yerine sonsuz sampler kullan; adım başına tohum `seed + step` olsun ki atlama sırasında veri yeniden okunmasın (ChangeMamba'nın yaptığının tersi).
- Hata kuralı: Her kapı başarısızlığı `raise` eder; "Run All" sonraki hücreye geçmez.

**Boşa koşu nedenleri ve önlemleri (özet)**

| Hata modu | Belirti | Önlem |
|---|---|---|
| `max_iters` birim karışıklığı | 16× uzun veya kısa koşu | Kapı 1; `iters` adıyla tek parametre |
| Resmi her 500 adımda val + test | Saatler boşa | `eval.every=2000`, test sona |
| Resume yalnız ağırlık yükler | LR ve Adam sıfırlanır | Tam durum, kapı 9 |
| Oturum 12 saatte kesilir | İlerleme kaybı | 1000 adımda checkpoint; idempotent notebook |
| CU biter | Backend kapanır | Kapı 7'de bakiye kontrolü, 80 GB'ı reddet |
| Drive'da eğitim okuması | I/O hatası, kota | Veri `/content`'te; Drive yalnız checkpoint |
| Drive kopyası sırasında kill | Bozuk checkpoint | `*.tmp` + `os.replace`; son 2'yi tut |
| bf16'da Lovász | Sıralama veya NaN sorunları | Kayıp `autocast` dışında fp32 |
| SAR yanlış normalizasyonu | Düşük yakınsama | Tek mean/std, kapı 3 |
| Flip/rot üç diziye eşit uygulanmıyor | Etiket kayması, sessiz düşük skor | Birim test + görsel kapı |
| `la_palma` ayrıştırması | Yanlış olay metriği | `rsplit`, test |
| Boş nodata kareleri | Bilgisiz adımlar | Ön kontrolde liste; maskeleme veya örnekleme kararı |
| Colab imajı değişir | `opencv 5`, numpy 2.x kırılması | Ön kontrolde freeze farkı; `cv2` bağımlılığından kaçın |
| High-RAM açık | %39 fazla yakım | Ortam kapısı |

### Gaps
- ChangeMamba çekirdeğinin Colab'da derlenmesi çalıştırılarak denenmedi; yukarıdaki maddeler kod okumaya ve kısmen belleğe dayanıyor. `compute_70`'in `nvcc` 13'te hata verdiğini gösteren doğrudan bir kaynak bulamadım (yalnız mimari desteğinin kaldırıldığı doğrulandı).
- `EVER-Z/torchange_bright` HF API'de sorgulanamadı; kapsamını ve erişimini doğrulayamadım. torchange'in BRIGHT için hazır eğitim config'ini (`configs/`) GitHub API limiti yüzünden göremedim.
- Colab'da `uv`'nin hazır kurulu olup olmadığı doğrulanamadı; pip listesinde yok, CLI README'si yalnız uv'yi tercih ettiğini söylüyor.
- `backend-info` deposunun son güncelleme tarihi alınamadı; sürümler canlı imajdan sapabilir.
- AlbumentationsX lisansı, `runtime.unassign()` belgesi, `torch.load` varsayılanı ve torchvision enum davranışı bu oturumda birincil kaynaktan kontrol edilmedi.
