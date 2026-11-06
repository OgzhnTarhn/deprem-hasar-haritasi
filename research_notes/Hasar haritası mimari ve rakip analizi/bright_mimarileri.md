# BRIGHT üzerinde kullanılan mimariler ve gerçekte işe yarayanlar (Kasım 2026 itibarıyla)

Amaç: görülmemiş bir olayda (zero-shot) semantik hasar haritası (background, intact, damaged, destroyed) için tek mimari seçmeye yetecek kaynaklı notlar. Kod okumaları 5 Kasım 2026'da yapıldı.

**Sayı etiketleri. Bu etiketler birbirine çevrilemez ve karıştırılmamalıdır:**
- `[ZS-EVENT mIoU]`: BRIGHT makalesi, bir olay dışarıda, 13 olayla eğitim, o olayda mIoU (Tablo 10).
- `[OS-EVENT mIoU]`: aynı tablo, hedef olaydan 1 etiketli eğitim + 1 etiketli doğrulama çifti eklenmiş.
- `[STD-SET mIoU]`: standart ML ayrımı, test kümesinin tamamında (Tablo 5). `[STD-EVENT mIoU]`: standart ayrım, olay başına (Tablo 6).
- `[DFC-TEST mIoU]`: DFC25 Track 2 son test, 9 olayla eğitim, Noto-EQ-2024 + Marshall-WF-2021 testi. `[DFC-DEV mIoU]`: geliştirme fazı, eğitim olaylarından doğrulama (alan içi).
- `[CVPR-TEST mAP]`: CVPRW 2026, örnek düzeyi segm AP, iki görülmemiş 2025 olayı. `[CVPR-HOLDOUT mAP]`: aynı metrik, alan içi holdout. `[TAKIM-VAL mAP]` ve `[TAKIM-EVENT-HOLDOUT mAP]`: ikinci takımın kendi protokolleri.

## 1. IEEE GRSS DFC 2025 Track 2 kazananları ne yaptı?

### Takeaway
İki kazanan da yüksek skoru tek başına mimariden değil, bina önbilgisi + test görüntüsüyle pseudo-label ince ayarı + nesne düzeyi son işlemden aldı. 54.28 ve 50.83 `[DFC-TEST mIoU]` değerleri Turkey zero-shot 41.28 ile aynı ölçekte değildir. Damaged sınıfı birincide bile 9.75 IoU'da kaldı.

### Cited Findings
**Sonuç ve veri**
- Birinci Henryljp (Wuhan Üniv.) 0.5428, ikinci zzzbnu (Beijing Normal Üniv.) 0.5081 ([GRSS kazananlar sayfası](https://www.grss-ieee.org/community/technical-committees/winners-of-the-2025-ieee-grss-data-fusion-contest-all-weather-land-cover-and-building-damage-mapping/)). zzzbnu'nun makale özeti ise 50.83% diyor; 0.5081 ile küçük bir çelişki var ([IGARSS 2025 kaydı](https://doi.org/10.1109/IGARSS55030.2025.11242942)).
- Eğitim/doğrulama 9 olay, test görülmemiş 2 olay: Noto-Earthquake-2024 ve Marshall-Wildfire-2021 ([README_DFC25](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/README_DFC25.md)). Liste dosyalarında train 2.228, holdout 661, val 349, test yaklaşık 156 tile var (noto 79, marshall 77) ve Turkey eğitimde ([splitname/dfc25](https://github.com/ChenHongruixuan/BRIGHT/tree/master/bda_benchmark/dataset/splitname/dfc25)).

**Birinci: Building-Guided Pseudo-Label Learning** ([makale](https://arxiv.org/abs/2505.04941), [kod](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping))
- Aşama 1: yalnız pre-event optik üzerinde PVT-v2-b2/b3/b4 + FPN tarzı decoder ile ikili bina çıkarımı, CE kaybı. Üç modelin yatay flip TTA'lı ortalaması test görüntülerinde pseudo-olasılık üretir, sonra düşük belirsizlikli eğitim yapılır ([makale](https://arxiv.org/abs/2505.04941)).
- Aşama 2: UACD'nin belirsizlik modülleri çıkarılmış hâli, encoder PVT-v2-b2. En iyi 5 model + TTA ile test üzerinde 4 kanallı pseudo-etiket üretilir ([makale](https://arxiv.org/abs/2505.04941)).
- Kayıp: normalize entropi < 0.3 pikseller için sert etiketle CE, kalanlar için yumuşak etiketle KL; ağırlık 2:1 ([makale](https://arxiv.org/abs/2505.04941)).
- Son sistem: bina için PVT-v2-b2 + b3 ensemble, hasar için **tek** PVT-v2-b2, ikisinde de flip TTA. 1024×1024, 100 epoch, AdamW, cosine, 1e-4, RTX 3090 24 GB ([makale](https://arxiv.org/abs/2505.04941)).
- Füzyon kodda çok basit: optik ve SAR aynı `self.backbone` üzerinden geçer (**paylaşımlı ağırlık, modaliteye özel encoder yok**), her evrede concat + 1×1 + 3×3, dört evre yukarı örneklenip birleştirilir. Bina rehberliği her evrede `özellik * (1 + maske)` çarpımıdır ([UABCD_v2.py](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping/blob/main/ChangeDetection/network/UABCD_v2.py)).
- Makale ile kod farkları: kod Adam + poly LR kullanıyor, ilk eğitimde varsayılan 40 epoch; hasar aşamasında entropi eşiği 0.2 ve normalize edilmiyor ([train.py](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping/blob/main/ChangeDetection/train.py), [train_pseudo_label.py](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping/blob/main/ChangeDetection/train_pseudo_label.py)). Veri artırımında 0.25 oranlı mosaic var ([dataloader.py](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping/blob/main/ChangeDetection/utils/dataloader.py)).
- Pseudo-label yeniden eğitimi `./hawaii-wildfire_test/` kökünde koşuyor ([train_pseudo_label.sh](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping/blob/main/ChangeDetection/train_pseudo_label.sh)). README'ye göre Hawaii-wildfire doğru bina etiketi sağlıyor, test kısmı bina çıkarım çıktısını kullanıyor ([README](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping)).
- Son işlem: bina bağlantılı bileşeninin %20'den fazlası hasarlıysa bütün bina damaged/destroyed çoğunluğuna atanır, 50 pikselden küçükler silinir. **Beş test dosyası adı koda sabit yazılmış ve farklı birleştirme alıyor** (`save_names`) ([post_cd_process.py](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping/blob/main/ChangeDetection/post_cd_process.py)).
- Sayılar ([makale Tablo I](https://arxiv.org/abs/2505.04941)):

| Yöntem | Faz | Bg | Intact | Damaged | Destroyed | mIoU |
|---|---|---|---|---|---|---|
| CGNet | `[DFC-DEV mIoU]` | 96.722 | 79.519 | 31.000 | 70.022 | 69.316 |
| ChangeMamba | `[DFC-DEV mIoU]` | 96.737 | 80.022 | 38.181 | 71.464 | 71.601 |
| UACD | `[DFC-DEV mIoU]` | 96.756 | 80.076 | 39.575 | 73.148 | 72.389 |
| Ours (bina rehbersiz) | `[DFC-DEV mIoU]` | 96.811 | 80.700 | 44.618 | 74.105 | 74.105 |
| Ours (bina rehberli) | `[DFC-DEV mIoU]` | 96.845 | 80.958 | 45.321 | 74.637 | 74.440 |
| Ours (bina rehberli) | `[DFC-TEST mIoU]` | 97.995 | 67.304 | 9.748 | 42.086 | 54.283 |

- Not: bina rehbersiz satırın sınıf ortalaması 74.06 ediyor, tabloda 74.105 yazıyor; olası yazım hatası, makaledeki gibi aktarıldı.
- Kod lisansı: repo kökünde LICENSE dosyası yok; README lisans belirtmiyor, ağırlık indirme bağlantısı da vermiyor ([repo](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping)).

**İkinci: Heterogeneous Feature Consistency and Knowledge Integration** ([özet](https://doi.org/10.1109/IGARSS55030.2025.11242942), [kod](https://github.com/zxqyiyang/DF2025))
- Özet: Mamba tabanlı çerçeve, modaliteler arası özellik tutarlılığı, bina konumu önbilgisi; testte bina konum güçlendirme, pseudo-label ince ayarı ve nesne oylaması ([IGARSS kaydı](https://doi.org/10.1109/IGARSS55030.2025.11242942)).
- Kod: VSSM-Base (`vssm_base_224.yaml`, `vssm_base_0229_ckpt_epoch_237.pth`), 512 crop, GPU başına batch 2, 100 epoch, 1e-4, OHEM CE (eşik 0.7) ([config/STMambaBDA.py](https://github.com/zxqyiyang/DF2025/blob/master/config/STMambaBDA.py)). Model dosyalarında tek `self.encoder` tanımlı, yani bu çözüm de paylaşımlı encoder kullanıyor ([STMambaBDA_DFv4.py](https://github.com/zxqyiyang/DF2025/blob/master/STMambaBDA_DFv4.py)).
- Tutarlılık kaybı: encoder'ın 3. ve 4. evrelerinde, etiketi 1 (intact) olan bölgelerde optik ve SAR özellikleri arasında MSE ([STMambaBDA_DFv4.py](https://github.com/zxqyiyang/DF2025/blob/master/STMambaBDA_DFv4.py)). Toplam kayıp bina + derin denetimli hasar çıktıları (1, 1, 0.6, 0.4, 1) + MSE ([train_BD.py](https://github.com/zxqyiyang/DF2025/blob/master/train_BD.py)).
- Üç aşamalı eğitim: normal eğitim, "tüm test verisiyle ince ayar", "yüksek güvenli hasar sonuçlarıyla ince ayar". Komutlar 3 GPU ile yazılmış ([train_BRIGHT.sh](https://github.com/zxqyiyang/DF2025/blob/master/train_BRIGHT.sh), [second](https://github.com/zxqyiyang/DF2025/blob/master/train_BRIGHT_Fine-tune_second.sh), [third](https://github.com/zxqyiyang/DF2025/blob/master/train_BRIGHT_Fine-tune_third.sh)). Test 4 yönlü flip TTA içeriyor ([test_BRIGHT.py](https://github.com/zxqyiyang/DF2025/blob/master/test_BRIGHT.py)).
- **Kod bağlantısı çelişkisi:** GRSS raporu zzzbnu için `CarryHJR/dfc2025`, Track 1 ikincisi için `zxqyiyang/DF2025` veriyor ([rapor](https://uverma.github.io/assets/pdf/DFC25Report.pdf)). İçerik tersini gösteriyor: `CarryHJR/dfc2025` InternImage/UPerNet ve `dfc2025/track1` veri yapısı içeriyor ([README](https://github.com/CarryHJR/dfc2025)); `zxqyiyang/DF2025` ise Mamba BDA kodu. IGARSS özeti ve BRIGHT README de `zxqyiyang/DF2025` diyor ([BRIGHT README_DFC25](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/README_DFC25.md)).

**Maliyet**
- Birinci çözümün tam hâli: 3 bina modeli + yeniden eğitimler + pseudo-label turu, README'de toplam 6 bina modeli ve seçilmiş 6 hasar epoch'u adı geçiyor ([README](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping)).

### Inferences
- Ana sınıflar üzerinden: birincinin üç bina sınıfı ortalaması (67.304 + 9.748 + 42.086) / 3 = 39.71. mIoU'nun 54.28 görünmesinin büyük kısmı 98'lik background'dan gelir.
- Birincinin geliştirme fazındaki ChangeMamba üstünlüğü (74.44'e karşı 71.60) alan içidir; ChangeMamba için `[DFC-TEST mIoU]` değeri yayımlanmadığından testte üstünlüğü ölçülmüş değil. Makalenin "ChangeMamba hasarlı binaları kaçırıyor" görsel karşılaştırması sayısal kanıt sayılmamalı.
- Birincinin kodundaki sabit test dosya adları ve Hawaii-wildfire + test karışımı, sonucun bir kısmının test kümesine özgü ayar olduğunu düşündürüyor. Bu, kör bir yeni olayda aynen tekrar edilemez.
- Öğrenci için indirgenmiş sürüm mümkün: 1 PVT-v2-b2 bina + 1 PVT-v2-b2 hasar + 1 pseudo-label turu = yaklaşık 3 tam eğitim, 24 GB'de batch 2-4, 1024². Tam ensemble bütçeyi aşar. (Tahmin, ölçülmedi.)

### Gaps
- Track 2 yöntemlerini anlatan JSTARS sonuç makalesi ([IEEE 11536770](https://ieeexplore.ieee.org/document/11536770)) ve zzzbnu tam metni ([PDF](https://ieeexplore.ieee.org/iel8/11242230/11242244/11242942.pdf)) okunamadı (boş yanıt / HTTP 418). zzzbnu için sınıf başına IoU ve ablation elde edilemedi; yalnız özet ve kod okundu.
- Pseudo-label'ın `[DFC-TEST mIoU]` üzerindeki ayrık katkısı iki takımda da bulunamadı.
- DFC-TEST skorunun küme düzeyinde mi hesaplandığı açıkça yazılı bulunamadı; kaynaklar yalnız "mIoU on test set" diyor.

## 2. BRIGHT Challenge (CVPRW 2026) kazananları ne yaptı?

### Takeaway
**İstenen iki kod deposu da incelenemedi:** BightSolution 404, Damage-Aware deposu boş. Yöntemler yalnız [sonuç makalesinden](https://arxiv.org/abs/2607.22746) doğrulandı. En güçlü iki ablation sinyali: erken füzyon yerine ayrı akış ve hedef olay profiline uyum. Ama bunlar örnek düzeyi AP'dir, semantik mIoU'ya çevrilemez.

### Cited Findings
**Depo durumu (5 Kasım 2026)**
- `huang-he99/BightSolution` HTTP 404, kullanıcının herkese açık depoları EI, LRDUN, STF. Wayback'te yalnız 2026-07-29 tarihli bir `.git` URL kaydı var ve açılmıyor ([GitHub profili](https://github.com/huang-he99?tab=repositories)). Birinci yöntemi için atıf verilen [20] numaralı kaynak ayrı bir makale değil, 2025 BGPL makalesi ([sonuç makalesi](https://arxiv.org/abs/2607.22746)). "Scene-Segregated" başlıklı ayrı makale bulunamadı.
- `whf-68/Damage-Aware-SAR-Optical-Query-Learning-framework` var ama sayfası "This repository is empty" diyor ve `git ls-remote` ref döndürmüyor; son push 2026-07-21 ([depo](https://github.com/whf-68/Damage-Aware-SAR-Optical-Query-Learning-framework)).

**Format** ([README](https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/README.md), [sonuç makalesi](https://arxiv.org/abs/2607.22746))
- 3 sınıf (intact, damaged, destroyed), COCO örnek segmentasyonu, ana metrik segm AP. 157 katılımcı, 1.289 gönderim, finalde 46 takım.
- Train+val 3.029 tile / 244.976 bina; holdout 366 / 32.840; test 426 / 13.384. Test: Kaliforniya yangını 2025 (104 tile, 7.321 bina, 4.729'u destroyed) ve Jamaika kasırgası 2025 (322 tile, 6.063 bina, 3.307'si damaged).
- Baseline: 4 kanal erken füzyon Mask R-CNN ResNet-50-FPN, 100 epoch ([mask_rcnn.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/src/model/mask_rcnn.py)). `[CVPR-HOLDOUT mAP]` 0.184, `[CVPR-TEST mAP]` 0.021 (intact 0.055, damaged 0.004, destroyed 0.003).
- Bu iki baseline değeri çelişki değil, farklı ayrımlardır. README yalnız 0.1839 holdout değerini verir ([README](https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/README.md)).

**Genelleme açığı** ([sonuç makalesi](https://arxiv.org/abs/2607.22746))
- En iyi holdout takımı swift 0.513 → test 0.069. Lider takımlar %70-86 düştü, holdout-test Spearman 0.35.

**Birinci gpt_lh: 0.182 `[CVPR-TEST mAP]`, Scene-Segregated Pseudo-Label Learning** ([sonuç makalesi](https://arxiv.org/abs/2607.22746)). Ekip DFC25 birincisiyle örtüşüyor (He Huang, Jiepan Li, Wei He).
- Adım 1: hasarlı bölge içeren yamalardan çift encoder ile afet türü sınıflandırma.
- Adım 2: ilk aşamada optik F1-F5 özelliklerinden bina maskesi; ikinci aşamada ayak izi + optik özellikler ile SAR özellikleri Cross-Layer Attention'dan geçer, Mask2Former decoder'a verilir. Yangında destroyed eşiği düşürülür, damaged eşiği yükseltilir.
- Adım 3: test tahminleri + afet tutarlılık filtresi ile yüksek güvenli pseudo-label, yeniden eğitim.
- Adım 4: 3 model varyantı, çoklu artırılmış görünüm, Mask NMS. Decoder olarak Mask2Former, YOLOv10, YOLO26 denenmiş.
- Eğitim: tek RTX 4090 24 GB, 100 epoch, AdamW 1e-4, batch 4 (lokalizasyon) ve 2 (değişim), 1024×1024.
- Ablation `[CVPR-TEST mAP]`: yalnız optik 0.0409; erken füzyon 0.0542; çift aşamalı füzyon 0.1095; tam çözüm (pseudo-label + Mask NMS) 0.1815. Karışık veri baseline'ı 0.0567 olarak ayrıca anılıyor.

**İkinci oooo: 0.181 `[CVPR-TEST mAP]`, Damage-Aware SAR-Optical Query Learning** ([sonuç makalesi](https://arxiv.org/abs/2607.22746))
- İki ResNet-50-FPN akışı (RGB ve 1 kanal SAR). Her FPN seviyesinde concat + 1×1; **optik geçişli başlatma**, eğitim optikten başlayıp SAR'ı kademeli ekler.
- Residual fark bloğu: optik, SAR, işaretli ve mutlak fark token'ları; α_l küçük başlatılmış öğrenilebilir ölçek.
- RPN önerileri sorguya dönüşür, çok modlu belleğe bakar, skor/etiket/maskeyi residual düzeltir.
- `[TAKIM-VAL mAP]`: SAR-only 0.002; optik 0.093; erken füzyon 0.120; geç füzyon çift backbone 0.264; modal fark 0.265; prior-guided 0.261 (damaged AP 0.105, destroyed 0.302); önerilen 0.267 (AP50 0.464, AP75 0.283, intact 0.354, damaged 0.038, destroyed 0.409).
- `[TAKIM-EVENT-HOLDOUT mAP]`: Mask R-CNN 0.049; damaged hard-negative 0.064; önerilen 0.148, damaged yine sıfıra yakın.

**Düzenleyicilerin dersleri** ([sonuç makalesi](https://arxiv.org/abs/2607.22746))
- Öğütlenen tarif: önce güvenilir ayak izi, sonra ayrı ve kalibre edilmiş şiddet kararı.
- Testte damaged en güçlü sınıf (AP yaklaşık 0.24-0.34), intact 0.1 civarı; alan içinde tam tersi. Şiddet sıralı/kalibre ele alınmalı önerisi var.
- Birinci, Kaliforniya yangınında 0.1 mAP üstüne çıkan tek takım.
- Sınırlılık: yalnız iki olay ("indicative"), hazır co-registration, ağır mühendislik, iki kazanan da ImageNet tarzı ön eğitim kullanmış.

### Inferences
- Birincinin 0.1095 → 0.1815 sıçraması pseudo-label, eşik, ensemble ve Mask NMS'in birleşimidir; hangisinin ne kattığı ayrıştırılmamış.
- Birincinin tam hattı en az: sahne sınıflandırıcı + sahne başına lokalizasyon ve hasar + pseudo-label yeniden eğitimi + 3 varyant, yani 8-10+ tam eğitim. Bir öğrencinin "birkaç tam koşu" bütçesini açıkça aşar.
- İkincinin optik geçişli başlatma + küçük residual SAR fikri düşük maliyetli ve semantik segmentasyona taşınabilir görünüyor, fakat semantik mIoU'da test edilmiş değil. (Spekülasyon)
- Eşik ayarı örnek skoru üzerinde çalışıyor. Semantik karşılığı sınıf logit kaydırma olurdu ve yeni olayda profil bilinmediği için risklidir. (Spekülasyon)

### Gaps
- Birinci çözümün encoder adları, kayıp fonksiyonları, üç varyantın kimliği ve sahne sınıflandırıcı doğruluğu hiçbir kaynakta yok.
- İkinci çözümün kayıp ve eğitim süresi bilgileri yok; kod boş olduğu için doğrulanamadı.
- Fig. 14'teki olay başına mAP değerleri yalnız şekil olarak var, sayı olarak yok.

## 3. "Works Building on BRIGHT" listesi ve Turkey-EQ-2023 zero-shot sonuçları

### Takeaway
Listede 11 çalışma var ([BRIGHT README](https://github.com/ChenHongruixuan/BRIGHT)); seçilecek mimariyle doğrudan ilgili olan esas olarak FlowMamba. Okuyabildiğim hiçbirinde Turkey-EQ-2023 zero-shot sonucu bulunamadı. Listede olmayan iki 2026 makalesi ise kısa yol riskini gösterdiği için daha faydalı.

### Cited Findings
**README listesi** (aksi belirtilmedikçe özetler README'nin tek satırından, tam metin okunmadı)

| Çalışma | Yer | Ne | İlgi |
|---|---|---|---|
| [CDML](https://github.com/lyxdlut/CDML) | TPAMI 2026 | Az örnekli sınıflandırma için first-order cross-domain meta-learning | Düşük |
| [SARCLIP](https://github.com/CAESAR-Radi/SARCLIP) | ISPRS 2025 | SAR için çok modlu foundation model, 400k veri | Dolaylı (SAR ön eğitimi adayı, denenmiş değil) |
| [DisasterM3](https://arxiv.org/abs/2505.21089) | NeurIPS 2025 | Vision-language veri seti, 123k çift | Düşük |
| [SARLANG-1M](https://github.com/Jimmyxichen/SARLANG-1M) | TGRS 2026 | SAR-metin benchmark | Düşük |
| [IM4CD](https://www.sciencedirect.com/science/article/pii/S0924271626000559) | ISPRS 2026 | Denetimsiz değişim tespiti + görüntü eşleme | Düşük-orta |
| [FlowMamba](https://github.com/flying318/FlowMamba) | TCSVT 2026 | Hizasızlığa dayanıklı Mamba BDA | **Yüksek** |
| [DSTCD](https://github.com/Lucky-DW/DSTCD) | JAG 2026 | Kayıt özelliği aktarımıyla az örnekli optik-SAR değişim | Orta-düşük |
| DDCL-GAN | ESWA 2026 | Denetimsiz çok modlu değişim | Düşük |
| [CDPrompt](https://github.com/zhanglimeng13/CDPrompt) | TGRS 2025 | Prompt tabanlı değişim, Bright-Extended | Düşük; **depoda yalnız "kod düzenleniyor" notu var** |
| [TSIG-GAN](https://www.tandfonline.com/doi/full/10.1080/15481603.2025.2565866) | GIS&RS 2025 | Doku-yapı etkileşimli GAN | Düşük |
| [DCIBCD](https://github.com/AIBox-IMU/Computer-Vision/tree/main/DCIBCD) | ESWA 2025 | Çift yönlü girişimi azaltan çift dal | Düşük |

**FlowMamba** ([özet](https://doi.org/10.1109/TCSVT.2025.3643612), [kod](https://github.com/flying318/FlowMamba))
- Vision Mamba backbone, özellik düzeyi diferansiyel optik akış hizalama, sınıf mesafesine duyarlı kayıp; xBD, BRIGHT ve dört dağılım dışı senaryoda test edilmiş.
- Kodda akış hizalama yalnız ilk iki evrede aktif ([STMambaBDA.py](https://github.com/flying318/FlowMamba/blob/main/changedetection/models/STMambaBDA.py)). Kayıp CE + Lovász + özel terim ([train.py](https://github.com/flying318/FlowMamba/blob/main/changedetection/script/train.py)). Lisans Apache-2.0.
- README tablosu **xBD** üzerindedir: FlowMamba F1 overall 80.53, 56.83 M parametre, 386.37 GFLOPs; ChangeMamba VSSM-Small 80.14, 52.11 M, 261.61 GFLOPs; ChangeOS ResNet-50 76.91, 50.60 M, 117.53 GFLOPs ([README](https://github.com/flying318/FlowMamba)). Fark 0.39 puan ve hesap yükü %48 fazla.
- BRIGHT için bulduğum tek sayı üçüncü taraf: STSF-Net tablosunda FlowMamba 67.04, DamageFormer 66.09, STSF-Net 67.91 mIoU; ayrım belirtilmemiş ve ChangeMamba satırı yok ([arXiv 2604.05527](https://arxiv.org/html/2604.05527)). O tablonun ChangeOS satırı BRIGHT makalesiyle birebir aynı, UNet satırı ise farklı (59.22'ye karşı 64.94), yani kaynakları karışık görünüyor.

**Listede olmayan ama ilgili çalışmalar** (Semantic Scholar'daki atıf listesi tarandı)
- **Neighbourhood shortcuts**: macro-F1, BRIGHT 2.099/233/697 ayrımı. UNet 0.764, SiamCRNN 0.743, DamageFormer 0.759, SiamAttnUNet 0.786. Komşularıyla uyumsuz binalarda dengeli doğruluk yaklaşık 0.5, izole hasar tespiti %7-11. Tight-crop uzman tek başına izole hasarın yalnız %12'sini buluyor ([arXiv 2610.38162](https://arxiv.org/html/2610.38162), [kod](https://github.com/SummerTide/neighbourhood-shortcut-bda)).
- Aynı çalışmanın İslahiye sahnesi Türkiye depreminden ama **optik pre + optik post, SAR yok** ve eğitim xBD gibi görünüyor; MambaBDA-S macro-F1 0.481 → 0.517 ([arXiv 2610.38162](https://arxiv.org/html/2610.38162)). Turkey zero-shot mIoU ile karşılaştırılamaz.
- **GFM shortcut**: dondurulmuş TerraMind/Prithvi/DOFA/DINOv3 ile tile düzeyi dengeli doğruluk. BRIGHT rastgele ayrım 0.818 → leave-one-event-out 0.568. Yalnız pre 0.571, tam çift 0.568, **yalnız bina sayısı 0.653** ([arXiv 2610.13766](https://arxiv.org/html/2610.13766), [kod](https://github.com/alishibli97/gfm-shortcut-learning)).
- **ADSFNet**: ResNet34, BRIGHT'ta pseudo-siamese; focal ağırlıkları [1,1,4,2] + 0.75 Lovász; Score 71.36, F1 damaged 46.80. Cross-event yok ([Sensors 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12846113/)).
- **STSF-Net**: SAM2 + Swin, BRIGHT 67.91 mIoU, cross-event yok, kod yayın sonrasına bırakılmış ([arXiv 2604.05527](https://arxiv.org/html/2604.05527)).
- **FAF-CD**: DINOv3 ConvNeXt + VMamba decoder; BRIGHT **validation** üzerinde NeXt2Former-CD'yi geçtiğini söylüyor, özette sayı yok ([arXiv 2606.03114](https://arxiv.org/abs/2606.03114)).
- **SA-CTDM**: tohum 42 örnek düzeyi BRIGHT ayrımında 611 etiketli çiftle %34.33 F1; hizalanmış 305 çiftli düzen fine-tuning'i geçemiyor ([Remote Sensing 2026](https://doi.org/10.3390/rs18152518)).
- **Maliyet duyarlı çerçeve**: seçilmiş olaylarda çapraz doğrulama; BRIGHT deprem olaylarında "%98'e kadar" macro-F1 ([Remote Sensing 2026](https://doi.org/10.3390/rs18172920)). Seçilmiş olay ve çapraz doğrulama olduğu için zero-shot'la karşılaştırılamaz.
- **SRAC** (süperpiksel + bellek sorgusu, sayı yok), **Adversarial-Equilibrium** (OA 94.78, F1 61.08), **ChessMamba** (BRIGHT dahil, özette sayı yok): cross-event sonucu görünmüyor ([SRAC](https://doi.org/10.1109/IGARSS60577.2026.11731104), [AE](https://doi.org/10.1109/IGARSS60577.2026.11730760), [ChessMamba](https://arxiv.org/abs/2511.19882)).
- **Smart Transfer** Türkiye-Suriye depremi üzerinde ama BRIGHT değil: dondurulmuş DINOv3 SAT493M, post-event Pléiades optik; makale aynı olay içi aktarım olduğunu kendisi söylüyor ([arXiv 2604.02627](https://arxiv.org/html/2604.02627)).
- **xBD-S12** (10 m Sentinel, BRIGHT değil): karmaşık mimariler görülmemiş afetlere genellemede zorlanıyor ([arXiv 2511.05461](https://arxiv.org/abs/2511.05461)).

### Inferences
- BRIGHT'ta tile düzeyi hasar sinyalinin önemli kısmı olay ve yerleşim bağlamından gelebilir: bina sayısı probu tam çifti geçiyor. Bu, tile düzeyi ve dondurulmuş encoder sonucudur; uçtan uca segmentasyona doğrudan taşınmamalı, ama nesne/kalibrasyon denetimi için güçlü uyarıdır.
- "En iyi" iddialarının neredeyse tamamı standart ayrımdadır ve genellemeyi kanıtlamaz; [CVPRW Spearman 0.35](https://arxiv.org/abs/2607.22746) bunun nedenini gösteriyor.

### Gaps
- IEEE/ScienceDirect tam metinleri çoğunlukla açılamadı (HTTP 403/418). FlowMamba'nın BRIGHT tablosu, IM4CD, DSTCD, CDPrompt, TSIG-GAN, DCIBCD ve DDCL-GAN içerikleri doğrulanmadı. DDCL-GAN bağlantısı README'de bir üniversite proxy adresi.
- Kapalı erişimli Information Fusion makalesi ([LFG-R2P](https://doi.org/10.1016/j.inffus.2026.104873)) ve [JAG SAR depremi çalışması](https://doi.org/10.1016/j.jag.2026.105260) özetsiz geldi; Turkey sonucu içerip içermedikleri bilinmiyor.

## 4. ChangeMamba, DamageFormer, ChangeOS, SiamCRNN nasıl çalışır; decoupling iddiası ne kadar güçlü?

### Takeaway
ChangeMamba BRIGHT eğitim betiği açık ve Apache-2.0, ancak **yalnız standart ayrım checkpoint'i var; Turkey zero-shot checkpoint'i ve zero-shot betiği yok.** Makalenin "decoupling genellemeyi iyileştirir" ifadesi kontrollü deney değil, model karşılaştırması gözlemidir ve encoder farkıyla karışıktır.

### Cited Findings
**BRIGHT benchmark karşılaştırması** ([arXiv v4](https://arxiv.org/html/2501.06019v4), [ESSD PDF](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.pdf), checkpoint boyutları [Zenodo 15349462](https://zenodo.org/records/15349462) ve [14037769](https://zenodo.org/records/14037769)). Turkey zero-shot satırı ESSD PDF'inde de birebir doğrulandı.

| Model | Tür | BRIGHT encoder'ı | `[STD-SET]` | `[STD-EVENT]` Turkey | `[ZS-EVENT]` ort. | `[ZS-EVENT]` Turkey | `[OS-EVENT]` Turkey | ckpt |
|---|---|---|---|---|---|---|---|---|
| UNet | doğrudan | 5 blok, 64-1024 | 64.94 | 49.34 | 34.13 | 37.82 | 35.83 | 124 MB |
| DeepLabV3+ | doğrudan | ResNet-50, kanal yığma | 64.80 | 49.53 | 35.60 | 37.28 | 40.49 | 170 MB |
| SiamAttnUNet | doğrudan | özgün ayar | 64.26 | 48.57 | 36.42 | 36.09 | 37.49 | 244 MB |
| SiamCRNN | ayrık | pseudo-siamese ResNet-18 + 4 ConvLSTM | 65.73 | 50.69 | 37.55 | 38.32 | 40.11 | 132 MB |
| ChangeOS | ayrık | pseudo-siamese ResNet-18 | 65.98 | 52.34 | 37.62 | 38.45 | 39.51 | yok |
| DamageFormer | ayrık | pseudo-siamese Swin-T | 67.09 | 51.36 | 37.48 | 37.20 | 40.27 | 269 MB |
| ChangeMamba | ayrık | pseudo-siamese MambaBDA-Tiny | 67.63 | 52.70 | 39.38 | **41.28** | **42.80** | 240 MB |

**Deprem olayları, `[ZS-EVENT mIoU]`** ([Tablo 10](https://arxiv.org/html/2501.06019v4)):

| Olay | UNet | DeepLabV3+ | SiamAttn | SiamCRNN | ChangeOS | DamageFormer | ChangeMamba |
|---|---|---|---|---|---|---|---|
| Les Cayes-EQ-2021 | 26.21 | 35.50 | 37.22 | 39.45 | 39.36 | 40.41 | 41.35 |
| Turkey-EQ-2023 | 37.82 | 37.28 | 36.09 | 38.32 | 38.45 | 37.20 | 41.28 |
| Morocco-EQ-2023 | 38.24 | 40.48 | 40.78 | 40.89 | 41.40 | 41.89 | 42.86 |
| Noto-EQ-2024 | 31.02 | 35.57 | 35.87 | 36.04 | 39.14 | 36.37 | 39.90 |

**Ortak eğitim**
- AdamW 1e-4, weight decay 5e-3, 50.000 iterasyon, batch 16, flip/rot/crop; kayıp CE + Lovász ([arXiv v4](https://arxiv.org/html/2501.06019v4)). Komutlardaki `max_iters=800000` ile uyumlu, çünkü `max_iters` örnek sayısı olarak kullanılıyor ve liste bu boya kesiliyor: 800.000 / 16 = 50.000 adım ([make_data_loader.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/dataset/make_data_loader.py)). Crop 640 ([README](https://github.com/ChenHongruixuan/BRIGHT)).
- Zero-shot betiği hedef olayı train/val/test listelerinden çıkarıp hepsini teste koyuyor, ama **model seçimini diğer olayların alan içi val'ı üzerinde yapıyor** ([train_SiamCRNN.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/script/cross_event/zero_shot/train_SiamCRNN.py)).

**ChangeMamba BRIGHT kodu** ([train_MambaBDA_bright.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/script/train_MambaBDA_bright.py), [ChangeMambaMMBDA.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/models/ChangeMambaMMBDA.py), [tasks/bright.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/tasks/bright.py))
- İki ayrı VSSM encoder. Bina decoder'ı yalnız pre özelliklerini, hasar decoder'ı pre + post özelliklerini alır.
- BRIGHT decoder'ı her evrede yalnız `cat` füzyon kullanır; ilk üç evrede VSS bloğu var ([ChangeDecoder_BRIGHT.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/models/ChangeDecoder_BRIGHT.py)). Fabrikada başka modlar da var ([decoder_factory.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/models/decoder_factory.py)).
- Kayıp: `CE_loc + CE_clf + 0.5·Lovasz_loc + 0.75·Lovasz_clf`, sınıf ağırlıkları [1,1,1,1].
- **Önemli ayrıntı:** hasar başı 4 sınıflı (background dahil) ve eğitim döngüsündeki mIoU `argmax(output_clf)` üzerinden hesaplanıyor; bina çıktısı maske olarak uygulanmıyor ve baş adı `aux_clf`. Makale ise nihai haritanın maske işlemiyle birleştirildiğini yazıyor ([tasks/bright.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/tasks/bright.py), [arXiv v4](https://arxiv.org/html/2501.06019v4)).
- SAR tek kanal olarak okunup 3 kanala kopyalanıyor ve ImageNet normalizasyonu alıyor ([multimodal_damage_assessment.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/datasets/multimodal_damage_assessment.py)).
- Betik varsayılanları: crop 512, batch 4, `max_iters=80000` (yani 20.000 adım), 1e-4, 5e-3; **makale ayarı değil**. Eğitim sırasında her 500 iterasyonda Validation ve Test değerlendiriliyor, seçim Validation mIoU ile ([metadata.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/tasks/metadata.py)).
- Kurulum: `kernels/selective_scan` derlenmeli, `timm==0.4.12` sabit, kodun Linux dışında test edilmediği yazıyor ([README](https://github.com/ChenHongruixuan/ChangeMamba)).

**Lisans ve checkpoint**
- Kod Apache-2.0 ([LICENSE](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/LICENSE)). BRIGHT deposu da Apache-2.0 ([LICENSE](https://github.com/ChenHongruixuan/BRIGHT/blob/master/LICENSE)).
- [Zenodo 14037769](https://zenodo.org/records/14037769) v1.1 (21 Mayıs 2025, CC BY 4.0): `ckpt_ChangeMamba_bright_standard_ML_split.pth` (240.4 MB); MambaBCD Tiny/Small/Base (LEVIR-CD+, SYSU, WHU); MambaSCD Tiny/Base (SECOND); `vssm_tiny/small/base` ImageNet backbone'ları; `prediction.zip`; `SECOND.zip`. BRIGHT için yalnız standart split. README'de xBD BDA checkpoint'i yok, yalnız MambaBDA-Tiny BRIGHT var ([README](https://github.com/ChenHongruixuan/ChangeMamba)).
- [Zenodo 15349462](https://zenodo.org/records/15349462) (CC BY 4.0): DamageFormer, DeepLabV3+, SiamAttnUNet, SiamCRNN, UNet için standart + zero-shot checkpoint'ler, `turkey-earthquake` dahil. **ChangeMamba ve ChangeOS yok.**
- BRIGHT deposunda ChangeOS modeli veya Mamba betiği yok; `model/` altında DamageFormer, DeepLabV3Plus, SiamAttnUNet, SiamCRNN, UNet var ([bda_benchmark/model](https://github.com/ChenHongruixuan/BRIGHT/tree/master/bda_benchmark/model)).

**Mimariler**
- **ChangeMamba**: Visual Mamba encoder + üç uzay-zaman mekanizmalı change decoder; MambaBDA hasar için uyarlama ([arXiv 2404.03425](https://arxiv.org/abs/2404.03425)).
- **DamageFormer (DamFormer)**: Siamese transformer encoder, çok zamanlı füzyon, hafif çift görevli decoder ([arXiv 2201.10953](https://arxiv.org/abs/2201.10953)). BRIGHT sürümünde özgün MixFormer yerine Swin-T ([DamageFormer.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/model/DamageFormer.py), [arXiv v4](https://arxiv.org/html/2501.06019v4)).
- **ChangeOS**: OBIA'daki süperpikselin yerine derin nesne lokalizasyonu; pre'den bina, ikiliden hasar; nesne bazlı son işlemle etiket tutarlılığı ([GitHub](https://github.com/Z-Zheng/ChangeOS)).
- **SiamCRNN**: iki ResNet-18 + her evrede 128 boyutlu ConvLSTM + iki baş ([SiamCRNN.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/model/SiamCRNN.py)). Parametre yükü en düşüklerden, zero-shot ortalaması ChangeOS'a yakın.

**Makale ne diyor**
- Cross-event için ifade: ayrık mimariler genelde doğrudan segmentasyonu geçiyor, "which confirms that separating" lokalizasyon ve sınıflandırma genellemeyi iyileştiriyor ([arXiv v4](https://arxiv.org/html/2501.06019v4)). Standart ayrımda da benzer yorum var ([arXiv v4](https://arxiv.org/html/2501.06019v4)).
- Makale mekanizma testi vermiyor. Metinde ayrı olarak SAR sınıf dağılımlarının olaylar arası farkı ve zero-shot ortalamasının %40 altına düşmesi konuşuluyor ([arXiv v4](https://arxiv.org/html/2501.06019v4)).
- Makale "ardından DamageFormer" diyor ama zero-shot tablosunda ChangeOS 37.62 ile DamageFormer'ın 37.48'inin üstünde; `[STD-EVENT]` ortalamasında da DamageFormer 52.26, ChangeMamba 51.39 ([arXiv v4](https://arxiv.org/html/2501.06019v4)).

### Inferences
- Benim hesabım, Tablo 10'dan: `[ZS-EVENT]` ortalamada doğrudan modeller 34.13-36.42, ayrık modeller 37.48-39.38. Turkey'de ise DamageFormer (37.20) UNet'in (37.82) altında, ChangeOS ile SiamCRNN farkı 1 puanın altında.
- Deprem ortalaması (dört olay, kendi hesabım): ChangeMamba 41.35, ChangeOS 39.59, DamageFormer 38.97, SiamCRNN 38.68, SiamAttn 37.49, DeepLabV3+ 37.21, UNet 33.32.
- Decoupling, çift akış encoder ve encoder ailesiyle karışık. Doğrudan ama Siamese olan SiamAttnUNet'in 36.42'si ayrık modellere yaklaşıyor; demek ki farkın bir kısmı ayrı akıştan geliyor olabilir. Tablo 10'da tohum veya standart sapma yok, 1-3 puanlık farklar gürültü olabilir.
- ChangeMamba'nın "decoupled" hâli kodda yumuşak çoklu görev gibi çalışıyor. Kanıtlı katkı sert maskeye değil, paylaşımlı özelliklere verilen yardımcı denetime ait olabilir. Birincinin DFC'de kullandığı sert bina birleştirme ise ayrı bir tasarımdır.
- `ckpt_ChangeMamba_bright_standard_ML_split.pth` Turkey'i eğitimde gördüğü için zero-shot deneyde kullanılırsa veri sızıntısı olur. Ancak kendi olayınızı bu checkpoint ile tahmin etmek meşrudur.
- Zero-shot ChangeMamba için BRIGHT'taki olay çıkarma mantığını ChangeMamba betiğine taşımak gerekir; bu küçük bir kod işidir ama 41.28'in yeniden üretilmesi garanti değildir.

### Gaps
- Decoupled/coupled farkını aynı encoder ile izole eden BRIGHT ablation'ı bulunamadı. En yakın kanıt CVPRW ablation'larıdır ve orada da füzyonla karışıktır.
- 41.28'i üreten tam betik ve ChangeMamba zero-shot komutu yayımlanmamış; sonucun tam ayarları (Tiny dışındaki ayrıntılar) doğrulanamadı.
- Tablo 10 için sınıf başına IoU yayımlanmamış.

## 5. Görülmemiş bir depreme genelleme için en güçlü tasarım seçimleri

### Takeaway
Kanıtla en iyi desteklenen paket: ChangeMamba-Tiny tabanı + modaliteleri geç birleştirme + önce bina + bina/nesne düzeyi son işlem + başka bir depremle model seçimi. Pseudo-label ayrı raporlanacak isteğe bağlı bir adım olarak düşünülmeli, çünkü kanıtı karışık.

### Cited Findings
- **Geç/çift aşamalı füzyon**: birinci 0.0542 → 0.1095 `[CVPR-TEST mAP]`; ikinci 0.120 → 0.264 `[TAKIM-VAL mAP]` ([sonuç makalesi](https://arxiv.org/abs/2607.22746)). **Karşı kanıt:** iki DFC25 birincisi de semantik görevde paylaşımlı encoder ile kazandı ([UABCD_v2.py](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping/blob/main/ChangeDetection/network/UABCD_v2.py), [DFv4](https://github.com/zxqyiyang/DF2025/blob/master/STMambaBDA_DFv4.py)).
- **Yalnız SAR ile lokalizasyon işe yaramıyor, optik çapa gerekli**: SAR-only 0.002, optik 0.093 `[TAKIM-VAL mAP]` ([sonuç makalesi](https://arxiv.org/abs/2607.22746)).
- **Hedef görüntülere uyum**: 0.1095 → 0.1815 `[CVPR-TEST mAP]` paket hâlinde ([sonuç makalesi](https://arxiv.org/abs/2607.22746)). **Karşı kanıt:** BRIGHT'ta klasik UDA'nın ortalama kazancı en çok +0.69, PyCDA −2.41; Turkey'de DeepLabV3+ source-only 37.28 iken AdaptSeg 33.69, AdvEnt 34.54, CLAN 36.68, PyCDA 34.58, FDA 36.82 `[ZS-EVENT mIoU]` ([Tablo 11, 16](https://arxiv.org/html/2501.06019v4)). Hepsi Turkey'de source-only'nin altında.
- **Az etiket**: Turkey'de ChangeMamba 41.28 → 42.80; DeepLabV3+ 37.28 → 40.49 ama UNet 37.82 → 35.83 ([Tablo 10](https://arxiv.org/html/2501.06019v4)). DeepLabV3+ one-shot'ta SSL: MT 38.96, CCT 39.72, GCT 36.50, CPS 37.15 `[OS-EVENT mIoU]`, hepsi 40.49'un altında ([Tablo 17](https://arxiv.org/html/2501.06019v4)).
- **Model seçimi**: holdout-test Spearman 0.35; düzenleyiciler event-holdout öneriyor; ikinci takım 0.267 → 0.148 düşüşü kendi içinde görmüş ([sonuç makalesi](https://arxiv.org/abs/2607.22746)). GFM çalışması da aynı öneride ([arXiv 2610.13766](https://arxiv.org/html/2610.13766)).
- **Şiddet kalibrasyonu**: damaged kırılgan; DFC test 9.75 IoU ([makale](https://arxiv.org/abs/2505.04941)), ikinci takımda 0.038 AP ve event-holdout'ta sıfıra yakın ([sonuç makalesi](https://arxiv.org/abs/2607.22746)). Sıralı/kalibre karar öneriliyor ama semantik zero-shot'ta sınanmamış.
- **Bağlam kısa yolu**: izole hasarda BRIGHT modelleri zayıf ([arXiv 2610.38162](https://arxiv.org/html/2610.38162)), bina sayısı probu güçlü ([arXiv 2610.13766](https://arxiv.org/html/2610.13766)).
- **Turkey'e özgü**: 1.114 tile ve 135.033 bina, Maxar + Capella + Umbra, GSD 0.30-0.35, en büyük olay; tahmini proxy kayıt hatası RMSE 0.937 piksel, binada 0.901 ([Tablo 2, 15](https://arxiv.org/html/2501.06019v4)). Makale deprem olaylarının tüm modeller için daha zor olduğunu belirtiyor ([arXiv v4](https://arxiv.org/html/2501.06019v4)).

### Inferences
**Kanıt düzeyi**

| Seçim | Durum | Maliyet |
|---|---|---|
| Event-holdout ile model seçimi | Güçlü kanıt | Çok düşük |
| Önce optikten bina, sonra hasar | Orta-güçlü, karışık | Düşük (zaten çift başlı) |
| Ayrı akış + geç füzyon | Örnek düzeyinde güçlü, semantikte karışık | Düşük |
| Mamba encoder | Orta: Tablo 10'da önde, tek koşu; DFC'de PVT daha iyi | Orta (derleme riski) |
| Bina/nesne çoğunluk son işlemi | Orta: iki DFC kazananı + ChangeOS; ayrı test sayısı yok | Çok düşük |
| Test görüntüsüyle pseudo-label | Karışık; transductive | Yüksek (1 ek tam koşu) |
| Sahne sınıflandırma + olay eşikleri | Tek takım, iki olay; deprem için kanıt yok | Yüksek |
| Ensemble/TTA | Kazananlarda var, ayrı katkı yok; TTA ucuz | TTA düşük, ensemble yüksek |
| Optik akış hizalama | Turkey hatası düşük, yararı zayıf görünüyor; zero-shot sayısı yok | Yüksek, GFLOPs +%48 |
| Foundation model / SAR ön eğitimi | Spekülasyon; kazananlar kullanmadı | Değişken |
| Sıralı/mesafe duyarlı kayıp | Spekülasyon + öneri; ucuz deneme | Düşük |

**Tek mimari önerim (yargı, kanıt değil)**
- Taban: ChangeMamba-MMBDA Tiny. Gerekçe: açık kod, Apache-2.0, BRIGHT'a özel betik, depremlerin dördünde de en yüksek `[ZS-EVENT]`, kıyaslanacak yayımlanmış hedef 41.28. Yedek plan: Mamba çekirdeği Colab'da derlenmezse DamageFormer (Swin-T, torchvision, BRIGHT'ta hazır betik, 37.20 hedefi).
- Ekle: (1) ayrı bina başı çıktısını kullanan bina düzeyi çoğunluk son işlemi; (2) flip TTA; (3) denetimli olarak hafif sınıf ağırlığı denemesi. Ağır Mask2Former/ensemble yok.
- Doğrulama: Turkey'e hiç bakmadan Noto veya Morocco'yu sahte hedef yapıp iterasyon sayısını sabitle, sonra 13 olayla son eğitim. Betiklerin eğitim sırasında Test yazdırması sızıntıya açık; loglarda Turkey sütununa bakılmamalı.
- Raporlama: "katı zero-shot" ile "etiketsiz Turkey görüntüsü kullanan transductive" sonuçları ayrı verilmeli; yalnız birincisi 41.28 ile kıyaslanabilir.

**Colab A100 bütçesi (tahmin; duvar saati süresi hiçbir kaynakta yok)**
- 50.000 adım, batch 16, crop 640 = 800.000 crop. Tam koşu için kaba tahminim 3-8 saat; önce 200 adımla ölçülmeli. Her 500 adımda Test değerlendirmesi süreyi belirgin uzatır, Turkey 1.114 tile olduğu için kapatılmalı veya seyrekleştirilmeli. A100 belleği `nvidia-smi` ile kontrol edilmeli.
- Olası plan, 3-4 koşu: (a) sahte hedefle seçim, (b) 13 olayla baz, (c) gerekirse sınıf ağırlığı veya geç füzyon değişikliği, (d) isteğe bağlı tek pseudo-label turu.

| Yaklaşım | Tam koşu sayısı (tahmin) | Karmaşıklık |
|---|---|---|
| SiamCRNN / UNet (hazır checkpoint'li) | 0-1 | Düşük |
| DamageFormer | 1 | Düşük |
| ChangeMamba-Tiny + son işlem | 1-2 | Orta |
| DFC25 1. indirgenmiş | 3 | Orta-yüksek |
| zzzbnu (VSSM-Base, 3 aşama) | 3, kendi komutları 3 GPU | Yüksek |
| CVPRW 1. tam hat | 8-10+ | Çok yüksek ve kod erişilemez |
| CVPRW 2. | 1-2 ama kod yok, örnek düzeyi | Yüksek |

### Gaps
- Deprem odaklı, aynı protokolde karşılaştırmalı ablation bulunamadı; öneri farklı metriklerden çıkarılmıştır.
- Sınıf ağırlığı ve sıralı kaybın BRIGHT zero-shot'ta etkisi ölçülmemiş.
- Colab A100'de ChangeMamba verimi ve `selective_scan` derlemesi denenmedi.

## 6. Zero-shot Turkey-EQ-2023'te güncel yayımlanmış en iyi sonuç nedir?

### Takeaway
Bulabildiğim en yüksek değer hâlâ **ChangeMamba 41.28 `[ZS-EVENT mIoU]`**. Bunu aynı protokolde geçen bir yayın bulamadım; bu "yok" demek değil, "bulunamadı" demektir.

### Cited Findings
- Aynı protokolün tam sıralaması: ChangeMamba 41.28, ChangeOS 38.45, SiamCRNN 38.32, UNet 37.82, DeepLabV3+ 37.28, DamageFormer 37.20, SiamAttnUNet 36.09 ([ESSD](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.pdf), [arXiv v4](https://arxiv.org/html/2501.06019v4)). **Görev tanımındaki listede ChangeOS 38.45 eksikti; SiamCRNN'den yüksektir.**
- UDA ile en iyi FDA 36.82, DeepLabV3+ tabanlı ([Tablo 16](https://arxiv.org/html/2501.06019v4)).
- Karşılaştırılamaz yakın değerler:
  - ChangeMamba 42.80 `[OS-EVENT mIoU]`, 1 etiketli hedef çifti var ([Tablo 10](https://arxiv.org/html/2501.06019v4)).
  - ChangeMamba 52.70 `[STD-EVENT mIoU]`, Turkey eğitimde ([Tablo 6](https://arxiv.org/html/2501.06019v4)).
  - İslahiye MambaBDA-S macro-F1 0.481 → 0.517, optik-optik ([arXiv 2610.38162](https://arxiv.org/html/2610.38162)).
  - Kahramanmaraş bina düzeyi LODO mIoU 0.47-0.51, BRIGHT değil, olay içi, ikili ([arXiv 2604.02627](https://arxiv.org/html/2604.02627)).
  - DFC25 54.28, Turkey eğitimde ve test Noto + Marshall ([makale](https://arxiv.org/abs/2505.04941)).
- Arama kapsamı: Semantic Scholar'ın BRIGHT'a atıf yapan 134 kaydı özetleriyle tarandı, ayrıca arXiv ve web aramaları yapıldı ([atıf API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2501.06019/citations)).

### Inferences
- Background IoU'nun yaklaşık 96-97 olduğu varsayılırsa 41.28, üç bina sınıfında ortalama yaklaşık 23 IoU demektir: (41.28×4 − 96.5) / 3 ≈ 22.9. Tablo 10 sınıf ayrımı vermediği için bu kaba tahmindir ama beklentiyi ayarlar.
- Kendi sonucu 41.28 ile kıyaslamak için: aynı 13 olay, aynı ayrım listesi, Turkey görüntüsü kullanmama ve aynı 4 sınıf mIoU gerekir. Ayrıca Tablo 10 tek koşu göründüğünden 1-2 puanlık farklar anlamlı sayılmamalı; 2-3 tohum bütçeyi aşıyorsa bu sınırlılık yazılmalıdır.

### Gaps
- Kapalı erişimli veya özetinde sayı olmayan makaleler (LFG-R2P, SRAC, ChessMamba, FlowMamba tam metni, JSTARS sonuç makalesi) Turkey zero-shot değeri içeriyor olabilir; doğrulanamadı.
- Codabench/Codalab post-challenge tabloları Turkey zero-shot protokolünü kapsamıyor; yayımlanmamış sonuçlar bu notların dışında.
