# Deprem Hasar Haritası: Tasarım Belgesi

**Durum:** Onay bekliyor · **Sürüm:** 3 (6 Kasım 2026; eksik belge, modüler yapı, yenilik ve iyileştirme
araştırmasıyla güncellendi) · **Süre:** 16 hafta

Bu belge bağlayıcıdır. Dayanakları [araştırma raporunda](../reports/Hasar%20haritası%20mimari%20ve%20rakip%20analizi.md).

## 1. Amaç

Afet öncesi optik ve afet sonrası SAR görüntülerinden bina hasarını haritalayan bir model.

Projenin sorusu: **Model, eğitimde hiç görmediği bir depremde ne kadar işe yarıyor ve bu sonuca ne kadar güvenebiliriz?**

Neden bu proje:

- CV'de hazır modelleri ürüne taşıma güçlü; eksik olan segmentasyon, yayımlanmış sonuçlarla dürüst kıyas ve bir
  modelin neden seçildiğini kanıta dayanarak savunmak.
- Hedef alan: yapay zekâ ve bilgisayarlı görü işleri.
- "Sayıların doğru olmasını önemsiyorum" cümlesini doğrudan sınayan bir proje.

## 2. Araştırmanın değiştirdiği üç şey

1. **"Türkiye'yi görmeden test" tek başına özgün değil.** BRIGHT makalesi, CS437 öğrenci projesi (BRIGHT optik +
   SAR, Türkiye ikili IoU 0,4953) ve Enkaz (optik + optik, Antakya) bunu yaptı.
2. **Mimari farklar Türkiye'de gürültü düzeyinde.** Modeller arası fark 1–3 puan, yayımlanan tablo tek koşudan.
   Alan uyarlaması (UDA) yöntemlerinin beşi de Türkiye'de tabanın altında kaldı.
3. **Sonucu asıl değerlendirme kararları belirliyor.** Yarışmada doğrulama ile test sıralaması arasındaki
   korelasyon 0,35; yarışma birincisinin "hasarlı" sınıfı yeni olayda 9,75 IoU'ya düştü.

Bu yüzden emek, tek ve doğru yapılmış bir ana eğitime ve **eğitim gerektirmeyen** değerlendirme işlerine gider.

## 3. Hedefler

**Hedefler**

1. Hazır Türkiye checkpoint'leriyle yayımlanmış sayıları eğitimsiz yeniden üretmek (hattın doğru olduğunun kanıtı).
2. Tek, önceden kilitlenmiş bir ana eğitim ve Türkiye'de tek seferlik test.
3. Sınıf bazlı ve bina düzeyinde metrikler, güven aralıkları.
4. Aynı binalarda üç referansın kıyası: BRIGHT optik yorum etiketi, HOT/OSM yıkık bina katmanı, öğrenmesiz radar
   tabanı (ARIA DPM). HOT/OSM de gönüllülerin hava görüntüsünden yaptığı optik yorumdur; BRIGHT etiketinden
   bağımsız değildir. Çerçeve "iki optik yorum + bir radar sinyali".
5. SAR hata analizi: Capella ve Umbra kırılımı, layover ve gölge, izole hasarlı binalar.
6. Belirsizlik ölçümü (iddia değil): risk-kapsam eğrisi, ECE.
7. Yabancı birinin tek komutla çalıştırabildiği açık kaynak depo, model kartı ve **gerçek 6 Şubat verisiyle**
   (BRIGHT'ın Türkiye kareleri) kısa demo.
8. Yenilik paketi (Y1–Y4): üç referanslı anlaşmazlık haritası, analist bütçeli önceliklendirme, etiketsiz SAR
   uyarlaması, tek fikirli varyant. Ayrıntı ve fazlar: [fazlar.md](fazlar.md).
9. Araştırmada çıkan iyileştirme ve yenilik fikirleri [fikir-havuzu.md](fikir-havuzu.md)'de bekler; ana eğitimin
   13 olaydaki sonucuna göre karar noktasında (KN1) seçilir.

**Hedef olmayanlar (16 hafta içinde)**

- Gerçek afet müdahalesinde kullanılacak sistem, kendi uydu verisi hattı
- Bina bazlı örnek segmentasyonu (`segm_AP`) ve yarışma sıralaması
- Türkiye görüntüsünü kullanan yöntemler (pseudo-label, UDA); yapılırsa ayrı "transdüktif" tabloda
- Birden çok mimari adayını eğitip karşılaştırmak
- İstanbul'da hasar demosu: deprem olmadığı için afet sonrası görüntü yok

**Anti-hedefler**

- Türkiye sonucuna bakarak herhangi bir şeyi seçmek veya ayarlamak
- "Yeni depremlere genelleniyor", "saha doğrulamalı", "Copernicus'tan iyi", "gerçek zamanlı" gibi kanıtsız iddialar
- Ağırlıklı F1 veya genel doğruluğu başlık sayısı yapmak (sağlam binalar baskın olduğu için yanıltıcı)
- Rakipleri anmadan "ilk" iddiası

## 4. Doğrulanmış gerçekler

| Konu | Değer |
|---|---|
| Veri | 14 olay, 4.246 çift, 1024×1024, uint8 GeoTIFF; optik 3 bant, SAR 1 bant; etiketler {0,1,2,3} |
| Türkiye | 1.114 kare, 135.033 bina, bina piksellerinin yaklaşık %52'si; UTM 37N, 0,35 m |
| Türkiye'de sınıflar | Bina piksellerinin %93,0'ı sağlam, %4,1'i hasarlı, %2,9'u yıkık |
| İndirme | Yaklaşık 13,2 GB zip, açılmış hâli 22,3 GB |
| Lisans | Veri CC BY-SA 4.0; BRIGHT ve ChangeMamba kodu Apache-2.0; Zenodo checkpoint'leri CC BY 4.0 |
| Hazır checkpoint | Zenodo 15349462: Türkiye zero-shot için UNet, SiamCRNN, SiamAttnUNet, DeepLabV3+, DamageFormer |

**Türkiye çıpaları `[ZS-EVENT mIoU]`:** ChangeMamba 41,28 · ChangeOS 38,45 · SiamCRNN 38,32 · UNet 37,82 ·
DeepLabV3+ 37,28 · DamageFormer 37,20 · SiamAttnUNet 36,09.

Bina sınıflarında bu kabaca ortalama 23 IoU demek. Gerçekçi beklenti 41,28 civarı, onu belirgin aşmak değil.

## 5. Model

| | Seçim | Neden |
|---|---|---|
| **Ana model** | ChangeMamba-MMBDA Tiny, ImageNet `vssm_tiny` başlangıcı | Dört depremin dördünde en iyi zero-shot sonuç; ayrı optik ve SAR kodlayıcı, ayrık bina ve hasar başlıkları |
| **Yedek** | Swin-T DamageFormer (BRIGHT deposu) | Çekirdek derlemesi gerektirmiyor; Türkiye çıpası 37,20 |
| **Kullanılmayacak** | DINOv3 (askeri kullanım yasağı), MaRS (lisans çelişkisi, Türkiye sahnelerini görmüş olabilir), standard_ML checkpoint'leri (Türkiye'yi görmüş) | Bölüm 3 anti-hedefleri |

**Yedeğe geçiş kuralı (otomatik):** ChangeMamba'nın `selective_scan` çekirdeği Colab'da en fazla 1 saatte derlenip
20 adımlık ileri-geri testten geçmezse ana eğitim DamageFormer ile yapılır. Hedef o zaman "37,20'yi açıkça geçmek,
41,28'e yaklaşmak" olur.

**Kendi mimari katkısı (karar C, Bölüm 11):** Ana eğitim ChangeMamba'yı değiştirmeden yapar; başlık sonucu ve
41,28 kıyası buradan gelir. Kendi değişikliği ikinci bir tam eğitimde, aynı protokol ve aynı reçeteyle denenir.

- Değişiklik **tek bir fikir** olur; tek koşuda birden çok değişikliğin katkısı ayrılamaz.
- Değişiklik yalnız araştırmadan ve 13 olayın doğrulama sonuçlarından türetilir. Adaylar (koherans kanalı, güçlü
  bina bulucu, sıralı kayıp, havuzlayan başlık, modalite bırakma, olay dengeli örnekleme)
  [fikir-havuzu.md](fikir-havuzu.md)'de; seçim KN1'de.
- Tasarımı ve yapılandırması, ana modelin Türkiye sonucu görülmeden önce `varyant-kilidi` etiketli commit'e girer.
- İki model de raporlanır; Türkiye sonucuna bakarak aralarından seçim yapılmaz. Fark 1–2 puanın altındaysa "eşit" yazılır.

## 6. Protokol

- **Bölme:** BRIGHT `cross_event/zero_shot` mantığı, `test_event = turkey-earthquake`. Türkiye'nin 1.114 karesinin
  tamamı teste gider. BRIGHT betiğiyle aynı biçimde eğitim = standart eğitim + standart test listesi, doğrulama =
  standart doğrulama listesi (ikisi de Türkiye hariç): **2.795 / 337 / 1.114** kare ([ADR-0001](adr/0001-zero-shot-bolme.md);
  ön kontrol kapısı 2'de doğrulanır).
- **Hesap kuralları:** [protokol.md](protokol.md) (taslak 0.1); kararların gerekçeleri [adr/](adr/README.md).
- **Seçim:** En iyi kayıt noktası 13 olayın doğrulama kümesinde EMA ağırlıklarının mIoU'suyla seçilir. Bu seçimin
  zayıf olduğu (korelasyon 0,35) raporda açıkça yazılır.
- **Başlık sonucu:** Türkiye'de tek seferlik, TTA'sız, `argmax` hasar çıktısıyla 4 sınıf olay düzeyi mIoU; resmî
  kod yolu. 41,28 ile yalnız bu kıyaslanır.
- **İkincil satırlar (önceden ilan):** flip TTA, bina düzeyi çoğunluk oylaması, bina maskeli harita. Ayarları 13 olayda
  seçilir.
- **Kilit:** Yapılandırma, eşikler ve metrik yolu `protokol-kilidi` etiketli commit'e girer; `final` adımı bu etiket
  olmadan çalışmaz.
- **Raporlama:** Sınıf başına IoU, üç bina sınıfının ortalaması, bina düzeyi makro F1, bootstrap güven aralığı.
  Komşu kareler ilişkili olduğu için bootstrap birimi mekânsal blok veya şehirdir; iki model eşleştirilmiş bootstrap
  ile kıyaslanır. "Eşit" kuralı tek bir eşik olarak protokolde kilitlenir.
- **Protokol belgesi:** Başlık mIoU'nun tam hesabı (biriktirme, arka plan, değerlendirmede nodata, tam kare veya
  kayan pencere), pikselden binaya geçiş, bootstrap ve "eşit" kuralı `docs/protokol.md`'de yazılır. Taslak Faz 0'da
  hazırlanır, `protokol-kilidi` ile Faz 4'te kilitlenir. Değerlendirmede nodata resmî kodla aynı işlenir; eğitimdeki
  yok sayma bizim reçetemizin parçasıdır.

## 7. Tek eğitim reçetesi

Dondurulmuş yapılandırma (`configs/ana.yaml`):

| Ayar | Değer | Dayanak |
|---|---|---|
| Optimizer | AdamW, lr 1e-4, weight decay 5e-3 | Resmî (kanıt) |
| Kırpma ve batch | 640 / 16; bellek yetmezse 512 / 8 | Resmî (kanıt) |
| Kayıp | CE_loc + CE_clf + 0,5·Lovász_loc + 0,75·Lovász_clf; sınıf ağırlığı yok; Lovász fp32 | Resmî (kanıt) |
| Süre | 40.000 adım (640.000 örnek) | 50.000 adım Pro bütçesine dar (yargı) |
| Zamanlama | 1.500 adım warmup, kosinüs, min lr 1e-6 | Düşük risk (yargı) |
| Hassasiyet ve EMA | bf16, EMA 0,999 | Düşük risk (yargı) |
| Gradyan kırpma | Ön kontrolde ölçülen p90 gradyan normu | Ölçülecek |
| Artırım | Flip, rot90; fotometrik yok | Resmî |
| Normalizasyon | Optik ImageNet. SAR: resmî biçim (3 kanala kopya, ImageNet istatistikleri); eğitim kümesi istatistiği fikir havuzunda (A3), Faz 4'te kısa provada ölçülür | Kanıt (resmî kod); kabul ([ADR-0004](adr/0004-sar-girdisi.md)) |
| Nodata | Optik ve SAR'ın ikisi de 0 olan pikseller kayıpta yok sayılır | Yargı |
| Doğrulama | 2.000 adımda bir, yalnız 13 olay | Türkiye eğitimde değerlendirilmez |
| Kayıt | 1.000 adımda bir Drive'a (son 2 + en iyi), 5.000 adımda bir HF Hub özel repoya; atomik yazma | Kopmaya dayanıklılık |
| Eklenmeyenler | Benek giderme, stil artırımı, UDA, SAM, MixStyle | UDA Türkiye'de tabanın altında; ötekilerin kanıtı yok |

## 8. Notebook ve ön kontrol kapıları

Notebook mantık içermez; `src/hasar` paketini çağırır ve "Run All" ile kopmadan sonra da aynı sonucu verir.

| Hücre | İş | Kopma sonrası |
|---|---|---|
| 1 | Drive'ı bağla, etiketli commit'i klonla, `pip install -e .`, sırları oku | Commit SHA kaydedilir, kirli çalışma ağacı reddedilir |
| 2 | Ortam kapısı: A100, VRAM, High-RAM kapalı, disk ≥ 60 GB | Koşul tutmazsa durur |
| 3 | Veriyi VM diskine indir, aç, bütünlüğünü denetle | `.done` varsa atlanır |
| 4 | Ön kontrol kapıları, sonuçlar `preflight.json`'a | Config hash'i aynıysa atlanır |
| 5 | Ana eğitim | `latest.pt` varsa tam durumla devam eder |
| 6 | Türkiye'de tek değerlendirme ve ikincil satırlar | Eğitim `DONE` değilse çalışmaz |
| 7 | Kopyalar doğrulanınca çalışma zamanını bırak | CU yakımı durur |

**Ön kontrol kapıları** (toplam tahminen 1–1,5 saat):

1. Hazır SiamCRNN, UNet ve DamageFormer Türkiye checkpoint'leri 38,32 / 37,82 / 37,20'ye yakın çıkmalı. Çıkmazsa
   hata veri veya metrik hattındadır ve ana eğitim başlamaz.
2. Kare sayıları ve olay listeleri beklenenle aynı; eğitim ve doğrulamada tek bir Türkiye karesi yok.
3. Veri yükleyici uzunluğu hedef adım sayısına eşit (`max_iters` tuzağı).
4. Başlangıç kayıpları beklenene yakın: hasar başlığı ln 4 ≈ 1,386, bina başlığı ln 2 ≈ 0,693.
5. Artırım kapalıyken 8 kare 300–500 adımda ezberlenmeli.
6. SAR sıfırlanınca hasar çıktısı değişmeli (SAR dalı bağlı).
7. Metrik uç durumları: kusursuz tahmin 1,0, hep arka plan tahmini = arka plan piksel oranı / 4 (doğrulama kümesinden; ~0,21 kaba tahmin).
8. 200 adımda hız, tepe VRAM ve CU projeksiyonu ölçülür; projeksiyon bütçeyi aşarsa eğitim başlamaz.
9. 2× lr ile 500 adımda p90 gradyan normu ölçülür, kırpma eşiği yapılır.
10. 60. adımda süreç öldürülür, devam testi yapılır; model, EMA, optimizer, zamanlayıcı, adım ve bütün RNG
    durumları geri gelmeli.
11. ChangeMamba çekirdek derleme kapısı (en fazla 1 saat; Bölüm 5).

## 8a. Kod mimarisi

Ayrıntılı düzen: [rapor, Bölüm 2](../reports/Eksik%20belgeler,%20modüler%20yapı%20ve%20yenilik%20değerlendirmesi.md).

- **Tek yönlü bağımlılık:** `data` ve `models` birbirini bilmez; `train` ikisini kullanır; `eval` `train`'i import
  etmez; `analysis` modeli değil tahmin dosyalarını okur.
- **Model sözleşmesi ve kayıt defteri:** Ana model, yedek ve varyant aynı arayüzü uygular; YAML'da tek satırla seçilir.
- **Türkiye tek kapıdan:** Veri katmanı Türkiye karelerini yalnız `guard` üzerinden açar; `guard` `protokol-kilidi`
  etiketini ve temiz çalışma ağacını denetler.
- **Tek çıkarım, çok analiz:** Türkiye'de model bir kez çalışır, olasılık haritaları önbelleğe yazılır; başlık
  satırı, ikincil satırlar, analizler ve demo bu önbelleği okur.
- **Sonuç şeması:** `results/**/*.json` dosyalarındaki her metrik kaydında `metric_tag` zorunlu alandır. Metrik
  olmayan sayım ve araştırma dosyaları (`results/veri/`, `results/arastirma/`) bunun yerine `kind` alanı taşır
  (`data-summary`, `research`); onlardaki sayılar metrik etiketi almaz ve 41,28 ile kıyaslanmaz.
- **Üçüncü taraf kod** `third_party/` altında sabit commit, lisans ve yama listesiyle durur.
- **Eklenmeyenler:** Hydra, eklenti sistemi, soyut Trainer hiyerarşisi.

## 9. Bütçe ve takvim

Hiçbir süre Colab'da ölçülmedi; hepsi tahmindir ve ilk Colab oturumunda ölçülür.

| Adım | Tam eğitim | Süre | CU (A100 40 GB, 5,40/saat) |
|---|---|---|---|
| Ortam, veri, hazır checkpoint'lerle yeniden üretim | 0 | 1–1,5 sa | 5–8 |
| Çekirdek derleme kapısı | 0 | ≤ 1 sa | ≤ 5,4 |
| Ön kontrol kapıları ve 2.000 adımlık prova | ~0,05 | 1–1,5 sa | 5–8 |
| **Ana eğitim, 40.000 adım** | **1** | 9–14 sa, 1–2 devam | **50–75** |
| Türkiye değerlendirmesi, TTA, belirsizlik, ONNX | 0 | 0,5–1 sa | 3–5 |
| **Toplam** | **1** | **13–19 sa** | **~65–100** |

- Colab Pro: ayda 100 CU (₺165,60), oturum en çok 12 saat, tarayıcı açık kalmalı. Pro+ ayda 600 CU (₺826,80),
  tarayıcı kapalıyken 24 saat.
- Hazırlık birinci ayda, ana eğitim ikinci ayın taze 100 CU'su ile başlar.
- High-RAM yakımı yaklaşık %39 artırır; kapalı tutulur.

Haftalık plan, fazlar ve karar noktaları: [fazlar.md](fazlar.md) (Faz 0–9).

İki tam eğitimle toplam tahminen 115–175 CU; 16 haftalık Pro'nun yaklaşık 400 CU'suna sığar, ama iki eğitim aynı aya
düşmemeli. Bugün başlanırsa bitiş yaklaşık 25 Şubat 2027.

## 10. Rakipler ve fark

| Rakip | Yaptığı | Eksiği (bu proje açısından) |
|---|---|---|
| BRIGHT makalesi | Zero-shot çıpaları | Sınıf kırılımı, güven aralığı, ikinci referans yok |
| CS437 group40 | BRIGHT optik + SAR, olay dışarıda, MC Dropout | İkili IoU, tek tohum, Fas ile Türkiye arasında büyük fark |
| Enkaz | Optik + optik, kalibrasyon, bootstrap, iyi sunum | SAR yok |
| DFC25 ve CVPRW26 kazananları | Yarışma reçeteleri | Türkiye eğitimde veya görev farklı; kodların bir kısmı erişilemez |
| Improved MambaBDA (arXiv 2603.01116) | MambaBDA'ya odak kaybı, dikkat kapıları, hizalama; "görülmemiş afette %27'ye kadar" | BRIGHT protokolü yok, metrik türü belirsiz, kod yok; 41,28 ile kıyaslanamaz |

**Fark (çoğu eğitim gerektirmez):** kilitli protokol, hazır checkpoint'lerle yeniden üretim, sınıf ve bina düzeyi
metrikler, aynı binalarda üç referansın anlaşmazlık tablosu, SAR'a özgü hata analizi, ölçülmüş belirsizlik. Rakipler
README'de adıyla anılır ve farklar tabloyla gösterilir.

## 11. Kullanıcının kararları

1. **Kendi mimari katkısı: karar C (6 Kasım 2026).** Önce ChangeMamba olduğu gibi, kendi değişikliği ikinci tam
   eğitimde (+50–75 CU). Kurallar Bölüm 5'te.
2. **Açık, 5. haftaya kadar:** Ana eğitimin yapılacağı ay için abonelik: Pro (tarayıcı açık, 1–2 devam) veya Pro+.
3. **İyileştirme fikirleri ertelendi (6 Kasım 2026).** Araştırmada çıkan bütün fikirler not edilir; ana eğitimin
   sonucuna göre KN1'de seçilir ([fikir-havuzu.md](fikir-havuzu.md)). Seçimi 13 olayın sonucu belirler, Türkiye
   sonucu belirlemez.
4. **Demo gerçek 6 Şubat verisiyle (6 Kasım 2026).** BRIGHT'ın Türkiye kareleri kullanılır. İstanbul hasar demosu
   yapılmaz: afet sonrası görüntü yok.

## 12. Claude'un verdiği kararlar (itiraz edilebilir)

- Ana metrik semantik mIoU, `segm_AP` değil.
- Seçim 13 olayın doğrulamasıyla. Les Cayes, Fas, Noto ile iç döngü, birden çok eğitim gerektirdiği için kaldırıldı.
- Türkiye görüntüsü hiçbir aşamada kullanılmaz; pseudo-label yok.
- Yedeğe geçiş otomatik ve önceden kabul edilmiş sayılır.
- Ukrayna, Myanmar ve Meksika optik görüntüleri veri sahibinin Hugging Face kopyasından kullanılır. Kaynakları
  README'de belirsiz olduğu için lisans tablosunda not düşülür.
- İkinci referans HOT/OSM; CSB saha verisi erişilebilirse eklenir.
- Kod mimarisi Bölüm 8a'daki ilkelerle kurulur. Uygulayana bırakılanlar: dosya düzeninin ayrıntısı, izleme aracı
  (JSONL zorunlu, W&B isteğe bağlı), testlerin biçimi.
- ChangeMamba'nın özel CUDA çekirdeği büyük olasılıkla ONNX'e çıkmaz (doğrulanmadı). Faz 4'te 10 dakikalık bir
  denemeyle ölçülür; dışa aktarım yolu buna göre seçilir.
- Belge fazı (Faz 0) koddan önce gelir: ADR'ler, protokol taslağı, veri ve lisans tablosu, risk kaydı, SAR notu.
