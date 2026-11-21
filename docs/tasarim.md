# Deprem Hasar Haritası: Tasarım Belgesi

**Durum:** Fine-tune kararı kullanıcı tarafından onaylandı (21 Kasım 2026, ADR-0007).
Önceki araştırma ve faz kayıtları korunur. **Süre:** 16 haftalık plan.

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

Bu yüzden emek, kontrollü iki kısa fine-tune’a ve **eğitim gerektirmeyen** değerlendirme işlerine gider.

## 3. Hedefler

**Hedefler**

1. Hazır Türkiye zero-shot checkpoint'leriyle yayımlanmış sayıları eğitimsiz yeniden üretmek (hattın doğru olduğunun kanıtı).
2. Önceden kilitlenmiş kontrol ve SAR varyantı fine-tune’u; her biri için Türkiye’de tek seferlik test.
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
   13 olay ölçümleriyle ele alınır; A4, KN1’de 21 Kasım 2026’da seçildi.

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

Bu sayılar geçmiş çalışmaların bağlamıdır. Güncel soru SAR artırımlı varyantın kontrol fine-tune’una farkıdır.

## 5. Model

Karar: [ADR-0007](adr/0007-ana-model-hazir-damageformer-fine-tune.md), kullanıcı onayı 21 Kasım 2026.

| Satır | Model | Eğitim |
|---|---|---|
| Referans | Hazır Zenodo DamageFormer zero-shot | Yok; 37,20 `[ZS-EVENT mIoU, repro]` (`results/yeniden-uretim/damageformer_test.json`) |
| Ana (kontrol) | Aynı checkpoint + fine-tune | `configs/ana.yaml`; SAR artırımı yok |
| Varyant | Aynı checkpoint + fine-tune | `configs/varyant.yaml`; yalnız SAR radyometrik artırımı eklenir |

Başlangıç Türkiye'yi görmemiş `ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth`;
Zenodo md5'i denetlenir. `standard_ML` ve Türkiye'yi görmüş olabilecek ağırlıklar dışarıda kalır.
ChangeMamba eğitimi kapsam dışıdır; 41,28 `[ZS-EVENT mIoU]` yalnız bağlamdır.

Kontrol, ek eğitimin ve proje kaybının etkisini SAR fikrinden ayırır. İki koşu aynı tohum, kırpma, örnek sırası
ve reçeteyi kullanır. A4, KN1'de yalnız 13 olayın ölçümleriyle seçildi; Türkiye sonucu ayarları belirlemez.
Her model bir kez eğitilir ve `final` ile Türkiye'de bir kez değerlendirilir. İki sonuç da raporlanır;
fark protokol §9'un “eşit” kuralıyla yazılır. Tasarımlar ana modelin Türkiye sonucundan önce kilitlenir.

## 6. Protokol

- **Bölme:** BRIGHT `cross_event/zero_shot` mantığı, `test_event = turkey-earthquake`. Türkiye'nin 1.114 karesinin
  tamamı teste gider. BRIGHT betiğiyle aynı biçimde eğitim = standart eğitim + standart test listesi, doğrulama =
  standart doğrulama listesi (ikisi de Türkiye hariç): **2.795 / 337 / 1.114** kare ([ADR-0001](adr/0001-zero-shot-bolme.md);
  ön kontrol kapısı 2'de doğrulanır).
- **Hesap kuralları:** [protokol.md](protokol.md) (sürüm 1.3); kararların gerekçeleri [adr/](adr/README.md).
- **Kayıt noktası:** Son adımın EMA ağırlıkları; doğrulama yalnız kayıt içindir. Eğitim olaylarının doğrulaması
  görülmemiş olaya genellemeyi ölçmez; seçim SAR varyantını haksız biçimde cezalandırabilir.
- **Başlık sonucu:** Türkiye'de tek seferlik, TTA'sız, `argmax` hasar çıktısıyla 4 sınıf olay düzeyi mIoU; resmî
  kod yolu. Birincil kıyas varyant − ana; ChangeMamba 41,28 `[ZS-EVENT mIoU]` yalnız bağlamdır.
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

## 7. Ortak fine-tune reçetesi

Dayanak: [onaylı tasarım §6.2](superpowers/specs/2026-11-21-sar-dayaniklilik-finetune-design.md),
`configs/ana.yaml` ve `configs/varyant.yaml`.

| Ayar | Değer | Gerekçe |
|---|---|---|
| Model | `bright_damageformer`, `init: bright_zero_shot` | Hazır checkpoint'in mimarisi |
| Başlangıç | `ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth`, md5 denetimli | Zenodo 15349462 |
| Girdi | Optik ImageNet, SAR `official` | Checkpoint'in girdi biçimi korunur |
| Kırpma / batch | 512 / 8; bellek yetmezse otomatik mikro parti | Etkin batch değişmez |
| Adım | 4.000; 32.000 örnek, yaklaşık 11,4 tur | Kısa fine-tune |
| Optimizer | AdamW, lr 2e-5, weight decay 5e-3 | Önceki 1e-4 öğrenme oranının beşte biri |
| Zamanlama | 200 adım ısınma, kosinüs, min lr 1e-7 | Ortak reçete |
| Kayıp | CE_loc + CE_clf + 0,5·Lovász_loc + 0,75·Lovász_clf; sınıf ağırlığı yok, Lovász fp32 | Kontrol, kaybın etkisini ayırır |
| Nodata | Optik ve SAR'ın ikisi de 0 ise kayıpta yok sayılır | Maske artırımdan önce hesaplanır |
| Hassasiyet / EMA | bf16 / 0,999; EMA checkpoint ağırlıklarıyla başlar | Ortak başlangıç |
| Gradyan kırpma | `train.clip: null`; geçmiş ön kontrolün 9. kapısındaki p90 normu | Protokol 1.2 kuralı korunur |
| Tohum | 42; iki koşuda aynı | Aynı örnek sırası ve geometri |
| Doğrulama | Her 500 adımda EMA, yalnız 13 olay | Yalnız kayıt; seçim yok |
| Kayıt noktası | Son adımın EMA'sı (`last_ema`) | Doğrulama genellemeyi ölçmez; SAR varyantını cezalandırmadan eşit davranır |
| Kayıt | Her 1.000 adımda Drive; son 2 ve seçilen 1, Hub aralığı 5.000 | Atomik kayıt ve devam |
| Ortak artırım | Flip, rot90; optik fotometrik artırım yok | Geometri aynı |
| Varyant farkı | `sar_p: 0.8`, `sar_gamma: [0.67, 1.5]`, `sar_gain: [0.657, 1.657]` | Yalnız eğitimde, geometriden sonra ve normalizasyondan önce |

Gama ve kazanç log-uniform çekilir. Kazanç sınırları `results/arastirma/sar_olcek.json` içindeki
41,3 / 62,9 ve 104,2 / 62,9 oranlarından gelir. Geçerli SAR pikselleri [1, 255]'e kırpılır, nodata 0 kalır.
Kontrolde `sar_p: 0.0`, gama ve kazanç [1.0, 1.0]; değerlendirmede iki koşuda da SAR artırımı yoktur.

## 8. Notebook ve ön kontrol kapıları

Notebook mantık içermez; `src/hasar` paketini çağırır ve "Run All" ile kopmadan sonra da aynı sonucu verir.

| Hücre | İş | Kopma sonrası |
|---|---|---|
| 1 | Drive'ı bağla, depoyu klonla, aşamayı etiketlerden seç (`protokol-kilidi` yoksa yalnız ön kontrol; `varyant-kilidi` varsa final de), `pip install -e .`, sırları oku | Commit SHA kaydedilir, kirli çalışma ağacı reddedilir |
| 2 | Ortam kapısı: A100 (Colab vermezse L4 22 GB), High-RAM kapalı, disk ≥ 60 GB | Koşul tutmazsa durur |
| 3 | Veri ve `hasar agirlik --config` ile iki koşunun md5 denetimli başlangıcı | `.done` varsa indirme atlanır |
| 4 | Ortak yeniden üretim denetimi ve `run_one` akışının tanımı | Ön kontrol kaydı yapılandırmayla eşleşir |
| 5 | Ana: ön kontrol → fine-tune → `sicaklik` → `final` (etiketlere göre) | `last.pt` ile devam; `DONE` ve `turkiye.json` varsa atlar |
| 6 | Varyant: aynı sıra, ayrı kayıt ve sonuçlar | Aynı devam kuralı |
| 7 | Kopyalar doğrulanınca çalışma zamanını bırak | CU yakımı durur |

**Ön kontrol kapıları** (toplam tahminen 1–1,5 saat):

1. Hazır SiamCRNN, UNet ve DamageFormer Türkiye checkpoint’leri sırasıyla 38,32 `[ZS-EVENT mIoU]`,
   37,82 `[ZS-EVENT mIoU]` ve 37,20 `[ZS-EVENT mIoU]` değerlerine yakın çıkmalı. Çıkmazsa
   hata veri veya metrik hattındadır ve ana eğitim başlamaz.
2. Kare sayıları ve olay listeleri beklenenle aynı; eğitim ve doğrulamada tek bir Türkiye karesi yok.
3. Veri yükleyici uzunluğu hedef adım sayısına eşit (`max_iters` tuzağı).
4. Yükleme denetimi: 74,48 `[VAL13 mIoU, repro]` değerine ±0,3 yakınlık; fp32, TTA yok.
5. Artırım kapalıyken 8 kare 300–500 adımda ezberlenmeli.
6. SAR sıfırlanınca hasar çıktısı değişmeli (SAR dalı bağlı).
7. Metrik uç durumları: kusursuz tahmin 1,0, hep arka plan tahmini = arka plan piksel oranı / 4 (doğrulama kümesinden; ~0,21 kaba tahmin).
8. 200 adımda hız, tepe VRAM ve CU projeksiyonu ölçülür; projeksiyon bütçeyi aşarsa eğitim başlamaz.
9. 2× lr ile 500 adımda p90 gradyan normu ölçülür, kırpma eşiği yapılır (eğitim bu değeri `preflight.json`'dan kendisi okur).
10. 60. adımda süreç öldürülür, devam testi yapılır; model, EMA, optimizer, zamanlayıcı, adım ve bütün RNG
    durumları geri gelmeli. CPU'da birebir; GPU'da bazı geri yayılım çekirdekleri bit düzeyinde belirlenimci
    olmadığından iki kesintisiz koşu bile ayrışır (yerelde ölçüldü), bu yüzden parametreler `atol` 1e-3 ve
    devamdan sonraki kayıplar %1 içinde karşılaştırılır; BatchNorm istatistikleri ve onlara bağlı doğrulama
    mIoU'su yalnız raporlanır.
11. DamageFormer için özel çekirdek gerekmez; kapı “uygulanmaz” kaydıyla geçer.

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

ChangeMamba'nın Colab L4 ön kontrolünde kapı 8 projeksiyonu 164,6 CU çıktı; 75 CU sınırını aştı.
Bu yüzden sıfırdan eğitim yerine iki kısa fine-tune seçildi.

| İş | Hedef / tahmin | Ölçüm |
|---|---|---|
| Kontrol fine-tune | Yaklaşık 1–2 saat, ≤ 15 CU | Ölçülmedi, kapı 8 ölçer |
| SAR varyantı fine-tune | Yaklaşık 1–2 saat, ≤ 15 CU | Ölçülmedi, kapı 8 ölçer |
| Ön kontrol ve değerlendirme | Aylık 100 CU toplam hedefi içinde | Fine-tune projeksiyonuna ön kontrol dahil değildir |

≤ 15 CU bir ölçüm sonucu değildir. Kapı 8 her yapılandırma için çalışan GPU'da hız ve eğitim maliyeti
projeksiyonunu ölçer; bütçe aşılırsa eğitim başlamaz. Proje, iki koşu ve ön kontrollerle Colab Pro'nun
aylık 100 CU'su içinde kalmayı hedefler. A100 veya L4 kullanılır; High-RAM kapalıdır.
Kopmada aynı defter “Run All” ile devam eder. Faz numaraları korunur; iki fine-tune aynı akışta çalışır.
Ayrıntı: [fazlar.md](fazlar.md), [colab.md](colab.md).

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

- **21 Kasım 2026:** Hazır DamageFormer zero-shot checkpoint'inden kontrollü fine-tune onaylandı.
  Ana = kontrol, varyant = SAR radyometrik dayanıklılık artırımı (A4, KN1). Her biri bir kez eğitilir.
  Bu karar, 6 Kasım'daki ChangeMamba ana model ve sonraki varyant seçimi kararının yerine geçer.
- Aynı reçete, son adımın EMA'sı ve önceden sabit SAR ayarları kullanılır. Birincil kıyas varyant − ana'dır.
- Eğitim başlamadı; yeni eğitim ve `final` için Türkiye'ye dokunulmadı. Faz 3 yeniden üretimi istisnası saklıdır.
  Kullanıcı `protokol-kilidi`ni 1.3'ü içeren commit'e taşır ve aynı commit'e `varyant-kilidi` koyar.
  İki tasarım da ana modelin Türkiye sonucundan önce kilitlenir.
- **Demo kararı (6 Kasım 2026) sürer:** Gerçek 6 Şubat verisi, BRIGHT Türkiye kareleri kullanılır.
  İstanbul hasar demosu yapılmaz; afet sonrası görüntü yok.

## 12. Claude'un verdiği kararlar (itiraz edilebilir)

- Ana metrik semantik mIoU, `segm_AP` değil.
- Kayıt noktası son adımın EMA’sıdır; doğrulamaya göre seçim yapılmaz (§7).
- Türkiye görüntüsü hiçbir aşamada kullanılmaz; pseudo-label yok.
- Ana model ve varyant hazır DamageFormer’dan başlar; ChangeMamba yedeğe geçiş kuralı bu akışta kullanılmaz.
- Ukrayna, Myanmar ve Meksika optik görüntüleri veri sahibinin Hugging Face kopyasından kullanılır. Kaynakları
  README'de belirsiz olduğu için lisans tablosunda not düşülür.
- İkinci referans HOT/OSM; CSB saha verisi erişilebilirse eklenir.
- Kod mimarisi Bölüm 8a'daki ilkelerle kurulur. Uygulayana bırakılanlar: dosya düzeninin ayrıntısı, izleme aracı
  (JSONL zorunlu, W&B isteğe bağlı), testlerin biçimi.
- ChangeMamba'nın özel CUDA çekirdeği büyük olasılıkla ONNX'e çıkmaz (doğrulanmadı). Faz 4'te 10 dakikalık bir
  denemeyle ölçülür; dışa aktarım yolu buna göre seçilir.
- Belge fazı (Faz 0) koddan önce gelir: ADR'ler, protokol taslağı, veri ve lisans tablosu, risk kaydı, SAR notu.
