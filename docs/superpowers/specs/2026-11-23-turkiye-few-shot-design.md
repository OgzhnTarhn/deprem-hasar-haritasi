# Türkiye few-shot uyarlama deneyi — tasarım

Tarih: 23 Kasım 2026. Durum: kullanıcı onayı bekliyor. Protokol 1.4'ün kaynağıdır.

## 1. Niyet

Zero-shot sonuç (ana 40,44 `[ZS-EVENT mIoU]`, `results/ana/turkiye.json`) değişmeden kalır. Bu ek deney şu soruyu
sorar: *Depremden birkaç gün sonra her ilden birkaç mahalle elle etiketlenirse model ne kadar iyileşir?* Zero-shot
sonuçta hasarlı ve yıkık sınıfların IoU'su %3'ün altında kaldı; deney bu zayıflığı hedefleyen bir kaybı kontrollü
biçimde sınar. Kullanıcının amacı: öğrenme ve CV için ikinci, savunulabilir bir sonuç; ezber olmadan.

Motivasyonun Türkiye zero-shot sonucundan geldiği açıkça yazılır. Bu bir "ikinci deney"dir; tek tarif önceden
kilitlenir, Türkiye test bölgesinde her model bir kez değerlendirilir.

## 2. Hedefler, hedef olmayanlar, karşı hedefler

**Hedefler**
- Kontrollü A/B: aynı Türkiye uyarlama kareleri, aynı tohum, aynı başlangıç; A düz kayıp, B hasar ağırlıklı kayıp.
- Türkiye'nin ayrılmış test bölgesinde eşleştirilmiş kıyaslar: B − A (birincil), A − ana zero-shot, B − ana zero-shot.
- Hasarlı ve yıkık sınıf IoU farkları (B − A) için önceden kayıtlı ikincil aralıklar.

**Hedef olmayanlar**
- Zero-shot başlık sonucunu değiştirmek ya da yeniden değerlendirmek.
- Hiperparametre araması, topluluk, SAR artırımı, test zamanı artırımı.
- Etiket miktarına göre öğrenme eğrisi (seçenek 3, kullanıcı seçmedi).

**Karşı hedefler** (teknik olarak çalışsa da başarısızlık sayılır)
- Bölmeyi, ağırlıkları ya da tarifi Türkiye test bölgesinin sonucuna veya ana modelin şehir/kare bazlı Türkiye
  sonuçlarına bakarak seçmek. `results/ana/turkiye_kare_matrisleri_*.npz` kilitten önce şehir/kare bazında açılmaz.
- Bir koşuyu sonucuna bakıp tekrarlamak; bölmeyi yeniden çekmek.
- `[FS-EVENT mIoU]` sayısını `[ZS-EVENT mIoU]` ile ya da 41,28 ile kıyaslamak.

## 3. Kısıtlar

- Proje kuralı 1'e (Türkiye kilitli) açık, dar bir istisna: yalnız kilitli uyarlama listesindeki kareler, yalnız
  `fewshot-kilidi` etiketinin atası olan commit'lerde eğitime girer. Test ve tampon kareleri eğitimde, doğrulamada,
  kayıt noktası seçiminde ve sıcaklıkta asla kullanılmaz. Zero-shot koşuların kuralları değişmez.
- Kural 2 korunur: başlangıç ağırlığı ana koşunun kendi checkpoint'i; Türkiye'yi görmemiştir.
- Kural 3'e üçüncü ve dördüncü eğitim eklenir (kullanıcı kararı, 22–23 Kasım 2026).
- Kural 4: yeni etiket `[FS-EVENT mIoU]` (few-shot, Türkiye test bölgesi). Zero-shot modelin aynı karelerdeki
  değeri `[FS-EVENT mIoU, zs-ana]` olarak yazılır.
- Kural 5 ve 6 aynen geçerli. Ön kontrol kapılarının hepsi geçmeden GPU eğitimi başlamaz.
- Ortam: Colab A100/L4, ince defter, Run All ile kopmadan devam.

## 4. Değerlendirilen yaklaşımlar

| Yaklaşım | Artı | Eksi | Karar |
|---|---|---|---|
| **1. Ana checkpoint'inden başla, Türkiye uyarlama + 13 olay karışık** | Ezberi yavaşlatır, genel bilgiyi korur, ana kazancını tutar | Örnekleyiciye karışım kodu | **Seçildi** (kullanıcı, 23 Kasım) |
| 2. Ana'dan başla, yalnız Türkiye kareleri | En basit | ~220 kare × 4000 adım ≈ kare başına 150 görüş; ezber; adım sayısını seçecek Türkiye doğrulaması yok | Elendi |
| 3. Hazır zero-shot'tan başla, karışık | Tek fine-tune hikâyesi | Ana'nın +3,24 kazancını atar | Elendi |

Bölme için şehir bazlı (A) ve blok bazlı (B) seçenekler tartışıldı; kullanıcı **blok bazlı**yı seçti (her şehirden
özellik görülsün), iyimserliğe karşı tampon Claude'un önerisi olarak eklendi. Deney boyutu için tek koşu,
A/B ve A/B + öğrenme eğrisi tartışıldı; kullanıcı **A/B**'yi seçti.

## 5. Kararlar

### 5.1 Bölme (yalnız koordinattan, piksel ve etiket okunmadan)

- Kaynak: `results/arastirma/turkiye_bloklar.json` `tile_centers` (1114 kare). Şehir = en yakın `clusters` merkezi
  (7 şehir). Blok = protokol §8'deki 2 km blok; `hasar.eval.bootstrap.turkey_blocks` ne üretiyorsa o (enlem-boylam
  ızgarası, en yakın şehir; 64 blok). Yeni bir ızgara kurulmaz; bölme ve bootstrap aynı blok kimliğini kullanır.
- Her şehirde bloklar sabit tohumla (`seed = 2023`) karıştırılır; bloklar sırayla uyarlamaya alınır, şehrin
  uyarlama karesi o şehrin karelerinin **%20**'sine ulaşınca durulur (son blok bütün olarak alınır).
- **Tampon:** merkezi herhangi bir uyarlama karesinin merkezine **0,5 km**'den yakın olan, uyarlamada olmayan kareler
  tampon olur; hiçbir yerde kullanılmaz. Kalanlar test bölgesidir.
- Çıktı: `results/fewshot/bolme.json` — `adapt`, `buffer`, `test` listeleri, sayılar, şehir başına sayılar, kural
  parametreleri ve listelerin sha256'sı. Komut deterministik; aynı girdiyle aynı dosyayı üretir; testi sentetik
  koordinatlarla yazılır.
- Bölme bir kez üretilir ve commit'lenir; sonradan yeniden çekilmez. Uyarlama karelerinde hasar az çıksa bile
  bölme değişmez.

### 5.2 Eğitim tarifi (A ve B ortak)

- Başlangıç: ana koşunun son EMA ağırlığı (`kosular/ana-708fdc1f5ecb/best.pt`, adım 4000). Yükleme denetimi:
  checkpoint'teki adım 4000 ve kapı 4'te 13 olay doğrulaması 74,55 ± 0,3 `[VAL13 mIoU]` (`results/ana/val13.json`).
  Dosyanın sha256'sı koşu kaydına yazılır.
- Veri karışımı: her partinin yarısı Türkiye uyarlama karelerinden, yarısı 13 olayın eğitim karelerinden. Hangi
  kaynaktan çekileceği ve kaynak içindeki kare yalnız `(seed, sample_no)` ile belirlenir; devam eden koşu aynı
  diziyi üretir. Her kaynak kendi permütasyon çağlarıyla döner.
- Artırım ve girdi işleme: mevcut eğitim yolu (512 kırpma, çevirme, 90° döndürme, nodata yok sayma, SAR resmî
  normalizasyon). SAR radyometrik artırımı kapalı (`sar_p = 0`).
- Optimizasyon: ana ile aynı (batch 8, 4000 adım, AdamW lr 2e-5, wd 5e-3, 200 adım ısınma, kosinüs, min_lr 1e-7,
  bf16, EMA 0,999 başlangıç ağırlığından, tohum 42). Kırpma eşiği kapı 9'dan.
- Doğrulama: yalnız 13 olayın doğrulaması (337 kare), yalnız izleme. Kayıt noktası son adımın EMA'sı (`last_ema`).
- Sıcaklık ve kalibrasyon bu deneyde yok (mIoU'yu değiştirmez; §9).

### 5.3 A ve B'nin tek farkı: sınıf ağırlıkları

- A: `loss.class_weights = (1, 1, 1, 1)`.
- B: arka plan ve sağlam 1; hasarlı ve yıkık sağlam sınıfa göre karekök ters sıklık:
  `w_c = sqrt(f_sağlam / f_c)`, f = 13 olay eğitim piksel sayıları (`results/veri/bolme.json` `class_pixels.train`).
  Bugünkü sayılarla yaklaşık `(1, 1, 2,01, 2,70)`. Değerler yapılandırmaya iki ondalıkla yazılır ve kilitlenir.
- Ağırlık yalnız sınıflandırma CE terimine uygulanır (mevcut kod); Lovász terimleri değişmez.
- Neden karekök (Claude'un kararı): tam ters sıklık (≈ 4,0 ve 7,3) Lovász ile birlikte yanlış pozitifleri artırıp
  sağlam ve arka plan IoU'sunu düşürebilir; Türkiye'ye bakarak ayar yapılamayacağı için ılımlı ve ilkeye dayalı
  değer seçildi. Türkiye uyarlama etiketleri ağırlık hesabına girmez; böylece ağırlık bölmeden önce kilitlenebilir.

### 5.4 Kilit ve erişim

- Protokol 1.4, `configs/fewshot_a.yaml`, `configs/fewshot_b.yaml`, `results/fewshot/bolme.json` aynı commit'te;
  kullanıcı bu commit'e `fewshot-kilidi` etiketini atar (Claude atmaz).
- Türkiye koruması: eğitim veri kümesi Türkiye karesini yalnız (a) `fewshot-kilidi` HEAD'in atasıysa, (b) kilitli
  dosyalar etiketten beri değişmediyse, (c) kare `bolme.json` `adapt` listesindeyse ve listenin sha256'sı tutuyorsa
  kabul eder. Diğer her durumda bugünkü `TurkeyLockError` davranışı sürer.
- Test değerlendirmesi yalnız `test` listesindeki karelerde, aynı etiket koşuluyla, model başına bir kez çalışır;
  sonuç dosyası varsa ikinci kez çalışmaz.

### 5.5 Değerlendirme ve kıyaslar

- Satır: `[FS-EVENT mIoU]` (4 sınıf IoU ve ortalaması, protokolün başlık yolu: resmî değerlendirme, bina maskesi
  uygulanmadan), şehir katmanlı blok bootstrap aralığıyla. Ek satır: bina makro F1 (örnek etiketi olan test
  karelerinde) `[FS-EVENT bina-makro-F1]`.
- Ana zero-shot modelin test karelerindeki değeri, `results/ana/turkiye_kare_matrisleri_headline.npz`'den test
  karelerine süzülerek hesaplanır: `[FS-EVENT mIoU, zs-ana]`. Bu süzme kilitten sonra kodla yapılır.
- Eşleştirilmiş blok bootstrap (aynı kareler, aynı bloklar, 10.000 örnek):
  - Birincil: **B − A** `[FS-EVENT mIoU, eslestirilmis-fark]`.
  - İkincil: A − zs-ana, B − zs-ana.
  - İkincil, sınıf bazlı: B − A hasarlı IoU ve yıkık IoU farkı, %95 aralıkla.
- "Eşit" kuralı protokol §9 ile aynı. Sonuçlar `results/fewshot/{a,b,kiyas}/`.

### 5.6 Defter

- Yeni ince defter `notebooks/fewshot.ipynb` (zero-shot defteri değişmez): kurulum → veri → ana checkpoint'in
  Drive'dan alınıp doğrulanması → A ön kontrol ve eğitim → B ön kontrol ve eğitim → ikisi de bitince A ve B test
  (§9.4). Etiketten aşama seçer,
  Run All ile kopmadan devam eder; önbellek kopyası yalnız klasör varsa yapılır.

## 6. Maliyet ve beklenti

- GPU: eğitim başına ≈ 3,7 CU (kapı 8 projeksiyonu, ana koşu; ölçülmedi, kapı 8 ölçer) + iki test değerlendirmesi.
- Bölme (23 Kasım 2026, `results/fewshot/bolme.json`): uyarlama 303, tampon 170, test 641 kare. Bütün bloklar alındığı
  için uyarlama %20 hedefini aşıp %27 oldu. Tampon ilk 1,0 km idi; o değerde test yalnız 355 kare (en küçük şehir 28)
  kalıyordu. Kullanıcı 23 Kasım'da 0,5 km'yi seçti: yan ve çapraz bütün komşular çıkar, test 641 (en küçük şehir 68).
  Seçim yalnız kare sayılarına bakılarak, hiçbir sonuç görülmeden yapıldı.
- Beklenti: sağlam ve hasar sınıflarında artış olası, garanti yok. Hangi sonuç çıkarsa yazılır.

## 7. Riskler

- Blok bazlı bölmede uzamsal benzerlik: 0,5 km tampon azaltır, tamamen gidermez; raporda yazılır.
- Uyarlama karelerinde yıkık piksel çok az olabilir; bölme yine değişmez.
- Ağırlıklı kayıp hasar sınıflarında yanlış pozitif artırabilir; B − A sınıf farkları bunu gösterir.
- Test bölgesi 1114 karenin alt kümesi; ana zero-shot sonucu bu karelerde toplu olarak zaten görüldü. Şehir/kare
  bazlı zero-shot sonuçlarına kilitten önce bakılmaz.

## 9. İnceleme sonrası kararlar (23 Kasım 2026, Claude'un kararları)

Bağımsız inceleme (konsey, Codex) on soru çıkardı. Hepsi aşağıdaki gibi kapatıldı; üstteki bölümlerle çelişirse
bu bölüm geçerlidir.

1. **Bootstrap hizalaması.** `paired_block_bootstrap` blokları iki modelin kendi kare sırasıyla kuruyor; sıralar
   farklıysa bloklar yanlış eşleşir (nokta tahmini değişmez, aralık ve "eşit" kararı bozulur). Düzeltme: bloklar
   `(şehir, blok kimliği)` ile hizalanır ve kare kümeleri ad bazında eşleştirilir; ters sıralı özdeş tahminlerin
   sıfır genişlikli fark aralığı verdiği bir test eklenir. Zero-shot kıyasları etkilenmedi: ana, varyant ve
   referans arşivleri aynı kare sırasında (yalnız adlar denetlendi, 23 Kasım).
2. **Kırpma eşiği.** İki kol aynı eşiği kullanır: ana koşunun kapı 9 ölçümü 3,1627 (`kosular/onkontrol/ana`),
   yapılandırmaya `train.clip: 3.1627` olarak yazılır. Kapı 9 iki kolda da koşar ama eşiği değiştirmez.
3. **Mikro parti.** İki kolda `train.micro_batch: 4` sabitlenir (A100 ve L4'te sığar); bellek hatasında yarıya
   inme kolları farklılaştırmasın diye yarılanma bu deneyde hata verir.
4. **Test sırası.** Önce A ve B'nin eğitimleri biter; Türkiye test değerlendirmesi ikisi de `DONE` olduktan sonra
   başlar. Böylece A'nın test sonucu görülüp B düzeltilemez.
5. **Erişim kapsamı.** HEAD `fewshot-kilidi`'nin torunu olmalı, çalışma ağacı temiz olmalı (eğitim ve test),
   kilitli dosyalar (`docs/protokol.md`, iki few-shot yapılandırması, `results/fewshot/bolme.json`) etiketten beri
   değişmemiş olmalı. Kod düzeltmeleri etiketten sonra gelebilir (zero-shot akışıyla aynı). İzin okuma sınırında
   uygulanır: eğitim kümesi yalnız `adapt`, test yolu yalnız `test` karelerini okuyabilir; doğrulama 13 olayda kalır.
6. **Başlangıç kimliği.** Kilit commit'inden önce kullanıcı Colab'da `best.pt`'nin sha256'sını hesaplar; değer
   yapılandırmaya (`model.init_sha256`) yazılır ve kilitlenir. Yükleme, ön kontrol ve devam bu değeri denetler;
   tutmazsa durur. Kapı 4 beklenen değeri `results/ana/val13.json` (74,55 ± 0,3 `[VAL13 mIoU]`).
7. **A − zs-ana yorumu.** A, Türkiye etiketlerine ek olarak 13 olayda 4000 adım daha eğitim görür. Bu fark
   "Türkiye uyarlama etiketleri + ek eğitim" diye raporlanır; yalnız Türkiye etiketlerinin etkisi diye yazılmaz.
   Ayrıştırmak için ek kontrol koşusu yapılmaz (kullanıcı iki koşu seçti).
8. **Değerlendirme yolu.** `run_final` yeniden kullanılmaz; yeni few-shot değerlendirmesi yalnız `test` listesini
   okur, etiketi `[FS-EVENT mIoU]`, 41,28 çapası ve flip-TTA yok. Tamamlandı işareti `results/fewshot/{a,b}/` altında,
   koşu kimliği ve bölme sha256'sıyla bağlı; varsa ikinci kez çalışmaz. Kare matrisleri her seferinde float32
   olasılıklardan hesaplanır.
9. **Bina F1.** Örnek etiket listesi ile `test` kesişimi kullanılır; kesişimdeki bir dosya eksikse değerlendirme
   başlamadan durur.
10. **Zero-shot defteri.** İki zero-shot final'i bitti; `ana_egitim.ipynb` protokol 1.4 sonrası çalıştırılmak üzere
    güncellenmez, başına "tarihî, protokol 1.3" notu eklenir.

## 8. Uygulayıcıya bırakılanlar

Komut ve fonksiyon adları, karışık örnekleyicinin iç yapısı, koruma API'sinin biçimi, ön kontrol kapılarının yeni
eğitim kümesine uyarlanması (kapı 2 beklenen sayısı 2795 + uyarlama, kapı 3 örnekleyici uzunluğu), defter hücre
düzeni, testler (sentetik, veri okumaz).
