# Öğrenme Rehberi: Deprem Hasar Haritası projesini baştan sona anlamak

Bu belge projede kullandığımız ve kullanacağımız her tekniği, neden seçildiğiyle birlikte anlatır. Hiçbir ön bilgi
varsaymaz; derin öğrenme, uydu görüntüsü ya da istatistik bilmeden de baştan okunabilir. Her bölüm bir öncekinin
üstüne kurulur.

- **Bağlayıcı kararlar** bu belgede değil, [tasarim.md](tasarim.md)'de durur. Bu rehber onları açıklar, değiştirmez.
- **Fazlar:** [fazlar.md](fazlar.md). **Ertelenmiş fikirler:** [fikir-havuzu.md](fikir-havuzu.md).
- Belge, proje ilerledikçe güncellenir. Bir şey tasarımla çelişirse tasarım geçerlidir.

**Okuma önerisi:** İlk okumada 1–4. bölümler (problem, uydu, veri, görev) projenin "ne"sini verir. 5–8. bölümler
(derin öğrenme, mimari, eğitim, değerlendirme) "nasıl"ını. 9–13. bölümler (dürüstlük, kontroller, belirsizlik,
yenilikler) projeyi rakiplerinden ayıran tarafı. 14–17. bölümler yazılım ve altyapı. Sonda bir sözlük var.

---

## İçindekiler

1. [Problem: depremden sonra hangi bina yıkıldı?](#1-problem-depremden-sonra-hangi-bina-yıkıldı)
2. [Uydu görüntüleri: optik ve radar](#2-uydu-görüntüleri-optik-ve-radar)
3. [Veri: BRIGHT](#3-veri-bright)
4. [Görev: bilgisayara tam olarak ne yaptırıyoruz?](#4-görev-bilgisayara-tam-olarak-ne-yaptırıyoruz)
5. [Derin öğrenmenin temelleri](#5-derin-öğrenmenin-temelleri)
6. [Mimari: ChangeMamba-MMBDA](#6-mimari-changemamba-mmbda)
7. [Eğitim: modeli nasıl öğreteceğiz?](#7-eğitim-modeli-nasıl-öğreteceğiz)
8. [Değerlendirme: ne kadar iyi olduğunu nasıl ölçeceğiz?](#8-değerlendirme-ne-kadar-iyi-olduğunu-nasıl-ölçeceğiz)
9. [Bilimsel dürüstlük: protokol ve kilitler](#9-bilimsel-dürüstlük-protokol-ve-kilitler)
10. [Ön kontrol kapıları](#10-ön-kontrol-kapıları)
11. [Belirsizlik ve kalibrasyon](#11-belirsizlik-ve-kalibrasyon)
12. [Yenilik analizleri](#12-yenilik-analizleri)
13. [Fikir havuzundaki teknikler](#13-fikir-havuzundaki-teknikler)
14. [Yazılım mimarisi](#14-yazılım-mimarisi)
15. [Altyapı: Colab, GPU ve bütçe](#15-altyapı-colab-gpu-ve-bütçe)
16. [Ürünleştirme ve demo](#16-ürünleştirme-ve-demo)
17. [Yolculuk: fazlar ve karar noktaları](#17-yolculuk-fazlar-ve-karar-noktaları)
18. [Sözlük](#18-sözlük)
19. [Kaynaklar ve ileri okuma](#19-kaynaklar-ve-ileri-okuma)

---

## 1. Problem: depremden sonra hangi bina yıkıldı?

6 Şubat 2023'te Kahramanmaraş merkezli iki büyük deprem on bir ili etkiledi. Böyle bir afetten sonraki ilk
saatlerde ve günlerde arama-kurtarma ekipleri, belediyeler ve yardım kuruluşları aynı soruyu sorar: **Hangi
binalar yıkıldı, hangileri hasarlı, hangileri sağlam?** Bu cevap ekiplerin nereye gönderileceğini, çadırların
nereye kurulacağını ve hasar tespitinin hangi mahallelerden başlayacağını belirler.

Binaları tek tek gezmek haftalar sürer. Uydu görüntüleri bütün bölgeyi birkaç gün içinde gösterebilir. Ama on
binlerce binaya bakıp her birini işaretlemek uzmanlar için de günler alır. Örneğin Copernicus Acil Durum Yönetim
Servisi'nin elle hazırladığı haritalar, sonradan yapılan saha sayımına karşı bina düzeyinde yalnız 0,15–0,23 F1
aldı (Ainscoe ve ark.). Yani elle yorumlama hem yavaş hem de sanıldığı kadar isabetli değil.

**Projenin fikri:** Bu işaretlemeyi bir yapay zekâ modeline yaptırmak. Model uydu görüntüsüne bakar ve her piksel
için "burası bina değil / sağlam bina / hasarlı bina / yıkık bina" der. Çıktı renkli bir hasar haritasıdır.

**Projenin asıl sorusu ise biraz farklı:**

> Model, eğitimde hiç görmediği bir depremde ne kadar işe yarıyor ve bu sonuca ne kadar güvenebiliriz?

Bu soru önemli, çünkü gerçek hayatta yeni bir deprem olduğunda o depremin etiketli verisi elimizde olmaz. Model
yalnızca geçmiş afetlerden öğrendikleriyle çalışmak zorundadır. Bir modelin "eğitildiği veriye benzeyen" veride
iyi olması kolaydır; asıl zor olan, farklı bir ülkede, farklı yapı stokunda, farklı bir uyduyla çekilmiş yeni bir
afette işe yaramasıdır. Buna **genelleme** denir.

Projenin ikinci yarısı ("ne kadar güvenebiliriz?") da en az birincisi kadar önemli. Tek bir sayı ("%41") vermek
yetmez. O sayının belirsizliği nedir? Hangi sınıflarda iyi, hangilerinde kötü? Kime göre doğru? Bu sorular
projenin karakterini belirliyor ve ileride ayrıntılı anlatılacak.

---

## 2. Uydu görüntüleri: optik ve radar

### 2.1 Optik görüntü

Optik uydu, bildiğimiz fotoğraf makinesi gibi çalışır: Güneş ışığının yeryüzünden yansımasını kaydeder. Görüntü
piksellerden oluşur. Her pikselin birkaç **bandı** vardır; bizim veride üç bant var (kırmızı, yeşil, mavi), yani
gözümüzün gördüğü renkli görüntü.

**Çözünürlük** burada "bir pikselin yerde kapladığı alan" demektir ve **GSD** (ground sampling distance) diye
anılır. BRIGHT'taki görüntülerde GSD yaklaşık 0,3–0,35 metre: bir piksel yerde 35 santimetrelik bir kareye denk
geliyor. Bu, bir arabayı, bir çatıyı, hatta enkaz yığınını görebilecek kadar ayrıntılıdır. Bu sınıfa **çok yüksek
çözünürlük** (VHR, very high resolution) denir. Türkiye görüntüleri Maxar şirketinin uydularından geliyor.

Optiğin iki zayıflığı var: **bulut** ve **gece**. Işık yoksa ya da bulut varsa görüntü yoktur. Şubat ayında, kışın
ve duman altında bu ciddi bir sorundur.

### 2.2 Radar (SAR) görüntüsü

**SAR** (Synthetic Aperture Radar, sentetik açıklıklı radar) farklı bir şekilde çalışır. Uydu kendi mikrodalga
sinyalini gönderir ve yeryüzünden geri dönen yansımayı ölçer. Yani:

- **Kendi ışığını kendisi getirir:** Gece de çalışır.
- **Mikrodalga bulutu geçer:** Hava koşulundan neredeyse bağımsızdır.

Bu yüzden afet müdahalesinde çok değerlidir. Bizim verideki radar görüntüleri **Capella Space** ve **Umbra**
şirketlerinin küçük uydularından geliyor. İkisi de **X-bandı** kullanır (dalga boyu yaklaşık 3 cm) ve optikle
benzer, santimetreler mertebesinde ayrıntı verebilir.

Radar görüntüsü bir fotoğrafa benzemez. Yorumlamak için bilinmesi gereken birkaç kavram var:

**Genlik (amplitude).** Her pikselin değeri, o noktadan uyduya ne kadar sinyal döndüğünü gösterir. Parlak piksel
"çok sinyal döndü", koyu piksel "az döndü" demektir. Düz ve pürüzsüz yüzeyler (sakin su, asfalt) sinyali ayna gibi
başka yöne yansıtır ve koyu görünür. Pürüzlü yüzeyler sinyali her yöne saçar ve orta tonda görünür.

**Çift sekme (double bounce).** Bir binanın duvarı ile önündeki zemin, birbirine dik iki ayna gibi bir **köşe
yansıtıcı** oluşturur. Sinyal önce zemine, sonra duvara çarpar ve doğrudan uyduya geri döner. Bu yüzden sağlam
binalar radarda parlak çizgiler hâlinde görünür. **Yıkılan bina bu köşeyi kaybeder**; yerinde sinyali dağınık
saçan bir enkaz yığını kalır. Radarla hasar bulmanın fiziksel temeli büyük ölçüde budur.

**Yandan bakış ve geometri bozulmaları.** Radar aşağıya dik değil, yana eğik bakar. Bu, iki bozulma yaratır:

- **Bindirme (layover):** Yüksek bir binanın tepesi uyduya tabanından daha yakın olduğu için görüntüde tabanından
  önce görünür; bina, uyduya doğru "devrilmiş" gibi durur ve önündeki şeylerle üst üste biner.
- **Gölge (shadow):** Binanın arkasında radarın hiç göremediği bir alan kalır; orası simsiyahtır. Optik gölgeden
  farklı olarak bu gölge Güneş'e değil uydunun bakış açısına bağlıdır.

Yoğun ve yüksek yapılaşmış bir şehirde bu iki etki binaları birbirine karıştırır. Hata analizinde (Bölüm 12) bu
yüzden "layover ve gölge" ayrı başlık olarak incelenecek.

**Benek (speckle).** Radar sinyali eş fazlıdır (coherent). Bir pikselin içindeki pek çok küçük yansıtıcının
dalgaları birbirini kimi yerde güçlendirir, kimi yerde söndürür. Sonuç, görüntünün üstüne serpilmiş tuz-biber
benzeri bir desen olur. Bu bir hata değil, radarın doğasıdır; ama modelin "gürültü ile hasarı" ayırmasını zorlaştırır.

**Desibel (dB).** Radar genliği çok geniş bir aralıkta değişir (bazı noktalar binlerce kat daha parlaktır).
Bu yüzden genellikle logaritmik ölçeğe, **desibele** çevrilerek kullanılır: `dB = 10 · log10(güç)`. BRIGHT'ta SAR
0–255 arası tam sayılar (uint8) olarak saklanıyor; bu değerlerin doğrusal genlik mi yoksa dB'den mi türetildiği
henüz doğrulanmadı (Faz 2'de bakılacak). Bu ayrıntı önemli, çünkü model girdisinin ölçeği öğrenmeyi etkiler.

**Koherans (coherence).** Aynı yerin aynı yörüngeden, birkaç gün arayla çekilmiş iki radar görüntüsünün yalnız
genliği değil **fazı** da karşılaştırılabilir. Hiçbir şey değişmediyse iki görüntü faz olarak birbirine çok
benzer (koherans 1'e yakın). Bina çöktüyse benzerlik düşer (koherans 0'a yaklaşır). Bu bilgi, ham karmaşık değerli
radar verisi (SLC, single look complex) gerektirir. BRIGHT'ta bu yok; ama ücretsiz Avrupa uydusu **Sentinel-1**
her yeri düzenli olarak çeker ve koheransı hesaplamaya izin verir (çözünürlüğü daha düşük: 10–30 m). NASA'nın
**ARIA Hasar Vekil Haritası (DPM)** tam olarak bu ilkeyle üretilir. Bu kavram Bölüm 12 ve 13'te tekrar karşımıza
çıkacak.

### 2.3 Neden "öncesi optik + sonrası radar"?

BRIGHT'taki her örnek bir çiftten oluşur:

```
Afet ÖNCESİ optik görüntü   +   Afet SONRASI radar görüntüsü
(binalar nerede, neye benziyor)   (şimdi nasıl görünüyorlar)
```

Bu eşleşme operasyonel bir senaryoyu taklit eder. Afetten önce bölgenin bulutsuz bir optik görüntüsü neredeyse
her zaman arşivde bulunur. Afetten sonra ise havanın açılmasını beklemek yerine radarla hemen çekim yapılabilir.

Bu kurgunun bir sınırı var: **Afet öncesi radar görüntüsü yok.** Klasik radar değişim tespiti "önceki radar ile
sonraki radarı karşılaştır" mantığıyla çalışır. Bizim model ise "önceden burada bina vardı (optikten biliyorum);
şimdi radarda böyle görünüyor; sağlam bir bina radarda nasıl görünmeliydi?" sorusunu kendisi çözmek zorunda.
İki farklı sensörün dilini birbirine çevirmesi gerekiyor. Görevi zorlaştıran şeylerden biri bu.

### 2.4 Coğrafi veri: GeoTIFF ve koordinat sistemleri

Uydu görüntüleri **GeoTIFF** dosyalarında saklanır. Bu, sıradan bir TIFF resim dosyasıdır, ama içinde görüntünün
dünya üzerindeki yerini anlatan bilgi (georeferans) de bulunur: Sol üst köşenin koordinatı, bir pikselin kaç metre
olduğu ve hangi **koordinat referans sistemi** (CRS) kullanıldığı.

Türkiye görüntüleri **UTM 37N** sisteminde. UTM, Dünya'yı 6 derecelik dilimlere bölüp her dilimi düz bir haritaya
açan bir sistemdir; birimi metredir, bu yüzden mesafe ve alan hesabı kolaydır. 37N, Türkiye'nin doğusunu kapsayan
kuzey yarımküre dilimidir. Bu bilgi bize iki şey için lazım:

1. Dış katmanları (OpenStreetMap binaları, ARIA hasar haritası, sarsıntı haritası) aynı binaların üstüne doğru
   oturtmak (Bölüm 12).
2. Demo haritasında tahminleri gerçek konumlarında göstermek (Bölüm 16).

Dosyaları okumak için iki kütüphane kullanacağız: Eğitimde hızlı olduğu için **tifffile**, coğrafi bilgiyi
korumamız gereken çıktılarda **rasterio**.

---

## 3. Veri: BRIGHT

### 3.1 İçerik

**BRIGHT**, Tokyo Üniversitesi ve ortaklarının hazırladığı, ESSD dergisinde 2025'te yayımlanan açık bir veri
setidir. Dünyanın farklı yerlerinden **14 afet olayı** içerir: depremler (Türkiye, Fas, Haiti/Les Cayes, Japonya/
Noto, Myanmar...), seller, orman yangınları, kasırgalar, bir yanardağ (La Palma) ve savaş hasarı (Ukrayna).

| Özellik | Değer |
|---|---|
| Görüntü çifti | 4.246 |
| Kare boyutu | 1024 × 1024 piksel |
| Optik | 3 bant, uint8 |
| SAR | 1 bant, uint8 |
| Etiket | Her piksel için {0, 1, 2, 3} |
| Boyut | Yaklaşık 13,2 GB zip, açılmış hâli 22,3 GB |
| Lisans | Veri CC BY-SA 4.0 |

Etiket değerlerinin anlamı:

| Değer | Sınıf |
|---|---|
| 0 | Arka plan (bina değil) |
| 1 | Sağlam bina |
| 2 | Hasarlı bina |
| 3 | Yıkık bina |

### 3.2 Türkiye'nin yeri

Türkiye olayı veri setinin açık ara en büyüğü: **1.114 kare, 135.033 bina**, veri setindeki bina piksellerinin
yaklaşık **%52'si**. Türkiye'deki bina piksellerinin %93,0'ı sağlam, %4,1'i hasarlı, %2,9'u yıkık.

Bunun iki sonucu var:

1. Türkiye'yi eğitimden çıkardığımızda modelin eğitim verisinin yarısından fazlasını kaybediyoruz. Görev bu yüzden
   doğal olarak zorlaşıyor.
2. Hasarlı piksellerin yaklaşık %63'ü Ukrayna ve Meksika olaylarında. Yani model "hasarın sık görüldüğü"
   olaylardan öğrenip "hasarın nadir olduğu" (sağlamların %93 olduğu) Türkiye'de sınanıyor. Buna **sınıf dağılımı
   kayması** (label shift) denir. Bölüm 13'te bununla başa çıkma yolları anlatılıyor.

### 3.3 Etiketler nereden geliyor?

Bu çok önemli bir ayrıntı: **BRIGHT etiketleri saha sayımı değildir.** Uzmanlar afet öncesi ve sonrası çok yüksek
çözünürlüklü **optik** görüntülere bakarak her binayı işaretlemiştir. Yani etiket de bir yorumdur ve hata
içerebilir. Örneğin:

- Çatısı sağlam görünen ama içten çökmüş bir bina optikte "sağlam" etiketlenebilir.
- "Hasarlı" ile "yıkık" arasındaki sınır insanlar arasında bile tartışmalıdır. Bir çalışmada aynı binaların drone ve
  uydu görüntülerinden etiketleri %29 oranında anlaşmadı.

Bu yüzden projede "model yanlış" dediğimiz bazı yerlerde aslında etiket yanlış olabilir. Bu gözlem projenin
yeniliklerinden birinin (Y1, referans anlaşmazlığı) temelidir.

### 3.4 Nodata

Bazı karelerde görüntünün bir kısmı ya da tamamı boştur: Uydu o bölgeyi çekmemiştir ve pikseller 0'dır. Buna
**nodata** denir. Fas'ta tamamen boş SAR kareleri var. Model bu piksellerden bir şey öğrenemez; öğrenmeye çalışırsa
yanlış şeyler öğrenir ("siyah = sağlam" gibi). Bu yüzden **eğitimde**, optiğin ve SAR'ın ikisinin de 0 olduğu
pikseller kayıp hesabında **yok sayılır** (Bölüm 7.3). **Değerlendirmede** ise resmî kodun yaptığı neyse aynısını
yaparız; aksi hâlde sonucumuz yayımlanmış sayılarla kıyaslanamaz (Bölüm 9).

### 3.5 Lisanslar

| Kaynak | Lisans | Anlamı |
|---|---|---|
| BRIGHT verisi | CC BY-SA 4.0 | Atıfla ve aynı lisansla paylaşılabilir |
| BRIGHT ve ChangeMamba kodu | Apache-2.0 | Atıfla serbest kullanım |
| Zenodo checkpoint'leri | CC BY 4.0 | Atıfla serbest |
| Maxar Open Data (demo için) | CC BY-NC 4.0 | Yalnız ticari olmayan kullanım |

Ukrayna, Myanmar ve Meksika optik görüntülerinin dağıtım hakkı BRIGHT'ın README'sinde belirsiz bırakılmış; bu
yüzden Faz 0'daki veri ve lisans tablosunda not düşülecek.

---

## 4. Görev: bilgisayara tam olarak ne yaptırıyoruz?

### 4.1 Semantik segmentasyon

Bilgisayarlı görüde üç temel görev vardır:

- **Sınıflandırma:** "Bu resimde ne var?" Tüm resme tek etiket.
- **Nesne tespiti:** "Nesneler nerede?" Her nesneye bir kutu.
- **Semantik segmentasyon:** "Her piksel neye ait?" Her piksele bir etiket.

Bizim görevimiz **semantik segmentasyon**: 1024 × 1024 = yaklaşık bir milyon pikselin her birine dört sınıftan
birini atamak. Çıktı, girdiyle aynı boyutta bir etiket haritasıdır.

Bir dördüncü görev daha var: **örnek segmentasyonu** (instance segmentation). Burada her binanın ayrı bir nesne
olarak çıkarılması gerekir ("bu pikseller 1 numaralı binaya, şunlar 2 numaralıya ait"). 2026 CVPR çalıştayındaki
BRIGHT yarışması bu görevi kullandı ve `segm_AP` diye bir metrikle ölçtü. Biz bu görevi yapmıyoruz; o sayılar
bizimkilerle kıyaslanamaz (Bölüm 8.7).

### 4.2 Değişim tespiti

Girdimiz tek bir resim değil, iki zamandan iki resim. İki zaman arasında ne değiştiğini bulmaya **değişim tespiti**
(change detection) denir. Hasar haritalama özel bir değişim tespitidir: Yalnız "değişti mi?" değil, "binada ne
kadar hasar var?" sorusunu sorar. Bu alanın literatüründe "BDA" (building damage assessment) kısaltması kullanılır.

### 4.3 İki alt görev

Hasar haritası aslında iki ayrı sorudan oluşur:

1. **Bina konumlandırma (localization):** Bu piksel bir binaya mı ait? (2 sınıf: evet/hayır)
2. **Hasar sınıflandırma (classification):** Bu bina sağlam mı, hasarlı mı, yıkık mı?

Birinci soru yalnız afet öncesi optikten cevaplanabilir ve görece kolaydır. İkincisi asıl zor kısımdır ve hem
öncesi hem sonrası görüntüye ihtiyaç duyar. Bizim modelimiz bu iki soruyu ayrı "başlıklarla" cevaplar (Bölüm 6.5).

### 4.4 Piksel düzeyi ve bina düzeyi

Model piksel piksel tahmin yapar. Ama gerçek dünyada karar **bina** düzeyinde verilir: "Bu bina yıkık." Tek bir
binanın pikselleri modelde farklı sınıflara düşebilir (çatının yarısı "hasarlı", yarısı "yıkık"). Bu yüzden iki
düzeyde de ölçüm yapacağız (Bölüm 8.4) ve piksel tahminlerini binalara toplamanın yollarını inceleyeceğiz (bina
oylaması, Bölüm 8.5).

---

## 5. Derin öğrenmenin temelleri

Bu bölüm, sonraki bölümlerin dilini kurar. Matematiği en aza indirdim; amaç sezgi.

### 5.1 Sinir ağı nedir?

Bir **sinir ağı**, girdiyi (bizde görüntüler) çıktıya (etiket haritası) dönüştüren, milyonlarca ayarlanabilir sayı
(**parametre** ya da **ağırlık**) içeren büyük bir matematiksel fonksiyondur. Başta ağırlıklar rastgeledir ve
çıktı anlamsızdır. **Eğitim**, bu ağırlıkları örneklere bakarak yavaş yavaş düzeltme işidir.

Eğitimin bir adımı şöyle işler:

1. **İleri geçiş:** Bir grup görüntü ağa verilir, ağ tahmin üretir.
2. **Kayıp (loss):** Tahmin doğru etiketle karşılaştırılır ve "ne kadar yanlış" tek bir sayıyla ölçülür. Kayıp ne
   kadar küçükse tahmin o kadar iyidir.
3. **Geri yayılım (backpropagation):** Her ağırlığın kaybı ne yönde ve ne kadar etkilediği hesaplanır. Bu bilgiye
   **gradyan** denir. Gradyan, "bu ağırlığı biraz artırırsan kayıp şu kadar artar" der.
4. **Güncelleme:** Bir **optimizer** (iyileştirici), her ağırlığı kaybı azaltacak yöne küçük bir adım kaydırır.
   Adımın büyüklüğüne **öğrenme oranı** (learning rate, lr) denir.

Bu döngü on binlerce kez tekrarlanır. Bizim planımız 40.000 adım. Her adımda ağa giden görüntü grubuna **batch**
denir; bizde 16 görüntü.

### 5.2 Özellik haritaları ve kodlayıcı-çözücü yapısı

Bir görüntü ağın içinden geçerken katman katman **özellik haritalarına** (feature map) dönüşür. İlk katmanlar
kenar, köşe ve doku gibi basit desenleri yakalar. Derin katmanlar bunları birleştirip "çatı", "yol", "enkaz yığını"
gibi soyut kavramlara ulaşır.

Segmentasyon ağları genellikle iki parçadan oluşur:

```
Görüntü (1024×1024)
   │
   ▼
KODLAYICI (encoder)      → görüntüyü adım adım küçültür, anlamı derinleştirir
   1/4 ölçek: kenarlar, dokular
   1/8 ölçek: parçalar
   1/16 ölçek: nesneler
   1/32 ölçek: sahne bağlamı
   │
   ▼
ÇÖZÜCÜ (decoder)         → özellikleri adım adım büyütür, piksel çözünürlüğüne geri döner
   │
   ▼
Etiket haritası (1024×1024)
```

Kodlayıcı "ne"yi, ilk katmanlar "nerede"yi iyi bilir. Çözücü, farklı ölçeklerdeki özellikleri birleştirerek hem
anlamı hem de keskin sınırları koruyan bir harita üretir. Bu birleştirmeye **atlamalı bağlantı** (skip connection)
denir; U-Net mimarisinin temel fikridir.

### 5.3 Ön eğitim ve transfer öğrenme

Sıfırdan, rastgele ağırlıklarla başlayan bir model, kenarın ne olduğunu bile öğrenmek için çok veri harcar. Bunun
yerine kodlayıcıyı önce büyük ve genel bir veri setinde eğitip sonra kendi görevimize uyarlarız. Buna **transfer
öğrenme** ve ilk aşamaya **ön eğitim** (pretraining) denir.

En yaygın ön eğitim verisi **ImageNet**: 1.000 sınıflı, yaklaşık 1,3 milyon gündelik fotoğraf. Uydu görüntüsü
gündelik fotoğrafa benzemese de kenar, doku ve şekil gibi temel kavramlar aktarılabilir. Bizim ana modelimizin
kodlayıcısı ImageNet'te ön eğitilmiş `vssm_tiny` ağırlıklarıyla başlıyor.

**Neden daha "akıllı" bir ön eğitim kullanmıyoruz?** Uydu görüntüsü için özel eğitilmiş modeller (temel modeller,
foundation models) var. Ama iki sorun var: Bazılarının ön eğitim verisi 2023 Türkiye sahnelerini içeriyor olabilir
(o zaman model Türkiye'yi "görmüş" olur, Bölüm 9) ve bazılarının lisansı sorunlu. Üstelik BRIGHT ekibinin 2026
karşılaştırmasında genel amaçlı temel modeller özel mimarilerin gerisinde kaldı.

### 5.4 CNN'den Transformer'a, oradan Mamba'ya

Ana modelimizi anlamak için görüntü işleyen ağların kısa tarihini bilmek gerekiyor.

**Evrişimli sinir ağları (CNN).** Görüntü üzerinde küçük bir pencere (ör. 3×3 piksel) kaydırılır ve her konumda
aynı işlem uygulanır. Bu, yerel desenleri yakalamakta çok iyidir ve hesap olarak ucuzdur. Zayıflığı: Bir piksel
başlangıçta yalnız yakın komşularını "görür". Uzak bağlamı görmek için çok sayıda katman üst üste konur. Bir
pikselin etkilendiği alana **alıcı alan** (receptive field) denir. UNet, DeepLabV3+ ve SiamCRNN bu ailedendir.

**Transformer ve dikkat (attention).** Dikkat mekanizmasında her konum diğer bütün konumlara bakar ve hangilerinin
kendisi için önemli olduğuna karar verir. Böylece ilk katmandan itibaren tüm görüntüyü görebilir. Bedeli:
N konum varsa N × N ilişki hesaplanır. 1024 × 1024'lük bir görüntüde bu çok pahalıdır (maliyet **karesel**
büyür). **Swin Transformer** bunu görüntüyü küçük pencerelere bölüp dikkati pencere içinde hesaplayarak ve
pencereleri katmandan katmana kaydırarak çözer. Yedek modelimiz DamageFormer Swin-T kullanır.

**Mamba ve durum uzay modelleri.** Mamba, 2023'te dil modelleri için önerilen ve Transformer'a alternatif olan
bir mimaridir. Sezgisi şöyle: Bir diziyi baştan sona okurken her adımda küçük bir **gizli durum** (hidden state)
güncellenir; bu durum o ana kadar okunanların özetidir.

```
h_t = A · h_(t-1) + B · x_t      (durumu güncelle: eskiyi biraz unut, yeniyi ekle)
y_t = C · h_t                    (durumdan çıktı üret)
```

Klasik durum uzay modellerinde A, B, C sabittir. Mamba'nın yeniliği **seçicilik**tir: Bu katsayılar girdiye göre
değişir. Model her adımda "bu bilgi önemli, hatırla" ya da "bu önemsiz, unut" diye karar verebilir. Maliyet dizi
uzunluğuyla **doğrusal** büyür (Transformer'daki gibi karesel değil) ama uzak bağlam yine taşınabilir.

Bu "seçici tarama" (**selective scan**) işlemi verimli çalışsın diye GPU için özel yazılmış bir **CUDA çekirdeği**
kullanılır. Bu ayrıntı projede önemli bir mühendislik riskine dönüşüyor (Bölüm 6.7 ve 15).

**Görüntüde Mamba: VMamba.** Görüntü bir dizi değildir; iki boyutludur. **VMamba**, görüntüyü dört farklı yönde
tarayarak (soldan sağa, sağdan sola, yukarıdan aşağıya, aşağıdan yukarıya) diziye çevirir ve sonuçları birleştirir
(**SS2D**, 2B seçici tarama). Böylece her piksel, dört yönden gelen bağlamı alır. VMamba'nın kodlayıcı ağına
**VSSM** (Visual State Space Model) denir; "Tiny" en küçük boyutudur.

---

## 6. Mimari: ChangeMamba-MMBDA

### 6.1 Genel resim

**ChangeMamba**, VMamba kodlayıcılarını değişim tespitine uyarlayan bir model ailesidir. Ailenin hasar haritalama
üyesine **MambaBDA** denir. BRIGHT ekibinin optik + SAR için uyarladığı sürüme **MMBDA** (multimodal BDA) diyoruz.
Bizim kullanacağımız boyut **Tiny**.

```
 Afet öncesi OPTİK (3 bant)            Afet sonrası SAR (1 bant)
          │                                     │
          ▼                                     ▼
 ┌──────────────────┐                  ┌──────────────────┐
 │ Optik kodlayıcı  │                  │  SAR kodlayıcı   │
 │   (VSSM Tiny)    │                  │   (VSSM Tiny)    │
 └──────────────────┘                  └──────────────────┘
   │  4 ölçekte özellik                     │  4 ölçekte özellik
   │                                        │
   ├───────────────┐                        │
   │               │                        │
   ▼               ▼                        │
 ┌─────────────┐  ┌──────────────────────────────────────┐
 │    BİNA     │  │         HASAR ÇÖZÜCÜSÜ               │
 │  ÇÖZÜCÜSÜ   │  │  her ölçekte optik + SAR birleştirme │
 │(yalnız optik)│ │           (cat)                      │
 └─────────────┘  └──────────────────────────────────────┘
       │                           │
       ▼                           ▼
  loc: 2 sınıf               clf: 4 sınıf
  (bina / değil)             (arka plan, sağlam, hasarlı, yıkık)
```

### 6.2 İki ayrı kodlayıcı

Optik ve radar görüntüleri çok farklı "dillerde" konuşur: Biri yansıyan güneş ışığını, öteki geri dönen mikrodalga
sinyalini kaydeder. Parlaklık, doku ve geometri bambaşkadır. Bu yüzden ChangeMamba her modalite için **ayrı bir
kodlayıcı** kullanır. İki kodlayıcı aynı mimariye sahiptir ama ağırlıkları ayrıdır; her biri kendi modalitesinin
desenlerini öğrenir.

Bu konuda literatür tek sesli değil: 2025 IEEE GRSS Veri Füzyonu Yarışması'nın (DFC25) iki birincisi semantik
görevde **paylaşımlı** kodlayıcı kullandı. Öte yandan 2026 CVPR çalıştayının iki birincisi bağımsız olarak
"modaliteye özgü kodlama ve geç birleştirme"yi tercih etti. Biz ChangeMamba'yı olduğu gibi kullanıyoruz; bu
tartışmayı çözmek projenin konusu değil.

### 6.3 Füzyon: optik ile radarı birleştirmek

İki kodlayıcının özellikleri hasar çözücüsünde her ölçekte **birleştirilir** (concatenation, kanal boyutunda yan
yana koymak). Basit bir yöntemdir: Çözücü, "şu konumda optik şunu, radar bunu söylüyor" bilgisini birlikte alır ve
ikisinden nasıl sonuç çıkaracağını kendisi öğrenir.

Daha karmaşık füzyon yöntemleri var (çapraz dikkat, özellik hizalama, optik akış). Araştırma raporuna göre
bunların görülmemiş olayda yararı gösterilmedi. Örneğin FlowMamba, xBD'de ChangeMamba'yı yalnız 0,39 F1 geçip
%48 daha fazla hesap istiyor; Türkiye'nin kayıt hatası (iki görüntünün hizalama hatası) zaten 1 pikselin altında.

### 6.4 Ayrık (decoupled) başlıklar

ChangeMamba'nın en önemli tasarım kararı, iki alt görevi (Bölüm 4.3) **ayrı çözücülerle** çözmesi:

- **Bina çözücüsü** yalnız afet öncesi optiği görür. Bina nerede sorusunun cevabı afetten önce de bellidir; radar
  burada gürültüden başka bir şey katmaz.
- **Hasar çözücüsü** hem optiği hem radarı görür. Hasarı anlamak için ikisine de ihtiyaç var.

BRIGHT makalesi bu ayrımın genellemeye yardım ettiğini savunuyor: Zero-shot ortalamasında ayrık modeller 37,48–39,38
`[ZS-EVENT mIoU]`, doğrudan segmentasyon modelleri 34,13–36,42 aldı. Dikkat: Bu kontrollü bir deney değil. Ayrık
modeller aynı zamanda farklı kodlayıcı ailelerine sahip ve Türkiye'de ayrık DamageFormer (37,20) doğrudan UNet'in
(37,82) altında kalıyor. "Ayrımın yararı" kanıtlanmış değil, olası.

### 6.5 Çıktılar ve nihai harita

Model iki çıktı verir:

- `loc`: Her piksel için 2 sınıflı **logit** (bina/değil).
- `clf`: Her piksel için 4 sınıflı logit (arka plan, sağlam, hasarlı, yıkık).

**Logit**, olasılığa dönüştürülmemiş ham puandır. **Softmax** fonksiyonu logitleri toplamı 1 olan olasılıklara
çevirir. **argmax** en yüksek puanlı sınıfı seçer.

Bir ayrıntı başlık sonucumuz için kritik: ChangeMamba'nın eğitim kodu mIoU'yu `clf` çıktısının argmax'ından, **bina
maskesi uygulamadan** hesaplıyor. Makale ise nihai haritanın `loc` maskesiyle birleştirildiğini yazıyor. Bu iki
yol farklı sayılar verebilir. Biz başlık sonucunu **resmî kod yolundan** (`argmax(clf)`, maskesiz) hesaplıyoruz;
maskeli hâl önceden ilan edilmiş bir ikincil satır olacak. Bu karar Türkiye'ye dokunmadan önce kilitlenir (Bölüm 9).

### 6.6 Neden ChangeMamba?

BRIGHT makalesinin Tablo 10'una göre ChangeMamba dört deprem olayının dördünde de en yüksek zero-shot sonucu aldı:
Les Cayes 41,35, Türkiye **41,28**, Fas 42,86, Noto 39,90 `[ZS-EVENT mIoU]`. Aynı protokolde Türkiye'de ikinci
sıradaki ChangeOS 38,45'te kalıyor ve 41,28'i geçen yayımlanmış bir yöntem bulunamadı.

Ama seçimin asıl nedeni "en iyi skor" değil: **Tek doğrudan çıpaya karşı dürüst bir ölçüm yapabilmek.** Aynı
modeli aynı protokolle eğitirsek, sonucumuz yayımlanmış 41,28 ile doğrudan kıyaslanabilir.

### 6.7 Yedek model: DamageFormer

ChangeMamba'nın riski mühendislik tarafında. Seçici tarama işlemi için CUDA çekirdeğinin derlenmesi gerekiyor.
Colab'ın güncel ortamı (CUDA 13) ile bu derlemenin çalıştığı doğrulanmadı; kodun `setup.py` dosyası artık
desteklenmeyen eski bir GPU mimarisi (`compute_70`) için de derleme istiyor ve eski bir `timm` sürümüne bağlı.

Bu yüzden önceden kabul edilmiş, otomatik bir yedek kuralı var: Çekirdek Colab'da **en fazla 1 saatte** derlenip
20 adımlık bir ileri-geri testten geçmezse ana eğitim **Swin-T DamageFormer** ile yapılır. DamageFormer BRIGHT
deposunda hazır, özel çekirdek istemiyor ve aynı protokoldeki Türkiye sonucu 37,20 `[ZS-EVENT mIoU]`. Yedeğe
geçilirse hedef "41,28'i geçmek" değil, "37,20'yi açıkça geçmek ve 41,28'e yaklaşmak" olur.

Bu kuralın mantığı: Belirsiz süreli bir hata ayıklamaya kontrolsüz GPU bütçesi harcamak, önceden tanımlanmış bir
yedeğe geçmekten daha pahalıdır.

### 6.8 Kullanmadığımız modeller ve nedenleri

| Model | Neden kullanmıyoruz |
|---|---|
| **DINOv3** (Meta, uydu sürümü) | Lisansı askerî ve savaş amaçlı kullanımı yasaklıyor; görülmemiş olay hasar görevinde test edilmedi |
| **MaRS** (SAR temel modeli) | Lisansı üç kaynakta üç farklı; ön eğitim verisi 2024 sonuna kadar uzandığı için 2023 Türkiye sahnelerini görmüş olabilir |
| **`*_standard_ML_split.pth` checkpoint'leri** | Rastgele bölmede eğitildiler; Türkiye eğitimlerinde vardı |
| **Alan uyarlama (UDA) yöntemleri** | BRIGHT'ta Türkiye için beş UDA yönteminin beşi de basit tabanın altında kaldı |
| **Görüntü-dil modelleri** | Hasar seviyesi tahmininde henüz zayıflar |
| **Birden çok mimariyi eğitip kıyaslamak** | Bütçe ve protokol: iki tam eğitim sınırı (Bölüm 9.3) |

---

## 7. Eğitim: modeli nasıl öğreteceğiz?

Bu bölümdeki her ayar `configs/ana.yaml` dosyasında dondurulur. Ayarların çoğu BRIGHT ve ChangeMamba'nın resmî
değerleri; birkaçı düşük riskli eklemeler. Tablo [tasarim.md Bölüm 7](tasarim.md)'de.

### 7.1 Veri hazırlama: kırpma ve batch

1024 × 1024'lük kareler GPU belleğine 16'lık gruplar hâlinde sığmaz. Bu yüzden her adımda her kareden rastgele
**640 × 640**'lık bir parça kesilir ve 16 parça bir batch oluşturur. Bellek yetmezse 512 × 512 ve 8'lik batch'e
inilir (ChangeMamba'nın kendi betiği böyle).

Rastgele kırpmanın bir yan faydası var: Her adımda aynı karenin farklı bir bölümü görüldüğü için model ezberlemek
yerine genellemeye zorlanır.

### 7.2 Veri artırımı (augmentation)

Eğitim görüntülerini yapay olarak çeşitlendirmeye **artırım** denir. Biz yalnız iki tür kullanıyoruz:

- **Çevirme (flip):** Yatay ve dikey ayna görüntüsü.
- **90 derece döndürme (rot90).**

Uydu görüntüsünde "yukarı" diye bir şey yoktur; bir bina 90 derece döndürüldüğünde de bina olarak kalır. Bu yüzden
bu dönüşümler etiketleri bozmadan veriyi çoğaltır.

**Kritik ayrıntı:** Aynı dönüşüm **üç diziye birden** uygulanmalı: optik, SAR ve etiket. Optiği çevirip etiketi
çevirmezsek model yanlış konumları öğrenir. Bu, kolay yapılan ve fark edilmesi zor bir hatadır; bu yüzden ayrı bir
birim testi var.

**Neden renk/parlaklık (fotometrik) artırımı yok?** Resmî reçetede yok ve radarın parlaklığı fiziksel bir anlam
taşır (çift sekme parlaklığı hasar ipucudur). Rastgele parlaklık oynamak bu bilgiyi bozabilir.

### 7.3 Normalizasyon

Sinir ağları, girdileri yaklaşık "ortalama 0, standart sapma 1" aralığındayken daha iyi öğrenir. Bu yüzden her
piksel değerinden bir ortalama çıkarılır ve bir standart sapmaya bölünür.

- **Optik:** ImageNet'in ortalama ve standart sapmaları. Kodlayıcı ImageNet'le ön eğitildiği için girdinin ona
  benzer istatistiklere sahip olması gerekir.
- **SAR:** Eğitim kümesinin (13 olay) kendi ortalama ve standart sapması. İki bağımsız kaynak 0–1 ölçeğinde yaklaşık
  0,22 ortalama ve 0,15–0,18 standart sapma veriyor.

**Neden yalnız eğitim kümesi?** Türkiye görüntülerini istatistik hesabına katsaydık, Türkiye hakkında bir bilgi
(etiketsiz de olsa) modelin girdisine sızmış olurdu. Bu küçük gibi görünen bir **sızıntıdır** ve kural 1 bunu
yasaklar. Türkiye'nin istatistikleriyle yeniden ölçeklemenin ne değiştirdiği, ayrı bir deney olarak (Y3, Bölüm
12.3) ve ayrı bir tabloda ölçülecek.

### 7.4 Kayıp fonksiyonu: CE + Lovász

Modelin iki çıktısı var, dolayısıyla iki kayıp terimi ailesi var. Toplam kayıp:

```
Kayıp = CE_loc + CE_clf + 0,5 · Lovász_loc + 0,75 · Lovász_clf
```

**Çapraz entropi (cross-entropy, CE).** Sınıflandırmanın standart kaybı. Her piksel için doğru sınıfa verilen
olasılığın logaritmasının eksisidir. Model doğru sınıfa %99 olasılık verirse kayıp neredeyse 0'dır; %1 verirse çok
büyüktür. Her piksel eşit ağırlıkla katkı yapar.

Bunun bir sorunu var: Bizim veride piksellerin büyük çoğunluğu arka plan. CE'yi düşürmenin kolay yolu "her yere
arka plan demek"tir. Nadir sınıflar (hasarlı, yıkık) toplam kayıpta küçük bir pay alır.

**Lovász-Softmax kaybı.** Değerlendirmede kullandığımız IoU metriğini (Bölüm 8.1) doğrudan optimize etmeye çalışır.
IoU'nun kendisi türevlenemez (sayma işlemlerinden oluşur), bu yüzden gradyanla optimize edilemez. Berman ve
arkadaşları (2018), Lovász genişlemesi adlı matematiksel bir araçla IoU'nun **türevlenebilir bir vekilini** kurdu.
Lovász kaybı her sınıfı eşit önemde ele aldığı için nadir sınıfları CE'den daha iyi korur.

İkisini birlikte kullanmak yaygın bir uygulamadır: CE kararlı ve hızlı öğrenme sağlar, Lovász metriğe yakınlaştırır.
Katsayılar (1, 1, 0,5, 0,75) resmî değerlerdir.

**Ek ayrıntılar:**
- **Sınıf ağırlığı yok** ([1, 1, 1, 1]). Hasarlı sınıfı ağırlıklandırmak cazip görünür, ama Türkiye'de binaların
  %93'ü sağlam olduğu için yanlış alarmları artırabilir.
- **Lovász fp32 hassasiyetle** hesaplanır; içindeki sıralama işlemi düşük hassasiyette kararsızlaşabilir.
- **ignore_index = 255:** Nodata pikselleri etikette 255 ile işaretlenir ve kayba katılmaz.

### 7.5 Optimizer: AdamW

**Adam**, en yaygın kullanılan optimizer'dır. Her ağırlık için gradyanın hem ortalamasını (yön) hem de büyüklüğünü
(ne kadar oynak olduğunu) takip eder ve adımı buna göre ayarlar: Gradyanı tutarlı olan ağırlıklar emin adımlarla,
oynak olanlar temkinli adımlarla güncellenir.

**AdamW**, Adam'ın **ağırlık sönümü** (weight decay) eklenmiş doğru biçimidir. Ağırlık sönümü her adımda bütün
ağırlıkları biraz küçültür; bu, modelin aşırı karmaşık çözümlere kaçmasını (aşırı uyum, overfitting) frenler.
"W" harfi, sönümün gradyandan ayrı (decoupled) uygulandığını belirtir.

Değerler: öğrenme oranı **1e-4** (0,0001), ağırlık sönümü **5e-3** (0,005). İkisi de resmî.

### 7.6 Öğrenme oranı zamanlaması: ısınma + kosinüs

Öğrenme oranı eğitim boyunca sabit kalmaz:

```
lr
 │      ┌──────╮
 │     /        ╲
 │    /          ╲___
 │   /               ╲____
 │  /                     ╲________
 └─┴──────────────────────────────── adım
   1.500                       40.000
  ısınma        kosinüs düşüş
```

- **Isınma (warmup), ilk 1.500 adım:** Öğrenme oranı 0'dan hedef değere yavaşça çıkar. Başta ağırlıklar rastgele
  ve Adam'ın istatistikleri henüz oturmamıştır; büyük adımlar modeli bozabilir.
- **Kosinüs düşüş:** Sonra öğrenme oranı bir kosinüs eğrisi boyunca 1e-6'ya iner. Başta büyük adımlarla kabaca doğru
  bölgeye gidilir, sonda küçük adımlarla ince ayar yapılır.

Bu eklemeler BRIGHT'ta denenmemiş ama derin öğrenmede yerleşik, düşük riskli uygulamalardır (Google Tuning
Playbook).

### 7.7 bf16: karma hassasiyet

Bilgisayarlar sayıları belirli bir bit sayısıyla saklar. Standart **fp32** 32 bit kullanır. **bf16** (brain float
16) 16 bit kullanır: fp32 ile aynı **sayı aralığını** korur ama daha az **ondalık hassasiyet** taşır.

Faydası: Bellek kullanımı ve hesap süresi neredeyse yarıya iner; A100 GPU'lar bf16'yı donanımda hızlı çalıştırır.
Derin öğrenme küçük yuvarlama hatalarına dayanıklıdır. fp16'nın aksine bf16'da sayı taşması nadir olduğu için ek
"kayıp ölçekleme" hilesine gerek kalmaz. Hassasiyetin önemli olduğu yerler (Lovász kaybı gibi) fp32'de kalır;
bu karışık kullanıma **karma hassasiyet** (mixed precision) denir.

### 7.8 EMA: ağırlıkların hareketli ortalaması

Eğitim sırasında ağırlıklar her adımda biraz zıplar. **EMA** (exponential moving average), ağırlıkların zaman
içindeki yumuşatılmış bir kopyasını tutar:

```
θ_EMA ← 0,999 · θ_EMA + 0,001 · θ
```

Her adımda gölge kopya, gerçek ağırlıklara %0,1 yaklaşır. Sonuç, son birkaç bin adımın ortalaması gibi davranan
daha kararlı bir modeldir ve genellikle yeni veride biraz daha iyi genelleşir. Doğrulama ve kayıt noktası seçimi
EMA ağırlıklarıyla yapılır.

### 7.9 Gradyan kırpma

Arada bir, kötü bir batch çok büyük bir gradyan üretebilir ve tek bir adım modeli bozabilir. **Gradyan kırpma**,
gradyanın toplam büyüklüğü (normu) bir eşiği aşarsa onu eşiğe ölçekler. Eşiği tahmin etmek yerine ölçüyoruz: Ön
kontrolde 2 katı öğrenme oranıyla 500 adım koşulur ve gradyan normlarının %90'lık dilimi (p90) eşik olarak alınır
(kapı 9, Bölüm 10).

### 7.10 Adım sayısı ve `max_iters` tuzağı

Resmî tarif 50.000 adım. Biz **40.000 adım** (640.000 örnek) planlıyoruz, çünkü 50.000 adım Colab Pro'nun aylık
bütçesine sığmıyor (Bölüm 15). Bu bir yargıdır; sonucu biraz etkileyebilir.

Burada sinsi bir tuzak var: BRIGHT kodunda `max_iters` parametresi **örnek** sayısıdır (800.000 örnek / 16 batch =
50.000 adım). ChangeMamba kodunda ise aynı adlı parametre **adım** sayısıdır. Biri ötekinin anlamıyla okunursa
eğitim 16 kat kısa ya da uzun olur. Araştırma sırasında iki ayrı not aynı betiği 20 bin ve 80 bin adım diye farklı
okudu. Bu yüzden ön kontrolde "veri yükleyicinin uzunluğu hedef adım sayısına eşit mi?" diye bir kapı var (kapı 3).

### 7.11 Tekrarlanabilirlik: tohum ve sonsuz örnekleyici

Derin öğrenmede birçok şey rastgeledir: Başlangıç ağırlıkları, kırpma konumları, artırım, örneklerin sırası.
Bunların hepsi bir **tohum** (seed) sayısından türetilen sözde rastgele sayılarla belirlenir. Aynı tohum aynı
sonucu verir.

Bizim örnekleyicimiz **sonsuz** ve **adım bazlı**: `seed + step` ile adım numarasından o adımın örneklerini üretir.
Faydası: Eğitim yarıda kesilip 25.000. adımdan devam ettiğinde, örnekleyici baştan 25.000 adım "boşa" çalışmak
zorunda kalmaz; doğrudan 25.000. adımın örneklerini üretir ve sonuç kesintisiz bir eğitimle aynı olur.

### 7.12 Kayıt noktası (checkpoint) ve kopmaya dayanıklılık

Colab oturumları en fazla 12 saat sürer ve bağlantı kopabilir. 9–14 saatlik bir eğitimin en az bir kez kesileceğini
varsayıyoruz. Bu yüzden:

- **Her 1.000 adımda** Google Drive'a kayıt (son 2 ve en iyi).
- **Her 5.000 adımda** Hugging Face Hub'daki özel bir depoya yedek.
- **Atomik yazma:** Dosya önce `*.tmp` adıyla yazılır, tamamlanınca tek bir işlemle (`os.replace`) asıl adına
  taşınır. Yazma sırasında kopma olursa eski kayıt bozulmaz.
- **Tam durum:** Yalnız model ağırlıkları değil; EMA kopyası, optimizer'ın iç istatistikleri, öğrenme oranı
  zamanlayıcısı, adım sayacı ve **bütün rastgele sayı üreteçlerinin durumu** kaydedilir. BRIGHT'ın kendi `--resume`
  seçeneği yalnız model ağırlıklarını yüklüyor; bu, devam eden eğitimin kesintisiz eğitimden farklı olması demek.

### 7.13 Doğrulama ve kayıt noktası seçimi

Eğitim sırasında modelin gerçekten iyileşip iyileşmediğini görmek için **doğrulama kümesi** kullanılır: Modelin
eğitimde görmediği ama Türkiye olmayan örnekler (13 olayın doğrulama listeleri). Her 2.000 adımda EMA modeli bu
kümede değerlendirilir ve en yüksek mIoU'yu veren kayıt noktası "en iyi" olarak saklanır.

Dürüst bir uyarı: Bu seçim zayıf bir seçimdir. 2026 CVPR yarışmasında doğrulama sıralaması ile test sıralaması
arasındaki korelasyon yalnız 0,35'ti. Yani doğrulamada en iyi olan model yeni olayda en iyi olmayabilir. Daha iyi
bir yol (bazı olayları sırayla dışarıda bırakmak) birden çok eğitim gerektirdiği için bütçeye sığmıyor. Bu zayıflık
raporda açıkça yazılacak.

**Eğitim kodunda Türkiye'yi değerlendirmenin yolu yok.** Resmî ChangeMamba betiği her 500 adımda test olayını da
yazdırıyor; bu, insanın farkında olmadan Türkiye sonucuna bakarak karar vermesine yol açar. Bizim kodda kapalı.

---

## 8. Değerlendirme: ne kadar iyi olduğunu nasıl ölçeceğiz?

### 8.1 Karışıklık matrisi, IoU ve mIoU

Her pikseli tahmin ve gerçek sınıfına göre bir tabloya yazarız. Bu tabloya **karışıklık matrisi** (confusion
matrix) denir:

```
                    TAHMİN
                arka  sağlam  hasarlı  yıkık
GERÇEK  arka     ■■■    ·       ·       ·
        sağlam    ·    ■■■      ·       ·
        hasarlı   ·     ■       ■       ·
        yıkık     ·     ·       ■       ■■
```

Köşegen doğru tahminleri, köşegen dışı karışıklıkları gösterir (yukarıdaki örnekte hasarlılar sık sık sağlam
sanılıyor).

**IoU** (Intersection over Union, kesişim bölü birleşim) bir sınıf için şöyle hesaplanır:

```
IoU = (doğru tahmin edilen pikseller) / (tahmin edilen ∪ gerçek pikseller)
    = TP / (TP + FP + FN)
```

- **TP** (doğru pozitif): Gerçekte yıkık, model de yıkık dedi.
- **FP** (yanlış pozitif): Gerçekte değil, model yıkık dedi.
- **FN** (yanlış negatif): Gerçekte yıkık, model başka bir şey dedi.

IoU 1 ise kusursuz örtüşme, 0 ise hiç örtüşme yok demektir. **mIoU** (mean IoU), dört sınıfın IoU'larının
ortalamasıdır. Bizim başlık metriğimiz budur.

**Neden doğruluk (accuracy) değil?** Piksellerin büyük kısmı arka plan, binaların %93'ü sağlam. "Her yere arka plan
veya sağlam de" diyen işe yaramaz bir model bile %95'in üstünde doğruluk alır. IoU her sınıfa eşit ağırlık verdiği
için bu tuzağa düşmez. Benzer şekilde **ağırlıklı F1** da yanıltıcıdır: CS437 projesinde aynı model ağırlıklı F1 ile
0,90, makro F1 ile 0,44 aldı. Bu yüzden anti-hedeflerimizden biri, bu metrikleri başlık yapmamak.

### 8.2 Olay düzeyi ve küme düzeyi

mIoU iki şekilde hesaplanabilir:

- **Olay düzeyi:** Bir olayın bütün kareleri için tek bir karışıklık matrisi biriktirilir, IoU onun üstünden
  hesaplanır.
- **Küme düzeyi:** Test kümesinin tamamı (farklı olaylar bir arada) tek bir matrise atılır.

Bir başka ayrım: Matris olay boyunca mı biriktiriliyor, yoksa her karenin IoU'su ayrı hesaplanıp kareler mi
ortalanıyor? Bu seçimler farklı sayılar verir. 41,28 olay düzeyinde bir sayıdır; bizim hesabımızın da **tam olarak
aynı yoldan** gitmesi gerekir. Bu yolun bütün ayrıntıları (biriktirme, arka planın ortalamaya girip girmediği,
nodata, tam kare mi kayan pencere mi) Faz 0'da `docs/protokol.md`'ye yazılacak ve Faz 3'teki yeniden üretim bunu
sınayacak.

### 8.3 41,28 ne anlama geliyor?

41,28 düşük görünebilir. Bir tahminle ne demek olduğunu anlayalım: Arka plan IoU'su genellikle 96–97 civarındadır
(arka planı bulmak kolaydır). Dört sınıfın ortalaması 41,28 ise üç bina sınıfının ortalaması kabaca **23 IoU**
demektir. Karşılaştırma için DFC25 birincisinin (Türkiye eğitimdeyken, farklı bir test kümesinde) sınıf kırılımı:
arka plan 98,0, sağlam 67,3, **hasarlı 9,7**, yıkık 42,1.

Buradan iki ders çıkıyor:

1. Görev gerçekten zor; özellikle "hasarlı" sınıfı neredeyse çözülmemiş bir problem.
2. Tek bir mIoU sayısı çok şeyi saklıyor. Bu yüzden **sınıf başına IoU** raporlamak zorunlu.

Gerçekçi beklentimiz 41,28 civarı, onu belirgin biçimde aşmak değil.

### 8.4 Bina düzeyi metrikler

Piksel metrikleri büyük binalara daha çok ağırlık verir (daha çok pikselleri var). Oysa her bina, içinde yaşayan
insanlar açısından eşit önemdedir. Bu yüzden **bina düzeyi** ölçüm de yapacağız:

1. Etiketteki her binayı ayrı bir nesne olarak çıkar.
2. Her binaya, içindeki piksellere göre tek bir tahmin sınıfı ata.
3. Bina sayıları üzerinden **makro F1** hesapla.

**F1**, kesinlik (precision: "yıkık dediklerimin kaçı gerçekten yıkık?") ile duyarlılığın (recall: "gerçek
yıkıkların kaçını buldum?") harmonik ortalamasıdır. **Makro** F1, her sınıfın F1'ini ayrı hesaplayıp eşit ağırlıkla
ortalar; nadir sınıfları ezmez.

Açık bir soru: BRIGHT'ın etiketi raster (piksel haritası); bina sınırları ayrı bir dosyada olmayabilir. Bitişik
binalar piksel haritasında birleşik görünür. Yoğun Antakya dokusunda bu, birkaç binanın tek bina sayılması demek.
BRIGHT'ın örnek (instance) etiketleri olup olmadığı Faz 2'de araştırılacak.

### 8.5 Test zamanı artırımı (TTA) ve bina oylaması

**TTA** (test-time augmentation): Aynı görüntü modele birkaç farklı hâliyle (çevrilmiş, döndürülmüş) verilir,
tahminler geri çevrilip ortalanır. Model her yönde biraz farklı hata yaptığı için ortalama daha kararlıdır.

**Bina oylaması:** Bir binanın içindeki piksellerin tahminleri çoğunluk oyuyla tek bir sınıfa indirilir. "Çatının
yarısı hasarlı, yarısı yıkık" gibi tutarsızlıkları giderir.

BRIGHT makalesinde TTA +0,56, nesne oylaması +0,69, ikisi birlikte +0,92 puan kazandırdı (olay düzeyi). Bunlar
başlık sonucuna değil, **önceden ilan edilmiş ikincil satırlara** girer. Neden? Başlık sonucu 41,28 ile aynı
koşullarda (TTA'sız) olmalı ki kıyas adil olsun.

### 8.6 Güven aralığı ve bootstrap

Diyelim modelimiz 42,1 aldı. Bu 41,28'den iyi mi? Cevap: Belki. Test kümesi biraz farklı olsaydı sonuç da biraz
farklı olurdu. Bu belirsizliği ölçmenin yolu **güven aralığı**dır.

**Bootstrap**, güven aralığını formül olmadan, bilgisayarla hesaplamanın yoludur:

1. Test kümesinden, yerine koyarak, aynı büyüklükte rastgele bir örneklem çek (bazı kareler iki kez, bazıları hiç
   seçilmez).
2. Metriği bu örneklemde hesapla.
3. Bunu binlerce kez tekrarla.
4. Sonuçların %2,5 ve %97,5'lik dilimleri, %95'lik güven aralığını verir.

İki önemli incelik:

- **Blok bootstrap.** Komşu kareler birbirine benzer (aynı mahalle, aynı hasar deseni). Kareleri bağımsızmış gibi
  yeniden örneklersek belirsizliği olduğundan küçük gösteririz. Bu yüzden örnekleme birimi tek tek kareler değil,
  **mekânsal bloklar ya da şehirler** olacak.
- **Eşleştirilmiş bootstrap.** İki modeli (ana model ve varyant) kıyaslarken, her yeniden örneklemde **iki modelin
  farkı** hesaplanır. Aynı kareler iki model için de kullanıldığı için bu yöntem farkları çok daha hassas ölçer.

**"Eşit" kuralı.** Fark gürültüden ayrılamıyorsa "eşit" yazılır. Belgelerde şimdilik "1–2 puanın altı" yazıyor,
ama bu bir aralık; sonuç görüldükten sonra 1'i mi 2'yi mi seçeceğimiz tartışmaya açık kalır. Bu yüzden protokolde
tek bir kural kilitlenecek (ör. "eşleştirilmiş farkın %95 aralığı 0'ı içeriyorsa veya fark 1,0 puanın altındaysa
eşit").

### 8.7 Metrik etiketleri: elmayla armudu karıştırmamak

Literatürde aynı veri setiyle elde edilmiş ama farklı koşullarda hesaplanmış çok sayıda "mIoU" var. Bunlar
kıyaslanamaz. Bu yüzden projede her sayının yanına türünü yazıyoruz (kural 4):

| Etiket | Ne ölçer | Örnek |
|---|---|---|
| `[ZS-EVENT mIoU]` | 13 olayla eğitim, Türkiye'nin bütün karelerinde olay düzeyi | ChangeMamba 41,28 |
| `[STD-EVENT mIoU]` | Rastgele bölme (Türkiye eğitimde), olay düzeyi | ChangeMamba 52,70 |
| `[STD-SET mIoU]` | Rastgele bölme, test kümesinin tamamı | ChangeMamba 67,63 |
| `[DFC-TEST mIoU]` | DFC25 yarışma kurgusu (Türkiye eğitimde) | 54,28 |
| `[CVPR-TEST segm_AP]` | Örnek segmentasyonu, farklı görev | 0,182 |

67,63'ü görüp "41,28 kötüymüş" demek, ya da birinin "Türkiye'de 54 aldık" cümlesini 41,28 ile kıyaslamak klasik
hatalardır. **Yalnız `[ZS-EVENT mIoU]` 41,28 ile kıyaslanır.**

---

## 9. Bilimsel dürüstlük: protokol ve kilitler

Bu bölüm projenin omurgası. Burada anlatılanlar, projeyi "bir model eğittim" demonstrasyonundan güvenilir bir
ölçüme dönüştürüyor.

### 9.1 Zero-shot bölme ve sızıntı türleri

**Zero-shot** (sıfır örnek), modelin test olayından tek bir etiketli örnek bile görmeden sınanması demektir. BRIGHT'ın
`cross_event/zero_shot` kurgusunda test olayı (`turkey-earthquake`) eğitim ve doğrulama listelerinden tamamen
çıkarılır ve bütün 1.114 karesi teste gider.

Buna karşılık **`standard_ML`** bölmesi rastgeledir ve Türkiye'nin kareleri eğitimde (777), doğrulamada (113) ve
testte (224) dağılmıştır. Araştırma notlarından biri yanlışlıkla bu bölmeyi önermişti; bizim için doğrudan bir
sızıntı olurdu.

**Sızıntı** (leakage), test verisi hakkındaki bilginin herhangi bir yoldan eğitime ya da karar sürecine
karışmasıdır. Projede dikkat ettiğimiz sızıntı yolları:

| Yol | Örnek | Önlem |
|---|---|---|
| Eğitim verisi | Türkiye kareleri eğitimde | Zero-shot bölme; bölme üreticisi Türkiye'yi görürse hata verir |
| Başlangıç ağırlığı | Türkiye'yi görmüş checkpoint | Yalnız ImageNet `vssm_tiny` (kural 2) |
| Ön eğitim verisi | 2023 Türkiye sahnelerini görmüş temel model | MaRS gibi modeller dışarıda |
| Normalizasyon | Türkiye'nin istatistikleri | SAR istatistikleri yalnız eğitim kümesinden |
| Model seçimi | Türkiye'de en iyi kayıt noktasını seçmek | Seçim yalnız 13 olayın doğrulamasıyla |
| Eşik ve ayar | Sıcaklık, eşik, TTA ayarını Türkiye'de seçmek | Bütün ayarlar 13 olayda |
| İnsan kararı | Türkiye sonucunu görüp bir şeyi değiştirmek | Git kilitleri (aşağıda) |

Son satır en sinsi olanıdır. Kimse kasıtlı hile yapmaz; ama sonucu gördükten sonra "şu ayarı biraz değiştirsek"
demek de testi eğitime çevirir.

### 9.2 Önceden kayıt ve git kilitleri

Bilimde **önceden kayıt** (pre-registration), deneyin nasıl yapılacağını ve nasıl değerlendirileceğini sonuçtan
önce yazıp kilitlemektir. Böylece sonradan "aslında şunu ölçmek istemiştik" deme imkânı kalmaz.

Projede bunu **git etiketleriyle** yapıyoruz:

- **`protokol-kilidi`:** Yapılandırma, eşikler, metrik yolu ve protokol belgesi bu etiketli commit'e girer.
  Türkiye'de değerlendirme yapan `final` adımı bu etiket olmadan **çalışmaz**. Kod, etiketi ve çalışma ağacının
  temiz olduğunu (commit edilmemiş değişiklik yok) kendisi denetler.
- **`varyant-kilidi`:** İkinci modelin (varyant) tasarımı, ana modelin Türkiye sonucu görülmeden bu etiketle
  kilitlenir.

Git etiketi bir zaman damgası gibidir: Herkes, kararların sonuçtan önce verildiğini commit geçmişinden doğrulayabilir.

### 9.3 Neden yalnız iki tam eğitim?

Bunun iki nedeni var:

1. **Bütçe.** Bir tam eğitim 50–75 CU tutuyor (Bölüm 15).
2. **Çatallanan yollar.** On farklı model eğitip Türkiye'de en iyisini raporlarsak, sonuç "görülmemiş depremde
   başarı" değil, "on denemeden şanslı olanı" olur. Her ek deneme, testi biraz daha eğitim verisine çevirir.

Bu yüzden: Ana model (ChangeMamba olduğu gibi) ve **tek fikirli** bir varyant. Tek fikir, çünkü bir eğitimde iki
değişiklik yaparsak hangisinin işe yaradığını ayıramayız. İki modelin ikisi de raporlanır; Türkiye sonucuna bakarak
aralarından seçim yapılmaz. Yeni bir tam eğitim eklemek kullanıcının kararıdır (kural 3).

### 9.4 İndüktif ve transdüktif

- **İndüktif** yöntem, test verisini hiç görmeden genel bir kural öğrenir ve sonra uygular. Başlık sonucumuz böyle.
- **Transdüktif** yöntem, test verisinin etiketsiz hâlini de kullanır (ör. Türkiye görüntülerinin istatistiklerine
  göre ayar yapmak, Türkiye görüntülerinde modelin kendi tahminlerini etiket gibi kullanmak).

Transdüktif yöntemler gerçek bir afette uygulanabilir (görüntü elimizdedir), bu yüzden meşrudur. Ama 41,28 indüktif
bir sayı olduğu için onunla kıyaslanamaz. Projede transdüktif sonuçlar (Y3, öncül düzeltmesi C1, olası pseudo-label
D6) **ayrı bir "transdüktif" tabloda** verilir.

### 9.5 Yeniden üretim: hattın doğru olduğunun kanıtı

Kendi eğitimimize başlamadan önce kendi veri ve metrik kodumuzun doğru olduğunu nasıl bileceğiz? BRIGHT ekibi,
Türkiye zero-shot için eğittiği beş modelin ağırlıklarını Zenodo'da yayımladı (SiamCRNN, UNet, SiamAttnUNet,
DeepLabV3+, DamageFormer). Bu modelleri **eğitmeden**, yalnız kendi hattımızdan geçirip Türkiye'de çalıştırırız.
Sonuç makaledeki 38,32 / 37,82 / 37,20 değerlerine ±0,3 içinde çıkmalı.

Eğitim yok, rastgelelik yok; sonuç belirlenimci olmalı. Tutmuyorsa hata modelde değil, veri okumada ya da metrik
hesabındadır ve ana eğitim başlamaz. Bu, bütün projenin en güçlü kontrolüdür ve rakip projelerin hiçbiri yapmıyor.
(Bu modellerin Türkiye çıktısını üretmek kural 1'in bilinçli bir istisnasıdır; seçim veya ayar için kullanılmaz.)

### 9.6 Olumsuz sonuç da sonuçtur

Varyant ana modeli geçmezse, Y3 hiçbir şeyi değiştirmezse, sonuç 41,28'in altında kalırsa, bunların hepsi
raporlanır (kural 6). Kilitli protokolle elde edilmiş, sınıf kırılımlı ve güven aralıklı bir 40, gevşek protokolle
elde edilmiş bir 44'ten daha çok şey söyler.

---

## 10. Ön kontrol kapıları

GPU saati pahalıdır. Bir eğitimin 10 saat sonra bir hata yüzünden çöpe gittiğini öğrenmek, en kötü senaryodur. Bu
yüzden ana eğitimden önce toplam 1–1,5 saat süren 11 kontrol var. Her biri belirli bir hata türünü yakalar.

| # | Kapı | Yakaladığı hata |
|---|---|---|
| 1 | Hazır checkpoint'lerle 38,32 / 37,82 / 37,20'yi yeniden üretmek | Veri okuma veya metrik hatası |
| 2 | Kare sayıları ve olay listeleri; eğitim ve doğrulamada tek bir Türkiye karesi yok | Yanlış bölme, sızıntı |
| 3 | Veri yükleyici uzunluğu = hedef adım sayısı | `max_iters` tuzağı |
| 4 | Başlangıç kayıpları: hasar başlığında ln 4 ≈ 1,386, bina başlığında ln 2 ≈ 0,693 | Yanlış kayıp kurulumu, bozuk etiket |
| 5 | Artırım kapalıyken 8 kare 300–500 adımda ezberlenmeli | Modelin öğrenemiyor olması (bağlantı hatası, yanlış lr) |
| 6 | SAR sıfırlanınca hasar çıktısı değişmeli | SAR dalının modele hiç bağlanmamış olması |
| 7 | Metrik uç durumları: kusursuz tahmin 1,0, hep arka plan yaklaşık 0,21 | Metrik kodu hatası |
| 8 | 200 adımda hız, tepe VRAM ve CU projeksiyonu | Bütçeyi aşacak bir eğitim |
| 9 | 2× lr ile 500 adımda p90 gradyan normu | Gradyan kırpma eşiği |
| 10 | 60. adımda süreç öldürülür, devam testi yapılır | Eksik kayıt durumu |
| 11 | ChangeMamba çekirdek derleme kapısı (en fazla 1 saat) | Ortam uyumsuzluğu; yedeğe geçiş |

Birkaçının arkasındaki mantık:

**Kapı 4 neden ln 4?** Rastgele başlatılmış bir model dört sınıfa yaklaşık eşit olasılık (%25) verir. Doğru sınıfa
%25 olasılık veren bir tahminin çapraz entropisi `-ln(0,25) = ln 4 ≈ 1,386`'dır. Başlangıç kaybı bundan çok
farklıysa bir şeyler yanlış kurulmuştur.

**Kapı 5 neden ezberleme?** Sağlıklı bir model, 8 görüntüyü kolayca ezberleyebilmelidir. Bunu bile yapamıyorsa
öğrenme mekanizmasında bir kopukluk vardır. Andrej Karpathy'nin "eğitim tarifi"nden gelen klasik bir kontrol.

**Kapı 7 neden 0,21?** Model her yere arka plan derse arka plan IoU'su yüksek (yaklaşık 0,85), diğer üç sınıfın
IoU'su 0 olur; ortalama yaklaşık 0,21 eder. Metrik kodu bu değeri vermiyorsa hatalıdır.

Sonuçlar `preflight.json` dosyasına yazılır. Hepsi geçmeden ana eğitim başlamaz.

---

## 11. Belirsizlik ve kalibrasyon

### 11.1 Modelin ne kadar emin olduğu

Model her piksel için yalnız bir sınıf değil, bir olasılık dağılımı verir ("%70 yıkık, %25 hasarlı, %5 sağlam").
Bu olasılıklar ne kadar güvenilir?

**Kalibrasyon**, modelin "%70 eminim" dediği durumların gerçekten yaklaşık %70'inde haklı çıkmasıdır. Derin
öğrenme modelleri genellikle **aşırı özgüvenlidir**: %95 eminim der ama %80 haklı çıkar. Yeni bir olayda (alan
kayması altında) bu daha da kötüleşir.

### 11.2 ECE

**ECE** (Expected Calibration Error, beklenen kalibrasyon hatası), kalibrasyonu ölçen standart sayıdır. Tahminler
güven düzeyine göre kutulara ayrılır (%0–10, %10–20, ...). Her kutuda ortalama güven ile gerçek doğruluk arasındaki
fark hesaplanır ve kutuların büyüklüğüne göre ağırlıklı ortalaması alınır. 0 kusursuz kalibrasyondur. Enkaz
projesinde ECE, eğitim şehrinde 0,01 iken görülmemiş şehirde 0,08–0,12'ye çıktı.

### 11.3 Sıcaklık ölçekleme

Aşırı özgüveni düzeltmenin en basit yolu **sıcaklık ölçekleme** (temperature scaling): Logitler tek bir T sayısına
bölünür. T > 1 olasılıkları yumuşatır (model daha temkinli olur), T < 1 keskinleştirir. T, doğrulama kümesinde
(13 olay) en iyi kalibrasyonu verecek şekilde seçilir. Türkiye'de yalnız **ölçülür**, ayarlanmaz.

### 11.4 TTA entropisi

**Entropi**, bir olasılık dağılımının ne kadar "kararsız" olduğunu ölçer: Tek sınıfa %100 verilirse entropi 0,
dört sınıfa eşit dağıtılırsa en yüksektir. TTA'da (Bölüm 8.5) modelin farklı çevirmelerdeki tahminleri birbirini
tutmuyorsa ortalama tahminin entropisi yüksek çıkar. Bu, "model burada kararsız" demenin pratik bir ölçüsüdür ve
bina düzeyinde belirsizlik puanı olarak kullanılacak.

### 11.5 Risk-kapsam eğrisi ve AURC

Bir model "emin olmadığım durumlarda karar vermeyeyim, insana bırakayım" diyebilseydi ne olurdu? Bu soruya
**seçici tahmin** (selective prediction) denir.

- **Kapsam** (coverage): Modelin karar verdiği örneklerin oranı.
- **Risk**: O örneklerdeki hata oranı.

Model en emin olduğu %10'luk kısımda karar verirse risk düşüktür; %100'ünde karar verirse risk en yüksektir. Bu
ilişkiyi çizen eğriye **risk-kapsam eğrisi**, altındaki alana **AURC** denir. Küçük AURC, modelin "neyi bilmediğini
bildiği" anlamına gelir.

Projede bu fikir operasyonel bir soruya dönüşüyor (Y2, Bölüm 12.2).

---

## 12. Yenilik analizleri

Projenin araştırma sorusu yalnız "kaç puan aldık?" değil. Planlı dört yenilik ve araştırmada çıkan üç aday analiz
var. Hepsinin ortak özelliği: Hiçbiri ek tam eğitim istemiyor (Y4 hariç).

### 12.1 Y1: Referans anlaşmazlığı haritası

**Soru:** Aynı binalarda farklı "doğru cevap" kaynakları ne kadar anlaşıyor; model kime göre yanlış?

Üç kaynak:

1. **BRIGHT etiketi:** Uzmanların optik yorumu.
2. **HOT/OSM yıkık bina katmanı:** OpenStreetMap gönüllülerinin (Humanitarian OpenStreetMap Team koordinasyonunda)
   Copernicus ve İTÜ'nün hava görüntülerine bakarak işaretlediği yıkık binalar. Bu da optik bir yorumdur; BRIGHT
   etiketinden bağımsız değildir. Muhtemelen yalnız "yıkık" sınıfını içerir.
3. **ARIA Hasar Vekil Haritası (DPM):** NASA ve Singapur Yer Gözlemevi'nin Sentinel-1 radar koheransından
   (Bölüm 2.2), öğrenme kullanmadan ürettiği 30 metrelik hasar haritası. Fiziksel bir sinyal; insan yorumu yok.

Bu yüzden çerçeve "üç bağımsız referans" değil, **"iki optik yorum + bir radar sinyali"**. Neden ilginç? Saha
sayımına karşı ölçüldüğünde ARIA DPM bina F1 0,47 aldı; Copernicus'un elle yaptığı optik haritalar 0,15–0,23'te
kaldı. Yani optik yorum, sanıldığı kadar güvenilir bir "doğru cevap" değil. Modelimiz BRIGHT etiketine göre
"yanlış" olduğu bazı yerlerde radar sinyaline göre haklı olabilir.

**Yapılacaklar:** Üç katmanı aynı binalara bağlamak. Bunun teknik zorlukları var: OSM poligonları ile uydu görüntüsü
arasında birkaç metrelik kayma olabilir; DPM'in 30 m'lik pikseli onlarca binayı kapsar; tarihler farklıdır. Bu
kuralların hepsi önceden yazılacak (Y1 birleştirme şartnamesi). Anlaşmazlık tablosunun yanında 100–200 bina elle
denetlenecek.

### 12.2 Y2: Analist bütçeli önceliklendirme

**Soru:** Bir uzman binaların yalnız %5, %10 veya %20'sini kontrol edebiliyorsa, model hangilerini göstermeli ve
gerçek hasarlı ve yıkık binaların yüzde kaçı yakalanır?

Afet müdahalesinde gerçek kısıt analist zamanıdır. Bu analiz belirsizliği soyut bir sayıdan ("ECE şu") operasyonel
bir sayıya ("%10'luk bütçeyle yıkıkların %X'i yakalanır") çevirir.

Binalar modelin hasar olasılığına ya da belirsizliğine göre sıralanır ve en üstteki %k incelenmiş sayılır. Kıyas
tabanları:

- **Rastgele sıra:** Hiç model kullanmasaydık.
- **DPM sırası:** Öğrenmesiz radar haritasına göre sıralasaydık. Model bu tabanı geçemiyorsa bu da bir bulgudur.

Benzer bir çerçeve sel hasarı için ALTIS çalışmasında kullanıldı; deprem, zero-shot ve VHR SAR için bulunamadı.

### 12.3 Y3: Etiketsiz SAR uyarlaması

**Soru:** Türkiye'nin yalnız SAR görüntü istatistikleriyle (etiket yok, gradyan yok) girdi yeniden standartlaştırılırsa
ne değişir?

Bölüm 7.3'te SAR'ı eğitim kümesinin istatistikleriyle normalize ettik. Ama Türkiye'nin radar görüntüleri (farklı
uydular, farklı bakış açıları, farklı yapı stoku) başka bir dağılıma sahip olabilir. BRIGHT makalesi SAR hasar
izinin olaydan olaya kaydığını gösteriyor. Y3, Türkiye'nin kendi ortalama ve standart sapmasını kullanmanın etkisini
ölçer. Fikir, alan uyarlamasındaki **AdaBN** tekniğine benzer ve yeni değildir; değeri ölçümdedir. Transdüktif
olduğu için ayrı tabloda raporlanır.

### 12.4 Y4: Tek fikirli varyant

İkinci tam eğitimde ChangeMamba'ya **tek bir** değişiklik eklenir. Fikir şimdi seçilmedi; ana eğitimden sonra, yalnız
13 olayın doğrulama sonuçlarına bakılarak birinci karar noktasında (KN1) seçilecek. Adaylar Bölüm 13'te.

### 12.5 Aday analizler: N1–N3

Araştırmada önerilen ana araştırma sorusu şu:

> Görülmemiş bir depremde model sıralaması, hangi referansa ve hangi istatistiğe baktığına göre değişiyor mu?

- **N1, yayımlanmış sıralamanın güven aralıkları:** Zenodo'daki 5 model ve bizim modelimiz için blok ve eşleştirilmiş
  bootstrap. 41,28 / 38,32 / 37,82 / 37,20 farklarından hangileri gürültüden ayrılabiliyor? BRIGHT'ın tablosu tek
  koşudan geliyor ve bunu ölçen bulunamadı.
- **N2, referansa göre sıralama:** Aynı 6 model üç referansa karşı ayrı ayrı. Şampiyon referansa göre değişiyor mu?
- **N3, sarsıntı kırılımı:** USGS ShakeMap'in yer ivmesi (PGA) katmanına göre hataların dağılımı. Model hataları
  sarsıntı şiddetiyle mi açıklanıyor, görüntü özellikleriyle mi?

Bunlar yanlışlanabilir sorulardır: "Sıralama değişmiyor" da yayımlanabilir bir bulgudur.

### 12.6 SAR hata galerisi

Hataların nerede yoğunlaştığını gösteren görsel bir inceleme:

- **Capella ve Umbra kırılımı:** İki uydu farklı bakış açıları ve işleme zincirleri kullanır; model birinde daha mı
  kötü?
- **Layover ve gölge:** Yüksek binaların bindirme ve gölge alanlarında hata artıyor mu? (Bölüm 2.2)
- **İzole hasarlı binalar:** Komşularından farklı durumda olan binalar. Literatüre göre bunların yalnız %7–11'i
  bulunuyor; yani modeller hasarı tek binada değil, "hasarlı mahalle" bağlamında öğreniyor olabilir (kısa yol
  öğrenme, Bölüm 13.5).

---

## 13. Fikir havuzundaki teknikler

Bu bölümdeki teknikler şimdi uygulanmayacak; [fikir-havuzu.md](fikir-havuzu.md)'de bekliyorlar ve ana eğitimden sonra
karar noktalarında seçilecekler. Burada ne olduklarını ve neden düşünüldüklerini anlatıyorum.

### 13.1 Hasarın altı izi

"Hasarlı bina bir yerden mutlaka bulunur" sezgisi kısmen doğru. Hasarın uzaydan görülebilen altı izi var ve
modelimiz bunların yalnız birini kullanıyor:

| İz | Neyle görülür | Modelde var mı? |
|---|---|---|
| Radar yansımasının değişmesi (çift sekme kaybı) | VHR SAR | Var (ama öncesi radar yok) |
| Radar koheransının kaybı | Sentinel-1 | Yok |
| Yükseklik kaybı | Stereo optik, LiDAR | Yok |
| Yer sarsıntısının şiddeti | ShakeMap | Yok |
| Binanın kırılganlığı (yaş, kat, yapı türü) | Kadastro, OSM | Yok |
| Komşuluk | Modelin kendi tahminleri | Dolaylı |

İyileştirme fikirlerinin çoğu, modele göremediği bir izi göstermeye ya da gördüğünü daha iyi kullanmaya dayanıyor.

### 13.2 Yeni fiziksel kanıt eklemek

**A1, Sentinel-1 koherans kanalı.** 14 olayın her biri için afet öncesi ve sonrası Sentinel-1 çiftlerinden koherans
değişimi hesaplanır (Alaska Uydu Tesisi'nin HyP3 servisi bunu ücretsiz yapıyor), BRIGHT karelerinin koordinatına
yeniden örneklenir ve hasar kodlayıcısına ek bir kanal olarak verilir. 30 m çözünürlük tek binayı ayıramaz; daha
çok "bu bölgede bir şeyler çöktü" diyen bir bölgesel öncül gibi çalışır. Bu birleşimi yapan bir çalışma
bulunamadı; bu yüzden en yenilikçi aday. Riski veri işinin büyüklüğü: Her olay için uygun bir Sentinel-1 çifti
bulunması gerekiyor. Faz 2'de 2–3 günlük bir **fizibilite kontrolü** var.

**A2, ShakeMap ile Bayesçi son işlem.** USGS, büyük depremlerden saatler sonra yer ivmesi haritası yayımlar.
Sarsıntı ne kadar güçlüyse hasar olasılığı o kadar yüksektir. **Bayesçi** düzeltme, modelin görüntüden çıkardığı
olasılığı bu "önsel bilgiyle" (prior) birleştirir. Katsayılar yalnız Türkiye dışındaki depremlerde (Les Cayes, Fas,
Noto) belirlenir. Benzer bir yaklaşım (ShakeMap + DPM) Li ve arkadaşlarının 2025 çalışmasında var.

**A3, SAR ön işleme.** SAR'ı dB ölçeğine çevirip z-skoru almak. SkyCap çalışması, çok yüksek çözünürlüklü SAR'da
sonuçların ön işlemeye çok duyarlı olduğunu gösterdi. Ucuz bir deneme; Faz 4'te ölçülecek.

### 13.3 Bedava kazançlar

**B1, TTA ve bina oylaması:** Bölüm 8.5. Zaten planlı.

**B2, heterojen topluluk (ensemble).** Birden çok modelin tahminlerinin ortalamasını almak, makine öğrenmesinin
en güvenilir iyileştirme yollarından biridir. Farklı mimariler farklı hatalar yapar; ortalama hataları kısmen
siler. Kendi ChangeMamba'mızın tahminleri, Zenodo'daki 5 hazır modelin tahminleriyle ortalanabilir. Hiçbiri
Türkiye'yi görmedi, hiçbiri yeni eğitim istemiyor. Ağırlıklar eşit ya da 13 olayın doğrulamasıyla seçilir. Sonuç
"topluluk" diye ayrı bir satırda raporlanır; başlık kıyası tek modelle yapılır.

**B3, ana model + varyant topluluğu.** İki tam eğitim zaten yapılacak; ortalamaları bedava.

### 13.4 Sınıf dağılımı kayması

Bölüm 3.2'de gördük: Model hasarın sık olduğu olaylardan öğrenip hasarın nadir olduğu Türkiye'de sınanıyor. Bu,
modelin Türkiye'de hasarı olduğundan fazla görmesine yol açabilir.

**C1, öncül düzeltmesi (EM).** Modelin olasılıkları, eğitimdeki sınıf oranlarını içerir. Test olayındaki gerçek
oranlar biliniyorsa olasılıklar Bayes kuralıyla düzeltilebilir. Oranlar bilinmiyorsa, modelin etiketsiz test
tahminlerinden **EM algoritmasıyla** (Saerens ve ark., 2002) tahmin edilebilir: Oranları tahmin et, olasılıkları
düzelt, düzeltilmiş olasılıklardan oranları yeniden tahmin et, oturana kadar tekrarla. Türkiye'nin etiketsiz
tahminlerini kullandığı için transdüktiftir.

**C2, olay dengeli örnekleme.** Eğitimde her olaydan eşit sayıda örnek çekmek. Model tek bir büyük olayın sınıf
oranını ezberlemez.

**C3, sıralı (ordinal) kayıp.** Sağlam < hasarlı < yıkık sıralı sınıflardır. Standart çapraz entropi bu sırayı
bilmez: "Yıkık"a "sağlam" demek ile "hasarlı" demek aynı cezayı alır. **CORN** gibi sıralı regresyon yöntemleri
problemi "hasar ≥ hasarlı mı?" ve "hasar ≥ yıkık mı?" gibi ardışık ikili sorulara böler ve sırayı korur. xBD
literatüründe denenmiş bir yaklaşım.

### 13.5 Model ve öğrenme tarafı

**D1, güçlü bina bulucu.** mIoU'nun dört sınıfından ikisi (arka plan ve sağlam) neredeyse tamamen bina bulmaya
bağlı, ve bina bulmak hasar bulmaktan çok daha kolay. Yalnız öncesi optikten bina bulan ayrı bir model, Türkiye
içermeyen büyük açık bina veri setleriyle (SpaceNet, WHU, Inria gibi) eğitilip ChangeMamba'nın bina başlığının yerine
konabilir. DFC25 birincisinin ilk aşaması tam olarak buydu.

**D2, bina ayak izinde havuzlama.** Bina oylamasının (Bölüm 8.5) öğrenilen hâli: Hasar başlığı, her binanın
pikselleri üzerinden özellikleri toplayıp (havuzlayıp) binaya tek bir karar verir.

**D3, komşuluk grafiği (GNN).** Binalar bir grafın düğümleri, komşuluklar kenarları olur; **grafik sinir ağı**
(graph neural network) her binanın kararında komşularının bilgisini kullanır. 2022'de olaylar arası genellemede
avantaj bildirildi. Ama kısa yol riski yüksek (aşağıda); bu yüzden park edildi.

**D4, modalite bırakma.** Eğitimde rastgele bazı örneklerde SAR sıfırlanır. Model hem yalnız optikle hem de ikisiyle
çalışmayı öğrenir. Böylece tek bir eğitimle "görülmemiş depremde VHR radar gerçekten ne katıyor?" sorusu temiz
biçimde cevaplanabilir.

**D5, temel model kodlayıcısı.** Ret önerisi: Kanıt zayıf, lisans ve "Türkiye'yi görmüş olabilir" riski var.

**D6, pseudo-label (sözde etiket).** Model Türkiye görüntülerinde tahmin yapar; en emin olduğu tahminler etiket gibi
kullanılıp model bunlarla ince ayardan geçirilir. Yarışma kazananları bunu kullandı. Transdüktif ve ek eğitim
istiyor; park edildi.

**D7 (görüntü-dil modelleri) ve D8 (sentetik hasar üretimi):** Biri hasar seviyesinde zayıf, öteki 16 haftaya
sığmıyor.

**Kısa yol öğrenme (shortcut learning) uyarısı.** Modeller bazen asıl görevi değil, onunla tesadüfen ilişkili kolay
bir ipucunu öğrenir. 2026'da yayımlanan bir çalışma, BRIGHT'ta **yalnız bina sayısını** bilen basit bir
sınıflandırıcının, olay dışarıda bırakıldığında 0,653 dengeli doğruluk aldığını gösterdi; yalnız öncesi görüntü de
hasarı tahmin ettirebiliyor. Yani modeller "yoğun yapılaşmış bölge = hasar" gibi bir kısa yol öğrenebilir. Bu, hem
GNN gibi komşuluk fikirlerine temkinle yaklaşmamızın hem de izole bina analizinin nedeni.

### 13.6 Dürüst sınır

"Hasarlı" (kısmen hasar) sınıfı insanlar için bile belirsiz ve 0,35 m'de yandan bakan radar çatı çatlağını iyi
görmez. Kazancın en çok "yıkık" ve "sağlam" sınıflarında gelmesi beklenmeli. 41,28'i birkaç puan geçmek mümkün
olabilir; "çok daha iyi" bir sonuç büyük olasılıkla yeni bir fiziksel kanıttan (A1) gelir, yeni bir mimariden değil.

---

## 14. Yazılım mimarisi

### 14.1 Neden modüler?

Projede üç model (ana, yedek, varyant), on bir ön kontrol, birkaç analiz ve bir demo var. Hepsi aynı veri ve metrik
kodunu kullanmalı; aksi hâlde aynı şeyi iki farklı yoldan hesaplayıp farklı sonuçlar alırız. Modüler yapı, her
parçayı bir kez yazıp her yerde aynı biçimde kullanmamızı sağlar.

### 14.2 Dizin düzeni

```
configs/        YAML yapılandırmaları (ana.yaml, varyant.yaml)
src/hasar/
  config.py     yapılandırma şeması, config hash
  guard.py      Türkiye kilidi
  results.py    sonuç JSON şeması
  data/         okuma, bölme, dönüşüm, örnekleyici, istatistik
  models/       model sözleşmesi, kayıt defteri, ChangeMamba, DamageFormer, varyant başlıkları
  losses.py     CE + Lovász
  train/        eğitim döngüsü, kayıt/devam, EMA, zamanlayıcı, log
  eval/         çıkarım, metrikler, pikselden binaya, bootstrap, final
  analysis/     Y1, Y2, Y3, hata galerisi
  preflight/    11 ön kontrol kapısı
  cli.py        hasar preflight | train | final | analyze
third_party/    ChangeMamba ve BRIGHT kodu (sabit commit, lisans, yama listesi)
notebooks/      ince Colab defteri
tests/          veri ve GPU gerektirmeyen testler
results/        her sayının kaynağı JSON dosyaları
docs/           tasarım, ADR, protokol, rehber
```

### 14.3 Tasarım ilkeleri

**Tek yönlü bağımlılık.** `data` ve `models` birbirini bilmez. `train` ikisini kullanır. `eval` `train`'i import
etmez. `analysis` modeli değil, tahmin dosyalarını okur. Bağımlılıklar tek yöne aktığında bir parçayı değiştirmek
ötekileri bozmaz ve her parça ayrı test edilebilir.

**Model sözleşmesi ve kayıt defteri.** Her model aynı arayüzü uygular:

```python
class DamageModel(Protocol):
    def forward(self, pre: Tensor, post: Tensor) -> dict[str, Tensor]:
        """pre: B×3×H×W optik, post: B×1×H×W SAR.
        Dönüş: {"loc": B×2×H×W, "clf": B×4×H×W} logit."""
```

Modeller bir **kayıt defterine** (registry) adlarıyla kaydedilir: `@register("changemamba_mmbda_tiny")`. YAML'da
`model.name` değiştirmek, eğitim ve değerlendirme koduna dokunmadan modeli değiştirir. Yedeğe geçiş tek satırlık bir
yapılandırma değişikliğine, varyant da `models/heads/` altına tek bir dosyaya iner. Zenodo modelleri de ince
sarmalayıcılarla aynı hattan geçer.

**Türkiye tek kapıdan.** Veri katmanı Türkiye karelerini yalnız `guard.py`'nin verdiği bir izinle açar. `guard`,
`protokol-kilidi` etiketinin var olduğunu ve çalışma ağacının temiz olduğunu denetler. Böylece kural 1 yalnız
disiplinle değil, kodla korunur: Yanlışlıkla Türkiye'yi okumaya çalışan bir kod hata verir.

**Tek çıkarım, çok analiz.** Türkiye'de model bir kez çalışır; olasılık haritaları (ve TTA çıktıları) diske yazılır.
Başlık metriği, ikincil satırlar, Y1, Y2, kalibrasyon, hata galerisi ve demo bu önbelleği okur. Faydaları:
Analizler GPU'suz, yerel makinede çalışır; Türkiye'ye "ikinci bir bakış" teknik olarak gerekmez; kilit denetimi
kolaylaşır.

**Yapılandırma ve config hash.** Bütün ayarlar tek bir YAML dosyasında ve Python `dataclass` şemasıyla doğrulanır.
Yapılandırmanın içeriğinden bir **özet değeri** (hash) hesaplanır ve her koşunun kimliğinin (run_id) parçası olur.
İki koşunun ayarları aynı mı, tek bakışta anlaşılır. Hydra gibi güçlü yapılandırma araçları bilinçli olarak
kullanılmıyor: İki eğitimlik bir projede esneklikten çok hata yüzeyi getirirler.

**Sonuç şeması.** Her sonuç `results/**/*.json` dosyasına yazılır ve şemada `metric_tag` alanı zorunludur. README'deki
her sayı bu dosyalardan otomatik üretilir (kural 5). Bir sayıyı elle yazmak ya da etiketsiz bırakmak böylece mümkün
olmaz.

**Üçüncü taraf kod ayrı.** ChangeMamba ve BRIGHT'tan alınan kod `third_party/` altında, alındığı commit'le, lisans
dosyasıyla ve yaptığımız değişikliklerin listesiyle (`PATCHES.md`) durur. Kendi kodumuz ona yalnız bir sarmalayıcıdan
dokunur.

### 14.4 Kalite araçları

| Araç | Ne yapar |
|---|---|
| **uv** | Python ortamı ve bağımlılık yöneticisi (yerelde). Torch sürümü sabitlenmez; Colab kendi sürümünü kullanır. |
| **ruff** | Kod stili ve hata denetleyicisi |
| **pytest** | Birim testleri |
| **nbstripout** | Notebook çıktılarını git'e girmeden temizler (büyük ve gereksiz değişiklikleri önler) |
| **GitHub Actions** | Her commit'te testleri CPU'da, gerçek veri olmadan, sentetik verilerle çalıştırır |

**Sentetik test** ne demek? Gerçek BRIGHT verisi 22 GB ve CI ortamında yok. Testler bunun yerine küçük, elle
oluşturulmuş sahte GeoTIFF'ler kullanır. Örneğin: "Optiği çevirince etiket de çevriliyor mu?", "`la_palma-volcano_
00000001` dosya adından olay adı doğru çıkıyor mu?", "Kusursuz tahminin mIoU'su 1,0 mı?", "Eğitim 60. adımda
öldürülüp devam ettirilince aynı sonuca varıyor mu?" Bu hatalar GPU saati harcanmadan yakalanır.

**Bir tuzak örneği: olay adı ayrıştırma.** Dosya adlarından olay adı çıkarılırken önce `_pre_disaster` /
`_post_disaster` eki atılır, sonra `rsplit('_', 1)[0]` alınır. `split('_')[0]` kullanılırsa `la_palma-volcano` olayı
`la` olarak okunur. Bu tür küçük hatalar bölmeyi sessizce bozar; bu yüzden testleri var.

---

## 15. Altyapı: Colab, GPU ve bütçe

### 15.1 Neden Colab?

Eğitim için büyük belleğe sahip bir GPU gerekiyor. Yerel RTX 4060 (8 GB) yalnız hata ayıklama ve küçük denemeler
için yeterli. Ana eğitim **Google Colab Pro** üzerinde, **A100** GPU ile yapılacak.

Colab kullanımı **compute unit** (CU) ile ölçülür. Tahmini yakım: A100 40 GB için saatte yaklaşık 5,40 CU. Colab Pro
ayda 100 CU verir (₺165,60). Oturum en fazla 12 saat sürer ve tarayıcının açık kalması gerekir. Pro+ ayda 600 CU
verir ve tarayıcı kapalıyken 24 saate kadar çalışır (₺826,80). High-RAM seçeneği yakımı yaklaşık %39 artırır; kapalı
tutulur.

### 15.2 Bütçe

| Adım | Süre | CU |
|---|---|---|
| Ortam, veri, hazır checkpoint'lerle yeniden üretim | 1–1,5 sa | 5–8 |
| Çekirdek derleme kapısı | ≤ 1 sa | ≤ 5,4 |
| Ön kontrol kapıları ve 2.000 adımlık prova | 1–1,5 sa | 5–8 |
| **Ana eğitim, 40.000 adım** | **9–14 sa** | **50–75** |
| Türkiye değerlendirmesi, TTA, belirsizlik | 0,5–1 sa | 3–5 |

İki tam eğitimle toplam tahminen 115–175 CU. 16 haftalık Pro yaklaşık 400 CU eder; sığar, ama iki eğitim aynı aya
düşmemeli. **Bu sürelerin hiçbiri ölçülmedi;** hepsi tahmin ve ilk Colab oturumunda ölçülecek.

### 15.3 Colab akışı: ince notebook

Notebook mantık içermez; yalnız `src/hasar` paketindeki komutları çağırır. "Run All" ile baştan sona çalışır ve bir
kopmadan sonra yeniden çalıştırıldığında kaldığı yerden aynı sonucu verir (**idempotent**).

| Hücre | İş | Kopma sonrası |
|---|---|---|
| 1 | Drive'ı bağla, etiketli commit'i klonla, paketi kur, gizli anahtarları oku | Commit SHA kaydedilir, kirli çalışma ağacı reddedilir |
| 2 | Ortam kapısı: A100, VRAM, High-RAM kapalı, disk ≥ 60 GB | Koşul tutmazsa durur |
| 3 | Veriyi Hugging Face'ten VM diskine indir, aç, denetle | `.done` varsa atlanır |
| 4 | Ön kontrol kapıları → `preflight.json` | Config hash'i aynıysa atlanır |
| 5 | Ana eğitim | `latest.pt` varsa tam durumla devam eder |
| 6 | Türkiye'de tek değerlendirme | Eğitim `DONE` değilse çalışmaz |
| 7 | Kopyalar doğrulanınca çalışma zamanını bırak | CU yakımı durur |

Veri her oturumda Drive'dan değil Hugging Face'ten VM'nin yerel diskine indirilir; Colab, Drive'dan çok sayıda küçük
dosya okumanın yavaş olduğunu belirtiyor. Kayıtlar JSONL formatında (her satır bir JSON nesnesi) Drive'a yazılır;
Weights & Biases isteğe bağlı.

### 15.4 CUDA çekirdeği derleme

ChangeMamba'nın seçici tarama işlemi C++/CUDA ile yazılmış ve kurulumda derlenmesi gereken bir uzantı. Derleme kapısında
denenecekler: `setup.py`'den eski `compute_70` hedefini çıkarıp yalnız A100'ün `compute_80`'ini bırakmak, derleyiciyi
(`nvcc`) pip paketinden göstermek, `--no-build-isolation` ile kurmak, eski `timm` sabitlemesini kaldırmak. Derlenen
paket Drive'a kaydedilir; Colab imajı değişirse yeniden derlemek gerekebilir. Bir saat içinde olmazsa yedeğe geçilir
(Bölüm 6.7).

---

## 16. Ürünleştirme ve demo

### 16.1 Dışa aktarım ve gecikme

Eğitilmiş bir modeli PyTorch dışındaki ortamlarda (C++, mobil, sunucu) çalıştırmak için **ONNX** gibi taşınabilir
formatlara aktarmak yaygındır. Projede bina başına çıkarım süresini (gecikme) de ölçeceğiz.

Bir risk var: ChangeMamba'nın özel CUDA çekirdeği büyük olasılıkla ONNX'e aktarılamaz (doğrulanmadı). Faz 4'te 10
dakikalık bir deneme yapılacak. Olmazsa alternatifler: saf PyTorch tarama uygulamasıyla aktarmak, `torch.export`
kullanmak ya da gecikmeyi doğrudan PyTorch'ta ölçmek.

### 16.2 Model kartı

**Model kartı**, bir modelin ne için eğitildiğini, hangi veriyle, nasıl değerlendirildiğini, nerede iyi nerede kötü
olduğunu ve nerede **kullanılmaması** gerektiğini anlatan standart bir belgedir. Bizim kartımızda örneğin "gerçek
afet müdahalesinde tek başına karar aracı olarak kullanılmamalıdır" ve "etiketler saha değil optik yorumdur" gibi
uyarılar yer alacak.

### 16.3 Demo: gerçek 6 Şubat verisi

Demo, BRIGHT'ın Türkiye kareleri üzerinde çalışacak. Bunlar 6 Şubat depreminin gerçek görüntüleri: Maxar'ın optik
ve Capella/Umbra'nın radar görüntüleri. Harita üzerinde şunlar gösterilecek:

- Afet öncesi optik ve afet sonrası radar görüntüsü
- Modelin hasar tahmini
- Üç referans (BRIGHT etiketi, HOT/OSM, ARIA DPM)
- Analistin kontrol listesi (Y2: önce bakılması gereken binalar)

Tahminler Faz 6'daki önbellekten gelir; demo için model yeniden çalıştırılmaz.

**BRIGHT dışındaki 6 Şubat bölgeleri** yalnız açık bir afet sonrası VHR radar kaynağı bulunursa eklenebilir. Afet
öncesi optik Maxar Open Data'da var (CC BY-NC 4.0), ama Capella ve Umbra'nın açık verisinde Türkiye bulunamadı.

**İstanbul'da hasar demosu yapılamaz:** Orada deprem olmadığı için afet sonrası görüntü yok.

### 16.4 Rakipleri anmak

README'de en yakın rakipler adıyla anılacak ve farklar bir tabloyla gösterilecek:

| Rakip | Yaptığı | Bu projeden farkı |
|---|---|---|
| BRIGHT makalesi | Zero-shot çıpaları | Sınıf kırılımı, güven aralığı, ikinci referans yok |
| CS437 group40 | BRIGHT optik + SAR, olay dışarıda | İkili IoU, tek tohum, yeniden üretim yok |
| Enkaz | Optik + optik, Antakya, kalibrasyon, bootstrap | SAR yok |
| DFC25 / CVPRW26 kazananları | Yarışma reçeteleri | Türkiye eğitimde veya görev farklı |
| Improved MambaBDA (2026) | MambaBDA'ya eklemeler | BRIGHT protokolü yok, kıyaslanamaz |

Ciddi bir değerlendiricinin bunları kendisinin bulmasındansa bizim göstermemiz daha güvenilirdir. "İlk" iddiası
yerine "bu aramada bulunamadı" denir.

---

## 17. Yolculuk: fazlar ve karar noktaları

| Faz | Ne yapılıyor | Neden bu sırada |
|---|---|---|
| **0. Belgeler ve protokol taslağı** | ADR'ler, metriğin tam hesabı, veri ve lisans tablosu, risk kaydı, SAR notu | Kod yazılırken belirsizlikler sessizce karara dönüşmesin |
| **1. Depo iskeleti** | Paket düzeni, guard, sonuç şeması, model sözleşmesi, CI | Her şeyin üstüne kurulacağı zemin |
| **2. Veri katmanı** | Okuma, bölme, dönüşüm, örnekleyici; A1 fizibilitesi | Veri doğru değilse hiçbir şey doğru değil |
| **3. Metrik ve yeniden üretim** | Metrik kodu; Zenodo modelleriyle 38,32 / 37,82 / 37,20 | Hattın doğru olduğunun eğitimsiz kanıtı |
| **4. Model, eğitim motoru, ön kontrol** | Derleme kapısı, eğitim döngüsü, 11 kapı, ONNX denemesi → `protokol-kilidi` | GPU harcamadan önce bütün hataları yakala |
| **5. Ana eğitim** | ChangeMamba, 40.000 adım | |
| **6. Karar noktası ve Türkiye** | KN1 → `varyant-kilidi` → Türkiye'de tek değerlendirme | Varyant, Türkiye sonucu görülmeden kilitlenmeli |
| **7. Analizler** | Y1, Y2, Y3, seçilen N'ler, hata galerisi; KN2 | Hepsi önbellekten, eğitimsiz |
| **8. Varyant eğitimi** | Tek fikirli varyant, bir kez | |
| **9. Demo ve ürünleştirme** | Demo, dışa aktarım, model kartı, README | |

**Karar noktaları:**

- **KN1 (Faz 6 başı):** Ana eğitim bitti, Türkiye görülmedi. 13 olayın sonuçlarına bakılarak varyant, topluluk
  satırları, analiz listesi ve transdüktif işler seçilir.
- **KN2 (Faz 7 sonu):** Türkiye sonucu ve analizler hazır. Yalnız raporlama derinliğine ve olası üçüncü bir eğitime
  (kullanıcı kararı; Türkiye sonucu ayrı "keşif" tablosunda) karar verilir.
- **KN3 (Faz 8 sonu):** README'ye ve demoya hangi bulguların gireceği.

---

## 18. Sözlük

| Terim | Anlamı |
|---|---|
| **ADR** | Architecture Decision Record; bir kararın neden verildiğini kaydeden kısa belge |
| **AdamW** | Ağırlık sönümünü ayrı uygulayan Adam optimizer'ı |
| **Ağırlık sönümü** | Ağırlıkları her adımda biraz küçülterek aşırı uyumu frenleyen düzenlileştirme |
| **Alan kayması** | Test verisinin eğitim verisinden farklı bir dağılıma sahip olması (domain shift) |
| **argmax** | En yüksek puanlı sınıfı seçme işlemi |
| **Artırım** | Eğitim görüntülerini çevirme, döndürme gibi dönüşümlerle çeşitlendirme |
| **AURC** | Risk-kapsam eğrisinin altındaki alan; küçükse model neyi bilmediğini biliyor |
| **Batch** | Bir eğitim adımında birlikte işlenen örnek grubu |
| **bf16** | 16 bitlik, fp32 ile aynı aralıkta ama daha az hassas sayı formatı |
| **Benek (speckle)** | Radarın eş fazlı doğasından gelen tuz-biber deseni |
| **Bindirme (layover)** | Radarda yüksek nesnenin tepesinin tabanından önce görünmesi |
| **Bootstrap** | Veriyi yerine koyarak yeniden örnekleyip belirsizlik ölçme yöntemi |
| **CE** | Çapraz entropi kaybı |
| **Checkpoint** | Eğitimin bir anındaki model ve durum kaydı |
| **Çift sekme** | Duvar ile zeminin oluşturduğu köşe yansıtıcıdan dönen güçlü radar sinyali |
| **Config hash** | Yapılandırma içeriğinden üretilen kısa kimlik |
| **CRS** | Koordinat referans sistemi |
| **CU** | Colab compute unit; GPU kullanım ölçüsü |
| **CUDA çekirdeği** | GPU üzerinde çalışan özel yazılmış hesap fonksiyonu |
| **dB** | Desibel; logaritmik güç ölçeği |
| **DPM** | Damage Proxy Map; radar koheransından üretilen hasar vekil haritası |
| **ECE** | Beklenen kalibrasyon hatası |
| **EMA** | Ağırlıkların üstel hareketli ortalaması |
| **Entropi** | Bir olasılık dağılımının kararsızlık ölçüsü |
| **F1** | Kesinlik ve duyarlılığın harmonik ortalaması |
| **Füzyon** | Farklı modalitelerin bilgisini birleştirme |
| **GeoTIFF** | Coğrafi konum bilgisi taşıyan TIFF dosyası |
| **Gradyan** | Kaybın her ağırlığa göre değişim yönü ve büyüklüğü |
| **GSD** | Bir pikselin yerde kapladığı mesafe |
| **IoU** | Kesişim bölü birleşim |
| **Kalibrasyon** | Modelin güven düzeyinin gerçek doğruluğuyla uyumu |
| **Kayıp** | Tahminin ne kadar yanlış olduğunu ölçen sayı |
| **Kodlayıcı / çözücü** | Görüntüyü özelliklere sıkıştıran / özellikleri haritaya açan ağ parçaları |
| **Koherans** | İki radar çekiminin faz benzerliği; düşüşü değişime işaret eder |
| **Logit** | Olasılığa çevrilmemiş ham sınıf puanı |
| **Lovász-Softmax** | IoU'nun türevlenebilir vekili olan kayıp |
| **Makro ortalama** | Her sınıfı eşit ağırlıkla ortalama |
| **Mamba** | Seçici durum uzay modeli; uzunlukla doğrusal maliyetli dizi modeli |
| **mIoU** | Sınıfların IoU ortalaması |
| **Modalite** | Veri türü (optik, radar) |
| **Nodata** | Görüntüde verinin olmadığı pikseller |
| **Ön eğitim** | Modeli önce genel bir veri setinde eğitme |
| **Önceden kayıt** | Değerlendirme kurallarını sonuçtan önce yazıp kilitleme |
| **Öncül (prior)** | Gözlemden önceki olasılık bilgisi (ör. sınıf oranları) |
| **Pseudo-label** | Modelin kendi emin tahminlerini etiket gibi kullanma |
| **SAR** | Sentetik açıklıklı radar |
| **Seçici tahmin** | Modelin emin olmadığı durumlarda karar vermekten kaçınması |
| **Semantik segmentasyon** | Her piksele bir sınıf atama görevi |
| **Sıcaklık ölçekleme** | Logitleri tek bir sayıya bölerek kalibrasyonu düzeltme |
| **Sızıntı** | Test bilgisinin eğitime veya karar sürecine karışması |
| **Softmax** | Logitleri toplamı 1 olan olasılıklara çeviren fonksiyon |
| **TTA** | Test zamanı artırımı |
| **Transdüktif** | Test verisinin etiketsiz hâlini kullanan yöntem |
| **Transfer öğrenme** | Bir görevde öğrenileni başka bir göreve aktarma |
| **UTM** | Dünyayı dilimlere bölen, metre birimli harita projeksiyonu |
| **VHR** | Çok yüksek çözünürlük (metrenin altı) |
| **VMamba / VSSM** | Mamba'nın görüntüye uyarlanmış hâli / onun kodlayıcı ağı |
| **Zero-shot** | Test olayından hiç örnek görmeden sınama |

---

## 19. Kaynaklar ve ileri okuma

**Projenin kendi belgeleri**

- [tasarim.md](tasarim.md): bağlayıcı kararlar
- [fazlar.md](fazlar.md): faz planı
- [fikir-havuzu.md](fikir-havuzu.md): ertelenmiş fikirler ve karar noktaları
- [Mimari ve rakip analizi](../reports/Hasar%20haritası%20mimari%20ve%20rakip%20analizi.md)
- [Eksik belgeler, modüler yapı ve yenilik](../reports/Eksik%20belgeler,%20modüler%20yapı%20ve%20yenilik%20değerlendirmesi.md)
- [ChangeMamba'nın ötesi](../reports/ChangeMamba'nın%20ötesi%20-%20çalışma%20belgesi.md)

**Veri ve çıpa**

- [BRIGHT, ESSD makalesi](https://essd.copernicus.org/articles/17/6217/2025/) ve [arXiv sürümü](https://arxiv.org/abs/2501.06019)
- [BRIGHT deposu](https://github.com/ChenHongruixuan/BRIGHT), [Hugging Face kopyası](https://huggingface.co/datasets/Kullervo/BRIGHT)
- [Zenodo 15349462: hazır zero-shot checkpoint'ler](https://zenodo.org/records/15349462)

**Mimari**

- [ChangeMamba deposu](https://github.com/ChenHongruixuan/ChangeMamba)
- Mamba: Gu ve Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- VMamba: Liu ve ark., "VMamba: Visual State Space Model" (2024)
- Swin Transformer: Liu ve ark. (2021)

**Eğitim ve kayıp**

- Lovász-Softmax: Berman ve ark., "The Lovász-Softmax loss" (CVPR 2018)
- AdamW: Loshchilov ve Hutter, "Decoupled Weight Decay Regularization" (2019)
- [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [Karpathy, "A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/)

**Değerlendirme, belirsizlik ve genelleme**

- Sıcaklık ölçekleme: Guo ve ark., "On Calibration of Modern Neural Networks" (2017)
- Öncül düzeltmesi: Saerens ve ark., "Adjusting the outputs of a classifier to new a priori probabilities" (2002)
- [DomainBed: Gulrajani ve Lopez-Paz](https://arxiv.org/pdf/2007.01434)
- [Kısa yol öğrenme ve BRIGHT (arXiv 2610.13766)](https://arxiv.org/abs/2610.13766)

**SAR ve referanslar**

- [Ainscoe ve ark., radar ve optik hasar haritaları](https://www.nature.com/articles/s43247-025-02623-4)
- [ARIA DPM, Türkiye](https://data.amerigeoss.org/it/dataset/aria-damage-proxy-map-copernicus-sentinel-1-for-the-turkiye-earthquakes)
- [HOT/OSM yıkık binalar](https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings)
- [SkyCap: VHR optik-SAR ve temel modeller](https://www.arxiv.org/abs/2512.14755)
- [Maxar Open Data, Türkiye 2023](https://maxar.com/open-data/turkey-earthquake-2023)
