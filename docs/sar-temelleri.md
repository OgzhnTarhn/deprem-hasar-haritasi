# SAR Temelleri: hata galerisinin sözlüğü

**Sürüm:** 1 (Faz 0, 6 Kasım 2026). Temel kavramların yumuşak anlatımı [öğrenme rehberi §2.2](ogrenme-rehberi.md)'de.
Bu not, Faz 7'deki SAR hata galerisinin ve bina düzeyi analizlerin dayanacağı kavramları ve bizim veriye özgü
sonuçlarını toplar.

## 1. Bizim verideki SAR

| Özellik | BRIGHT'taki durum | Sonuç |
|---|---|---|
| Sensörler | Capella Space ve Umbra, X-bandı (dalga boyu yaklaşık 3 cm) | Kısa dalga boyu yüzey pürüzlülüğüne ve küçük ayrıntıya duyarlı; bitki örtüsünü geçemez |
| Çözünürlük | Optikle aynı ızgaraya örneklenmiş, yaklaşık 0,3–0,35 m | Tek binanın ayrıntısı görünür |
| Bant | Tek bant, yalnız genlik | Faz bilgisi yok: koherans ve interferometri BRIGHT'tan hesaplanamaz |
| Değer aralığı | uint8 (0–255) | Doğrusal mı dB'den mi türetildiği bilinmiyor (Faz 2, protokol §4) |
| Zaman | Yalnız afet sonrası | Öncesi radar yok; radar değişim tespiti yapılamaz |
| Kayıt hatası | Optikle yaklaşık 0,9 piksel (BRIGHT) | Hizalama iyi; ama bu, yer seviyesindeki hizalamadır (aşağıda §3) |
| Bakış geometrisi | Sahneden sahneye farklı (yörünge yönü, geliş açısı) | Aynı bina farklı karelerde farklı görünür; meta veride varsa hata analizine girer |

## 2. Saçılma mekanizmaları ve hasar imzaları

| Mekanizma | Ne olur | Radarda görünüşü | Sağlam bina | Yıkık bina |
|---|---|---|---|---|
| **Ayna yansıması** | Düz yüzey sinyali uzağa yansıtır | Koyu | Düz çatı, avlu | Düzleşmiş molozun bazı kısımları |
| **Yüzey saçılması** | Pürüzlü yüzey her yöne saçar | Orta ton | Toprak, çakıl | Enkaz yığını (dağınık, orta-parlak doku) |
| **Çift sekme** | Duvar + zemin köşe yansıtıcısı | Bina cephesinde çok parlak çizgi | Belirgin | **Kaybolur veya parçalanır** |
| **Hacim saçılması** | Sinyal bir hacimde çok kez saçılır | Orta ton, gürültülü | Ağaç tepeleri | Karmaşık moloz |

**Yargı:** Yıkık binanın en güçlü radar izi, çift sekme çizgisinin kaybolması ve yerine düzensiz bir saçılma
dokusunun gelmesidir. Hasarlı (kısmen) binada çift sekme çoğunlukla korunur; çatı çatlakları X-bandında bile
görünmeyebilir. Bu, "hasarlı" sınıfının radarda neden zor olduğunu açıklar.

## 3. Geometri: bindirme ve gölge

Radar yana, geliş açısı θ ile bakar. Yüksekliği h olan bir bina için yer menzilinde yaklaşık:

```
bindirme (layover) uzunluğu ≈ h / tan(θ)    (uyduya doğru)
gölge uzunluğu              ≈ h · tan(θ)    (uydudan uzağa)
```

Örnek: 5 katlı (yaklaşık 15 m) bir bina, θ = 35°:

- Bindirme ≈ 15 / 0,70 ≈ **21 m** → 0,35 m pikselde yaklaşık **60 piksel**
- Gölge ≈ 15 × 0,70 ≈ **10,5 m** → yaklaşık **30 piksel**

**Sonuç (yargı):** Çok katlı bir binanın radar izi, optikteki çatı ayak izinden onlarca piksel kayık ve komşu
binalarla üst üste binmiş olabilir. "Optikle 0,9 piksel hizalı" bilgisi yalnız yer seviyesi için geçerlidir. Bu
yüzden:

- Hasar başlığının bir binanın kararını yalnız ayak izinin içindeki SAR piksellerinden vermesi yetersiz kalabilir
  (fikir havuzu D2 tasarlanırken dikkate alınır).
- Hata galerisinde yüksek ve yoğun yapılaşmış bölgeler (Antakya, Kahramanmaraş merkezi) ayrı incelenir.
- Bina yüksekliği bilinmiyorsa, ayak izi alanı ve yoğunluk yaklaşık bir vekil olarak kullanılabilir.

## 4. Benek ve ön işleme

- **Benek** eş fazlı görüntülemenin doğası; çarpımsal bir gürültü gibi davranır. Klasik çözüm filtreleme (Lee,
  Frost) ya da çok bakışlı ortalama. Tasarım benek giderme eklemiyor (kanıt yok); model dokuyu kendisi öğreniyor.
- **dB dönüşümü** çarpımsal benek gürültüsünü toplamsal hâle getirir ve dinamik aralığı sıkıştırır. VHR SAR'da
  sonuçlar ön işlemeye çok duyarlı (SkyCap). BRIGHT'ın uint8 değerleri muhtemelen zaten sıkıştırılmış bir ölçekte;
  Faz 2'de histogramlarla kontrol edilir (fikir havuzu A3).

## 5. Capella ve Umbra farkı

İki sağlayıcı aynı bantta çalışsa da işleme zincirleri, çözünürlük modları, polarizasyon ve geliş açısı aralıkları
farklı olabilir. Model, eğitimde bir sağlayıcıyı daha çok gördüyse ötekinde zayıflayabilir. **Faz 2'de yapılacak:**
Her karenin sağlayıcısı (dosya meta verisinden ya da BRIGHT belgelerinden) çıkarılır; 13 olayda ve Türkiye'de
sağlayıcı dağılımı sayılır (Türkiye için yalnız meta veri, etiket değil). Hata galerisi sonuçları sağlayıcıya göre
kırılır.

## 6. Koherans (BRIGHT'ta yok, Sentinel-1'de var)

İki karmaşık değerli (SLC) çekimin yerel benzerliği:

```
γ = |E[s1 · s2*]| / sqrt(E[|s1|²] · E[|s2|²])     (0 ≤ γ ≤ 1)
```

Değişmeyen yapılar yüksek koherans korur; çöken yapılar koheransı düşürür. ARIA DPM, deprem öncesi ve sonrası
koherans çiftlerinin farkından üretilir. Sentinel-1'in çözünürlüğü 10–30 m'dir; tek binayı değil mahalleyi görür.
Bitki örtüsü, tarım ve su da koheransı düşürür; DPM'in kendi notu bu yüzden yalnız kentsel alanlarda
kullanılmasını önerir. Fikir havuzu A1 ve Y1'in DPM referansı bu sinyale dayanır.

## 7. Hata galerisinin kategorileri (Faz 7)

| Kategori | Tanım | Ölçü |
|---|---|---|
| Sağlayıcı | Capella / Umbra | Sınıf başına IoU, bina F1 |
| Yükseklik ve yoğunluk | Ayak izi alanı ve komşu bina yoğunluğu (yükseklik vekili) | Hata oranı |
| Bindirme / gölge bölgesi | Bina ayak izinin bakış yönündeki komşuluğu | Örnek galeri |
| İzole hasar | Komşularından farklı sınıftaki hasarlı/yıkık binalar | Duyarlılık (literatür: %7–11) |
| Nodata kenarı | Boş piksellere komşu bölgeler | Hata oranı |
| Sınıf karışıklığı | Hasarlı↔sağlam, hasarlı↔yıkık | Karışıklık matrisi |
