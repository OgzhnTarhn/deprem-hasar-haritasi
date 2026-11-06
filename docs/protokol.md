# Değerlendirme Protokolü (önceden kayıt)

- **Durum:** Taslak, sürüm 0.1 (Faz 0, 6 Kasım 2026). Faz 4 sonunda son hâline getirilir ve `protokol-kilidi`
  etiketli commit'e girer. Kilitten sonra bu dosya değiştirilmez; değişiklik gerekirse yeni sürüm, gerekçesi ve
  tarihiyle §14'e yazılır ve başlık sonucunu etkileyen her değişiklik raporda belirtilir.
- **Kapsam:** Türkiye'deki tek değerlendirmenin bütün hesap kuralları: bölme, başlık metriği, girdi işleme, kayıt
  noktası seçimi, ikincil satırlar, bina düzeyi metrikler, güven aralıkları, "eşit" kuralı, kalibrasyon.
- **Dayanaklar:** [ADR-0001](adr/0001-zero-shot-bolme.md), [ADR-0004](adr/0004-sar-girdisi.md),
  [ADR-0005](adr/0005-baslik-metrigi.md); BRIGHT ve ChangeMamba kodunun Faz 0 okuması.

**İşaretler:** ✅ kararlaştırıldı · 🟨 öneri, kullanıcı onayı bekliyor · ⬜ açık, belirtilen fazda ölçülüp
kararlaştırılacak.

---

## 1. Değişmezler

1. Türkiye'nin ne görüntüsü ne etiketi eğitimde, doğrulamada, kayıt noktası seçiminde, eşik/sıcaklık ayarında veya
   normalizasyon istatistiğinde kullanılır. ✅
2. Her model Türkiye'de yalnız `final` adımında, bir kez değerlendirilir. `final`, `protokol-kilidi` etiketi ve temiz
   çalışma ağacı olmadan çalışmaz (`guard`). ✅
3. Varyantın tasarımı ana modelin Türkiye sonucu görülmeden `varyant-kilidi` ile kilitlenir. ✅
4. Bu dosyadaki her ayar Türkiye'ye dokunmadan, yalnız 13 olayın verisiyle seçilir. ✅

## 2. Bölme ✅

BRIGHT `cross_event/zero_shot` mantığı, `test_event = turkey-earthquake` (ADR-0001).

| Küme | Kaynak | Kare |
|---|---|---|
| Eğitim | Standart eğitim + standart test listesi, Türkiye hariç | 2.795 |
| Doğrulama | Standart doğrulama listesi, Türkiye hariç | 337 |
| Test | Türkiye'nin üç standart listedeki bütün kareleri | 1.114 |

13 eğitim olayı: bata-explosion, beirut-explosion, congo-volcano, haiti-earthquake, hawaii-wildfire,
la_palma-volcano, libya-flood, marshall-wildfire, mexico-hurricane, morocco-earthquake, myanmar-hurricane,
noto-earthquake, ukraine-conflict.

Liste dosyaları BRIGHT deposunun sabit bir commit'inden alınır; commit SHA'sı ve dosya özetleri
`results/veri/bolme.json`'a yazılır (Faz 2).

## 3. Başlık metriği ✅

Etiket: `[ZS-EVENT mIoU]`. Resmî kod yolu (ADR-0005):

1. **Girdi:** Türkiye'nin her karesi 1024 × 1024 tam olarak, kırpmadan, tek tek modele verilir.
2. **Tahmin:** `pred = argmax(clf)`; TTA yok, bina maskesi yok, sıcaklık yok.
3. **Matris:** 4×4 karışıklık matrisi `C` bütün 1.114 kare boyunca biriktirilir. Etiketi [0, 4) dışında kalan
   pikseller matrise girmez. Nodata pikselleri ayrıca dışlanmaz (resmî kodla aynı).
4. **IoU:** her sınıf k için `IoU_k = C[k,k] / (Σ_j C[k,j] + Σ_i C[i,k] − C[k,k] + 1e-7)`.
5. **mIoU:** dört sınıfın (arka plan, sağlam, hasarlı, yıkık) ortalaması.
6. **Kıyas:** Yalnız bu sayı 41,28 ile kıyaslanır; fark §9'daki kurala göre yazılır.

Her karenin 4×4 matrisi ayrıca saklanır (`results/<model>/turkiye_kare_matrisleri.npz`). Bootstrap (§8) ve bütün
alt kırılımlar bu matrislerden hesaplanır; model yeniden çalıştırılmaz.

## 4. Girdi işleme

| Madde | Kural | Durum |
|---|---|---|
| Optik | İlk 3 bant; ImageNet ortalama/standart sapması (0–255 ölçeği: 123,675 / 116,28 / 103,53; 58,395 / 57,12 / 57,375) | ✅ |
| SAR | Resmî biçim: tek bant 3 kanala kopyalanır, ImageNet istatistikleriyle normalize edilir (ADR-0004, seçenek C) | 🟨 |
| SAR ölçeği | BRIGHT'taki uint8 değerlerin doğrusal mı dB'den mi türetildiği | ⬜ Faz 2 |
| Eğitimde nodata | Optik ve SAR'ın ikisi de 0 olan pikseller kayıpta `ignore_index = 255` ile yok sayılır | ✅ (reçete; değerlendirmeyi etkilemez) |
| Değerlendirmede nodata | Resmî kodla aynı; ayrıca dışlanmaz. Yok sayılmış hâl ikincil satır | ✅ |

## 5. Kayıt noktası seçimi ✅

- Her 2.000 adımda **EMA** ağırlıkları doğrulama kümesinin (337 kare) tamamında §3'teki yoldan değerlendirilir
  (tam kare, `argmax(clf)`, maskesiz).
- En yüksek doğrulama mIoU'su seçilir; eşitlikte erken adım.
- Ham (EMA'sız) ağırlıkların doğrulama sonucu da kaydedilir ama seçimde kullanılmaz.
- Bu seçimin zayıf olduğu (doğrulama-test sıralama korelasyonu 0,35, CVPRW26) raporda yazılır.

## 6. İkincil satırlar (önceden ilan)

Bütün ikincil satırlar Faz 6'daki tek çıkarımın önbelleğinden hesaplanır.

| Satır | Tanım | Ayar kaynağı | Durum |
|---|---|---|---|
| Flip TTA | Kimlik, yatay, dikey ve ikisi birden; softmax olasılıklarının ortalaması, sonra argmax | Ayar yok | ✅ |
| Bina maskeli harita | `loc` argmax'ı 0 olan piksellerde tahmin arka plan; ötekilerde `argmax(clf[1:]) + 1` | Ayar yok | ✅ |
| Bina düzeyi oylama | Her bina örneğinin piksellerinde çoğunluk sınıfı | Örnek kaynağı §7 | ⬜ Faz 2 |
| Nodata yok sayılmış | §3'te nodata pikselleri matrise alınmaz | Ayar yok | ✅ |
| Topluluk satırları | B2/B3 (fikir havuzu) | KN1'de kabul edilirse ağırlıklar 13 olayda | ⬜ KN1 |

Her satırın etiketi `[ZS-EVENT mIoU]` ile birlikte satır adını taşır (ör. `[ZS-EVENT mIoU, flip-TTA]`); başlık
kıyası yalnız §3 ile yapılır.

## 7. Bina düzeyi metrikler ⬜ Faz 2

- **Örnek kaynağı:** BRIGHT'ın bina örnek etiketleri varsa onlar. Yoksa etiketteki bina piksellerinin (sınıf 1–3)
  8-komşuluklu bağlı bileşenleri. Bağlı bileşen kullanılırsa 13 olayda bitişik binaların birleşme oranı raporlanır.
- **Gerçek sınıf:** Örneğin etiket piksellerinin çoğunluk sınıfı.
- **Tahmin sınıfı:** Örneğin içindeki piksellerin `argmax(clf)` çoğunluğu; arka plan çoğunluktaysa "kaçırıldı".
- **Metrik:** Sağlam, hasarlı ve yıkık sınıflarının F1'leri ve makro ortalaması `[ZS-EVENT bina makro F1]`.
- Ayrıca piksel düzeyi "üç bina sınıfının IoU ortalaması" raporlanır.

## 8. Güven aralıkları

| Madde | Kural | Durum |
|---|---|---|
| Yöntem | Yerine koyarak yeniden örnekleme; her örneklemde seçilen birimlerin kare matrisleri toplanır, §3'teki metrik yeniden hesaplanır | ✅ |
| Birim | Mekânsal blok: komşu kareler aynı bloğa girer. Blok tanımı (şehir mi, sabit km ızgarası mı) Türkiye karelerinin yalnız koordinatlarına bakılarak belirlenir; etikete bakılmaz | ⬜ Faz 2 |
| Örnek sayısı | 10.000 | ✅ |
| Aralık | %95, yüzdelik (2,5 ve 97,5) | ✅ |
| Tohum | Sabit (`results` dosyasına yazılır) | ✅ |
| İki modelin kıyası | Eşleştirilmiş bootstrap: her örneklemde iki modelin farkı aynı bloklarla hesaplanır | ✅ |

## 9. "Eşit" kuralı 🟨

Öneri: İki sayı arasındaki fark **1,0 puanın altındaysa veya** eşleştirilmiş bootstrap farkının %95 aralığı 0'ı
içeriyorsa sonuç "eşit" yazılır. Aksi hâlde fark ve aralığı yazılır ("ana model 41,28'in X puan üstünde, %95 aralık
[a, b]").

- 41,28 ile kıyasta eşleştirilmiş bootstrap yapılamaz (yayımlanmış modelin kare çıktıları yok). Bu kıyasta yalnız
  bizim sayımızın aralığı verilir; 41,28 aralığın içindeyse "eşit".
- Onaylanırsa `CLAUDE.md` kural 6'daki "1–2 puan" ifadesi bu kurala bağlanır.

## 10. Kalibrasyon ve belirsizlik

| Madde | Kural | Durum |
|---|---|---|
| Sıcaklık | `clf` logitleri için tek bir T, 13 olayın doğrulamasında negatif log olabilirlikle seçilir; Türkiye'de yalnız ölçülür | ✅ |
| ECE | Bina pikselleri (etiket 1–3) üzerinde, en yüksek olasılıklı sınıfın güveni, 15 eşit genişlikli kutu | ✅ |
| Bina belirsizliği | Flip TTA tahminlerinin ortalamasının entropisi, bina örneği içinde ortalanır | ✅ |
| Risk-kapsam | Binalar belirsizliğe göre sıralanır; AURC raporlanır | ✅ |

## 11. Türkiye erişimi ve önbellek ✅

- Türkiye karelerine erişen tek kod yolu `eval/final.py`; `guard` etiketi ve temiz çalışma ağacını denetler, commit
  SHA'sını sonuç dosyasına yazar.
- Tek çıkarımda her kare için saklananlar: `clf` ve `loc` softmax olasılıkları (float16), flip TTA olasılıkları,
  kare matrisleri. Bütün ikincil satırlar, analizler ve demo bu önbelleği okur.

## 12. Raporlama ✅

Her sonuç `results/**/*.json`'a `metric_tag`, commit SHA, config hash ve tohumla yazılır. Başlık tablosu:
4 sınıfın IoU'ları, mIoU ve %95 aralığı, üç bina sınıfının IoU ortalaması, bina düzeyi makro F1, ikincil satırlar,
41,28 ile kıyas ve §9'a göre yorum.

## 13. Açık maddeler

| Madde | Bölüm | Ne zaman | Girdi |
|---|---|---|---|
| SAR girdi biçimi onayı | §4 | Faz 1 öncesi | Kullanıcı (ADR-0004) |
| SAR uint8 ölçeği | §4 | Faz 2 | 13 olayın SAR histogramları |
| Bina örnek kaynağı | §6, §7 | Faz 2 | BRIGHT deposu ve veri incelemesi |
| Bootstrap blok tanımı | §8 | Faz 2 | Türkiye karelerinin koordinatları (etiketsiz) |
| "Eşit" kuralı onayı | §9 | Faz 4 öncesi | Kullanıcı |
| Topluluk satırları | §6 | KN1 | Fikir havuzu B2/B3 |
| Ön kontrol kapısı 7'nin beklenen değeri | – | Faz 3 | "Hep arka plan" mIoU'su = arka plan oranı / 4; sabit 0,21 değil, doğrulama kümesinin etiket dağılımından hesaplanır |

## 14. Değişiklik günlüğü

| Sürüm | Tarih | Değişiklik |
|---|---|---|
| 0.1 | 6 Kasım 2026 | İlk taslak; BRIGHT ve ChangeMamba değerlendirici kodu okunarak yazıldı |
