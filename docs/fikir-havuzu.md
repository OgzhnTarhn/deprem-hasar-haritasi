# Fikir Havuzu

Araştırmada çıkan ama henüz karara bağlanmamış bütün iyileştirme ve yenilik fikirleri burada durur. Fikirler
**karar noktalarında** (aşağıda) ele alınır. Kaynak raporlar:

- [Eksik belgeler, modüler yapı ve yenilik değerlendirmesi](../reports/Eksik%20belgeler,%20modüler%20yapı%20ve%20yenilik%20değerlendirmesi.md) (N kodları)
- [ChangeMamba'nın ötesi: çalışma belgesi](../reports/ChangeMamba'nın%20ötesi%20-%20çalışma%20belgesi.md) (A–D kodları)

Bir fikir kabul edilince `tasarim.md`'ye ve bir ADR'ye taşınır, buradaki durumu güncellenir. Reddedilen fikir
silinmez; gerekçesiyle "Ret" olarak kalır.

## "Eğitim sonucuna göre ilerlemek" ne demek?

Kural 1 yüzünden iki tür sonuç birbirinden ayrılır:

- **13 olayın doğrulama sonuçları** (ana eğitimden hemen sonra): Varyant, topluluk ve protokol eklentileri **yalnız
  bunlara bakılarak** seçilir. Seçim, Türkiye'ye dokunulmadan `varyant-kilidi` etiketiyle kilitlenir.
- **Türkiye sonucu** (kilitten sonra): Yalnız raporlamayı ve analizlerin derinliğini etkiler. Türkiye sonucu
  görüldükten sonra yeni bir tam eğitim istenirse (kural 3, kullanıcı kararı), onun Türkiye sonucu temiz zero-shot
  sayılmaz; ayrı "keşif" tablosunda raporlanır.

## Karar noktaları

| Kod | Ne zaman | Girdiler | Kararlar | Kilit |
|---|---|---|---|---|
| **KN1** | Faz 6 başı: ana eğitim bitti, Türkiye görülmedi | 13 olayın sınıf bazlı doğrulama sonuçları, hata örnekleri, A1 fizibilite sonucu, A3 ölçümü | Varyant fikri (V adayları), topluluk satırları, analiz listesi, transdüktif işler | `varyant-kilidi` |
| **KN2** | Faz 7 sonu: ana modelin Türkiye sonucu ve analizler hazır | Türkiye sonucu, Faz 7 bulguları | Raporlama derinliği; üçüncü tam eğitim isteniyor mu (keşif etiketiyle) | – |
| **KN3** | Faz 8 sonu: varyant değerlendirildi | İki modelin sonuçları | Hangi bulgular README'ye ve demoya girer | – |

## Fikirler

**Durumlar:** Planlı (bir fazda zaten var) · Aday (karar noktasında seçilebilir) · Park (16 hafta içinde öngörülmüyor) ·
Ret önerisi (araştırma zayıf buldu; karar noktasında kesinleşir).

### Model ve doğruluk iyileştirmeleri

| Kod | Fikir | Ek tam eğitim | Rapor türü | Durum | Karar |
|---|---|---|---|---|---|
| A1 | Sentinel-1 koherans kanalı (ek girdi) | 1 (varyant olarak) | Ayrı satır: "ek girdi" | Aday · fizibilite geçti (14/14 olay, Faz 2) | KN1 |
| A2 | ShakeMap sarsıntı öncülüyle Bayesçi son işlem | 0 | Ayrı satır | Aday | KN1 |
| A3 | SAR z-skoru ön işleme (değerler zaten log/dB benzeri; olay başına germe farklı) | 0 (küçük deneme) | Reçetenin parçası | Planlı · ölçüm Faz 4'te, karar 13 olaya göre ve `protokol-kilidi`'nden önce | Faz 4 |
| B1 | Flip TTA + bina düzeyi oylama | 0 | İkincil satır | Planlı | – |
| B2 | Heterojen topluluk: ana model + Zenodo'daki hazır zero-shot checkpoint'ler (DeepLabV3+ hariç: Faz 3'te Türkiye'yi görmüş çıktı) | 0 | Ayrı satır: "topluluk" | Aday | KN1 |
| B3 | Ana model + varyant topluluğu | 0 | Ayrı satır: "topluluk" | Aday | KN1 |
| C1 | Etiketsiz öncül (sınıf oranı) düzeltmesi, EM | 0 | Transdüktif tablo | Aday | KN1 |
| C2 | Olay dengeli örnekleme | 1 | Varyant | Aday (V) | KN1 |
| C3 | Sıralı (ordinal) kayıp, CORN | 1 | Varyant | Aday (V) | KN1 |
| D1 | Türkiye içermeyen açık veriyle ayrı bina bulucu | 1 küçük | Varyant | Aday (V) | KN1 |
| D2 | Bina ayak izinde havuzlayan hasar başlığı (eski Y4) | 1 | Varyant | Aday (V) | KN1 |
| D3 | Komşuluk grafiği (GNN) | 1 | Varyant | Park · kısa yol riski yüksek | – |
| D4 | Modalite bırakma (SAR'ın gerçek katkısı) | 1 | Varyant | Aday (V) | KN1 |
| D5 | Temel model kodlayıcısı | 1 | – | Ret önerisi · kanıt zayıf, lisans ve Türkiye riski | KN1 |
| D6 | Türkiye görüntüsüyle pseudo-label | +0,25 | Transdüktif tablo | Park · +13–19 CU | KN1 |
| D7 | Görüntü-dil modelleri | – | – | Ret önerisi · hasar seviyesinde zayıf | – |
| D8 | Sentetik hasar üretimi | – | – | Park · 16 haftaya sığmaz | – |

**Varyant (V) adayları:** A1, C2, C3, D1, D2, D4. KN1'de tek bir fikir seçilir (kural 3: tek fikirli varyant).
Araştırmanın önceliği: A1 (fizibilite geçerse), sonra D1.

### Faz 2 bulguları (13 olay ve yalnız meta veri)

Sayıların kaynağı `results/arastirma/*.json`; betikler `scripts/arastirma/`.

- **A1 fizibilitesi geçti** (`s1_kapsam.json`). 14 olayın hepsinde, Türkiye'de yedi şehrin her birinde, aynı yörüngede
  olaydan önce en az iki ve olaydan sonra en az bir Sentinel-1 SLC çekimi var (pencere: 36 gün önce, 24 gün sonra).
  İlk uygun sonrası çekim 0–8 gün içinde. Uyarılar: Beirut'ta "0 gün" çekimin patlama saatinden önce mi sonra mı
  olduğu ayrıca denetlenmeli; Ukrayna'da hasar haftalara yayıldığı için olay tarihi yaklaşık. Koherans ürünü
  (ör. ASF HyP3 InSAR) 40–80 m pikselde; BRIGHT karesinin (yaklaşık 300 m) birkaç pikseline düşer, yani A1 kare ya
  da mahalle ölçeğinde bir öncül olur, bina ölçeğinde değil.
- **A3 için SAR ölçeği** (`sar_olcek.json`). BRIGHT'taki uint8 SAR değerleri doğrusal genlik gibi davranmıyor:
  8×8 bloklarda log(std)–log(ortalama) eğimi havuzda 0,2 (çarpımsal benekte ~1 beklenir), ham değerlerin çarpıklığı
  0,43, log değerlerinki −1,69. Değerler log/dB benzeri, kırpılmış bir ölçekte. Olay ortalamaları 41–104 arasında
  değişiyor: olay başına farklı bir germe uygulanmış olabilir. Bu, A3'ü (eğitim kümesi z-skoru) ve olay düzeyinde
  normalizasyon sorusunu destekler; dB'ye dönüştürme adımı gerekmez.
- **Bina örnekleri** (`ornek_etiket_kapsami.json`, `bina_birlesme.json`). CVPRW26 örnek etiketleri (COCO biçimi)
  10 eğitim olayının zero-shot eğitim karelerinde var; doğrulama karelerinde ve Meksika, Myanmar, Ukrayna'da yok.
  Türkiye'de 1.114 karenin 1.001'inde var (yalnız dosya adlarından sayıldı). Etiketteki 8-komşuluklu bağlı
  bileşenler örnek sayısının yalnız 0,61'i kadar: bitişik binaların yaklaşık %39'u birleşiyor. Bina piksellerinin
  %5,9'u birden çok hasar sınıfı içeren bileşenlerde. D2 (ayak izinde havuzlama) eğitimde örnek etiketlerini
  kullanabilir; bağlı bileşenle yetinmek bina sayısını ciddi biçimde düşürür.

### Değerlendirme ve yenilik analizleri

| Kod | Fikir | Ek tam eğitim | Durum | Karar |
|---|---|---|---|---|
| Y1 | Aynı binalarda referans anlaşmazlığı: BRIGHT etiketi, HOT/OSM, ARIA DPM. HOT/OSM de optik yoruma dayanır; çerçeve "iki optik yorum + bir radar sinyali". | 0 | Planlı (Faz 7) | – |
| Y2 | Analist bütçeli önceliklendirme; DPM sıralaması taban olarak eklenir | 0 | Planlı (Faz 7) | – |
| Y3 | Etiketsiz SAR yeniden standartlaştırma | 0 | Planlı (Faz 7, transdüktif); yenilik iddiası taşımaz | – |
| N1 | Yayımlanmış zero-shot sıralamasının güven aralıkları (6 model, blok ve eşleştirilmiş bootstrap) | 0 | Aday · kural 1 istisnasının genişletilmesi kullanıcı kararı | KN1 |
| N2 | Model sıralamasının referansa göre değişimi (Y1'in genişlemesi) | 0 | Aday | KN1 |
| N3 | Hataların ShakeMap sarsıntı şiddetine göre kırılımı | 0 | Aday | KN1 |

**Önerilen ana araştırma sorusu (KN1'de karar):** "Görülmemiş bir depremde model sıralaması, hangi referansa ve
hangi istatistiğe baktığına göre değişiyor mu?"

### Demo

| Kod | Fikir | Durum |
|---|---|---|
| DM1 | BRIGHT'ın Türkiye karelerinde (gerçek 6 Şubat verisi: Maxar optik + Capella/Umbra SAR) önce/sonra, tahmin, üç referans | Planlı (Faz 9) |
| DM2 | BRIGHT dışındaki 6 Şubat bölgeleri: öncesi optik Maxar Open Data'dan (CC BY-NC 4.0) | Aday · afet sonrası VHR SAR'ın açık kaynağı doğrulanmadı (Capella/Umbra açık veride Türkiye bulunamadı) |
| DM3 | İstanbul | Ret önerisi (hasar demosu olarak) · deprem olmadığı için afet sonrası görüntü yok. Yalnız öncesi optikten "bina envanteri" gösterilebilir, ama VHR görüntü lisansı gerekir ve projenin sorusuna katkısı yok. |
