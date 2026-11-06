# ChangeMamba'nın ötesi: hasarlı binayı daha iyi bulmak için ne denenebilir?

**Tarih:** 6 Kasım 2026 · **Tür:** Çalışma belgesi (bağlayıcı değil). Kabul edilen fikirler `docs/tasarim.md`'ye,
ADR'ye ve `docs/fazlar.md`'ye taşınır. Faz numaraları yazıldığı andaki plana göredir; güncel plan
`docs/fazlar.md` sürüm 2, fikirlerin durumu `docs/fikir-havuzu.md`.

Önceki raporlarda olduğu gibi: **Kanıt** = kaynak ya da belge okuması, **Yargı** = çıkarım veya öneri. Beklenen
kazançların hepsi yargıdır; hiçbiri bu veride ölçülmedi.

---

## 0. Kısa cevap

- **Bugün 41,28'i geçen yayımlanmış bir yöntem bulunamadı.** BRIGHT ekibinin 2026 karşılaştırması da "genel amaçlı
  temel modeller hâlâ özel mimarilerin gerisinde, ChangeMamba en iyi genelleyen" diyor
  ([EGU26-7216](https://meetingorganizer.copernicus.org/EGU26/EGU26-7216.html)).
- **"Çok daha iyi" bir sonuç büyük olasılıkla daha iyi bir mimariden gelmez.** Mimari farklar Türkiye'de 1–3 puanda
  kalıyor (eski rapor). Fark yaratabilecek üç kaldıraç var:
  1. **Modele yeni bir fiziksel kanıt vermek** (radar koheransı, yer sarsıntısı): modelin bugün göremediği bilgi.
  2. **Ücretsiz kazançları toplamak** (topluluk, TTA, bina oylaması): sıfır yeni eğitim.
  3. **Sınıf dağılımı kaymasını düzeltmek**: Türkiye'de binaların %93'ü sağlam, eğitim olayları ise hasar ağırlıklı.
- **Dürüst sınır:** "Hasarlı" (kısmen hasar) sınıfı insanlar için bile belirsiz ve 0,35 m yandan bakan radar çatı
  çatlağını iyi görmez. Kazanç en çok "yıkık" ve "sağlam" sınıflarında beklenmeli, "hasarlı"da değil.

---

## 1. Önce sezgi: hasar uydudan neden ve nereden görünür?

"Bir yerden mutlaka bulunur" düşüncesi kısmen doğru. Hasarın uzaydan görülebilen **altı ayrı izi** var; modelimiz
bunların yalnız birini, kısmen kullanıyor:

| # | Hasarın izi | Neyle görülür | Bizim modelde var mı? |
|---|---|---|---|
| 1 | **Radar yansımasının değişmesi.** Sağlam binada duvar ile zemin bir "köşe yansıtıcı" oluşturur ve radarda parlak bir çizgi verir (çift sekme). Yıkılan bina bu köşeyi kaybeder; enkaz sinyali dağınık saçar. | Afet sonrası VHR SAR (BRIGHT'taki Capella/Umbra) | **Evet**, ama öncesi radar yok; model "önceden nasıldı"yı optikten tahmin etmek zorunda |
| 2 | **Radar koheransının kaybı.** Aynı yerin art arda çekilen iki radar görüntüsü normalde "faz olarak" benzerdir; yapı yıkılınca bu benzerlik düşer. ARIA DPM tam olarak budur. | Sentinel-1 (ücretsiz, 10–30 m, afetten önce ve sonra) | **Hayır** |
| 3 | **Yükseklik kaybı.** Çöken binanın boyu kısalır. | Stereo optik, LiDAR, InSAR yükseklik modeli | Hayır (bu projede veri yok) |
| 4 | **Yer sarsıntısının şiddeti.** Hasar olasılığı sarsıntıyla artar. | USGS ShakeMap (ücretsiz, depremden saatler sonra) | **Hayır** |
| 5 | **Binanın kırılganlığı.** Yaş, kat sayısı, yapı türü. | Kadastro, OSM etiketleri | Hayır (Türkiye için erişim belirsiz) |
| 6 | **Komşuluk.** Hasar kümelenir; yıkık binanın komşusu da hasarlı olma eğilimindedir. | Modelin kendi tahminleri | Dolaylı (evrişimin gördüğü pencere kadar) |

**Kanıt:** Saha sayımına göre yalnız 2 numaralı izi kullanan öğrenmesiz ARIA DPM bina F1 0,47 aldı. Optik
yorumlu Copernicus haritaları 0,15–0,23'te, Microsoft'un CNN'i 0,24'te kaldı
([Ainscoe ve ark.](https://www.nature.com/articles/s43247-025-02623-4)). **Yargı:** Modelimizin göremediği en güçlü
ve ücretsiz kanıt koheranstır. Sentinel-1 koheransını VHR görüntüyle birleştiren bir derin öğrenme çalışması bu
taramada bulunamadı (yakın işler: koherans zaman serisiyle RNN,
[Stephenson ve ark.](https://meetingorganizer.copernicus.org/EGU21/EGU21-6995.html); yalnız VHR SAR ile
makine öğrenmesi, [PMC12457488](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12457488/)).

**Uyarı (kısa yol):** Modeller hasar yerine kolay ipuçlarını da öğrenebiliyor. BRIGHT'ta yalnız bina sayısını bilen
bir sınıflandırıcı, olay dışarıda bırakıldığında dengeli doğrulukta 0,653 alıyor; yalnız öncesi görüntüsü de hasarı
tahmin ettirebiliyor ([arXiv 2610.13766](https://arxiv.org/abs/2610.13766)). Komşularından farklı, izole hasarlı
binaların yalnız %7–11'i bulunuyor ([arXiv 2610.38162](https://arxiv.org/html/2610.38162)). Yani "komşuluk" (iz 6)
hem kaldıraç hem tuzaktır.

---

## 2. Fikir kataloğu

Her fikir dört ölçüyle verildi: **beklenen kazanç** (yargı), **ek tam eğitim**, **kural uyumu** ve **kanıt düzeyi**.

**Kural uyumu kısaltmaları:**
- ✅ Başlık tablosuna girer, 41,28 ile kıyaslanır.
- 🟨 Aynı protokol, ama ayrı satır olarak etiketlenir (ör. "topluluk" veya "ek girdi").
- 🟥 Türkiye'nin etiketsiz görüntüsünü kullanır; yalnız ayrı "transdüktif" tabloda verilir.

### A. Yeni fiziksel kanıt eklemek (en büyük potansiyel, en büyük iş)

| # | Fikir | Nasıl | Kazanç | Ek eğitim | Uyum | Kanıt |
|---|---|---|---|---|---|---|
| A1 | **Sentinel-1 koherans kanalı** | 14 olay için afet öncesi ve sonrası Sentinel-1 çiftlerinden koherans değişimi üretilir (ASF HyP3 ücretsiz hesaplıyor), BRIGHT karesine yeniden örneklenir ve hasar kodlayıcısına ek bir kanal olarak verilir. | Yargı: "yıkık" sınıfında belirgin olabilir; 30 m çözünürlük tek binayı ayıramaz, bölgesel bir öncül gibi çalışır. Bilinmiyor. | 1 (varyant olabilir) | 🟨 (girdi kümesi farklı) | Dolaylı: DPM saha F1 0,47; birleşim çalışması bulunamadı, bu yüzden **yenilikçi** |
| A2 | **ShakeMap sarsıntı öncülü** | Tahmin sonrası Bayesçi düzeltme: P(hasar \| görüntü, PGA). Katsayılar Türkiye dışındaki depremlerde (Les Cayes, Fas, Noto) uydurulur. | Yargı: küçük-orta; deprem dışı olaylarda yok, yalnız son işlem | 0 | 🟨 | Benzer: ShakeMap + DPM ile nedensellik bilgili Bayesçi tahmin ([Li ve ark. 2025](https://journals.sagepub.com/doi/10.1177/87552930241290501)) |
| A3 | **SAR ön işlemesi** | SAR'ı dB ölçeğine çevirip z-skoru almak, eğitim öncesi küçük denemeyle. | Yargı: küçük ama ucuz | 0 (küçük deneme) | ✅ (eğitim reçetesinin parçası) | VHR SAR'da sonuç ön işlemeye çok duyarlı ([SkyCap](https://www.arxiv.org/abs/2512.14755)). BRIGHT SAR'ının uint8 olarak hangi ölçekte saklandığı kontrol edilmeli. |

**Risk (A1):** 14 olayın hepsi için Sentinel-1 çifti bulmak ve hizalamak 1–2 haftalık bir veri işi. ARIA'nın hazır
DPM'i bazı olaylar için var (Türkiye, Haiti 2021), Noto ve Fas için bulunamadı; bu yüzden hazır DPM değil kendi
koheransımızı üretmek gerekir. Türkiye koheransı test anında girdi olarak kullanılır; bu, Türkiye'nin afet sonrası
SAR görüntüsünü kullanmakla aynı statüdedir, etiket kullanılmaz.

### B. Sıfır eğitimle ücretsiz kazançlar (önce bunlar)

| # | Fikir | Nasıl | Kazanç | Ek eğitim | Uyum | Kanıt |
|---|---|---|---|---|---|---|
| B1 | **TTA + bina düzeyi oylama** | Zaten ikincil satır olarak planlı. | +0,56 / +0,69 / birlikte +0,92 | 0 | ✅ ikincil | Kanıt ([ESSD](https://essd.copernicus.org/articles/17/6217/2025/)) |
| B2 | **Heterojen topluluk** | Kendi ChangeMamba'mız + Zenodo'daki 5 hazır Türkiye zero-shot checkpoint'i (UNet, SiamCRNN, SiamAttnUNet, DeepLabV3+, DamageFormer). Hiçbiri Türkiye'yi görmedi. Ağırlıklar eşit veya 13 olayın doğrulamasıyla seçilir. | Yargı: +1–3. Farklı mimarilerin hataları farklıdır. | 0 | 🟨 "topluluk" satırı | DFC25 birincisi çok modelli füzyon kullandı ([Li ve ark.](https://arxiv.org/abs/2505.04941)); BRIGHT'ta ayrıştırılmış ölçüm bulunamadı |
| B3 | **Ana model + varyant topluluğu** | İki tam eğitim zaten yapılacak; ikisinin ortalaması bedava. | Yargı: +0,5–2 | 0 | 🟨 | Genel bilgi |

### C. Sınıf dağılımı kayması

**Kanıt:** Türkiye'de bina piksellerinin %93'ü sağlam; hasarlı piksellerin yaklaşık %63'ü ise Ukrayna ve Meksika
olaylarında (eski rapor). Model "hasar sık görülür" diye öğrenip "hasar nadirdir" olan bir yerde sınanıyor. Bu,
yanlış pozitifler demek.

| # | Fikir | Nasıl | Kazanç | Ek eğitim | Uyum | Kanıt |
|---|---|---|---|---|---|---|
| C1 | **Öncül düzeltmesi (EM)** | Modelin Türkiye'deki etiketsiz tahminlerinden sınıf oranları tahmin edilir ve olasılıklar buna göre yeniden ölçeklenir (Saerens 2002 yöntemi). | Yargı: hasarlı ve yıkık sınıflarında kesinlik artabilir, duyarlılık düşebilir; net etki bilinmiyor | 0 | 🟥 | Genel yöntem iyi kurulmuş ([LEIP](https://arxiv.org/pdf/2604.01651), [Label Shift Adapter](https://arxiv.org/pdf/2308.08810)); hasar haritalamada uygulaması bulunamadı |
| C2 | **Olay dengeli örnekleme** | Eğitimde her olaydan eşit örnek: model tek bir olayın sınıf oranını ezberlemez. | Yargı: küçük | 1 (reçete değişikliği) | ✅ | Zayıf |
| C3 | **Sıralı (ordinal) kayıp** | Sağlam < hasarlı < yıkık sırasını kayba öğretmek (CORN). "Yıkık"ı "sağlam" demek, "hasarlı" demekten daha ağır cezalanır. | Yargı: küçük; komşu sınıf karışıklığını azaltabilir | 1 (varyant) | ✅ | xBD'de sıralı yaklaşımla iyi sonuç bildirildi ([xBD](https://arxiv.org/pdf/1911.09296)) |

### D. Model ve öğrenme tarafı

| # | Fikir | Kazanç | Ek eğitim | Uyum | Kanıt ve not |
|---|---|---|---|---|---|
| D1 | **Bina konumlandırmayı dış veriyle güçlendirmek.** Yalnız öncesi optikten bina bulan ayrı bir model, Türkiye içermeyen açık bina veri setleriyle (ör. SpaceNet, WHU, Inria) eğitilir; ChangeMamba'nın bina başlığının yerine ya da yanına konur. | Yargı: orta. mIoU'nun 4 sınıfından ikisi (arka plan, sağlam) neredeyse tamamen bina bulmaya bağlı; bu iş hasar bulmaktan çok daha kolay. | 1 küçük eğitim (bina modeli) | ✅ | DFC25 birincisinin ilk aşaması tam olarak bu ([Li ve ark.](https://arxiv.org/abs/2505.04941)). Dış verinin Türkiye içermediği doğrulanmalı. |
| D2 | **Bina ayak izinde havuzlama** (mevcut Y4) | Küçük (+0,69'luk oylamanın öğrenilen hâli) | 1 | ✅ | |
| D3 | **Komşuluk grafiği (GNN)** | Belirsiz; olaylar arası genellemede avantaj bildirilmiş, ama kısa yol riski yüksek | 1 | ✅ | [Ismail ve Awad 2022](https://arxiv.org/abs/2201.10395) |
| D4 | **Modalite bırakma** (eğitimde SAR'ı rastgele sıfırlama) | Doğrudan kazanç belirsiz; SAR'ın gerçek katkısını ölçmeye yarar | 1 | ✅ | Bilimsel değeri yüksek (önceki rapor, N4) |
| D5 | **Temel model kodlayıcısı** | Düşük. BRIGHT ekibine göre genel temel modeller özel mimarilerin gerisinde. Lisans ve "Türkiye'yi görmüş olabilir" riski var. | 1 | ✅/❌ | [EGU26](https://meetingorganizer.copernicus.org/EGU26/EGU26-7216.html), [SkyCap](https://www.arxiv.org/abs/2512.14755) |
| D6 | **Türkiye'de pseudo-label** | Yarışmalarda büyük katkı, ama transdüktif | +0,25 | 🟥 | Önceki rapor |
| D7 | **Görüntü-dil modelleri (VLM)** | Düşük; hasar seviyesinde zayıflar | – | – | [DisasterInsight](https://arxiv.org/abs/2601.18493), [GeBDA](https://arxiv.org/abs/2608.28567) (sayı yok) |
| D8 | **Sentetik hasar üretimi** (difüzyon, SAR benzetimi) | Belirsiz, 16 haftaya sığmaz | – | – | SAR benzetimiyle hasar ([GFZ](https://gfzpublic.gfz-potsdam.de/pubman/item/item_5018516)); çoğu optik |

---

## 3. Önerilen yol

**İlke:** Tasarımdaki "iki tam eğitim" sınırını bozmadan en büyük beklenen kazancı almak. Önce bedava olanlar,
sonra tek varyant hakkını en güçlü fikre harcamak.

| Sıra | İş | Ne zaman | Maliyet |
|---|---|---|---|
| 1 | **A3 SAR ön işleme denemesi** (dB/z-skoru, küçük ve yerel; 13 olayın doğrulamasıyla) | Faz 1–3, ana eğitimden önce | ~0–2 CU |
| 2 | **Ana eğitim** (ChangeMamba olduğu gibi) | Faz 4 | 50–75 CU |
| 3 | **B1 + B2 + B3**: TTA, oylama, topluluk satırları | Faz 5 (kilitli satırlar) | <5 CU |
| 4 | **Varyant seçimi**: aday listesi A1 (koherans kanalı), D1 (güçlü bina bulucu), C3+D2 (sıralı kayıp + havuzlama), D4 (modalite bırakma). Tek fikir, yalnız 13 olayın doğrulamasıyla seçilir. | Faz 5 | 0 |
| 5 | **C1 öncül düzeltmesi**: transdüktif tabloda | Faz 6 | 0 |
| 6 | **Varyant eğitimi** | Faz 7 | 50–75 CU (+ A1 seçilirse veri işi) |

**Varyant için önerim A1 (koherans kanalı).** Üç nedeni var: modele gerçekten yeni bir bilgi veriyor, literatürde bu
birleşim bulunamadı (yenilik), ve sonuç ne olursa olsun yayımlanabilir bir soruya cevap veriyor: "Ücretsiz Sentinel-1
koheransı, VHR SAR modeline ne katıyor?" Riski veri işinin büyüklüğü. Bu yüzden önce bir **fizibilite ön kontrolü**
gerekiyor: 14 olayın kaçında uygun Sentinel-1 çifti var? Kapsam 10 olayın altındaysa D1'e geçilir.

---

## 4. Açık sorular (araştırılmalı)

1. **BRIGHT SAR'ı hangi ölçekte?** uint8 değerler doğrusal genlik mi, dB mi? (A3'ün anlamı buna bağlı.)
2. **Sentinel-1 kapsamı:** 14 olayın her biri için afet öncesi ve sonrası, aynı yörüngeden SLC çifti var mı?
   Ukrayna (savaş, tarihi belirsiz) ve yangınlarda koherans neyi ölçer?
3. **Hazır checkpoint'lerin 13 olaydaki doğrulama çıktısı:** B2'deki topluluk ağırlıklarını seçmek için gerekli;
   Zenodo modelleri bizim hattımızda doğrulama kümesinde çalıştırılabiliyor mu? (Faz 2'de ölçülür.)
4. **Dış bina veri setleri (D1):** Lisansları ve Türkiye içermedikleri.
5. **Topluluk satırı ve 41,28:** Topluluk aynı protokolde, ama tek model değil. README'de "tek model" ile
   "topluluk" ayrı satırlarda verilir; başlık kıyası tek modelle yapılır.

---

## 5. Kullanıcının kararları

| # | Karar | Seçenekler ve maliyet | Önerim |
|---|---|---|---|
| K1 | Bedava kazançlar (B1–B3) | (a) Ekle, ikincil ve topluluk satırları olarak protokolde kilitle (<5 CU) · (b) Yalnız B1 | (a) |
| K2 | Varyantın aday listesi | (a) Mevcut Y4 (havuzlama) · (b) A1, D1, C3+D2 ve D4 arasından 13 olaya göre seçim · (c) Doğrudan A1 | (b), A1 öncelikli |
| K3 | A1 fizibilitesi | (a) Faz 1 sırasında 2–3 günlük kapsam kontrolü · (b) Atla | (a) |
| K4 | Transdüktif işler (C1, D6) | (a) Yalnız C1 (eğitimsiz) · (b) C1 + D6 (+13–19 CU) · (c) Hiçbiri | (a) |
| K5 | Üçüncü tam eğitim (ör. A1 ve D1'in ikisini birden denemek) | Kural 3'e göre senin kararın; +50–75 CU, takvime +3 hafta | Şimdilik hayır; varyant sonucundan sonra yeniden bak |

## Kaynaklar

- [EGU26-7216: BRIGHT üzerinde çok modlu hasar haritalama karşılaştırması](https://meetingorganizer.copernicus.org/EGU26/EGU26-7216.html)
- [Ainscoe ve ark., radar ve optik hasar haritaları](https://www.nature.com/articles/s43247-025-02623-4)
- [Shortcut Learning in GFM Embeddings for Cross-Disaster Damage Assessment (arXiv 2610.13766)](https://arxiv.org/abs/2610.13766)
- [SkyCap: VHR optik-SAR dörtlüleri ve temel modeller (arXiv 2512.14755)](https://www.arxiv.org/abs/2512.14755)
- [DFC25 birincisi, bina rehberli pseudo-label (arXiv 2505.04941)](https://arxiv.org/abs/2505.04941)
- [Li ve ark. 2025, ShakeMap ile nedensellik bilgili Bayesçi hasar tahmini](https://journals.sagepub.com/doi/10.1177/87552930241290501)
- [Stephenson ve ark., InSAR koherans zaman serisiyle derin öğrenme](https://meetingorganizer.copernicus.org/EGU21/EGU21-6995.html)
- [VHR SAR ve makine öğrenmesiyle bina düzeyi deprem hasarı (PMC12457488)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12457488/)
- [ARIA DPM, Haiti 2021](https://maps.disasters.nasa.gov/ags04/rest/services/haiti_earthquake_202108/aria_dpm_sentinel1_v3_20210815/ImageServer/info/iteminfo)
- [ARIA DPM, Türkiye 2023](https://data.amerigeoss.org/it/dataset/aria-damage-proxy-map-copernicus-sentinel-1-for-the-turkiye-earthquakes)
- [Ismail ve Awad, olaylar arası hasar için GCN (arXiv 2201.10395)](https://arxiv.org/abs/2201.10395)
- [xBD (arXiv 1911.09296)](https://arxiv.org/pdf/1911.09296)
- [Label Shift Adapter (arXiv 2308.08810)](https://arxiv.org/pdf/2308.08810)
- [LEIP, sonradan öncül tahmini (arXiv 2604.01651)](https://arxiv.org/pdf/2604.01651)
- [DisasterInsight (arXiv 2601.18493)](https://arxiv.org/abs/2601.18493)
- [GeBDA (arXiv 2608.28567)](https://arxiv.org/abs/2608.28567)
- [BRIGHT, ESSD](https://essd.copernicus.org/articles/17/6217/2025/)
- [RaySAR ile hasar benzetimi (GFZ)](https://gfzpublic.gfz-potsdam.de/pubman/item/item_5018516)
