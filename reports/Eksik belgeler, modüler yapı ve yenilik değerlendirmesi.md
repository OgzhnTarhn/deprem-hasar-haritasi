# Eksik belgeler, modüler yapı ve yenilik değerlendirmesi

**Tarih:** 6 Kasım 2026 · **Kapsam:** `docs/tasarim.md` (sürüm 2), `docs/fazlar.md`, araştırma raporu ve 6 Kasım'da
yapılan literatür taraması. Bu belge bağlayıcı değildir; kabul edilen maddeler `tasarim.md`'ye ve ADR'ye taşınır.
Faz numaraları yazıldığı andaki plana göredir; aynı gün faz planı sürüm 2'ye geçti ve fazlar bir kaydı (güncel:
`docs/fazlar.md`). Kararların durumu: `docs/fikir-havuzu.md`.

Önceki raporda olduğu gibi iki tür cümle ayrı tutuldu: **Kanıt** bir kaynak ya da belge okumasıdır, **Yargı** çıkarım
ya da öneridir.

## Özet

1. **Tasarım iyi, ama dört yerde belirsiz ve bu belirsizlikler kod yazılırken sessizce karara dönüşür:** başlık
   metriğinin tam hesap yolu, pikselden binaya geçiş, bootstrap'in birimi ve "eşit" eşiği. Bunlar Faz 0'dan önce
   yazılı bir **protokol belgesine** girmeli.
2. **İki mimari kör nokta var:** ChangeMamba'nın özel CUDA çekirdeği büyük olasılıkla ONNX'e çıkmaz (Faz 8 planı
   buna dayanıyor) ve HOT/OSM yıkık bina katmanı optik yorumdan bağımsız değildir (Y1'in "bağımsız referans"
   cümlesi buna dayanıyor).
3. **Modüler yapı için önerim:** model sözleşmesi + kayıt defteri, Türkiye erişimini tek kapıya bağlayan bir koruma
   modülü ve **"tek çıkarım, çok analiz"** düzeni: Türkiye tahminleri bir kez diske yazılır, bütün yenilik analizleri
   modeli değil bu önbelleği okur.
4. **Proje şu hâliyle titiz ama yenilikçi değil.** Y2–Y4'ün her birinin yakın bir öncülü var. Yenilik, model
   tarafında değil **değerlendirme biliminde** kurulabilir: "Görülmemiş depremde model sıralaması, hangi referansa
   ve hangi istatistiğe baktığına göre değişiyor mu?" Bu soru sıfır ek eğitimle, keskin ve yanlışlanabilir bir
   bulgu üretir.

---

## 1. Eksik belgeler

### 1.1 Planlanmış ama olmayanlar

| Belge | Nerede söz verildi | Neden şimdi gerekli |
|---|---|---|
| `docs/adr/` ve geriye dönük ADR'ler | `CLAUDE.md`, Faz 0 | Verilmiş kararlar yalnız tasarım metninde: zero-shot bölme, ChangeMamba/DamageFormer yedeği, SAR normalizasyonu, başlık metriği yolu, DINOv3/MaRS dışlaması. Her biri için 0001–0006 arası kısa ADR. |
| Veri ve lisans tablosu | Tasarım §9, hafta 1–2 | Ukrayna, Myanmar ve Meksika optiğinin kaynağı açık karar (eski rapor, karar 4). Bu karar 41,28 kıyasını bozabileceği için Faz 1'den önce yazılı olmalı. |
| `docs/bulgular.md` | Faz 6 | Faz 6'da açılır; şimdi gerekmez. Şablonu (olumsuz sonuç bölümü dahil) Faz 0'da konabilir. |

### 1.2 Hiç planlanmamış ama gerekenler

| # | Belge | Eksik olan karar | Etki |
|---|---|---|---|
| B1 | **`docs/protokol.md` (önceden kayıt)** | Tasarım §6 neyin kilitleneceğini söylüyor, nasıl hesaplanacağını söylemiyor. Aşağıdaki 1.3'teki dört belirsizlik buraya. `protokol-kilidi` etiketi bu dosyayı kilitler. | Yüksek: başlık sayısının geçerliliği |
| B2 | **Sonuç JSON şeması** (`results/SCHEMA.md` veya `pydantic` modeli) | `CLAUDE.md` kural 4 ve 5 (metrik etiketi, sayılar dosyadan) yalnız disiplinle korunuyor. Şemada `metric_tag` zorunlu alan olursa kural kodla korunur. | Orta, ucuz |
| B3 | **Y1 birleştirme şartnamesi** | HOT/OSM poligonları, BRIGHT raster etiketleri ve 30 m ARIA DPM aynı binaya nasıl bağlanacak? Koordinat kayması toleransı, 30 m pikselin 0,35 m binaya atanma kuralı, tarih uyumsuzluğu, elle denetimin örnekleme planı (100–200 bina nasıl seçilecek). | Yüksek: Y1'in geçerliliği |
| B4 | **Y2 değerlendirme şartnamesi** | Belirsizlik ölçüsü (TTA entropisi mi, hasar olasılığı mı), sıralama birimi, tabanlar. Özellikle **DPM'e göre sıralama tabanı** eksik: model, öğrenmesiz radarın sıralamasını geçemiyorsa bu bulgudur. | Orta |
| B5 | **Risk kaydı** | Colab imajı değişir (derlenmiş wheel kırılır), HF veri kopyası kalkar, Zenodo checkpoint'i yeni kodla yüklenmez, ARIA/HDX verisi taşınır. Her biri için tetik ve yedek. | Orta |
| B6 | **SAR temelleri notu** | Tasarım §9 hafta 1–2'de "SAR temelleri" var ama çıktısı tanımlı değil. Layover, gölge, benek, Capella/Umbra farkı: hata galerisinin sözlüğü. Portföyde alan okuryazarlığını gösterir. | Düşük, ama Faz 6 galerisi buna dayanır |

### 1.3 Protokol belgesine girmesi gereken dört belirsizlik

**1. Başlık metriğinin tam hesap yolu.** *Kanıt:* Tasarım "resmî kod yolu" diyor; eski rapor ChangeMamba eğitim
döngüsünün bina maskesiz `argmax` kullandığını yazıyor. *Eksik:* Karışıklık matrisi olay boyunca mı biriktiriliyor,
yoksa kare başına IoU mu ortalanıyor? Arka plan sınıfı ortalamaya giriyor mu (giriyor görünüyor: "hep arka plan ≈
0,21")? Değerlendirmede nodata pikselleri yok sayılıyor mu? Eğitimde yok saymak (tasarım §7) bizim kararımız;
değerlendirmede yok saymak resmî yoldan sapma olur ve 41,28 kıyasını bozar. **Yargı:** Değerlendirmede nodata
resmî kodla aynı işlenmeli; yok sayılmış hâli ikincil satır olur. Tam görüntü (1024²) mü kayan pencere mi, o da
yazılmalı. Faz 2'deki yeniden üretim bu kararları ancak dolaylı olarak doğrular; doğrudan BRIGHT değerlendirici
kodu okunup protokole yazılmalı.

**2. Pikselden binaya geçiş.** "Bina düzeyi makro F1", "bina düzeyi oylama", Y2 ve Y4 hep bina örneğine ihtiyaç
duyuyor. *Eksik:* BRIGHT etiketi raster; bina örnekleri nereden gelecek? Bitişik binaların bağlı bileşenle
ayrılması yoğun Antakya dokusunda binaları birleştirir. CVPRW26 örnek düzeyi göreviyle örnek etiketleri
yayımlanmış olabilir. **Araştırılmalı (Faz 1 öncesi):** BRIGHT'ın örnek poligonları/maskeleri Türkiye dahil var mı,
yoksa bağlı bileşen kullanılacaksa birleşme oranı ne?

**3. Bootstrap'in birimi.** *Eksik:* "Kareler üzerinde bootstrap" komşu karelerin ilişkili olduğunu yok sayar ve
güven aralığını dar gösterir. **Yargı:** Birim, mekânsal blok ya da şehir olmalı (Türkiye karelerinin hangi
şehirlere dağıldığı Faz 1'de sayılır). İki modelin kıyası için **eşleştirilmiş** bootstrap (aynı yeniden örneklemde
iki modelin farkı) kullanılmalı. Örnek sayısı (ör. 10.000) ve aralık türü (yüzdelik) sabitlenmeli.

**4. "Eşit" eşiği.** *Kanıt:* Belgeler "1–2 puanın altı" diyor. Bu bir aralıktır, eşik değil; sonuç görüldükten sonra
1 mi 2 mi seçileceği tartışmaya açık kalır. **Yargı:** Tek kural kilitlenmeli, ör. "eşleştirilmiş bootstrap farkının
%95 aralığı 0'ı içeriyorsa *veya* fark 1,0 puanın altındaysa eşit".

### 1.4 Tasarımda düzeltilmesi gereken iki cümle

**HOT/OSM bağımsız değil.** *Kanıt:* HDX kaydına göre HOT/OSM yıkık bina katmanı Copernicus EMS ve İTÜ-CSCRS hava
görüntülerinden, OSM ve MapRoulette gönüllülerince işaretlendi ([HDX](https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings)).
Yani bu katman da optik yoruma dayanıyor ve muhtemelen yalnız "yıkık" sınıfını içeriyor. **Yargı:** Y1 "üç bağımsız
referans" değil, "iki optik yorum (uzman ve gönüllü) + bir fiziksel radar sinyali" olarak çerçevelenmeli. Bu
zayıflık değil; iki optik yorumun birbiriyle anlaşmazlığı zaten ölçülmeye değer bir bulgu.

**ONNX planı ChangeMamba ile büyük olasılıkla çalışmaz.** *Yargı (doğrulanmadı):* `selective_scan` özel bir CUDA
uzantısıdır; ONNX dışa aktarımı bu işlemi tanımaz. Faz 8 için seçenekler: (a) saf PyTorch `selective_scan`
başvuru uygulamasıyla dışa aktarmak (hızı bilinmiyor), (b) ONNX yerine `torch.export`/TorchScript ile gecikme
ölçmek, (c) gecikme ölçümünü PyTorch çıkarımıyla raporlamak. Derleme kapısı (Faz 3) sırasında 10 dakikalık bir
ONNX denemesi bu riski erkenden kapatır.

---

## 2. Modüler kod yapısı önerisi

### 2.1 İlkeler

1. **Bağımlılık tek yönlü:** `data` ve `models` birbirini bilmez; `train` ikisini kullanır; `eval` `train`'i import
   etmez; `analysis` modeli değil tahmin dosyalarını okur.
2. **Model sözleşmesi:** Her model aynı arayüzü uygular. Ana model, yedek (DamageFormer) ve Y4 varyantı yalnız
   YAML'da `model.name` değiştirilerek seçilir; eğitim ve değerlendirme kodu değişmez.
3. **Türkiye tek kapıdan:** Veri katmanı Türkiye karelerini yalnız `final` modülünün verdiği bir koruma nesnesiyle
   açar; bu nesne `protokol-kilidi` etiketini ve temiz çalışma ağacını denetler. Kural 1 disiplinden koda geçer.
4. **Tek çıkarım, çok analiz:** Türkiye'de model bir kez çalışır; olasılık haritaları (ve TTA çıktıları) diske
   yazılır. Başlık metriği, ikincil satırlar, Y1, Y2, kalibrasyon ve hata galerisi bu önbelleği okur. Faydası: analiz
   kodu CPU'da ve yerel makinede çalışır, Türkiye'ye "ikinci bakış" teknik olarak gerekmez, kilit denetimi kolaylaşır.
5. **Saf fonksiyonlar test edilir:** Metrik, bootstrap, olay adı ayrıştırma, eşleştirilmiş dönüşümler NumPy ile,
   torch'suz yazılır ve CI'da sentetik veriyle sınanır.
6. **Üçüncü taraf kod ayrı:** ChangeMamba ve BRIGHT'tan alınan kod `third_party/` altında, sabit commit, lisans
   dosyası ve yapılan değişikliklerin listesiyle durur. Kendi kodumuz ona yalnız bir sarmalayıcıdan dokunur.

### 2.2 Dizin düzeni

```
configs/
  ana.yaml                  # ana model
  varyant.yaml              # Y4; ana.yaml'dan yalnız fark
src/hasar/
  config.py                 # dataclass şeması, YAML okuma, config hash
  guard.py                  # TurkeyGuard: etiket + temiz ağaç denetimi; Türkiye karelerine tek erişim yolu
  results.py                # sonuç JSON şeması (metric_tag zorunlu), atomik yazma
  data/
    events.py               # olay adı ayrıştırma, zero-shot bölme üretimi
    io.py                   # tifffile okuma, nodata maskesi
    dataset.py              # BrightDataset -> {pre, post, label, valid}
    transforms.py           # üç diziye aynı crop/flip/rot90
    sampler.py              # seed+step sonsuz örnekleyici
    stats.py                # SAR ortalama/std yalnız eğitim kümesinden
  models/
    base.py                 # DamageModel sözleşmesi: forward(pre, post) -> {loc_logits, clf_logits}
    registry.py             # @register("changemamba_mmbda_tiny") ...
    changemamba.py          # third_party sarmalayıcısı
    damageformer.py         # yedek
    heads/                  # Y4 varyantının tek fikri burada
  losses.py                 # CE + Lovász (fp32)
  train/
    loop.py                 # adım döngüsü; Türkiye'yi bilmez
    state.py                # tam durum kaydı/devam (model, EMA, optim, sched, adım, RNG)
    ema.py  schedule.py  log.py
  eval/
    inference.py            # tam kare/kayan pencere, TTA, olasılık önbelleği yazımı
    metrics.py              # karışıklık matrisi, IoU, makro F1 (saf NumPy)
    instances.py            # pikselden binaya geçiş
    bootstrap.py            # blok ve eşleştirilmiş bootstrap
    final.py                # Türkiye'de tek değerlendirme; guard ister
  analysis/                 # yalnız önbellek + dış katmanlar okur, model import etmez
    references.py           # Y1
    triage.py               # Y2
    sar_restd.py            # Y3 (transdüktif)
    gallery.py              # hata galerisi
  preflight/
    gates.py                # 11 kapı; her biri GateResult döndürür -> preflight.json
  cli.py                    # hasar preflight | train | final | analyze
third_party/
  changemamba/  bright/     # sabit commit, LICENSE, PATCHES.md
notebooks/
  01_colab_egitim.ipynb     # yalnız cli çağrıları
tests/                      # sentetik, veri ve GPU gerektirmez
```

### 2.3 Sözleşmenin kendisi (taslak)

```python
class DamageModel(Protocol):
    def forward(self, pre: Tensor, post: Tensor) -> dict[str, Tensor]:
        """pre: B×3×H×W optik, post: B×1×H×W SAR.
        Dönüş: {"loc": B×2×H×W, "clf": B×4×H×W} logit."""
```

Bu sözleşme üç şeyi ucuzlatır: yedeğe geçiş (Bölüm 5'teki otomatik kural tek satırlık config değişikliği olur),
Y4 varyantı (yalnız `heads/` altına bir dosya) ve Faz 2'de Zenodo checkpoint'lerinin aynı değerlendirme hattından
geçirilmesi (her biri için ince bir sarmalayıcı).

**Bilinçli olarak eklenmeyenler:** Hydra, eklenti sistemi, soyut "Trainer" sınıf hiyerarşisi. İki tam eğitimli bir
projede bunlar esneklik değil hata yüzeyi getirir (eski raporun Hydra yargısıyla aynı).

---

## 3. Yenilik değerlendirmesi

### 3.1 Bu proje bugün yenilikçi mi?

**Kısa cevap: titiz, ama yenilikçi değil.** Projenin en güçlü yanı yöntem disiplini (kilit, yeniden üretim, sınıf
kırılımı); bu bir işe alım değerlendiricisi için değerli, ama "yenilik" değil. Y1–Y4'ün literatürdeki öncülleri:

| # | Fikir | Bulunan en yakın öncül | Kalan yenilik | Değerlendirme |
|---|---|---|---|---|
| Y1 | Aynı binalarda referans anlaşmazlığı | Drone ile uydu etiketleri aynı binalarda %29 anlaşmıyor ([arXiv 2505.08117](https://arxiv.org/html/2505.08117v1)); ARIA DPM, Copernicus ve Microsoft CNN saha sayımına karşı kıyaslandı ([Ainscoe ve ark.](https://www.nature.com/articles/s43247-025-02623-4)) | VHR SAR modeliyle, görülmemiş olay protokolünde ve aynı binalarda üç kaynak: bulunamadı | **Orta, en güçlüsü** |
| Y2 | Analist bütçeli önceliklendirme | ALTIS, sel hasarını sabit saha bütçesi altında sıralama problemi olarak kuruyor ve IoU'nun önceliklendirmede yanıltabileceğini savunuyor ([arXiv 2603.13803](https://arxiv.org/pdf/2603.13803)); Damage-TriageFormer sınıfları müdahale maliyetine bağlıyor ([arXiv 2606.12248](https://arxiv.org/pdf/2606.12248)) | Depremde, zero-shot, DPM sıralamasına karşı: bulunamadı | **Zayıf-orta** (çerçeve var, uygulama yeni) |
| Y3 | Etiketsiz SAR yeniden standartlaştırma | AdaBN tarzı istatistik uyarlaması 2016'dan beri bilinen bir teknik; hasar için etiketsiz test zamanı uyarlaması DAVI'de var (optik, xBD) ([arXiv 2406.08020](https://arxiv.org/pdf/2406.08020)) | Yalnız ölçüm; teknik yeni değil | **Zayıf** |
| Y4 | Bina ayak izinde havuzlayan başlık | BRIGHT'ın nesne oylaması (+0,69); CVPRW26 kazananları zaten örnek düzeyinde çalışıyor ([sonuç makalesi](https://arxiv.org/abs/2607.22746)) | Oylamanın öğrenilen hâli; küçük artım | **Zayıf** |

**Bir rakip daha:** Mart 2026'da MambaBDA üzerine odak kaybı, dikkat kapıları ve hizalama modülü ekleyen ve
"görülmemiş afetlerde %27'ye kadar kazanç" bildiren bir çalışma çıktı; Türkiye depremini veri olarak anıyor ama
BRIGHT'ı ve protokolünü anmıyor ([arXiv 2603.01116](https://arxiv.org/abs/2603.01116)). **Yargı:** README'deki rakip
tablosuna girmeli; metrik türü belirsiz olduğu için 41,28 ile kıyaslanamaz (kural 4).

### 3.2 Yenilik nerede kurulabilir

**Yargı.** Bu projenin kısıtları (iki tam eğitim, Türkiye kilidi) model tarafında yeniliği neredeyse imkânsız
kılıyor: eski raporun kendi kanıtına göre mimari farklar Türkiye'de gürültü düzeyinde. Aynı kısıtlar
**değerlendirme biliminde** ise avantaj: kilitli protokol, hazır checkpoint'ler ve birden çok referans, kimsenin
bir arada sormadığı bir soruyu sormaya izin veriyor.

**Önerilen ana araştırma sorusu:**

> *Görülmemiş bir depremde hasar modellerinin sıralaması, (a) hangi referansa karşı ölçüldüğüne ve (b) sonucun
> istatistiksel belirsizliğine göre değişiyor mu?*

Bu soru yanlışlanabilir (sıralama değişmeyebilir; o da bulgu), keskin bir başlık üretir ("Referans değişince
şampiyon değişiyor" ya da "değişmiyor") ve neredeyse tamamı eğitimsizdir. Parçaları:

| # | Yeni iş | Ne gösterir | Ek tam eğitim | Kısıt notu |
|---|---|---|---|---|
| N1 | **Yayımlanmış sıralamanın güven aralıkları:** Zenodo'daki 5 Türkiye zero-shot checkpoint'i + kendi modelimiz için blok bootstrap ve eşleştirilmiş farklar | 41,28 / 38,32 / 37,82 / 37,20 ... farklarının hangileri gürültüden ayrılabiliyor? Tablo 10 tek koşudan; bunu ölçen bulunamadı. | 0 | Hazır checkpoint'lerin Türkiye çıktısı Faz 2 istisnasıyla zaten üretiliyor; kararı etkilememesi için analiz `protokol-kilidi`'nden sonra yapılmalı. **Kullanıcı kararı.** |
| N2 | **Referansa göre sıralama:** aynı 6 model, BRIGHT etiketi, HOT/OSM ve DPM'e karşı ayrı ayrı (Y1'in genişlemesi) | Sıralama referans seçimine dayanıklı mı? | 0 | Y1 birleştirme şartnamesi (B3) önce yazılmalı |
| N3 | **Fiziksel dördüncü eksen: yer sarsıntısı.** USGS ShakeMap PGA katmanına göre hataların kırılımı | Model hataları sarsıntı şiddetiyle mi, görüntü özellikleriyle mi açıklanıyor? Hiçbir optik/SAR etiketine bağlı olmayan bağımsız bir eksen. | 0 | Yalnız analiz, hiçbir ayara girmez |
| N4 | **SAR'ın marjinal değeri:** Y4'e aday olarak *modalite bırakma* (eğitimde SAR'ı rastgele sıfırlama). Tek eğitimle aynı model hem optik-yalnız hem birleşik çalışabilir. | "Görülmemiş depremde VHR SAR gerçekten katkı veriyor mu?" sorusuna temiz cevap. Bina havuzlama başlığından daha ilginç bir soru. | 0 (Y4'ün yerine) | Y4 tanımı zaten Faz 5'te seçiliyor; aday listesine eklenir |

**Bırakılması ya da küçültülmesi önerilenler:** Y3 tek başına yenilik iddiası taşımıyor; ayrı transdüktif tabloda
bir satır olarak kalabilir, "yenilik paketi"nden çıkarılmalı. Y4 olarak bina havuzlama başlığı yerine N4 daha güçlü
bir soru soruyor; seçim yine 13 olayın doğrulamasıyla Faz 5'te yapılır.

### 3.3 Yeniliği nasıl korursun

- **Önce soruyu, sonra sonucu yaz:** N1–N3'ün hipotezleri ve analiz kodu `protokol-kilidi`'nden önce commit'lenir.
  Önceden kayıtlı bir değerlendirme çalışması, sonradan anlatılandan çok daha inandırıcıdır.
- **"İlk" deme, "bulunamadı" de:** Taramanın tarihi ve sorguları belgede durur (bu rapor ve öncekiler).
- **Yenilik tek cümleye sığmalı:** README'nin ilk paragrafı model skoruyla değil, ana araştırma sorusunun
  cevabıyla başlar; 41,28 kıyası ikinci paragrafta.
- **Olumsuz sonuç başlık olabilir:** "Sıralama referansa göre değişmiyor" da yayımlanabilir bir bulgudur.

---

## 4. Kullanıcının kararları

| # | Karar | Seçenekler ve maliyet | Önerim |
|---|---|---|---|
| K1 | Protokol belgesi (B1) | (a) Faz 0'dan önce yaz, ~1 oturum · (b) Faz 2 yeniden üretiminden sonra yaz | (a); değerlendirici kodu okunarak, yeniden üretim ona karşı test olur |
| K2 | Ana araştırma sorusu | (a) Mevcut Y1–Y4 · (b) Önerilen soru + N1–N3, Y3 küçültülür · (c) (b) + N4 Y4 adayı olur | (c); ek tam eğitim yok, 0–5 CU |
| K3 | N1 için hazır checkpoint'lerin Türkiye çıktısını kullanmak | Kural 1'in Faz 2 istisnasının genişletilmesi demek. (a) Kabul, kilitten sonra · (b) Ret, N1 yalnız kendi modelimizle | (a); seçim ve ayara girmez, `CLAUDE.md`'ye tek satırla yazılır |
| K4 | Modüler yapı | (a) Bölüm 2 olduğu gibi · (b) Daha düz: `data/`, `models/`, `train/`, `eval/` ve tek `analysis.py` | (a); fark küçük, ama guard ve önbellek düzeni kilit kuralını koda bağlıyor |
| K5 | ONNX | (a) Faz 3'te 10 dakikalık deneme · (b) Faz 8'e kadar bekle | (a) |

## Kaynaklar

- [HOTOSM Turkey Destroyed Buildings (HDX)](https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings)
- [Ainscoe ve ark., radar ve optik hasar haritaları, Communications Earth & Environment](https://www.nature.com/articles/s43247-025-02623-4)
- [Now you see it, Now you don't: Damage Label Agreement in Drone & Satellite Post-Disaster Imagery](https://arxiv.org/html/2505.08117v1)
- [ALTIS: Sentinel-1 ile mülk düzeyi sel hasarı önceliklendirme](https://arxiv.org/pdf/2603.13803)
- [Damage-TriageFormer](https://arxiv.org/pdf/2606.12248)
- [DAVI: Vision foundation model ile genellenebilir hasar tespiti](https://arxiv.org/pdf/2406.08020)
- [Improved MambaBDA Framework for Robust Building Damage Assessment Across Disaster Domains](https://arxiv.org/abs/2603.01116)
- [2026 Bright Challenge sonuç makalesi](https://arxiv.org/abs/2607.22746)
- [BRIGHT, ESSD](https://essd.copernicus.org/articles/17/6217/2025/)
- [Li ve ark., ShakeMap ve uydu ile nedensellik bilgili Bayesçi hasar tahmini](https://journals.sagepub.com/doi/10.1177/87552930241290501)
