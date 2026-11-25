# Faz Planı

**Güncelleme:** 21 Kasım 2026; ADR-0007 ve protokol 1.3. Faz numaraları korunur; iki fine-tune aynı Colab akışındadır.

Her faz ayrı bir Claude Code oturumunda yapılır. Kendi çıkış koşulu sağlanmadan sonraki faza geçilmez. Kod işleri
orkestra ile, belge işleri doğrudan yapılır. Her fazın başında oturuma şu verilir:

```
docs/fazlar.md içindeki Faz N'i uygula. Önce CLAUDE.md, docs/tasarim.md ve docs/durum.md'yi oku.
```

Faz bitince: çıkış koşulları doğrulanır, değişiklikler atomik commit'lere bölünür, `docs/durum.md` güncellenir.

## Genel bakış

| Faz | Ad | Hafta | CU | Tür | Kilit / karar |
|---|---|---|---|---|---|
| 0 | Belgeler ve protokol taslağı | 1 | 0 | Belge | |
| 1 | Depo iskeleti | 1–2 | 0 | Kod | |
| 2 | Veri katmanı | 2–3 | 0 | Kod + araştırma | |
| 3 | Metrik ve yeniden üretim | 3–4 | 5–8 | Kod + Colab | |
| 4 | Model, eğitim motoru, ön kontrol | 4–5 | 10–15 | Kod + Colab | `protokol-kilidi` |
| 5 | Kontrol fine-tune | 5–6 | ≤ 15 hedef; ölçülmedi | Colab | İki kilitten sonra |
| 6 | Türkiye değerlendirmesi ve kıyas | 6–7 | Ölçülmedi | Colab + yerel | KN1 eğitim öncesinde kapandı |
| 7 | Analizler | 7–10 | 2–5 | Kod (CPU ağırlıklı) | KN2 |
| 8 | SAR varyantı fine-tune | Ana sonrası aynı akış | ≤ 15 hedef; ölçülmedi | Colab | KN3 |
| 9 | Demo ve ürünleştirme | 14–16 | < 2 | Kod + belge | |
| Ek | Türkiye few-shot A/B | Zero-shot sonrası | Ölçülmedi | Kod + Colab | `fewshot-kilidi` |

Faz 5, 6 ve 8 bitti (22 Kasım 2026); sonuçlar [durum.md](durum.md) ve `results/` altında.
İki fine-tune ve ön kontroller için aylık 100 CU içinde kalma hedefi var. Koşu başına ≤ 15 CU
ölçülmedi; kapı 8 eğitim projeksiyonunu ölçer. Ön kontrol bu projeksiyonun dışındadır.

## Yenilik paketi ve fikir havuzu

Rakip projeler model skoruna odaklanıyor. Bu projenin yeniliği, tek doğru yapılmış eğitimin üstüne kurulan
değerlendirme işleri. Planlı olanlar:

| # | Yenilik | Soru | Ek tam eğitim | Faz |
|---|---|---|---|---|
| Y1 | **Referans anlaşmazlığı haritası** | Aynı binalarda BRIGHT optik yorum etiketi, HOT/OSM yıkık bina katmanı ve öğrenmesiz radar haritası (ARIA DPM) ne kadar anlaşıyor; model kime göre "yanlış"? HOT/OSM de optik yoruma dayandığı için çerçeve "iki optik yorum + bir radar sinyali". | 0 | 7 |
| Y2 | **Analist bütçeli önceliklendirme** | Uzman binaların yalnız %5, %10 veya %20'sini kontrol edebiliyorsa model hangilerini göstermeli? Taban: rastgele sıra ve DPM sırası. | 0 | 7 |
| Y3 | **Etiketsiz SAR uyarlaması** | Türkiye'nin yalnız SAR istatistikleriyle giriş yeniden standartlaştırılırsa ne değişir? Ayrı transdüktif tablo; yenilik iddiası taşımaz. | 0 | 7 |
| Y4 | **Tek fikirli varyant** | A4 SAR radyometrik dayanıklılığı KN1’de, 21 Kasım 2026’da seçildi. Diğer fikirler: [fikir-havuzu.md](fikir-havuzu.md). | 1 | 6, 8 |

KN1 sonucu (21 Kasım 2026): A4 seçildi; dayanak yalnız 13 olayın SAR ölçeği ve hazır DamageFormer
doğrulaması. Topluluk 1.3 kapsamı dışında. Varyantın ana modelin Türkiye sonucundan önce kilitlenmesi sürer.
Karar: [ADR-0007](adr/0007-ana-model-hazir-damageformer-fine-tune.md). Diğer adaylar fikir havuzunda saklanır.

---

## Faz 0: Belgeler ve protokol taslağı (Hafta 1, belge)

Kod yazılırken sessizce karara dönüşecek belirsizlikleri önceden yazıya geçirmek.

- `docs/adr/` ve şablon; geriye dönük ADR'ler: zero-shot bölme, ChangeMamba ana ve DamageFormer yedek, SAR
  normalizasyonu, başlık metriği yolu, DINOv3/MaRS ve `standard_ML` checkpoint dışlaması, modüler kod yapısı.
- `docs/protokol.md` taslağı (önceden kayıt). İçinde olacaklar:
  - Başlık mIoU'nun tam hesabı: karışıklık matrisi olay boyunca mı biriktiriliyor, arka plan ortalamaya giriyor
    mu, değerlendirmede nodata nasıl işleniyor, tam kare mi kayan pencere mi. BRIGHT değerlendirici kodu okunarak
    yazılır.
  - Pikselden binaya geçiş kuralı (Faz 2 araştırmasıyla kesinleşir).
  - Bootstrap: mekânsal blok veya şehir birimi, eşleştirilmiş fark, yeniden örnek sayısı, aralık türü.
  - Tek "eşit" kuralı (ör. eşleştirilmiş farkın %95 aralığı 0'ı içeriyorsa veya fark 1,0 puanın altındaysa).
  - İkincil satırlar ve ayar kaynakları.
- Veri ve lisans tablosu (`docs/veri-lisans.md`): BRIGHT, Zenodo, HOT/OSM, ARIA DPM, Maxar Open Data; Ukrayna,
  Myanmar ve Meksika optiğinin kaynak notu.
- Risk kaydı (`docs/riskler.md`): Colab imajı değişir, HF kopyası kalkar, Zenodo checkpoint'i yeni kodla yüklenmez,
  ChangeMamba ONNX'e çıkmaz, dış katman taşınır; her biri için tetik ve yedek.
- SAR temelleri notu (`docs/sar-temelleri.md`): çift sekme, layover, gölge, benek, Capella/Umbra farkı.

**Çıkış:** Belgeler var; protokol taslağında yalnız Faz 2–3'te ölçülecek maddeler "açık" olarak işaretli.

## Faz 1: Depo iskeleti (Hafta 1–2)

Hedef düzen ve ilkeler: [rapor, Bölüm 2](../reports/Eksik%20belgeler,%20modüler%20yapı%20ve%20yenilik%20değerlendirmesi.md).

- `pyproject.toml` (uv; torch sabitlenmez), `src/hasar/` alt paketleri (`data`, `models`, `train`, `eval`,
  `analysis`, `preflight`), `configs/ana.yaml`, `tests/`, `notebooks/`, `results/`, `third_party/`.
- `config.py` (dataclass şeması, config hash), `results.py` (sonuç JSON şeması, `metric_tag` zorunlu, atomik yazma),
  `guard.py` (Türkiye karelerine tek erişim yolu; `protokol-kilidi` etiketi ve temiz çalışma ağacı denetimi).
- `models/base.py` model sözleşmesi ve `models/registry.py`.
- `ruff`, `pytest`, `nbstripout`, GitHub Actions (CPU, veri yok).

**Çıkış:** CI yeşil; `uv run pytest` ve `uv run ruff check .` temiz; guard, şema ve config hash testleri geçer.

## Faz 2: Veri katmanı (Hafta 2–3)

- Hugging Face'ten indirme ve `.done` işaretli açma.
- Olay adı ayrıştırma (`CLAUDE.md` tuzakları), zero-shot bölme üreticisi: Türkiye eğitim ve doğrulamada bulunursa hata.
- Nodata maskesi; SAR ortalama ve standart sapması yalnız eğitim kümesinden.
- Optik, SAR ve etikete aynı kırpma, flip, rot90; `seed + step` ile sonsuz örnekleyici.
- Sentetik GeoTIFF'lerle testler.
- **Araştırma ve sayım (Türkiye etiketine bakmadan, yalnız meta veri):**
  - BRIGHT'ın bina örnek etiketleri var mı; yoksa bağlı bileşenle bitişik binaların birleşme oranı (13 olayda).
  - Türkiye karelerinin şehir ve mekânsal blok dağılımı (bootstrap birimi için; yalnız koordinatlar).
  - BRIGHT SAR uint8 değerlerinin ölçeği: doğrusal genlik mi, dB mi (A3 için).
  - **A1 fizibilitesi:** 14 olayın kaçı için uygun Sentinel-1 öncesi/sonrası çift var (2–3 gün).

**Çıkış:** Testler geçer; kare sayıları ve sınıf dağılımı `results/veri/bolme.json`'da; araştırma sonuçları protokol
taslağına ve `fikir-havuzu.md`'ye işlendi.

## Faz 3: Metrik ve yeniden üretim (Hafta 3–4, ilk Colab, ~5–8 CU)

- Resmî değerlendiriciyle aynı olay düzeyi mIoU, sınıf başına IoU, bina düzeyi makro F1; blok ve eşleştirilmiş
  bootstrap.
- Zenodo 15349462'deki hazır zero-shot checkpoint'lerin çıkarımı: Türkiye için SiamCRNN, UNet, DamageFormer;
  denetim için bir başka deprem. Aynı checkpoint'lerin 13 olaydaki doğrulama çıktıları da kaydedilir (B2 topluluk
  adayı için).
- Metriğin uç durum testleri (kusursuz 1,0; hep arka plan = arka plan piksel oranı / 4, doğrulama kümesinin etiket dağılımından).

**Çıkış:** Türkiye'de 38,32 / 37,82 / 37,20 `[ZS-EVENT mIoU]` ±0,3 içinde (eğitim yok, sonuç belirlenimci olmalı).
Tutmazsa sonraki faza geçilmez.

## Faz 4: Model, eğitim motoru ve ön kontrol (Hafta 4–5, ~10–15 CU)

- Güncel model hazır DamageFormer zero-shot; md5 yükleme denetimi var, özel çekirdek gerekmiyor.
- **ONNX denemesi** (10 dakika): ChangeMamba dışa aktarılabiliyor mu? Sonuç risk kaydına ve Faz 9 planına.
- Eğitim döngüsü: AdamW, warmup ve kosinüs, bf16, EMA, gradyan kırpma, atomik kayıt, tam durumla devam, JSONL log.
- Doğrulama yalnız 13 olayda; eğitim kodunda Türkiye'yi değerlendirmenin yolu yok.
- SAR `official` biçimde kalır. A4 yalnız varyantın eğitiminde uygulanır; değerlendirmede yoktur.
- 7 hücreli ince notebook (`tasarim.md` Bölüm 8) ve 11 ön kontrol kapısı.
- `docs/protokol.md` son hâli.

**Çıkış:** İki yapılandırma ve protokol 1.3 kullanıcı tarafından aynı commit’te iki etiketle kilitlenir.
Defterde her koşunun bütün ön kontrol kapıları geçmeden o koşunun eğitimi başlamaz.

## Faz 5: Kontrol fine-tune

- `configs/ana.yaml`: hazır DamageFormer zero-shot başlangıcı, 4.000 adım; SAR artırımı yok.
- Kullanıcı protokol 1.3'ü içeren commit'e `protokol-kilidi`ni taşır; aynı commit'e `varyant-kilidi` koyar.
- Tek “Run All” önce ana için ön kontrol → fine-tune → `sicaklik` → `final`, sonra varyant için aynı sırayı izler.
  Defter `origin/main` kullanır; dal değişiklikleri oraya alınmış olmalı.
- Kopmada `last.pt` ile devam eder; `DONE` varsa eğitim atlanır. `best.pt` son adımın EMA'sıdır.

**Çıkış:** Ana `DONE`, doğrulama kaydı ve kilitliyse tek `final`. ≤ 15 CU hedefi ölçülmedi; kapı 8 ölçer.

## Faz 6: Türkiye değerlendirmesi ve kıyas

KN1 eğitimden önce kapandı: A4 seçildi (21 Kasım 2026). İki tasarım da ana modelin Türkiye sonucundan önce kilitli.
Her koşu `sicaklik` için yalnız 13 olayın doğrulamasını kullanır; ardından Türkiye'de bir kez `final` çalışır.
`turkiye.json` varsa tekrar çalışmaz; yarım iş önbellekten devam eder.

**Çıkış:** `results/ana/turkiye.json`, varyant sonrası `results/varyant/turkiye.json`.
Birincil kıyas `hasar kiyas` ile varyant − ana; ikincil kıyaslar hazır DamageFormer referansına karşı.
Eşleştirilmiş blok bootstrap ve protokol §9 “eşit” kuralı kullanılır. ChangeMamba 41,28 `[ZS-EVENT mIoU]` yalnız bağlamdır.

## Faz 7: Analizler (Hafta 7–10, 2–5 CU)

Bütün analizler Faz 6'nın tahmin önbelleğini okur; model yeniden çalıştırılmaz.

- Y1: HOT/OSM ve ARIA DPM katmanlarını aynı binalara bağla, anlaşmazlık tablosu, elle denetlenmiş 100–200 bina.
- Y2: Bina düzeyi belirsizlik (TTA entropisi), risk-kapsam eğrisi, analist bütçesi tablosu, DPM tabanı.
- Y3 ve KN1'de seçildiyse C1: ayrı transdüktif tablo.
- KN1'de seçilen N analizleri (sıralama güven aralıkları, referansa göre sıralama, ShakeMap kırılımı).
- SAR hata galerisi: Capella ve Umbra kırılımı, layover ve gölge, izole hasarlı binalar.
- **KN2:** Raporlama derinliği; üçüncü tam eğitim isteniyor mu (istenirse "keşif" etiketiyle, kullanıcı kararı).

**Çıkış:** Her analizin JSON'u ve görselleri; olumsuz bulgular dahil `docs/bulgular.md`.

## Faz 8: SAR varyantı fine-tune

- Aynı defter ana sonrası `configs/varyant.yaml` için ön kontrol → fine-tune → `sicaklik` → `final` çalıştırır.
- Ortak başlangıç ve reçete korunur; tek fark eğitimde SAR radyometrik artırımıdır (A4).
- 4.000 adım; kayıt noktası son adımın EMA'sı. ≤ 15 CU hedefi ölçülmedi; kapı 8 ölçer.
- Topluluk 1.3 kapsamı dışında. **KN3:** README ve demoya girecek bulgular.

**Çıkış:** İki fine-tune ve referans birlikte raporlanır. Fark protokolün “eşit” kuralına göre yazılır;
Türkiye sonucuna göre model veya ayar seçilmez.

## Faz 9: Demo ve ürünleştirme (Hafta 14–16, < 2 CU)

- **Demo, gerçek 6 Şubat verisiyle:** BRIGHT'ın Türkiye kareleri (Maxar optik + Capella/Umbra SAR). Harita
  üzerinde önce/sonra görüntüleri, model tahmini, üç referans ve analistin kontrol listesi. Tahminler Faz 6
  önbelleğinden gelir; demo için model yeniden çalıştırılmaz.
- BRIGHT dışındaki 6 Şubat bölgeleri (DM2) yalnız açık bir afet sonrası VHR SAR kaynağı doğrulanırsa eklenir.
  İstanbul hasar demosu yapılmaz: deprem olmadığı için afet sonrası görüntü yok ([fikir-havuzu.md](fikir-havuzu.md)).
- Faz 4'teki ONNX sonucuna göre dışa aktarım (ONNX, `torch.export` veya PyTorch) ve bina başına gecikme ölçümü.
- Model kartı, lisans ve atıf tablosu (Maxar Open Data CC BY-NC 4.0 dahil), rakip karşılaştırma tablosu.
- README: bütün sayılar `results/` dosyalarından üretilir.
- CV ve LinkedIn metni.

**Çıkış:** Yabancı biri depoyu tek komutla çalıştırabiliyor; demo yayında.

## Ek deney: Türkiye few-shot A/B (protokol 1.4 §15)

Zero-shot sonuçları kesinleştikten sonra eklendi ([ADR-0008](adr/0008-turkiye-few-shot-ek-deneyi.md),
[tasarım](superpowers/specs/2026-11-23-turkiye-few-shot-design.md)). Soru: Türkiye etiketlerinin küçük bir kısmı ve
hasar sınıflarına kayıp ağırlığı, hasarlı ve yıkık sınıflarındaki açığı kapatır mı?

- **Bölme:** `hasar fewshot-bolme` yalnız koordinattan `results/fewshot/bolme.json`'u üretir. Her şehirden 2 km
  bloklarla karelerin ~%20'si uyarlamaya, 0,5 km tampon atılır, kalanı test olur.
- **Kilit:** `fewshot-kilidi` protokolü, iki yapılandırmayı ve bölmeyi dondurur. Eğitim yalnız uyarlama, değerlendirme
  yalnız test karelerini okur (`FewshotAccess`).
- **Kollar:** İkisi de ana koşunun `best.pt`'sinden (sha256 kilitli), 4.000 adım, lr 2e-5, son adımın EMA'sı; her
  adımda yarı Türkiye uyarlama, yarı 13 olay (`MixedStepSampler`). Tek fark: A düz kayıp, B hasar ağırlıklı kayıp.
- **Değerlendirme:** `hasar fewshot-test` iki kol da bitince her model için bir kez; `hasar fewshot-kiyas` B − A
  (birincil), A ve B − zero-shot ana, sınıf bazlı B − A. Metrik etiketi `[FS-EVENT mIoU]`.

**Çıkış:** `results/fewshot/{a,b}/test.json`, `results/fewshot/kiyas/`; README'de ayrı few-shot tablosu.
