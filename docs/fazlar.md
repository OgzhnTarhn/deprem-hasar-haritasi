# Faz Planı

Her faz ayrı bir Claude Code oturumunda yapılır ve kendi çıkış koşulu sağlanmadan sonraki faza geçilmez. Kod işleri
orkestra ile yapılır; her fazın başında oturuma şu verilir:

```
/orkestra docs/fazlar.md içindeki Faz N'i uygula. Önce CLAUDE.md ve docs/tasarim.md'yi oku.
```

Faz bitince: çıkış koşulları doğrulanır, değişiklikler atomik commit'lere bölünür, `docs/durum.md` güncellenir.

## Yenilik paketi

Rakip projeler model skoruna odaklanıyor. Bu projenin yeniliği, tek doğru yapılmış eğitimin üstüne kurulan dört
parça. Dördü de mevcut açık projelerde birlikte bulunamadı (araştırma raporu, Bölüm "Rakipler"); bu "yok" değil,
"bulunamadı" demek.

| # | Yenilik | Soru | Ek tam eğitim | Faz |
|---|---|---|---|---|
| Y1 | **Üç referanslı anlaşmazlık haritası** | Aynı binalarda BRIGHT optik yorum etiketi, HOT/OSM yıkık bina katmanı ve öğrenmesiz radar haritası (ARIA DPM) ne kadar anlaşıyor; model kime göre "yanlış"? | 0 | 6 |
| Y2 | **Analist bütçeli önceliklendirme** | Bir uzman binaların yalnız %5, %10 veya %20'sini kontrol edebiliyorsa model hangilerini göstermeli; hasarlı ve yıkık binaların yüzde kaçı yakalanır? | 0 | 6 |
| Y3 | **Etiketsiz SAR uyarlaması** | Türkiye'nin yalnız SAR görüntü istatistikleriyle (etiket yok, gradyan yok) giriş yeniden standartlaştırılırsa ne değişir? Ayrı "transdüktif" tabloda raporlanır. | 0 | 6 |
| Y4 | **Tek fikirli varyant** | Aday: bina ayak izi içinde özellikleri havuzlayıp her binaya tek karar veren hasar başlığı. Seçim Faz 5'te yalnız 13 olayın doğrulamasıyla. | 1 | 5, 7 |

Neden bu dördü:

- **Y1:** Asıl bulgu skor değil, referansların ayrışması olabilir. Saha sayımına göre Copernicus'un elle haritası F1
  0,15–0,23, öğrenmesiz radar 0,47 aldı; BRIGHT etiketi saha değil optik yorum.
- **Y2:** Afet müdahalesinde gerçek kısıt analist zamanı. Belirsizliği "ECE şu" diye değil, operasyonel bir sayıya çevirir.
- **Y3:** BRIGHT makalesi SAR hasar imzasının olaydan olaya kaydığını gösteriyor. Uygulaması eğitim istemiyor.
- **Y4:** Bina düzeyinde oylama son işlemi makalede +0,69 `[ZS-EVENT mIoU]` kazandırdı; bunun öğrenilen hâli tek ve
  savunulabilir bir fikir.

Y3 ve Y4 iddia değil, ölçümdür: işe yaramazlarsa bu da raporlanır.

## Faz 0: Depo iskeleti (Hafta 1)

- `pyproject.toml` (uv; torch sabitlenmez), `src/hasar/` paketi, `configs/ana.yaml`, `tests/`, `notebooks/`, `results/`.
- `ruff`, `pytest`, `nbstripout`, GitHub Actions (CPU, veri yok).
- `docs/durum.md` ve `docs/adr/0001-*.md` (ADR şablonu).

**Çıkış:** CI yeşil, `uv run pytest` ve `uv run ruff check .` temiz.

## Faz 1: Veri katmanı (Hafta 1–2)

- Hugging Face'ten indirme ve `.done` işaretli açma.
- Olay adı ayrıştırma (`CLAUDE.md` tuzakları), zero-shot bölme üreticisi: Türkiye eğitim ve doğrulamada bulunursa hata.
- Nodata maskesi; SAR ortalama ve standart sapması yalnız eğitim kümesinden.
- Optik, SAR ve etikete aynı kırpma, flip, rot90; `seed + step` ile sonsuz örnekleyici.
- Sentetik GeoTIFF'lerle testler.

**Çıkış:** Testler geçer; kare sayıları ve sınıf dağılımı `results/veri/bolme.json` dosyasında.

## Faz 2: Metrik ve yeniden üretim (Hafta 2–3, ilk Colab, ~5–8 CU)

- Resmî değerlendiriciyle aynı olay düzeyi mIoU, sınıf başına IoU, bina düzeyi makro F1, kareler üzerinde bootstrap.
- Zenodo 15349462'deki hazır zero-shot checkpoint'lerin çıkarımı: Türkiye için SiamCRNN, UNet, DamageFormer;
  denetim için bir başka deprem.
- Metriğin uç durum testleri (kusursuz 1,0; hep arka plan ≈ 0,21).

**Çıkış:** Türkiye'de 38,32 / 37,82 / 37,20 `[ZS-EVENT mIoU]` ±0,3 içinde (eğitim yok, sonuç belirlenimci olmalı).
Tutmazsa sonraki faza geçilmez.

## Faz 3: Model, eğitim motoru ve ön kontrol (Hafta 3–4, ~10–15 CU)

- ChangeMamba çekirdek derleme kapısı (en çok 1 saat); geçmezse otomatik olarak Swin-T DamageFormer.
- Eğitim döngüsü: AdamW, warmup ve kosinüs, bf16, EMA, gradyan kırpma, atomik kayıt, tam durumla devam, JSONL log.
- Doğrulama yalnız 13 olayda; eğitim kodunda Türkiye'yi değerlendirmenin yolu yok.
- 7 hücreli ince notebook (`tasarim.md` Bölüm 8) ve 11 ön kontrol kapısı.

**Çıkış:** `preflight.json` içindeki bütün kapılar geçer; CU projeksiyonu bütçede; `protokol-kilidi` etiketi atılır.

## Faz 4: Ana eğitim (Hafta 5–6, 50–75 CU)

- ChangeMamba olduğu gibi, 40.000 adım, yeni ayın CU'su ile.
- Kopma olursa aynı notebook "Run All" ile devam eder.

**Çıkış:** `DONE` işareti, doğrulama eğrisi ve seçilen kayıt noktası `results/ana/` altında. Türkiye'ye dokunulmadı.

## Faz 5: Varyant kilidi ve Türkiye değerlendirmesi (Hafta 6, < 5 CU)

1. Varyant (Y4) 13 olayın doğrulama hatalarına bakılarak yazılır; yapılandırması `varyant-kilidi` etiketiyle commit'lenir.
2. Ancak bundan sonra ana model Türkiye'de bir kez değerlendirilir: başlık satırı ve önceden ilan edilen ikincil satırlar.

**Çıkış:** `results/ana/turkiye.json`; başlık sonucu 41,28 ile kıyaslanmış ve sonuç ne olursa olsun yazılmış.

## Faz 6: Yenilik analizleri (Hafta 6–10, 2–5 CU)

- Y1: HOT/OSM ve ARIA DPM katmanlarını aynı bina ayak izlerine bağla, anlaşmazlık tablosu, elle denetlenmiş 100–200 bina.
- Y2: Bina düzeyi belirsizlik (TTA entropisi), risk-kapsam eğrisi, analist bütçesi tablosu.
- Y3: Etiketsiz SAR yeniden standartlaştırma; ayrı transdüktif tablo.
- SAR hata galerisi: Capella ve Umbra kırılımı, layover ve gölge, izole hasarlı binalar.

**Çıkış:** Her analizin JSON'u ve görselleri; olumsuz bulgular dahil `docs/bulgular.md`.

## Faz 7: Varyant eğitimi (Hafta 11–13, 50–75 CU)

- Kilitli varyant, aynı reçete ve aynı ön kontrollerle bir kez eğitilir ve Türkiye'de bir kez değerlendirilir.

**Çıkış:** İki model yan yana; fark 1–2 puanın altındaysa "eşit". Seçim Türkiye sonucuna göre yapılmaz.

## Faz 8: Ürünleştirme (Hafta 14–16, < 2 CU)

- ONNX dışa aktarımı ve bina başına gecikme ölçümü.
- Model kartı, lisans ve atıf tablosu, rakip karşılaştırma tablosu.
- README: bütün sayılar `results/` dosyalarından üretilir.
- Demo sayfası: Türkiye bölgesinde önce/sonra görüntüleri, tahmin, üç referans ve analistin kontrol listesi.
- CV ve LinkedIn metni.

**Çıkış:** Yabancı biri depoyu tek komutla çalıştırabiliyor; demo yayında.
