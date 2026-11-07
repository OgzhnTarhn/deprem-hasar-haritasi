# Deprem Hasar Haritası

Afet öncesi optik ve afet sonrası SAR uydu görüntülerinden bina hasarı haritalayan model. BRIGHT veri setiyle
eğitilir; 6 Şubat 2023 Türkiye depreminde, eğitimde hiç görülmeden **bir kez** test edilir. Çıpa: ChangeMamba
41,28 `[ZS-EVENT mIoU]`.

- Bağlayıcı tasarım: [docs/tasarim.md](docs/tasarim.md). Değişen karar orada ve ADR'de güncellenir.
- Faz planı ve yenilik paketi: [docs/fazlar.md](docs/fazlar.md). Güncel durum: [docs/durum.md](docs/durum.md); her faz sonunda güncellenir.
- Ertelenmiş fikirler ve karar noktaları (KN1–KN3): [docs/fikir-havuzu.md](docs/fikir-havuzu.md).
- Görsel genel bakış (bütün fazlar, her parça neye yarıyor): [docs/proje-haritasi.md](docs/proje-haritasi.md).
- Ders kitabı (26 ders, elle çözülmüş örnekler ve sorularla): [docs/dersler.md](docs/dersler.md).
- Bütün tekniklerin baştan sona anlatımı: [docs/ogrenme-rehberi.md](docs/ogrenme-rehberi.md). Teknik bir karar değişince güncellenir.
- Araştırma raporları: `reports/` (mimari ve rakip analizi; eksik belgeler, modüler yapı ve yenilik; ChangeMamba'nın ötesi).
- Ham notlar: `research_notes/`. Bu klasörlerin içeriği değiştirilmez; yeni araştırma yeni klasöre gider.

## Çalışma biçimi

- Kullanıcıya Türkçe yaz. Kod ve kod içi adlar İngilizce, açıklamalar Türkçe olabilir.
- Kod ve test işleri orkestra ile yapılır (genel kural). Belge ve ayar işleri doğrudan yapılır.
- Ürün kararları kullanıcıya maliyeti yazılı seçeneklerle getirilir, açık uçlu soru olarak değil.

## Git

- **Atomik commit:** bir commit tek bir mantıksal değişiklik. Karışık değişiklik ayrı commit'lere bölünür.
- **Claude imzası yok:** `Co-Authored-By` satırı yok, "Generated with Claude" satırı yok, PR açıklamalarında da yok.
- Commit mesajı kısa, Türkçe ve emir kipinde: `Veri yükleyiciye nodata maskesini ekle`.
- Commit ve push yalnız kullanıcı isteyince.
- Veri, checkpoint, zip, `.pt`/`.pth`, büyük çıktılar git'e girmez (`.gitignore`).

## Değişmez kurallar

1. **Türkiye kilitli.** Eğitimde, doğrulamada, model ve kayıt noktası seçiminde, eşik ve sıcaklık ayarında,
   normalizasyon istatistiklerinde Türkiye'nin ne görüntüsü ne etiketi kullanılır. Her model Türkiye'de yalnız
   `final` adımında bir kez değerlendirilir; bu adım protokol commit'i etiketlenmeden (`protokol-kilidi`) çalışmaz.
   Varyant modelin tasarımı, ana modelin Türkiye sonucu görülmeden `varyant-kilidi` etiketiyle kilitlenir.
   İki istisna, ikisi de başlık sonucunu etkilemez: Faz 3'te hazır zero-shot checkpoint'lerin yeniden üretimi ve
   Faz 7'de Türkiye'nin etiketsiz SAR istatistikleriyle yapılan, ayrı "transdüktif" tabloda raporlanan Y3 analizi.
   Fikir havuzundaki seçimler 13 olayın sonucuyla yapılır, Türkiye sonucuyla yapılmaz.
2. **Türkiye'yi görmüş ağırlık yok.** `*_standard_ML_split.pth` checkpoint'leri ve 2023 Türkiye sahnelerini
   ön eğitimde görmüş olabilecek modeller (ör. MaRS) başlangıç ağırlığı olamaz.
3. **İki tam eğitim, fazlası yok.** Ana model (ChangeMamba olduğu gibi) ve tek fikirli bir varyant, her biri bir kez.
   GPU saati harcamadan önce ön kontrol kapılarının hepsi geçmeli. Yeni bir tam eğitim eklemek kullanıcının kararıdır.
4. **Metrik etiketleri karışmaz.** Her sayının yanında türü yazılır: `[ZS-EVENT mIoU]`, `[STD-EVENT mIoU]`,
   `[STD-SET mIoU]`, `[DFC-TEST mIoU]`, `[CVPR-TEST segm_AP]`. Yalnız `[ZS-EVENT mIoU]` 41,28 ile kıyaslanır.
5. **Sayılar dosyadan gelir.** README ve raporlardaki her sayı `results/**/*.json` dosyasından üretilir; elle yazılmaz.
6. **Olumsuz sonuç da yazılır.** 1–2 puanın altındaki farklar "eşit" diye raporlanır.

## Bilinen tuzaklar

- `max_iters`: BRIGHT kodunda **örnek** sayısı (800.000 / 16 = 50.000 adım), ChangeMamba kodunda **adım** sayısı.
- Olay adı: önce `_pre_disaster` / `_post_disaster` eki atılır, sonra `rsplit('_', 1)[0]` alınır
  (`la_palma-volcano_00000001` → `la_palma-volcano`). `split('_')[0]` alt çizgili olay adlarında yanlış sonuç verir.
- `standard_ML` bölmesinde Türkiye eğitimde (777/113/224). Bu projede yalnız `cross_event/zero_shot` mantığı.
- Resmî betikler eğitim sırasında test olayını da değerlendiriyor; bizim kodda kapalı.
- ChangeMamba'nın eğitimdeki mIoU'su bina maskesi uygulamadan hesaplanıyor; başlık metriğinin yolu tasarımda kilitli.
- Bazı karelerde SAR tamamen boş (nodata); kayıpta yok sayılır.

## Ortam

- **Eğitim:** Colab Pro, A100. Colab'ın kurulu `torch` ve `timm` sürümleri kullanılır; venv açılmaz, torch yeniden
  kurulmaz. High-RAM kapalı. Veri her oturumda Hugging Face'ten VM diskine indirilir.
- **Yerel:** RTX 4060 8 GB yalnız hata ayıklama ve küçük denemeler için. Yerel ortam `uv`; `pyproject.toml` torch'u
  sabitlemez.
- **Notebook ince:** mantık `src/` paketinde. Notebook "Run All" ile baştan sona, kopmadan sonra aynı sonucu
  verecek biçimde çalışır.
- **Kalite:** `ruff`, `pytest`, `nbstripout`. CI veri gerektirmeyen sentetik testlerle CPU'da çalışır.

## Planlanan düzen

```
configs/     tek YAML yapılandırması (config hash'i run_id'nin parçası)
src/hasar/   veri, model, eğitim, metrik, değerlendirme
notebooks/   ince Colab defterleri
tests/       veri gerektirmeyen testler
results/     her sayının kaynağı JSON dosyaları
docs/        tasarım, ADR'ler, raporlar
```
