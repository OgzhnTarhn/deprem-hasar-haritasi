# Deprem Hasar Haritası

Afet öncesi optik ve afet sonrası SAR uydu görüntülerinden bina hasarını piksel düzeyinde haritalayan model.
[BRIGHT](https://github.com/ChenHongruixuan/BRIGHT) veri setinin 13 olayıyla eğitilir; 6 Şubat 2023 Türkiye
depreminde, eğitimde hiç görülmeden, **bir kez** test edilir. Referans: hazır DamageFormer 37,20 `[ZS-EVENT mIoU, repro]`;
ChangeMamba 41,28 `[ZS-EVENT mIoU]` yalnız bağlamdır.

## Kısaca sonuç

- **Fine-tune işe yaradı.** Hazır DamageFormer'ı kısa, kontrollü bir fine-tune'la güncelledik; model Türkiye'yi hiç
  görmeden referansı geçti ve eşleştirilmiş farkın %95 aralığı 0'ı içermiyor (aşağıdaki tabloda
  `kiyas/ana_vs_referans.json` satırı).
- **SAR dayanıklılığı fikri başlık metriğini değiştirmedi.** SAR artırımlı varyant ana modelle "eşit" çıktı
  (`kiyas/varyant_vs_ana.json`). Olumsuz sonucu da olduğu gibi yazıyoruz.
- **Açık problem: hasar sınıfları.** Kazancın neredeyse tamamı sağlam binalardan geldi; hasarlı ve yıkık binalar
  Türkiye'de hâlâ çok zor ([docs/durum.md](docs/durum.md)). Bunu hedefleyen few-shot ek deneyi hazır, sonucu
  aşağıdaki ayrı tabloya düşecek.

## Bu projede neyi iyi yaptık

- **Sızıntısız test.** Türkiye'ye erişim kodda kilitli: veri katmanı bir erişim belirteci olmadan Türkiye karesi
  okumaz, belirteç yalnız git etiketli (`protokol-kilidi`, `varyant-kilidi`) temiz bir commit'te verilir. Her model
  Türkiye'de bir kez değerlendirildi; Türkiye sonucuna göre ne ayar ne kayıt noktası seçildi.
- **Önceden kayıtlı protokol.** Metrik yolu, kıyaslar ve "eşit" kuralı eğitimden önce
  [docs/protokol.md](docs/protokol.md)'de yazıldı ve kilitlendi.
- **Önce yeniden üretim.** Hazır BRIGHT checkpoint'lerinin yayımlanan değerlerini kendi değerlendirme hattımızla
  tolerans içinde tutturduk; hat doğru olmadan model eğitilmedi. Bu sırada "zero-shot" diye yayımlanan DeepLabV3+
  ağırlığının Türkiye'yi görmüş olduğunu yakaladık ve dışarıda bıraktık.
- **Kontrollü deney.** Ana ve varyant aynı checkpoint'ten, aynı tohum ve örnek sırasıyla eğitildi; tek fark SAR
  artırımı. Fark bu yüzden fikrin kendisine aittir.
- **Dürüst istatistik.** Bütün farklar şehir bloklarına göre eşleştirilmiş bootstrap ile, %95 aralıklarıyla verilir;
  her sayının yanında metrik türü yazar ve README'deki her sayı `results/` dosyasından üretilir.
- **GPU yakmadan önce sigorta.** On bir ön kontrol kapısı (bellek, hız, maliyet projeksiyonu, devam testi) Colab
  eğitiminden önce çalışır; defter kopmalara dayanıklıdır, yarıda kalan iş kaldığı yerden sürer.
- **Mühendislik düzeni.** Mantık `src/hasar/` paketinde, defter ince; yapılandırma hash'i koşu kimliğinin parçası;
  veri gerektirmeyen kısa testler CI'da CPU'da koşar.

Ayrıntılar: tasarım [docs/tasarim.md](docs/tasarim.md) · görsel genel bakış [docs/proje-haritasi.md](docs/proje-haritasi.md) ·
bulgular [docs/bulgular.md](docs/bulgular.md) · ders kitabı [docs/dersler.md](docs/dersler.md) ·
güncel durum [docs/durum.md](docs/durum.md) · Colab'da çalıştırma [docs/colab.md](docs/colab.md)

## Sonuçlar

Her sayı `results/**/*.json` dosyasından `hasar readme` ile üretilir ve yanında metrik türünü taşır. Tabloyu okumak için:

- `ana/turkiye.json` ve `varyant/turkiye.json`: iki fine-tune'un Türkiye başlık sonucu `[ZS-EVENT mIoU]`, %95 aralıklı.
  Aynı klasörlerdeki `turkiye_<satır>.json` dosyaları önceden ilan edilmiş ikincil satırlardır.
- `kiyas/*.json`: eşleştirilmiş farklar. Birincil kıyas varyant − ana; ana − referans ve varyant − referans ikincildir.
  Fark protokol §9'daki "eşit" kuralıyla yorumlanır.
- `yeniden-uretim/*.json` (`repro`): hazır BRIGHT checkpoint'leri; bu projenin modeli değildir. DeepLabV3+ satırı
  **olumsuz bulgudur**: Türkiye'yi görmüş sayıldı ([docs/bulgular.md](docs/bulgular.md)).
- `[VAL13 mIoU]` satırları 13 olayın doğrulamasıdır; Türkiye başarısını ölçmez.

<!-- sonuclar:basla -->
Bu tablo `hasar readme` ile results/ dosyalarından üretildi; elle düzenlemeyin.

| Model | Bölme | Metrik | Değer | Kaynak |
| --- | --- | --- | --- | --- |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU] | 40,44 [39,54; 41,41] | `ana/turkiye.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU] | 40,35 [39,43; 41,35] | `varyant/turkiye.json` |
| bright_damageformer | zero_shot/val | [VAL13 mIoU, sicaklik] | 74,55 | `ana/sicaklik.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, bina-maskeli] | 40,48 [39,59; 41,43] | `ana/turkiye_bina-maskeli.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, bina-oylama] | 40,67 [39,71; 41,71] | `ana/turkiye_bina-oylama.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT bina-makro-F1] | 30,76 | `ana/turkiye_bina_f1.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, flip-tta] | 40,76 [39,81; 41,81] | `ana/turkiye_flip-tta.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, kalibrasyon] | 40,44 | `ana/turkiye_kalibrasyon.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, nodata-yok] | 40,16 [39,25; 41,14] | `ana/turkiye_nodata-yok.json` |
| BrightBaseline | zero_shot/val | [VAL13 mIoU, ema] | 74,55 | `ana/val13.json` |
| referans | zero_shot/test | [ZS-EVENT mIoU, eslestirilmis-fark] | 3,24 | `kiyas/ana_vs_referans.json` |
| ana | zero_shot/test | [ZS-EVENT mIoU, eslestirilmis-fark] | -0,09 | `kiyas/varyant_vs_ana.json` |
| referans | zero_shot/test | [ZS-EVENT mIoU, eslestirilmis-fark] | 3,15 | `kiyas/varyant_vs_referans.json` |
| bright_damageformer | zero_shot/val | [VAL13 mIoU, sicaklik] | 74,45 | `varyant/sicaklik.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, bina-maskeli] | 40,47 [39,57; 41,45] | `varyant/turkiye_bina-maskeli.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, bina-oylama] | 40,68 [39,68; 41,84] | `varyant/turkiye_bina-oylama.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT bina-makro-F1] | 30,88 | `varyant/turkiye_bina_f1.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, flip-tta] | 40,69 [39,73; 41,78] | `varyant/turkiye_flip-tta.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, kalibrasyon] | 40,35 | `varyant/turkiye_kalibrasyon.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, nodata-yok] | 40,07 [39,14; 41,08] | `varyant/turkiye_nodata-yok.json` |
| BrightBaseline | zero_shot/val | [VAL13 mIoU, ema] | 74,45 | `varyant/val13.json` |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, repro] | 37,20 | `yeniden-uretim/damageformer_test.json` |
| bright_damageformer | zero_shot/val | [VAL13 mIoU, repro] | 74,48 | `yeniden-uretim/damageformer_val.json` |
| bright_deeplabv3plus | zero_shot/test | [ZS-EVENT mIoU, repro] | 65,16 | `yeniden-uretim/deeplabv3plus_test.json` |
| bright_siamcrnn | zero_shot/test | [ZS-EVENT mIoU, repro] | 38,32 | `yeniden-uretim/siamcrnn_test.json` |
| bright_siamcrnn | zero_shot/val | [VAL13 mIoU, repro] | 73,32 | `yeniden-uretim/siamcrnn_val.json` |
| bright_unet | zero_shot/test | [ZS-EVENT mIoU, repro] | 37,82 | `yeniden-uretim/unet_test.json` |
| bright_unet | zero_shot/val | [VAL13 mIoU, repro] | 70,69 | `yeniden-uretim/unet_val.json` |
<!-- sonuclar:bitir -->

### Few-shot ek deneyi

Zero-shot sonuçları kesinleştikten sonra eklenen ayrı bir soru: Türkiye'nin küçük, kilitli bir bölgesinin etiketleri
ve hasar sınıflarına kayıp ağırlığı, hasarlı ve yıkık binalardaki açığı kapatır mı? Türkiye 2 km bloklarla bölünür;
her şehirden karelerin bir kısmı uyarlamaya, 0,5 km tampondan sonrası teste gider. İki kol ana modelden başlar,
her adımda yarı Türkiye yarı 13 olay görür; A düz, B hasar ağırlıklı kayıpla eğitilir. Sonuçlar `[FS-EVENT mIoU]`
etiketiyle bu ayrı tabloda raporlanır ve zero-shot sonuçlarıyla ya da 41,28 ile kıyaslanmaz
([protokol §15](docs/protokol.md), [ADR-0008](docs/adr/0008-turkiye-few-shot-ek-deneyi.md)).

<!-- fewshot:basla -->
<!-- fewshot:bitir -->

## Hızlı başlangıç

```bash
uv sync                      # paket ve geliştirme araçları (torch sabitlenmez)
uv run pytest -q             # veri gerektirmeyen testler
uv run hasar indir --root data/bright --cache data/cache   # BRIGHT'ı Hugging Face'ten indir
uv run hasar veri-ozet --root data/bright                  # bölme özeti -> results/veri/bolme.json
```

Eğitim Colab'da, ince defterlerle yapılır; adımlar [docs/colab.md](docs/colab.md)'de.

- **Zero-shot (bitti):** [notebooks/ana_egitim.ipynb](notebooks/ana_egitim.ipynb) önce ana, sonra varyant için ön kontrol →
  fine-tune → `sicaklik` → `final` çalıştırdı. Artık tarihî defterdir.
- **Few-shot:** [notebooks/fewshot.ipynb](notebooks/fewshot.ipynb) `fewshot-kilidi` etiketi olmadan eğitime geçmez;
  iki kolu eğitir, sonra test bölgesini model başına bir kez değerlendirir. Kıyaslar `uv run hasar fewshot-kiyas` ile.
- **GPU:** A100 hedeflenir; Colab L4 verirse defter onu da kabul eder ve kapı 8 maliyeti çalışan karta göre hesaplar.
- **Kilitler:** etiketleri kullanıcı yönetir; defter `origin/main` kullanır.

Sonuç tabloları `uv run hasar readme` ile yenilenir (`--check` güncel olup olmadığını denetler).

## Değişmez kurallar (özet)

1. Türkiye'nin ne görüntüsü ne etiketi eğitimde, doğrulamada, seçimde veya ayarda kullanılır; zero-shot için Türkiye'ye
   tek yol kilitli, temiz bir commit'teki `final` adımıdır. Tek istisna few-shot ek deneyi: yalnız kilitli uyarlama
   kareleri eğitime girer, test kareleri model başına bir kez değerlendirilir.
2. Türkiye'yi görmüş ağırlık (`*_standard_ML_split.pth`, MaRS) başlangıç olamaz.
3. İki zero-shot eğitimi (kontrol ve SAR varyantı) ve iki few-shot kolu; her biri bir kez.
4. Her sayının yanında metrik türü yazılır; sayılar dosyadan gelir; olumsuz sonuç da yazılır.

## Lisans ve atıf

Kod Apache-2.0 lisanslı üçüncü taraf kod içerir (`third_party/`). Veri ve ağırlıkların lisansları:
[docs/veri-lisans.md](docs/veri-lisans.md). Model çıktıları ve demo görselleri BRIGHT'tan (CC BY-SA 4.0) ve Maxar
Open Data'dan (CC BY-NC 4.0) türediği için bu şartlara tabidir.
