# Deprem Hasar Haritası

Afet öncesi optik ve afet sonrası SAR uydu görüntülerinden bina hasarını piksel düzeyinde haritalayan model.
[BRIGHT](https://github.com/ChenHongruixuan/BRIGHT) veri setinin 13 olayıyla eğitilir; 6 Şubat 2023 Türkiye
depreminde, eğitimde hiç görülmeden, **bir kez** test edilir. Referans: hazır DamageFormer 37,20 `[ZS-EVENT mIoU, repro]`;
ChangeMamba 41,28 `[ZS-EVENT mIoU]` yalnız bağlamdır.

- Tasarım: [docs/tasarim.md](docs/tasarim.md) · Protokol (önceden kayıt): [docs/protokol.md](docs/protokol.md)
- Görsel genel bakış: [docs/proje-haritasi.md](docs/proje-haritasi.md) · Ders kitabı: [docs/dersler.md](docs/dersler.md)
- Güncel durum: [docs/durum.md](docs/durum.md) · Colab'da çalıştırma: [docs/colab.md](docs/colab.md)

## Sonuçlar

Her sayı `results/**/*.json` dosyasından üretilir ve yanında metrik türünü taşır. `[ZS-EVENT mIoU]`
için birincil kıyas varyant − ana, ikincil kıyaslar referansa karşıdır; fark protokol §9'daki "eşit" kuralıyla yazılır.

Şimdilik tabloda yalnız Faz 3'ün yeniden üretim satırları (`repro`) var: hazır BRIGHT checkpoint'lerinin yayımlanan
değerleri tutturup tutturmadığını gösterir, bu projenin modeli değildir. DeepLabV3+ satırı **olumsuz bulgudur**:
"zero-shot" diye yayımlanan checkpoint yayımlanan değerin çok üstünde çıktı ve Türkiye'yi görmüş sayıldı
([docs/bulgular.md](docs/bulgular.md)). Ana modelin satırları Colab'daki eğitim ve tek `final` adımından sonra
`hasar readme` ile eklenir.

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

<!-- fewshot:basla -->
<!-- fewshot:bitir -->

## Hızlı başlangıç

```bash
uv sync                      # paket ve geliştirme araçları (torch sabitlenmez)
uv run pytest -q             # veri gerektirmeyen testler
uv run hasar indir --root data/bright --cache data/cache   # BRIGHT'ı Hugging Face'ten indir
uv run hasar veri-ozet --root data/bright                  # bölme özeti -> results/veri/bolme.json
```

Kontrol ve SAR varyantı hazır DamageFormer zero-shot checkpoint’inden fine-tune edilir.
Tek “Run All” önce ana, sonra varyant için ön kontrol → fine-tune → `sicaklik` → `final` çalıştırır
(etiketlere göre). Eğitim Colab’da, ince defterle yapılır: [notebooks/ana_egitim.ipynb](notebooks/ana_egitim.ipynb). Adımlar
[docs/colab.md](docs/colab.md)'de.

- **GPU:** A100 hedeflenir; Colab L4 verirse defter onu da kabul eder ve kapı 8 maliyeti çalışan karta göre hesaplar.
- **Bellek:** bir adım belleğe sığmazsa eğitim döngüsü mikro partiyi yarıya indirip gradyan biriktirir; etkin batch
  değişmez (`train.micro_batch`).
- **Kilitler:** protokol 1.3'ü içeren commit'te `protokol-kilidi` ve `varyant-kilidi` gerekir; kullanıcı yönetir.
  Defter `origin/main` kullanır; dal değişiklikleri oraya alınmış olmalı.
- **Bütçe:** fine-tune başına ≤ 15 CU hedefi; ölçülmedi, kapı 8 ölçer. Özel ChangeMamba çekirdeği artık gerekmiyor.

Sonuç tablosu `uv run hasar readme` ile yenilenir (`--check` güncel olup olmadığını denetler).

## Değişmez kurallar (özet)

1. Türkiye'nin ne görüntüsü ne etiketi eğitimde, doğrulamada, seçimde veya ayarda kullanılır; Türkiye'ye tek yol
   `protokol-kilidi` etiketli, temiz bir commit'teki `final` adımıdır.
2. Türkiye'yi görmüş ağırlık (`*_standard_ML_split.pth`, MaRS) başlangıç olamaz.
3. İki eğitim: aynı hazır DamageFormer zero-shot checkpoint’inden kontrol ve SAR varyantı; her biri bir kez.
4. Her sayının yanında metrik türü yazılır; sayılar dosyadan gelir; olumsuz sonuç da yazılır.

## Lisans ve atıf

Kod Apache-2.0 lisanslı üçüncü taraf kod içerir (`third_party/`). Veri ve ağırlıkların lisansları:
[docs/veri-lisans.md](docs/veri-lisans.md). Model çıktıları ve demo görselleri BRIGHT'tan (CC BY-SA 4.0) ve Maxar
Open Data'dan (CC BY-NC 4.0) türediği için bu şartlara tabidir.
