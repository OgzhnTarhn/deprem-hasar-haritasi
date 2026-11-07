# Deprem Hasar Haritası

Afet öncesi optik ve afet sonrası SAR uydu görüntülerinden bina hasarını piksel düzeyinde haritalayan model.
[BRIGHT](https://github.com/ChenHongruixuan/BRIGHT) veri setinin 13 olayıyla eğitilir; 6 Şubat 2023 Türkiye
depreminde, eğitimde hiç görülmeden, **bir kez** test edilir. Kıyas çıpası: ChangeMamba 41,28 `[ZS-EVENT mIoU]`.

- Tasarım: [docs/tasarim.md](docs/tasarim.md) · Protokol (önceden kayıt): [docs/protokol.md](docs/protokol.md)
- Görsel genel bakış: [docs/proje-haritasi.md](docs/proje-haritasi.md) · Ders kitabı: [docs/dersler.md](docs/dersler.md)
- Güncel durum: [docs/durum.md](docs/durum.md) · Colab'da çalıştırma: [docs/colab.md](docs/colab.md)

## Sonuçlar

Her sayı `results/**/*.json` dosyasından üretilir ve yanında metrik türünü taşır. Yalnız `[ZS-EVENT mIoU]`
41,28 ile kıyaslanır; fark protokol §9'daki "eşit" kuralına göre yazılır.

Şimdilik tabloda yalnız Faz 3'ün yeniden üretim satırları (`repro`) var: hazır BRIGHT checkpoint'lerinin yayımlanan
değerleri tutturup tutturmadığını gösterir, bu projenin modeli değildir. DeepLabV3+ satırı **olumsuz bulgudur**:
"zero-shot" diye yayımlanan checkpoint yayımlanan değerin çok üstünde çıktı ve Türkiye'yi görmüş sayıldı
([docs/bulgular.md](docs/bulgular.md)). Ana modelin satırları Colab'daki eğitim ve tek `final` adımından sonra
`hasar readme` ile eklenir.

<!-- sonuclar:basla -->
Bu tablo `hasar readme` ile results/ dosyalarından üretildi; elle düzenlemeyin.

| Model | Bölme | Metrik | Değer | Kaynak |
| --- | --- | --- | --- | --- |
| bright_damageformer | zero_shot/test | [ZS-EVENT mIoU, repro] | 37,20 | `yeniden-uretim/damageformer_test.json` |
| bright_damageformer | zero_shot/val | [VAL13 mIoU, repro] | 74,48 | `yeniden-uretim/damageformer_val.json` |
| bright_deeplabv3plus | zero_shot/test | [ZS-EVENT mIoU, repro] | 65,16 | `yeniden-uretim/deeplabv3plus_test.json` |
| bright_siamcrnn | zero_shot/test | [ZS-EVENT mIoU, repro] | 38,32 | `yeniden-uretim/siamcrnn_test.json` |
| bright_siamcrnn | zero_shot/val | [VAL13 mIoU, repro] | 73,32 | `yeniden-uretim/siamcrnn_val.json` |
| bright_unet | zero_shot/test | [ZS-EVENT mIoU, repro] | 37,82 | `yeniden-uretim/unet_test.json` |
| bright_unet | zero_shot/val | [VAL13 mIoU, repro] | 70,69 | `yeniden-uretim/unet_val.json` |
<!-- sonuclar:bitir -->

## Hızlı başlangıç

```bash
uv sync                      # paket ve geliştirme araçları (torch sabitlenmez)
uv run pytest -q             # veri gerektirmeyen testler
uv run hasar indir --root data/bright --cache data/cache   # BRIGHT'ı Hugging Face'ten indir
uv run hasar veri-ozet --root data/bright                  # bölme özeti -> results/veri/bolme.json
```

Eğitim Colab A100'de, ince defterle yapılır: [notebooks/ana_egitim.ipynb](notebooks/ana_egitim.ipynb).

## Değişmez kurallar (özet)

1. Türkiye'nin ne görüntüsü ne etiketi eğitimde, doğrulamada, seçimde veya ayarda kullanılır; Türkiye'ye tek yol
   `protokol-kilidi` etiketli, temiz bir commit'teki `final` adımıdır.
2. Türkiye'yi görmüş ağırlık (`*_standard_ML_split.pth`, MaRS) başlangıç olamaz.
3. İki tam eğitim: ana model (ChangeMamba olduğu gibi) ve tek fikirli bir varyant.
4. Her sayının yanında metrik türü yazılır; sayılar dosyadan gelir; olumsuz sonuç da yazılır.

## Lisans ve atıf

Kod Apache-2.0 lisanslı üçüncü taraf kod içerir (`third_party/`). Veri ve ağırlıkların lisansları:
[docs/veri-lisans.md](docs/veri-lisans.md). Model çıktıları ve demo görselleri BRIGHT'tan (CC BY-SA 4.0) ve Maxar
Open Data'dan (CC BY-NC 4.0) türediği için bu şartlara tabidir.
