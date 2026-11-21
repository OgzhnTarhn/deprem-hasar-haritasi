# Colab'da çalıştırma rehberi

Ana model artık hazır DamageFormer zero-shot checkpoint'inin kontrol fine-tune'u. Varyant aynı checkpoint'ten
SAR radyometrik artırımıyla başlar. Karar: [ADR-0007](adr/0007-ana-model-hazir-damageformer-fine-tune.md),
[protokol 1.3](protokol.md). ChangeMamba 41,28 `[ZS-EVENT mIoU]` yalnız bağlamdır.

**Doldurulacak alan yok.** [Defter](../notebooks/ana_egitim.ipynb) tek “Run All” ile önce ana, sonra varyant için
**ön kontrol → fine-tune → `sicaklik` → `final`** sırasını çalıştırır. Aşamayı git etiketleri belirler.

| Etiketler | Her iki koşuda defterin yaptığı |
|---|---|
| `protokol-kilidi` yok | Yalnız ön kontrol; eğitim başlamaz |
| Yalnız `protokol-kilidi` | Ön kontrol → fine-tune; Türkiye değerlendirmesi yok |
| `protokol-kilidi` + `varyant-kilidi` | Ön kontrol → fine-tune → `ornekler` → `sicaklik` → tek `final` |

Defter `origin/main` kullanır. Bu dalın değişiklikleri orada bulunmadan yeni akış çalışmaz. Mevcut
`protokol-kilidi` 1.2'yi gösteriyor; kullanıcı, 1.3'ü içeren commit'e taşımalı ve aynı commit'e
`varyant-kilidi` koymalı. Bu belgeleri hazırlama adımı etiket taşımaz veya push yapmaz.
Varyant ana modelin Türkiye sonucundan önce kilitlenir.

`protokol-kilidi` denetimi `configs/ana.yaml`, `configs/varyant.yaml`, `docs/protokol.md` dosyalarını;
`varyant-kilidi` denetimi `configs/` altını kapsar. Kilitli dosya değişmişse veya etiket HEAD'in atası değilse durur.

| Adım | Yer | Süre / CU | Durum |
|---|---|---|---|
| Hazır modelleri yeniden üretme | Yerel; Faz 3 tamamlandı | Eğitim yok | Referans kayıtları depoda |
| Protokol 1.3 ve varyant kilitleri | Kullanıcı | GPU gerekmez | Eğitimden ve ana `final`den önce |
| Ana ön kontrol | Colab A100 veya L4 | Ölçülmedi, kapı 8 ölçer (eğitim projeksiyonu) | Ayrı ön kontrol kaydı |
| Ana fine-tune | Colab | Yaklaşık 1–2 saat; ≤ 15 CU hedefi; ölçülmedi, kapı 8 ölçer | 4.000 adım |
| Ana `sicaklik` → `final` | Colab | Ölçülmedi; kapı 8 eğitim projeksiyonunu ölçer, değerlendirme süresini ölçmez | Türkiye'de bir kez |
| Varyant ön kontrol → fine-tune | Aynı “Run All” | Yaklaşık 1–2 saat fine-tune; ≤ 15 CU hedefi; ölçülmedi, kapı 8 ölçer | 4.000 adım; ayrı kayıt |
| Varyant `sicaklik` → `final` | Aynı “Run All” | Ölçülmedi; kapı 8 yalnız eğitim projeksiyonunu ölçer | Türkiye'de bir kez |
| Kıyas, analizler, demo | Yerel | GPU eğitimi yok | Saklanan matris ve önbellekler |

Fine-tune süreleri [tasarımın](superpowers/specs/2026-11-21-sar-dayaniklilik-finetune-design.md) tahminidir.
Kapı 8'in ≤ 15 CU denetimine ön kontrol maliyeti dahil değildir. İki koşu ve ön kontrollerin toplam hedefi aylık
100 CU içinde kalmaktır; ölçülmüş toplam gibi yazılmaz.

## 0. Hazırlık

- A100 seçin; Colab L4 verirse defter onu da kabul eder. High-RAM kapalı kalsın.
- Kurulu `torch` ve `timm` kullanılır; torch yeniden kurulmaz.
- Veri VM diskine indirilir. Drive'da `MyDrive/deprem-hasar/` altında kayıtlar ve sonuçlar tutulur.
- `HF_TOKEN` Colab Secrets'e eklenebilir; zorunlu değildir.
- Etkin batch 8'dir. Bellek yetmezse mikro parti otomatik küçülür, gradyan biriktirilir; etkin batch değişmez.

### ChangeMamba çekirdeği: artık gerekmiyor

DamageFormer özel `selective_scan` çekirdeği istemez. Derleme, yama ve Drive tekerleği kurulumu bu akışta yoktur.
Kapı 11 model kaydını “uygulanmaz” notuyla geçirir.

## 1. Veri ve başlangıç ağırlığı

Defter `hasar indir` ile veriyi alır; `.done` varsa atlar. İki yapılandırma için aşağıdaki komut çağrılır:

```bash
hasar agirlik --config configs/ana.yaml --out /content/data/checkpoints
hasar agirlik --config configs/varyant.yaml --out /content/data/checkpoints
```

Komut md5 denetimli DamageFormer zero-shot checkpoint'inin yolunu son satıra yazar. Ön kontrol ve eğitim
bu yolu `--pretrained` ile alır. `standard_ML` veya ImageNet ağırlığı bu fine-tune'un başlangıcı değildir.
Faz 3'ün SiamCRNN, UNet ve DamageFormer yeniden üretim kayıtları varsa ortak yeniden üretim adımı atlanır.
Referans DamageFormer 37,20 `[ZS-EVENT mIoU, repro]`; kaynak `results/yeniden-uretim/damageformer_test.json`.

## 2. Ön kontrol

Her koşunun kaydı `kosular/onkontrol/<ad>/preflight.json` dosyasındadır. Yapılandırma hash'i tutan,
tamamlanmış ve geçmiş kayıt yeniden kullanılabilir. Bir kapı kalırsa eğitim başlamaz.

| Kapı | Fine-tune akışındaki kural |
|---|---|
| 1–3 | Yeniden üretim, Türkiye'siz bölme ve örnekleyici denetimleri korunur |
| 4 | Yüklenen model, 13 olay doğrulamasında 74,48 `[VAL13 mIoU, repro]` değerine ±0,3 yakın olmalı; fp32, TTA yok |
| 5–7 | Ezberleme, SAR dalı ve metrik uç durumları korunur |
| 8 | Çalışan kartta hız ve CU projeksiyonu; yapılandırmadaki 15 CU sınırını aşarsa durur |
| 9 | Ortak reçetenin 2× lr provasında p90 gradyan normu; eğitim bunu kırpma eşiği olarak okur |
| 10 | Tam durumla kopma ve devam denetimi korunur |
| 11 | DamageFormer için özel çekirdek gerekmez; “uygulanmaz” |

Kapı 4'ün kaynağı `results/yeniden-uretim/damageformer_val.json`; yoksa kapı geçmez.
`train.clip: null` kalır. Eğitim, `--preflight` kaydındaki 9. kapıyı okur; eksik veya farklı yapılandırmaya
ait kaydı reddeder. Eşik elle yazılmaz.

## 3. İki fine-tune ve kopmadan devam

Defter `RUNS` listesiyle önce `configs/ana.yaml`, sonra `configs/varyant.yaml` çalıştırır. `CONFIG` veya `NAME`
değiştirmek gerekmez. İkisi de aynı checkpoint, tohum, kırpma ve örnek sırasını kullanır. Varyantın tek eğitim
farkı SAR artırımıdır; değerlendirmede uygulanmaz.

Kopmada yeniden “Run All” yeterlidir. `last.pt` tam durumla devam eder; `DONE` varsa eğitim atlanır.
`kosular/<run_id>/best.pt` adı korunur ama içerik **son adımın EMA ağırlıklarıdır**; en yüksek doğrulama sonucu
aranmaz. EMA başlangıç checkpoint'inden başlar. `val_log.jsonl`, `val13.json` ve `DONE` ayrı koşu altında tutulur.

## 4. Sıcaklık ve Türkiye'de tek değerlendirme

İki kilit varsa her koşu kendi eğitiminden sonra `ornekler`, `sicaklik`, `final` sırasını izler.
`sicaklik` yalnız 13 olayın doğrulamasında ayarlanır; `sicaklik.json` varsa atlanır. `final` temiz çalışma ağacı
ve iki etiketin HEAD'in atası olmasını ister. Çıktılar depo dışına yazılır.

```bash
R=/content/drive/MyDrive/deprem-hasar
hasar ornekler --dest /content/data/ornekler --cache /content/data/cache
hasar sicaklik --config configs/ana.yaml --data /content/data/bright --runs $R/kosular --device cuda --out $R/sonuclar/ana
hasar final --config configs/ana.yaml --data /content/data/bright --runs $R/kosular --device cuda --out $R/sonuclar/ana --cache /content/data/final_cache --instances /content/data/ornekler
```

Varyant için defter aynı komutları `configs/varyant.yaml` ve `sonuclar/varyant` ile çağırır.
`ornekler` sabit bina karesi listesiyle uyumlu klasörü hazırlar. Model başına ayrı önbellek kullanılır.
Yarım `final`, aynı VM ve önbellek korunmuşsa oradan devam eder. VM yenilenirse yarım önbellek kaybolabilir;
defter Drive kopyasını `final` tamamlandıktan sonra alır. Başlık dosyası `turkiye.json` en son yazılır. Dosya varsa defter ikinci
`final`i atlar. `final` komutu da tamamlanmış değerlendirmeyi yeniden çalıştırmayı reddeder.

Drive'daki `sonuclar/ana/` ve `sonuclar/varyant/` içerikleri yerelde `results/ana/` ve `results/varyant/`
altına alınır. Her koşunun `val13.json` kaydı da ilgili sonuç klasörüne kopyalanır.
Önbellekler `MyDrive/deprem-hasar/final_cache/<run_id>/` altında saklanır; analizler ve demo bunları okur.

## 5. Üç kıyas

`hasar kiyas` yalnız kare matrislerini okur; modeli veya Türkiye görüntülerini yeniden çalıştırmaz.
Farkın yönü **b − a**. `--blocks results/arastirma/turkiye_bloklar.json` açıkça verilmelidir; verilmezse her kare ayrı blok sayılır.

```bash
# Birincil: varyant − ana
uv run hasar kiyas --a results/ana/turkiye_kare_matrisleri_headline.npz --b results/varyant/turkiye_kare_matrisleri_headline.npz --name-a ana --name-b varyant --out results/kiyas/varyant_vs_ana.json --blocks results/arastirma/turkiye_bloklar.json
# İkincil: ana − referans
uv run hasar kiyas --a results/yeniden-uretim/damageformer_test_matrices.npz --b results/ana/turkiye_kare_matrisleri_headline.npz --name-a referans --name-b ana --out results/kiyas/ana_vs_referans.json --blocks results/arastirma/turkiye_bloklar.json
# İkincil: varyant − referans
uv run hasar kiyas --a results/yeniden-uretim/damageformer_test_matrices.npz --b results/varyant/turkiye_kare_matrisleri_headline.npz --name-a referans --name-b varyant --out results/kiyas/varyant_vs_referans.json --blocks results/arastirma/turkiye_bloklar.json
```

Kare kümeleri eşleşmelidir; eşleşmezse komut hata verir. Eşleştirilmiş blok bootstrap ve “eşit” kuralı
[protokol §8–9](protokol.md) ile sabittir. Referans eşleştirilemiyorsa yalnız aralıkla kıyas raporlanır.
Sonuca göre model veya ayar seçilmez.

## 6. Faz 7: Analizler (yerel CPU)

Model yeniden çalışmaz; `results/ana/turkiye.json` yoksa reddeder. Önbellek ve örnek etiketleri yerele indirilir:

```python
from hasar.analysis.runner import run_cache_analyses
from hasar.guard import require_turkey_access
run_cache_analyses("data/final_cache/<run_id>", "data/bright", "data/ornekler", "results/analiz",
                   final_dir="results/ana", access=require_turkey_access("."))
```

Çıktılar: `results/analiz/y2_butce.json`, `y2_risk_kapsam.json`, `galeri/`. Y1 için HOT/OSM ve ARIA DPM
katmanları ayrıca indirilir ve karenin koordinat sistemine çevrilir (`docs/veri-lisans.md` §4;
`hasar.analysis.agreement`). Y3 (transdüktif) Türkiye'de yeniden çıkarım ister; ayrı tabloda raporlanır.
Bulgular `docs/bulgular.md`'ye işlenir.

## 7. Faz 9: Demo, README, dışa aktarım

- README tablosu: `uv run hasar readme` (denetim: `--check`). Elle sayı yazılmaz.
- Demo: `uv run hasar demo --cache data/final_cache/<run_id> --data data/bright --out demo/`. `demo/` git dışında
  (Maxar CC BY-NC görüntü içerir); yayımlama kararı sizde.
- Dışa aktarım: `uv run python scripts/disa_aktar.py --model <ad> --checkpoint best.pt` →
  `results/arastirma/disa_aktarim_<ad>.json` (ONNX için `onnx` paketi gerekir).
