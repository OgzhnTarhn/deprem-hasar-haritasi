# Colab'da çalıştırma rehberi

Bu belge, kodu yazılmış ama GPU gerektirdiği için çalıştırılmamış adımları sırayla anlatır. Bütün mantık
`src/hasar` paketinde; defter yalnız komutları çağırır. Her adım yapılmış işi atlar: kopma olursa aynı defter
"Run All" ile kaldığı yerden devam eder.

| Adım | Faz | Nerede | Süre / CU (tahmin) | Geri alınabilir mi |
|---|---|---|---|---|
| 1. Yeniden üretim | 3 | **Yapıldı** (yerel RTX 4060, 7 Kasım 2026) | ~16 dk / 0 CU | Evet |
| 2. Ön kontrol kapıları | 4 | Colab A100 | 1–1,5 saat / ~10 CU | Evet |
| 3. `protokol-kilidi` etiketi | 4 | Yerel | 1 dk | Etiket silinebilir ama `final` sonrası anlamı kalmaz |
| 4. Ana eğitim | 5 | Colab A100 | 40.000 adım / 50–75 CU | Evet (devam eder) |
| 5. KN1 ve `varyant-kilidi` | 6 | Yerel (karar) | – | Evet |
| 6. Sıcaklık ve Türkiye `final` | 6 | Colab A100 | < 1 saat / < 5 CU | **Hayır: Türkiye bir kez değerlendirilir** |
| 7. Analizler | 7 | Yerel CPU | – | Evet |
| 8. Varyant eğitimi ve `final` | 8 | Colab A100 | 50–75 CU | `final` yine bir kez |
| 9. Demo, README | 9 | Yerel | – | Evet |

## 0. Hazırlık (bir kez)

- Colab'da **A100** seçin, High-RAM kapalı kalsın. Colab'ın kurulu `torch` ve `timm` sürümleri kullanılır.
  Colab A100 seçili görünse de bazen **L4** (22 GB) veriyor (`!nvidia-smi` gerçek kartı gösterir). Defter L4'ü de
  kabul eder: kapı 8 CU'yu karta göre (A100 ~13, L4 ~5 CU/saat) hesaplar. Bellek yetmezse eğitim döngüsü mikro
  partiyi kendisi yarıya indirir ve gradyanları biriktirir; etkin batch 16 kalır (`train.micro_batch`).
- ChangeMamba çekirdeği Colab'ın CUDA 13'ünde ancak yamayla derlenir (`cub::LaneId`, `cub::CTA_SYNC` kaldırıldı,
  sm_70 desteklenmiyor). Defterin 2. hücresi yamayı kendisi uygular ve derlenen tekerleği
  `MyDrive/deprem-hasar/wheels/` altına kaydeder; sonraki oturumlar derlemeden kurar.
- Colab "Secrets" bölümüne `HF_TOKEN` ekleyin (Hugging Face indirme hızı için; zorunlu değil).
- Drive'da `MyDrive/deprem-hasar/` klasörü defter tarafından açılır; kayıt noktaları oraya yazılır.

## 1. Faz 3: hazır modellerin yeniden üretimi

> **Durum:** yerelde yapıldı ve commit'lendi (`results/yeniden-uretim/`): SiamCRNN, UNet, DamageFormer ±0,3 içinde;
> DeepLabV3+ checkpoint'i tutmadı (Türkiye'yi görmüş görünüyor, `docs/durum.md` Faz 3). Defterin 4. hücresi bu
> dosyaları bulduğu için adımı atlar. Aşağıdaki komutlar yalnız yeniden çalıştırmak gerekirse.

Zenodo 15349462'deki Türkiye zero-shot checkpoint'leri, eğitim yapılmadan Türkiye'de çalıştırılır (kural 1'in
birinci istisnası). Çıkış ölçütü: SiamCRNN 38,32, UNet 37,82, DamageFormer 37,20 `[ZS-EVENT mIoU]` değerlerine
±0,3 içinde yakınlık. Tutmazsa hata veri ya da metrik hattındadır ve sonraki faza geçilmez.

```bash
# Çalışma ağacı temiz olmalı (commit'lenmemiş değişiklik varsa guard reddeder).
uv run hasar yeniden-uret --model siamcrnn     --data data/bright --out results/yeniden-uretim --device cuda
uv run hasar yeniden-uret --model unet         --data data/bright --out results/yeniden-uretim --device cuda
uv run hasar yeniden-uret --model damageformer --data data/bright --out results/yeniden-uretim --device cuda
# B2 topluluk adayı için aynı checkpoint'lerin 13 olay doğrulaması (Türkiye'ye dokunmaz):
uv run hasar yeniden-uret --model damageformer --split val --data data/bright --out results/yeniden-uretim --device cuda
```

Sonuçlar `results/yeniden-uretim/<model>_test.json` dosyalarına `[ZS-EVENT mIoU, repro]` etiketiyle yazılır;
`extra.within_tolerance` alanı çıkış ölçütünü gösterir. Defterin 4. hücresi bu adımı da kendisi çalıştırır.

## 2. Faz 4: ön kontrol kapıları

Defter (`notebooks/ana_egitim.ipynb`) ilk satırında `REF` olarak henüz etiket yoksa `main` ya da bir commit
SHA'sı verin; `STEPS_OVERRIDE = 2000` ile kısa prova yapılabilir. 4. hücre 11 kapıyı çalıştırır ve
`MyDrive/deprem-hasar/kosular/preflight.json` dosyasını yazar:

| # | Kapı | Geçme koşulu |
|---|---|---|
| 1 | Yeniden üretim | En az 3 `repro` sonucu ±0,3 içinde |
| 2 | Bölme sayıları | 2.795 / 337 / 1.114; eğitim ve doğrulamada Türkiye yok |
| 3 | Örnekleyici uzunluğu | adım × batch |
| 4 | Başlangıç kaybı | hasar ≈ ln 4, bina ≈ ln 2 (±0,25) |
| 5 | Ezberleme | 8 kare, artırımsız; kayıp başlangıcın %20'sinin altına |
| 6 | SAR dalı bağlı | SAR sıfırlanınca hasar çıktısı değişir |
| 7 | Metrik uç durumları | kusursuz = 1,0; hep arka plan = arka plan oranı / 4 (≈ 0,226) |
| 8 | Hız ve CU | projeksiyon ≤ 75 CU |
| 9 | Gradyan kırpma | 2× lr'de p90 gradyan normu → `train.clip` |
| 10 | Devam testi | 60. adımda kesilip devam eden koşu, kesintisiz koşuyla aynı ağırlıklar (GPU'da `atol` 1e-3, kayıplar %1 içinde) |
| 11 | Çekirdek | 20 adım ileri-geri sonlu; ChangeMamba CUDA çekirdeğiyle çalışıyorsa torch yoluyla fark < 1e-2. Kayıtta model ve tarama yolu yazar |

Çekirdek derlenmemişse model kurulurken otomatik olarak Swin-T DamageFormer yedeğine geçilir (tasarım §5) ve kapılar
yedekle koşar; `preflight.json`'daki kapı 11 kaydında `model: BrightBaseline` görünür. Ana eğitimin ChangeMamba ile
yapılacağından emin olmak için bu alana bakın.
Kapı 9'un verdiği eşik `configs/ana.yaml` içinde `train.clip` alanına yazılıp commit'lenir.

## 3. Protokol kilidi

Bütün kapılar geçtikten ve `train.clip` yazıldıktan sonra, yerelde:

```bash
git status            # temiz olmalı
git tag protokol-kilidi
git push origin main --tags
```

Bu etiketten sonra `configs/ana.yaml` ve `docs/protokol.md` değiştirilmez.

## 4. Faz 5: ana eğitim

Defterde `REF = "protokol-kilidi"`, `STEPS_OVERRIDE = None`, `RUN_FINAL = False` ile "Run All". Kopma olursa yine
"Run All": `last.pt` varsa tam durumla devam eder, `DONE` varsa atlar. Sonunda Drive'da
`kosular/<run_id>/` altında `best.pt` (EMA), `val_log.jsonl`, `val13.json` ve `DONE` bulunur.
`val13.json` dosyasını depoya `results/ana/val13.json` olarak kopyalayın.

## 5. Faz 6: KN1 ve varyant kilidi (karar)

Ana modelin 13 olaydaki sonuçlarına bakarak `docs/fikir-havuzu.md`'deki adaylardan varyant fikri seçilir; tasarımı
ve yapılandırması (`configs/varyant.yaml`) commit'lenip `varyant-kilidi` etiketi atılır. **Türkiye sonucu
görülmeden.** `final` komutu bu etiket olmadan çalışmaz.

## 6. Faz 6: Türkiye'de tek değerlendirme

Defterde `REF = "varyant-kilidi"`, `RUN_FINAL = True` ile "Run All". 6. hücre sırayla şunları yapar (elle de
çalıştırılabilir; çıktılar depo dışına yazılır, çünkü `final` temiz çalışma ağacı ister):

```bash
R=/content/drive/MyDrive/deprem-hasar
hasar ornekler --dest /content/data/ornekler --cache /content/data/cache        # yalnız sabit 1.001 kare
hasar sicaklik --config configs/ana.yaml --data /content/data/bright --runs $R/kosular --device cuda --out $R/sonuclar/ana
hasar final    --config configs/ana.yaml --data /content/data/bright --runs $R/kosular --device cuda --out $R/sonuclar/ana                --cache /content/data/final_cache --instances /content/data/ornekler
```

Sonra yerelde `MyDrive/deprem-hasar/sonuclar/ana/` içeriği `results/ana/` altına kopyalanıp commit'lenir;
önbellek `MyDrive/deprem-hasar/final_cache/<run_id>/` altında kalır (analizler ve demo onu okur).

- `final` yalnız `protokol-kilidi` ve `varyant-kilidi` etiketleri HEAD'in atasıysa ve ağaç temizse çalışır.
- `sicaklik` yalnız 13 olayın doğrulamasını okur (337 kare; yerel denemede kare başına ~6 sn, A100'de daha kısa;
  NLL ızgarası CPU'da). Sonucu `--out` klasöründeki `sicaklik.json`; `final` aynı klasörden okur, yoksa uyarıp
  T=1 ile devam eder. Önce `sicaklik`.
- `--instances` klasörü `results/arastirma/turkiye_bina_kareleri.json` listesindeki 1.001 kareyle birebir
  tutmalı; tutmazsa `final` Türkiye'ye dokunmadan durur. `hasar ornekler` tam bu klasörü üretir.
- Bootstrap blokları varsayılan olarak `results/arastirma/turkiye_bloklar.json`'dan okunur (protokol §8).
- Önbellek model başına ayrıdır (`<cache>/<run_id>/`): varyantın `final`'i ana modelin olasılıklarını okuyamaz.
- Yarıda kalırsa aynı komutu yeniden çalıştırın: biten kareler önbellekten okunur. Başlık dosyası
  (`turkiye.json`) en son yazılır; o yazıldıktan sonra komut reddedilir.
- Sonuç kayıtlarındaki `model` alanı gerçekten çalışan modeli yazar (çekirdek yoksa `damageformer_swin_t`).
  Ana eğitim ChangeMamba ile yapıldıysa `final` da çekirdeği derlenmiş bir oturumda çalışmalıdır; aksi hâlde
  ağırlıklar yedek modele yüklenemez ve komut hata verir.

## 7. Faz 7: Analizler (yerel CPU)

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

## 8. Faz 8: Varyant

Aynı defter `CONFIG = "configs/varyant.yaml"`, `NAME = "varyant"` ile: ön kontrol, eğitim, `RUN_FINAL = True`
ile tek `final`. İki modelin farkı kare matrislerinden, eşleştirilmiş blok bootstrap ile:

```python
from hasar.eval.bootstrap import turkey_blocks
from hasar.eval.compare import compare_models, compare_record
from hasar.eval.metrics import ConfusionAccumulator
a = ConfusionAccumulator.load_npz("results/ana/turkiye_kare_matrisleri_headline.npz").per_tile
b = ConfusionAccumulator.load_npz("results/varyant/turkiye_kare_matrisleri_headline.npz").per_tile
result = compare_models(a, b, blocks=turkey_blocks("results/arastirma/turkiye_bloklar.json"))
```

Sonuç "eşit" kuralına göre yazılır (protokol §9); seçim Türkiye sonucuna göre yapılmaz.

## 9. Faz 9: Demo, README, dışa aktarım

- README tablosu: `uv run hasar readme` (denetim: `--check`). Elle sayı yazılmaz.
- Demo: `uv run hasar demo --cache data/final_cache/<run_id> --data data/bright --out demo/`. `demo/` git dışında
  (Maxar CC BY-NC görüntü içerir); yayımlama kararı sizde.
- Dışa aktarım: `uv run python scripts/disa_aktar.py --model <ad> --checkpoint best.pt` →
  `results/arastirma/disa_aktarim_<ad>.json` (ONNX için `onnx` paketi gerekir).
