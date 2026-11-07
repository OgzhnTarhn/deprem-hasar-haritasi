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

```bash
hasar sicaklik --config configs/ana.yaml --data /content/data/bright --runs /content/drive/MyDrive/deprem-hasar/kosular
hasar final    --config configs/ana.yaml --data /content/data/bright --runs /content/drive/MyDrive/deprem-hasar/kosular \
               --instances <CVPRW26 örnek etiketleri klasörü>
```

Defterde `RUN_FINAL = True` aynı şeyi yapar. Sonuçlar `results/ana/turkiye*.json`; olasılık önbelleği
`data/final_cache/` (Drive'a kopyalayın: analizler ve demo onu okur). `final` ikinci kez çalışmaz.

## 7–9. Analizler, varyant, demo

- Analizler önbellekten, yerel CPU'da: `hasar.analysis.runner.run_cache_analyses(...)`. Y1 için HOT/OSM ve ARIA DPM
  katmanları ayrıca indirilir (`docs/veri-lisans.md` §4).
- Varyant: aynı defter, `configs/varyant.yaml` ile; iki modelin farkı `hasar.eval.compare` ile "eşit" kuralına göre.
- Demo: `hasar demo --cache data/final_cache --data data/bright --out demo/`; README tablosu: `hasar readme`.
