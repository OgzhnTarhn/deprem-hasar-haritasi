# SAR Dayanıklılığı Fine-Tune'u Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ana modeli ChangeMamba'dan hazır DamageFormer zero-shot checkpoint'inin kısa fine-tune'una çevirmek;
SAR radyometrik artırımlı varyantı eklemek; ikisini tek "Run All" ile Colab'da eğitip Türkiye'de birer kez
değerlendirmek.

**Architecture:** Yapılandırmaya üç alan grubu eklenir: SAR artırımı, fine-tune başlangıcı, bütçe ve kayıt
noktası seçimi. Mevcut eğitim motoru, ön kontrol ve `final` akışı aynen kullanılır; yalnız model kurulumu, veri
artırımı, kapı 4 ve kapı 8 genişler. Kıyas için küçük bir `hasar kiyas` komutu eklenir.

**Tech Stack:** Python ≥3.10, PyTorch (Colab'ın sürümü), numpy, pytest, ruff, nbstripout, `uv`.

**Spec:** [docs/superpowers/specs/2026-11-21-sar-dayaniklilik-finetune-design.md](../specs/2026-11-21-sar-dayaniklilik-finetune-design.md)

## Global Constraints

- Türkiye kilidi: testler ve eğitim Türkiye karesi okumaz; artırım istatistikleri yalnız 13 olaydan (`sar_olcek.json`).
- Türkiye'yi görmüş ağırlık yok: başlangıç yalnız `ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth`; md5 kodda kayıtlı `BASELINES["damageformer"].md5` ile denetlenir (elle yazılmaz).
- Reçete (iki yapılandırmada ortak): `bright_damageformer`, crop 512, batch 8, 4.000 adım, AdamW lr 2e-5, wd 5e-3, warmup 200, kosinüs, min lr 1e-7, bf16, EMA 0,999, clip null, seed 42, eval her 500 adım, kayıt noktası son adımın EMA'sı, fine-tune bütçesi 15 CU.
- Varyant farkı yalnız: `aug.sar_p: 0.8`, `aug.sar_gamma: [0.67, 1.5]`, `aug.sar_gain: [0.657, 1.657]` ve `experiment: varyant`.
- Kontrol (ana): `aug.sar_p: 0.0`, `sar_gamma: [1.0, 1.0]`, `sar_gain: [1.0, 1.0]`, `experiment: ana`.
- Kod ve adlar İngilizce, kullanıcıya dönük iletiler Türkçe. ruff satır sınırı 120.
- Commit: atomik, Türkçe, emir kipi; `Co-Authored-By` ve "Generated with Claude" satırı yok (`CLAUDE.md`). Commit'ler yapıldığı günün tarihiyle atılır. Push ve etiketler kullanıcıda.
- Başlangıç: `uv run pytest` 737 geçiyor, `uv run ruff check .` temiz. Her görev bu durumu korur.

## Review Focus

1. Artırım sonrası nodata pikseli 0 kalmalı, geçerli piksel 0'a düşmemeli (yoksa model onu nodata sanır). Task 2'de test.
2. Kopup devam eden varyant aynı artırımı yeniden üretmeli: aynı `(index, sample_no)` iki çağrıda bit bit aynı çıktı. Task 2'de test.
3. Yanlış checkpoint (`standard_ML` ya da `vssm_tiny`) verilirse eğitim başlamadan md5 hatasıyla durmalı. Task 3'te test.
4. `sicaklik` ve `final` `--pretrained` olmadan çalışmalı: ham `bright_damageformer` kurulup `best.pt` yüklenir. Task 3'te test.
5. Kapı 4'ün yeniden üretim kaydı (`damageformer_val.json`) yoksa kapı çökmemeli, açık Türkçe gerekçeyle "kaldı" demeli. Task 5'te test.

---

### Task 1: Yapılandırma şeması ve iki yapılandırma dosyası

**Files:**
- Modify: `src/hasar/config.py` (`Aug`, `Model`, `Train`, `Eval`, `Data`, `Optim`, `Sched`, `Ckpt` varsayılanları ve doğrulamalar)
- Modify: `configs/ana.yaml` (kontrol reçetesi, her alan açık ve yorumlu)
- Create: `configs/varyant.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Aug.sar_p: float = 0.0`, `Aug.sar_gamma: tuple[float, ...] = (1.0, 1.0)`, `Aug.sar_gain: tuple[float, ...] = (1.0, 1.0)`; `Model.name = "bright_damageformer"`, `Model.init = "bright_zero_shot"` (izinli: `"vssm_tiny_imagenet"`, `"bright_zero_shot"`); `Train.budget_cu: float = 15.0`; `Eval.select = "last_ema"` (izinli: `"ema_val_miou"`, `"last_ema"`). `Config()` varsayılanları `configs/ana.yaml` ile birebir aynı.

- [ ] **Step 1: Failing testleri yaz** (`tests/test_config.py`)
  - `test_ana_yaml_equals_defaults`: mevcut testi koru; `load_config("configs/ana.yaml") == Config()`.
  - `test_varyant_differs_only_in_sar_aug_and_name`: `a, v = load_config(ana), load_config(varyant)`; `dataclasses.replace(v, experiment="ana", aug=a.aug) == a`; `v.aug.sar_p == 0.8`, `v.aug.sar_gamma == (0.67, 1.5)`.
  - `test_sar_gain_bounds_match_sar_olcek`: `sar_olcek.json` içinden `lo = min(e["median_mean"]) / pooled["mean"]`, `hi = max(...) / pooled["mean"]`; `v.aug.sar_gain == (round(lo, 3), round(hi, 3)) == (0.657, 1.657)`.
  - `test_finetune_recipe_values`: ana için crop 512, batch 8, steps 4000, lr 2e-5, warmup 200, min_lr 1e-7, eval.every 500, select "last_ema", budget_cu 15.0, init "bright_zero_shot".
  - Hata testleri (`pytest.raises(ConfigError)`): `sar_p` 1,2 ya da −0,1; `sar_gamma` uzunluğu 3; `sar_gain` alt > üst ya da ≤ 0; `eval.select: "best"`; `model.init: "x"`; `budget_cu: 0`.
  - 290–303. satırlardaki açık payload testi yeni varsayılanlarla güncellenir; `test_config_hash_changes_for_each_section`'a `{"aug": {"sar_p": 0.5}}` eklenir.

- [ ] **Step 2: Testleri koş, düştüklerini gör**
  Run: `uv run pytest tests/test_config.py -q`
  Expected: yeni testler FAIL (alan yok / değerler eski)

- [ ] **Step 3: `config.py` alanlarını, doğrulamalarını ve varyant varsayılanlarını yaz; iki YAML'ı yaz**
  `configs/ana.yaml` her alanı yorumuyla yazar (mevcut dosyanın üslubu). `aug.sar_gain` yorumu türetimi gösterir: "41,3/62,9 ve 104,2/62,9 (`sar_olcek.json`)". `configs/varyant.yaml` aynı dosyanın kopyası; yalnız `experiment` ve üç `aug.sar_*` alanı farklı.

- [ ] **Step 4: Testleri koş**
  Run: `uv run pytest tests/test_config.py -q` → PASS. Ardından `uv run pytest -q`: başka testler eski varsayılanlara (crop 640, steps 40000, `changemamba_mmbda_tiny`) dayanıyorsa test içinde açık değerle düzelt; davranış değiştirme.

- [ ] **Step 5: Commit**
  `git commit -m "Yapılandırmaya fine-tune reçetesini ve SAR artırımı alanlarını ekle"`

---

### Task 2: SAR radyometrik artırımı

**Files:**
- Modify: `src/hasar/data/transforms.py`
- Modify: `src/hasar/data/dataset.py` (`BrightDataset.__init__`, `_train_item`)
- Modify: `src/hasar/train/run.py` (`build_datasets` aug alanlarını geçirir)
- Test: `tests/test_transforms.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: Task 1'deki `cfg.aug.sar_p`, `sar_gamma`, `sar_gain`.
- Produces: `sar_radiometric(post: np.ndarray, valid: np.ndarray, rng: np.random.Generator, p: float, gamma: tuple[float, float], gain: tuple[float, float]) -> np.ndarray`. `BrightDataset(..., sar_p: float = 0.0, sar_gamma=(1.0, 1.0), sar_gain=(1.0, 1.0))`.

Algoritma (sıra sabit, belirlenim buna dayanır):

```
if p == 0: return post            # hiç çekiliş yok -> kontrol geometrisi aynı kalır
if rng.random() >= p: return post
g = exp(rng.uniform(log gamma[0], log gamma[1]))
k = exp(rng.uniform(log gain[0],  log gain[1]))
x = 255 * (post/255) ** g * k  (float32)
x[valid]  = clip(x[valid], 1, 255);  x[~valid] = 0
```

Çağrı `transforms.augment`'tan **sonra**, aynı `rng` ile, `normalize.sar_to_input`'tan önce yapılır. `sar_to_input` float32 girdiyi zaten kabul ediyor.

- [ ] **Step 1: Failing testleri yaz**
  - `test_sar_radiometric_p0_returns_input_without_draws`: p=0; çıktı girdiye eşit ve `rng` durumu değişmemiş (sonraki `rng.random()` kopya üreteçle aynı).
  - `test_sar_radiometric_keeps_nodata_zero_and_valid_positive`: rastgele uint8 SAR, yarısı `valid=False`; p=1; `out[~valid] == 0` ve `out[valid].min() >= 1`, `out.max() <= 255`.
  - `test_sar_radiometric_identity_ranges`: gamma (1,1), gain (1,1), p=1 → geçerli piksellerde çıktı `clip(post, 1, 255)`.
  - `test_sar_radiometric_gain_scales_mean`: gamma (1,1), gain (2,2), p=1, değerler 10–100 → çıktı = 2× girdi.
  - `test_sar_radiometric_deterministic`: aynı tohumlu iki üreteç aynı çıktıyı verir.
  - `test_dataset_control_and_variant_share_geometry` (`test_dataset.py`, sentetik kareler, mevcut fixture deseni): `sar_p=0` ve `sar_p=1` iki dataset, aynı `(i, s)` için `pre`, `clf`, `valid` eşit; `post` farklı.
  - `test_dataset_variant_item_repeatable`: aynı `(i, s)` iki kez → `post` bit bit aynı.

- [ ] **Step 2: Koş, düştüğünü gör**
  Run: `uv run pytest tests/test_transforms.py tests/test_dataset.py -q` → FAIL (`sar_radiometric` yok)

- [ ] **Step 3: `sar_radiometric`'i yaz, `_train_item` ve `build_datasets`'e bağla**
  Değerlendirme yolu (`_eval_item`) ve kapı 5'in `_fixed_batch`'i dokunulmadan kalır (artırım yok).

- [ ] **Step 4: Koş**
  Run: `uv run pytest -q` → hepsi PASS

- [ ] **Step 5: Commit**
  `git commit -m "Eğitime SAR radyometrik artırımını ekle"`

---

### Task 3: Zenodo checkpoint'inden başlatma

**Files:**
- Modify: `src/hasar/train/run.py` (`build_model`, `run_training`, `run_preflight`)
- Modify: `src/hasar/cli.py` (`agirlik` alt komutuna `--config`)
- Test: `tests/test_train_run_preflight.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `bright_baselines.BASELINES["damageformer"]`, `file_md5`, `load_checkpoint`, `download_checkpoint`.
- Produces: `build_model(cfg, pretrained, device)` imzası aynı. `cfg.model.name == "bright_damageformer"` iken: `pretrained` verilmişse md5 `BASELINES["damageformer"].md5` ile denetlenir, tutmazsa `ValueError` (Türkçe, dosya adı ve beklenen checkpoint adıyla); tutarsa `load_checkpoint` uygulanır. `pretrained is None` ise ham model döner (`final`/`sicaklik` yolu). `run_training` ve `run_preflight`, `cfg.model.init == "bright_zero_shot"` ve `pretrained is None` ise başlamadan `ValueError` verir. `hasar agirlik --config X --out D`: init `bright_zero_shot` ise DamageFormer zero-shot checkpoint'ini indirip (md5 denetimli) yolunu, değilse vssm yolunu **son satıra** yazar.

- [ ] **Step 1: Failing testleri yaz** (gerçek checkpoint yok; md5'i monkeypatch ile ver, `load_checkpoint`'u izle)
  - `test_build_model_loads_zenodo_checkpoint_when_md5_matches`
  - `test_build_model_rejects_wrong_checkpoint_md5` → `ValueError`, iletide "md5"
  - `test_build_model_without_pretrained_returns_raw_damageformer` (`final` yolu)
  - `test_run_training_requires_pretrained_for_bright_zero_shot` ve aynısı `run_preflight` için
  - `test_cli_agirlik_with_config_downloads_damageformer` (indirici monkeypatch; çıktının son satırı dönen yol)

- [ ] **Step 2: Koş** → FAIL

- [ ] **Step 3: Uygula**

- [ ] **Step 4: Koş** → `uv run pytest -q` PASS

- [ ] **Step 5: Commit**
  `git commit -m "Fine-tune'u md5 denetimli Zenodo DamageFormer checkpoint'inden başlat"`

---

### Task 4: Son adımın EMA'sını kayıt noktası seç

**Files:**
- Modify: `src/hasar/train/loop.py` (`train` içindeki `run_eval`)
- Test: `tests/test_train_loop.py`

**Interfaces:**
- Consumes: `cfg.eval.select`.
- Produces: `select == "last_ema"` iken `best.pt` yalnız `at_step == total_steps` değerlendirmesinde yazılır ve `best_step`/`best_miou` o değerlendirmeyi gösterir. `"ema_val_miou"` davranışı değişmez.

- [ ] **Step 1: Failing test yaz**
  - `test_last_ema_selection_writes_best_at_final_step`: mevcut küçük sentetik eğitim fixture'ıyla 6 adım, eval.every 2; `torch.load(best.pt)["step"] == 6`, `DONE["best_step"] == 6`.
  - `test_last_ema_selection_survives_resume`: `stop_at=4` ile kes, yeniden çağır; `best.pt` adımı 6.

- [ ] **Step 2: Koş** → FAIL
- [ ] **Step 3: Uygula**
- [ ] **Step 4: Koş** → `uv run pytest -q` PASS
- [ ] **Step 5: Commit**
  `git commit -m "Kayıt noktası olarak son adımın EMA ağırlıklarını seçme kuralını ekle"`

---

### Task 5: Kapı 4 yükleme denetimi, kapı 8 bütçesi yapılandırmadan

**Files:**
- Modify: `src/hasar/preflight/gates.py` (yeni `gate_loaded_checkpoint`)
- Modify: `src/hasar/train/run.py` (kapı 4 ve 8 dalları; `CU_BUDGET` sabiti kaldırılır)
- Test: `tests/test_preflight.py`, `tests/test_train_run_preflight.py`

**Interfaces:**
- Produces: `gate_loaded_checkpoint(model: torch.nn.Module, val_set, expected_miou_pct: float, device: str, tolerance: float = 0.3) -> GateResult`. `hasar.eval.predict.evaluate` ve `summary` ile Faz 3'le aynı yoldan (fp32, TTA yok) ölçer; `value` içinde `miou_pct`, `expected_miou_pct`, `tolerance`. Kapı no 4, ad `GATE_NAMES[3]`. `run_preflight` kapı 4'te: `init == "bright_zero_shot"` ise beklenen değer `<repro_dir>/damageformer_val.json` → `values.miou_pct`; dosya yoksa `_result(4, False, ...)`, iletide dosya yolu. Öteki init'lerde eski `gate_initial_loss`. Kapı 8: `gate_speed(seconds, cfg.train.steps, cu_per_hour(gpu), cfg.train.budget_cu)`.

- [ ] **Step 1: Failing testleri yaz**
  - `test_gate_loaded_checkpoint_passes_within_tolerance` / `..._fails_outside_tolerance` (`evaluate`/`summary` monkeypatch'li, 74,48'e karşı 74,60 ve 75,00)
  - `test_preflight_gate4_fails_cleanly_without_repro_record` (boş `repro_dir`)
  - `test_preflight_gate8_uses_config_budget` (gate_speed'i yakala, `budget_cu == 15.0`)
  - Kapı 11'in ChangeMamba dışı modelde geçtiğini gösteren test yoksa ekle: `BrightBaseline` + CPU dışı koşul atlanır (`pytest.mark.skipif(not torch.cuda.is_available())`).

- [ ] **Step 2: Koş** → FAIL
- [ ] **Step 3: Uygula**
- [ ] **Step 4: Koş** → `uv run pytest -q` PASS
- [ ] **Step 5: Commit**
  `git commit -m "Kapı 4'ü fine-tune'da yükleme denetimine çevir, kapı 8 bütçesini yapılandırmadan oku"`

---

### Task 6: `hasar kiyas` komutu

**Files:**
- Modify: `src/hasar/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ConfusionAccumulator.load_npz(...).per_tile`, `turkey_blocks(path)`, `compare_models`, `compare_record`, `write_result`.
- Produces: `hasar kiyas --a A.npz --b B.npz --name-a ana --name-b varyant --out results/kiyas/varyant_vs_ana.json [--blocks results/arastirma/turkiye_bloklar.json]`. `diff_pct = b − a` (mevcut sözleşme). Kare kümeleri farklıysa Türkçe hata ve çıkış kodu 1. Türkiye görüntüsü okumaz, yalnız matris dosyalarını okur.

- [ ] **Step 1: Failing test yaz**: iki sentetik npz (aynı 4 kare), blok dosyası yerine `--blocks` verilmez → JSON yazılır, `extra.verdict` var, `values.diff_pct` beklenen işaretle; farklı kare kümesinde çıkış 1.
- [ ] **Step 2: Koş** → FAIL
- [ ] **Step 3: Uygula**
- [ ] **Step 4: Koş** → `uv run pytest -q` PASS
- [ ] **Step 5: Commit**
  `git commit -m "Eşleştirilmiş model kıyası için kiyas komutunu ekle"`

---

### Task 7: Colab defteri: iki fine-tune tek "Run All"

**Files:**
- Modify: `notebooks/ana_egitim.ipynb` (başlığı "Fine-tune: kontrol ve SAR varyantı")
- Modify: `scripts/defter_kontrol.py` (`REQUIRED`'e `"configs/varyant.yaml"` ve `"hasar agirlik --config"`; `FORBIDDEN`'a `"selective_scan"`)
- Test: `tests/test_smoke.py` ya da defter kontrolünü çağıran mevcut test; yoksa `tests/test_defter_kontrol.py`

Hücre yapısı:
1. Drive, etiketler, kurulum (mevcut 1. hücre; `CONFIG`/`NAME` kaldırılır, yerine `RUNS = [("configs/ana.yaml", "ana"), ("configs/varyant.yaml", "varyant")]`). Kilit denetimi: `protokol-kilidi` için `configs/ana.yaml configs/varyant.yaml docs/protokol.md`; `varyant-kilidi` için `configs/`.
2. Ortam kapısı: yalnız GPU (A100/L4) ve disk. Çekirdek derleme kodu silinir.
3. Veri; her yapılandırma için `hasar agirlik --config C --out CKPT`, son satır `PRETRAINED[name]`.
4. `run_one(config, name)` fonksiyonu: ön kontrol → (kilitliyse) `hasar egit ... --pretrained PRETRAINED[name] --preflight ...` → (`varyant-kilidi` varsa) `ornekler`, `sicaklik`, `final`. Mevcut 4–6. hücrelerin mantığı bu fonksiyona taşınır; davranış aynı (geçen ön kontrol atlanır, `DONE` atlanır, `turkiye.json` varsa `final` atlanır).
5. `run_one(*RUNS[0])`
6. `run_one(*RUNS[1])`
7. Kopyaları doğrula, `runtime.unassign()`.

- [ ] **Step 1: Defter kontrolüne yeni gereklilikleri ekle, eski defterde düştüğünü gör**
  Run: `uv run python scripts/defter_kontrol.py notebooks/ana_egitim.ipynb` → exit 1
- [ ] **Step 2: Defteri yeniden yaz**, `uv run nbstripout notebooks/ana_egitim.ipynb`
- [ ] **Step 3: Doğrula**
  Run: `uv run python scripts/defter_kontrol.py notebooks/ana_egitim.ipynb` → exit 0; `uv run pytest -q` PASS; `uv run ruff check .` temiz
- [ ] **Step 4: Commit**
  `git commit -m "Defteri iki fine-tune'u tek Run All ile koşacak biçimde yeniden yaz"`

---

### Task 8: Yerel duman testi (RTX 4060, commit yok)

Gerçek veri `data/bright`, checkpoint `data/checkpoints/ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth`. Çıktılar scratchpad'e.

- [ ] **Step 1: Kapı 4, 5, 6, 11 iki yapılandırmayla**
  Run: `uv run hasar on-kontrol --config configs/ana.yaml --data data/bright --out <scratch>/pre_ana --device cuda --pretrained data/checkpoints/ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth --gates 4,5,6,11`
  Expected: 4 geçer ve `miou_pct` 74,48 ± 0,3; 5, 6, 11 geçer. Aynısını `configs/varyant.yaml` ile koş.
- [ ] **Step 2: 20 adımlık prova**: `configs/ana.yaml`'ın scratchpad kopyasında `train.clip: 1.0`, `data.workers: 2`; `uv run hasar egit --config <kopya> --data data/bright --out <scratch>/runs --device cuda --pretrained <ckpt> --steps 20`. Varyant kopyasıyla da. Expected: ikisi de `DONE` yazar, kayıp sonlu.
- [ ] **Step 3: Sonuçları kaydet**. Adım başına süreyi ve tepe belleği not et. Kapı 5 düşerse eşiği değiştirme; dur ve bulguyu kullanıcıya getir. Kodda düzeltme gerekirse ilgili görevin dosyasında yap, testini ekle, ayrı commit at.

---

### Task 9: Karar belgeleri

**Files:**
- Create: `docs/adr/0007-ana-model-hazir-damageformer-fine-tune.md` (şablon `0000-sablon.md`; ADR-0002'nin ana model kararının yerine geçer, ADR-0002'nin başına "Yerini aldı: ADR-0007" satırı)
- Modify: `docs/protokol.md` → 1.3: başlık ve durum satırı; §4'e "Varyant: eğitimde SAR radyometrik artırımı (p, gama, kazanç); değerlendirmede yok"; §5 "son adımın EMA'sı, seçim yok" ve gerekçesi; §6 topluluk satırı "kapsam dışı"; §9'a birincil kıyas (varyant − ana) ve ikincil kıyaslar (referansa karşı, `damageformer_test_matrices.npz`); §14'e 1.3 satırı (tarih, gerekçe, başlık sonucunu etkiliyor mu)
- Modify: `docs/tasarim.md` §5, §7 (reçete tablosu spec §6.2'den), §9 (bütçe), §11 (21 Kasım kararı)
- Modify: `CLAUDE.md` (özet paragrafı: çıpa 37,20 referans + 41,28 bağlam; kural 3: "ana = kontrol fine-tune'u, varyant = SAR fine-tune'u")
- Modify: `docs/fikir-havuzu.md` (yeni satır "A4 SAR radyometrik dayanıklılık artırımı — Seçildi (KN1, 21 Kasım 2026)"; KN1 satırına sonuç)

- [ ] **Step 1: Yaz**. Sayılar spec'ten ve `results/` dosyalarından; elle yeni sayı yok.
- [ ] **Step 2: Doğrula**: `uv run hasar readme --check` → değişiklik yok; `uv run pytest -q` PASS (protokol metnini okuyan test varsa geçmeli)
- [ ] **Step 3: Commit** (iki commit): `Ana modeli hazır DamageFormer'ın kontrollü fine-tune'una çevir (ADR-0007)` ve `Protokolü 1.3'e taşı: SAR varyantı ve son adım seçimi`

---

### Task 10: İşletim ve öğretim belgeleri

**Files:**
- Modify: `docs/colab.md` (yeni akış tablosu, adımlar, maliyet tahmini "ölçülmedi, kapı 8 ölçer"; ChangeMamba çekirdeği bölümü "artık gerekmiyor")
- Modify: `docs/durum.md`, `docs/fazlar.md`, `README.md` (Colab notları)
- Modify: `docs/ogrenme-rehberi.md`, `docs/dersler.md` (yeni ders: fine-tune nedir, düşük lr, EMA'nın checkpoint'ten başlaması, kontrol koşusu, veri artırımıyla alan farkına dayanıklılık; elle çözülmüş küçük örnek: gama/kazanç uygulanmış 3 piksel), `docs/proje-haritasi.md`

- [ ] **Step 1: Yaz**
- [ ] **Step 2: Doğrula**: `uv run hasar readme --check`; belgelerdeki bağlantılar var olan dosyalara gidiyor (`grep` ile kontrol)
- [ ] **Step 3: Commit** (iki commit): `Colab rehberini ve durum belgelerini fine-tune akışına göre güncelle` ve `Ders kitabına fine-tune ve SAR artırımı dersini ekle`

---

### Task 11: Son doğrulama ve devir

- [ ] **Step 1**: `uv run pytest -q` (≥ 737 + yeni testler, hepsi PASS), `uv run ruff check .`, `uv run python scripts/defter_kontrol.py notebooks/ana_egitim.ipynb`, `uv run hasar readme --check`
- [ ] **Step 2**: `git status` temiz; `git log --oneline` atomik commit'leri gösterir
- [ ] **Step 3**: Kullanıcıya devir: push, `protokol-kilidi`'nin 1.3 commit'ine taşınması (`git tag -f protokol-kilidi <sha>` + `git push -f origin protokol-kilidi`), aynı commit'e `varyant-kilidi`, ardından Colab'da "Run All". Komutlar kullanıcıya PowerShell bloklarıyla verilir; Claude etiket atmaz, push etmez.
