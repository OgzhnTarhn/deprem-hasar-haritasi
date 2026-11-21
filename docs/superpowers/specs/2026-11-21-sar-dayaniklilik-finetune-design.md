# Tasarım: SAR dayanıklılığı fine-tune'u (ana model değişikliği)

- **Tarih:** 21 Kasım 2026
- **Durum:** Taslak, kullanıcı onayı bekliyor
- **Etkilediği kararlar:** ADR-0002 (ana model), tasarım §5, §7, §9, §11; protokol 1.2 → 1.3; `CLAUDE.md` kural 3

## 1. Niyet ve neden

ChangeMamba'yı sıfırdan eğitmek Colab'da pahalı çıktı. Ön kontrolün 8. kapısı L4'te adım başına ~3,0 sn ölçtü:
40.000 adım ~33 saat ve ~165 CU eder, bütçe ise 75 CU. Proje öğrenme amaçlı. Kullanıcı şunu yazabilmek istiyor:
"Hazır bir hasar modelini kendi geliştirdiğim bir fikirle fine-tune ettim, 6 Şubat 2023 depreminde kilitli bir
protokolle bir kez test ettim, şu sonucu aldım." Bu cümle CV'ye girecek.

Bu yüzden yeni plan şu:

- Sıfırdan eğitim yok. Başlangıç noktası, BRIGHT'ın yayımladığı ve Türkiye'yi görmemiş **DamageFormer zero-shot
  checkpoint'i**. Türkiye'de `[ZS-EVENT mIoU, repro]` **37,20**, Faz 3'te yeniden üretildi.
- Yeni fikir: **SAR radyometrik dayanıklılık artırımı.** Fine-tune sırasında SAR görüntüsünün parlaklığı ve
  kontrastı rastgele bozulur; model, depremden depreme değişen SAR ölçeğine dayanıklı olmayı öğrenir.

## 2. Kanıt (yalnız 13 olay)

| Bulgu | Kaynak |
|---|---|
| Olayların SAR medyan ortalaması 41,3 (Beirut) ile 104,2 (Myanmar) arasında; havuz ortalaması 62,9 | `results/arastirma/sar_olcek.json` |
| Olay içinde de büyük fark var (La Palma kareleri 50–140) | aynı dosya |
| Dağılımın biçimi olaydan olaya değişiyor: medyan log-çarpıklık −0,92 ile −3,29 arasında | aynı dosya |
| Hazır DamageFormer 13 olayın doğrulamasında `[VAL13 mIoU, repro]` 74,48 alıyor; hasarlı 52,9, yıkık 74,9 | `results/yeniden-uretim/damageformer_val.json` |

Doğrulama kümesi eğitim olaylarından geliyor. Bu yüzden "görülmemiş olaya genelleme" sorusunu ölçemez; yalnız
"fine-tune modeli bozdu mu?" sorusuna cevap verir. Fikir eğitimden önce, yukarıdaki 13 olay kanıtıyla seçildi.
Türkiye'nin sınıf sonuçları Faz 3'te görüldü (kural 1'in birinci istisnası), ama fikrin tasarımında ve
ayarlarında kullanılmaz.

## 3. Hedefler, hedef dışı, karşı hedefler

**Hedefler**

1. Aynı koşullarda iki kısa fine-tune: **kontrol** (fikirsiz) ve **varyant** (SAR artırımlı). Tek fark fikir.
2. Her model Türkiye'de bir kez değerlendirilir. Kıyas, eşleştirilmiş blok bootstrap ve "eşit" kuralıyla yapılır.
3. Toplam maliyet Colab Pro'nun aylık 100 CU'su içinde kalır. Hedef: fine-tune başına ≤ 15 CU, ön kontrol dahil.
4. Öğrenme belgeleri (ders kitabı, rehber) fine-tune'u anlatacak biçimde güncellenir.

**Hedef dışı**

- ChangeMamba eğitimi. Kod depoda kalır ama kullanılmaz; silinmez.
- Birden çok artırım gücünü deneyip seçmek. Tek ayar, önceden yazılır.
- Koherans kanalı (A1), bina ayak izi başlığı (D2) ve öteki varyant adayları.

**Karşı hedefler** (çalışsa bile başarısızlık sayılır)

- Kontrol ile varyantın farklı kırpma, tohum, adım veya örnek sırası görmesi. O zaman fark fikre bağlanamaz.
- Türkiye sonucunu görüp bir ayarı değiştirmek ya da yeniden eğitmek.
- Fark "eşit" çıktığı hâlde "iyileştirdik" yazmak.

## 4. Kısıtlar

- `CLAUDE.md` değişmez kuralları: Türkiye kilidi, Türkiye'yi görmüş ağırlık yok, sayılar dosyadan gelir, metrik
  etiketleri karışmaz.
- **Kural 3 (iki eğitim) yeni anlamıyla:** ana = kontrol fine-tune'u, varyant = SAR fine-tune'u. Hazır checkpoint
  eğitim sayılmaz; o yalnız başlangıç noktası ve referans satırı.
- Colab Pro, L4 ya da A100. Oturum en çok 12 saat. Kopmada "Run All" kaldığı yerden devam eder.
- Yerel RTX 4060 yalnız duman testi ve kısa deneme için.
- `protokol-kilidi` 1.2'nin commit'inde duruyor. `configs/ana.yaml` ve `docs/protokol.md` değişeceği için kilit
  1.3'e taşınır. Eğitim başlamadı, Türkiye'ye dokunulmadı; bu yüzden taşımak kurallara uygun.

## 5. Değerlendirilen yollar

| Yol | Ne | Karşılığı | Karar |
|---|---|---|---|
| A. Kısa ChangeMamba | 512/8, 20.000 adım, sıfırdan | ~6 sa, ~30 CU; 41,28 ile bağ korunur | Seçilmedi: fine-tune öğretmez, ikinci eğitim yine pahalı |
| **B. Hazır DamageFormer + kontrollü fine-tune** | Zenodo checkpoint'inden iki kısa fine-tune | ~1–2 sa ve ≤ 15 CU / koşu (ölçülmedi); temiz A/B | **Seçildi** (kullanıcı, 21 Kasım 2026) |
| C. Eğitimsiz topluluk (B2) | 3 hazır modelin ortalaması | 0 eğitim CU'su | Seçilmedi: eğitim ve fine-tune öğretmez |

B'nin içindeki fikir adayları: SAR dayanıklılığı (seçildi), sıralı kayıp (C3), modalite bırakma (D4).
SAR dayanıklılığı seçildi; 13 olaydan ölçülmüş kanıtı var ve mimariye dokunmuyor.

**Kontrol koşusu neden var:** Hazır modeli yalnız fikirli fine-tune ile kıyaslarsak fark, fikirden mi, bizim kayıp
fonksiyonumuzdan mı (CE + Lovász, nodata yok sayma), yoksa sadece fazladan eğitimden mi geliyor ayırt edilemez.
Aynı reçeteyle fikirsiz bir kontrol bu üçünü ayırır.

## 6. Karar verilmiş tasarım

### 6.1 Modeller ve satırlar

| Satır | Model | Eğitim | Türkiye |
|---|---|---|---|
| Referans | Zenodo DamageFormer, olduğu gibi | Yok | Var: `results/yeniden-uretim/damageformer_test.json` (37,20; Faz 3) |
| **Ana (kontrol)** | Aynı checkpoint + fine-tune, artırım yok | `configs/ana.yaml` | `final` ile bir kez |
| **Varyant** | Aynı checkpoint + fine-tune + SAR artırımı | `configs/varyant.yaml` | `final` ile bir kez |

### 6.2 Ortak fine-tune reçetesi

İki yapılandırma, `aug.sar_radiometric` dışında birebir aynıdır.

| Ayar | Değer | Gerekçe |
|---|---|---|
| Model | `bright_damageformer` (iki akışlı, BRIGHT girdi biçimi) | Checkpoint'in mimarisi |
| Başlangıç | `ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth`, md5 `9276c98a…` denetlenir | Zenodo 15349462 |
| Girdi | Optik ImageNet, SAR `official` | Checkpoint bu biçimle eğitildi; değişirse model bozulur |
| Kırpma / batch | 512 / 8 (mikro parti bellek yetmezse otomatik) | BRIGHT/ChangeMamba resmî betiği |
| Adım | 4.000 (32.000 örnek ≈ eğitim karelerinin 11,4 turu) | Kısa fine-tune |
| Optimizer | AdamW, lr 2e-5, weight decay 5e-3 | Sıfırdan eğitimin lr'sinin (1e-4) beşte biri |
| Zamanlama | 200 adım ısınma, kosinüs, min lr 1e-7 | |
| Kayıp | Projenin kaybı: CE_loc + CE_clf + 0,5·Lovász_loc + 0,75·Lovász_clf, nodata yok sayılır | Kontrol bu etkiyi zaten ayırıyor |
| Hassasiyet, EMA | bf16, EMA 0,999 (EMA checkpoint ağırlıklarıyla başlar) | |
| Gradyan kırpma | Ön kontrolün 9. kapısından (protokol 1.2 kuralı aynı) | |
| Tohum | 42, iki koşuda aynı | Aynı örnek sırası |
| Doğrulama | Her 500 adımda EMA ile 13 olayın doğrulaması; yalnız kayıt | |
| **Kayıt noktası** | **Son adımın EMA ağırlıkları**; doğrulamaya göre seçim yok | Doğrulama genellemeyi ölçmüyor (§2). SAR artırımı doğrulamada varyantı haksız biçimde cezalandırabilir. Seçimsiz kural iki koşuya eşit davranır |

### 6.3 SAR radyometrik artırımı (yalnız varyant)

Eğitim örneğinde, geometrik artırımdan **sonra** ve normalizasyondan **önce**, 0–255 ölçeğindeki SAR'a uygulanır.
Değerlendirmede ve kontrolde uygulanmaz.

1. Olasılık `p = 0,8` ile uygulanır.
2. Gama: `γ ~ logUniform[0,67, 1,5]`, `v' = 255 · (v/255)^γ`. Dağılımın biçimini değiştirir; olaylar arası
   çarpıklık farkını taklit eder.
3. Kazanç: `g ~ logUniform[g_min, g_max]`, `v'' = g · v'`. `g_min = 41,3 / 62,9 ≈ 0,66` ve
   `g_max = 104,2 / 62,9 ≈ 1,66`: en karanlık ve en parlak olayın medyan ortalamasının havuz ortalamasına oranı.
   Sınırlar `sar_olcek.json`'dan hesaplanır, elle yazılmaz.
4. Geçerli pikseller `[1, 255]` aralığına kırpılır. Nodata pikselleri (0) 0 kalır. Nodata maskesi artırımdan önce
   hesaplanır.
5. Rastgele sayılar, örneğin kendi üretecinden geometrik çekilişlerden **sonra** çekilir. Böylece kontrol ve
   varyant her örnekte aynı kırpmayı ve aynı çevirmeyi görür. Kontrolde bu çekilişler hiç yapılmaz.

### 6.4 Ön kontrol kapılarının uyarlanması

| Kapı | Değişiklik |
|---|---|
| 1, 2, 3, 5, 6, 7, 9, 10 | Aynı |
| 4 | Rastgele başlangıç beklentisi (ln 4 / ln 2) yerine **yükleme kapısı**: yüklenen modelin 13 olay doğrulaması `[VAL13 mIoU]` Faz 3 değerine (74,48) ±0,3 içinde olmalı. Bu, checkpoint'in, girdi biçiminin ve metrik yolunun doğru olduğunu tek ölçümde gösterir |
| 8 | Bütçe fine-tune başına 15 CU. Bütçe kod sabiti olmaktan çıkar, yapılandırmaya taşınır |
| 11 | Çekirdek gerekmiyor: model kaydı yazılır, kapı "uygulanmaz" notuyla geçer |

### 6.5 Kıyas ve raporlama

- Başlık metriği değişmez: `[ZS-EVENT mIoU]` (protokol §3).
- **Birincil kıyas: varyant – ana (kontrol).** Eşleştirilmiş blok bootstrap ve §9 "eşit" kuralı.
- İkincil kıyaslar: ana – referans ve varyant – referans. Referansın kare matrisleri Faz 3'ten var
  (`damageformer_test_matrices.npz`).
- 41,28 (ChangeMamba) yalnız bağlam olarak yazılır, hedef değildir.
- Sınıf bazında IoU, özellikle hasarlı ve yıkık sınıfları, ve bina düzeyi F1 her üç satır için raporlanır.
- İkincil satırlar (flip TTA, bina maskeli, bina oylama, nodata yok) protokol §6'daki gibi kalır.

### 6.6 Kilitler ve sıra

1. Kod, yapılandırmalar, protokol 1.3 ve belgeler hazırlanır; testler geçer; yerelde duman testi yapılır.
   Kapı 4, birkaç adım ve iki yapılandırma.
2. Kullanıcı `protokol-kilidi`'ni 1.3'ün commit'ine taşır ve aynı commit'e **`varyant-kilidi`** atar. Varyantın
   tasarımı şimdi tamamen belli; ana modelin Türkiye sonucundan önce kilitlenmiş olur.
3. Colab'da tek "Run All": ana için ön kontrol → fine-tune → `sicaklik` → `final`, ardından varyant için aynısı.
   ChangeMamba çekirdeği hücresi, model ChangeMamba değilse atlanır.
4. Sonuçlar `results/ana/` ve `results/varyant/` altına alınır, kıyas ve README tablosu üretilir.

## 7. Güncellenecek belgeler

`CLAUDE.md` (proje özeti, kural 3), yeni ADR-0007 (ADR-0002'nin ana model kararının yerine geçer), `tasarim.md`
§5, §7, §9, §11, `protokol.md` 1.3 (§4, §5, §6 topluluk satırı, §9 kıyaslar, §14), `fikir-havuzu.md` (yeni fikir
satırı ve KN1'in sonucu), `durum.md`, `fazlar.md`, `colab.md`, README'nin Colab notları,
`ogrenme-rehberi.md` ve `dersler.md` (fine-tune ve veri artırımı anlatımı), `proje-haritasi.md`.

## 8. Uygulayıcıya bırakılanlar

- Yapılandırma alanlarının adları ve yeri (`model.init`, `aug.sar_radiometric`, gate 8 bütçesi).
- Notebook'un iki yapılandırmayı arka arkaya nasıl çalıştırdığı.
- Testlerin ayrıntısı. Asgari kapsam: artırımın nodata'yı koruması, aralık sınırları, tohumla belirlenim,
  kontrol ile varyantın aynı geometrik çekilişleri görmesi, kapı 4'ün yeni hâli, md5 denetimi.

Uygulayıcının denetlemesi gerekenler:

- Kapı 4'teki doğrulama, Faz 3 yeniden üretimiyle aynı yoldan ve aynı hassasiyetle çalışmalı. Yoksa ±0,3
  toleransı hassasiyet farkını ölçer.
- Referansın kare matrisleri (`damageformer_test_matrices.npz`) `final`'in ürettiği matrislerle aynı biçimde ve
  aynı kare sırasında olmalı. Eşleştirilmiş bootstrap buna dayanıyor; tutmazsa referans kıyası yalnız aralıkla
  yapılır.
- Kapı 8'in 15 CU bütçesi yalnız fine-tune projeksiyonunu kapsar; ön kontrol bunun dışındadır.

## 9. Claude'un verdiği kararlar (kullanıcı düzeltebilir)

| Karar | Gerekçe |
|---|---|
| Kontrol koşusu eklemek (ana = fikirsiz fine-tune) | A/B'yi temiz yapar; kural 3'ün iki eğitimine oturur |
| lr 2e-5, 4.000 adım, 512/8 | Kısa ve ucuz bir fine-tune; resmî kırpma |
| Kayıt noktası: son adımın EMA'sı | Doğrulama genellemeyi ölçmüyor; iki koşuya eşit davranır. Protokol §5 değişir |
| Artırım aralıkları `sar_olcek.json`'dan, gama [0,67, 1,5], p 0,8 | 13 olayın ölçülmüş farkını kapsar |
| Kapı 4'ü yükleme kapısına çevirmek | Rastgele başlangıç beklentisi fine-tune'da anlamsız |
| `varyant-kilidi`'ni `protokol-kilidi` ile aynı commit'e atmak | Fikir şimdiden kesin; tek "Run All" iki modeli de bitirebilir |
| Fine-tune başına 15 CU bütçe | İki koşu ve ön kontroller Pro'nun 100 CU'suna sığar |

## 10. Riskler

| Risk | Etki | Önlem |
|---|---|---|
| Fine-tune modeli bozar (öğrenme oranı yüksek) | Ana ve varyant referansın altında kalır | Düşük lr; kontrol satırı bunu görünür kılar; sonuç yine raporlanır |
| Varyantın 13 olay doğrulaması kontrolün altında kalır | Beklenen; artırım dağılım içindeki başarıdan biraz feda eder | Seçim doğrulamaya göre yapılmıyor; raporda yazılır |
| Fark "eşit" çıkar | CV cümlesi "etkisi ölçülemeyecek kadar küçük" olur | Kural 6; olumsuz sonuç da yazılır |
| Maliyet tahmini tutmaz | Bütçe aşılır | Kapı 8 her koşudan önce ölçer ve gerekirse durdurur |
