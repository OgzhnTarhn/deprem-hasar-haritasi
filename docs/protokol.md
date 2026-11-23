# Değerlendirme Protokolü (önceden kayıt)

- **Durum:** Son metin, sürüm 1.4 (23 Kasım 2026). Zero-shot kuralları 1.3 ile aynı; 1.4 yalnız §15'teki few-shot ek deneyini ekler ve `fewshot-kilidi` etiketiyle kilitlenir. Sürüm 1.3 (21 Kasım 2026): `protokol-kilidi` etiketi bu sürümü içeren commit'e taşınır; aynı commit'e `varyant-kilidi` atılır.
  `train.clip` yapılandırmada boş kalır; eğitim eşiği aynı yapılandırmanın geçmiş ön kontrolünün 9. kapısından
  kendisi okur (tasarım §7–8). Bu yüzden kilit, ön kontrolden önce atılabilir. Kilitten sonra bu dosya değiştirilmez; değişiklik gerekirse yeni sürüm, gerekçesi ve
  tarihiyle §14'e yazılır ve başlık sonucunu etkileyen her değişiklik raporda belirtilir.
- **Kapsam:** Türkiye'deki tek değerlendirmenin bütün hesap kuralları: bölme, başlık metriği, girdi işleme, kayıt
  noktası seçimi, ikincil satırlar, bina düzeyi metrikler, güven aralıkları, "eşit" kuralı, kalibrasyon.
- **Dayanaklar:** [ADR-0001](adr/0001-zero-shot-bolme.md), [ADR-0004](adr/0004-sar-girdisi.md),
  [ADR-0005](adr/0005-baslik-metrigi.md), [ADR-0007](adr/0007-ana-model-hazir-damageformer-fine-tune.md); BRIGHT ve ChangeMamba kodunun Faz 0 okuması.

**İşaretler:** ✅ kararlaştırıldı · 🟨 öneri, kullanıcı onayı bekliyor · ⬜ açık, belirtilen fazda ölçülüp
kararlaştırılacak.

---

## 1. Değişmezler

1. Türkiye'nin ne görüntüsü ne etiketi eğitimde, doğrulamada, kayıt noktası seçiminde, eşik/sıcaklık ayarında veya
   normalizasyon istatistiğinde kullanılır. ✅
2. Her model Türkiye'de yalnız `final` adımında, bir kez değerlendirilir. `final`, `protokol-kilidi` etiketi ve temiz
   çalışma ağacı olmadan çalışmaz (`guard`). ✅
3. Varyantın tasarımı ana modelin Türkiye sonucu görülmeden `varyant-kilidi` ile kilitlenir. ✅
4. Bu dosyadaki her ayar Türkiye'ye dokunmadan, yalnız 13 olayın verisiyle seçilir. ✅

## 2. Bölme ✅

BRIGHT `cross_event/zero_shot` mantığı, `test_event = turkey-earthquake` (ADR-0001).

| Küme | Kaynak | Kare |
|---|---|---|
| Eğitim | Standart eğitim + standart test listesi, Türkiye hariç | 2.795 |
| Doğrulama | Standart doğrulama listesi, Türkiye hariç | 337 |
| Test | Türkiye'nin üç standart listedeki bütün kareleri | 1.114 |

13 eğitim olayı: bata-explosion, beirut-explosion, congo-volcano, haiti-earthquake, hawaii-wildfire,
la_palma-volcano, libya-flood, marshall-wildfire, mexico-hurricane, morocco-earthquake, myanmar-hurricane,
noto-earthquake, ukraine-conflict.

Liste dosyaları BRIGHT deposunun sabit bir commit'inden alınır (`third_party/bright/SOURCE.md`). Kare sayıları, olay
başına sayılar, eğitim ve doğrulamanın sınıf piksel dağılımı ve liste özetleri (satır sonundan bağımsız sha256)
`results/veri/bolme.json`'dadır; `hasar veri-ozet` üretir. Türkiye için yalnız kare sayısı yazılır, etiketi okunmaz.

## 3. Başlık metriği ✅

Etiket: `[ZS-EVENT mIoU]`. Resmî kod yolu (ADR-0005):

1. **Girdi:** Türkiye'nin her karesi 1024 × 1024 tam olarak, kırpmadan, tek tek modele verilir.
2. **Tahmin:** `pred = argmax(clf)`; TTA yok, bina maskesi yok, sıcaklık yok.
3. **Matris:** 4×4 karışıklık matrisi `C` bütün 1.114 kare boyunca biriktirilir. Etiketi [0, 4) dışında kalan
   pikseller matrise girmez. Nodata pikselleri ayrıca dışlanmaz (resmî kodla aynı).
4. **IoU:** her sınıf k için `IoU_k = C[k,k] / (Σ_j C[k,j] + Σ_i C[i,k] − C[k,k] + 1e-7)`.
5. **mIoU:** dört sınıfın (arka plan, sağlam, hasarlı, yıkık) ortalaması.
6. **Kıyas:** Birincil fark varyant − ana, ikincil farklar referansa karşıdır (§9).
   ChangeMamba 41,28 `[ZS-EVENT mIoU]` yalnız bağlamdır.

Her karenin 4×4 matrisi ayrıca saklanır (`results/<model>/turkiye_kare_matrisleri_<satır>.npz`; başlık için `_headline`). Bootstrap (§8) ve bütün
alt kırılımlar bu matrislerden hesaplanır; model yeniden çalıştırılmaz.

## 4. Girdi işleme

| Madde | Kural | Durum |
|---|---|---|
| Optik | İlk 3 bant; ImageNet ortalama/standart sapması (0–255 ölçeği: 123,675 / 116,28 / 103,53; 58,395 / 57,12 / 57,375) | ✅ |
| SAR | Resmî biçim: tek bant 3 kanala kopyalanır, ImageNet istatistikleriyle normalize edilir (ADR-0004, seçenek C) | ✅ |
| SAR ölçeği | Ölçüldü (13 olay): değerler doğrusal genlik değil, log/dB benzeri ve kırpılmış; olay ortalamaları 41–104 (`results/arastirma/sar_olcek.json`). Kontrolde ek dönüşüm yok; varyantta yalnız eğitimde aşağıdaki artırım var | ✅ |
| Varyant SAR artırımı | Yalnız eğitimde p = 0,8; γ ~ logUniform[0,67, 1,5], kazanç ~ logUniform[0,657, 1,657]. Kazanç sınırları `results/arastirma/sar_olcek.json` içindeki 41,3 / 62,9 ve 104,2 / 62,9 oranlarından gelir. Kontrolde ve değerlendirmede uygulanmaz | ✅ |
| Eğitimde nodata | Optik ve SAR'ın ikisi de 0 olan pikseller kayıpta `ignore_index = 255` ile yok sayılır | ✅ (reçete; değerlendirmeyi etkilemez) |
| Değerlendirmede nodata | Resmî kodla aynı; ayrıca dışlanmaz. Yok sayılmış hâl ikincil satır | ✅ |

SAR artırımı geometriden sonra, normalizasyondan önce uygulanır: `255·(v/255)^γ·k`.
Geçerli pikseller [1, 255]'e kırpılır; nodata 0 kalır. Nodata maskesi artırımdan önce hesaplanır.
Ortak tohum ve örnek sırası korunur; SAR çekilişleri geometrik çekilişlerden sonra yapılır.

## 5. Kayıt noktası ✅

- Kayıt noktası **son adımın EMA ağırlıklarıdır** (`last_ema`); doğrulamaya göre seçim yoktur.
  EMA hazır checkpoint ağırlıklarıyla başlar. İki koşu aynı kurala tabidir.
- Her 500 adımda EMA, yalnız 13 olayın doğrulamasında §3'teki yoldan değerlendirilir; sonuç yalnız kaydedilir.
- Gerekçe: doğrulama eğitim olaylarından gelir; görülmemiş olaya genellemeyi ölçmez. SAR artırımı varyantı
  bu doğrulamada haksız biçimde cezalandırabilir. Seçimsiz kural kontrol ve varyanta eşit davranır
  ([onaylı tasarım §6.2](superpowers/specs/2026-11-21-sar-dayaniklilik-finetune-design.md)).
- Ham ağırlıkların son adım doğrulaması `val13.json` içinde `last_raw_miou_pct` olarak kaydedilir; kullanılmaz.

## 6. İkincil satırlar (önceden ilan)

Bütün ikincil satırlar Faz 6'daki tek çıkarımın önbelleğinden hesaplanır.

| Satır | Tanım | Ayar kaynağı | Durum |
|---|---|---|---|
| Flip TTA | Kimlik, yatay, dikey ve ikisi birden; softmax olasılıklarının ortalaması, sonra argmax | Ayar yok | ✅ |
| Bina maskeli harita | `loc` argmax'ı 0 olan piksellerde tahmin arka plan; ötekilerde `argmax(clf[1:]) + 1` | Ayar yok | ✅ |
| Bina düzeyi oylama | Tahmin edilen bina maskesinin (`loc` argmax) 8-komşuluklu bağlı bileşenlerinde `argmax(clf[1:]) + 1` çoğunluğu; birim tahminden gelir, Türkiye etiketi kullanılmaz | Ayar yok | ✅ |
| Nodata yok sayılmış | §3'te nodata pikselleri matrise alınmaz | Ayar yok | ✅ |
| Topluluk satırları | kapsam dışı (1.3) | Uygulanmaz | ✅ |

Her satırın etiketi `[ZS-EVENT mIoU]` ile birlikte satır adını taşır: `flip-tta`, `bina-maskeli`, `bina-oylama`,
`nodata-yok` (ör. `[ZS-EVENT mIoU, flip-tta]`, dosya `turkiye_flip-tta.json`); başlık kıyası yalnız §3 ile yapılır.

## 7. Bina düzeyi metrikler ✅

- **Faz 2 ölçümü:** BRIGHT'ın CVPRW26 örnek etiketleri (COCO biçimi) Türkiye'nin 1.114 karesinin 1.001'inde var
  (yalnız dosya adlarından sayıldı). 13 olayda etiketin bağlı bileşenleri örnek sayısının 0,61'i kadar; bitişik
  binaların yaklaşık %39'u birleşiyor (`results/arastirma/bina_birlesme.json`). Bağlı bileşen bina birimi olarak
  zayıf.
- **Örnek kaynağı:** CVPRW26 örnek etiketleri. Metrik örnek etiketi olan 1.001 karede hesaplanır; kare listesi
  dosya adlarından `protokol-kilidi`'nden önce sabitlendi (`results/arastirma/turkiye_bina_kareleri.json`, sha256
  `1db74d313eff…`), `final` bu listeyle tutmayan örnek klasörünü reddeder; sonuçta "1.001 kare" diye yazılır. Kalan 113 kare
  bina düzeyi metriğe girmez (piksel metrikleri bütün 1.114 karede).
- **Gerçek sınıf:** Örneğin etiket piksellerinin çoğunluk sınıfı.
- **Tahmin sınıfı:** Örneğin içindeki piksellerin `argmax(clf)` çoğunluğu; arka plan çoğunluktaysa "kaçırıldı".
- **Metrik:** Sağlam, hasarlı ve yıkık sınıflarının F1'leri ve makro ortalaması `[ZS-EVENT bina-makro-F1]` (`turkiye_bina_f1.json`).
- Ayrıca piksel düzeyi "üç bina sınıfının IoU ortalaması" raporlanır.

## 8. Güven aralıkları

| Madde | Kural | Durum |
|---|---|---|
| Yöntem | Yerine koyarak yeniden örnekleme; her örneklemde seçilen birimlerin kare matrisleri toplanır, §3'teki metrik yeniden hesaplanır | ✅ |
| Birim | Şehre göre katmanlı blok bootstrap. Türkiye kareleri GeoTIFF başlıklarına göre 7 şehirde (Gaziantep 181, Osmaniye 179, Adıyaman 171, Malatya 168, Kahramanmaraş 160, Antakya 130, İslahiye 125); 2 km ızgarada 64 blok, blok başına 1–42 kare. Her örneklemde her şehrin blokları kendi içinde yerine koyarak çekilir; şehir payları sabit kalır. 7 birimle şehir düzeyi bootstrap güvenilir değil. Yalnız koordinat okundu (`results/arastirma/turkiye_bloklar.json`) | ✅ |
| Örnek sayısı | 10.000 | ✅ |
| Aralık | %95, yüzdelik (2,5 ve 97,5) | ✅ |
| Tohum | Sabit (`results` dosyasına yazılır) | ✅ |
| İki modelin kıyası | Eşleştirilmiş bootstrap: her örneklemde iki modelin farkı aynı bloklarla hesaplanır | ✅ |

## 9. "Eşit" kuralı ✅

Birincil kıyas **varyant − ana (kontrol)**. İkincil kıyaslar **ana − referans** ve **varyant − referans**.
Referans hazır Zenodo DamageFormer'dır: 37,20 `[ZS-EVENT mIoU, repro]`;
kare matrisleri `results/yeniden-uretim/damageformer_test_matrices.npz` dosyasındadır.

`hasar kiyas` farkı **b − a** olarak hesaplar. Örnek birincil kıyas:

```bash
uv run hasar kiyas --a results/ana/turkiye_kare_matrisleri_headline.npz --b results/varyant/turkiye_kare_matrisleri_headline.npz --name-a ana --name-b varyant --out results/kiyas/varyant_vs_ana.json --blocks results/arastirma/turkiye_bloklar.json
```

Üç kıyasta da `--blocks results/arastirma/turkiye_bloklar.json` açıkça verilir.
Aynı kareler ve şehir katmanlı bloklarla eşleştirilmiş bootstrap kullanılır (§8).
Kare kümeleri farklıysa komut reddeder; referans eşleştirmesi sağlanamazsa yalnız aralıkla kıyas yazılır.

Farkın mutlak büyüklüğü **1,0 puanın altındaysa veya** eşleştirilmiş bootstrap farkının %95 aralığı 0'ı
içeriyorsa sonuç “eşit” yazılır. Aksi hâlde fark ve aralığı raporlanır. Bu kural üç kıyasa da uygulanır.
ChangeMamba 41,28 `[ZS-EVENT mIoU]` yalnız bağlamdır; hedef veya birincil kıyas değildir.

## 10. Kalibrasyon ve belirsizlik

| Madde | Kural | Durum |
|---|---|---|
| Sıcaklık | `clf` logitleri için tek bir T, 13 olayın doğrulamasında negatif log olabilirlikle seçilir; Türkiye'de yalnız ölçülür | ✅ |
| ECE | Bina pikselleri (etiket 1–3) üzerinde, en yüksek olasılıklı sınıfın güveni, 15 eşit genişlikli kutu | ✅ |
| Bina belirsizliği | Flip TTA tahminlerinin ortalamasının entropisi, bina örneği içinde ortalanır | ✅ |
| Risk-kapsam | Binalar belirsizliğe göre sıralanır; AURC raporlanır | ✅ |

## 11. Türkiye erişimi ve önbellek ✅

- Türkiye karelerine erişen tek kod yolu `eval/final.py`; `guard` etiketi ve temiz çalışma ağacını denetler, commit
  SHA'sını sonuç dosyasına yazar.
- Tek çıkarımda her kare için saklananlar: `clf` ve `loc` softmax olasılıkları (float16), flip TTA olasılıkları,
  kare matrisleri. Bütün ikincil satırlar, analizler ve demo bu önbelleği okur.

## 12. Raporlama ✅

Her sonuç `results/**/*.json`'a `metric_tag`, commit SHA, config hash ve tohumla yazılır. Başlık tablosu:
4 sınıfın IoU'ları, mIoU ve %95 aralığı, üç bina sınıfının IoU ortalaması, bina düzeyi makro F1, ikincil satırlar,
varyant − ana ve referansa karşı ikincil farklar, §9'a göre yorum. ChangeMamba 41,28 `[ZS-EVENT mIoU]` yalnız bağlamdır.

Referansın Faz 3 kaydı sınıf IoU'larını içerir; bina düzeyi F1 içermez. Kare piksel matrislerinden
bina F1 türetilemez. Bu alan referans satırında “mevcut değil” yazılır; yeni Türkiye çıkarımı yapılmaz.
Ana ve varyantın bina F1 kayıtları kendi kilitli `final` adımlarından gelir.

## 13. Açık maddeler

Kapı 7'nin beklenen değeri Faz 3'te kapandı (arka plan oranı / 4, ≈ 0,226; kapı kodu doğrulama etiketlerinden
hesaplar). KN1 kapandı: A4 seçildi; topluluk 1.3 kapsamı dışında. Bu başlıkta açık karar kalmadı.

## 14. Değişiklik günlüğü

| Sürüm | Tarih | Değişiklik |
|---|---|---|
| 0.1 | 6 Kasım 2026 | İlk taslak; BRIGHT ve ChangeMamba değerlendirici kodu okunarak yazıldı |
| 0.2 | 6 Kasım 2026 | Faz 2 ölçümleri: SAR ölçeği (§4 ✅), bölme özetleri (§2), bina örnek kaynağı ve oylama birimi önerisi (§6–7 🟨), katmanlı blok bootstrap önerisi (§8 🟨), kapı 7 beklenen değeri |
| 0.3 | 7 Kasım 2026 | Kullanıcı bekleyen önerilerin hepsini kabul etti: SAR girdisi (§4, ADR-0004 C), oylama birimi (§6), bina örnek kaynağı (§7), şehre göre katmanlı blok bootstrap (§8), "eşit" kuralı (§9) ✅ |
| 1.0 | 7 Kasım 2026 | Faz 3–4 sonrası son metin. Faz 3: SiamCRNN, UNet, DamageFormer ±0,3 içinde (`results/yeniden-uretim/`), hat resmî değerlendiriciyle birebir; DeepLabV3+ zero-shot checkpoint'i yayımlanan değerin çok üstünde çıktı, Türkiye'yi görmüş sayılıp topluluk adaylarından çıkarıldı (§6). Kapı 1 yalnız bu üç modeli şart koşar. §5: ham ağırlıkların doğrulaması son adımda bir kez. §6 flip TTA kodda dört görünüm. Kapı 10'un GPU ölçütü (tasarım §7): parametreler `atol` 1e-3, kayıplar %1; BatchNorm istatistikleri yalnız raporlanır. Kapı 7 kapandı. §7 bina karesi listesi dosyaya sabitlendi |
| 1.1 | 7 Kasım 2026 | Faz 6 kodu: adlar koddakiyle hizalandı (§3 kare matrisi dosyası satır başına, §6 satır etiketleri, §7 `[ZS-EVENT bina-makro-F1]`). `final` başlık dosyasını (`turkiye.json`) bütün öteki kayıtlardan sonra yazar; yarıda kalan koşu önbellekten devam eder, başlık yazıldıktan sonra ikinci koşu reddedilir. Hesap kuralı değişmedi |
| 1.2 | 18 Kasım 2026 | Gradyan kırpma eşiği elle yazılmıyor: `train.clip: null` iken `hasar egit` eşiği, yapılandırma hash'i tutan, tamamlanmış ve bütün kapıları geçmiş `preflight.json`'ın 9. kapısından (2× lr'de 500 adım, p90 gradyan normu) okur; kayıt yoksa ya da tutmazsa eğitim başlamaz. Kural önceden sabit olduğu için kilit ön kontrolden önce atılır ve Colab defteri tek "Run All" ile ön kontrol + eğitim yapar. Eşik Türkiye'ye dokunmadan seçilir; hesap kuralı değişmedi |
| 1.3 | 21 Kasım 2026 | Ana model hazır DamageFormer kontrol fine-tune'u; varyant SAR radyometrik artırımı (§4); son adımın EMA'sı, seçim yok (§5); topluluk kapsam dışı (§6); birincil varyant − ana, ikincil referans kıyasları (§9). Neden: Colab L4 kapı 8 ölçümünde ChangeMamba projeksiyonu 164,6 CU > 75 CU. Eğitim başlamadı; yeni eğitim ve `final` için Türkiye'ye dokunulmadı (Faz 3 yeniden üretimi istisnası saklı). Kullanıcı 21 Kasım 2026'da onayladı. Başlık sonucunu etkiler: **evet, model değişiyor** (ADR-0007). |
| 1.4 | 23 Kasım 2026 | Few-shot ek deneyi (§15): Türkiye blok bazlı uyarlama/tampon/test bölmesi, kontrollü A/B (düz ve hasar ağırlıklı kayıp), ana checkpoint'inden başlangıç, yeni etiket `[FS-EVENT mIoU]`, `fewshot-kilidi`. Neden: kullanıcı 22–23 Kasım 2026'da az etiketle uyarlamayı ikinci deney olarak istedi; motivasyon zero-shot sonucundaki zayıf hasar sınıfları, açıkça yazılır. Zero-shot başlık sonucunu etkiler: **hayır** (ADR-0008). |

## 15. Few-shot ek deneyi (1.4) ✅

Bağlayıcı ayrıntı: [tasarım](superpowers/specs/2026-11-23-turkiye-few-shot-design.md) (§9 üstteki bölümlerin önüne geçer),
[ADR-0008](adr/0008-turkiye-few-shot-ek-deneyi.md). Zero-shot sonuçları (`results/ana`, `results/varyant`) değişmez.

| Konu | Kural |
|---|---|
| Bölme | Yalnız koordinattan: `turkey_blocks` blokları; her şehirde `seed 2023` ile karıştırılan bütün bloklar, şehrin karelerinin %20'sine ulaşana dek uyarlama. Uyarlama merkezine 0,5 km'den yakın diğer kareler tampon (hiç kullanılmaz), kalanı test. `results/fewshot/bolme.json`, listelerin sha256'sıyla; yeniden çekilmez |
| Erişim | `fewshot-kilidi` HEAD'in atası, temiz ağaç, kilitli dosyalar (`docs/protokol.md`, `configs/fewshot_a.yaml`, `configs/fewshot_b.yaml`, `results/fewshot/bolme.json`) etiketten beri aynı. Eğitim yalnız `adapt`, test yolu yalnız `test` karelerini okur. Doğrulama, kayıt noktası, eşik 13 olaydan |
| Başlangıç | Ana koşunun son EMA'sı (`ana-708fdc1f5ecb/best.pt`); sha256 yapılandırmada kilitli; kapı 4 74,55 ± 0,3 `[VAL13 mIoU]` (`results/ana/val13.json`) |
| Tarif | Ana ile aynı (4000 adım, batch 8, lr 2e-5, EMA 0,999, son adım EMA); her partinin yarısı Türkiye uyarlama, yarısı 13 olay; SAR artırımı kapalı; `train.clip` 3,1627 (ana kapı 9) iki kolda aynı; `micro_batch` 4 sabit |
| A/B | Tek fark `loss.class_weights`: A `(1, 1, 1, 1)`; B `(1, 1, 2,01, 2,70)` = karekök(f_sağlam / f_c), 13 olay eğitim pikselleri |
| Sıra | İki eğitim de bitmeden Türkiye testi yok; her model test bölgesinde bir kez |
| Metrik | `[FS-EVENT mIoU]` test karelerinde, §3 başlık yolu; şehir katmanlı blok bootstrap. Ek: `[FS-EVENT bina-makro-F1]` (örnek etiketli test kareleri). 41,28 ile ve `[ZS-EVENT mIoU]` ile kıyaslanmaz; sıcaklık, kalibrasyon, TTA yok |
| Kıyaslar | Birincil B − A; ikincil A − zs-ana, B − zs-ana (zs-ana = ana zero-shot matrislerinin test karelerine süzülmesi, kilitten sonra kodla); ikincil sınıf bazlı B − A hasarlı ve yıkık IoU. Bloklar `(şehir, blok)` ile hizalanır. §9 "eşit" kuralı |
| Raporlama | A − zs-ana "Türkiye uyarlama etiketleri + ek 4000 adım eğitim" diye yazılır. Motivasyonun zero-shot sonucundan geldiği ve blok bölmesinin uzamsal benzerlik riski raporda yazılır |
