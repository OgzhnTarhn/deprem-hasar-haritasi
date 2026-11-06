# Rakip analizi: uydu tabanlı bina hasar haritalama (optik + SAR) ve Şubat 2023 Türkiye depremi

Gözlem tarihi: 5 Kasım 2026, yaklaşık 18:00–18:30 UTC. Yıldız, fork ve "son push" değerleri bu tarihte GitHub API'den alındı. Yöntem notları:

- "Son push", API'deki `pushed_at` alanıdır. Ana daldaki son commit'ten daha yeni olabilir, otomatik push'ları da içerebilir.
- Hiçbir repo çalıştırılmadı. "Tekrarlanabilirlik" sütunu yalnız README ve dosya yapısında görülenleri anlatır.
- Makale sayıları çoğunlukla sayfanın özetlenmiş çıktısından alındı. Rapora tablo değeri olarak geçmeden önce birincil tablodan bakılmalı. Çıplak gözle doğrulanmayanlar işaretlendi.
- `[YARGI]` etiketi kaynaklı bilgi değil, benim değerlendirmemdir.

## GitHub'da açık kaynak: BRIGHT, xView2/xBD ve SAR tabanlı hasar haritalama

### Takeaway
Ekosistem üç kümeye ayrılıyor: bakımı süren BRIGHT çevresi (optik öncesi + SAR sonrası, Temmuz 2026'da güncellendi), 2019–2020'den kalan ve çoğu yalnız arşiv durumundaki xView2 çözümleri, dağınık SAR araçları. Yıldız sayıları düşük, en güçlü iş BRIGHT ve onun görülmemiş olay kurulumu. Bu kurulum Türkiye'yi test olayı olarak seçmeye doğrudan izin veriyor, yani "Türkiye'yi eğitimde görmedik" iddiası tek başına yeni değil.

### Cited Findings

**A. BRIGHT çevresi ve optik + SAR çalışmaları**

| Repo | Yıldız / fork | Son push | Yöntem | Veri | Bildirilen metrik | Durum ve tekrarlanabilirlik |
|---|---|---|---|---|---|---|
| [ChenHongruixuan/BRIGHT](https://github.com/ChenHongruixuan/BRIGHT) | 262 / 44 | 2026-07-21 | Veri seti + kıyaslama: standart ML, sıfır/tek örnekli olaylar arası, UDA, SSL, gözetimsiz değişim, görüntü eşleme | Öncesi optik, sonrası SAR, 14 olay, 0,3–1 m | [ChangeMamba: standart mIoU 67,63; Türkiye 52,70; olaylar arası ort. 39,38 (sıfır örnek), 43,23 (tek örnek); Türkiye sıfır örnek 41,28, tek örnek 42,80](https://arxiv.org/html/2501.06019v2) | Bakımda. Apache-2.0. Kod yalnız Linux'ta denenmiş. [Ağırlıklar Zenodo'da](https://zenodo.org/records/15349462), TorchGeo ve TorChange'e eklenmiş. Örnek komut 800.000 iterasyon |
| [ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba) | 657 / 49 | 2026-04-24 | BRIGHT makalesinde en iyi model | BRIGHT makalesinde kullanıldı | BRIGHT satırına bakın | Yalnız üst veri okundu |
| [Henryjiepanli/Building-Guided-Pseudo-Label...](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping) | 20 / 0 | 2025-11-27 | DFC25 Track 2 birincisi. İki aşama: bina çıkarımı, ardından düşük belirsizlikli sözde etiketli değişim tespiti | DFC25 bölünmesi, test olayları [Noto ve Marshall](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/README_DFC25.md) | [mIoU 54,28](https://arxiv.org/abs/2505.04941) | Lisans görünmüyor. README yalnız çıkarım adımlarını anlatıyor |
| [zxqyiyang/DF2025](https://github.com/zxqyiyang/DF2025) | 4 / 0 | 2025-04-17 | DFC25 ikincisi | DFC25 | [0,5081, metrik adı sayfada yok](https://www.grss-ieee.org/community/technical-committees/winners-of-the-2025-ieee-grss-data-fusion-contest-all-weather-land-cover-and-building-damage-mapping/) | Yalnız üst veri okundu |
| huang-he99/BightSolution | Okunamadı | Okunamadı | [CVPRW 2026 BRIGHT Challenge birincisi, sahne ayrımlı sözde etiket](https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/README.md) | Örnek düzeyinde BRIGHT, test: 2025 Kaliforniya yangını, 2025 Jamaika kasırgası | Skor okunmadı | BRIGHT README'sindeki bağlantı 5 Kasım 2026'da 404 döndü |
| [whf-68/Damage-Aware-SAR-Optical-Query...](https://github.com/whf-68/Damage-Aware-SAR-Optical-Query-Learning-framework) | 0 / 0 | 2026-07-21 | CVPRW 2026 ikincisi | Aynı | Skor okunmadı | Yalnız üst veri okundu |
| [KotlinWang/DisasterEO](https://github.com/KotlinWang/DisasterEO) | 13 / 1 | 2026-08-30 | DamageNet + ReliaDA, görülmemiş olaya aktarım | Öncesi optik, en erken optik/SAR kanıtı, [Zenodo](https://zenodo.org/records/18918459) | Okuduğum kısımda sayı yok | Etkin, Ağustos 2026 Nepal olayı için harita yayınlamış |
| [chunnn26/DisasterBridge](https://github.com/chunnn26/DisasterBridge) | 0 / 0 | 2026-10-06 | Köprü difüzyonu ile hasara koşullu SAR sentezi | README'de veri ayrıntısına bakılmadı | Okunmadı | Yeni, MIT |
| [ya0-sun/PostEQ-SARopt-BuildingDamage](https://github.com/ya0-sun/PostEQ-SARopt-BuildingDamage) (QuickQuakeBuildings) | 43 / 2 | 2024-10-04 | Olay sonrası SAR + optik, ikili sınıflama | Capella ve Maxar, OSM/HOT etiketleri, 4.000+ bina, Türkiye 2023 | [İslahiye: F1 0,282, AUROC 0,769 (Russo et al. aktarımı)](https://arxiv.org/html/2506.22338) | MIT, veri Terabox'ta. Öncesi görüntü yok |
| [m-usmanzahid/cs437-group40...](https://github.com/m-usmanzahid/cs437-group40-multimodal-earthquake-damage-assessment) | 1 / 0 | 2026-05-08 | Ders projesi, CAFUNet füzyonu, MC Dropout | BRIGHT, xBD | Türkiye dışarıda tutulunca ikili IoU 0,4953, histogram eşlemeyle 0,5117; ECE 0,0352 | Lisans görünmüyor. Çok tohumlu test yok, kendisi söylüyor |

**B. xView2 / xBD (optik öncesi + optik sonrası)**

| Repo | Yıldız / fork | Son push | Yöntem | Bildirilen metrik | Durum ve tekrarlanabilirlik |
|---|---|---|---|---|---|
| [DIUx-xView/xView2_baseline](https://github.com/DIUx-xView/xView2_baseline) | 243 / 93 | 2023-07-26 | Resmi referans: yerelleştirme + sınıflama | [F1 toplam 0,27 (ikincil kaynak)](https://github.com/NSNNTL/ProtoNet-xBD) | Arşiv niteliğinde, Python 3.6 dönemi |
| [vdurnov/xview2_1st_place_solution](https://github.com/vdurnov/xview2_1st_place_solution) | 105 / 47 | 2023-03-25 | Birinci: U-Net yerelleştirme, Siyam sınıflama, topluluk | README'de skor yok | MIT. PyTorch 1.1, Apex, 2 GPU'da yaklaşık 7 gün. Yazar "eğitim tekrarlanabilirliği için düzenlenmedi" diyor. Ağırlık bağlantısı var, canlılığı denenmedi |
| [selimsef/xview2_solution](https://github.com/selimsef/xview2_solution) | 61 / 18 | 2020-04-17 | İkinci: U-Net, Siyam U-Net | README'de skor yok | Dockerfile var, bakılmıyor |
| [BloodAxe/xView2-Solution](https://github.com/BloodAxe/xView2-Solution) | 50 / 10 | 2021-11-25 | Üçüncü: topluluk | 0,803 genel, 0,805 özel | Bakılmıyor |
| [Z-Zheng/xview2_4th_solution](https://github.com/Z-Zheng/xview2_4th_solution) | 6 / 4 | 2026-08-12 | Dördüncü | README'de skor yok | Push yeni, içeriği incelenmedi |
| [ethanweber/xview2](https://github.com/ethanweber/xview2) | 81 / 25 | 2021-01-08 | Track 3 ikincisi, Detectron2 | Skor sıralaması README'de | Bakılmıyor |
| [michal2409/xView2](https://github.com/michal2409/xView2) | 62 / 8 | 2023-10-03 | Segmentasyon modelleri, Siyam varyantları | Taramada skor çıkmadı | Dockerfile var |
| [PaulBorneP/Xview2_Strong_Baseline](https://github.com/PaulBorneP/Xview2_Strong_Baseline) | 27 / 2 | 2023-08-08 | Birincilik modeli üzerinde deneyler | Taramada skor çıkmadı | Bakılmıyor |
| [DIUx-xView/xView2-deploy](https://github.com/DIUx-xView/xView2-deploy) | 9 / 4 | 2023-11-19 | Birincilik ağırlıklarıyla dağıtım | Yok | Docker var |
| [Z-Zheng/ChangeOS](https://github.com/Z-Zheng/ChangeOS) | 95 / 16 | 2021-12-21 | Nesne tabanlı semantik değişim | "Şehir ölçeği bir dakikada" (yazarın iddiası) | `pip install changeos`, PyTorch 1.10 |
| [ShaneShen/BDANet](https://github.com/ShaneShen/BDANet-Building-Damage-Assessment) | 41 / 6 | 2023-11-15 | Çapraz yönlü dikkat, birincilik çözümüne dayalı | README taramasında skor çıkmadı | Türkiye'ye [Xia et al. 2023](https://link.springer.com/article/10.1007/s13753-023-00526-6) tarafından yeniden eğitimsiz taşındı |
| [nka77/DAHiTra](https://github.com/nka77/DAHiTra) | 39 / 8 | 2023-06-14 | Hiyerarşik transformer | Taramada skor çıkmadı | Bakılmıyor |

**C. Kurumsal araçlar ve SAR'a odaklı işler**

| Repo | Yıldız / fork | Son push | Yöntem ve veri | Metrik | Durum |
|---|---|---|---|---|---|
| [google-research/skai](https://github.com/google-research/skai) | 168 / 24 | 2026-10-23 | Hava görüntüsünde otomatik hasar | README'de yok | Bakımda, Apache-2.0, 29 açık kayıt |
| [microsoft/building-damage-assessment](https://github.com/microsoft/building-damage-assessment) | 165 / 36 | 2026-08-11 | Olaya göre ince ayar, etiketleme aracı, DPM analiz akışı | README'de yok | Bakımda, MIT |
| [microsoft/building-damage-assessment-cnn-siamese](https://github.com/microsoft/building-damage-assessment-cnn-siamese) | 79 / 23 | 2026-08-14 | Siyam CNN, Docker ile çıkarım | Taramada sayı çıkmadı | Push yeni, içeriği incelenmedi |
| [prs-eth/xbd-s12](https://github.com/prs-eth/xbd-s12) | 30 / 2 | 2026-04-22 | [10.315 Sentinel-1/2 çifti; karmaşık mimariler görülmemiş olaya daha kötü genellemiş](https://arxiv.org/abs/2511.05461) | Sayılar özette yok | Bakımda, MIT, veri Zenodo'da |
| [oballinger/PWTT](https://github.com/oballinger/PWTT) | [70 / 6](https://github.com/oballinger/PWTT) | Görülmedi | Sentinel-1 piksel bazlı t-testi, Earth Engine, öğrenmesiz | [AUC 0,88 Ukrayna, 0,81 Gazze](https://arxiv.org/abs/2405.06323) | Açık kaynak, yıldız sayısı web sayfasından okundu |
| [deepankkumar/QuakeMetaFormer](https://github.com/deepankkumar/QuakeMetaFormer) | 0 / 0 | Görülmedi | [Görüntü + sismik/SAR üst verisi, Türkiye](https://arxiv.org/html/2412.04664v3) | Sonraki bölüme bakın | MetaFormer fork'u, PyTorch 1.5.1, ağırlık sürümü görünmüyor |
| [ghasemi1997/Building-Level-Earthquake-Damage-CNN](https://github.com/ghasemi1997/Building-Level-Earthquake-Damage-CNN) | 1 / 0 | 2025-12-23 | Bina başına 7 sayısal özellikle 1D CNN | README'de sayı yok | %50 doğrulama bölmesi, olay adı README'de yok |
| [sepal-contrib/damage_proxy_maps](https://github.com/sepal-contrib/damage_proxy_maps) | 2 / 4 | 2026-09-04 | DPM üretim aracı | Yok | Üst veri okundu |
| [salaheldinaz/pwtt-qgis-plugin](https://github.com/salaheldinaz/pwtt-qgis-plugin) | 0 / 0 | 2026-04-23 | PWTT için QGIS eklentisi | Yok | Üst veri okundu |

BRIGHT'a özgü olgular:

- Türkiye olayı [0,30–0,35 m, Maxar + Capella + Umbra, 1.114 parça, 135.033 bina](https://arxiv.org/html/2501.06019v2) ile veri setinin en büyük olayı. Toplam 4.246 çift ve 384.596 bina veriliyor, ancak olay bazlı sayıların toplamı yaklaşık 358.000 ediyor. Makaledeki bu tutarsızlığın kaynağı doğrulanmadı.
- [Sıfır örnekli kurulum bir olayı ayırıp 13 olayla eğitiyor](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/README_cross_event.md), `--test_event_name` ile seçiliyor.
- [BRIGHT hasarı uzmanların optik görüntüden görsel yorumuna dayanıyor, etiket gürültüsü ve küçük hizalama hataları kabul ediliyor, Türkiye eğitimde baskın olabilir](https://arxiv.org/html/2501.06019v2). Türkiye için vekil kayıt hatası yaklaşık 0,9 piksel.
- [Ukrayna, Myanmar ve Meksika optik verisi yeniden dağıtılamıyor](https://github.com/ChenHongruixuan/BRIGHT).
- [CVPRW 2026 yarışması: 157 katılımcı, 1.289 gönderim, finalde 46 takım, metrik `segm_AP`](https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/README.md).

### Inferences
- BRIGHT, DFC25 ve CVPRW26 mIoU değerleri çok sınıflıdır. CS437'nin ikili IoU'su, Enkaz'ın bina düzeyi makro F1'i ve Russo'nun bina düzeyi F1'i ile karşılaştırılamaz.
- BRIGHT'taki Zenodo ağırlıkları ve standart bölünme Türkiye'yi eğitimde içeriyor olabilir. "Görülmemiş" iddiası için bu ağırlıklardan başlanmamalı. Ağırlıkların hangi bölünmeye ait olduğu tek tek kontrol edilmeli.
- Maliyet açısından, BRIGHT tam görüntü segmentasyonu ve xView2 birincilik çözümü öğrenci donanımı için ağır. Bina kırpıntısı yaklaşımı [Colab T4'te yaklaşık 54 dakika](https://github.com/emrtunali/enkaz) sürüyor.
- xBD resmi bölünmesinin olay bazlı olmadığı benim ön bilgimdir, bu araştırmada birincil kaynakta doğrulanmadı.

### Gaps
- xView2 birincisinin özel test skoru ve resmi skor tablosu okunamadı. Yalnız üçüncünün 0,805 değeri README'den görüldü.
- DFC25 ve CVPRW26 kazananlarının sözde etiketleri hedef olay görüntülerinden üretip üretmediği (transdüktif eğitim) ayrıntılı okunmadı. Türkiye testine taşınırken önemli.
- GitHub API sınırı (saatte 60 istek) dolduğu için [torchgeo](https://github.com/microsoft/torchgeo) ve [pytorch-change-models](https://github.com/Z-Zheng/pytorch-change-models) değerleri alınamadı.
- BRIGHT makale tablo değerleri özet çıktıdan alındı, tablo gözle doğrulanmadı.

## Öğrenci ve portföy düzeyi projeler

### Takeaway
2023–2025 arası öğrenci projelerinin çoğu xBD üzerinde sınıflayıcı veya U-Net kuruyor ve tek bir ortalama skor veriyor. Ancak 2026'da seviye yükseldi: Enkaz (Ekim 2026) ve CS437 (Mayıs 2026), görevde tarif edilen zayıflıkların çoğunu zaten kapatıyor. Fark yaratmak için "şehir dışarıda, kalibre, belirsizlikli" olmak artık yetmiyor.

### Cited Findings

Doğrudan rakipler:

- [emrtunali/enkaz](https://github.com/emrtunali/enkaz) (0 yıldız, push 2026-10-22, 19 Ekim 2026'da açılmış, lisans görünmüyor). Bu görevin kancasına en yakın proje. 6 Şubat depremi, Maxar sonrası + BRIGHT öncesi ve maskeler, yani **ikisi de optik, SAR yok**.
  - Şehir bazlı bölünme: eğitim 5 şehir, doğrulama İslahiye, test Antakya (13.555 bina).
  - Öğrenmesiz taban, yalnız sonrası CNN, Siyam ve topluluk karşılaştırması. Antakya'da makro F1 0,490 (0,463–0,515), AUROC 0,736, en üst %10 içinde yıkılanların %55'i.
  - Kalibrasyon: doğrulama şehrinde ECE yaklaşık 0,01, Antakya'da 0,08–0,12.
  - Parça düzeyinde küme bootstrap, ikinci test şehri (Kahramanmaraş, AUROC 0,793), hata galerisi, etkileşimli sayfa, İstanbul için hazırlık sayfası.
  - Dürüst bulgu: öncesi görüntü bu kurulumda yardım etmemiş, eşikler şehirler arası taşınmıyor, sıralama taşınıyor.
- [m-usmanzahid/cs437-group40](https://github.com/m-usmanzahid/cs437-group40-multimodal-earthquake-damage-assessment) (1 yıldız, 2026-05-08). **BRIGHT optik + SAR füzyonu**, tek modlu tabanlar, MC Dropout, ECE, bir-olay-dışarıda değerlendirme.
  - Dışarıda tutulan olayda IoU: Türkiye 0,4953, Fas 0,1719, Haiti 0,3519.
  - Eksikleri kendi söylüyor: çok tohumlu istatistik yok, dış karşılaştırma vurgulanmıyor. Şehir kırılımı README'de görülmedi.
- [antoniopiogrossi/bright-dfc2025-reproduction](https://github.com/antoniopiogrossi/bright-dfc2025-reproduction) (0 yıldız, 2026-10-23). Tez, DFC25 birincisini yeniden üretmeyi hedefliyor. Henüz model eğitilmemiş, ama metrik kodunu sentetik tahminlerle doğrulamış.

Tipik örnekler (kolayda örneklem, yalnız README anahtar kelime taraması):

| Repo | Ne yapıyor | Zayıf nokta |
|---|---|---|
| [zak-510/xView2](https://github.com/zak-510/xView2) | Hasar sınıflayıcı | Manşet ağırlıklı F1: 82,7% |
| [Ali-Alii/Building-Damage-Detection](https://github.com/Ali-Alii/Building-Damage-Detection-from-Satellite-Imagery) | ResNet50, xBD | Doğruluk ve makro F1 var, olaylar arası test taramada çıkmadı |
| [Mr-Espada/building-damage-assessment-xbd](https://github.com/Mr-Espada/building-damage-assessment-xbd) | Lisans bitirme, U-Net | Doğrulama hold + test birleşimi, bağımsız test değil, kendisi belirtiyor. Eğitim geçmişi eksik |
| [mustafabukh/BDAM-XBD](https://github.com/mustafabukh/BDAM-XBD) | Rastgele Orman | Belirsizlik "gelecek iş" |
| [jamesrak/Building_Damage_Assessment_of_Turkey_Earthquake](https://github.com/jamesrak/Building_Damage_Assessment_of_Turkey_Earthquake) | Türkiye | README iki satır |
| [vxhll/eo-sar-change-detection](https://github.com/vxhll/eo-sar-change-detection) | EO-SAR U-Net | Doğrulama F1 0,1245, başka analiz taramada çıkmadı |
| [Pankajadas17/eosar-change-detection](https://github.com/Pankajadas17/eosar-change-detection) | EO-SAR, sahne bazlı test | Doğrulama F1 0,6030'tan test F1 0,0969'a düşüş, açıkça raporlanmış |
| [husnainalix77/disaster-damage-assessment](https://github.com/husnainalix77/disaster-damage-assessment) | Dizüstü ile xBD alt kümesi | Konum kimliğine göre sızıntısız bölme var, 3 afet tipi ile sınırlı, kapsam notu dürüst |
| [Pio-code/xbd-siamese-explainable](https://github.com/Pio-code/xbd-siamese-explainable-damage_assessment) | Siyam + LIME | Grup K-kat var, olaylar arası test taramada çıkmadı |
| [NSNNTL/ProtoNet-xBD](https://github.com/NSNNTL/ProtoNet-xBD) | Az örnekli öğrenme | F1 toplam 0,61 tam gözetimli, 0,55 beş örnekli |
| [MatteoLarrode/BuildingDamageViaSAR](https://github.com/MatteoLarrode/BuildingDamageViaSAR) | Oxford MSc, Gazze, Sentinel-1 | Deprem değil, savaş hasarı |

Bölünme şişirmesinin ölçeği için dış kanıt: [aynı veride çapraz doğrulama F1 0,886, şehir dışarıda F1 0,12–0,38](https://arxiv.org/html/2506.22338).

### Inferences
- Görevdeki dört zayıflık (rastgele bölme, taban yok, olaylar arası test yok, belirsizlik yok) xBD odaklı eski öğrenci işlerinde yaygın görünüyor. Bu bir sayım değil, yaklaşık 15 reponun taramasına dayanan izlenimdir. Anahtar kelime çıkmaması özelliğin yok olduğunu kanıtlamaz.
- En yaygın raporlama kusuru ağırlıklı F1. CS437'de aynı model için [ağırlıklı F1 0,9021, makro F1 0,4424](https://github.com/m-usmanzahid/cs437-group40-multimodal-earthquake-damage-assessment).
- Enkaz ve CS437'nin 0–1 yıldızlı olması görünürlüklerinin düşük olduğunu gösterir, kalitelerinin değil. Ciddi bir incelemeci bu repoları bulabilir. Portföyde bunlardan söz edip farkı açıkça göstermek daha güvenli. `[YARGI]`

### Gaps
- GitHub araması yıldız sıralı ve eksik. Türkçe adlı ya da README'si zayıf projeler kaçmış olabilir.
- Notebook içleri okunmadı. Bir README'de görünmeyen doğrulama notebook içinde olabilir.
- Enkaz sahibinin kim olduğu, kullanıcıyla ilişkisi araştırılmadı. Rakip olarak ele alındı.

## Kurumsal ve ticari sistemler

### Takeaway
Hızı ve kapsamı en iyi belgelenenler Copernicus EMS, ARIA ve Microsoft. Bağımsız saha doğrulamasına göre, 2023 olayında bina düzeyinde hiçbiri güçlü değil: radar DPM F1 0,47 ile en iyisi, optik ürünler F1 0,15–0,24. ICEYE, Capella, Umbra ve Maxar için Türkiye'ye özgü, metrikli bağımsız hasar doğrulaması bulamadım.

### Cited Findings

| Sistem | Ne teslim ediyor | Hız | Kendi doğrulaması | Bağımsız bulgu | Not |
|---|---|---|---|---|---|
| Microsoft AI for Good | [AFAD için 4 şehirde bina düzeyi GeoPackage; xBD ön eğitimli CNN, her alan için elle etiketle ince ayar; Planet 50 cm, Maxar 30 cm; 3.849 hasarlı bina](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/02/Turkey-Earthquake-Report-2_MS.pdf) | Görüntü 7–9 Şubat, rapor 16 Şubat 2023 | Raporda doğruluk metriği bulunmadı | [F1 0,24, toplamın 0,41 katı](https://www.nature.com/articles/s43247-025-02623-4) | Raporda çelişki: metin İslahiye'de "13.215 hasarlı" diyor, tabloda 13.215 toplam bina, 328 hasarlı |
| Google SKAI / WFP / UN | [Sıfır örnekli ve ince ayarlı modeller, uzman düzeltmesi, UNOSAT üzerinden dağıtım](https://www.unglobalpulse.org/ai-from-google-research-and-un-boosts-humanitarian-disaster-response-wider-coverage-faster-damage-assessments) | "Bir günün altında" yön gösterici bulgu | Bina düzeyinde 0,75 ve üzeri, büyük ve dengeli örnekle. Referans standardı belirtilmemiş | Maraş/Antep sayfada Suriye altında | Kurum ortağı anlatımı. [Clear SKAI vaka çalışmasındaki yüksek oran ve süre iddiaları](https://objectcomputing.com/client-outcomes/revolutionizing-disaster-response-with-ai) doğrulanmadı |
| Microsoft ve Google karşılaştırması | [Sağlık tesisleri üzerine bindirme: duyarlılık %71,4 ve %57,1, Kappa 0,38](https://pmc.ncbi.nlm.nih.gov/articles/PMC12558478) | Yok | Yok | [Duyarlılık düşük, ölçüler farklı](https://www.crisisready.io/publications/remote-sensing-models-2023-earthquake-turkey/) | PMC sayfası tarayıcı kontrolüne takıldı, sayılar arama özetinden |
| Copernicus EMS EMSR648 | [20 alan, elle derecelendirme, ilk aşamada 53 harita, ön sayımda 4.042 bina/blok](https://mapping.emergency.copernicus.eu/news/information-bulletin-165-the-copernicus-emergency-management-service-assesses-the-damage-caused-by-the-earthquake-in-t%C3%BCrkiye/) | İlk derecelendirme aktivasyondan 40 saat sonra | [JRC kalite kontrolü, sınıflar: yok, muhtemel, hasarlı, yıkık](https://elib.dlr.de/209366/1/s10518-024-01877-1.pdf) | [F1 0,15–0,23, hassasiyet 0,64–0,79, toplamın 0,10–0,21 katı](https://www.nature.com/articles/s43247-025-02623-4) | [37 derecelendirme ürünü](https://data.jrc.ec.europa.eu/dataset/a8a6fa40-853a-4a1f-826a-51510e28cf48) |
| NASA ARIA / EOS-RS DPM | [Sentinel-1'den 30 m harita, 10 Şubat görüntüsü](https://data.amerigeoss.org/it/dataset/aria-damage-proxy-map-copernicus-sentinel-1-for-the-turkiye-earthquakes); [ALOS-2'den 8 Şubat](https://maps.disasters.nasa.gov/ags04/rest/services/turkey_earthquake_2023/dpm/MapServer/info/iteminfo) | Görüntüye bağlı, aşağıda | Ön doğrulama: yüksek çözünürlüklü optik ve medya. Bitkili, dağlık, karlı alanda güvenilirlik düşer | [F1 0,47, duyarlılık 0,53–0,59, toplamın 1,5 katı; il r² 0,94, ilçe 0,81, mahalle 0,66](https://www.nature.com/articles/s43247-025-02623-4) | Hasar proxy'si, bina hasar sınıfı değil. Yayınlı piksel 30 m |
| UNOSAT | [6 Şubat'ta Uluslararası Şart aktivasyonu; Mart 2023 raporu Suriye odaklı](https://unosat.org/static/unosat_filesystem/3534/UNOSAT_Preliminary_Comprehensive_DA_Report_March2023_EQ20230206SYR.pdf) | Rapor Mart 2023 | Türkiye için metrik görülmedi | Yok | Türkiye bina haritası doğrulanamadı |
| Maxar | [Olay için açık veri](https://maxar.com/open-data/turkey-earthquake-2023), [CC BY-NC 4.0 (Enkaz README aktarımı)](https://github.com/emrtunali/enkaz) | [Aynı dönemde 11.000 km², alanın %5,4'ü](https://www.nature.com/articles/s43247-025-02623-4) | Hasar sayımı yayını bulunmadı | Görüntü sağlayıcı, hasar ürünü değerlendirilmedi | Görüntü, hasar haritası değil |
| ICEYE | [Pazarlama: "saatler içinde tüm olay çevresi"](https://www.iceye.com/newsroom/press-releases/iceye-expands-multi-peril-change-detection-to-include-earthquake-with-fire-following), başlangıçta Kaliforniya, sigorta odaklı | İddia | Metrik veya bağımsız doğrulama yok | Yok | **Pazarlama, doğrulanmış doğruluk değil.** [9–10 Şubat 2023 Türkiye strip görüntüleri veri sayfasında](https://www.iceye.com/resources/datasets) |
| Capella | [9 Şubat 2023 açık görüntü, İslahiye çalışmalarında](https://github.com/ya0-sun/PostEQ-SARopt-BuildingDamage) | Görüntü tarihi | Şirketin hasar ürünü görülmedi | [Akademik kullanım, Capella ile sınıflama](https://elib.dlr.de/209366/1/s10518-024-01877-1.pdf) | Hasar ürünü değil, veri |
| Umbra | [Açık veri programı 15 Mart 2023'te başladı](https://prnewswire.com/news-releases/umbra-releases-over-1-million-of-free-sar-data-301887643.html) | Yok | Türkiye hasar ürünü görülmedi | [BRIGHT Türkiye verisinde kullanılmış](https://arxiv.org/html/2501.06019v2) | Pazarlama dili, yalnız olgu olarak alındı |
| STM (Türkiye) | [SSB çatısı altında, uydu ve İHA görüntüleri, MAKS adres verisiyle yoğunluk haritası, arama kurtarmaya yönlendirme](https://www.dailysabah.com/turkiye/turkiye-quakes-ai-satellite-imagery-assess-building-damage/news/amp) | Belirtilmemiş | "Doğru hasar tespiti sağlandı", metrik yok | Yok | Basın anlatımı, doğrulanmamış |
| Hollanda Kızılhaçı | [Antakya otomatik değerlendirme](https://research.itu.edu.tr/tr/publications/complementarity-of-field-and-remote-sensing-based-damage-assessme/) | Şubat 2023 görüntüsü | Yok | Saha hasarının yalnız %59'unu bulmuş, Nisan görsel yorumu %24 daha fazla hasar buldu | Barınak kaybı odaklı |

Zamanlama ve kapsam:

- [İlk SAR sahnesi ilk depremden yaklaşık 56 saat sonra, tam kapsama 134 saat. Radar 204.000 km², kullanılabilir optik ürünler 664–800 km²](https://www.nature.com/articles/s43247-025-02623-4). Aynı grubun [özetinde "on gün içinde"](https://earthobservatory.sg/research/publications/earthquake-damage-mapped-more-comprehensively-and-accurately-by-radar-satellites-than-optical-imagery) deniyor, kapsam tanımı farklı olabilir.
- [Simülasyon çalışması, X bandı takımyıldızlarının ShakeMap yönlendirmesiyle saatler içinde harita üretebileceğini söylüyor](https://doaj.org/article/157d842882fa40ba87b131b7f630754a). Bu simülasyondur, 2023'te fiilen yaşanan gecikme değildir.

### Inferences
- Ainscoe et al. sonuçları güçlü ama tarafsızlığı sınırlı: yazar grubu radar DPM üreten ekiplerle iç içe ([EOS-RS, ALOS-2 DPM'lerini üretmiş](https://maps.disasters.nasa.gov/ags04/rest/services/turkey_earthquake_2023/dpm/MapServer/info/iteminfo)). Optik ürünler o günkü halleriyle kullanılmış, yeniden analiz edilmemiş. Karşılaştırmayı "radar her zaman optikten iyi" diye genellemek hatalı olur.
- Rapordaki oran ve hassasiyetten aritmetikle: elle derecelendirmede duyarlılık yaklaşık %8–14, Microsoft CNN için yaklaşık %17. Bu, hazır optik ürünlerin ağır hasarlı binaların çoğunu kaçırdığı anlamına gelir.
- Kurumsal sistemlerin çoğu görülmemiş olayda "sıfır örnekle" değil, olay başına elle etiket ve uzman düzeltmesiyle çalışıyor (Microsoft ince ayar, SKAI ince ayar + UNOSAT). Tamamen otomatik sıfır örnek, operasyonel uygulamanın kendisi değil.

### Gaps
- Capella, Umbra ve ICEYE'nin Türkiye'ye fiili teslim süreleri bulunamadı. Vitale ve Milillo tam metni okunmadı.
- UNOSAT'ın Türkiye bina ürünleri doğrulanamadı. Bir arama sonucundaki SDGSAT-1 gece ışığı değerlendirmesi UNOSAT'a atfedilmişti, ancak bağlantı [Çin Bilimler Akademisi](https://english.cas.cn/newsroom/news/202303/t20230307_327945.shtml) sayfası olduğundan kullanılmadı.
- Maxar'ın isim değişikliği ve güncel ticari ürünleri araştırılmadı.
- EMSR648 için resmi doğruluk matrisi bulunamadı.

## 2023 Kahramanmaraş/Türkiye depremi için yayınlar ve yer gerçeği

### Takeaway
Kanıt hiyerarşisi net: görülmemiş şehirde bina düzeyi F1 genellikle 0,12–0,48, kendi içi çapraz doğrulamada 0,886'ya kadar çıkıyor. En önemli bulgu ise etiket sorunu: Copernicus ve HOT gibi optik yorum etiketleri saha sayımının çok gerisinde. BRIGHT Türkiye etiketleri üzerindeki bir test, gerçek hasarla değil uzman optik yorumla uyumu ölçer.

### Cited Findings

| Çalışma | Görüntü | Yer gerçeği | Bölme | Sonuç |
|---|---|---|---|---|
| [Ainscoe et al. 2025, Commun. Earth Environ.](https://www.nature.com/articles/s43247-025-02623-4) | Sentinel-1, ALOS-2, öğrenme yok (koherans anomali) | CSB sahası: yaklaşık 8.000 personel, 2 milyon bina; 911.181 nokta, yaklaşık 327 bin ayak izi (228.859 hasarsız, 50.395 hasarlı) | Eşik 0,84 | F1 0,47. Yıkık ile ağır hasarı ayıramıyor. Kar ve sıvılaşma yanlış pozitif. Veri yeniden dağıtılamıyor |
| [Singh, Hoskere, Milillo 2025 (QuakeMetaFormer)](https://arxiv.org/html/2412.04664v3) | Maxar olay sonrası optik + DPM + sismik, zemin | CSB, Gece Yazılım ve Yer Çizenler işlemiş; yaklaşık 95.800 bina (Türkiye-L), 15.600 (Türkiye-S) | %85/15 ve bölgeler arası | F1 0,49, AUC 0,82. Bölgeler arası AUC 0,783, F1 0,430. SAR maskelenince AUC yalnız 0,81'den 0,80'e düştü |
| [Russo et al. 2025 (arXiv)](https://arxiv.org/html/2506.22338) | COSMO-SkyMed + DSM + GEM, yalnız olay sonrası | HOT/OSM yıkık katmanı, 13.487 bina | 5 kat ve şehir dışarıda | Çapraz doğrulama F1 0,886, şehir dışarıda 0,12–0,38, AUROC 0,55–0,85. Kod kabul sonrası denmiş |
| [Macchiarulo et al. 2024, Bull. Earthq. Eng.](https://elib.dlr.de/209366/1/s10518-024-01877-1.pdf) | Capella, olay sonrası, doku özellikleri, Rastgele Orman | HOT (İslahiye), Copernicus (Kahramanmaraş), saha (Haiti) | Vaka içi ve yeni ortam | Vaka içi genel doğruluk %73–86. Yeni ortamda ortalama %63, Kappa 0,26, duyarlılık %6–88 |
| [Yu et al. 2024, npj Nat. Hazards](https://www.nature.com/articles/s44304-024-00003-0) | Sentinel-1, ALOS-2, Sentinel-2, PGA; 30 m piksel | Bakanlık hasar haritası, yanlılık olabileceği belirtilmiş | Rastgele %80/20 | ROC-AUC yaklaşık 0,69–0,70 |
| [Xia et al. 2023, Int. J. Disaster Risk Sci.](https://link.springer.com/article/10.1007/s13753-023-00526-6) | WorldView-2, BDANet | Hasar için yok, üç bölgede görsel | xBD eğitimli, yeniden eğitim yok | Hasar doğruluğu İslahiye için bildirilmemiş |
| [Wiguna et al. 2024, JSTARS (Taşkın derlemesi üzerinden)](https://elib.dlr.de/220803/1/Post-Earthquake_Damage_Mapping_via_Remote_Sensing_Lessons_from_the_2023_Trkiye_Disaster.pdf) | Swin + CE-Focal, xBD eğitimli | Türkiye ve Tohoku görülmemiş test | Görülmemiş olay | Yok ve yıkık iyi, ara sınıflar karışıyor |
| [BRIGHT, Chen et al. 2025, ESSD](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.pdf) | Öncesi optik, sonrası SAR | Uzman optik yorum; Copernicus, UNOSAT, FEMA | Türkiye dışarıda dahil | ChangeMamba Türkiye sıfır örnekli mIoU 41,28 |
| [Yang et al. 2024, Earthq. Spectra](https://journals.sagepub.com/doi/abs/10.1177/87552930241262761) | Sentinel-1 koherans zaman serisi, RNN | Metrik arama özetinden | Okunmadı | Kahramanmaraş hasar oranı %24,08, saha ile uyumlu denmiş |

Türk grupları ve Türkiye verisi:

- [ITÜ: KATE-PD veri seti, 3.293 km², olay sonrası tek görüntü, Musaoğlu, Taşkın ve diğerleri, Payten ile, IGARSS 2025](https://research.itu.edu.tr/en/publications/a-rapid-damage-assessment-using-remote-sensing-turkiye-2023-post-/). Sayfada metrik yok.
- [Taşkın, Musaoğlu, Erten, Hänsch, Yang: JSTARS özel sayı kapanış derlemesi](https://elib.dlr.de/220803/1/Post-Earthquake_Damage_Mapping_via_Remote_Sensing_Lessons_from_the_2023_Trkiye_Disaster.pdf). "En kritik zorluk görülmemiş bölge ve afetlere zayıf genelleme" sonucuna varıyor, ayrıca etiket gürültüsü ve ara sınıf belirsizliğini sıralıyor. ITÜ-CSCRS açık verisinin kritik rol oynadığını belirtiyor.
- [Ekkazan ve Karsligil: TE23D, 1.183 yama, 2.080 hasar poligonu, ikili segmentasyon](https://elib.dlr.de/220803/1/Post-Earthquake_Damage_Mapping_via_Remote_Sensing_Lessons_from_the_2023_Trkiye_Disaster.pdf). Kurum bağlantıları doğrulanmadı.
- [Buznego Puerto, Kundak, Göksu ve diğerleri, Antakya saha karşılaştırması](https://research.itu.edu.tr/tr/publications/complementarity-of-field-and-remote-sensing-based-damage-assessme/).
- [EEFIT, Voelker ve diğerleri: uzaktan algılama ile saha noktası seçimi ve saha doğrulaması](https://repository.tudelft.nl/record/uuid:c16e9079-2f1b-4394-a5f3-f396e422b277).

Yer gerçeği kaynakları:

| Kaynak | Nitelik | Erişim |
|---|---|---|
| CSB saha sayımı | En güçlü, ama eksik ve hatalı kayıtlar içeriyor, [nokta-ayak izi eşlemesi hata katıyor](https://www.nature.com/articles/s43247-025-02623-4) | Ainscoe yeniden dağıtamıyor. [Singh verisi Foursquare'de, parola yazardan alınıyor](https://arxiv.org/html/2412.04664v3). [Yu: hasar.6subatdepremi.org](https://www.nature.com/articles/s44304-024-00003-0), buradan erişilemedi |
| Copernicus EMS | Optik uzman, erken ürünler eksik | [Açık](https://mapping.emergency.copernicus.eu/activations/EMSR648/) |
| HOT/OSM | Gönüllü, [dört adımlı doğrulama](https://elib.dlr.de/209366/1/s10518-024-01877-1.pdf), [yanlış negatif riski](https://arxiv.org/html/2506.22338) | [Açık](https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings) |
| BRIGHT Türkiye | [Bina poligonu uzman, hasar uzman optik yorum](https://arxiv.org/html/2501.06019v2); [Enkaz Copernicus kaynaklı diyor](https://github.com/emrtunali/enkaz) | Açık, makalede Türkiye için olay bazlı kaynak belirtilmedi |

Bağlam: Bakan [232.000'den fazla binanın ağır hasarlı veya yıkılacak olduğunu](https://en.yenisafak.com/turkiye/turkiye-says-over-232000-buildings-damaged-or-to-be-demolished-in-quake-hit-region-3661736) açıklamış. Bu sayı Microsoft'un 4 şehirdeki 3.849 hasarlı bina tahminiyle ölçek olarak karşılaştırılamaz, kapsam farklı.

### Inferences
- Bir model BRIGHT Türkiye etiketlerine göre "yanlış pozitif" görünen bir binada gerçekte haklı olabilir. Hedef SAR ise, optik yorum etiketleri modeli sistematik olarak cezalandırabilir. Bu, projenin hem riski hem en ilginç deneyi.
- Singh'de SAR'ın katkısının küçük çıkması çözünürlükten kaynaklanabilir (yazarlar da öyle düşünüyor). VHR SAR ile aynı soruyu sormak hâlâ anlamlı.
- Enkaz'ın öncesi görüntü bulgusu ile Ainscoe'nun radar bulgusu birlikte şu açık soruyu doğuruyor: aynı binalarda, aynı referansla, sonrası optik mi sonrası SAR mı, öncesi optik ne ekliyor?
- Türkiye dışarıda bırakılınca BRIGHT eğitim verisinin yaklaşık üçte biri gider (135.033 / 384.596). Sıfır örnekli sonuçların düşüşü kısmen bundan da kaynaklanabilir.

### Gaps
- [ASCE derin topluluk çalışması](https://ascelibrary.org/doi/10.1061/AOMJAH.AOENG-0043) 403 döndü, okunamadı.
- [Sensors 2023 SAR değişim çalışması](https://pmc.ncbi.nlm.nih.gov/articles/PMC10385665/) ve [ITÜ sahne sınıflama çalışması](https://research.itu.edu.tr/tr/publications/deep-learning-based-scene-classification-of-very-high-resolution-/) tam metin okunmadı. [Kahramanmaraş test sahalarında %75, %84,4, %73,8 genel doğruluk](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/s11069-023-06154-z.pdf) yalnız arama özetinden görüldü, bir ayna siteden geliyor.
- ODTÜ, AFAD ve TÜBİTAK kaynaklı ML hasar makalesi için doğrudan metrikli kayıt bulamadım. [ODTÜ'de saha/dijital araç çalışması var](https://avesis.metu.edu.tr/yayin/c30cd112-f892-462f-8809-732780bc24ef/utilizing-digital-technologies-for-rapid-damage-assessment-and-reconnaissance-the-february-6-2023-kahramanmaras-turkiye-earthquakes-mw-7-7-and-mw-7-6), okunmadı.
- hasar.6subatdepremi.org bu ortamdan açılmadı, resmi olup olmadığı ve lisansı doğrulanmadı. [Zenodo kaydı](https://doi.org/10.5281/zenodo.10995299) API'den okunamadı.
- Ainscoe veri erişim beyanı tam metinde bulunamadı. "Yeniden dağıtılamaz" bilgisi makalenin özetlenmiş içeriğinden geliyor.

## Dürüst boşluk, haksız iddialar ve işe alım yöneticisini ikna edecek şeyler

### Takeaway
`[YARGI]` Boşluk "görülmemiş depremde test" değil, çünkü bu yapıldı. Kalan boşluk: öncesi optik + VHR sonrası SAR modelinin, aynı binalarda optik yorum etiketi ve saha sayımı ile ayrı ayrı, öğrenmesiz radar tabanına (DPM) ve sonrası optik modele karşı, kilitli protokol ve güven aralıklarıyla sınanması. Bunu yapan açık bir repo bulamadım, ancak bu "kesinlikle yok" değil, "bu aramada bulunamadı" demektir.

### Cited Findings

Zaten yapılmış olanlar (iddia edilemeyecek özgünlük):

- Türkiye sıfır örnekli optik + SAR: [BRIGHT makalesi](https://arxiv.org/html/2501.06019v2) ve [CS437](https://github.com/m-usmanzahid/cs437-group40-multimodal-earthquake-damage-assessment).
- Şehir dışarıda, kalibre, sıralama metrikli, bootstrap'li Türkiye: [Enkaz](https://github.com/emrtunali/enkaz), ama optik.
- Saha doğrulamalı radar-optik karşılaştırması: [Ainscoe et al.](https://www.nature.com/articles/s43247-025-02623-4), ama derin öğrenme yok ve optik ürünler hazır haliyle.
- CSB etiketli görüntü + üst veri modeli: [Singh et al.](https://arxiv.org/html/2412.04664v3), ama VHR SAR yok, öncesi görüntü yok.
- Topluluk önceliği: [Taşkın et al.](https://elib.dlr.de/220803/1/Post-Earthquake_Damage_Mapping_via_Remote_Sensing_Lessons_from_the_2023_Trkiye_Disaster.pdf) genelleme ve daha güvenilir yer gerçeğini ana açık konu sayıyor, yani hedef güncel.

Sektör sinyalleri (zayıf kanıt):

- [TÜBİTAK UZAY, uzay ve uydu projeleri için 25 yarı zamanlı aday araştırmacı aradı, son başvuru 20 Temmuz 2026](https://uzay.tubitak.gov.tr/en/?p=26832).
- [Türkiye'de bir görüntü işleme ilanı en az 3 yıl Ar-Ge, C++ ve Python istiyor](https://web.kariyer.net/is-ilani/cutesafe-teknoloji-anonim-sirketi-cutesafe-tech-computer-vision-engineer-2250283) (arama özetinden, ilan metni okunmadı).
- [Yurt dışı bir SAR ilanı interferometri ve polarimetri bilgisi istiyor](https://djinni.co/jobs/808909-sar-processing-engineer-ai-raw-sar-level-0) (arama özetinden).
- Aselsan, Havelsan, TUSAŞ, Baykar veya TÜBİTAK UZAY'ın SAR + derin öğrenme ölçütlerini sayan güncel bir ilan bulamadım.

### Inferences

**Önerilen fark (`[YARGI]`), önem sırasıyla:**

1. **İki referansla değerlendirme.** Aynı model hem BRIGHT/Copernicus etiketine hem erişilebilirse CSB verisine karşı. Anlaşmazlık tablosu ve elle denetlenmiş 100–200 binalık örnek. Erişim yoksa bunu açıkça yazmak, HOT/OSM ile ikinci referans kullanmak.
2. **Aynı bina kümesinde modalite karşılaştırması.** Öğrenmesiz değişim veya ARIA DPM, yalnız sonrası SAR, öncesi optik + sonrası SAR, sonrası optik (Enkaz tarifi). Sonuç füzyon lehine çıkmasa da yayımlanmalı.
3. **Kilitli sıfır örnek protokolü.** Türkiye'ye dokunmadan önce commit hash, eşik ve model seçimi doğrulama olayında sabitlenir. Hiçbir ön eğitimli ağırlık Türkiye görmemiş olmalı. Çok tohum, parça düzeyinde küme bootstrap.
4. **Karar metriği.** Üst %10'da yakalanan yıkık oranı, şehir ve sensör kırılımı, makro F1, sınıf bazlı IoU, ECE. Genel doğruluk veya ağırlıklı F1 manşet olmaz.
5. **SAR okuryazarlığı.** Kayma (layover), gölge, leke gürültüsü, arazi düzeltmesi, bakış geometrisi ile hata analizi. Kayıt hatası piksel olarak ölçülür.
6. **Maliyet ve gecikme.** Bina başına çıkarım süresi, GPU belleği, ONNX veya benzeri dışa aktarma, gerçek görüntü tarihleriyle "ilk ne zaman üretilebilirdi" tablosu.
7. **Tek komutla yeniden üretim.** Sabit ortam, ağırlıklar, model kartı, veri ve lisans tablosu, kısa bir etkileşimli sayfa.
8. **Olumsuz bulgular.** Eşiğin şehirler arası taşınmaması gibi sonuçlar ayrı başlıkla.

**Haksız olacak iddialar:**

| İddia | Neden haksız |
|---|---|
| "Yeni depremlere genelleniyor" | Tek olay, aynı sensör ailesi. [Fas IoU 0,17, Türkiye 0,50](https://github.com/m-usmanzahid/cs437-group40-multimodal-earthquake-damage-assessment) farkı olaya bağımlılığı gösteriyor |
| "Görülmemiş veride test edildi" (kanıtsız) | Ön eğitimli ağırlık, hiperparametre veya eşik Türkiye'ye dokunduysa geçersiz. Capella ve Umbra eğitimde de varsa sensör görülmemiş sayılmaz |
| "Saha doğrulamalı" veya "gerçek yer gerçeği" | BRIGHT etiketi [uzman optik yorumu](https://arxiv.org/html/2501.06019v2) |
| "Copernicus, Microsoft veya Google'dan iyi" | Etiket, kapsam, zaman ve insan döngüsü farklı. Karşılaştırma ancak aynı binalarda ve aynı referansta yapılırsa ve yalnız o kapsam için söylenebilir |
| "%9x doğruluk" | [Binaların yaklaşık %4'ü hasarlı olduğunda (Enkaz verisi)](https://github.com/emrtunali/enkaz) anlamsız |
| "Tüm hava koşullarında, gerçek zamanlı, operasyonel" | [İlk SAR yaklaşık 56 saat, tam kapsama 134 saat](https://www.nature.com/articles/s43247-025-02623-4), VHR kapsama daha dar, sahada deneme yok |
| "Yeni model" veya "ilk" | BRIGHT, DFC25, CVPRW26, DisasterEO mevcut |
| "Belirsizlik kalibre" | Yeni şehirde ölçülmeden söylenemez. [Enkaz'da ECE yaklaşık 0,01'den 0,08–0,12'ye çıktı](https://github.com/emrtunali/enkaz) |
| "Sağlam bina güvenlidir" | [Uydular çatı görür, pancake göçme sağlam görünebilir](https://github.com/emrtunali/enkaz) |
| Ağır/orta hasar ayrımı güvenilir | [Ara sınıflar literatürde karışıyor](https://elib.dlr.de/220803/1/Post-Earthquake_Damage_Mapping_via_Remote_Sensing_Lessons_from_the_2023_Trkiye_Disaster.pdf), [Enkaz F1 "hasarlı" en fazla 0,24](https://github.com/emrtunali/enkaz) |
| Ticari kullanıma hazır | [Maxar CC BY-NC 4.0, ağırlıklar kısıtı devralıyor (Enkaz yorumu)](https://github.com/emrtunali/enkaz); hukuki doğrulama yapılmadı |

**Ankara savunma ve uzay sektörü için (`[YARGI]`, kaynak yok):**

- Yönetici büyük olasılıkla hasar haritasının kendisini değil, belirsiz veride mühendislik disiplinini arar: sızıntı önleme, kayıt hatası ölçümü, hata analizi.
- SAR görüntüsünü RGB gibi kullanmak en hızlı elenme nedeni olabilir. Fiziksel neden gösterebilen birkaç hata örneği değerlidir.
- Sistem yönü etkili olabilir: Python yanında dışa aktarılmış model, C++ çıkarım veya gecikme ölçümü. Bu, [C++ isteyen ilanlarla](https://web.kariyer.net/is-ilani/cutesafe-teknoloji-anonim-sirketi-cutesafe-tech-computer-vision-engineer-2250283) uyumlu, ama savunma firmaları için doğrulanmadı.
- Sözde askeri uygulama anlatımı veya şirket adı anmak yerine sade bir sınırlar bölümü daha güven verir.
- Kısa teknik not, 2 dakikada okunacak README ve dürüst sonuç tablosu; akademik makale biçimi gerekmez.
- Düşük skorlar sorun değil, bağlama oturtulmaları gerekir. Referans: [Russo LOCO F1 0,12–0,38](https://arxiv.org/html/2506.22338), [Singh bölgeler arası F1 0,43](https://arxiv.org/html/2412.04664v3), [ARIA DPM F1 0,47](https://www.nature.com/articles/s43247-025-02623-4), [ChangeMamba Türkiye sıfır örnekli mIoU 41,28](https://arxiv.org/html/2501.06019v2). Bunlar farklı görevler, yan yana bağlam olarak verilebilir, sıralama olarak değil.

### Gaps
- Ankara'daki firmaların işe alım ölçütleri için birincil kaynak yok. Bölümdeki tavsiyeler kişisel yargıdır, bir yönetici görüşüyle sınanmadı.
- CSB verisinin portföy projesinde kullanım izni bilinmiyor. Kullanılabilse bile yalnız toplu metrik yayımlanabilir olabilir, yazarlara ve kuruma sorulmalı.
- BRIGHT Türkiye parçalarının CSB noktalarıyla ne kadar örtüştüğü ve hangi tarihte SAR çekildiği (molozun kaldırılması etkisi) denenmedi. Fizibilite için ilk küçük deney bu olmalı.
- "Hiçbir açık repo bu kombinasyonu yapmıyor" sonucu sınırlı aramaya dayanıyor. GitHub kod araması (kimlik doğrulama gerektiriyor), Hugging Face ve Papers with Code taranmadı.
