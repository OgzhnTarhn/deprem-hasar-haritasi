# Kilitli tek koşuyla 41,28 çıtasını zorlamak

Bu proje için en savunulabilir seçim şu: BRIGHT'ın Türkiye'yi tümüyle dışarıda bırakan resmi sıfır-örnek ayrımında, iki ayrı kodlayıcılı ve bina ile hasar başlıkları ayrık **ChangeMamba-MMBDA Tiny** modelini **tek, önceden kilitlenmiş, 40 bin adımlık bir koşuyla** eğitmek. Aynı protokolde yayımlanmış en iyi değer hâlâ onun **41,28 `[ZS-EVENT mIoU]`** skoru. İkinci sıradaki ChangeOS 38,45'te kalıyor ve bu skoru geçen bir yayın bulunamadı. Tek koşu, A100 40 GB üzerinde tahminen **9–14 saat ve 50–75 compute unit (CU)** yakar. Ortam, veri, hazır checkpoint'lerle protokol doğrulaması ve son değerlendirme eklenince toplam **yaklaşık 65–100 CU** eder. Bu, Colab Pro'nun aylık 100 CU'suna ancak sığar ve 12 saatlik oturum sınırı yüzünden en az bir devam (resume) gerektirir. Mamba çekirdeğinin Colab'ın CUDA 13 imajında derlendiği doğrulanmadı. Bu yüzden reçete bir saatle sınırlı bir derleme kapısıyla başlar. Kapı geçmezse yedek, BRIGHT deposunda hazır duran Swin-T DamageFormer'dır; bu modelin aynı protokoldeki skoru 37,20. Rakip analizinin asıl uyarısı da şu: "Türkiye'yi eğitimde görmedik" iddiasını BRIGHT makalesi ve 2026'daki CS437 öğrenci projesi zaten kullandı. Fark yaratacak işlerin çoğu ek eğitim istemiyor: kilitli protokol, yayımlanmış Türkiye checkpoint'lerinin yeniden üretimi, sınıf bazlı ve bina düzeyi metrikler, ikinci bir referans etiket (HOT/OSM) ile öğrenmesiz radar tabanına (ARIA DPM) karşı aynı binalarda kıyas, sensör ve geometri hata analizi. Gerçekçi beklenti 41,28 civarıdır, onu belirgin biçimde aşmak değil. Yayımlanmış tablo tek koşudan geliyor, 1–2 puanlık farklar gürültüden ayrılamıyor ve 41,28 kabaca bina sınıflarında ortalama 23 IoU anlamına geliyor. Yarışma kazandıran pseudo-label, sahne sınıflandırıcı ve topluluk gibi adımların sorunu da açık: Türkiye görüntüsünü kullandıklarında 41,28 ile kıyaslanamıyorlar, kullanmadıklarında ise 3–10 tam koşu istiyorlar.

Raporda iki tür cümle ayrı tutuldu. **Kanıt** kaynaklı bir sayı ya da kod okumasıdır. **Yargı** benim çıkarımım ya da önerimdir. Süre ve CU değerlerinin hiçbiri Colab'da ölçülmedi; hepsi tahmindir ve ilk iş olarak ölçülmeleri gerekir.

## Önce etiket: 41,28 yalnız aynı protokoldeki sayılarla kıyaslanır

**Kanıt.** BRIGHT'ın sıfır-örnek kurgusu hedef olayı eğitim, doğrulama ve test listelerinden çıkarır ve o olayın bütün karelerini teste koyar. Model kalan 13 olayla eğitilir. Model seçimi ise bu 13 olayın alan içi doğrulama kümesinde yapılır ([train_SiamCRNN.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/script/cross_event/zero_shot/train_SiamCRNN.py)). Türkiye olayı **1.114 kare ve 135.033 bina** ile veri setinin en büyüğü; görüntüler Maxar optik ile Capella ve Umbra SAR'dan geliyor, GSD 0,30–0,35 m, tahmini kayıt hatası yaklaşık 0,9 piksel ([BRIGHT arXiv v4](https://arxiv.org/html/2501.06019v4)). Hugging Face kopyası üzerinde araştırma sırasında yapılan sayıma göre Türkiye tek başına **bina piksellerinin yaklaşık %52'sini** taşıyor. Bina içi dağılımı %93,0 sağlam, %4,1 hasarlı, %2,9 yıkılmış. Hasarlı piksellerin yaklaşık %63'ü ise Ukrayna ve Meksika olaylarında ([Kullervo/BRIGHT](https://huggingface.co/datasets/Kullervo/BRIGHT)). Türkiye dışarıda kalınca model hem verinin büyük bölümünü kaybediyor hem de hasarı bol olaylardan öğrenip sağlamı baskın bir olayda test ediliyor. **Yargı:** Bu sınıf önceliği kayması, Türkiye'de eşik veya sınıf ağırlığı ayarlamanın neden hem tehlikeli hem de cazip olduğunu açıklıyor.

Aşağıdaki tablo, notlarda geçen her sayıyı türüyle birlikte veriyor. Yalnız ilk satır 41,28 ile aynı ölçektedir.

| Etiket | Ne ölçer | Değer | 41,28 ile kıyaslanır mı |
|---|---|---|---|
| `[ZS-EVENT mIoU]` | 13 olayla eğitim, Türkiye'nin bütün karelerinde 4 sınıf mIoU | ChangeMamba **41,28**, ChangeOS 38,45, SiamCRNN 38,32, UNet 37,82, DeepLabV3+ 37,28, DamageFormer 37,20, SiamAttnUNet 36,09 ([ESSD](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.pdf)) | Evet, tek doğrudan çıpa |
| `[OS-EVENT mIoU]` | Türkiye'den 1 etiketli eğitim ve 1 doğrulama çifti eklenir | ChangeMamba 42,80 ([arXiv v4](https://arxiv.org/html/2501.06019v4)) | Hayır |
| `[STD-EVENT mIoU]` | Rastgele ayrım, Türkiye eğitimde, olay bazlı | ChangeMamba 52,70 | Hayır |
| `[STD-SET mIoU]` | Rastgele ayrım, test kümesinin tamamı | ChangeMamba 67,63, DamageFormer 67,09 | Hayır |
| `[DFC-TEST mIoU]` | Türkiye dahil 9 olayla eğitim, test Noto ve Marshall | 54,28 ve 50,81 ([Li ve ark.](https://arxiv.org/abs/2505.04941), [GRSS](https://www.grss-ieee.org/community/technical-committees/winners-of-the-2025-ieee-grss-data-fusion-contest-all-weather-land-cover-and-building-damage-mapping/)) | Hayır |
| `[CVPR-TEST segm_AP]` | 3 sınıf örnek segmentasyonu, 2025 yangın ve kasırga | 0,182 ve 0,181 ([sonuç makalesi](https://arxiv.org/abs/2607.22746)) | Hayır, farklı görev |
| İkili IoU | CS437, Türkiye dışarıda | 0,4953 ([repo](https://github.com/m-usmanzahid/cs437-group40-multimodal-earthquake-damage-assessment)) | Hayır |
| Bina düzeyi makro F1 | Enkaz, Antakya, optik + optik | 0,490 ([repo](https://github.com/emrtunali/enkaz)) | Hayır |
| Saha sayımına karşı bina F1 | ARIA DPM, öğrenmesiz radar, CSB sahası | 0,47 ([Ainscoe ve ark.](https://www.nature.com/articles/s43247-025-02623-4)) | Hayır, referans farklı |

**Kanıt.** DFC25 birincisinin 54,28'lik test skoru arka planda 97,995, sağlamda 67,304, hasarlıda **9,748**, yıkılmışta 42,086 IoU'dan oluşuyor ([Li ve ark.](https://arxiv.org/abs/2505.04941)). Üç bina sınıfının ortalaması yalnız 39,71. **Yargı:** Arka plan IoU'su 96–97 kabul edilirse 41,28, bina sınıflarında kabaca ortalama 23 IoU demektir. Tablo 10 sınıf kırılımı vermediği için bu kaba bir tahmin, ama beklentiyi doğru ayarlar. Tablo 10'da tohum veya standart sapma da yok. Notlardaki kendi hesabıma göre dört deprem olayının ortalaması ChangeMamba'da 41,35, ChangeOS'ta 39,59, DamageFormer'da 38,97. Türkiye'de modeller arasındaki 1–3 puanlık farklar bu yüzden tek koşu gürültüsü olabilir. Projenin kendi sonucu için "41,28'i geçtik" demek ancak aynı 13 olayla, aynı listelerle, aynı 4 sınıflı mIoU ile ve Türkiye görüntüsüne hiç dokunmadan mümkündür. Fark 1–2 puanın altındaysa bunu "eşit" diye raporlamak gerekir.

## ChangeMamba-Tiny taban, Swin-T yedek: kanıtın ve riskin gösterdiği yer

**Kanıt.** Dört deprem olayının dördünde de en yüksek `[ZS-EVENT mIoU]` değerini ChangeMamba alıyor: Les Cayes 41,35, Türkiye 41,28, Fas 42,86, Noto 39,90 ([Tablo 10](https://arxiv.org/html/2501.06019v4)). Sıfır-örnek ortalamasında bina konumunu ve hasarı ayıran ("ayrık") modeller 37,48–39,38 alırken doğrudan segmentasyon modelleri 34,13–36,42'de kalıyor. Makale bunu "ayırma genellemeyi iyileştiriyor" diye yorumluyor ([arXiv v4](https://arxiv.org/html/2501.06019v4)). Ancak bu kontrollü bir deney değil. Ayrık modeller aynı zamanda çift akışlı ve kodlayıcı aileleri farklı. Türkiye'de ayrık DamageFormer (37,20), doğrudan UNet'in (37,82) altında kalıyor. Kod düzeyinde ChangeMamba'nın BRIGHT sürümü iki ayrı VSSM kodlayıcı kullanıyor. Bina dekoderi yalnız olay öncesi optiği, hasar dekoderi ise optik ile SAR özelliklerini birleştirerek görüyor. Füzyon her evrede basit birleştirme (`cat`); kayıp `CE_loc + CE_clf + 0,5·Lovász_loc + 0,75·Lovász_clf` ([ChangeMambaMMBDA.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/models/ChangeMambaMMBDA.py), [tasks/bright.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/tasks/bright.py)). Önemli bir ayrıntı daha var: eğitim döngüsündeki mIoU, 4 sınıflı hasar başlığının `argmax` çıktısından hesaplanıyor ve bina maskesi uygulanmıyor. Makale ise nihai haritanın maskeyle birleştirildiğini yazıyor. Bu yüzden başlık metriğinin hangi yoldan hesaplanacağı önceden kilitlenmeli.

**Kanıt.** Füzyon konusunda iki şampiyonun sonuçları çelişiyor. CVPRW 2026'da birinci takımın test ablasyonunda erken füzyon 0,0542, çift aşamalı füzyon **0,1095** `[CVPR-TEST segm_AP]` aldı. İkinci takımda erken füzyon 0,120, geç füzyonlu çift omurga **0,264** `[TAKIM-VAL segm_AP]` aldı. Yalnız SAR ile konumlandırma ise 0,002'de kaldı ([sonuç makalesi](https://arxiv.org/abs/2607.22746)). Buna karşılık DFC25'in iki birincisi de semantik görevde optik ve SAR için **paylaşımlı** kodlayıcıyla kazandı ([UABCD_v2.py](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping/blob/main/ChangeDetection/network/UABCD_v2.py), [STMambaBDA_DFv4.py](https://github.com/zxqyiyang/DF2025/blob/master/STMambaBDA_DFv4.py)). **Yargı:** ChangeMamba zaten optik çapalı, ayrı akışlı ve geç birleştirmeli bir yapıda. Bu konuda kanıtın yön gösterdiği her şeyi taşıyor. Çapraz dikkat, Mask2Former veya optik akış hizalaması eklemenin görülmemiş olayda yararı gösterilmedi. FlowMamba xBD'de ChangeMamba'yı yalnız 0,39 F1 geçiyor ama %48 daha fazla GFLOPs istiyor ([FlowMamba](https://github.com/flying318/FlowMamba)), Türkiye'nin kayıt hatası da zaten 1 pikselin altında.

**Kanıt.** Önceden eğitilmiş kodlayıcı tarafında doğrudan kanıt yok. DINOv3 ViT-L SAT-493M, 0,6 m Maxar görüntüsüyle eğitildi ve GEO-Bench-2'de alt-10 m görevlerde önde ([model kartı](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m), [GEO-Bench-2](https://arxiv.org/html/2511.15658v2)). Ama hiçbir görülmemiş olay hasar görevinde test edilmedi. Lisansı da askeri ve savaş amaçlı kullanımı yasaklıyor ([DINOv3 lisansı](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)). Ankara savunma sektörünü hedefleyen bir portföy için bu ciddi bir sürtünme. SAR tarafında Umbra ve Capella ile eğitildiği doğrulanan MaRS'ın lisansı üç kaynakta üç farklı. Verisi 30 Aralık 2024'e kadar uzandığı için 2023 Türkiye sahnelerini görmüş olabilir ([MaRS](https://rsidea.whu.edu.cn/mars.htm), [Zenodo](https://zenodo.org/records/17800805)). Bu, "görülmemiş" iddiasını sessizce bozar. İki CVPRW şampiyonu da ImageNet tarzı ön eğitim kullandı ([sonuç makalesi](https://arxiv.org/abs/2607.22746)). Alan genellemesi hileleri de umut vermiyor. BRIGHT'ta Türkiye için DeepLabV3+ yalnız kaynak verisiyle 37,28 alırken AdaptSeg 33,69, AdvEnt 34,54, CLAN 36,68, PyCDA 34,58, FDA 36,82 aldı; yani **beş UDA yönteminin hepsi tabanın altında** ([Tablo 16](https://arxiv.org/html/2501.06019v4)). DomainBed de iyi ayarlanmış ERM'yi hiçbir yöntemin bir puandan fazla geçmediğini gösteriyor ([Gulrajani ve Lopez-Paz](https://arxiv.org/pdf/2007.01434)). **Yargı:** Kodlayıcı Zenodo'daki ImageNet `vssm_tiny` olmalı ([Zenodo 14037769](https://zenodo.org/records/14037769)). Aynı kayıttaki `ckpt_ChangeMamba_bright_standard_ML_split.pth` Türkiye'yi eğitimde gördüğü için başlangıç ağırlığı olarak kullanılmamalı.

**Kanıt.** ChangeMamba'nın asıl riski mühendislik tarafında. `selective_scan` CUDA çekirdeği derlenmeli. `setup.py` koşulsuz olarak `compute_70` bayrağı ekliyor, `requirements.txt` ise `timm==0.4.12` sürümünü sabitliyor ([setup.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/kernels/selective_scan/setup.py), [requirements.txt](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/requirements.txt)). CUDA 13 ise Volta için çevrimdışı derleme desteğini kaldırdı ([NVIDIA](https://docs.nvidia.com/cuda/cuda-features-archive/index.html)). Colab imajı `torch 2.14.1+cu130` ile geliyor ve apt'te CUDA toolkit görünmüyor ([pip-freeze.gpu](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt)). Yazarlar kodun Linux dışında test edilmediğini belirtiyor ve Türkiye sıfır-örnek checkpoint'i ya da betiği yayımlamamışlar ([README](https://github.com/ChenHongruixuan/ChangeMamba)). BRIGHT'ın Zenodo kaydında ise DamageFormer, DeepLabV3+, SiamAttnUNet, SiamCRNN ve UNet için `turkey-earthquake` sıfır-örnek checkpoint'leri hazır duruyor ([Zenodo 15349462](https://zenodo.org/records/15349462)). Bu tabloya göre karar şöyle:

| Seçenek | Türkiye `[ZS-EVENT]` çıpası | Hazırlık durumu | Ana risk | Tam koşu | CU (tahmin, @5,40) |
|---|---|---|---|---|---|
| **ChangeMamba-MMBDA Tiny (önerilen)** | 41,28 | Apache-2.0 kod; ZS checkpoint ve betik yok, BRIGHT'ın olay çıkarma mantığı taşınmalı | Çekirdek derlemesi, `timm` sabitlemesi | 1 | 50–75 + derleme kapısı ≤5 |
| Swin-T DamageFormer (yedek) | 37,20 | BRIGHT deposunda model, Zenodo'da Türkiye ZS checkpoint'i | Düşük; çıpa 4 puan aşağıda | 1 | 50–75 |
| Hazır ZS checkpoint'ler (SiamCRNN, UNet, DamageFormer) | 38,32 / 37,82 / 37,20 | Yalnız çıkarım | Yeni kod sürümüyle yükleme doğrulanmadı | 0 | 3–5 |
| DFC25 1. çözümün indirgenmiş hâli, PVT-v2-b2 | Yok | Lisans yok, test dosya adları koda gömülü | Transdüktif, kıyaslanamaz | ~3 | ~150–225 (yargı) |
| zzzbnu, VSSM-Base 3 aşama | Yok | Komutlar 3 GPU için | Derleme ve test görüntüsüyle ince ayar | 3 | ~150+ (yargı) |
| CVPRW 1. çözümün tam hattı | Yok, metrik segm_AP | Repo 404 | Ulaşılamıyor | 8–10+ | 400+ (yargı) |
| DINOv3 SAT ViT-L + SAR kolu | Yok | Kapılı ağırlık | Lisans, VRAM ölçülmedi | ≥1 | >75 (yargı, ölçülmedi) |

**Yargı.** Bu iki adımlı bir karardır ve ikinci adım otomatiktir. Önce ChangeMamba çekirdeği en fazla bir saat içinde derlenmeye çalışılır. Derleme ve 20 adımlık bir ileri-geri test geçerse ana koşu ChangeMamba ile yapılır. Geçmezse ana koşu, aynı reçeteyle Swin-T DamageFormer'a döner ve hedef "41,28'i geçmek" yerine "37,20'yi açıkça geçmek ve 41,28'e yaklaşmak" olur. Bu geri dönüşü önceden kabul etmek, hata ayıklamaya kontrolsüz CU harcamaktan daha ucuzdur.

## Rakipler "Türkiye'yi görmedik" cümlesini çoktan kullandı

**Kanıt.** GitHub'daki ekosistem üç kümeye ayrılıyor. Birincisi bakımı süren BRIGHT çevresi. İkincisi büyük kısmı arşiv durumundaki xView2/xBD çözümleri; birincilik çözümü 2 GPU'da yaklaşık 7 gün sürüyor ve yazarı tekrarlanabilir olmadığını söylüyor ([xview2_1st](https://github.com/vdurnov/xview2_1st_place_solution)). Üçüncüsü dağınık SAR araçları. Projenin doğrudan rakipleri aşağıda (yıldız sayıları 5 Kasım 2026 tarihli):

| Repo | ★ | Ne yaptı | Metrik (türüyle) | Bu proje için anlamı |
|---|---|---|---|---|
| [BRIGHT](https://github.com/ChenHongruixuan/BRIGHT) | 262 | Veri seti ve kıyaslama; ZS, OS, UDA, SSL | 41,28 `[ZS-EVENT]` | Çıpa, betikler, Apache-2.0 |
| [ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba) | 657 | Taban model | 67,63 `[STD-SET]` | Kod tabanı |
| [DFC25 1.](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping) | 20 | Bina rehberli pseudo-label | 54,28 `[DFC-TEST]` | Lisans yok; son işlemde test dosya adları koda gömülü ([post_cd_process.py](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping/blob/main/ChangeDetection/post_cd_process.py)) |
| [DF2025](https://github.com/zxqyiyang/DF2025) | 4 | Mamba tabanlı, 3 aşamalı test ince ayarı | 50,81 `[DFC-TEST]` | GRSS raporundaki kod bağlantısı ile içerik çelişiyor |
| [BightSolution](https://github.com/huang-he99/BightSolution) / [Damage-Aware](https://github.com/whf-68/Damage-Aware-SAR-Optical-Query-Learning-framework) | – / 0 | CVPRW26 1. ve 2. | 0,182 / 0,181 `[CVPR-TEST segm_AP]` | 404 ve boş repo; yalnız makaleden biliniyor |
| [CS437 group40](https://github.com/m-usmanzahid/cs437-group40-multimodal-earthquake-damage-assessment) | 1 | **BRIGHT optik + SAR**, bir olay dışarıda, MC Dropout, ECE | Türkiye ikili IoU 0,4953, Fas 0,1719 | En yakın öğrenci rakibi; çok tohum yok |
| [Enkaz](https://github.com/emrtunali/enkaz) | 0 | Şehir dışarıda, kalibrasyon, bootstrap, etkileşimli sayfa | Antakya makro F1 0,490, ECE 0,01'den 0,08–0,12'ye | Sunum çıtası yüksek; ama optik + optik, SAR yok |
| [bright-dfc2025-reproduction](https://github.com/antoniopiogrossi/bright-dfc2025-reproduction) | 0 | DFC25 1. çözümünü yeniden üretme tezi | Henüz eğitim yok | Yeniden üretim tek başına özgün değil |
| [DisasterEO](https://github.com/KotlinWang/DisasterEO) | 13 | Görülmemiş olaya aktarım, operasyonel haritalar | Sayı okunmadı | Etkin rakip araştırma hattı |

**Kanıt.** Literatür ve kurumsal sistemler ölçek duygusu veriyor. Russo ve arkadaşlarında aynı veride çapraz doğrulama F1'i 0,886 iken şehir dışarıda tutulunca 0,12–0,38'e düşüyor ([arXiv 2506.22338](https://arxiv.org/html/2506.22338)). Singh ve arkadaşlarında bölgeler arası F1 0,430 ([arXiv 2412.04664](https://arxiv.org/html/2412.04664v3)). CSB saha sayımına karşı ölçüldüğünde Copernicus EMS'in elle derecelendirmesi F1 0,15–0,23, Microsoft'un CNN'i F1 0,24, öğrenmesiz ARIA radar DPM'i ise F1 0,47 alıyor. Aynı çalışmaya göre ilk SAR sahnesi depremden yaklaşık 56 saat sonra, tam kapsama ise 134 saat sonra geldi ([Ainscoe ve ark.](https://www.nature.com/articles/s43247-025-02623-4)). BRIGHT etiketleri saha sayımı değil, uzmanların optik görüntüden yaptığı yorumdur ([arXiv v4](https://arxiv.org/html/2501.06019v4)). Kısa yol çalışmaları ayrı bir uyarı veriyor. Dondurulmuş temel modellerde tile düzeyinde **yalnız bina sayısı probu 0,653** alarak tam görüntü çiftini (0,568) geçiyor ([arXiv 2610.13766](https://arxiv.org/html/2610.13766)). Komşularıyla uyumsuz, yani izole hasarlı binaların yalnız %7–11'i tespit ediliyor ([arXiv 2610.38162](https://arxiv.org/html/2610.38162)).

**Yargı.** Projenin boşluğu "görülmemiş depremde test" değil, çünkü bu yapıldı. Kalan boşluk, önce optik ve sonra VHR SAR kullanan bir modelin aynı binalar üzerinde hem optik yorum etiketine hem bağımsız bir referansa karşı, öğrenmesiz radar tabanıyla yan yana, kilitli protokol ve güven aralıklarıyla sınanması. Bunu yapan açık bir repo bulunamadı; bu "yok" değil, "bu aramada bulunamadı" demektir. Aşağıdaki farkların neredeyse hepsi sıfır ek eğitim istiyor. Tek koşu kısıtında bu durum projenin lehine:

| Fark | Neden işe yarar | Tam koşu | CU (tahmin) |
|---|---|---|---|
| Zenodo'daki Türkiye ZS checkpoint'leriyle 38,32 / 37,20 / 37,82'yi yeniden üretmek | Veri ve metrik hattının tam hedef protokolde doğru olduğunu eğitimsiz kanıtlar; CS437 ve Enkaz bunu yapmıyor | 0 | 3–5 |
| Kilitli protokol: commit hash, eşikler ve başlık metriği Türkiye'ye dokunmadan önce etiketlenir; Türkiye'yi gören hiçbir ağırlık kullanılmaz | CVPRW'de holdout ile test sıralaması arasındaki Spearman korelasyonu 0,35 ([rapor](https://arxiv.org/abs/2607.22746)) | 0 | 0 |
| Sınıf bazlı IoU, 3 bina sınıfının ortalaması, bina düzeyi makro F1, kare kümelerine göre bootstrap güven aralığı | Hasarlı sınıfın kırılganlığını açıkça gösterir (DFC testinde 9,75) | 0 | <1 |
| İkinci referans: HOT/OSM yıkık bina katmanı ([HDX](https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings)), anlaşmazlık tablosu ve elle denetlenmiş 100–200 bina | Optik yorum etiketinin modeli haksız yere cezalandırdığı yerleri bulur | 0 | 0 (CPU) |
| Öğrenmesiz radar tabanı: ARIA Sentinel-1 DPM ([30 m](https://data.amerigeoss.org/it/dataset/aria-damage-proxy-map-copernicus-sentinel-1-for-the-turkiye-earthquakes)) aynı binalarda | Saha doğrulamalı tek kurumsal taban (F1 0,47) | 0 | 0 (CPU) |
| Capella / Umbra kırılımı, layover ve gölge galerisi, izole hasar analizi | SAR okuryazarlığı gösterir, bağlam kısa yolunu sınar | 0 | <1 |
| TTA entropisi, risk-kapsam eğrisi (AURC), sıcaklık 13 olayda seçilir, ECE Türkiye'de kilit sonrası ölçülür | Kayma altında kalibrasyonun ne kadar bozulduğunu ölçer, iddia etmez | 0 | 1–2 |
| ONNX dışa aktarımı, bina başına gecikme, gerçek görüntü tarihleriyle "en erken ne zaman" tablosu | Sistem disiplini; C++ ve çıkarım isteyen ilanlarla uyumlu (yargı) | 0 | <1 |
| İkinci tohum | Koşudan koşuya varyansı ölçer, 2 üyeli topluluk verir | 1 | 50–75 |

**Yargı.** Bazı iddialar haksız olur ve kurulmamalıdır. "Yeni depremlere genelleniyor" demek yanlış olur; CS437'de Fas 0,17 ve Türkiye 0,50 farkı olay bağımlılığını gösteriyor. "Saha doğrulamalı" demek yanlış olur, çünkü etiket optik yorumdur. "Copernicus'tan iyi" demek yanlış olur, çünkü referans, kapsam ve zaman farklı. "Gerçek zamanlı ve her havada" demek yanlış olur, çünkü ilk SAR 56 saatte geldi. Ağırlıklı F1 veya genel doğruluk manşet olmamalı. Aynı CS437 modelinde ağırlıklı F1 0,9021, makro F1 ise 0,4424 ([repo](https://github.com/m-usmanzahid/cs437-group40-multimodal-earthquake-damage-assessment)). Enkaz ve CS437'yi README'de anmak ve farkı tabloyla göstermek, ciddi bir değerlendiricinin onları kendisinin bulmasından daha güvenlidir.

## Tek reçete: Türkiye'ye dokunmadan 40 bin adım, 50–75 CU

**Kanıt ve düzeltme.** Eğitim reçetesi notu tek koşu için `standard_ML` ayrımını öneriyor. Ama bu ayrımda Türkiye 777 eğitim, 113 doğrulama ve 224 test karesiyle eğitimin içinde ([splitname](https://github.com/ChenHongruixuan/BRIGHT/tree/master/bda_benchmark/dataset/splitname)). Yani bu proje için doğrudan sızıntı. Doğru ayrım, BRIGHT'ın `cross_event/zero_shot` mantığıyla `--test_event_name turkey-earthquake` kullanmak ([README_cross_event](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/README_cross_event.md)). Liste sayımından yaptığım hesaba göre, betik standart listeleri kullanıyorsa yaklaşık 2.141 eğitim ve 337 doğrulama karesi kalır; Türkiye'nin 1.114 karesinin tamamı teste gider. Bu sayılar kapı 2'de doğrulanmalı. BRIGHT'ın ortak tarifi şöyle: AdamW, 1e-4, ağırlık sönümü 5e-3, batch 16, kırpma 640, 50 bin adım, CE + Lovász, sınıf ağırlıkları [1,1,1,1] ([arXiv v4](https://arxiv.org/html/2501.06019v4)). Bir tuzak var. BRIGHT kodunda `max_iters` örnek sayısıdır: 800.000 / 16 = 50.000 adım ([make_data_loader.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/dataset/make_data_loader.py)). ChangeMamba'nın güncel kodunda ise adım sayısıdır ([common.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/datasets/common.py)). İki araştırma notu bile aynı betiği 20 bin ve 80 bin adım diye farklı okudu. Bu tek parametre 16 katlık bir bütçe hatasına yol açabilir.

Önerilen dondurulmuş yapılandırma:

```yaml
data:   {protocol: zero_shot, test_event: turkey-earthquake, crop: 640, workers: 8,
         nodata_ignore: "pre==0 & post==0"}
aug:    {flips: true, rot90: true, photometric: false}
norm:   {optical: imagenet, sar: [train_mean, train_std]}   # ~0,22 / 0,15-0,18 (0-1 ölçeği)
model:  {name: changemamba_mmbda_tiny, init: vssm_tiny_imagenet, fallback: swin_t_damageformer}
optim:  {name: adamw, lr: 1.0e-4, wd: 5.0e-3}
sched:  {warmup_steps: 1500, type: cosine, min_lr: 1.0e-6}
train:  {batch: 16, iters: 40000, amp: bf16, ema: 0.999, clip: <kapı 8 p90>, seed: 42}
loss:   {ce_loc: 1.0, ce_clf: 1.0, lov_loc: 0.5, lov_clf: 0.75,
         class_weights: [1,1,1,1], ignore_index: 255, lovasz_fp32: true}
eval:   {every: 2000, split: val_13_events, turkey_during_training: false, select: ema_val_miou}
final:  {turkey_once: true, headline: raw_argmax_clf, secondary: [tta_flip, building_vote]}
ckpt:   {every: 1000, keep_last: 2, keep_best: 1, atomic: true, sinks: [drive, hf_hub]}
```

| Ayar | Karar | Dayanak | Kanıt mı, yargı mı | Ek maliyet |
|---|---|---|---|---|
| AdamW 1e-4, WD 5e-3, kayıp katsayıları | Resmi değerler | BRIGHT, ChangeMamba ve DF2025'te aynı ([train_MambaBDA.py](https://github.com/zxqyiyang/DF2025/blob/master/changedetection/script/train_MambaBDA.py)) | Kanıt | 0 |
| Kırpma 640, batch 16 | Resmi değerler; bellek yetmezse 512/8 | ChangeMamba betiği 512/8 kullanıyor ([sh](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/sh_file/train_MMBDA_BRIGHT.sh)) | Kanıt; Mamba'nın bellek kullanımı ölçülmedi | 0 |
| 40 bin adım (640 bin örnek) | Resmi 50 binin altında | ChangeMamba betiğinin örnek bütçesi; 50 bin adım Pro'ya sığmaz | Yargı | 50 bin seçilirse +13–19 CU |
| Sınıf ağırlığı yok | [1,1,1,1] | Resmi; Türkiye sağlam baskın olduğu için hasarı ağırlıklandırmak yanlış pozitifleri artırabilir | Yargı | 0 |
| Warmup + kosinüs, bf16, EMA | Ekle | [Tuning Playbook](https://github.com/google-research/tuning_playbook); BRIGHT'ta denenmemiş | Yargı, düşük risk | ~0 |
| SAR için eğitim kümesi ortalaması ve standart sapması | ImageNet yerine | İki bağımsız kaynak 0,22 / 0,15–0,18 veriyor ([disaster.yaml](https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/config/disaster.yaml), [torchange](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/metrics/bright.py)) | Yargı | 0 |
| Boş (nodata) pikselleri kayıpta yok saymak | Ekle | Fas'ta tamamen boş SAR kareleri var (örneklem) | Yargı, etkisi ölçülmedi | 0 |
| Benek giderme, stil artırımı, UDA, SAM, MixStyle | Ekleme | UDA Türkiye'de tabanın altında; benek giderme için doğrudan test yok | Kanıt (UDA), yargı (diğerleri) | Bütçe korunur |
| Türkiye'yi eğitim sırasında değerlendirmek | Kapat | ChangeMamba her 500 adımda Test'i yazdırıyor ([metadata.py](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/tasks/metadata.py)) | Kanıt (sızıntı riski) | Ayrıca saatler kazandırır |
| TTA ve bina düzeyi oylama | Yalnız son çıkarımda, ikincil satır | BRIGHT'ta TTA +0,56, nesne oylaması +0,69, hepsi birlikte +0,92 `[olay düzeyi]`; nesne oylaması küme düzeyinde −0,41 ([ESSD](https://essd.copernicus.org/articles/17/6217/2025/)) | Kanıt (küçük kazanç) | 0 koşu, 4 kat çıkarım |
| Türkiye görüntüsüyle pseudo-label | Varsayılan değil | Şampiyonlarda var ama katkısı ayrıştırılmamış; transdüktif | Karışık kanıt | +~0,25 koşu, 13–19 CU |

**Yargı: önceden kayıt.** Başlık sonucu, 13 olayın doğrulama kümesinde en iyi EMA checkpoint'inin Türkiye'de tek seferlik, TTA'sız ve `argmax(clf)` ile hesaplanan 4 sınıf mIoU'sudur. Bu değer resmi kod yolunu izler ve 41,28 ile kıyaslanır. İkincil satırlar baştan ilan edilir: flip TTA, bina düzeyi çoğunluk oylaması ve bina maskeli harita. Her birinin ayarı 13 olayda seçilir. Türkiye sonuçları görüldükten sonra hiçbir satır seçilmez ve hiçbir eşik değiştirilmez. Pseudo-label yapılırsa ayrı bir "transdüktif" tabloda verilir.

| Adım | Tam koşu | Süre (tahmin) | CU @5,40 (40 GB) | CU @7,52 (80 GB) |
|---|---|---|---|---|
| 0. Ortam, veri indirme, Zenodo ZS checkpoint'leriyle Türkiye yeniden üretimi | 0 | 1–1,5 sa | 5–8 | 8–11 |
| 1. Mamba derleme kapısı, en fazla 1 saat | 0 | ≤1 sa | ≤5,4 | ≤7,5 |
| 2. Ön kontrol kapıları ve 2000 adımlık prova | ~0,05 | 1–1,5 sa | 5–8 | 8–11 |
| 3. **Ana koşu, 40 bin adım** | **1** | 9,3–13,9 sa, ayrıca 1–2 devam | **50–75** | 70–105 |
| 4. Türkiye'de tek değerlendirme, TTA, oylama, belirsizlik, ONNX | 0 | 0,5–1 sa | 3–5 | 4–8 |
| **Asgari plan toplamı** | **1** | **13–19 sa** | **~65–100** | ~90–140 |
| 5. İsteğe bağlı ikinci tohum | +1 | 9,3–13,9 sa | +50–75 | +70–105 |
| 6. İsteğe bağlı pseudo-label ince ayarı, 10 bin adım | +~0,25 | 2,5–3,5 sa | +13–19 | +19–26 |

**Kanıt.** Hız tahmini, timm'in RTX 3090 ölçümlerinden ve "A100 ile 3090 yaklaşık eşit" varsayımından türetildi: Swin-T için 0,8–1,2 adım/saniye ([timm benchmark](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-train-amp-nhwc-pt112-cu113-rtx3090.csv)). Mamba'nın hızı hiç ölçülmedi; kapı 7 bunu 200 adımda ölçer. A100 yakım hızını Google yayımlamıyor. En güncel ve şeffaf tahmin 40 GB için 5,40, 80 GB için 7,52 CU/saat. High-RAM'i açmak yakımı yaklaşık %39 artırıyor ([McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)). Colab Pro aylık 100 CU veriyor ve oturumlar "genel olarak en çok 12 saat" sürüyor. Tarayıcı kapalıyken çalışmaya devam etmek (background execution) yalnız Pro+'ta var; Pro+ 600 CU ve 24 saat veriyor. Türkiye fiyatları Pro için ₺165,60, Pro+ için ₺826,80, CU'lar 90 günde sona eriyor ([Colab FAQ](https://research.google.com/colaboratory/faq.html), [Colab signup](https://colab.research.google.com/signup)). **Yargı.** 16 hafta boyunca Pro, 4 × ₺165,60 = ₺662,40'a toplam 400 CU demektir. Bu, asgari planı ve ikinci tohumu rahatça taşır, ama tek bir ayda ikisi birden sığmaz. Asgari plan ayın üst sınırına dayanırsa ₺165,60'lık 100 CU'luk kullandıkça öde paketi tampon olur. Önerim şu: adım 0–2 birinci ayda (~15–20 CU), ana koşu ikinci ayın başında. Böylece ana koşu taze bir 100 CU ile başlar.

| Hafta | İş | Colab CU |
|---|---|---|
| 1–2 | Veri lisans tablosu, CPU birim testleri, protokol kilidi ve commit etiketi | 0 |
| 3 | Adım 0–2: yeniden üretim, derleme kapısı, ön kontrol | ~15–20 |
| 5–6 | Ana koşu (yeni ayın CU'su) | 50–75 |
| 6–10 | Sıfır koşulu analizler: HOT/OSM, DPM, sensör ve hata galerisi, belirsizlik | ~2–5 |
| 9–10 | İsteğe bağlı ikinci tohum (üçüncü ay) | 0 veya 50–75 |
| 11–16 | ONNX ve gecikme, model kartı, README, etkileşimli sayfa, olumsuz bulgular, tampon | <2 |

## Araç zinciri ve notebook: yedi hücre, on bir kapı

**Kanıt.** BRIGHT dosyaları sıkıştırılmamış GeoTIFF olarak geliyor. Optik 3 bant, SAR 1 bant, ikisi de uint8. Etiketler yalnız {0,1,2,3}; 4.246 karenin hiçbirinde başka değer yok. Koordinat sistemi olaya göre değişiyor; Türkiye UTM 37N ve 0,35 m. Zip dosyaları yaklaşık 13,2 GB, açılmış hâli 22,3 GB tutuyor. Hugging Face kartının lisansı CC BY-SA 4.0 ([Kullervo/BRIGHT](https://huggingface.co/datasets/Kullervo/BRIGHT)). Ukrayna, Myanmar ve Meksika'nın optik verisinin dağıtılamadığı README'de yazıyor. Yine de HF zip'i 4.246 karenin hepsini içeriyor ve bu olaylardan örneklenen 12 karede görüntü benzeri veri var; bunların kaynağı doğrulanmadı ([README](https://github.com/ChenHongruixuan/BRIGHT/blob/master/README.md)). Olay adı ayrıştırmasında `la_palma-volcano` alt çizgi içerdiği için `split('_')[0]` yanlış sonuç veriyor, `rsplit('_',1)[0]` doğru çalışıyor. Colab imajında `torch 2.14.1+cu130`, `timm 1.0.31`, `rasterio 1.5.2`, `tifffile 2026.10.15`, `huggingface_hub 1.33.0`, `hf-xet` ve `wandb 0.28.1` kurulu geliyor ([pip-freeze.gpu](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt)). `hf_transfer` terk edildi; varsayılan `hf_xet` ([HF docs](https://huggingface.co/docs/huggingface_hub/hf_transfer)). Colab, Drive'dan çok sayıda küçük okuma yapılmamasını ve arşivin VM diskine kopyalanıp orada açılmasını öneriyor ([FAQ](https://research.google.com/colaboratory/faq.html)).

| Bileşen | Seçim | Gerekçe (yargı) |
|---|---|---|
| PyTorch, timm, torchvision | Colab'da kurulu gelen sürümler; venv yok, yeniden kurulum yok | CUDA wheel'ini yeniden indirmek disk ve zaman harcar |
| Model kodu | ChangeMamba deposu, BRIGHT'ın olay çıkarma mantığı taşınarak; yedek BRIGHT DamageFormer | İkisi de Apache-2.0 |
| Okuma ve yazma | Eğitimde `tifffile`, çıktı GeoTIFF'i için `rasterio` | Sıkıştırmasız dosyalar ucuz okunur; georeferans korunur |
| Artırım | Yaklaşık 30 satırlık NumPy (crop, flip, rot90, üç diziye aynı dönüşüm) | Resmi artırım seti bu kadar |
| Yapılandırma | Tek YAML ve `dataclass`; config hash'i run_id'nin parçası | Hydra tek koşuda risk ekler |
| İzleme | Drive'da JSONL ana kayıt; W&B isteğe bağlı ve `offline` güvenli | Anahtar sorunu eğitimi durdurmamalı |
| Checkpoint | Her 1000 adımda Drive (son 2 ve en iyi), her 5000 adımda HF Hub özel repo; `*.tmp` ardından `os.replace` | Tek kopma noktası kalmaz |
| Kalite | `ruff`, `pytest`, `nbstripout`; GitHub Actions'ta CPU ile sentetik testler | Dönüşüm eşleşmesi, `la_palma`, metrik ve resume hataları GPU yakmadan yakalanır |
| Referans olarak kalanlar | TorchGeo (yalnız DFC25 kurgusu), torchange (`ever` bağımlılığı), SMP | Çalışma zamanına ek risk getirir |
| `torch.compile`, Colab CLI | Kullanma | Tek koşuda ek hata yüzeyi; CLI yalnız Linux ve macOS'ta çalışıyor |

**Yargı: derleme kapısında denenecekler.** Çekirdeği derlemeyi denemeden önce dört şey yapılmalı. `setup.py`'den `compute_70` çıkarılıp yalnız `compute_80` bırakılmalı. `CUDA_HOME`, pip'teki `nvidia-cuda-nvcc` ikilisine yönlendirilmeli. Paket `pip install --no-build-isolation .` ile kurulmalı. `timm==0.4.12` sabitlemesi kaldırılmalı. Derlenen wheel Drive'a alınmalı; Colab imajı değişirse ABI kırılabilir. Bu maddelerin hiçbiri çalıştırılarak doğrulanmadı. Saf PyTorch `selective_scan_easy` yalnız bir örnek olarak duruyor; hızı ve doğruluğu bilinmiyor ([kernels README](https://github.com/ChenHongruixuan/ChangeMamba/blob/master/kernels/selective_scan/README.md)).

Notebook mantık içermemeli, yalnız paketteki modülleri çağırmalı ve "Run All" ile yukarıdan aşağı, her seferinde aynı sonucu verecek (idempotent) biçimde çalışmalı:

| Hücre | İş | Kopma sonrası davranış |
|---|---|---|
| 1 | Drive'ı bağla, etiketli commit'i `git clone` et, `pip install -e .`, secret'ları oku | Commit SHA kaydedilir, kirli çalışma ağacı reddedilir |
| 2 | Ortam kapısı: A100, VRAM, High-RAM kapalı, disk ≥60 GB, CU bakiyesi elle girilir | Koşul tutmazsa `raise` |
| 3 | Zip'leri VM diskine indir veya Drive'dan kopyala, aç, bütünlük kontrolü | `.done` dosyası varsa atlanır; süre loglanır |
| 4 | Ön kontrol kapıları (aşağıda) ve sonuçların `preflight.json`'a yazılması | Config hash'i aynıysa atlanır |
| 5 | Eğitim, subprocess olarak; çıktı Drive'a da yazılır | `latest.pt` varsa tam durumla devam eder |
| 6 | Türkiye'de tek değerlendirme: EMA ve ham ağırlık, TTA, olay ve sınıf tabloları, örnek GeoTIFF'ler | `DONE` dosyası yoksa çalışmaz |
| 7 | Kopyalar doğrulandıktan sonra `runtime.unassign()` | Yakım durur |

Ön kontrol kapıları toplam 1–1,5 saat ve 5–8 CU tutar (tahmin). Her biri belli bir GPU israfını önlüyor. Veri yükleyicinin uzunluğu hedef adım sayısına eşit olmalı; bu, `max_iters` tuzağını yakalar. Başlangıç kayıpları beklenen değerlere yakın çıkmalı: hasar başlığında ln 4 ≈ 1,386, bina başlığında ln 2 ≈ 0,693. Augmentation kapalıyken 8 kare 300–500 adımda ezberlenebilmeli. SAR sıfırlanınca hasar çıktısı değişmeli; bu, SAR dalının bağlı olduğunu gösterir. Gerçek hatla 200 adım koşulup adım hızı, tepe VRAM ve CU projeksiyonu ölçülmeli. 2×LR ile 500 adım koşulup p90 gradyan normu kırpma eşiği yapılmalı. Adım 60'ta süreç `SIGKILL` ile öldürülüp devam testi yapılmalı. Metrik iki uç durumda doğru çalışmalı: kusursuz tahmin 1,0, hep arka plan tahmini yaklaşık 0,21 vermeli. En güçlü metrik kontrolü ise adım 0'daki yeniden üretimdir ([Karpathy](https://karpathy.github.io/2019/04/25/recipe/), [Playbook](https://github.com/google-research/tuning_playbook)). Bu adımda SiamCRNN'in Türkiye ZS checkpoint'i 38,32'ye yaklaşamıyorsa hata modelde değil veri ya da metrik hattındadır ve ana koşu başlamamalıdır. BRIGHT'ın kendi `--resume` seçeneği yalnız model ağırlığını yüklüyor; optimizer ve adım sayacı geri gelmiyor ([train_UNet.py](https://github.com/ChenHongruixuan/BRIGHT/blob/master/bda_benchmark/script/standard_ML/train_UNet.py)). Bu yüzden devam durumu model, EMA, optimizer, scheduler, adım ve bütün RNG durumlarını içermeli. Veri tarafında `seed + step` ile çalışan sonsuz bir sampler, atlanan adımlarda verinin yeniden okunmasını önler.

## Sonuç

Tek koşu kısıtı projeyi zayıflatmıyor, odağını değiştiriyor. Yayımlanmış kanıt, mimari tercihlerin Türkiye'de birkaç puanlık ve büyük ölçüde gürültü içinde kalan farklar ürettiğini gösteriyor. Buna karşılık değerlendirme kararları sonucu değiştiriyor. Holdout ile test sıralaması arasındaki korelasyon 0,35; UDA yöntemlerinin beşi de tabanın altında; yarışma kazananlarının hasarlı sınıfı yeni olayda 9,75 IoU'ya düşüyor. Bu yüzden en değerli yatırım, ana koşuyu doğru yapmak ve geri kalan bütün emeği sıfır koşulu işlere vermek: kilit, yeniden üretim, ikinci referans, radar tabanı, sensör ve geometri analizi. ChangeMamba seçimi en iyi skoru kovalamak için değil, tek doğrudan çıpaya karşı dürüst bir ölçüm yapabilmek için yapılıyor.

İşe alım açısından en güçlü bulgu, büyük ihtimalle model skorunun kendisi olmayacak. Asıl bulgu, aynı binalarda optik yorum etiketi, HOT/OSM ve radar DPM'in birbirinden ne kadar ayrıştığını gösteren anlaşmazlık tablosu olacak. Hiçbir rakip repo bunu VHR SAR modeliyle birlikte yapmıyor. Sonuç 41,28'in altında kalsa bile kilitli protokolle elde edilmiş, sınıf kırılımlı ve güven aralıklı bir sayı, gevşek protokolle elde edilmiş daha yüksek bir sayıdan daha fazla şey söyler.

## Yalnız proje sahibinin verebileceği kararlar

1. **Abonelik:** Pro mu (₺165,60/ay, 100 CU, tarayıcı açık, 12 saatte 1–2 devam), yoksa ana koşunun yapılacağı ay için Pro+ mı (₺826,80, 600 CU, tarayıcı kapalıyken 24 saat)? Kullandıkça öde tamponu kabul edilebilir mi?
2. **Koşu sayısı:** Yalnız 1 tam koşu mu, yoksa varyansı ölçmek ve 2 üyeli topluluk kurmak için ikinci tohum da (+50–75 CU) mı?
3. **Derleme kapısı geçmezse:** Swin-T DamageFormer'a dönüş (çıpa 37,20) önceden kabul ediliyor mu, yoksa Mamba hata ayıklamasına ek CU ayrılacak mı?
4. **Ukrayna, Myanmar ve Meksika optiği:** Kaynağı doğrulanmamış HF kopyaları mı kullanılacak, BRIGHT tutorial'ına göre yeniden mi hazırlanacak, yoksa bu olaylar çıkarılacak mı? Çıkarılırlarsa hasarlı piksellerin yaklaşık %63'ü kaybolur ve 41,28 ile kıyas bozulur.
5. **Başlık metriği:** Kod yolu `argmax(clf)` mi, yoksa bina maskeli harita mı? TTA ve oylama başlığa mı girecek, ikincil satırda mı kalacak? Karar Türkiye'ye dokunmadan önce commit'lenmeli.
6. **Transdüktif kullanım:** Türkiye'nin etiketsiz görüntüsüyle pseudo-label (+13–19 CU) yapılacak mı? Yapılırsa yalnız ayrı bir tabloda raporlanır.
7. **İkinci referans ve lisans:** CSB saha verisine erişim için yazarlara başvurulacak mı, yoksa yalnız HOT/OSM ile mi yetinilecek? DINOv3 (askeri kullanım yasağı) ve MaRS (lisans çelişkisi, olası Türkiye sahneleri) kapsam dışı mı tutulacak?
8. **Rakiplerin anılması:** Enkaz ve CS437 README'de adıyla anılıp farklar tabloyla gösterilecek mi?
