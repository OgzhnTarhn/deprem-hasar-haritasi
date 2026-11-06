# Optik + SAR segmentasyon modellerini görülmemiş afete genelleme ve 0,3-1 m görüntü için önceden eğitilmiş kodlayıcılar

Durum tarihi: 5 Kasım 2026. Varsayım: Colab A100 (yaklaşık 40 GB VRAM), birkaç tam eğitim koşusu.

Kanıt ölçeği: **Güçlü** = doğrudan görev (VHR optik+SAR hasar, görülmemiş olay) ve birden çok bağımsız kaynak. **Orta** = doğrudan görevde tek kaynak ya da az olay, veya yakın görevde birden çok kaynak. **Zayıf** = analoji (başka görev veya alan) ya da ablasyonu ayrıştırılmamış sonuç. **Yok** = kanıt bulunamadı, öneri akıl yürütmeye dayanıyor.

Maliyet ölçeği: "koşu" = bir tam eğitim. "0 koşu" = ek eğitim gerekmez. Yüzdeler koşu başına ek süre tahminimdir; ölçülmedi.

Bağlam için en önemli sayı: görülmemiş olaylarda sorun henüz çözülmüş değil. BRIGHT Challenge 2026 şampiyonları test mAP 0,182 ve 0,181 aldı; en iyi alan-içi holdout skoru 0,513 idi. Holdout lideri takım testte 0,069'a düştü, holdout-test sıra korelasyonu 0,346 ([Bright Challenge raporu](https://arxiv.org/html/2607.22746)).

## Önceden eğitilmiş kodlayıcılar: hangileri 0,3-1 m görüntüye ve Capella/Umbra SAR'a uygun?

### Takeaway
Optik kol için en iyi desteklenen seçim DINOv3 ViT-L/16 SAT-493M'dir (0,6 m Maxar RGB ile eğitilmiş, alt-10 m görevlerde önde), ancak görülmemiş olaylarda hasar görevi üzerinde test edilmemiştir ve lisans kısıtları vardır. SAR kolunda Umbra ve Capella ile eğitildiği doğrulanan ağırlıklar yalnızca MaRS ve (Capella, ICEYE, Umbra sahneleri içeren) DINOSAR'dır; ikisinin de lisans durumu temiz değildir ve hiçbiri için görülmemiş olay kazancı gösterilmemiştir. Her iki BRIGHT Challenge şampiyonu da genel ImageNet tarzı ön eğitim kullandı, yani özel kodlayıcının kazancı henüz kanıtlanmamış bir hipotezdir.

### Cited Findings

**Doğrudan görev (BRIGHT ve benzeri)**
- BRIGHT Challenge 2026'nın iki şampiyonu da genel ImageNet tarzı ön eğitim kullandı; rapor DINOv2, DINOv3, Swin, ConvNeXt veya SAM'den söz etmiyor. Rapor SAR'a veya hasara özgü ön eğitimi yüksek getirili bir gelecek yönü olarak nitelendiriyor; bu bir sonuç değil, öneridir. İkinci takım iki ResNet-50-FPN akışı kullandı, birinci takım Mask2Former dekoderli ayrı optik ve SAR omurgaları kullandı (omurga adı rapor metninde yok) — [Bright Challenge raporu](https://arxiv.org/html/2607.22746)
- DFC25 Track 2 birincisi PVT-v2 (b2, b3, b4) kodlayıcıları kullandı, görüntüleri 1024x1024 korudu ve 24 GB RTX 3090 üzerinde eğitti. Değişim aşamasında tek bir PVT-v2-b2 yeterli oldu — [Li ve ark. 2025](https://arxiv.org/pdf/2505.04941)
- BRIGHT standart bölünmesinde ChangeMamba mIoU 67,63, DamageFormer 67,09, UNet 64,94. Olay düzeyinde ortalama 51,39 ve 52,26. Sıfır-atışlı olay aktarımında tüm temel modeller 40'ın altında, ChangeMamba 39,38 — [BRIGHT, ESSD 2025](https://essd.copernicus.org/articles/17/6217/2025/)
- MaRS proje sayfası DFC25-T2 hasar değerlendirmesi için IoU yaklaşık 41,49 bildiriyor. Sayfa metriğin tanımını ve aynı protokoldeki rakipleri vermiyor, bu yüzden DFC25 şampiyonunun 54,28 mIoU'su ile karşılaştırılamaz — [MaRS sayfası](https://rsidea.whu.edu.cn/mars.htm)
- STSF-Net, BRIGHT'ta öncül üreteci karşılaştırdı: öncülsüz 65,64, SAM 66,86, DINOv3 66,72, SAM2 67,91 mIoU. Optik kolda adaptörlü SAM2 Hiera, SAR kolunda sıfırdan küçük Swin kullanıldı. Olaylar arası test yok — [STSF-Net](https://arxiv.org/html/2604.05527v1)

**DINOv2 / DINOv3 (optik)**
- SAT-493M: 512x512 boyutunda 493 milyon Maxar RGB ortorektifiye görüntü, 0,6 m. Uydu ağırlıkları yalnızca ViT-L/16 (300M parametre, ViT-7B'den damıtılmış) ve ViT-7B/16 (6.716M). Kart GEO-Bench segmentasyon ortalamasını ViT-L için 74,5, ViT-7B için 75,0 veriyor. Erişim kapılı — [model kartı](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m)
- SAT-493M ağırlıkları için normalizasyon ortalaması (0,430, 0,411, 0,296), standart sapması (0,213, 0,156, 0,143); ImageNet değerleri kullanılmamalı — [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- GEO-Bench-2: DINOv3-ViT-L-SAT ve ConvNeXt modelleri alt-10 m ve RGB/NIR yeteneklerinde önde. Coğrafi temel modeller ResNet-50'yi geçiyor ama özellikle yüksek çözünürlüklü RGB görevlerde ConvNeXt'i geçemiyor. Dondurulmuş kodlayıcı sistematik performans düşüşü yaptı ve model çiftlerinin yüzde 20'sinden fazlasında sıralamayı değiştirdi. Kullanılan yüksek çözünürlüklü veri setlerinde hasar görevi yok (FLAIR2, SpaceNet2, EverWatch, NZCattle) — [GEO-Bench-2](https://arxiv.org/html/2511.15658v2)
- Dondurma konusunda çelişki var. DINOv3 kartı ince ayarı son çare olarak öneriyor ([kart](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m)). Rein++ sentetikten gerçeğe sürüş verisinde DINOv2-L için tam ince ayar 61,7, dondurulmuş 65,6, Rein-G 67,5 mIoU veriyor ([Rein++](https://arxiv.org/html/2508.01667)). Uzaktan algılama DG'sinde dondurulmuş DINOv2-L 51,8, Rein 53,1, Earth-Adapter 56,2 ([Earth-Adapter](https://arxiv.org/html/2504.06220v4)). Luo ve ark. DINOv2 ViT-L'in hem dondurulmuş hem ince ayarlı durumda diğer omurgaları geçtiğini bildiriyor (yalnızca özet düzeyinde okundu) ([ISPRS 2025 kaydı](https://search.napstic.cn/literature/periodical/0220260200197171))
- CrossEarth'te D2M ayarında DINOv2 tabanlı model MTP'nin gerisinde kaldı (50,5 ve 54,3); yazarlar bunu uzaktan algılama ön eğitimi eksikliğine bağlıyor ve MTP'nin RVSA omurgasına geçince 55,5 alındığını bildiriyor — [CrossEarth](https://arxiv.org/html/2410.22629v2)

**DINO ailesi SAR üzerinde**
- AFRL: en yakın komşu başlıklı 10 görev ortalaması dengeli doğruluk: torch.hub DINOv2-ViTB 29,5; DINOv3-ViT7b-Web 34,4; DINOv3-ViTL-Sat 25,6; SARATR-X 44,6; SAR-SimCLR 60,3; AFRL-DINOv2-ViTB-p1 61,6. LoRA ince ayarı DINOv2-ViTB'yi 29,5'ten 71,7'ye çıkardı; tam ince ayarlı SARATR-X 75,1. Uydu ön eğitimi SAR'a yardım etmedi (Sat, Web'in altında). Görev ATR sınıflandırması, segmentasyon değil; bu bir analojidir. -p3 ve -p4 skorları transdüktif sızıntı nedeniyle doğrudan karşılaştırılamaz — [AFRL](https://arxiv.org/html/2509.21722v1)
- Dondurulmuş DINOv3 ViT-S+ düşük çözünürlüklü, ağır benekli sivil araç SAR'ında yüzde 26,9 doğruluk aldı; LoRA 29,8; optik prototip hizalama 33,3 — [Hirsch ve ark. 2026](https://arxiv.org/pdf/2609.07753)
- DINOv3-PEFT, dondurulmuş DINOv3 ile hafif uyarlamayı 0,5 m Capella içeren SAR yol segmentasyonuna uyguluyor (yalnızca arama özeti okundu; sayı doğrulanmadı) — [MDPI](https://www.mdpi.com/2072-4292/18/7/973)

**SAR'a özgü modeller**
- MaRS: Umbra ve Capella'dan 4.225 VHR SAR görüntüsü, 16.785.168 SAR-optik yama çifti, 0,35 m, X-bandı HH veya VV, 30 Aralık 2024'e kadar — [MaRS sayfası](https://rsidea.whu.edu.cn/mars.htm). İkili SwinV2 kodlayıcı, RGB ve SAR için ayrı yalnız-kodlayıcı ağırlıkları, timm ile yükleme, ön eğitimde 1024, aşağı akış örneklerinde 512 girdi — [MaRS GitHub](https://github.com/WanderRainy/MaRS). Dosyalar: base 347,7 MB, large yaklaşık 634-640 MB — [Zenodo](https://zenodo.org/records/17800805)
- MaRS lisansı çelişkili: README yalnızca akademik araştırma için olduğunu söylüyor ([GitHub](https://github.com/WanderRainy/MaRS)), proje sayfası tüm hakları saklı diyor ([sayfa](https://rsidea.whu.edu.cn/mars.htm)), Zenodo kaydı CC BY 4.0 gösteriyor ([Zenodo](https://zenodo.org/records/17800805))
- DINOSAR: UniSAR-7M (7.047.666 örnek; beş kamu veri seti ve 2.274 Capella, ICEYE, Umbra sahnesi), ViT-S/16 ve ViT-B/16, kod MIT ([GitHub](https://github.com/kenanking/DINOSAR)). Ağırlık lisansı köken incelemesi nedeniyle beklemede; girdi tek kanal, 224x224, ortalama 0,219 standart sapma 0,220 ([model kartı](https://huggingface.co/YTang/DINOSAR)). Sonuçlar [IJDE makalesinde](https://doi.org/10.1080/17538947.2026.2757416), bu oturumda okunmadı
- SARATR-X: HiViT, 0,18 M hedef örneği, veri kaynakları gemi, araç ve uçak ağırlıklı (MSTAR, HRSID, SSDD vb.), Capella ve Umbra anılmıyor, CC BY-NC 4.0, ticari olmayan — [GitHub](https://github.com/waterdisappear/SARATR-X)
- SARATR-X-v2: iTPN-B/L, ön eğitim 8 A800 GPU üzerinde 1200 epoch, ön eğitim tablosunda Capella ve Umbra yok. Segmentasyon: AIR-PolSAR-Seg-2.0'da belirgin kazanç (95,3 mIoU, +3,2 iTPN-B için). Tablo metni PDF'ten çıkarıldı, sütun hizalaması bozuk olabilir. OpenEarthMap-SAR'da U-Net 35,1, SegFormer 35,8, VMamba 34,7, SARATR-X-v2 iTPN-L 36,2 (+0,4). WHU-OPT-SAR'da PIDNet ile yaklaşık başa baş — [SARATR-X-v2](https://arxiv.org/pdf/2607.23238). Ağırlıklar kapılı, yazara e-posta gerekiyor; arama özetinde cc-by-4.0 görünüyor ama v1'in NC lisansıyla çelişiyor — [Hugging Face](https://huggingface.co/waterdisappear/SARATR-X-v2)
- SARMAE: ViT-B/L, SAR-1M (1.312.902 görüntü, 0,1-60 m; listelenen sensörlerde Capella ve Umbra yok), benek-farkında yeniden yapılandırma, dondurulmuş DINOv3 optik çapa. AIR-PolSAR-Seg 67,51 mIoU, en iyi temel model 52,58. Kod ve modeller için yalnızca yayınlanacak deniyor — [SARMAE](https://arxiv.org/html/2512.16635v1)

**Sentinel odaklı temel modeller**
- PANGAEA xView2 mIoU (yüzde 100 etiket): Scale-MAE 60,72; DOFA 59,64; GFM-Swin 59,15; UNet 58,68; RemoteCLIP 57,41; ViT 57,43; CROMA 53,27; SatlasNet 52,23; Prithvi 49,35; SpectralGPT 48,40. Yüzde 10 etikette UNet 46,77'ye düşüyor, Scale-MAE 56,06 ve DOFA 55,60 kalıyor. Temel modeller denetimli modelleri tutarlı biçimde geçmiyor. Bu görev RGB-only ve olaylar arası ayrım raporlanmıyor — [PANGAEA](https://arxiv.org/html/2412.04204)
- Prithvi-EO-2.0: 30 m HLS, altı bant, 300M parametre, Apache-2.0 — [model kartı](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
- Clay: GEO-Bench-2 onu 1-30 m olarak listeliyor; 8 yama boyutu sayesinde üst sıralarda ama yaklaşık 4 kat hesap maliyetiyle ([GEO-Bench-2](https://arxiv.org/html/2511.15658v2)). Ağırlıklar Apache-2.0 listeli ([Hugging Face](https://huggingface.co/made-with-clay/Clay)); eski v1.0 notları OpenRAIL-M diyordu ([v1.0 notları](https://clay-foundation.github.io/model/release-notes/specification.html))
- DOFA: dalga boyu koşullu, ViT-B 224 girdi, demo SAR dalga listesi Sentinel-1; kod MIT, ağırlık lisansı sayfada yok ([GitHub](https://github.com/zhu-xlab/DOFA)). GEO-Bench-2 1-30 m listeliyor ([kaynak](https://arxiv.org/html/2511.15658v2))
- CROMA: varsayılan 120x120, Sentinel-1 2 kanal ve Sentinel-2 12 kanal; kod MIT, ağırlık lisansı sayfada yok — [GitHub](https://github.com/antofuller/CROMA)
- Copernicus-FM: Sentinel görevleri, kod Apache-2.0 (bazı parçalar CC-BY-NC-4.0), ağırlıklar CC-BY-4.0; sayfada keyfi GSD desteği açıklanmıyor — [GitHub](https://github.com/zhu-xlab/Copernicus-FM)
- SatlasPretrain: Swin ve ResNet, Sentinel-2, Sentinel-1, Landsat ve 0,5-2 m hava görüntüsü; metadata Apache-2.0, metin ODC-BY diyor — [model kartı](https://huggingface.co/allenai/satlas-pretrain)

**Lisans: DINOv3**
- Kraliyetsiz, münhasır olmayan, devredilemez lisans. Dağıtımda sözleşme kopyası ve aynı sözleşme altında dağıtım, ürün veya web sitesinde "Built with DINOv3" ibaresi, yayınlarda atıf gerekli. ITAR kapsamı ile askeri veya savaş amaçları, nükleer uygulamalar, casusluk ve silah kullanımı yasak. Metin ticari kullanımı açıkça düzenlemiyor. Meta sözleşmeyi değiştirebilir, ihlalde feshedebilir — [DINOv3 Lisansı](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)

### Inferences

**Çözünürlük ve lisans özeti (benim sentezim)**

| Model | 0,3-1 m optik uyumu | Capella/Umbra SAR uyumu | Lisans | Girdi | Karar |
|---|---|---|---|---|---|
| DINOv3 ViT-L SAT | Yüksek (0,6 m Maxar) | Düşük (dondurulmuş) | Kısıtlı özel | 16'nın katı, 512 ile eğitildi | Optik birincil |
| DINOv3 ConvNeXt-L WEB | Yüksek (GEO-Bench-2) | Bilinmiyor | Kısıtlı özel | Değişken | Optik yedek |
| ImageNet ConvNeXt / Swin / PVT-v2 | Yüksek, BRIGHT'ta kanıtlı aile | Orta (eğitilirse) | Doğrulanmadı | Değişken | Temel çizgi |
| VMamba (ChangeMamba) | BRIGHT'ta en iyi temel modellerden | OEM-SAR'da U-Net ile benzer | Doğrulanmadı | Değişken | Temel çizgi |
| MaxViT | Bu görevde kanıt bulunamadı | Kanıt yok | Doğrulanmadı | Değişken | Önerilmez |
| MaRS SwinV2 | Orta | En yüksek (Umbra+Capella) | Çelişkili | 512-1024, SAR 1 kanal | SAR birincil adayı |
| DINOSAR ViT-B | - | Yüksek | Ağırlık beklemede | 224'te test edilmiş, 1 kanal | SAR yedek |
| SARATR-X / v2 | - | Düşük-orta (hedef odaklı) | NC / belirsiz | Değişken | Önerilmez |
| SARMAE | - | Düşük-orta | Belirsiz | Değişken | Yayın kontrolü gerekli |
| Prithvi, CROMA, Copernicus-FM | Düşük (10-30 m, çok bantlı) | Düşük (S1) | Açık | Bant uyumsuz | Eleyin |
| Clay, DOFA | Orta (1-30 m) | Düşük (S1 odaklı) | Çoğunlukla açık | Değişken | Yedeğin yedeği |
| SatlasPretrain Aerial | Orta (0,5-2 m) | S1 10 m, uygun değil | Çelişkili | Değişken | Yedek |

**Öneriler**

| Karar | Kanıt | Maliyet |
|---|---|---|
| Optik: DINOv3 ViT-L SAT-493M, SAT normalizasyonuyla; lisans engel olursa ConvNeXt-L veya PVT-v2 ImageNet | Orta (alt-10 m kıyaslar; BRIGHT olaylar arası testi yok) | 0 ek koşu, seçim bir kilit kararı |
| Optik: tam dondurma yerine PEFT veya çok düşük öğrenme oranı | Zayıf-orta, kaynaklar çelişiyor | Yüzde 0-20 ek süre |
| SAR: MaRS ve DINOSAR adaylarını ImageNet ön eğitimine karşı event-holdout üzerinde kısa sonda ile tara; net kazanç yoksa ImageNet'te kal | Zayıf (kazanç hipotez; şampiyonlar ImageNet kullandı) | Kısa sonda: yaklaşık 0,1 koşu aday başına |
| SAR kodlayıcı her zaman eğitilsin, dondurulmuş DINO kullanılmasın | Orta-zayıf (ATR analojisi, tutarlı) | Fark yok |
| ViT-7B ile eğitme | Hesaba dayalı | Parametreler tek başına yaklaşık 26,9 GB (32 bit); gradyanlarla 40 GB'ı aşar |

- ViT-L 300M yaklaşık 1,2 GB, ConvNeXt-L 198M yaklaşık 0,8 GB parametre belleğidir (32 bit, hesabım). 1024 girişte ViT/16 yaklaşık 4.096 token işler; gerçek tepe VRAM ölçülmedi. Eğitimden önce 1 toplu duman testi şart. Kanıt: 24 GB ile PVT-v2 1024 çalışmış ([Li ve ark.](https://arxiv.org/pdf/2505.04941)); BRIGHT Challenge birincisi 1024'te toplu boyutu 4 ve 2 kullanmış ([rapor](https://arxiv.org/html/2607.22746)).
- SAT-493M ile BRIGHT olay-öncesi optik görüntü (Maxar Open Data, [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/)) aynı sağlayıcıdan. Bu, kapsam uyumu açısından güçlü bir akıl yürütmedir, ölçülmüş bir kazanç değildir.
- **Doğrulama sızıntısı riski:** MaRS ve DINOSAR kendi verilerinde Umbra ve Capella afet sahnelerini görmüş olabilir. MaRS verisi 30 Aralık 2024'e kadar. BRIGHT'ın 2025 öncesi olaylarında event-holdout skoru iyimser olabilir. 2025 olaylarında bu sorun MaRS için tarih gereği beklenmez. AFRL aynı tür transdüktif sorunu kendi modelleri için açıkça işaret ediyor ([AFRL](https://arxiv.org/html/2509.21722v1)).
- BRIGHT silahlı çatışma olayları içeriyor ([BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/)). DINOv3 lisansındaki askeri kullanım yasağı ile etkileşimi hukuki bir sorudur; ürün veya müşteri savunma bağlantılıysa danışılmalı. Bu hukuki görüş değildir.
- MaRS "akademik" ifadesi çözülmeden ticari veya yarışma dışı ürüne konmamalı.

### Gaps
- DINOv3 SAT, MaRS, DINOSAR, SARATR-X-v2 veya SARMAE için BRIGHT event-holdout veya başka görülmemiş olay üzerinde sonuç bulamadım. Kodlayıcı seçiminin görülmemiş olay etkisi doğrudan ölçülmemiş.
- MaxViT için bu görevde hiçbir sonuç bulamadım. Swin, ConvNeXt, VMamba için aynı protokolde cross-event karşılaştırma yok.
- MaRS AAAI makalesine erişemedim (erişim engeli). DFC25-T2 tablosu ve karşılaştırma modelleri doğrulanmadı.
- DINOSAR makalesi, DINOSAR segmentasyon sonuçları ve SARATR-X-v2 ağırlık lisansı doğrulanmadı.
- DINOv2 lisansı (Apache-2.0 olduğunu biliyorum) bu oturumda birincil kaynaktan doğrulanmadı. ImageNet Swin, ConvNeXt, MaxViT, VMamba ağırlık lisansları da kontrol edilmedi.
- Modellerin 1024 girişte gerçek VRAM tüketimi ölçülmedi; herhangi bir kaynakta bu görev için tablo bulamadım.
- ChangeMamba'nın VMamba omurgası kullandığı genel bilgidir, okunan bölümlerde ayrıca doğrulanmadı.
- BRIGHT Challenge 1. ve 2. takım kodları okunamadı: [ilk repo](https://github.com/huang-he99/BightSolution) 404 döndü, [ikinci repo](https://github.com/whf-68/Damage-Aware-SAR-Optical-Query-Learning-framework) boş göründü. Omurga adları doğrulanmadı.

## SAR işleme: benek giderme, log ölçek, normalizasyon, geometri ve sensör kayması

### Takeaway
Görülmemiş olaya aktarımda benek gidermenin, belirli bir dB aralığının veya Capella/Umbra uyumlamasının yardım ettiğini gösteren kontrollü çalışma bulamadım. Savunulabilir varsayılan: benek giderme yapmamak, log benzeri ölçek ve görüntü başına dayanıklı normalizasyon, benek ve parlaklık artırımı. SAR tarafındaki kanıt zayıftır, bu nedenle bu kararlar ucuz ve geri alınabilir tutulmalıdır.

### Cited Findings
- BRIGHT SAR verisi VV veya HH genlik, arazi düzeltmeli. Sağlayıcının 8-bit ürünü varsa o, yoksa sağlayıcının önerdiği dönüşüm kullanılmış. Spotlight tercih edilmiş, Stripmap da var. Tek polarizasyon — [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/)
- BRIGHT yazarları SAR hasar imzasının olay ve afet türüne göre değiştiğini, olası nedenlerin kentsel doku, yapı malzemesi ve sensör geliş açısı olduğunu söylüyor. Makalenin okunan bölümlerinde benek, eğik açı veya normalizasyon önerisi yok ve Capella/Umbra karşılaştırması yok — [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/)
- BRIGHT Challenge test olaylarının hem optik hem SAR değer dağılımları eğitimden farklı; olaylar sensör, mevsim ve coğrafyada ayrışıyor. Rapor SAR ön işlemesini tarif etmiyor — [Bright Challenge raporu](https://arxiv.org/html/2607.22746)
- Capella GEO ve GEC piksel değerleri sigma-sıfır cinsinden kalibre genliktir; SLC beta-sıfırdır; sigma = beta x sin(geliş açısı), elipsoide göre. Spotlight Ultra GEO azimutta 5 kez çok-bakışlı — [Capella ürün kılavuzu](https://support.capellaspace.com/hc/en-us/article_attachments/30675126547092). Dijital sayıdan sigma-sıfıra sabit ölçek faktörü ve yardımcı üründe sabit bir geliş açısı var — [Esri Capella GEO](https://pro.arcgis.com/en/pro-app/3.3/help/analysis/image-analyst/capella-geo-workflow.htm)
- Umbra sigma-sıfır, GEC GeoTIFF içindeki ölçek faktörlerinden türetilir ve dB'dir; GEC sabit yükseklikli elipsoid yüzeyine projekte edilir, yani arazi düzeltmesi Capella GEO ile aynı değildir. NASA hizmetlerinin etiketleri Gamma0 ve sigma arasında tutarsız — [NASA Umbra metadata](https://gis.earthdata.nasa.gov/gis05/rest/services/DISASTERS_202604_TYPHON_SINLAKU/202606_umbra/ImageServer/info/iteminfo). Umbra veri CC BY 4.0 — [AWS kaydı](https://registry.opendata.aws/umbra-open-data/)
- Yalnız olay-sonrası Capella ile bina düzeyi deprem hasarı: bakış açıları kabaca 24-49 derece, 1x9 çok-bakış, doku özellikleri ve Random Forest. Görülmemiş bölgede genel doğruluk en iyi modellerde yüzde 53-72. Layover, çift yansıma, çatı ve gölge bölgeleri bina imzasını ve dolayısıyla hasar yorumunu belirliyor. Derin öğrenme yok, ama aynı sensör ailesi — [Macchiarulo ve ark. 2024](https://doi.org/10.1007/s10518-024-01877-1)
- Log dönüşüm dinamik aralığı sıkıştırır ve benek varyansını dengeler; yoksa büyük değerler kırpılır. Bu, restorasyon için bir derleme bulgusudur, hasar segmentasyonu için değil — [Restorasyon derlemesi](https://arxiv.org/html/2101.11852)
- Benek giderme ön işlemi için doğrudan kontrollü segmentasyon ablasyonu bulamadım. En yakın örnek: düşük çözünürlüklü Yellow River değişim veri setinde derin benek giderme DDNet doğruluğunu 86,65'ten 90,79'a, LANTNet'i 87,16'dan 91,1'e çıkardı (yalnızca arama özeti) — [Reading eprint](https://reading-9.eprints-hosting.org/114518/1/Deep%20Despeckling%20of%20SAR%20Images%20to%20Improve%20Change%20Detection%20Performance.pdf)
- SARATR-X-v2 piksel yeniden yapılandırmanın lekeye bağlı kaldığını savunuyor, sentetik benek değişiminde hedef sapmasını büyük oranda azaltıyor (segmentasyon değil, denetim sinyali ölçümü) — [SARATR-X-v2](https://arxiv.org/pdf/2607.23238). SARMAE, gamma çok-bakış modeliyle yüzde 50 olasılıkla ek bozulma uyguluyor — [SARMAE](https://arxiv.org/html/2512.16635v1)
- AFRL tek kanalı üçe kopyalayıp artırımda gürültü ve bulanıklık kullanıyor — [AFRL](https://arxiv.org/html/2509.21722v1). DINOSAR gri tonlama, [0,1] tensör ve global ortalama ile sapma kullanıyor; kart bu istatistiklerin ham SAR için değil hazırlanmış görüntü için olduğunu belirtiyor — [DINOSAR kartı](https://huggingface.co/YTang/DINOSAR)
- Sensör ve açı kayması: ATRNet-STAR'da görüntüleme açısı değişimi bant ve polarizasyondan daha etkili, yazarlar iyi bir yöntem bulmadıklarını söylüyor ([ATRNet-STAR](https://arxiv.org/pdf/2501.13354), yalnızca özet; ATR analojisi). Bir derleme geliş açısı, bant genişliği, çözünürlük ve ön işlemi kayma nedeni sayıyor ([Remote Sensing 2026](https://doi.org/10.3390/rs18152567)). SED-FOD, sensörler arası tam değişmezliği zorlamanın optimal olmayabileceğini savunuyor ([SED-FOD](https://arxiv.org/pdf/2608.18755), özet)
- BRIGHT'ta ortalama kayıt hatası yaklaşık 1 piksel, optik-SAR otomatik kayıt dağlık alanda zorlanıyor, FLSS ve HOPC daha kararlı — [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/)

### Inferences

| Karar | Kanıt | Maliyet |
|---|---|---|
| Ön işlem olarak benek giderme yapma; SAR ürünleri zaten çok-bakışlı, ince saçılım hasarı taşıyabilir | Zayıf (doğrudan test yok) | 0 koşu |
| Çarpımsal log-normal veya gamma benek artırımı | Zayıf (ön eğitim literatüründen analoji) | Yüzde 0-5 süre |
| BRIGHT 8-bit veride görüntü başına dayanıklı normalizasyon (örneğin yüzdelik 1-99, ardından ortalama/sapma) | Zayıf; dağılım farkı gözlemi var, kazanç testi yok | 0 koşu |
| Kalibre dB'yi yalnızca ham GeoTIFF'lere geri dönülebiliyorsa düşün | Yok | Veri işleme süresi, bir koşuya gerek yok |
| Geliş açısını kanal veya koşul olarak eklemeyi temel plana alma; önce açı meta verisi ile hata korelasyonunu ölç | Zayıf | 0 koşu ölçüm; koşullama 1 koşu |
| Çevirme ve 90 derece döndürmeyi koru, optik ve SAR'a aynı dönüşümü uygula: kuzey-yukarı ürünlerde bakış yönü sahneler arasında değiştiği için ([Macchiarulo](https://doi.org/10.1007/s10518-024-01877-1): sağ ve sol bakış, artan ve azalan yörünge) gölge ve layover yönü çeşitliliği sağlayabilir | Yok (fiziksel akıl yürütme); [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/) bu artırımları kullanıyor ama ablasyon yok | 0 koşu |

- Spesifik bir dB kırpma aralığı önermiyorum, çünkü VHR Capella/Umbra için doğrulanmış aralık bulamadım ve BRIGHT'taki 8-bit ürünler zaten bir germe içeriyor. Aralık, olay başına histogramlardan seçilmeli.
- BRIGHT 8-bit ürünleri sağlayıcılar arasında aynı radyometrik anlama sahip olmayabilir. Bu nedenle Capella ve Umbra için ayrı histogram raporu, herhangi bir eğitimden önce yapılacak ücretsiz ve yüksek değerli bir kontroldür.
- Macchiarulo'nun 53-72 aralığı, klasik öznitelikli Random Forest için geçerlidir; derin model beklentisine doğrudan aktarılamaz.
- Gerçekçi iddia: SAR işleme kararı tek başına farkı yaratmayabilir; karar sensör ve event başına hata raporu ile doğrulanmalı.

### Gaps
- Aynı VHR hasar görevinde ham, despeckled ve log girdi karşılaştıran kontrollü çalışma bulamadım.
- Capella ve Umbra arasında kayma ölçen çalışma bulamadım. BRIGHT'ta olay-sensör eşlemesi tam olarak doğrulanmadı.
- BRIGHT 8-bit ürünlerinin doğrusal mı, dB mi olduğu, hangi germeyi içerdiği doğrulanmadı.
- Geliş açısı, layover ve gölge için hasar segmentasyonunda derin model ablasyonu bulamadım; Umbra ürün kılavuzu okunmadı.
- Şampiyonların SAR ön işlemesi bilinmiyor; raporda yok, kodları okunamadı.

## Füzyon: erken, geç, çapraz dikkat, ağırlık paylaşımı ve modalite düşürme

### Takeaway
Görülmemiş olay altında en güçlü tutarlı mimari bulgu: modaliteye özgü kodlayıcılar ve geç veya kademeli füzyon erken füzyonu belirgin biçimde geçiyor; bina konumu ile hasar kararını ayırmak da yardım ediyor. Çapraz dikkatin geç füzyondan üstün olduğu görülmemiş olayda gösterilmedi. Modalite düşürme için hasar görevinde doğrudan kanıt yok ve yanlış uygulanırsa zarar verebilir.

### Cited Findings
- BRIGHT Challenge 1.'si test mAP ablasyonu: yalnız optik temel 0,0409; erken füzyon 0,0542; ikili aşamalı füzyon 0,1095; tam çözüm (sözde etiket ve Mask NMS) 0,1815. Testte ölçülmüş, ama sözde etiket ile NMS ayrıştırılmamış — [Bright Challenge raporu](https://arxiv.org/html/2607.22746)
- 2.'si doğrulamada: erken RGB+SAR 0,120, geç füzyonlu ikili omurga 0,264, yalnız SAR Mask R-CNN 0,002. Füzyon her FPN seviyesinde birleştirme ve 1x1 katman, optik geçişi olarak başlatılıyor; artık çapraz modal fark bloğu var. Takım, dört kanallı girdinin erken filtreleri modaliteleri karıştırmaya zorladığını savunuyor; optik geçiş ablasyonu ayrı değil — [rapor](https://arxiv.org/html/2607.22746)
- İki şampiyon da bağımsız olarak modaliteye özgü kodlayıcı, geç veya kademeli füzyon, optik baskın konum ile hasar ayrımına yöneldi — [rapor](https://arxiv.org/abs/2607.22746)
- BRIGHT sıfır-atışlı aktarımda ayrışık modeller doğrudan segmentasyondan daha iyi — [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/). DFC25 birincisi de iki aşamalı; binalar önce optikten, sonra bina öncülüyle değişim tespiti — [Li ve ark.](https://arxiv.org/pdf/2505.04941)
- Olay-sonrası modalite, beş olay, DamageFormer: yalnız optik 69,76; yalnız SAR 65,56; füzyon 70,79 mIoU. Mükemmel bina maskesiyle UNet Damaged IoU yüzde 35,83 (SAR) ve 44,83 (optik+SAR); DeepLabV3+ Damaged 39,63 ve 40,45 — [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/)
- Kontrollü füzyon karşılaştırması: Myanmar 2025, ResNet-18, 2 m, ikili sınıflandırma, EF, LF, ayak izi rehberli orta füzyon. Yazarlar orta füzyonun üstün olduğunu söylüyor, ama üç modalite F1 değerleri yaklaşık 0,30-0,36 ve standart sapmalar örtüşüyor (tablo PDF'ten okundu, sütun ataması kesin değil). Tek olay, 5 katlı çapraz doğrulama, olaylar arası değil — [Russo ve ark. 2026](https://isprs-annals.copernicus.org/articles/XI-3-2026/927/2026/isprs-annals-XI-3-2026-927-2026.pdf)
- Ağırlık paylaşımı: Benson ve Ecker paylaşmayan iki akışlı ResNet50 kullanıyor, ama paylaşımlı ile karşılaştırmıyor ([kaynak](https://ar5iv.labs.arxiv.org/html/2011.10328)). MaRS ayrı RGB ve SAR SwinV2 kullanıyor ve meta-modalite dikkati ekliyor ([MaRS arama özeti](https://ojs.aaai.org/index.php/AAAI/article/view/38153)). STSF-Net optikte SAM2, SAR'da küçük Swin kullanıyor ([STSF-Net](https://arxiv.org/html/2604.05527v1))
- Modalite düşürme: QDFNet optik-SAR nesne tespitinde rastgele modalite düşürme protokolü kullanıyor ([kaynak](https://arxiv.org/abs/2512.22447), özet). Ürün haritalamada geç (karar) füzyon eksik optik altında en dayanıklı, erken füzyon en kırılgan, zamansal düşürme yardım ediyor ([Garnot ve ark.](https://arxiv.org/pdf/2112.07558), özet; analoji)

### Inferences

| Karar | Kanıt | Maliyet |
|---|---|---|
| Erken füzyon kullanma; iki ayrı kodlayıcı (pseudo-siamese), geç veya kademeli füzyon | **Orta-güçlü** (iki bağımsız şampiyon, biri testte; yalnız iki test olayı) | 0 ek koşu |
| Konum ile hasarı ayır: optikten bina maskesi, hasar için optik+SAR | Orta (3 bağımsız kaynak, aynı alan) | Bir çok görev başlığı: yaklaşık yüzde 10-20; iki aşama: 2 model eğitimi |
| Füzyon katmanını optik geçişi olarak başlat ve FPN seviyesi başına basit birleştirme + artık SAR farkı ile başla | Zayıf-orta (tek takım, ablasyon yok) | 0 koşu |
| Çapraz dikkati yalnız düşük çözünürlüklü aşamalarda ve bütçe kalırsa ekle | Zayıf | VRAM artışı, olası +1 koşu |
| Rastgele modalite düşürmeyi varsayılan yapma; yalnız SAR kolunu optiksiz yardımcı başlıkla zorlamak daha mantıklı | Yok, akıl yürütme | Yardımcı başlık için yüzde 5-10 |
| Bina düzeyinde sonradan oylama ve dış ayak izi varsa kullanma | Zayıf-orta ([BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/): nesne tabanlı oylama olay düzeyinde +0,69, küme düzeyinde -0,41) | 0 koşu |

- SAR'ı düşürmek hasar etiketini çoğu zaman öğrenilemez kılar, çünkü olay-öncesi optikte hasar bilgisi yoktur. Bu yüzden hasar başlığı için düşürme etiket gürültüsü gibi davranır. Optiği düşürmek ise yalnız SAR konumlandırmasının çok zayıf olduğunu gösteren 0,002 mAP ile uyumsuz. Küçük bir olasılıkla (ve yalnızca bir yardımcı öğrenme hedefi olarak) düşürme hipotezi test edilebilir, ama önce kilitlenmemeli.
- Optik dengesizliği riski: geç füzyon optik başlatmayla SAR'ı yok sayabilir. Eğitim sırasında SAR'a özgü yardımcı kayıp veya gradyan izlemesi ucuz bir önlemdir. Bu da hipotezdir.
- 1. takımın 0,0409 temel değeri ile resmi 0,021 temel çizgisi farklı kurulumlar, doğrudan karşılaştırılmamalı.
- 2.'nin 0,120/0,264 değerleri doğrulama, 1.'nin 0,0542/0,1095 değerleri test; aynı yönü göstermeleri güven verir, büyüklükler karşılaştırılamaz.

### Gaps
- BRIGHT olay-dışı (event-holdout) üzerinde paylaşımlı ile pseudo-siamese karşılaştıran çalışma bulamadım.
- Çapraz dikkat ile geç füzyonu aynı omurga ve olay-dışı protokolde kıyaslayan çalışma bulamadım.
- Hasar görevinde modalite düşürmenin etkisini gösteren çalışma bulamadım.
- 1. takımın Cross-Layer Attention ayrıntıları ve şampiyon kodları okunamadı.

## Segmentasyon için alan genellemesi: stil, normalizasyon, SAM, EMA ve kayıplar

### Takeaway
Etiketli hedef olmadan büyük DG kazancı beklenmemeli. Bu görevde iki şey tutarlı biçimde en büyük farkı yarattı: olay dışı doğrulama ile model seçimi ve hedefe uyarlanmış ayarlar (eşikler, sözde etiket). Ağırlık ortalaması ucuz ve en iyi desteklenen düzenleyicidir; stil ve Fourier yöntemleri küçük ve tutarsız kazanç gösteriyor; SAM ile MixStyle için hasar görevinde kanıt yok.

### Cited Findings

**Olumsuz ve temel çizgi sonuçları**
- DomainBed: dikkatle ayarlanmış ERM ortalamada en iyi; aynı koşullarda hiçbir algoritma ERM'yi bir puandan fazla geçmiyor; model seçimi yöntemin parçasıdır — [Gulrajani ve Lopez-Paz](https://arxiv.org/pdf/2007.01434). Sınıflandırma, analoji. İkincil özet oracle 68,9, eğitim alanı doğrulama 67,0, bir-alan-dışı 63,8 veriyor ([özet](https://sotaverified.org/papers/in-search-of-lost-domain-generalization))
- BRIGHT sıfır-atışlı UDA (DeepLabV3+ 35,60): FDA +0,69, AdaptSeg +0,45, CLAN +0,41, AdvEnt önemsiz, PyCDA -2,41. Yazarlar kazançları küçük ve olaylar arasında tutarsız buluyor. Bir-atışlı SSL (38,82): Mean Teacher +1,18, CCT +0,67, GCT +0,52, CPS gelişme yok. Etiketsiz değişim tespiti diğer veri setlerinde F1 70-85, BRIGHT'ta yaklaşık 20 — [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/)
- Çok etiketli sahne sınıflandırmada ERM 66,5, en iyi stil artırımı 71,5, alan-içi oracle 95,4; yazarlar kalan farkı GSD, sensör ve sınıf istatistiklerine bağlıyor — [Label-Decoupled Style](https://arxiv.org/pdf/2607.12704) (özet, analoji)

**Doğrudan hasar kanıtı**
- BRIGHT Challenge: holdout liderleri testte yüzde 70-86 düştü; alan-içi skor aktarımı öngörmüyor. 2.'nin doğrulaması standart bölünmede 0,2670'ten event-holdout'ta 0,1482'ye düştü. Rapor tek-olay-dışı doğrulama öneriyor. Yalnız iki test olayı var — [Bright Challenge raporu](https://arxiv.org/html/2607.22746)
- Hasar sınıfları kararsız: test yangın olayında 7.321 örneğin 4.729'u yıkık, kasırga olayında 6.063'ün 3.307'si hasarlı. Birinci takım tek 0,1 üzeri wildfire skorunu aldı. Hasarlı sınıfı alan-içinde en zayıf, test liderlerinde en güçlü — [rapor](https://arxiv.org/html/2607.22746)
- Ters yönde: DFC25 şampiyonunun geliştirme aşamasında Damaged IoU 45,321, Destroyed 74,637, test aşamasında Damaged 9,748, Destroyed 42,086, Intact 67,304 (mIoU 74,44'ten 54,28'e) — [Li ve ark.](https://arxiv.org/pdf/2505.04941)
- Sahne-farkında uyarlama: sahne sınıflandırıcı, yangın için yıkık eşiğini düşürüp hasarlı eşiğini yükseltme, felaket-farkında sözde etiket. Karışık veri temel çizgisi 0,0567, tam sistem 0,1815 (sahne katkısı ayrıştırılmamış). Sözde etiketin doğrulama yanlılığı riski ölçülmemiş — [rapor](https://arxiv.org/html/2607.22746)
- DFC25 şampiyonu test görüntülerinde topluluk+TTA ile sözde etiket, normalleştirilmiş entropi 0,3 altı için hard CE, üstü için KL (2:1) — [Li ve ark.](https://arxiv.org/pdf/2505.04941)
- xBD, Benson ve Ecker: iki akışlı ResNet50 IID 0,74, OOD 0,60; Dual-HRNet 0,73 ve 0,67. Gupta bölünmesinde iki akışlı 0,44 iken SWA + çok-alanlı AdaBN ile 0,59; OOD-xBD'de yaklaşık +0,05. Dual-HRNet'te AdaBN 0,67'den 0,69'a, SWA tek başına yardım etmedi (0 ve -0,01). Tüm birini-dışarıda bırakma pahalı olduğu için üç kat kullanıyorlar — [Benson ve Ecker](https://ar5iv.labs.arxiv.org/html/2011.10328)
- Bilinen hedef etiketle ince ayar (SDA) yeni olayda belirleyici olabiliyor, ama hedef etiketi gerekiyor — [2603.14694](https://arxiv.org/abs/2603.14694) (özet)
- BRIGHT'ta kaynak doğrulamayla model seçimi sıfır-atışlı durumda büyük fark bırakıyor, bir etiketli hedef çifti farkı daraltıyor ama kapatmıyor. En zayıf aktarım olayları Acapulco-HC-2023 ve Ukraine-AC-2022. Bir-atışlı ortalama mIoU ChangeMamba için 43,23, sıfır-atışlı 39,38 — [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/)

**Uzaktan algılama DG yöntemleri**
- CrossEarth, DINOv2 + stil enjeksiyonu + MIM. Rein'e göre: Potsdam→Vaihingen +7,2; V→P +0,6; LoveDA R→U +0,2; U→R +1,0; D2M +0,8. CASID ablasyonunda stil eklemek Sub2Tem 45,5→47,8, Sub2Tms 64,4→61,5, Sub2Trf 61,9→63,2. Yazarlar temel modellerin alan dışında DA yöntemlerini tutarlı biçimde geçmediğini söylüyor. Tam ERM Mask2Former satırı yok — [CrossEarth](https://arxiv.org/html/2410.22629v2)
- Earth-Adapter: LoRA tipi yöntemler uzaktan algılamada bozulabiliyor; artefakt iddiası yazarların yorumu; DG ortalama Rein +3,1 — [Earth-Adapter](https://arxiv.org/html/2504.06220v4)

**Ağırlık ortalaması, SAM, MixStyle**
- EoA: DomainBed'de ResNet-50 68,0 (ERM'den yaklaşık 4 puan yüksek), RegNetY-16GF 76,6 (yaklaşık 6). Hareketli ortalama alan-içi doğrulamayı dış performansın daha iyi göstergesi yapıyor ve ortalamalı modelleri topluluk yapmak daha iyi — [Arpit ve ark.](https://arxiv.org/abs/2110.10832). Sınıflandırma, analoji
- SWAD yoğun ağırlık ortalamasıyla DomainBed'de düz minimum arıyor; SWA ve SAM alan-dışı doğruluğu iyileştirirken Mixup ve CutMix iyileştirmiyor — [SWAD](https://arxiv.org/pdf/2102.08604). DGSAM, global SAM'ın sahte düz minimumlar bulabildiğini savunuyor (ikincil özet) — [özet](https://themoonlight.io/review/dgsam-domain-generalization-via-individual-sharpness-aware-minimization)
- MixStyle sınıflandırma, örnek eşleme ve RL üzerinde değerlendirilmiş; düşük katman CNN öznitelik istatistiklerine çalışıyor — [Zhou ve ark.](https://arxiv.org/abs/2104.02008)

**Kayıp ve dengesizlik**
- BRIGHT piksel oranları: Intact yüzde 82,8, Damaged 10,7, Destroyed 6,5; arka plan-bina 7:1. Temel ayar CE + Lovasz softmax — [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/)
- 2.'nin önceliğe dayalı varyantı Damaged AP'yi 0,105'e çıkarırken Destroyed AP'yi 0,409'dan 0,302'ye düşürdü. Hasarlı hard-negative event-holdout'ta 0,064, önerilen model 0,148, standart Mask R-CNN 0,049 — [rapor](https://arxiv.org/html/2607.22746)
- xBD'de öğrenci projesi Dice+focal'ı en iyi buldu (zayıf kaynak, olay dışı değil) — [Stanford](https://web.stanford.edu/~markcx/sample-project/IISE_Building_Damage_Detection_from_Satellite_Imagery.pdf). MambaBDA'ya CE ve Lovasz yanına focal eklenmiş (arama özeti) — [2603.01116](https://arxiv.org/html/2603.01116v2)

### Inferences

| Karar | Kanıt | Maliyet |
|---|---|---|
| Model seçimini event-holdout ile yap; 2-3 çeşitli olay (biri deprem, biri yangın veya kasırga) | **Güçlü** (doğrudan iki bağımsız kaynak, DomainBed ile uyumlu) | 2-3 koşu ama aynı koşular topluluk üyesi olarak yeniden kullanılır |
| Rastgele bölünme skorunu genelleme göstergesi sayma | Güçlü | 0 |
| EMA veya SWA, EMA ağırlıklarını kaydet | Orta (Benson'da bu görevde karışık; EoA ve SWAD analoji) | Yaklaşık 0 |
| Temel artırım + güçlü optik fotometrik; ağır stil yöntemi kilitleme | Zayıf-orta | 0-yüzde 5 |
| Kaynak olaylar arası FDA veya stil artırımı isteğe bağlı | Zayıf; BRIGHT'ta +0,69, tutarsız | Yüzde 0-10 |
| MixStyle ve IBN bu görevde önerilmez, DINO/Swin LayerNorm kullanıyor | Yok | Atla |
| SAM ana eğitimde kullanılmasın | Zayıf, sınıflandırma analojisi | Genel bilgiye göre adım başına iki gradyan, yaklaşık 2 kat süre: birkaç koşu bütçesine pahalı |
| Çok-alanlı AdaBN yalnızca BN'li kolda (ör. ResNet SAR) düşünülsün; ViT, ConvNeXt, Swin için geçerli değil | Orta (doğrudan görev, CNN) | Küme örnekleme değişikliği; yaklaşık 0 |
| CE + Lovasz + hafif sınıf ağırlığı; agresif ağırlıklandırma veya hard-negative yok; sınıf eşiği olay dışı doğrulamadan | Zayıf-orta | 0 |
| Etiketsiz test görüntüleri yasal ve erişilebilirse: 1 tur güven filtreli sözde etiket | Orta (iki şampiyonda, ablasyon ayrışmamış; onay yanlılığı riski) | 1 koşu (ince ayar) |

- Model seçimi kanıtın en güçlü olduğu nokta; 0,346 korelasyon tek başına alan-içi metriği kullanılmaz yapıyor.
- Hasarlı sınıf için iki kaynak ters yönde görünüyor (Challenge test liderlerinde en güçlü, DFC25 testinde 9,75). Bunun en makul açıklaması olay karışımıdır (Challenge kasırga olayında hasarlı baskın), yani sınıf önceliği kayması. Bu nedenle tek bir olay dışı skor raporlanmamalı; olay başına tablo şart.
- Hedef sınıf önceliği bilinmez: sabit eşik riskli. Afet türü dışarıdan biliniyorsa (operasyonda bilinir) 2-3 afet türü için hazır eşik tablosu ucuz bir uyarlama yoludur. Bir sınıflandırıcı eğitmek gerekmez.
- İncelenen düzenleyici ve stil yöntemlerinin hiçbiri küçük ek kazançların ötesinde tutarlı sonuç vermiyor. Bu nedenle bütçeyi yeni düzenleyicilere değil, doğrulama, füzyon mimarisi ve topluluğa harcamak daha iyi.
- Çok koşu gerektiren yöntemler: DomainBed tarzı hiperparametre araması, tam birini-dışarıda, CrossEarth'ün bütün ablasyonları ve GEO-Bench-2'nin önerdiği hiperparametre seçimi birkaç koşu bütçesine uymaz. Benson'un üç katı uygun bir emsal.

### Gaps
- BRIGHT olay dışında MixStyle, IBN, SAM, EMA veya FDA'yı (DG kipinde) ERM'ye karşı ölçen çalışma bulamadım.
- Hasar için bir DomainBed benzeri ERM karşılaştırması yok; en yakını BRIGHT UDA tablosu.
- IBN-Net özgün makalesi okunmadı; uzaktan algılama veya SAR kanıtı aramada çıkmadı.
- Uzaktan algılama için SAM ile segmentasyon DG sonucu bulamadım. SAM'ın iki gradyanlık maliyeti genel bilgidir, bu oturumda birincil kaynaktan okunmadı.
- Focal, Dice ve Lovasz'ı olay dışı protokolde kıyaslayan çalışma bulamadım.
- BRIGHT olay-dışı DG tablosu F1 ve G1 ekleri ile Şekil 14 olay başına değerleri okunamadı.

## Hedef etiketi olmadan belirsizlik ve kalibrasyon

### Takeaway
Hiçbir yöntem alan kaymasında güveni korumuyor. Sıcaklık ölçekleme ucuzdur ama kayma altında az yardım eder; topluluklar en güvenilir seçimdir ve TTA en iyi hafif alternatiftir. Ek eğitim gerektirmeyen en ucuz paket: TTA + maksimum softmax/entropi + bina düzeyinde toplama + event-holdout üzerinde sıcaklık ve risk-kapsam değerlendirmesi. Hedef kalibrasyonu etiket olmadan doğrulanamaz; yalnız dışarıda tutulmuş olaylarla tahmin edilebilir.

### Cited Findings
- Ovadia ve ark.: belirsizlik kalitesi tüm yöntemlerde artan kaymayla bozuluyor. Doğrulamada kalibrasyon kayma altında kalibrasyonu garanti etmiyor; sıcaklık ölçeklemenin ECE'si kaymayla artıyor. Topluluklar en iyi ve en dayanıklı, M=5 kazançların çoğunu veriyor. Dropout sıcaklık ölçeklemeyi geçiyor. ECE uygun bir skor kuralı değil, Brier farkları daha net gösteriyor. Görüntü sınıflandırma ve diğer görevler, analoji — [Ovadia ve ark. 2019](https://ar5iv.arxiv.org/html/1906.02530)
- Segmentasyon güvenilirliği: Cityscapes'ten ACDC'ye mIoU hatası yaklaşık 2-3 kat, kalibrasyon hatası yaklaşık 40 kat artıyor. Sıcaklık ölçekleme iyileştiriyor ama alan-içi düzeye indirmiyor. Local TS en iyi; hedef görüntüsü olmadan küme temelli TS sınırlı. Yeni modeller daha dayanıklı ama belirsizlikte daha iyi değil; transformer ve CNN arasında net kazanan yok — [Reliability in Semantic Segmentation](https://ar5iv.labs.arxiv.org/html/2303.11298). Sürüş verisi, analoji
- On Calibrating Semantic Segmentation Models tablosunda BDD100K'dan CityScapes'e SegFormer-B5 ECE: kalibresiz 0,087; TS 0,083; topluluk 0,087; Local TS 0,085; Meta-Cal 0,076; Selective 0,069. ADE20K'dan COCO'ya TS 0,467→0,466. Yani TS bu kaymada neredeyse sıfır kazanç — [Wang ve ark.](https://arxiv.org/pdf/2212.12053.pdf). Tablo PDF metninden okundu. Selective'in hedef verisine gereksinimi tam doğrulanmadı
- ValUES: topluluklar genel olarak en sağlam, TTA hafif ve uygun alternatif. Görüntü düzeyi başarısızlık tespitinde AURC kullanılıyor; topluluk en iyi, TTA yakın. Kalibrasyonda (ACE) tahmin belirsizliği en iyi; MC dropout (TTD) ortalama ya da üstü. Toplama yöntemi kritik ve veri setine bağımlı; alt görev için ucuz olduğundan farklı toplamaları sınamayı öneriyorlar. Tıbbi ve sürüş verisi, analoji — [ValUES](https://arxiv.org/pdf/2401.08501)
- Jaeger ve ark.: yaygın başarısızlık tespit yöntemlerinin hiçbiri softmax temel çizgisini gerçekçi hata kaynaklarında geçemiyor; AURC birincil metrik öneriliyor; sıralama için logitler 64 bit olmalı ya da sıcaklık ölçeklenmeli — [Jaeger ve ark.](https://arxiv.org/pdf/2211.15259). Görüntü sınıflandırma, analoji
- Evidential: EDL'nin öğrenilen belirsizliği sonsuz veride bile kaybolmuyor ve aşağı akış başarısı aslında bir enerji temelli OOD dedektörü gibi. Önerdikleri düzeltme model belirsizliği eklemek, yani ek hesap — [Shen ve ark. NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3177be226ee12e34d6ba3b5e6fe6a5b-Abstract.html)
- BRIGHT sonrası işlem: TTA +0,87 küme ve +0,56 olay; topluluk +0,82 ve +0,75; hepsi +1,23 ve +0,92 mIoU. Topluluk boyutu okunan metinde yok — [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/). Kalibrasyon ölçülmemiş
- DFC25 şampiyonu topluluk+yatay çevirme TTA ile olasılık üretip normalleştirilmiş entropi 0,3 eşiğini sözde etiket güvenilirliği için kullandı; belirsizlik bina sınırlarında yoğunlaştı — [Li ve ark.](https://arxiv.org/pdf/2505.04941)
- Bu alandaki adaylar okunmadı: BayeSiamMTL xBD'de varyasyonel MC ile alan kaymasını inceliyor ([kayıt](https://www.space4water.org/bibcite/reference/488)); CNES/Thales posteri MC dropout, topluluk, Bayes by Backprop ve TTA'yı coğrafi kaymada karşılaştırıyor ama yalnız özet ([poster](https://airdrive.eventsair.com/eventsairwesteuprod/production-nikal-public/1ede2abfad2d4b03bf63df725bf9e770))

### Inferences

| Yöntem | Kanıt | Kaymada beklenti | Maliyet |
|---|---|---|---|
| MSR/entropi + bina düzeyi toplama | Orta (Jaeger, ValUES; analoji) | Sıralama için güçlü temel | 0 koşu |
| Sıcaklık ölçekleme, olay dışı | Orta; çok sayıda kaynak: kaymada az kazanç | Küçük iyileşme; ama zarar riski de var (Ovadia Criteo Brier) | 0 koşu, tek parametre |
| TTA (4-8 görünüm) | Orta (ValUES, BRIGHT doğruluk kazancı) | Topluluğa yakın, ucuz | 0 koşu; çıkarım 4-8 kat |
| EMA kontrol noktaları | Zayıf | Çok az çeşitlilik | 0 koşu, depolama |
| Derin topluluk (3-5) | Güçlü (genel), bu görevde ölçülmemiş | En iyi | 3-5 koşu ama olay dışı koşularla paylaşılır; çıkarım N kat; VRAM az |
| MC Dropout | Zayıf-orta | Topluluktan zayıf | 0 koşu ancak eğitimden önce dekodere dropout konmalı; çıkarım N kat |
| Evidential | Olumsuz | Güvenilmez epistemik | Kayıp değişikliği, doğruluk riski; önerilmez |

**En ucuz paket (ek eğitim yok):** Olay dışı modeli al; TTA; entropi ya da 1-MSR; footprint üstünde ortalama; sıcaklığı dışarıda tutulan olaylarda seç; AURC ile raporla.

**Bina düzeyinde ölçüm reçetesi (benim önerim):**
- Bina olasılığı: pikseller üzerinde logit veya olasılık ortalaması ayak izi içinde. Alternatif olarak bina sınırı halkasını dışarıda bırakma, çünkü belirsizlik sınırda toplanıyor ve kayıt hatası yaklaşık 1 piksel.
- Olay başına ayrı ve eşit kütleli (adaptif) ECE: sınıf bazında, bina başına 10-15 kutu. Küçük olaylarda bootstrap güven aralığı. ECE tek başına kullanılmamalı, Brier ve NLL ile birlikte.
- Risk-kapsam: binaları güven sırasına diz, kapsam arttıkça riski (1-doğruluk ya da 1-makro F1) çiz; AURC ve E-AURC raporla, `coverage@risk` ekle. Sınıf öncelikleri kaydığı için dengeli risk de raporlanmalı.
- Sıralama metriğini sınıf koşullu ver: Destroyed ve Damaged için ayrı. Sadece Intact binalardan oluşan bir kapsam yanıltıcıdır.
- Tek sayı verme: olay başına aralık göster. Etiketsiz gerçek hedefte yalnız güven dağılımını, entropiyi, modeller arası anlaşmazlığı ve tahmin sınıf oranlarını izle. Kalibrasyon iddiası yapma.

- Hedefte sınıf önceliği kaymasını gösteren sayılar (yıkık yüzde 65 ve hasarlı yüzde 55) bu görevde sabit kalibrasyonun neden zayıf kalabileceğini açıklar. Bu benim çıkarımımdır.
- Olay dışı koşuları topluluk üyesi yapmak: her biri farklı olayı dışarıda tuttuğundan çeşitlilik sağlar, ama her üye verinin bir kısmını kaybeder. Son modelin tüm veriyle yeniden eğitilmesi başka bir koşu gerektirir. Bütçeye göre seçilmeli.

**Toplu kilit listesi (eğitimden önce)**

| Sıra | Kilit | Kanıt | Maliyet |
|---|---|---|---|
| 1 | Event-holdout doğrulama ve olay başına metrik | Güçlü | 2-3 koşu (yeniden kullanılır) |
| 2 | Ayrı kodlayıcı + geç/kademeli füzyon, erken füzyon yok | Orta-güçlü | 0 |
| 3 | Bina konumu ve hasar başlığını ayır | Orta | Yüzde 10-20 |
| 4 | Optik: DINOv3 SAT ViT-L (lisans uygunsa) + PEFT/düşük LR; yedek ConvNeXt/PVT | Orta | 0 |
| 5 | SAR: eğitilir kodlayıcı, MaRS veya DINOSAR için kısa sonda, lisans onayı; yedek ImageNet | Zayıf | Yaklaşık 0,1 koşu/aday |
| 6 | Benek giderme yok; log/dayanıklı normalizasyon; benek artırımı; sensör histogram raporu | Zayıf | 0 |
| 7 | EMA aç, kontrol noktalarını sakla | Orta-zayıf | 0 |
| 8 | CE + Lovasz; olay dışı eşikler | Zayıf-orta | 0 |
| 9 | Dekodere dropout istenirse şimdi ekle; EDL ekleme | Zayıf | 0 |
| 10 | Son aşama: TTA + 3 üyeli topluluk + bina düzeyi toplama | Orta | Koşular paylaşılır |
| 11 | İzinliyse: 1 tur güven filtreli sözde etiket | Orta | 1 koşu |
| Atla | SAM, MixStyle, EDL, Sentinel temelli modeller, çok koşulu arama | Zayıf veya yok | Bütçe koruması |

Önerilen 4-5 koşu dağılımı (benim tahminim): 3 event-holdout koşusu (ve topluluk), 1 tüm veri son eğitimi, 1 sözde etiket veya risk yedeği.

### Gaps
- Hasar tahmininde bina düzeyinde ECE veya risk-kapsam eğrisi veren birincil çalışma bulamadım; reçete genel literatürden uyarlanmıştır.
- Sıcaklığı, MC dropout ve topluluğu BRIGHT olay dışında karşılaştıran çalışma bulamadım. xBD için BayeSiamMTL ve CNES/Thales posteri okunmadı.
- Selective scaling'in tam eğitim gereksinimi ve Local TS'in bu görevde çalışıp çalışmadığı doğrulanmadı.
- MetaSeg veya bina düzeyi meta-sınıflandırma için arama sonuç vermedi; yöntem bu notta değerlendirilmedi.
- Etiketsiz gerçek hedefte kalibrasyon ölçmenin güvenilir bir yolunu bulamadım; ValUES ve Ovadia sonuçları bunu desteklemiyor.
