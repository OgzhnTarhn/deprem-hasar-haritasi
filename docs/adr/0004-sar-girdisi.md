# ADR-0004: SAR girdisi: resmî biçim (3 kanal, ImageNet normalizasyonu)

- **Durum:** Kabul (7 Kasım 2026; kullanıcı bekleyen önerilerin hepsini kabul etti)
- **Tarih:** 6 Kasım 2026
- **İlgili:** tasarım §7 "Normalizasyon" satırı, fikir havuzu A3, [protokol.md](../protokol.md) §4

## Bağlam

Tasarım sürüm 3, SAR'ı **eğitim kümesinin ortalama ve standart sapmasıyla** normalize etmeyi öngörüyor (yargı;
iki kaynak 0–1 ölçeğinde yaklaşık 0,22 / 0,15–0,18 veriyor). Aynı tasarım, ana modelin "ChangeMamba olduğu gibi"
eğitileceğini söylüyor.

**Kanıt (Faz 0 kod okuması):** Hem BRIGHT (`bda_benchmark/dataset/make_data_loader.py`) hem güncel ChangeMamba
(`changedetection/datasets/multimodal_damage_assessment.py`) SAR'ı şöyle işliyor:

1. Tek bantlı SAR üç kez yan yana konup **3 kanallı** yapılıyor (`np.stack((post,)*3, axis=-1)`).
2. Optikle **aynı ImageNet istatistikleriyle** normalize ediliyor (ortalama 123,675 / 116,28 / 103,53, standart
   sapma 58,395 / 57,12 / 57,375; 0–255 ölçeğinde).

41,28 bu girdi biçimiyle elde edildi. SAR kodlayıcısı ImageNet ağırlıklarıyla başladığı için 3 kanal bekliyor.

Tasarımdaki yol iki şeyi birden değiştiriyor: kanal sayısı (veya ilk katmanın uyarlanması) ve normalizasyon
istatistiği. İkisi de ana modeli "olduğu gibi" olmaktan çıkarıyor ve sonucun 41,28'den farkının bir kısmını
açıklanamaz hâle getiriyor.

## Seçenekler

| Seçenek | Artı | Eksi | Maliyet |
|---|---|---|---|
| **A. Resmî biçim:** 3 kanal kopya, ImageNet normalizasyonu | 41,28 ile aynı girdi; "olduğu gibi" ilkesiyle tutarlı; ImageNet ağırlıkları olduğu gibi yüklenir | SAR istatistiği ImageNet'e uymuyor (fiziksel olarak anlamsız, ama ağ bunu öğrenebiliyor) | 0 |
| B. Tasarımdaki yol: eğitim kümesi istatistiği | Fiziksel olarak daha temiz girdi | Resmî reçeteden sapma; etkisi ölçülmedi | 0 |
| C. A ana eğitimde; B (ve dB/z-skoru) Faz 4'te kısa provada ölçülür, fikir havuzunda kalır | Ana model temiz kalır, soru kaybolmaz | Provanın 13 olayda ölçülmesi ~1–2 CU | ~1–2 CU |

## Karar

**C.** Ana eğitim resmî SAR biçimini kullanır. Eğitim kümesi istatistiği ve dB/z-skoru ön işlemesi fikir
havuzunda (A3) kalır; Faz 4'te kısa provada 13 olayın doğrulamasıyla ölçülür. Ana reçeteye ancak `protokol-kilidi`
öncesinde ve yalnız 13 olayın sonucuna göre girebilir; girerse "olduğu gibi" ilkesinden sapma olarak raporlanır.

## Sonuçlar

- Tasarım §7'nin normalizasyon satırı "SAR: resmî biçim (3 kanal, ImageNet)" olarak güncellendi.
- Model sözleşmesindeki `post: B×1×H×W` korunur; 3 kanala kopyalama ChangeMamba sarmalayıcısının içinde yapılır.
  Böylece öteki modeller ve analizler tek bantlı SAR görür.
- Y3 (Türkiye istatistikleriyle yeniden standartlaştırma) bu girdi biçimi üzerinde tanımlanır.
