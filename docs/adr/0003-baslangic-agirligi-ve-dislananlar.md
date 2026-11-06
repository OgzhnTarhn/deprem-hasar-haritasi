# ADR-0003: Başlangıç ağırlığı ImageNet; Türkiye'yi görmüş olabilecek modeller dışarıda

- **Durum:** Kabul
- **Tarih:** 6 Kasım 2026 (geriye dönük kayıt)
- **İlgili:** `CLAUDE.md` kural 2, tasarım §5

## Bağlam

Başlangıç ağırlığı Türkiye 2023 sahnelerini (etiketli ya da etiketsiz) görmüşse, "görülmemiş deprem" iddiası
sessizce bozulur.

**Kanıt:**
- Zenodo 14037769'daki `ckpt_ChangeMamba_bright_standard_ML_split.pth`, `standard_ML` bölmesinde eğitildi;
  Türkiye eğitimdeydi.
- **MaRS** (SAR temel modeli): Umbra ve Capella ile eğitildi; verisi 30 Aralık 2024'e kadar uzanıyor; lisansı üç
  kaynakta üç farklı.
- **DINOv3 SAT-493M:** Lisansı askerî ve savaş amaçlı kullanımı yasaklıyor; görülmemiş olay hasar görevinde test
  edilmedi.
- BRIGHT ekibinin 2026 karşılaştırmasına göre genel amaçlı temel modeller özel mimarilerin gerisinde.

## Seçenekler

| Seçenek | Türkiye riski | Not |
|---|---|---|
| ImageNet `vssm_tiny` (Zenodo 14037769) | Yok | Resmî başlangıç |
| `standard_ML` checkpoint'leri | Kesin (eğitimde Türkiye) | Kural 2 ihlali |
| MaRS | Olası | Lisans belirsiz |
| DINOv3 SAT | Düşük ama doğrulanamaz | Lisans sürtünmesi |

## Karar

Kodlayıcılar ImageNet `vssm_tiny` ağırlıklarıyla başlar. `*_standard_ML_split.pth`, MaRS ve 2023 Türkiye
sahnelerini görmüş olabilecek hiçbir model başlangıç ağırlığı olamaz. DINOv3 lisans nedeniyle kullanılmaz.

## Sonuçlar

- Fikir havuzundaki D5 (temel model kodlayıcısı) bu ADR yüzünden "Ret önerisi" durumunda; bir aday ancak ön eğitim
  verisinin kapsamı belgelenmişse yeniden ele alınır.
- Faz 3'te kullanılan Zenodo 15349462 zero-shot checkpoint'leri Türkiye'yi görmedi; yalnız yeniden üretim ve
  (KN1'de kabul edilirse) topluluk/güven aralığı analizi için kullanılır, başlangıç ağırlığı olarak kullanılmaz.
