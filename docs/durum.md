# Durum

Her faz sonunda güncellenir: ne bitti, ne ölçüldü, sıradaki faz ne.

| Faz | Durum | Not |
|---|---|---|
| Araştırma ve tasarım | Bitti (6 Kasım 2026) | `reports/` (üç rapor), `docs/tasarim.md` sürüm 3, `docs/fazlar.md` sürüm 2, `docs/fikir-havuzu.md` |
| 0. Belgeler ve protokol taslağı | Sırada | ADR'ler, `docs/protokol.md` taslağı, veri ve lisans tablosu, risk kaydı, SAR notu |
| 1–9 | Bekliyor | |

## 6 Kasım 2026 araştırmasının sonuçları

- Bugün 41,28 `[ZS-EVENT mIoU]` değerini geçen yayımlanmış bir yöntem bulunamadı.
- Proje titiz ama yenilikçi değil; Y2–Y4'ün yakın öncülleri var. Yenilik değerlendirme tarafında kurulabilir
  (fikir havuzu, N1–N3).
- HOT/OSM yıkık bina katmanı optik yoruma dayanır, BRIGHT etiketinden bağımsız değil.
- ChangeMamba'nın ONNX'e çıkması şüpheli (Faz 4'te denenir).
- En güçlü varyant adayı Sentinel-1 koherans kanalı (A1); fizibilitesi Faz 2'de.
- Demo gerçek 6 Şubat verisiyle (BRIGHT Türkiye kareleri). İstanbul hasar demosu yapılamaz.

## Açık kararlar

- Ana eğitimin yapılacağı ay için Colab Pro mu Pro+ mı (Faz 5'ten önce).
- Fikir havuzundaki adaylar: varyant fikri, topluluk satırları, N analizleri, transdüktif işler (KN1, Faz 6).
- N1 için kural 1 istisnasının hazır checkpoint'lerin güven aralığı analizine genişletilmesi (KN1).

## Ölçülmemiş tahminler

- A100 adım hızı, tepe VRAM, CU yakımı, Hugging Face indirme süresi: Faz 3–4'te ölçülecek.
