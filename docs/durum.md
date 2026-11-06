# Durum

Her faz sonunda güncellenir: ne bitti, ne ölçüldü, sıradaki faz ne.

| Faz | Durum | Not |
|---|---|---|
| Araştırma ve tasarım | Bitti (6 Kasım 2026) | `reports/` (üç rapor), `docs/tasarim.md` sürüm 3, `docs/fazlar.md` sürüm 2, `docs/fikir-havuzu.md` |
| 0. Belgeler ve protokol taslağı | Bitti (6 Kasım 2026) | `docs/adr/` (0001–0006), `docs/protokol.md` 0.1, `docs/veri-lisans.md`, `docs/riskler.md`, `docs/sar-temelleri.md` |
| 1. Depo iskeleti | Bitti (6 Kasım 2026) | `src/hasar/` paketi, `configs/ana.yaml`, sonuç şeması, Türkiye guard'ı, model sözleşmesi, CI; 125 kısa test (~3 sn) |
| 2. Veri katmanı | Sırada | |
| 3–9 | Bekliyor | |

## Faz 1'de kurulanlar

- **Paket:** `uv` projesi (Python ≥3.10), torch ve timm `train` ekstrasında ve sabitlenmemiş (Colab'ın kurulu
  sürümleri). Alt paketler: `data`, `models`, `train`, `eval`, `analysis`, `preflight`.
- **Yapılandırma:** `hasar.config` dondurulmuş dataclass'lar; `configs/ana.yaml` varsayılanların tamamını açık
  yazar. `config_hash` sıralı JSON'un sha256'sının ilk 12 hanesi; `run_id = <deney>-<hash>`.
- **Sonuç şeması:** `hasar.results` metrik etiketi olmayan ya da tanınmayan etiketli kaydı yazmaz (kural 4–5).
- **Guard:** `hasar.guard.require_turkey_access` yalnız HEAD `protokol-kilidi` etiketindeyse ve çalışma ağacı
  temizse `TurkeyAccess` verir (kural 1).
- **Model sözleşmesi:** her model `{"loc": B×2×H×W, "clf": B×4×H×W}` döndürür; `registry` adla kurar.
- **Üçüncü taraf:** BRIGHT `standard_ML` bölme listeleri sabit commit'ten (`third_party/bright/SOURCE.md`).
- **Not:** `.gitignore`'daki `data/` kuralı `src/hasar/data`'yı da yutuyordu; kök klasöre bağlandı (`/data/`).

## Faz 0'da ölçülen ve öğrenilenler

- **Zero-shot bölme (BRIGHT kodu okundu):** eğitim = standart eğitim + standart test listesi, doğrulama = standart
  doğrulama (Türkiye hariç). Kare sayıları **2.795 / 337 / 1.114** (eski tahmin 2.141 eğitimdi).
- **Başlık metriği yolu (BRIGHT ve ChangeMamba kodu):** tam 1024 kare, `argmax(clf)`, maskesiz, olay boyunca
  biriktirilen 4×4 matris, arka plan dahil 4 sınıf ortalaması, IoU paydasında 1e-7; nodata için ayrı işlem yok.
- **SAR girdisi:** Resmî kod SAR'ı 3 kanala kopyalayıp ImageNet istatistikleriyle normalize ediyor. Tasarımdaki
  "eğitim kümesi istatistiği" bundan sapıyordu → ADR-0004 (öneri: resmî biçim, eğitim kümesi istatistiği A3'e).
- **ChangeMamba resmî reçetesi:** crop 512, batch 8, 80.000 adım (= 640.000 örnek; bizim 40.000 × 16 ile aynı).
- **Kapı 7:** "Hep arka plan" mIoU'su sabit 0,21 değil, arka plan piksel oranı / 4.
- **Geometri:** 5 katlı bir binanın radar bindirmesi 0,35 m pikselde yaklaşık 60 piksel; SAR izi optik ayak izinden
  kayık olabilir (`docs/sar-temelleri.md` §3).
- Eğitimde yalnız üç deprem var (Haiti, Fas, Noto); öteki olaylar patlama, yanardağ, yangın, kasırga, sel, savaş.

## Açık kararlar

- **ADR-0004 (SAR girdisi):** Önerilen seçenek C, kullanıcı onayı gelene kadar varsayılan olarak uygulanır; kod
  iki biçimi de yapılandırmayla destekler.
- **Protokol §9 "eşit" kuralı:** Öneri bekliyor (Faz 4 öncesi).
- Ana eğitimin yapılacağı ay için Colab Pro mu Pro+ mı (Faz 5'ten önce).
- Fikir havuzundaki adaylar (KN1, Faz 6). N1 için kural 1 istisnasının genişletilmesi (KN1).

## Ölçülmemiş tahminler

- A100 adım hızı, tepe VRAM, CU yakımı, Hugging Face indirme süresi: Faz 3–4'te ölçülecek.
