# ADR-0001: Zero-shot bölme: Türkiye tamamen test

- **Durum:** Kabul
- **Tarih:** 6 Kasım 2026 (geriye dönük kayıt)
- **İlgili:** tasarım §6, `CLAUDE.md` kural 1, [protokol.md](../protokol.md) §2

## Bağlam

Projenin sorusu, modelin eğitimde hiç görmediği bir depremde ne kadar işe yaradığı. BRIGHT iki bölme sunuyor:

- **`standard_ML`:** Rastgele bölme. Türkiye kareleri eğitimde (777), doğrulamada (113) ve testte (224).
- **`cross_event/zero_shot`:** Hedef olay bütün listelerden çıkarılır, bütün kareleri teste gider.

**Kanıt (kod okuması, `bda_benchmark/script/cross_event/zero_shot/train_SiamCRNN.py`):** Zero-shot betiği üç
standart listeyi okur. Test olayının adıyla başlayan bütün kareler üç listeden toplanıp teste konur. Eğitim
listesi, standart eğitim listesinin ve **standart test listesinin** Türkiye dışı karelerinin birleşimidir.
Doğrulama listesi, standart doğrulama listesinin Türkiye dışı kareleridir.

**Kanıt (liste sayımı, 6 Kasım 2026):** Standart listeler 2.918 / 450 / 878 kare. Türkiye çıkarılınca eğitim
**2.795**, doğrulama **337**, test **1.114** kare. 13 olayın hepsi eğitimde ve doğrulamada temsil ediliyor.
Araştırma raporundaki "yaklaşık 2.141 eğitim" tahmini, standart test listesinin eğitime katıldığını hesaba
katmamıştı.

## Seçenekler

| Seçenek | Artı | Eksi |
|---|---|---|
| `standard_ML` | Daha çok eğitim verisi | Türkiye eğitimde: doğrudan sızıntı; 41,28 ile kıyaslanamaz |
| BRIGHT zero-shot mantığı, `test_event = turkey-earthquake` | 41,28 ile aynı protokol | Verinin yarısından fazlası (bina pikselleri) eğitim dışı |
| Kendi bölmemiz (ör. Türkiye'yi şehir bazında ayırmak) | Esnek | Yayımlanmış hiçbir sayıyla kıyaslanamaz |

## Karar

BRIGHT'ın zero-shot mantığı birebir uygulanır: eğitim = standart eğitim + standart test (Türkiye hariç),
doğrulama = standart doğrulama (Türkiye hariç), test = Türkiye'nin üç listedeki bütün kareleri.

## Sonuçlar

- Bölme üreticisi Türkiye karesini eğitim veya doğrulamada görürse hata verir (Faz 2 testi).
- Kare sayıları (2.795 / 337 / 1.114) ön kontrol kapısı 2'nin beklenen değerleridir.
- Olay adı `rsplit('_', 1)[0]` ile ayrıştırılır (`la_palma-volcano` tuzağı). BRIGHT'ın kodu `startswith` ile
  süzüyor; ikisi bu veri için aynı sonucu verir, bu eşitlik de test edilir.
