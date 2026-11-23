# ADR-0008: Türkiye few-shot ek deneyi

- **Durum:** Kabul
- **Tarih:** 23 Kasım 2026
- **İlgili:** [protokol §15](../protokol.md), [tasarım](../superpowers/specs/2026-11-23-turkiye-few-shot-design.md), ADR-0001, ADR-0007

## Bağlam

Kanıt: zero-shot ana 40,44 `[ZS-EVENT mIoU]` aldı (`results/ana/turkiye.json`); hasarlı ve yıkık IoU %3'ün altında.
Gerçek afette birkaç gün içinde her ilden birkaç mahalle elle etiketlenebilir. Kullanıcı 22–23 Kasım 2026'da bu
durumu ölçen ikinci bir deney istedi.

Yargı: Türkiye etiketlerini kullanan her deney zero-shot sorusundan ayrıdır; aynı tabloda karışırsa iki sonuç da
anlamını yitirir. Motivasyon Türkiye sonucundan geldiği için tarif önceden ve tek seferde kilitlenmelidir.

## Seçenekler

| Seçenek | Artı | Eksi | Maliyet |
|---|---|---|---|
| Few-shot, blok bazlı bölme + 0,5 km tampon, A/B | Her şehirden örnek; kayıp etkisi ölçülür; gerçek senaryoya yakın | Uzamsal benzerlik tamamen gitmez | 2 eğitim (≈ 3,7 CU tahmin, ölçülmedi) |
| Few-shot, şehir bazlı bölme | Sızıntı riski en düşük | Uyarlama çeşitliliği az | Aynı |
| `standard_ML` bölmesi | Hazır | Türkiye'nin çoğu eğitimde; ezber, az bilgi | Aynı |

## Karar

Kural 1'e dar bir istisna: kilitli bölmenin `adapt` kareleri `fewshot-kilidi` altında eğitime girer; ana
checkpoint'inden düz (A) ve hasar ağırlıklı (B) iki fine-tune, test bölgesinde birer kez, `[FS-EVENT mIoU]` etiketiyle.

## Sonuçlar

Zero-shot sonuçları değişmez. Few-shot sayıları ayrı tabloda, 41,28 ile kıyaslanmadan yazılır. Bootstrap'ın kare
sırasına bağlı hizalama hatası bu karar sırasında bulundu ve düzeltildi; zero-shot kıyasları etkilenmedi (arşivler
aynı sırada).
