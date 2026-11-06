# Mimari Karar Kayıtları (ADR)

Her ADR tek bir kararı, bağlamını, değerlendirilen seçenekleri ve sonuçlarını kaydeder. Kararın kendisi
[tasarim.md](../tasarim.md)'de de yer alır; ADR "neden"i saklar.

- Yeni ADR için [0000-sablon.md](0000-sablon.md) kopyalanır, sıradaki numara verilir.
- Kabul edilmiş bir ADR değiştirilmez. Karar değişirse yeni bir ADR yazılır ve eskisinin durumu
  "Yerini aldı: ADR-XXXX" olur.
- Durumlar: **Önerildi** (kullanıcı onayı bekliyor) · **Kabul** · **Yerini aldı** · **Reddedildi**.

| No | Başlık | Durum |
|---|---|---|
| [0001](0001-zero-shot-bolme.md) | Zero-shot bölme: Türkiye tamamen test | Kabul |
| [0002](0002-ana-model-ve-yedek.md) | Ana model ChangeMamba-MMBDA Tiny, yedek DamageFormer | Kabul |
| [0003](0003-baslangic-agirligi-ve-dislananlar.md) | Başlangıç ağırlığı ImageNet; Türkiye'yi görmüş olabilecek modeller dışarıda | Kabul |
| [0004](0004-sar-girdisi.md) | SAR girdisi: resmî biçim (3 kanal, ImageNet normalizasyonu) | Önerildi |
| [0005](0005-baslik-metrigi.md) | Başlık metriği: resmî kod yolu | Kabul |
| [0006](0006-moduler-kod-yapisi.md) | Modüler kod yapısı | Kabul |
