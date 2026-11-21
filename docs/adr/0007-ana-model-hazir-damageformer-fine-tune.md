# ADR-0007: Ana model hazır DamageFormer ile kontrollü fine-tune

- **Durum:** Kabul
- **Tarih:** 21 Kasım 2026
- **İlgili:** ADR-0002'nin ana model kararının yerine geçer; tasarım §5, §7, §9, §11;
  protokol 1.3; [onaylı tasarım](../superpowers/specs/2026-11-21-sar-dayaniklilik-finetune-design.md)

## Bağlam

**Kanıt:** ChangeMamba için Colab L4 ön kontrolünün 8. kapısı maliyeti bütçenin üstünde buldu.
40.000 adım projeksiyonu 164,6 CU; sınır 75 CU. Eğitim başlamadı; yeni eğitim ve `final` için Türkiye'ye
dokunulmadı. Faz 3 yeniden üretimi ayrı, önceden izin verilmiş istisnadır.

Hazır DamageFormer zero-shot checkpoint'i Türkiye'yi görmedi. Referans sonucu 37,20 `[ZS-EVENT mIoU, repro]`
(`results/yeniden-uretim/damageformer_test.json`); doğrulaması 74,48 `[VAL13 mIoU, repro]`
(`results/yeniden-uretim/damageformer_val.json`). SAR ölçeği olaylar arasında değişiyor
(`results/arastirma/sar_olcek.json`).

**Yargı:** Kısa fine-tune, öğrenme amacına ve bütçeye daha uygun. Fikrin etkisini ek eğitimin ve kaybın etkisinden
ayırmak için aynı reçeteyle bir kontrol gerekir. Kullanıcı bu yolu 21 Kasım 2026'da onayladı.

## Seçenekler

| Seçenek | Artı | Eksi | Maliyet |
|---|---|---|---|
| Kısa ChangeMamba | Mevcut modelle bağ korunur | Fine-tune öğretmez | Yaklaşık 30 CU; ölçülmedi |
| Hazır DamageFormer + kontrollü fine-tune | Ortak başlangıçla fikrin etkisi ayrılır | Fine-tune modeli bozabilir | Koşu başına ≤ 15 CU hedefi; ölçülmedi, kapı 8 ölçer |
| Eğitimsiz topluluk | Eğitim maliyeti yok | Fine-tune öğretmez | Eğitim: 0 CU |

## Karar

Ana model, hazır DamageFormer zero-shot checkpoint'inin kontrol fine-tune'u; tek varyant ise aynı checkpoint ve
reçeteyle SAR radyometrik dayanıklılık fine-tune'udur.

## Sonuçlar

- `configs/ana.yaml` ve `configs/varyant.yaml` yalnız deney adı ve SAR artırımı alanlarında ayrılır.
  Ortak tohum, kırpma, örnek sırası ve adım sayısı korunur. Her biri bir kez eğitilir.
- Başlangıç `ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth`; md5 denetlenir.
  EMA da bu ağırlıklardan başlar. Kayıt noktası son adımın EMA'sıdır; doğrulamaya göre seçim yapılmaz.
- A4, yalnız 13 olayın SAR ölçümleriyle KN1'de seçildi. Değerlendirmede artırım uygulanmaz.
- Birincil kıyas varyant − ana; ikincil kıyaslar iki fine-tune'un referansa farkıdır.
  Eşleştirilmiş blok bootstrap ve protokol §9'un “eşit” kuralı kullanılır.
- ChangeMamba 41,28 `[ZS-EVENT mIoU]` yalnız bağlamdır. Eğitimi ve topluluk protokol 1.3 kapsamında değildir.
- Kullanıcı `protokol-kilidi`ni 1.3'ü içeren commit'e taşır ve aynı commit'e `varyant-kilidi` koyar.
  Ana modelin Türkiye sonucundan önce iki tasarım da kilitlenir. Tek “Run All” iki koşuyu sırayla tamamlar.
- ≤ 15 CU fine-tune hedefi ölçülmedi. Kapı 8 eğitim projeksiyonunu ölçer; ön kontrol maliyeti bu projeksiyonun dışındadır.
