# ADR-0005: Başlık metriği: resmî kod yolu

- **Durum:** Kabul
- **Tarih:** 6 Kasım 2026 (geriye dönük kayıt; Faz 0 kod okumasıyla ayrıntılandırıldı)
- **İlgili:** tasarım §6, `CLAUDE.md` kural 4, [protokol.md](../protokol.md) §3

## Bağlam

Başlık sayısı yalnız aynı yoldan hesaplanırsa 41,28 ile kıyaslanabilir. ChangeMamba makalesi nihai haritanın bina
maskesiyle birleştirildiğini yazıyor; kod ise maskesiz çalışıyor.

**Kanıt (Faz 0 kod okuması):**
- BRIGHT `util_func/metrics.py` ve ChangeMamba `evaluation/multiclass.py` aynı hesabı yapıyor: 4×4 karışıklık
  matrisi, IoU = köşegen / (satır toplamı + sütun toplamı − köşegen + 1e-7), mIoU = 4 sınıfın `nanmean`'i
  (**arka plan dahil**). Etiketi [0, 4) dışında olan pikseller (ör. 255) matrise girmez.
- Değerlendirmede tahmin `argmax(clf)`; **bina maskesi uygulanmıyor** (`loc` çıktısı mIoU'ya girmiyor).
- Değerlendirmede **kırpma yok**: kareler 1024 × 1024 tam olarak modele veriliyor (`is_train` değilse dönüşüm yok).
- Matris bütün test kareleri boyunca **biriktiriliyor**; kare başına IoU ortalanmıyor. Test kümesi yalnız Türkiye
  olduğu için bu olay düzeyi mIoU'dur.
- Resmî kodda nodata için ayrı bir işlem yok: nodata pikselleri etiketlerinde ne yazıyorsa öyle sayılıyor.
- Raporlanan test sayısı, doğrulama mIoU'su en iyi olan kayıt noktasının test sonucudur.

## Karar

Başlık sonucu `[ZS-EVENT mIoU]`: 13 olayın doğrulamasında seçilen EMA kayıt noktası, Türkiye'nin 1.114 karesinde
bir kez, tam kare, TTA'sız, `argmax(clf)`, maskesiz, olay boyunca biriktirilmiş 4×4 matristen, arka plan dahil 4
sınıf ortalaması. Nodata pikselleri değerlendirmede resmî kodla aynı işlenir (ayrıca yok sayılmaz).

## Sonuçlar

- Metrik kodu BRIGHT'ın formülünü (1e-7 dahil) birebir uygular; Faz 3 yeniden üretimi bunu sınar.
- Önceden ilan edilen ikincil satırlar: flip TTA, bina düzeyi oylama, bina maskeli harita, nodata yok sayılmış hâl.
- Eğitimde nodata yok sayma (tasarım §7) bizim reçetemizin parçasıdır ve değerlendirme yolunu değiştirmez.
