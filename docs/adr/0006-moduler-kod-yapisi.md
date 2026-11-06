# ADR-0006: Modüler kod yapısı

- **Durum:** Kabul
- **Tarih:** 6 Kasım 2026
- **İlgili:** tasarım §8a, [rapor Bölüm 2](../../reports/Eksik%20belgeler,%20modüler%20yapı%20ve%20yenilik%20değerlendirmesi.md), rehber §14

## Bağlam

Projede üç model (ana, yedek, varyant), 11 ön kontrol, birkaç analiz ve bir demo aynı veri ve metrik kodunu
kullanacak. Kural 1 (Türkiye kilidi) ve kural 5 (sayılar dosyadan) yalnız disiplinle korunursa kırılgan kalır.

## Seçenekler

| Seçenek | Artı | Eksi |
|---|---|---|
| Resmî betikleri olduğu gibi kullanmak | Az kod | Türkiye'yi eğitimde değerlendiriyor, resume eksik, `max_iters` belirsiz |
| Düz paket (`data`, `models`, `train`, `eval`, tek `analysis.py`) | Basit | Kilit ve önbellek düzeni koda bağlı değil |
| **Katmanlı paket + model sözleşmesi + guard + tahmin önbelleği** | Kurallar kodla korunur; yedeğe geçiş ve varyant tek dosya | Biraz daha fazla iskelet |
| Hydra ve eklenti sistemi | Esnek | İki eğitimlik projede hata yüzeyi |

## Karar

Katmanlı `src/hasar` paketi: tek yönlü bağımlılık (`data`, `models` → `train`; `eval` `train`'i import etmez;
`analysis` yalnız tahmin önbelleğini okur), `DamageModel` sözleşmesi ve kayıt defteri, Türkiye karelerine tek erişim
yolu olan `guard`, `metric_tag` zorunlu sonuç şeması, `third_party/` altında sabit commit'li üçüncü taraf kod.

## Sonuçlar

- Faz 1'in çıkış koşulları guard, şema ve config hash testlerini içerir.
- Türkiye'de tek çıkarım yapılır; başlık, ikincil satırlar, analizler ve demo aynı önbelleği okur.
- Hydra, eklenti sistemi ve soyut Trainer hiyerarşisi kullanılmaz.
