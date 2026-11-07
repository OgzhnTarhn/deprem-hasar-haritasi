# Risk Kaydı

**Sürüm:** 1 (Faz 0, 6 Kasım 2026). Her faz sonunda gözden geçirilir; gerçekleşen risk "Durum" sütununa işlenir.

**Olasılık / etki:** Y yüksek · O orta · D düşük.

| No | Risk | Olasılık | Etki | Tetik (nasıl anlarız) | Önlem | Yedek plan | Faz |
|---|---|---|---|---|---|---|---|
| R1 | ChangeMamba `selective_scan` çekirdeği Colab'da derlenmez | O | Y | Derleme kapısı 1 saatte geçmez | `compute_70`'i çıkar, `CUDA_HOME` → pip `nvcc`, `--no-build-isolation`, `timm` sabitlemesini kaldır | Otomatik olarak DamageFormer (ADR-0002) | 4 |
| R2 | Colab imajı değişir, derlenmiş paket kırılır | O | O | Oturum başında import hatası | Derlenmiş wheel ve imaj sürümü Drive'da; ortam kapısı torch/CUDA sürümünü kaydeder | Yeniden derleme (≤ 1 saat) veya yedek | 4–8 |
| R3 | Faz 3 yeniden üretimi ±0,3 içinde tutmaz | O | Y | 38,32 / 37,82 / 37,20'den sapma | Metrik ve okuma kodu resmî kodla birebir; BRIGHT'ın test yolu (tam kare, maskesiz) | Fark kaynağı bulunana kadar Faz 4'e geçilmez; checkpoint ile kod sürümü uyumsuzsa BRIGHT'ın o commit'i kullanılır | 3 |
| R4 | ChangeMamba ONNX'e aktarılamaz | Y | D | Faz 4 denemesi başarısız | Erken deneme | `torch.export` veya PyTorch'ta gecikme ölçümü | 4, 9 |
| R5 | CU bütçesi aşılır | O | Y | Kapı 8 projeksiyonu > bütçe | Hız ve VRAM 200 adımda ölçülür; High-RAM kapalı; iki eğitim ayrı aylarda | 512/8 kırpma, kullandıkça öde paketi, Pro+ (açık karar) | 4–8 |
| R6 | Oturum kopması, eksik devam durumu | Y | O | 12 saat sınırı, tarayıcı kapanması | Tam durum kaydı, atomik yazma, `seed + step` örnekleyici, kapı 10 | Drive + HF Hub iki kopya | 5, 8 |
| R7 | Hugging Face veri kopyası kalkar veya değişir | D | Y | İndirme hatası veya özet uyuşmazlığı | İlk indirmede dosya özetleri `results/veri/`'ye yazılır; zip'ler Drive'da yedeklenir | Drive yedeği | 2 |
| R8 | Türkiye bilgisi karar sürecine sızar (insan hatası) | O | Y | Türkiye'ye `final` dışından erişim | `guard`, git kilitleri, eğitim kodunda test yolu yok, protokol | Sızıntı olursa olay raporlanır, etkilenen sonuç "keşif" etiketi alır | 1–8 |
| R9 | SAR girdi biçimi resmî reçeteden sapar | O | O | ADR-0004 (kabul) uygulanmaz | ADR-0004 seçenek C | Sapma raporda açıkça yazılır | 1, 4 |
| R10 | BRIGHT'ta bina örnek etiketi yok, bağlı bileşenler binaları birleştirir | O | O | Faz 2 incelemesi | 13 olayda birleşme oranı ölçülür | Bina düzeyi metrikler sınırlamasıyla raporlanır | 2 |
| R11 | Sentinel-1 kapsamı A1 için yetersiz | O | D | Uygun çift < 10 olay | Faz 2 fizibilitesi | Varyant adayı D1'e geçer | 2 |
| R12 | Dış katmanlar (HDX, ARIA, ShakeMap) taşınır veya kaldırılır | D | O | Bağlantı hatası | İndirildiği anda yerel kopya, özet ve tarih kaydı | Arşiv kopyaları (Wayback, NASA kataloğu) | 7 |
| R13 | Lisans belirsizliği (Ukrayna/Myanmar/Meksika optiği, Maxar NC şartı) | O | O | İtiraz veya kaynak netleşmesi | Veri ve lisans tablosu, README notu | İlgili olaylar ayrı tabloda yeniden değerlendirilir | 0, 9 |
| R14 | Takvim kayması (16 hafta) | O | O | Bir faz planlanan haftayı 1 haftadan fazla aşar | Faz çıkış koşulları; kapsam fikir havuzunda | Fikir havuzu adayları ertelenir; iki tam eğitim ve başlık sonucu korunur | Hepsi |
| R15 | Eğitim çöker veya ıraksar | D | Y | Kayıp NaN, doğrulama düşüşü | Kapı 4, 5, 9; bf16 + fp32 Lovász; gradyan kırpma | Son sağlam kayıt noktasından düşük lr ile devam; kayıp raporlanır | 5, 8 |

## Gerçekleşen riskler

| Tarih | Risk | Ne oldu | Ne yapıldı |
|---|---|---|---|
| – | – | – | – |
