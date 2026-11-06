# third_party

Bu klasör, ChenHongruixuan/BRIGHT ve ChenHongruixuan/ChangeMamba depolarından alınan (vendored) kodu barındırır.
Her iki depo da Apache-2.0 lisanslıdır.

Her alt klasör şunları içermek zorundadır:

- kaynak commit SHA'sı,
- kaynağın `LICENSE` dosyası,
- yaptığımız değişiklikleri listeleyen `PATCHES.md`.

Kendi kodumuz bu klasöre yalnızca `src/hasar/models` altındaki sarmalayıcılar (wrapper) üzerinden dokunur.
