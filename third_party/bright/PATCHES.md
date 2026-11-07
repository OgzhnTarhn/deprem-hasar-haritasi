# Yamalar

Bölme listelerinde (`splitname/`) değişiklik yapılmadı.

Model dosyalarında yalnız başlangıç ağırlığı indirmesi kapatıldı. Zero-shot checkpoint'i yüklenirken bütün ağırlıklar
zaten üzerine yazıldığı için çıkarım sonucunu etkilemez; güncel torchvision'da kaldırılmış `pretrained=` argümanı da
böylece ortadan kalkar.

| Dosya | Satır | Değişiklik |
|---|---|---|
| `models/SiamCRNN.py` | 52–53 | `torchvision.models.resnet18(pretrained=True)` → `resnet18(weights=None)` |
| `models/DamageFormer.py` | 77–78 | `swin_t(weights=Swin_T_Weights)` → `swin_t(weights=None)` |
| `models/DeepLabV3Plus.py` | 40 | `models.resnet50(pretrained=True)` → `models.resnet50(weights=None)` |
