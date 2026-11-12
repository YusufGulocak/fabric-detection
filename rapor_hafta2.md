# 2. Hafta Raporu: Kumaşlarda Hata Sınıflandırması

## Yapılanlar

### 1. Madde 2: transfer learning stratejileri
ConvNeXt-Tiny, aynı ayarlar (160 px, 15 epoch, seed 42, eski bölme):

| Strateji | Val macro-F1 | Test acc | Test macro-F1 |
|---|---|---|---|
| frozen (yalnızca sınıflandırıcı) | 0,698 | 0,698 | 0,733 |
| gradual (kademeli açma) | 0,774 | 0,778 | 0,803 |
| full fine-tune | 0,824 | 0,803 | 0,831 |

Gövdeyi ne kadar açarsak sonuç o kadar iyi; ImageNet özellikleri kumaş hatası için tek başına yetmiyor. Gradual, frozen'a göre +0,07 kazandırdı ama full'ün 0,03 altında kaldı. Discriminative lr koşulmadı. Tek seed olduğu için farklar kesin değildir.

### 2. Temiz bölme
Hata analizinde test örneklerinin bir kısmının augmentation kopyası olduğu ve aynı hatanın birden fazla kırpmayla sayıldığı görüldü. Kaynak başına en az augmente kopya alındı, çakışan kutular birleştirildi, grup içi neredeyse aynı kırpmalar atıldı: 2927 kırpma, 1744 grup. Bölme sınıf dengeli ve gruba göre yapıldı: train 2091, val 418, test 418 (cut test: 28). Yukarıdaki sonuçlar eski bölmeye aittir; tüm modeller bu bölmede yeniden koşulacak.

## Gelecek
- Discriminative lr ve tüm stratejilerin temiz bölmede tekrarı.
- Kaggle GPU'da temiz bölmede, 224 px ve 3 seed ile ana deneylerin tekrarı (paket hazır: `kaggle_upload_clean.zip`).
- Madde 3-5: ViT, ViT+CNN hibrit, Vision Mamba.
