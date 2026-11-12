# 1. Hafta Raporu: Kumaşlarda Hata Sınıflandırması

## Amaç
Kumaş hata görüntülerini 5 sınıfa (cut, hole, knot, stain, thread_error) ayırmak ve beş model ailesini (CNN, transfer learning, ViT, ViT+CNN hibrit, Vision Mamba) data leakage olmadan, aynı koşullarda karşılaştırmak.

## Yapılanlar

### 1. Veri seti seçimi
- Lusitano ve Roboflow "Fabric Defects 5-class" incelendi.
- Lusitano'nun eğitim kısmında hatalı görüntü yok (anomali tespiti formatı, 4096×1024), bu yüzden sınıflandırma için uygun bulunmadı.
- Roboflow veri seti seçildi (CC BY 4.0, COCO formatı, 5 sınıf).

### 2. Leakage analizi ve veri hazırlama
- Roboflow'un eğitim kümesinde her kaynak görüntünün yaklaşık 3 augmentation kopyası var. Hazır bölme olduğu gibi kullanılırsa aynı görüntünün kopyaları train ve test'e dağılır (sızıntı).
- 5358 görüntü, augmentation kopyaları atılarak 1968 benzersiz kaynağa indirildi.
- Perceptual hash (dHash, çevrilmiş varyantlar dahil) ile kaynaklar arası 225 benzer görüntü birleştirildi: 1743 grup.
- Hata kutuları kırpılarak 3095 örneklik sınıflandırma verisi üretildi: stain 859, thread_error 752, hole 671, knot 616, cut 197.

### 3. Bölme ve değerlendirme protokolü
- Gruba göre tek seferlik train/val/test bölmesi (yaklaşık 70/15/15). Gruplar arası kesişim kodla denetleniyor.
- Normalizasyon istatistikleri yalnızca train'den hesaplanıyor; augmentation yalnızca train'de.
- En iyi epoch val macro-F1'e göre seçiliyor; test seti en sonda bir kez kullanılıyor.
- Sınıf dengesizliği için ağırlıklı cross-entropy; ana metrik macro-F1.

### 4. Altyapı
PyTorch tabanlı eğitim kodu: 6 model, epoch bazlı loglama, test metrikleri (accuracy, macro/weighted F1, precision, recall, sınıf bazlı F1, confusion matrix) ve karşılaştırma tablosu.

### 5. Madde 1 (CNN) deneyleri
Sıfırdan eğitilen küçük bir CNN bu veride yetersiz kaldı (val macro-F1 ≈ 0,49; 22. epoch'ta durduruldu). Bu nedenle ImageNet-pretrained CNN'ler tam fine-tune edildi (15 epoch, AdamW, lr 1e-4, cosine, 160×160 giriş):

| Model | Parametre | Eğitim süresi (CPU) | Val macro-F1 | Test acc | Test macro-F1 |
|---|---|---|---|---|---|
| ResNet50 | 23,5M | ~43 dk | 0,785 | 0,769 | 0,803 |
| EfficientNet-B0 | 4,0M | ~21 dk | 0,777 | 0,780 | 0,810 |
| ConvNeXt-Tiny | 27,8M | ~1 saat | 0,824 | 0,803 | 0,831 |

Sınıf bazlı test F1 (ConvNeXt-Tiny): cut 1,00; hole 0,84; stain 0,83; thread_error 0,75; knot 0,73.

### 6. Cut sınıfı için dış veri denemesi
Cut sınıfı küçük olduğundan (ana veride 197 kırpma; train 138, val 30, test 29) dış veri araştırıldı:
- **TILDA-OBB** (195 cut kırpması) ve **Roboflow "fabric" / Afrasiab** (25) eklendi; toplam 220 kırpma, yalnızca train'e (138 → 358).
- **Figshare FD Dataset** tamamen elendi: dosya adları ve perceptual hash ile görüntülerin %100'ünün (190/190 cut) ana veri setinde zaten bulunduğu görüldü. Elenmeseydi val/test sızıntısı olacaktı.
- Dış veri, val/test'e ve birbirine benzeyen görüntülerden hash ile arındırıldı; val ve test değişmedi.

Aynı ayarlarla (160 px, 15 epoch, seed 42) sonuçlar:

| Model | Test macro-F1 (ana) | Test macro-F1 (+dış cut) | Test acc (ana) | Test acc (+dış cut) | Cut F1 (ana → +dış) |
|---|---|---|---|---|---|
| ResNet50 | 0,803 | 0,816 | 0,769 | 0,794 | 1,00 → 0,95 |
| EfficientNet-B0 | 0,810 | 0,782 | 0,780 | 0,763 | 1,00 → 0,92 |
| ConvNeXt-Tiny | 0,831 | 0,822 | 0,803 | 0,798 | 1,00 → 0,97 |

Yorum: tutarlı bir iyileşme gözlenmedi (ResNet50 +0,013, EfficientNet-B0 -0,028, ConvNeXt-Tiny -0,009; tek seed ve 451 örneklik test setinde bu farklar gürültü bandında). Üç modelde de cut precision'ı düştü (0,91-0,94), recall 1,00 kaldı. Olası sebep alan kayması: TILDA görüntüleri gri tonlu endüstriyel dokular, ana veri ise renkli telefon fotoğrafları. Ana sonuçlar dış veri olmadan raporlanmıştır.

## Değerlendirme
- ConvNeXt-Tiny en iyi sonucu verdi (ResNet50'ye göre test macro-F1 +0,029).
- EfficientNet-B0, ResNet50'den 6 kat az parametreyle benzer sonuç verdi.
- Tüm modellerde en zayıf sınıflar knot ve thread_error.
- Üç modelde de train accuracy ≈ 0,97-0,99, val ≈ 0,75-0,80: belirgin aşırı öğrenme var (küçük veri).
- Cut sınıfının test setinde yalnızca 29 örneği olduğundan 1,00 sonucu güvenilir değil.

## Sınırlılıklar
- Tek seed; modeller arası farklar küçük (test seti 451 örnek), anlamlılık için seed tekrarı gerekli.
- Eğitim CPU'da yapıldığı için giriş 160×160 ile sınırlı; daha büyük giriş sonuçları değiştirebilir.

## Gelecek hafta
- Madde 2: transfer learning varyantları (frozen backbone, kademeli açma).
- Madde 3-5: ViT, ViT+CNN hibrit, Vision Mamba.
- Kaggle GPU'da 224 px ve 3 seed ile ana deneylerin tekrarı
- Birden fazla seed ile tekrar (ortalama ± std), augmentation güçlendirme ve görüntü boyutu deneyleri.
