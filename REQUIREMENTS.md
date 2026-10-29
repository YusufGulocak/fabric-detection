# Gereksinim Analizi

## 1. Amaç
Kumaş görüntülerinde hata sınıflandırması için 5 farklı model ailesini (CNN, transfer learning, ViT, ViT+CNN hibrit, Vision Mamba) data leakage olmadan, adil koşullarda karşılaştırmak.

## 2. Fonksiyonel gereksinimler
| ID | Gereksinim |
|----|-----------|
| F1 | Görüntüleri sınıf klasörlerinden okuyup grup-bazlı train/val/test'e bölmek (70/15/15) |
| F2 | 5 modeli ortak arayüzle eğitmek ve aynı bölmeyi kullanmak |
| F3 | Her epoch için loss, accuracy, macro-F1, lr, süre loglamak |
| F4 | Testte accuracy, macro/weighted F1, precision, recall, sınıf bazlı F1, confusion matrix üretmek |
| F5 | Modelleri tek tabloda (`summary.csv`) karşılaştırmak |
| F6 | Sonuçları tekrarlanabilir yapmak (seed, config kaydı) |

## 3. Fonksiyonel olmayan gereksinimler
- **Leakage yok:** bkz. README "Data leakage kuralları"; bölmede grup kesişimi kontrolü.
- **Tekrarlanabilirlik:** sabit seed, `config.json`, kaydedilen `splits.json`. Hakem/hoca için en az 3 farklı seed ile tekrar önerilir (ortalama ± std).
- **Sınıf dengesizliği:** ağırlıklı cross-entropy; ana metrik accuracy değil macro-F1.
- **Donanım:** CPU'da çalışır; vision_mamba (ardışık tarama) ve ViT için GPU önerilir.
- **Taşınabilirlik:** Windows uyumlu, CUDA'ya özel paket yok (`mamba-ssm` kullanılmadı).

## 4. Veri gereksinimleri
- Görüntü seviyesinde sınıf etiketi (bounding box varsa kırpılmalı veya etikete çevrilmeli).
- Aynı kaynaktan türeyen görüntüler (kırpmalar, augmentation kopyaları) aynı grup kimliğini taşımalı veya kopyalar silinmeli.
- Sınıf başına yeterli örnek (öneri: ≥ 100; az ise sınıfı birleştir veya k-fold düşün).
- Lisans/kullanım izni kontrolü.

### Aday veri setleri
| Veri seti | Uygunluk | Not |
|-----------|----------|-----|
| Roboflow fabric-defects 5 class | Uygun (5 sınıf) | Augmentation'sız sürümü al, kopyaları temizle, kutuları kırp |
| Lusitano | Sınıflandırma için uygun değil | Eğitimde sadece normal görüntü, 4096×1024; anomali tespiti için kullanılabilir |

## 4.1 Kullanılan veri setleri
| Veri seti | Rol | Kaynak | Lisans | Durum |
|-----------|-----|--------|--------|-------|
| Roboflow Fabric Defects 5-class (v3) | Ana veri (train/val/test) | https://universe.roboflow.com/jaswant-oyc4f/fabric-defects-5-class-b2mwz | CC BY 4.0 | Kullanıldı |
| TILDA-OBB (Mendeley) | Dış veri, yalnızca train; yalnızca cut kullanıldı | https://data.mendeley.com/datasets/8y46pm2m6y/1 | Kontrol edilecek | Kullanıldı (cut: 195 kırpma) |
| FD Dataset (Figshare) | Elendi: görüntülerin %100'ü ana veri setinde zaten var (hash/dosya adı) | https://figshare.com/articles/dataset/FD_Dataset_7z/25546465 | CC BY 4.0 (data.yaml) | Elendi (0 kırpma) |
| Roboflow "fabric" (Afrasiab) | Dış veri, yalnızca train; cut (45 kutu) | https://universe.roboflow.com/afrasiab/fabric-5pxpq | CC BY 4.0 | Kullanıldı (25 kırpma) |
| Mendeley Fabric Defects Dataset (Ranathunga) | Kullanılmadı: cut yok | https://data.mendeley.com/datasets/663j22s43c/3 | Kontrol edilecek | Kullanılmadı |
| Lusitano | Kullanılmadı (eğitimde hatalı görüntü yok, anomali tespiti formatı) | https://kailashhambarde.github.io/Lusitano/ | Belirtilmemiş | Elendi |

Dış veri kuralları: yalnızca train'e eklenir; val/test'e benzer görüntüler hash ile çıkarılır; her kırpmanın kaynağı kaydedilir; sonuçlar "+dış veri" deneyi olarak ayrı raporlanır. Amaç cut sınıfı için örnek sayısını artırmak (ana veride 197 kırpma).

## 5. Değerlendirme metrikleri
Birincil: macro-F1. İkincil: accuracy, macro precision/recall, sınıf bazlı F1, confusion matrix. Pratik: parametre sayısı, eğitim süresi.

## 5.1 Teknoloji yığını
Python, PyTorch, torchvision, scikit-learn, Pillow, NumPy. Veri: Roboflow Fabric Defects 5-class (COCO, CC BY 4.0). Vision Mamba saf PyTorch (mamba-ssm kullanılmadı).

## 5.2 Model seçimi (Madde 1: CNN)
ImageNet-pretrained, tam fine-tune. Küçük veri (~3k kırpma) nedeniyle sıfırdan CNN eğitilmez.
| Model | Parametre | Gerekçe |
|-------|-----------|---------|
| ResNet50 | 25M | Literatürdeki standart baseline |
| EfficientNet-B0 | 5M | Küçük/verimli; küçük veride düşük overfit riski, CPU'da hızlı |
| ConvNeXt-Tiny | 28M | Modern CNN; ViT ile adil kıyas sağlar |
Ortak ayar: aynı split, aynı normalizasyon, ağırlıklı CE, AdamW lr 1e-4, cosine schedule.

## 6. Deney planı
1. Veri hazırlık + kopya temizliği + split (tek sefer)
2. resnet50 → efficientnet_b0 → convnext_tiny → vit → hybrid → vision_mamba (aynı epoch, lr, boyut)
3. Seed tekrarları, sonuç tablosu ve confusion matrix karşılaştırması
4. (Opsiyonel) ablation: görüntü boyutu, augmentation

## 7. Riskler
| Risk | Önlem |
|------|-------|
| Veri sızıntısı (augmentation kopyaları) | perceptual hash ile kopya tespiti, grup split |
| Küçük veri → ViT/Mamba sıfırdan zayıf kalır | Beklenen sonuç olarak raporla; transfer ile kıyasla |
| Vision Mamba CPU'da çok yavaş | Küçük görüntü (128), GPU |
| Sınıf dengesizliği | Ağırlıklı loss, macro-F1 |

## 8. Yazılım gereksinimleri
Python ≥ 3.10, `torch`, `torchvision`, `scikit-learn`, `pillow`, `numpy` (bkz. `requirements.txt`).
