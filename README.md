# Kumaş Hata Tespiti (Fabric Defect Classification)

Danışman hocanın istediği 5 yaklaşımın aynı veri ve aynı bölmeyle karşılaştırılması.

| # | Model | `--model` | Not |
|---|-------|-----------|-----|
| 1 | CNN (ImageNet-pretrained, tam fine-tune) | `resnet50`, `efficientnet_b0`, `convnext_tiny` | klasik / verimli / modern |
| 2 | Transfer learning varyantları | _(sonra)_ | frozen backbone, kademeli açma |
| 3 | ViT (sıfırdan) | `vit` | |
| 4 | ViT + CNN hibrit | `hybrid` | conv stem + transformer |
| 5 | Vision Mamba | `vision_mamba` | saf PyTorch, bidirectional SSM |

Çalışma sırası: resnet50 → efficientnet_b0 → convnext_tiny → vit → hybrid → vision_mamba.

## Kurulum
```
pip install -r requirements.txt
```
Veri: `data_cls/<sinif_adi>/<grupid>__<ad>.png`  (aynı kaynak görüntüden gelenler aynı `grupid`).

## Çalıştırma
```
python prepare_data.py --src data --out data_cls   # Roboflow COCO -> kırpma + grup
python train.py --model resnet50 --data data_cls --epochs 15 --lr 1e-4 --size 160
python train.py --model all --data data_cls         # hepsi sırayla
```
GPU yoksa `--size 128` kullan (özellikle vision_mamba için).

## Çıktılar (`runs/<model>/`)
- `config.json` – hiperparametreler, sınıf sayıları, parametre sayısı, normalizasyon istatistikleri
- `epochs.csv` – epoch başına train/val loss, accuracy, macro-F1, lr, süre
- `test_metrics.json`, `test_report.txt`, `confusion_matrix.csv` – tek seferlik test sonucu
- `best.pt` – val macro-F1'e göre en iyi ağırlıklar
- `runs/summary.csv` – tüm modellerin karşılaştırma tablosu

## Data leakage kuralları
1. Bölme tek sefer, gruba göre (`GroupShuffleSplit`), `splits.json`'a kaydedilir; tüm modeller aynısını kullanır.
2. Gruplar split'ler arasında kesişirse kod `assert` ile durur.
3. Normalizasyon ortalaması/std sadece train'den hesaplanır.
4. Augmentation sadece train'de.
5. Model/epoch seçimi val ile; test seti yalnızca en sonda bir kez.
6. Veri setinde önceden augmentation uygulanmışsa (ör. Roboflow) kopya/benzer görüntüler bölmeden önce temizlenmeli.

Detaylı gereksinimler: [REQUIREMENTS.md](REQUIREMENTS.md)
