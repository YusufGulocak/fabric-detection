"""Leak-free data pipeline.

Expected layout (ImageFolder style):  data/<class_name>/<image>.png
Optional: if several images come from the same source (e.g. crops of one big
fabric roll image), name them  <groupid>__<anything>.png  so they stay in the
same split. Otherwise each file is its own group.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def scan(root):
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    paths, labels, groups = [], [], []
    for ci, c in enumerate(classes):
        for p in sorted((root / c).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                paths.append(str(p))
                labels.append(ci)
                groups.append(p.stem.split("__")[0] if "__" in p.stem else str(p))
    return classes, np.array(paths), np.array(labels), np.array(groups)


def make_splits(root, out="splits.json", seed=42, val=0.15, test=0.15, stratified=False):
    """Split ONCE by group, before any augmentation / normalisation stats.
    Saved to disk so every model uses the identical split.
    stratified=True keeps class proportions (StratifiedGroupKFold) - used for the clean split."""
    if Path(out).exists():
        return json.load(open(out))
    classes, paths, labels, groups = scan(root)
    idx = np.arange(len(paths))
    if stratified:
        def carve(ii, frac):
            k = round(1 / frac)
            sgk = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
            rest, held = next(sgk.split(ii, labels[ii], groups[ii]))
            return ii[rest], ii[held]
        trv, te = carve(idx, test)
        tr, va = carve(trv, val / (1 - test))
    else:
        gss = GroupShuffleSplit(1, test_size=test, random_state=seed)
        trv, te = next(gss.split(idx, labels, groups))
        gss2 = GroupShuffleSplit(1, test_size=val / (1 - test), random_state=seed)
        tr, va = next(gss2.split(trv, labels[trv], groups[trv]))
        tr, va = trv[tr], trv[va]
    # hard leakage check: no group may appear in two splits
    g = lambda i: set(groups[i])
    assert not (g(tr) & g(va) or g(tr) & g(te) or g(va) & g(te)), "group leakage!"
    s = {"classes": classes}
    for name, ii in (("train", tr), ("val", va), ("test", te)):
        s[name] = {"paths": [p.replace(chr(92), "/") for p in paths[ii]], "labels": labels[ii].tolist()}
    json.dump(s, open(out, "w"))
    return s


def train_mean_std(paths, size):
    """Normalisation stats from the TRAIN split only."""
    t = T.Compose([T.Resize((size, size)), T.ToTensor()])
    s = torch.zeros(3); s2 = torch.zeros(3)
    for p in paths:
        x = t(Image.open(p).convert("RGB"))
        s += x.mean((1, 2)); s2 += (x ** 2).mean((1, 2))
    m = s / len(paths)
    return m.tolist(), (s2 / len(paths) - m ** 2).sqrt().tolist()


def transforms(size, mean, std, train):
    aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
           T.ColorJitter(0.2, 0.2)] if train else []
    return T.Compose([T.Resize((size, size)), *aug, T.ToTensor(), T.Normalize(mean, std)])


class DS(Dataset):
    def __init__(self, split, tf):
        self.p = [p.replace(chr(92), "/") for p in split["paths"]]  # splits.json made on Windows
        self.y, self.tf = split["labels"], tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), self.y[i]
