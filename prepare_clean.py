"""Cleaner crop dataset from the Roboflow COCO export  ->  data_clean/

python prepare_clean.py --src data --out data_clean
python train.py --data data_clean --splits splits_clean.json --stratified ...

Differences from prepare_data.py (found by inspecting ConvNeXt test errors):
 1. Per source image keep the LEAST augmented copy: valid/test copies are originals
    (Roboflow only augments train); for train-only sources pick the copy with the
    smallest black border (rotation / shear leaves black corners).
 2. Overlapping boxes inside one image (IoU > 0.6) are merged -> one crop.
 3. Near-identical crops inside one group (dHash incl. flips, same class) are dropped,
    so one defect is not counted 3 times in val/test.
 4. Same cross-source near-duplicate grouping as before (group id in the file name).
The stratified group split is then made by train.py --stratified.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_data import HAM, MIN_SIDE, dhash, hashes, square_crop

CROP_HAM, IOU = 6, 0.6


def black_frac(im):
    a = np.asarray(im.convert("L").resize((64, 64)))
    return float((a < 6).mean())


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / (a[2] * a[3] + b[2] * b[3] - i)


def merge_boxes(boxes):
    """boxes: [(cls, bbox)] -> drop same-class boxes overlapping > IOU (keep larger)."""
    kept = []
    for c, b in sorted(boxes, key=lambda x: -x[1][2] * x[1][3]):
        if not any(c == c2 and iou(b, b2) > IOU for c2, b2 in kept):
            kept.append((c, b))
    return kept


def main(a):
    src, out = Path(a.src), Path(a.out)
    cands = defaultdict(list)  # source name -> [(is_original, path, boxes)]
    for sp in ("train", "valid", "test"):
        d = json.load(open(src / sp / "_annotations.coco.json"))
        cat = {c["id"]: c["name"] for c in d["categories"] if c["supercategory"] != "none"}
        anns = defaultdict(list)
        for an in d["annotations"]:
            anns[an["image_id"]].append((cat[an["category_id"]], an["bbox"]))
        for im in d["images"]:
            if anns[im["id"]]:
                cands[im["extra"]["name"]].append((sp != "train", src / sp / im["file_name"], anns[im["id"]]))

    items, n_orig = {}, 0
    for name, lst in cands.items():
        scored = sorted(lst, key=lambda c: (not c[0], black_frac(Image.open(c[1]).convert("RGB"))))
        items[name] = scored[0]
        n_orig += scored[0][0]
    print(f"{len(items)} sources; {n_orig} use an original (valid/test) copy, "
          f"{len(items) - n_orig} use the least-augmented train copy")

    names = list(items)
    H = {n: hashes(Image.open(items[n][1]).convert("RGB")) for n in names}
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]; n = parent[n]
        return n

    base = np.stack([H[n][0] for n in names])
    for n in names:
        for v in H[n][1:]:
            for j in np.nonzero((base != v).sum(1) <= HAM)[0]:
                r1, r2 = find(n), find(names[j])
                if r1 != r2:
                    parent[r1] = r2
    gid = {}
    for n in names:
        gid.setdefault(find(n), f"g{len(gid):05d}")

    cnt, dropped_iou, dropped_dup = Counter(), 0, 0
    kept_h = defaultdict(list)  # (group, class) -> [hash arrays of kept crops]
    for n in names:
        _, path, boxes = items[n]
        im = Image.open(path).convert("RGB")
        merged = merge_boxes(boxes)
        dropped_iou += len(boxes) - len(merged)
        g = gid[find(n)]
        for k, (c, b) in enumerate(merged):
            if min(b[2], b[3]) < MIN_SIDE:
                continue
            crop = square_crop(im, b)
            hs = hashes(crop)
            if any(((np.stack(kept_h[(g, c)]) != h).sum(1) <= CROP_HAM).any() for h in hs) \
                    if kept_h[(g, c)] else False:
                dropped_dup += 1; continue
            kept_h[(g, c)].append(hs[0])
            (out / c).mkdir(parents=True, exist_ok=True)
            crop.save(out / c / f"{g}__{n[:40]}_{k}.png")
            cnt[c] += 1
    print(f"groups: {len(gid)} | dropped overlapping boxes: {dropped_iou} | "
          f"dropped near-identical crops in a group: {dropped_dup}")
    print("crops per class:", dict(cnt), "| total:", sum(cnt.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data"); p.add_argument("--out", default="data_clean")
    main(p.parse_args())
