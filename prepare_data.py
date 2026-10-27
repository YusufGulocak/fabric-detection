"""Roboflow COCO export  ->  leak-safe crop classification dataset.

python prepare_data.py --src data --out data_cls

Steps (all aimed at avoiding leakage):
 1. Merge train/valid/test of the export (Roboflow's own split is NOT trusted:
    its train set contains ~3 augmented copies of every source image).
 2. Keep ONE image per source (extra.name), dropping Roboflow's augmentation copies.
 3. Cluster near-duplicate images across sources (dHash incl. flipped variants);
    each cluster becomes one group.
 4. Crop every bounding box (square, 15% margin) -> out/<class>/<group>__<n>.png
    The group id in the file name is what data.make_splits() splits on.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MARGIN, MIN_SIDE, HAM = 0.15, 32, 4


def dhash(im):
    a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hashes(im):
    return [dhash(im), dhash(ImageOps.mirror(im)), dhash(ImageOps.flip(im)),
            dhash(ImageOps.flip(ImageOps.mirror(im)))]


def square_crop(im, b):
    x, y, w, h = b
    s = max(w, h) * (1 + 2 * MARGIN)
    s = min(s, im.width, im.height)
    cx, cy = x + w / 2, y + h / 2
    x0 = min(max(cx - s / 2, 0), im.width - s)
    y0 = min(max(cy - s / 2, 0), im.height - s)
    return im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s)))


def main(a):
    src, out = Path(a.src), Path(a.out)
    items = {}  # source name -> (path, [(cls, bbox)])
    n_imgs = 0
    for sp in ("train", "valid", "test"):
        d = json.load(open(src / sp / "_annotations.coco.json"))
        cat = {c["id"]: c["name"] for c in d["categories"] if c["supercategory"] != "none"}
        anns = defaultdict(list)
        for an in d["annotations"]:
            anns[an["image_id"]].append((cat[an["category_id"]], an["bbox"]))
        for im in d["images"]:
            n_imgs += 1
            name = im["extra"]["name"]
            if name not in items and anns[im["id"]]:
                items[name] = (src / sp / im["file_name"], anns[im["id"]])
    print(f"{n_imgs} exported images -> {len(items)} unique sources with boxes")

    names = list(items)
    H = {n: hashes(Image.open(items[n][0])) for n in names}
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]; n = parent[n]
        return n

    base = np.stack([H[n][0] for n in names])
    merged = 0
    for n in names:
        for v in H[n][1:]:
            for j in np.nonzero((base != v).sum(1) <= HAM)[0]:
                r1, r2 = find(n), find(names[j])
                if r1 != r2:
                    parent[r1] = r2; merged += 1
    print(f"near-duplicate merges (cross-source): {merged}")
    gid = {}
    for n in names:
        gid.setdefault(find(n), f"g{len(gid):05d}")

    cnt = Counter()
    for n in names:
        path, boxes = items[n]
        im = Image.open(path).convert("RGB")
        for k, (c, b) in enumerate(boxes):
            if min(b[2], b[3]) < MIN_SIDE:
                continue
            (out / c).mkdir(parents=True, exist_ok=True)
            square_crop(im, b).save(out / c / f"{gid[find(n)]}__{n[:40]}_{k}.png")
            cnt[c] += 1
    print("groups:", len(gid), "| crops per class:", dict(cnt), "| total:", sum(cnt.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data"); p.add_argument("--out", default="data_cls")
    main(p.parse_args())
