"""External cut samples -> data_ext/cut/*.png  (TRAIN-ONLY extra data).

python prepare_external.py [--classes cut]

Sources (see REQUIREMENTS.md 4.1):
  TILDA-OBB-ALL (YOLO-OBB)   class Cut                     -> cut
  FD_Dataset/YOLO (YOLO bbox) class 2 (tears, checked visually) -> cut
  fabric.v1i.coco (Afrasiab)  class cut                    -> cut

Leakage rules: every candidate source image is compared (dHash incl. flips,
hamming <= HAM) against ALL main-dataset images (train+valid+test of the
Roboflow 5-class export) and against already kept external images; matches
are dropped. Extras are only ever appended to TRAIN by train.py --extra.
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_data import HAM, MIN_SIDE, dhash, hashes, square_crop

EXT = Path("external")


def yolo_boxes(label_file, W, H, keep):
    out = []
    for l in open(label_file):
        p = l.split()
        if not p or int(p[0]) not in keep:
            continue
        v = list(map(float, p[1:]))
        if len(v) == 4:                       # cx cy w h
            cx, cy, w, h = v
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:                                 # 4 polygon points (OBB)
            xs, ys = v[0::2], v[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        x0, x1 = max(x0, 0), min(x1, 1)
        y0, y1 = max(y0, 0), min(y1, 1)
        out.append([x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H])
    return out


def candidates():
    """yield (source_tag, image_path, [bbox px])"""
    # TILDA: Cut = class 0
    A = EXT / "TILDA-OBB/TILDA-OBB/TILDA-OBB-ALL"
    for lf in sorted(glob.glob(str(A / "*/labels/*.txt"))):
        sp = Path(lf).parent.parent
        img = glob.glob(str(sp / "images" / (Path(lf).stem + ".*")))[0]
        W, H = Image.open(img).size
        b = yolo_boxes(lf, W, H, {0})
        if b:
            yield "tilda", img, b
    # FD: class 2 = cut
    Y = EXT / "FD_Dataset/YOLO"
    for lf in sorted(glob.glob(str(Y / "*/labels/*.txt"))):
        sp = Path(lf).parent.parent
        img = glob.glob(str(sp / "images" / (Path(lf).stem + ".*")))[0]
        W, H = Image.open(img).size
        b = yolo_boxes(lf, W, H, {2})
        if b:
            yield "fd", img, b
    # Afrasiab COCO: cut
    R = EXT / "fabric.v1i.coco"
    for sp in ("train", "valid", "test"):
        d = json.load(open(R / sp / "_annotations.coco.json"))
        cid = {c["name"]: c["id"] for c in d["categories"]}["cut"]
        by = {}
        for a in d["annotations"]:
            if a["category_id"] == cid:
                by.setdefault(a["image_id"], []).append(a["bbox"])
        for im in d["images"]:
            if im["id"] in by:
                yield "afrasiab", str(R / sp / im["file_name"]), by[im["id"]]


def main(a):
    main_h = np.stack([dhash(Image.open(f)) for f in
                       glob.glob("data/*/*.jpg")])
    print("main images hashed:", len(main_h))
    kept_h, out = [], Path(a.out) / "cut"
    out.mkdir(parents=True, exist_ok=True)
    stats = {k: [0, 0, 0] for k in ("tilda", "fd", "afrasiab")}  # seen, dup_main, dup_ext
    n = 0
    for tag, path, boxes in candidates():
        st = stats[tag]; st[0] += 1
        im = Image.open(path).convert("RGB")
        hs = hashes(im)
        if any(((main_h != h).sum(1) <= HAM).any() for h in hs):
            st[1] += 1; continue
        if kept_h and any(((np.stack(kept_h) != h).sum(1) <= HAM).any() for h in hs):
            st[2] += 1; continue
        kept_h.append(hs[0])
        for k, b in enumerate(boxes):
            if min(b[2], b[3]) < MIN_SIDE:
                continue
            square_crop(im, b).save(out / f"ext_{tag}_{n:05d}_{k}.png"); n += 1
    print("per source [images seen, dup of main, dup of external]:", stats)
    print("external cut crops written:", len(list(out.glob('*.png'))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", default="data_ext")
    main(p.parse_args())
