"""python train.py --model resnet50 --data data_cls --epochs 30 --lr 1e-4
Models: resnet50 | efficientnet_b0 | convnext_tiny | vit | hybrid | vision_mamba | all   (all runs in that order)

Leakage rules enforced here:
 - group-aware split done once, saved to splits.json (same for all models)
 - normalisation stats from train split only
 - augmentation only on train
 - best epoch chosen on VAL; TEST evaluated exactly once at the end

Logs per model in runs/<model>/ :
 config.json, epochs.csv, test_metrics.json, test_report.txt, confusion_matrix.csv, best.pt
Summary of all models: runs/summary.csv
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader

from data import DS, make_splits, train_mean_std, transforms
from models import REGISTRY

ORDER = ["resnet50", "efficientnet_b0", "convnext_tiny", "vit", "hybrid", "vision_mamba"]


def run(name, a, dev):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    s = make_splits(a.data, seed=a.seed)
    nc = len(s["classes"])
    mean, std = train_mean_std(s["train"]["paths"], a.size)
    dl = lambda k, tr: DataLoader(DS(s[k], transforms(a.size, mean, std, tr)),
                                  a.bs, shuffle=tr, num_workers=a.workers)
    tr, va, te = dl("train", True), dl("val", False), dl("test", False)

    model = REGISTRY[name](nc, size=a.size).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    cnt = np.bincount(s["train"]["labels"], minlength=nc)
    lossf = nn.CrossEntropyLoss(torch.tensor(cnt.sum() / (nc * cnt), dtype=torch.float32).to(dev))

    json.dump({**vars(a), "model": name, "device": dev, "n_params": n_params,
               "classes": s["classes"], "n_train": len(s["train"]["paths"]),
               "n_val": len(s["val"]["paths"]), "n_test": len(s["test"]["paths"]),
               "train_class_counts": cnt.tolist(), "norm_mean": mean, "norm_std": std},
              open(out / "config.json", "w"), indent=2)

    def evaluate(loader):
        model.eval(); P, Y, L = [], [], 0.0
        with torch.no_grad():
            for x, y in loader:
                o = model(x.to(dev))
                L += lossf(o, y.to(dev)).item() * len(y)
                P += o.argmax(1).cpu().tolist(); Y += y.tolist()
        return Y, P, L / len(Y)

    log = open(out / "epochs.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
                "val_macro_f1", "epoch_sec"])
    best, t_start = -1, time.time()
    for ep in range(a.epochs):
        t0 = time.time(); model.train(); tl = tc = n = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); o = model(x); l = lossf(o, y); l.backward(); opt.step()
            tl += l.item() * len(y); tc += (o.argmax(1) == y).sum().item(); n += len(y)
        lr = opt.param_groups[0]["lr"]; sched.step()
        Y, P, vl = evaluate(va)
        f1 = f1_score(Y, P, average="macro", zero_division=0)
        row = [ep + 1, lr, tl / n, tc / n, vl, float(np.mean(np.array(Y) == np.array(P))),
               f1, time.time() - t0]
        w.writerow(row); log.flush()
        print(f"[{name}] ep{ep+1}/{a.epochs} loss={row[2]:.4f} acc={row[3]:.4f} "
              f"val_loss={vl:.4f} val_f1={f1:.4f}")
        if f1 > best:
            best = f1; torch.save(model.state_dict(), out / "best.pt")
    log.close()
    train_sec = time.time() - t_start

    model.load_state_dict(torch.load(out / "best.pt"))
    Y, P, _ = evaluate(te)  # single test pass
    names = s["classes"]
    rep = classification_report(Y, P, target_names=names, digits=4, zero_division=0)
    (out / "test_report.txt").write_text(rep)
    cm = confusion_matrix(Y, P, labels=range(nc))
    np.savetxt(out / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(names), comments="")
    m = {"model": name, "params": n_params, "train_sec": train_sec, "val_best_macro_f1": best,
         "test_acc": float(np.mean(np.array(Y) == np.array(P))),
         "test_macro_f1": f1_score(Y, P, average="macro", zero_division=0),
         "test_weighted_f1": f1_score(Y, P, average="weighted", zero_division=0),
         "test_macro_precision": precision_score(Y, P, average="macro", zero_division=0),
         "test_macro_recall": recall_score(Y, P, average="macro", zero_division=0),
         "per_class_f1": dict(zip(names, f1_score(Y, P, average=None, labels=range(nc),
                                                  zero_division=0).tolist()))}
    json.dump(m, open(out / "test_metrics.json", "w"), indent=2)
    print(rep)
    return m


def update_summary(res, outdir):
    keys = ["model", "params", "train_sec", "val_best_macro_f1", "test_acc", "test_macro_f1",
            "test_weighted_f1", "test_macro_precision", "test_macro_recall"]
    path = Path(outdir) / "summary.csv"
    rows = {}
    if path.exists():
        rows = {r["model"]: r for r in csv.DictReader(open(path))}
    rows[res["model"]] = {k: res[k] for k in keys}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50"); p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for n in (ORDER if a.model == "all" else [a.model]):
        update_summary(run(n, a, dev), a.out)
