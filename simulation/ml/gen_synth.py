"""Generate a synthetic FLS detection dataset from the simulator.

Produces YOLO-format images and labels using exactly the class indices of the
public real dataset, so a model trained here can be scored directly on the
real held-out test split.

Labels are analytic: each object's footprint is projected into the fan image
from known geometry, so there is no annotation noise and no human labelling
cost. That is the operational argument for the simulator, since no labelled
sonar exists for Indian flood scenarios.
"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image

from varuna.validation import labelled_tank_frame, DATASET_CLASSES
from varuna.acoustics import preset

ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=2400)
ap.add_argument("--n-val", type=int, default=300)
ap.add_argument("--size", type=int, default=640)
ap.add_argument("--out", default=os.path.expanduser("~/dev/datasets/fls_synth"))
args = ap.parse_args()

OUT = args.out
for split in ("train", "val"):
    for sub in ("images", "labels"):
        os.makedirs(f"{OUT}/{sub}/{split}", exist_ok=True)
        for f in os.listdir(f"{OUT}/{sub}/{split}"):
            os.remove(f"{OUT}/{sub}/{split}/{f}")

rng = np.random.default_rng(0)
counts = {c: 0 for c in DATASET_CLASSES}
n_empty = 0

for split, n in (("train", args.n_train), ("val", args.n_val)):
    base = 0 if split == "train" else 900000
    for i in range(n):
        seed = base + i
        # Vary the sonar configuration so the model does not latch onto one
        # fixed range window or noise level.
        r = rng.random()
        cfg = preset("aris", seed=int(rng.integers(1 << 30)),
                     r_min=0.5,
                     r_max=float(rng.uniform(5.0, 9.5)),
                     ssc_g_per_l=float(rng.uniform(0.0, 1.2)),
                     n_looks=float(rng.uniform(1.2, 2.4)),
                     noise_floor_db=float(rng.uniform(30, 42)),
                     dynamic_range_db=float(rng.uniform(38, 52)))
        img, boxes = labelled_tank_frame(seed, cfg, size=args.size,
                                         clutter=r > 0.12)
        if not boxes:
            n_empty += 1
        Image.fromarray(img).save(f"{OUT}/images/{split}/s{seed:06d}.png")
        with open(f"{OUT}/labels/{split}/s{seed:06d}.txt", "w") as fh:
            for c, xc, yc, w, h in boxes:
                counts[DATASET_CLASSES[c]] += 1
                fh.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        if i % 200 == 0:
            print(f"  {split} {i}/{n}", flush=True)

with open(f"{OUT}/data.yaml", "w") as fh:
    fh.write(f"path: {OUT}\ntrain: images/train\nval: images/val\n\nnames:\n")
    for i, c in enumerate(DATASET_CLASSES):
        fh.write(f"  {i}: {c}\n")

print()
print("synthetic dataset written to", OUT)
print(f"  train {args.n_train}, val {args.n_val}, frames with no label {n_empty}")
for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"    {c:16s} {n}")
