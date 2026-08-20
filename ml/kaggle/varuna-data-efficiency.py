"""Does simulator pretraining help when real sonar data is scarce?

Zero-shot transfer from simulation to real sonar fails, which is measured
separately. That is not the question that decides whether the simulator is
worth building. The operational situation is that a small amount of real data
can be collected from a target site, but nothing like enough to train from
scratch. The question is therefore whether pretraining on simulator output
reduces how much real data is needed.

Two arms, identical in every other respect, trained on the same N real images
and evaluated on the same untouched real test split:

    COCO init        the usual starting point, generic natural-image weights
    simulator init   weights pretrained on physics-simulated sonar only

Sweeping N gives a data-efficiency curve. If the simulator has value, the
simulator-initialised arm reaches a given accuracy with fewer real images.
"""
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


def sh(*a):
    print("$", " ".join(a), flush=True)
    subprocess.run(list(a), check=False)


INPUT = Path("/kaggle/input")
print("inputs:", [p.name for p in INPUT.iterdir()])

REAL = None
for p in INPUT.rglob("data.yaml"):
    if (p.parent / "images" / "test").exists():
        REAL = p.parent
SIM_W = None
for p in INPUT.rglob("synth_trained.pt"):
    SIM_W = p
print("real dataset :", REAL)
print("sim weights  :", SIM_W)
assert REAL is not None and SIM_W is not None

import torch  # noqa: E402

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("gpu", torch.cuda.get_device_name(0), cap)
    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        sh(sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1",
           "torchvision==0.20.1", "--index-url",
           "https://download.pytorch.org/whl/cu121")
        os.execv(sys.executable, [sys.executable] + sys.argv)

sh(sys.executable, "-m", "pip", "install", "-q", "ultralytics")
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

WORK = Path("/kaggle/working")
CLASSES = ["mine", "can", "bottle", "drink-carton", "chain", "propeller",
           "tire", "hook", "valve", "shampoo-bottle", "standing-bottle"]

BASE = WORK / "real_full"
if not BASE.exists():
    shutil.copytree(REAL, BASE)

train_imgs = sorted((BASE / "images" / "train").glob("*"))
print("real train pool:", len(train_imgs))
random.seed(0)
random.shuffle(train_imgs)

dev = 0 if torch.cuda.is_available() else "cpu"
SIZES = [40, 80, 160, 320, 640]
results = []


def build_subset(n):
    """Materialise a subset of N real training images."""
    d = WORK / f"sub{n}"
    if d.exists():
        shutil.rmtree(d)
    for split in ("train", "val"):
        (d / "images" / split).mkdir(parents=True, exist_ok=True)
        (d / "labels" / split).mkdir(parents=True, exist_ok=True)
    for img in train_imgs[:n]:
        lbl = BASE / "labels" / "train" / (img.stem + ".txt")
        shutil.copy(img, d / "images" / "train" / img.name)
        if lbl.exists():
            shutil.copy(lbl, d / "labels" / "train" / lbl.name)
    # Validation stays the full real val split for every arm.
    for img in sorted((BASE / "images" / "val").glob("*")):
        lbl = BASE / "labels" / "val" / (img.stem + ".txt")
        shutil.copy(img, d / "images" / "val" / img.name)
        if lbl.exists():
            shutil.copy(lbl, d / "labels" / "val" / lbl.name)
    y = WORK / f"sub{n}.yaml"
    y.write_text(yaml.safe_dump({
        "path": str(d), "train": "images/train", "val": "images/val",
        "names": {i: c for i, c in enumerate(CLASSES)}}))
    return y


test_yaml = WORK / "real_test.yaml"
test_yaml.write_text(yaml.safe_dump({
    "path": str(BASE), "train": "images/train", "val": "images/val",
    "test": "images/test", "names": {i: c for i, c in enumerate(CLASSES)}}))

for n in SIZES:
    yml = build_subset(n)
    for arm, init in (("coco", "yolov8s.pt"), ("sim", str(SIM_W))):
        name = f"n{n}_{arm}"
        print(f"\n===== {name} =====", flush=True)
        m = YOLO(init)
        m.train(data=str(yml), epochs=70, imgsz=640, batch=16, patience=20,
                device=dev, project=str(WORK / "runs"), name=name,
                exist_ok=True, pretrained=True, seed=0, workers=2, verbose=False,
                fliplr=0.5, flipud=0.0, degrees=4.0, translate=0.08, scale=0.35,
                shear=0.0, perspective=0.0, mosaic=0.7, mixup=0.05,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.35)
        best = WORK / "runs" / name / "weights" / "best.pt"
        v = YOLO(str(best)).val(data=str(test_yaml), split="test", imgsz=640,
                                device=dev, verbose=False)
        row = {"n_real": n, "arm": arm,
               "mAP50": float(v.box.map50), "mAP50_95": float(v.box.map),
               "precision": float(v.box.mp), "recall": float(v.box.mr)}
        results.append(row)
        print(json.dumps(row), flush=True)
        shutil.rmtree(WORK / "runs" / name / "weights", ignore_errors=True)
        (WORK / "data_efficiency.json").write_text(json.dumps(results, indent=1))
    shutil.rmtree(WORK / f"sub{n}", ignore_errors=True)

print("\n===== SUMMARY =====")
print(f"{'N real':>8}{'COCO mAP50':>13}{'SIM mAP50':>12}{'gain':>9}")
for n in SIZES:
    c = next((r for r in results if r["n_real"] == n and r["arm"] == "coco"), None)
    s = next((r for r in results if r["n_real"] == n and r["arm"] == "sim"), None)
    if c and s:
        print(f"{n:>8}{c['mAP50']*100:>12.1f}%{s['mAP50']*100:>11.1f}%"
              f"{(s['mAP50']-c['mAP50'])*100:>+8.1f}")

shutil.rmtree(BASE, ignore_errors=True)
print("done")
