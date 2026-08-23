"""Train a sonar detector on simulated data only, then test it on real sonar.

This is the operationally relevant transfer direction. No labelled sonar
exists for Indian flood scenarios and none will exist before deployment, so
the question that matters is whether a physics based simulator can serve as
the training source for a detector that then works on real acoustic imagery.

Training set : frames from the isonavi sonar simulator, labels derived
               analytically from scene geometry, no human annotation
Test set     : the real ARIS Explorer 3000 held-out split, untouched

A real-trained baseline on the same test split is reported alongside, so the
gap attributable to the simulator is explicit.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sh(*a):
    print("$", " ".join(a), flush=True)
    subprocess.run(list(a), check=False)


INPUT = Path("/kaggle/input")
print("inputs:", [p.name for p in INPUT.iterdir()] if INPUT.exists() else "none")

SYNTH = None
REAL = None
for p in INPUT.rglob("data.yaml"):
    txt = p.read_text()
    if "images/train" in txt and (p.parent / "images" / "train").exists():
        if "synth" in str(p).lower():
            SYNTH = p.parent
        else:
            REAL = p.parent
if SYNTH is None or REAL is None:
    # Fall back to identifying by directory name.
    for d in INPUT.iterdir():
        if "synth" in d.name and SYNTH is None:
            cands = list(d.rglob("images"))
            if cands:
                SYNTH = cands[0].parent
        elif REAL is None:
            cands = list(d.rglob("images"))
            if cands:
                REAL = cands[0].parent
print("synthetic:", SYNTH)
print("real     :", REAL)

import torch  # noqa: E402

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("gpu", torch.cuda.get_device_name(0), cap, "torch", torch.__version__)
    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        sh(sys.executable, "-m", "pip", "install", "-q",
           "torch==2.5.1", "torchvision==0.20.1",
           "--index-url", "https://download.pytorch.org/whl/cu121")
        os.execv(sys.executable, [sys.executable] + sys.argv)

sh(sys.executable, "-m", "pip", "install", "-q", "ultralytics")
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

WORK = Path("/kaggle/working")
CLASSES = ["mine", "can", "bottle", "drink-carton", "chain", "propeller",
           "tire", "hook", "valve", "shampoo-bottle", "standing-bottle"]

# Working copies, because Kaggle inputs are read only and YOLO writes caches.
SDIR, RDIR = WORK / "synth", WORK / "real"
for src, dst in ((SYNTH, SDIR), (REAL, RDIR)):
    if not dst.exists():
        shutil.copytree(src, dst)

syn_yaml = WORK / "synth.yaml"
syn_yaml.write_text(yaml.safe_dump({
    "path": str(SDIR), "train": "images/train", "val": "images/val",
    "names": {i: c for i, c in enumerate(CLASSES)}}))

real_yaml = WORK / "real.yaml"
real_yaml.write_text(yaml.safe_dump({
    "path": str(RDIR), "train": "images/train", "val": "images/val",
    "test": "images/test", "names": {i: c for i, c in enumerate(CLASSES)}}))

for tag, d in (("synth", SDIR), ("real", RDIR)):
    for split in ("train", "val", "test"):
        p = d / "images" / split
        if p.exists():
            print(f"  {tag}/{split}: {len(list(p.glob('*')))}")

dev = 0 if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- train on sim
model = YOLO("yolov8s.pt")
model.train(
    data=str(syn_yaml), epochs=80, imgsz=640, batch=16, patience=20,
    device=dev, project=str(WORK / "runs"), name="synth", exist_ok=True,
    pretrained=True, seed=0, workers=2,
    fliplr=0.5, flipud=0.0, degrees=4.0, translate=0.08, scale=0.4,
    shear=0.0, perspective=0.0, mosaic=0.7, mixup=0.05,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.45,
)
best = WORK / "runs" / "synth" / "weights" / "best.pt"
shutil.copy(best, WORK / "synth_trained.pt")

out = {"n_train": len(list((SDIR / "images" / "train").glob("*")))}

# ---------------------------------------------------------------- evaluate
print("\n=== simulator-trained model, evaluated on the REAL test split ===",
      flush=True)
m = YOLO(str(best)).val(data=str(real_yaml), split="test", imgsz=640,
                        device=dev, conf=0.001)
out.update({
    "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
    "precision": float(m.box.mp), "recall": float(m.box.mr),
    "per_class_mAP50": {CLASSES[int(i)]: float(v)
                        for i, v in zip(m.box.ap_class_index, m.box.ap50)},
})
print(json.dumps(out, indent=1))

# Sanity check that the simulator-trained model works on its own domain.
print("\n=== same model on the simulated validation split ===", flush=True)
ms = YOLO(str(best)).val(data=str(syn_yaml), split="val", imgsz=640, device=dev)
out["sim_val_mAP50"] = float(ms.box.map50)
out["sim_val_mAP50_95"] = float(ms.box.map)

(WORK / "synth_transfer_metrics.json").write_text(json.dumps(out, indent=1))
shutil.rmtree(SDIR, ignore_errors=True)
shutil.rmtree(RDIR, ignore_errors=True)
shutil.rmtree(WORK / "runs" / "synth" / "weights", ignore_errors=True)
print("done")
