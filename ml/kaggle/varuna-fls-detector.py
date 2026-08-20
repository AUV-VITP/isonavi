"""Train a forward-looking sonar object detector on real ARIS 3000 imagery.

Runs as a Kaggle GPU kernel. The dataset is the public marine debris FLS
collection captured with an ARIS Explorer 3000 in a water tank, repackaged in
YOLO format, plus added mine targets.

Two environment quirks are handled explicitly:

  the input mount path varies, so the dataset is located by searching for its
  data.yaml rather than being hard coded

  Kaggle may allocate a Tesla P100, which is compute capability 6.0. Recent
  preinstalled PyTorch builds drop Pascal support, so a compatible build is
  installed when that card is detected.

Outputs written to /kaggle/working: best.pt, metrics.json, results.csv
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sh(*args):
    print("$", " ".join(args), flush=True)
    subprocess.run(list(args), check=False)


# ---------------------------------------------------------------- locate data
INPUT = Path("/kaggle/input")
print("input tree:")
for p in sorted(INPUT.rglob("*"))[:15]:
    print("   ", p)

cands = list(INPUT.rglob("data.yaml"))
if not cands:
    cands = [p for p in INPUT.rglob("images") if p.is_dir()]
    if not cands:
        raise SystemExit("could not locate the dataset under /kaggle/input")
    ROOT = cands[0].parent
    YAML = None
else:
    YAML = cands[0]
    ROOT = YAML.parent
print("dataset root:", ROOT)

# ---------------------------------------------------------------- gpu / torch
import torch  # noqa: E402

need_reinstall = False
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"gpu {name} capability {cap}, torch {torch.__version__}")
    supported = torch.cuda.get_arch_list()
    print("torch arch list:", supported)
    if f"sm_{cap[0]}{cap[1]}" not in supported:
        need_reinstall = True
        print("this GPU is not supported by the installed torch, reinstalling")
else:
    print("no CUDA device")

if need_reinstall:
    sh(sys.executable, "-m", "pip", "install", "-q",
       "torch==2.5.1", "torchvision==0.20.1",
       "--index-url", "https://download.pytorch.org/whl/cu121")
    os.execv(sys.executable, [sys.executable] + sys.argv)  # restart with new torch

sh(sys.executable, "-m", "pip", "install", "-q", "ultralytics")

import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available())

WORK = Path("/kaggle/working")
DATA = WORK / "data"
if not DATA.exists():
    shutil.copytree(ROOT, DATA)

cfg = yaml.safe_load(YAML.read_text()) if YAML else {}
if "names" not in cfg:
    raise SystemExit("dataset yaml has no class names")
cfg["path"] = str(DATA)
cfg["train"], cfg["val"], cfg["test"] = "images/train", "images/val", "images/test"
(WORK / "data.yaml").write_text(yaml.safe_dump(cfg))
print(json.dumps(cfg, indent=1))
for split in ("train", "val", "test"):
    print(f"  {split}: {len(list((DATA/'images'/split).glob('*')))} images")

# ---------------------------------------------------------------- train
dev = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8s.pt")
model.train(
    data=str(WORK / "data.yaml"),
    epochs=90, imgsz=640, batch=16, patience=25,
    device=dev, project=str(WORK / "runs"), name="fls", exist_ok=True,
    pretrained=True, optimizer="auto", seed=0, workers=2,
    # Sonar geometry is fixed: the fan is always the same way up and range maps
    # to one axis, so vertical flips and large rotations would produce images
    # the sensor can never generate.
    fliplr=0.5, flipud=0.0, degrees=4.0, translate=0.08, scale=0.35,
    shear=0.0, perspective=0.0, mosaic=0.7, mixup=0.05,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.35,
)

best = WORK / "runs" / "fls" / "weights" / "best.pt"
shutil.copy(best, WORK / "best.pt")

out = {}
for split in ("val", "test"):
    m = YOLO(str(best)).val(data=str(WORK / "data.yaml"), split=split,
                            imgsz=640, device=dev)
    out[split] = {
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class_mAP50": {cfg["names"][int(i)]: float(v)
                            for i, v in zip(m.box.ap_class_index, m.box.ap50)},
    }
    print(split, json.dumps(out[split], indent=1), flush=True)

out["classes"] = cfg["names"]
out["torch"] = torch.__version__
(WORK / "metrics.json").write_text(json.dumps(out, indent=1))
csv = WORK / "runs" / "fls" / "results.csv"
if csv.exists():
    shutil.copy(csv, WORK / "results.csv")
# Keep the output small: the copied dataset would otherwise be published too.
shutil.rmtree(DATA, ignore_errors=True)
shutil.rmtree(WORK / "runs" / "fls" / "weights", ignore_errors=True)
print("done")
