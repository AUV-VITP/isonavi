"""Symmetric cross-domain evaluation between real and simulated sonar.

Both directions are measured the same way, with standard detection mAP on a
held-out split, so the two numbers are directly comparable:

    real-trained model    -> simulated validation split
    simulator-trained     -> real test split   (measured on Kaggle)

The simulated split used here is the same one the synthetic training set was
generated with, so the sensor geometry, clutter and labelling are identical to
what a simulator-trained model would have seen. An earlier version of this test
used a different, sparser frame geometry, which made the simulated frames
unrepresentative and the comparison unfair.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = f"{ROOT}/results/logs"
SYNTH = os.path.expanduser("~/dev/datasets/fls_synth")
REAL = os.path.expanduser("~/dev/datasets/fls_sonar/Sonar_Dataset")
MODEL = f"{ROOT}/ml/models/fls_yolov8s.pt"

import yaml
from ultralytics import YOLO

CLASSES = ["mine", "can", "bottle", "drink-carton", "chain", "propeller",
           "tire", "hook", "valve", "shampoo-bottle", "standing-bottle"]

tmp = "/tmp/xdomain"
os.makedirs(tmp, exist_ok=True)
syn_yaml = f"{tmp}/synth.yaml"
with open(syn_yaml, "w") as fh:
    yaml.safe_dump({"path": SYNTH, "train": "images/train", "val": "images/val",
                    "names": {i: c for i, c in enumerate(CLASSES)}}, fh)

print("=" * 70)
print("CROSS-DOMAIN EVALUATION")
print("=" * 70)
n_val = len(os.listdir(f"{SYNTH}/images/val"))
print(f"  real-trained detector -> simulated validation split ({n_val} images)")

m = YOLO(MODEL).val(data=syn_yaml, split="val", imgsz=640, device="cpu",
                    verbose=False)
out = {
    "real_to_sim": {
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "n_images": n_val,
    }
}
print(f"    mAP@0.5      {100*m.box.map50:.1f} %")
print(f"    mAP@0.5:0.95 {100*m.box.map:.1f} %")
print(f"    precision    {100*m.box.mp:.1f} %")
print(f"    recall       {100*m.box.mr:.1f} %")

st = f"{LOG}/synth_transfer_metrics.json"
if os.path.exists(st):
    s = json.load(open(st))
    print()
    print("  simulator-trained detector -> real test split")
    print(f"    mAP@0.5      {100*s['mAP50']:.1f} %")
    print(f"    mAP@0.5:0.95 {100*s['mAP50_95']:.1f} %")
    print()
    print("  same model on its own simulated split")
    print(f"    mAP@0.5      {100*s['sim_val_mAP50']:.1f} %")
    out["sim_to_real"] = {"mAP50": s["mAP50"], "mAP50_95": s["mAP50_95"]}
    out["sim_self"] = {"mAP50": s["sim_val_mAP50"],
                       "mAP50_95": s["sim_val_mAP50_95"]}

rt = f"{ROOT}/ml/models/train_metrics.json"
if os.path.exists(rt):
    r = json.load(open(rt))
    out["real_self"] = {"mAP50": r["test"]["mAP50"],
                        "mAP50_95": r["test"]["mAP50_95"]}
    print()
    print("  real-trained model on its own real test split")
    print(f"    mAP@0.5      {100*r['test']['mAP50']:.1f} %")

json.dump(out, open(f"{LOG}/cross_domain_metrics.json", "w"), indent=1)
print()
print("  wrote cross_domain_metrics.json")
