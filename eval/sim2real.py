"""Zero-shot sim-to-real transfer test for the sonar model.

A detector is trained only on real ARIS Explorer 3000 water-tank imagery. It
is then run, without any fine tuning or domain adaptation, on frames produced
by this simulator for equivalent scenes. If it detects the simulated objects,
the simulator is placing target returns, acoustic shadows and speckle inside
the distribution the network learned from real data.

This is a stronger statement about simulator fidelity than any pixel
similarity metric, because it tests exactly the features a perception model
relies on rather than the appearance of the image as a whole.

Reported:
    recall            fraction of simulated objects the real-trained detector finds
    class agreement   fraction of those given the correct class
    confidence        score distribution, compared against the real test set
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from varuna.validation import random_tank, TANK_CLASSES
from varuna.acoustics import ForwardLookingSonar, preset
from varuna.dynamics import rot_body_to_world

ROOT = os.path.expanduser("~/dev/rakshatech")
FIG, LOG = f"{ROOT}/results/figures", f"{ROOT}/results/logs"
MODEL = f"{ROOT}/ml/models/fls_yolov8s.pt"
REAL = os.path.expanduser("~/dev/datasets/fls_sonar/Sonar_Dataset/images/test")
OUTIMG = f"{ROOT}/results/sim2real"
os.makedirs(OUTIMG, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", type=int, default=45)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--size", type=int, default=640)
args = ap.parse_args()

from ultralytics import YOLO
model = YOLO(MODEL)
NAMES = model.names

# Sensor geometry matched to the tank captures: the head sits low and looks
# across the floor, so targets appear at a few metres with long shadows.
CFG = preset("aris", seed=17, r_max=12.0, r_min=0.7, ssc_g_per_l=0.35)


def project(pose, world_xy_z, size, rmax, half_fov, rmin=0.0):
    """World point to pixel in the Cartesian fan image."""
    R = rot_body_to_world(pose[3], pose[4], pose[5])
    local = R.T @ (np.asarray(world_xy_z, float) - pose[:3])
    r = float(np.linalg.norm(local))
    b = float(np.arctan2(local[1], local[0]))
    if r > rmax or abs(b) > half_fov:
        return None
    if r < rmin:
        return None
    X, Y = r * np.sin(b), r * np.cos(b)
    xs0, xs1 = -rmax * np.sin(half_fov), rmax * np.sin(half_fov)
    col = (X - xs0) / (xs1 - xs0) * (size - 1)
    row = (size - 1) - (Y - rmin) / max(rmax - rmin, 1e-6) * (size - 1)
    return col, row, r, b


records = []
gallery = []
rng = np.random.default_rng(0)

for si in range(args.scenes):
    scene, truth = random_tank(rng.integers(3, 6), seed=100 + si)
    fls = ForwardLookingSonar(CFG, scene)
    pitch = np.radians(rng.uniform(8.0, 17.0))
    pose = np.array([0.0, 0.0, float(np.mean([t["z"] for t in truth])) + rng.uniform(1.6, 2.8),
                     0.0, pitch, 0.0])
    fr = fls.ping(pose)
    img = fr.to_cartesian(args.size)
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([u8] * 3, axis=-1)

    half_fov = float(fr.bearings.max())
    rmax = float(fr.ranges.max())
    gt = []
    for t in truth:
        p = project(pose, (t["x"], t["y"], t["z"] + 0.1), args.size, rmax,
                    half_fov, rmin=float(fr.ranges.min()))
        if p is None:
            continue
        col, row, r, b = p
        # Pixel scale at this range, used to size the acceptance region.
        px_per_m = (args.size - 1) / max(rmax - float(fr.ranges.min()), 1e-6)
        rad_px = max(t["radius"] * px_per_m, 9.0)
        gt.append({"class": t["class"], "col": col, "row": row,
                   "r": r, "rad_px": rad_px})

    res = model.predict(rgb, conf=args.conf, imgsz=args.size, verbose=False)[0]
    dets = []
    if res.boxes is not None and len(res.boxes):
        for bx, cf, cl in zip(res.boxes.xywh.cpu().numpy(),
                              res.boxes.conf.cpu().numpy(),
                              res.boxes.cls.cpu().numpy().astype(int)):
            dets.append({"cx": float(bx[0]), "cy": float(bx[1]),
                         "w": float(bx[2]), "h": float(bx[3]),
                         "conf": float(cf), "class": NAMES[cl]})

    used = set()
    for g in gt:
        best, bd = None, np.inf
        for k, dd in enumerate(dets):
            if k in used:
                continue
            dist = np.hypot(dd["cx"] - g["col"], dd["cy"] - g["row"])
            if dist < bd:
                best, bd = k, dist
        hit = best is not None and bd <= max(g["rad_px"] * 2.2, 26.0)
        if hit:
            used.add(best)
        records.append({
            "scene": si, "class": g["class"], "range_m": g["r"],
            "detected": bool(hit),
            "pred_class": dets[best]["class"] if hit else None,
            "conf": dets[best]["conf"] if hit else 0.0,
            "px_error": float(bd) if hit else None,
        })
    if si < 6:
        gallery.append((rgb, gt, dets, si))
    if si % 10 == 0:
        print(f"  scene {si}/{args.scenes}: {len(gt)} targets, {len(dets)} detections",
              flush=True)

# ---------------------------------------------------------------- summary
n = len(records)
det = sum(r["detected"] for r in records)
agree = sum(1 for r in records if r["detected"] and r["pred_class"] == r["class"])
confs = [r["conf"] for r in records if r["detected"]]

print()
print("=" * 66)
print("ZERO-SHOT SIM-TO-REAL TRANSFER")
print("=" * 66)
print(f"  detector trained on : real ARIS 3000 tank imagery only")
print(f"  evaluated on        : {args.scenes} simulated scenes, {n} target instances")
print(f"  recall              : {det}/{n} = {100*det/max(n,1):.1f} %")
print(f"  class agreement     : {agree}/{max(det,1)} = {100*agree/max(det,1):.1f} % of detections")
print(f"  mean confidence     : {np.mean(confs):.3f}" if confs else "  no detections")

by = {}
for r in records:
    b = by.setdefault(r["class"], [0, 0, 0])
    b[1] += 1
    if r["detected"]:
        b[0] += 1
        if r["pred_class"] == r["class"]:
            b[2] += 1
print()
print("  per class")
for c in sorted(by, key=lambda k: -by[k][1]):
    a, tot, ok = by[c]
    print(f"    {c:16s} recall {a:3d}/{tot:3d} = {100*a/tot:5.1f} %"
          f"   correct class {ok:3d}")

# Confidence on the real test set, for comparison.
real_files = sorted([f"{REAL}/{f}" for f in os.listdir(REAL)])[:120]
rc = []
for f in real_files:
    r = model.predict(f, conf=args.conf, imgsz=args.size, verbose=False)[0]
    if r.boxes is not None and len(r.boxes):
        rc.extend(r.boxes.conf.cpu().numpy().tolist())
print()
print(f"  confidence on real test images : mean {np.mean(rc):.3f} (n={len(rc)})")
print(f"  confidence on simulated images : mean {np.mean(confs):.3f} (n={len(confs)})")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 145, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
for ax, (rgb, gt, dets, si) in zip(axes.ravel(), gallery):
    ax.imshow(rgb)
    for g in gt:
        ax.add_patch(plt.Circle((g["col"], g["row"]), g["rad_px"] * 1.6,
                                fill=False, ec="#39d3ff", lw=1.3, ls="--"))
        ax.annotate(g["class"], (g["col"], g["row"] - g["rad_px"] * 1.8),
                    color="#39d3ff", fontsize=7, ha="center")
    for dd in dets:
        ax.add_patch(plt.Rectangle((dd["cx"] - dd["w"] / 2, dd["cy"] - dd["h"] / 2),
                                   dd["w"], dd["h"], fill=False, ec="#3fb950", lw=1.5))
        ax.annotate(f"{dd['class']} {dd['conf']:.2f}",
                    (dd["cx"] - dd["w"] / 2, dd["cy"] + dd["h"] / 2 + 12),
                    color="#3fb950", fontsize=7)
    ax.set_title(f"simulated scene {si}", fontsize=9)
    ax.axis("off")
plt.suptitle("Detector trained only on real sonar, run zero-shot on simulated frames\n"
             "dashed cyan: simulated ground truth      green: detector output", y=0.99)
plt.tight_layout(); plt.savefig(f"{FIG}/f9_sim2real.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax[0].hist(rc, bins=22, alpha=0.6, density=True, label=f"real (n={len(rc)})")
ax[0].hist(confs, bins=22, alpha=0.6, density=True, label=f"simulated (n={len(confs)})")
ax[0].set_xlabel("detection confidence"); ax[0].set_ylabel("density")
ax[0].set_title("Confidence, real versus simulated")
ax[0].legend()

cls = sorted(by, key=lambda k: -by[k][1])
rec = [100 * by[c][0] / by[c][1] for c in cls]
ax[1].barh(cls, rec, color="#1f6feb")
ax[1].set_xlabel("zero-shot recall on simulated frames (%)")
ax[1].set_xlim(0, 100)
ax[1].axvline(100 * det / max(n, 1), color="#d1242f", ls="--",
              label=f"overall {100*det/max(n,1):.0f} %")
ax[1].legend()
plt.tight_layout(); plt.savefig(f"{FIG}/f9b_sim2real_stats.png"); plt.close()

json.dump({
    "scenes": args.scenes, "instances": n, "detected": det,
    "recall": det / max(n, 1),
    "class_agreement": agree / max(det, 1),
    "mean_conf_sim": float(np.mean(confs)) if confs else None,
    "mean_conf_real": float(np.mean(rc)) if rc else None,
    "per_class": {c: {"recall": by[c][0] / by[c][1], "n": by[c][1],
                      "correct_class": by[c][2]} for c in by},
}, open(f"{LOG}/sim2real_metrics.json", "w"), indent=1)
print()
print("  wrote f9_sim2real.png, f9b_sim2real_stats.png, sim2real_metrics.json")
