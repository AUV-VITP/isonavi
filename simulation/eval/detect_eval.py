"""Evaluate geometric target detection from the mapped bathymetry."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from isonavi.scene import DisasterSite
from isonavi.mapping import (BathymetryMap, detect_objects_from_residual,
                            match_detections)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, LOG = f"{ROOT}/results/figures", f"{ROOT}/results/logs"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 145, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})

d = np.load(f"{LOG}/mission_isonavi_s1.npz", allow_pickle=True)
site = DisasterSite()
bm = BathymetryMap(site.cfg.x_min, site.cfg.x_max, site.cfg.y_min, site.cfg.y_max, 0.5)
bm.sum, bm.count = d["map_sum"], d["map_count"]

dets, res, bed = detect_objects_from_residual(bm)
PRIMARY = {k: v for k, v in site.targets.items()
           if v["class"] in ("vehicle_large", "vehicle_small", "structure")}
m_all, miss_all, fp_all = match_detections(dets, site.targets, gate=4.0)
m_pri, miss_pri, _ = match_detections(dets, PRIMARY, gate=5.0)

print("=" * 64)
print("GEOMETRIC TARGET DETECTION FROM MAPPED BATHYMETRY")
print("=" * 64)
print(f"  clusters found            : {len(dets)}")
print()
print("  PRIMARY TARGETS (vehicles and structure)")
for x in sorted(m_pri, key=lambda a: a["error"]):
    t = site.targets[x["name"]]
    print(f"    {x['name']:15s} {x['class']:14s} localisation error {x['error']:5.2f} m"
          f"   height {x['det']['height']:.2f} m")
for x in miss_pri:
    print(f"    {x['name']:15s} {x['class']:14s} MISSED")
errs = [x["error"] for x in m_pri]
print(f"    detected {len(m_pri)}/{len(PRIMARY)}, "
      f"mean localisation error {np.mean(errs):.2f} m, max {np.max(errs):.2f} m")
print()
byc = {}
for x in m_all:
    byc.setdefault(x["class"], [0, 0])[0] += 1
for x in miss_all:
    byc.setdefault(x["class"], [0, 0])[1] += 1
print("  ALL REGISTERED OBJECTS, by class")
for c, (a, b) in sorted(byc.items()):
    print(f"    {c:15s} {a}/{a+b} detected ({100*a/max(a+b,1):.0f} %)")
print()
print("  Small boulders below roughly 1 m across fall under the map cell size")
print("  and the sonar footprint, so they are not separable from bed texture.")

# ---------------------------------------------------------------- figure
mx = bm.x0 + np.arange(res.shape[1]) * bm.res
my = bm.y0 + np.arange(res.shape[0]) * bm.res
fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))

im = ax[0].pcolormesh(mx, my, np.where(bm.count > 0, res, np.nan),
                      shading="auto", cmap="magma", vmin=0, vmax=3.2)
plt.colorbar(im, ax=ax[0], fraction=0.035, label="height above fitted bed (m)")
ax[0].set_title("Residual above the bare-bed surface")
ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")

ax[1].pcolormesh(mx, my, np.where(bm.count > 0, res, np.nan),
                 shading="auto", cmap="Greys", vmin=0, vmax=2.5)
for det in dets:
    ax[1].plot(*det["centre"], "o", ms=4, mfc="none", mec="#1f6feb", mew=1.0)
for name, tg in site.targets.items():
    cls = tg["class"]
    col = {"vehicle_large": "#d1242f", "vehicle_small": "#bf3989",
           "structure": "#8250df"}.get(cls, "#57606a")
    big = cls in ("vehicle_large", "vehicle_small", "structure")
    ax[1].plot(tg["centre"][0], tg["centre"][1], "x", color=col,
               ms=9 if big else 4, mew=1.8 if big else 1.0)
    if big:
        ax[1].annotate(name, (tg["centre"][0], tg["centre"][1]), fontsize=7.5,
                       color=col, xytext=(6, 5), textcoords="offset points")
for x in m_pri:
    ax[1].plot([x["truth"][0], x["detected"][0]],
               [x["truth"][1], x["detected"][1]], "-", color="#1a7f37", lw=1.6)
ax[1].set_title(f"Detections against ground truth\n"
                f"primary targets {len(m_pri)}/{len(PRIMARY)}, "
                f"mean error {np.mean(errs):.2f} m")
ax[1].set_xlabel("x (m)")
for a in ax:
    a.set_aspect("equal"); a.set_xlim(-15, 60); a.set_ylim(-28, 28)
plt.tight_layout(); plt.savefig(f"{FIG}/f7_detection.png"); plt.close()

json.dump({
    "clusters": len(dets),
    "primary_detected": len(m_pri),
    "primary_total": len(PRIMARY),
    "primary_mean_error_m": float(np.mean(errs)),
    "primary_max_error_m": float(np.max(errs)),
    "by_class": {c: {"detected": a, "total": a + b} for c, (a, b) in byc.items()},
    "matches": [{"name": x["name"], "class": x["class"], "error_m": x["error"],
                 "height_m": x["det"]["height"]} for x in m_pri],
}, open(f"{LOG}/detection_metrics.json", "w"), indent=1)
print()
print("  wrote results/figures/f7_detection.png and detection_metrics.json")
