"""Aggregate mission results across seeds.

A single run says little about whether the system works. Repeating the mission
with different sensor noise, DVL dropout patterns and speckle realisations, and
reporting the spread, is what distinguishes a result from an anecdote.

The scene itself is fixed, so what varies between seeds is everything
stochastic in the sensing and perception chain.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from varuna.scene import DisasterSite
from varuna.mapping import (BathymetryMap, detect_objects_from_residual,
                            match_detections)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, LOG = f"{ROOT}/results/figures", f"{ROOT}/results/logs"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 145, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})

site = DisasterSite()
PRIMARY = {k: v for k, v in site.targets.items()
           if v["class"] in ("vehicle_large", "vehicle_small", "structure")}

runs = []
for f in sorted(glob.glob(f"{LOG}/mission_varuna_s*.json")):
    s = json.load(open(f))
    seed = s.get("seed")
    npz = f.replace(".json", ".npz")
    if not os.path.exists(npz):
        continue
    d = np.load(npz, allow_pickle=True)

    bm = BathymetryMap(site.cfg.x_min, site.cfg.x_max,
                       site.cfg.y_min, site.cfg.y_max, 0.5)
    bm.sum, bm.count = d["map_sum"], d["map_count"]

    mean_z = bm.mean_z
    Xg, Yg = np.meshgrid(bm.xs, bm.ys)
    diff = mean_z - site.bed_height(Xg, Yg)
    ok = np.isfinite(diff)
    map_rmse = float(np.sqrt(np.nanmean(diff[ok] ** 2)))

    dets, _, _ = detect_objects_from_residual(bm)
    m, miss, fp = match_detections(dets, PRIMARY, gate=5.0)
    errs = [x["error"] for x in m]

    # Recall against the primary targets is only half the picture. The
    # detector flags any bed anomaly above its height threshold, so the honest
    # figure is how many contacts it raises that correspond to nothing in the
    # scene at all. Matching against every object, boulders and debris
    # included, separates real bed features from spurious returns.
    m_any, _, fp_any = match_detections(dets, site.targets, gate=5.0)

    sc = s.get("scour", {}) or {}
    rec = [100 * v["max_depth"] / v["truth_depth"]
           for v in sc.values() if v and v.get("truth_depth")]

    runs.append({
        "seed": seed,
        "duration": s["duration"],
        "coverage": s["coverage"] * 100,
        "nav_mean": s["nav_error_mean"],
        "nav_max": s["nav_error_max"],
        "dvl": s["dvl_availability"] * 100,
        "map_rmse": map_rmse,
        "detected": len(m),
        "det_total": len(PRIMARY),
        "det_err": float(np.mean(errs)) if errs else np.nan,
        "scour_rec": float(np.mean(rec)) if rec else np.nan,
        "completed": s["phases_reached"][-1] == "DONE",
        "n_det": len(dets),
        "on_object": len(m_any),
        "false_alarms": len(fp_any),
    })

if not runs:
    raise SystemExit("no mission runs found")

print("=" * 78)
print(f"MISSION REPEATABILITY OVER {len(runs)} SEEDS")
print("=" * 78)
hdr = (f"{'seed':>5}{'done':>6}{'cover%':>8}{'navMean':>9}{'navMax':>8}"
       f"{'DVL%':>7}{'mapRMSE':>9}{'targets':>9}{'detErr':>8}{'scour%':>8}")
print(hdr)
for r in runs:
    print(f"{r['seed']:>5}{'yes' if r['completed'] else 'NO':>6}"
          f"{r['coverage']:>8.1f}{r['nav_mean']:>9.2f}{r['nav_max']:>8.2f}"
          f"{r['dvl']:>7.1f}{r['map_rmse']:>9.3f}"
          f"{r['detected']}/{r['det_total']:<7}{r['det_err']:>8.2f}"
          f"{r['scour_rec']:>8.0f}")


def ms(key):
    v = np.array([r[key] for r in runs], dtype=float)
    v = v[np.isfinite(v)]
    return v.mean(), v.std()


print("-" * 78)
summary = {}
for key, label, fmt in (("coverage", "search coverage (%)", "{:.1f}"),
                        ("nav_mean", "nav error mean (m)", "{:.3f}"),
                        ("nav_max", "nav error max (m)", "{:.3f}"),
                        ("dvl", "DVL availability (%)", "{:.1f}"),
                        ("map_rmse", "map RMSE (m)", "{:.3f}"),
                        ("det_err", "detection error (m)", "{:.2f}"),
                        ("scour_rec", "scour recovery (%)", "{:.0f}")):
    mu, sd = ms(key)
    summary[key] = {"mean": float(mu), "std": float(sd)}
    print(f"  {label:24s} {fmt.format(mu)} +/- {fmt.format(sd)}")

n_done = sum(r["completed"] for r in runs)
n_full = sum(r["detected"] == r["det_total"] for r in runs)
print(f"  {'missions completed':24s} {n_done}/{len(runs)}")
print(f"  {'all primary targets found':24s} {n_full}/{len(runs)}")
fa = [r["false_alarms"] for r in runs]
nd = [r["n_det"] for r in runs]
area_ha = ((site.cfg.x_max - site.cfg.x_min)
           * (site.cfg.y_max - site.cfg.y_min)) / 1e4
print(f"  {'contacts raised per run':24s} {np.mean(nd):.1f}")
print(f"  {'false alarms per run':24s} {np.mean(fa):.1f} +/- {np.std(fa):.1f}")
print(f"  {'false alarms per hectare':24s} {np.mean(fa) / area_ha:.1f}")
summary["contacts_per_run"] = float(np.mean(nd))
summary["false_alarms_per_run"] = float(np.mean(fa))
summary["false_alarms_std"] = float(np.std(fa))
summary["false_alarms_per_hectare"] = float(np.mean(fa) / area_ha)
summary["survey_area_ha"] = float(area_ha)
summary["missions_completed"] = n_done
summary["runs"] = len(runs)
summary["all_targets_found"] = n_full
summary["per_run"] = runs

json.dump(summary, open(f"{LOG}/repeatability.json", "w"), indent=1)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9))
panels = [("coverage", "search coverage (%)", 100),
          ("nav_mean", "mean navigation error (m)", None),
          ("map_rmse", "bathymetric map RMSE (m)", None),
          ("det_err", "target localisation error (m)", None)]
for a, (key, label, ref) in zip(ax, panels):
    vals = np.array([r[key] for r in runs], dtype=float)
    seeds = [r["seed"] for r in runs]
    a.bar([str(s) for s in seeds], vals, color="#1f6feb", width=0.62)
    mu = np.nanmean(vals)
    a.axhline(mu, color="#d1242f", ls="--", lw=1.1, label=f"mean {mu:.2f}")
    if ref:
        a.set_ylim(0, ref * 1.05)
    a.set_xlabel("seed")
    a.set_title(label, fontsize=9.5)
    a.legend(fontsize=8)
plt.suptitle(f"Mission repeatability over {len(runs)} independent runs "
             f"(sensor noise, DVL dropout and speckle re-randomised)", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f11_repeatability.png", bbox_inches="tight")
plt.close()
print()
print("wrote f11_repeatability.png and repeatability.json")
