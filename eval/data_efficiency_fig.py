"""Plot the real-data efficiency curve and report the pretraining gain."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.expanduser("~/dev/rakshatech")
FIG, LOG = f"{ROOT}/results/figures", f"{ROOT}/results/logs"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 145, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})

src = f"{LOG}/data_efficiency.json"
if not os.path.exists(src):
    raise SystemExit("data_efficiency.json not present yet")
rows = json.load(open(src))
ns = sorted({r["n_real"] for r in rows})


def series(arm, key="mAP50"):
    return [next((r[key] for r in rows if r["n_real"] == n and r["arm"] == arm), np.nan)
            for n in ns]


coco = np.array(series("coco")) * 100
sim = np.array(series("sim")) * 100
coco95 = np.array(series("coco", "mAP50_95")) * 100
sim95 = np.array(series("sim", "mAP50_95")) * 100

fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6))
ax[0].plot(ns, coco, "o-", color="#57606a", lw=1.8, ms=6,
           label="natural-image initialisation")
ax[0].plot(ns, sim, "s-", color="#1f6feb", lw=1.8, ms=6,
           label="simulator pretraining")
ax[0].set_xscale("log")
ax[0].set_xticks(ns)
ax[0].set_xticklabels([str(n) for n in ns])
ax[0].set_xlabel("real training images")
ax[0].set_ylabel("mAP@0.5 on the real test split (%)")
ax[0].set_title("Real-data efficiency")
ax[0].legend(loc="lower right", fontsize=8.5)

gain = sim - coco
colors = ["#1a7f37" if g >= 0 else "#d1242f" for g in gain]
ax[1].bar([str(n) for n in ns], gain, color=colors, width=0.6)
ax[1].axhline(0, color="k", lw=0.9)
ax[1].set_xlabel("real training images")
ax[1].set_ylabel("mAP@0.5 difference (percentage points)")
ax[1].set_title("Gain from simulator pretraining")
for i, g in enumerate(gain):
    ax[1].annotate(f"{g:+.1f}", (i, g), ha="center",
                   va="bottom" if g >= 0 else "top", fontsize=8.5)
plt.tight_layout()
plt.savefig(f"{FIG}/f10_data_efficiency.png")
plt.close()

print(f"{'N real':>8}{'natural':>11}{'simulator':>12}{'gain':>9}")
for n, c, s in zip(ns, coco, sim):
    print(f"{n:>8}{c:>10.1f}%{s:>11.1f}%{s-c:>+9.1f}")
print()
best = int(np.nanargmax(gain))
print(f"largest gain: {gain[best]:+.1f} points at N = {ns[best]}")

# How many real images does the natural-image arm need to match what the
# simulator-pretrained arm reaches at the smallest N?
target = sim[0]
need = None
for n, c in zip(ns, coco):
    if c >= target:
        need = n
        break
if need is not None and need > ns[0]:
    print(f"natural-image arm needs {need} real images to match what "
          f"simulator pretraining reaches with {ns[0]}")

json.dump({"n": ns, "coco_mAP50": coco.tolist(), "sim_mAP50": sim.tolist(),
           "coco_mAP": coco95.tolist(), "sim_mAP": sim95.tolist(),
           "gain": gain.tolist(), "best_gain": float(gain[best]),
           "best_n": int(ns[best])},
          open(f"{LOG}/data_efficiency_summary.json", "w"), indent=1)
print("\nwrote f10_data_efficiency.png")
