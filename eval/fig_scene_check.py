"""Build the disaster site, render sonar frames, and compare against real data."""
import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from varuna.scene import DisasterSite, SiteConfig
from varuna.acoustics import ForwardLookingSonar, preset

OUT = os.path.expanduser("~/dev/rakshatech/results/figures")
os.makedirs(OUT, exist_ok=True)
REAL = os.path.expanduser("~/dev/datasets/fls_sonar/Sonar_Dataset/images/train")

t0 = time.perf_counter()
site = DisasterSite()
print(site.summary())
print(f"build time: {time.perf_counter()-t0:.2f} s")

# ---------------------------------------------------------------- bathymetry
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
im = ax[0].pcolormesh(site.xs, site.ys, site.H, shading="auto", cmap="viridis")
ax[0].set_title("Bathymetry (m, z up)")
ax[0].set_xlabel("x downstream (m)"); ax[0].set_ylabel("y cross-stream (m)")
ax[0].set_aspect("equal")
plt.colorbar(im, ax=ax[0], label="bed elevation (m)")
for name, t in site.targets.items():
    ax[0].plot(t["centre"][0], t["centre"][1], "rx", ms=7)
    ax[0].annotate(name, (t["centre"][0], t["centre"][1]), fontsize=7,
                   color="w", xytext=(3, 3), textcoords="offset points")

# Current magnitude on a horizontal slice 2 m above the bed.
X, Y = np.meshgrid(site.xs[::4], site.ys[::4])
Z = site.bed_height(X, Y) + 2.0
V = site.current(X, Y, Z)
speed = np.linalg.norm(V, axis=-1)
im2 = ax[1].pcolormesh(site.xs[::4], site.ys[::4], speed, shading="auto", cmap="magma")
ax[1].set_title("Current speed 2 m above bed (m/s)")
ax[1].set_xlabel("x downstream (m)"); ax[1].set_aspect("equal")
plt.colorbar(im2, ax=ax[1], label="m/s")
plt.tight_layout()
plt.savefig(f"{OUT}/site_overview.png", dpi=130)
plt.close()
print(f"current: min {speed.min():.2f} max {speed.max():.2f} m/s")

# ---------------------------------------------------------------- sonar frames
cfg = preset("oculus", seed=3, r_max=45.0, ssc_g_per_l=site.cfg.ssc_g_per_l)
fls = ForwardLookingSonar(cfg, site.scene)

views = [
    ("approach to pier P2", [-26.0, -8.0, -7.0, 0, np.radians(9), 0.0]),
    ("bus 1 in debris field", [10.0, 6.5, -8.5, 0, np.radians(11), np.radians(-2)]),
    ("collapsed span", [-6.0, 17.0, -7.5, 0, np.radians(13), 0.0]),
    ("wide search transit", [30.0, -9.0, -7.5, 0, np.radians(10), np.radians(178)]),
]

fig, axes = plt.subplots(2, 4, figsize=(19, 9))
for k, (title, pose) in enumerate(views):
    t0 = time.perf_counter()
    fr = fls.ping(pose)
    dt = time.perf_counter() - t0
    axes[0, k].imshow(fr.normalised(), aspect="auto", cmap="inferno", origin="lower",
                      extent=[np.degrees(fr.bearings[0]), np.degrees(fr.bearings[-1]),
                              fr.ranges[0], fr.ranges[-1]])
    axes[0, k].set_title(f"{title}\npolar, {dt*1000:.0f} ms", fontsize=9)
    axes[0, k].set_xlabel("bearing (deg)")
    if k == 0:
        axes[0, k].set_ylabel("slant range (m)")
    axes[1, k].imshow(fr.to_cartesian(420), cmap="inferno")
    axes[1, k].set_title("fan view", fontsize=9)
    axes[1, k].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/sonar_views.png", dpi=125)
plt.close()

# ---------------------------------------------------------------- realism check
real_files = sorted(glob.glob(f"{REAL}/*.png"))[:4]
print(f"real samples found: {len(real_files)}")
if real_files:
    import cv2
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    for k, f in enumerate(real_files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        axes[0, k].imshow(img, cmap="inferno")
        axes[0, k].set_title(f"REAL ARIS 3000\n{os.path.basename(f)[:30]}", fontsize=8)
        axes[0, k].axis("off")
    acfg = preset("aris", seed=7, ssc_g_per_l=1.0)
    afls = ForwardLookingSonar(acfg, site.scene)
    aris_views = [
        [16.0, 6.5, -9.6, 0, np.radians(16), 0.0],
        [40.0, 3.0, -10.4, 0, np.radians(18), 0.0],
        [-8.0, -8.0, -9.0, 0, np.radians(14), 0.0],
        [2.0, 15.0, -9.4, 0, np.radians(15), np.radians(10)],
    ]
    for k, pose in enumerate(aris_views):
        fr = afls.ping(pose)
        axes[1, k].imshow(fr.to_cartesian(430), cmap="inferno")
        axes[1, k].set_title("SIMULATED (this work)", fontsize=8)
        axes[1, k].axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT}/real_vs_sim.png", dpi=125)
    plt.close()

    # First-order statistical comparison of intensity distributions.
    reals = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
             for f in sorted(glob.glob(f"{REAL}/*.png"))[:120]]
    rstack = np.concatenate([r.ravel() for r in reals])
    sims = []
    rng = np.random.default_rng(0)
    for _ in range(40):
        p = [rng.uniform(-40, 50), rng.uniform(-28, 28), 0, 0,
             np.radians(rng.uniform(10, 20)), rng.uniform(0, 2 * np.pi)]
        p[2] = float(site.bed_height(p[0], p[1])) + rng.uniform(2.5, 4.5)
        sims.append(afls.ping(p).normalised().ravel())
    sstack = np.concatenate(sims)
    print()
    print("intensity statistics, normalised to [0, 1]")
    print(f"  real : mean {rstack.mean():.3f}  std {rstack.std():.3f}  "
          f"p50 {np.percentile(rstack,50):.3f}  p95 {np.percentile(rstack,95):.3f}")
    print(f"  sim  : mean {sstack.mean():.3f}  std {sstack.std():.3f}  "
          f"p50 {np.percentile(sstack,50):.3f}  p95 {np.percentile(sstack,95):.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(rstack, bins=80, density=True, alpha=0.55, label="real ARIS 3000")
    ax.hist(sstack, bins=80, density=True, alpha=0.55, label="simulated")
    ax.set_xlabel("normalised intensity"); ax.set_ylabel("density")
    ax.set_title("Intensity distribution, real vs simulated FLS")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/intensity_hist.png", dpi=130)
    plt.close()

print()
print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
