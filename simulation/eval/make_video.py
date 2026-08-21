"""Render an operator-console video of the reference mission.

Layout per frame:

    left    plan view: bathymetry, track, targets, live vehicle
    right   forward-looking sonar and the bathymetric map building up
    centre  vertical profile against the riverbed
    bottom  telemetry strip and a mission phase timeline

The map is accumulated honestly: every sonar ping up to the current video
time is rendered once and folded into the map, so the operator view fills in
at the rate the survey actually achieves.
"""
import argparse
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle

from varuna.scene import DisasterSite
from varuna.acoustics import ForwardLookingSonar, preset
from varuna.mapping import BathymetryMap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG, VID = f"{ROOT}/results/logs", f"{ROOT}/results/video"
os.makedirs(VID, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--seconds", type=float, default=45.0)
ap.add_argument("--dpi", type=int, default=100)
ap.add_argument("--tag", default="varuna_s1")
args = ap.parse_args()

d = np.load(f"{LOG}/mission_{args.tag}.npz", allow_pickle=True)
site = DisasterSite()
t, eta, est, phase = d["t"], d["eta"], d["est_pos"], d["phase"]
nav_err = np.linalg.norm(est - eta[:, :3], axis=1)
bed_under = site.bed_height(eta[:, 0], eta[:, 1])
ping_t, ping_pose = d["ping_t"], d["ping_pose"]

# The DVL reports at 8 Hz while the log runs at the control rate, so rows
# between reports carry no sample. Hold the last report for display.
lock_raw = d["dvl_lock"].astype(bool)
lock_disp = np.zeros_like(lock_raw)
cur = True
for i, v in enumerate(lock_raw):
    if i % 4 == 0:
        cur = bool(v)
    lock_disp[i] = cur

n_frames = int(args.fps * args.seconds)
idx = np.linspace(0, len(t) - 1, n_frames).astype(int)

cfg = preset("oculus", seed=5, r_max=45.0, ssc_g_per_l=site.cfg.ssc_g_per_l)
fls = ForwardLookingSonar(cfg, site.scene)
bm = BathymetryMap(site.cfg.x_min, site.cfg.x_max, site.cfg.y_min, site.cfg.y_max, 0.6)

PH_COL = {"DEPLOY": "#a371f7", "ACQUIRE": "#e3b341", "SEARCH": "#58a6ff",
          "INSPECT": "#3fb950", "RETURN": "#8b949e", "REPORT": "#8b949e",
          "DONE": "#8b949e"}
PH_ORDER = ["DEPLOY", "ACQUIRE", "SEARCH", "INSPECT", "RETURN", "REPORT", "DONE"]
BG, FG, GRID = "#0b1020", "#e6edf3", "#243049"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": "#111a2e", "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": "#8b98b0",
    "ytick.color": "#8b98b0", "axes.edgecolor": GRID, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

spans, curp, t0 = [], phase[0], t[0]
for i in range(1, len(phase)):
    if phase[i] != curp:
        spans.append((curp, t0, t[i]))
        curp, t0 = phase[i], t[i]
spans.append((curp, t0, t[-1]))

frames_dir = f"{VID}/frames"
os.makedirs(frames_dir, exist_ok=True)
for f in os.listdir(frames_dir):
    os.remove(os.path.join(frames_dir, f))

sonar_img = None
next_ping = 0
print(f"rendering {n_frames} frames, {len(ping_t)} pings to fold in", flush=True)

for fi, k in enumerate(idx):
    now = t[k]
    # Fold in every ping that has happened by now.
    while next_ping < len(ping_t) and ping_t[next_ping] <= now:
        fr = fls.ping(ping_pose[next_ping])
        if fr.all_point is not None and len(fr.all_point):
            bm.add(fr.all_point, max_incidence_deg=78.0, incidence=fr.all_incidence)
        sonar_img = fr.to_cartesian(320)
        next_ping += 1

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1.5, 1, 1],
                  height_ratios=[1.05, 0.95, 0.5], hspace=0.45, wspace=0.26,
                  left=0.045, right=0.985, top=0.90, bottom=0.055)

    # ------------------------------------------------------------- plan view
    axp = fig.add_subplot(gs[0:2, 0])
    axp.pcolormesh(site.xs, site.ys, site.H, shading="auto", cmap="gist_earth",
                   alpha=0.9)
    for ph, a, b in spans:
        if a > now:
            break
        m = (t >= a) & (t <= min(b, now))
        axp.plot(eta[m, 0], eta[m, 1], lw=1.7, color=PH_COL.get(ph, "#ccc"))
    for i, p in enumerate(site.cfg.piers):
        axp.add_patch(Circle((p.x, p.y), p.radius, fc="#2b3550", ec="#c9d1d9",
                             lw=1.0, zorder=5))
        axp.annotate(f"P{i+1}", (p.x, p.y), color="w", ha="center", va="center",
                     fontsize=7, zorder=6)
    for name, tg in site.targets.items():
        if tg["class"] in ("vehicle_large", "vehicle_small", "structure"):
            axp.plot(tg["centre"][0], tg["centre"][1], marker="s", ms=8,
                     mfc="none", mec="#ff7b72", mew=1.7, zorder=7)
    yaw = eta[k, 5]
    axp.plot(eta[k, 0], eta[k, 1], marker="o", ms=9, color="#ffd33d", zorder=9)
    axp.arrow(eta[k, 0], eta[k, 1], 6.5 * np.cos(yaw), 6.5 * np.sin(yaw),
              head_width=2.4, color="#ffd33d", zorder=9, length_includes_head=True)
    axp.set_xlim(-58, 64); axp.set_ylim(-33, 33); axp.set_aspect("equal")
    axp.set_title("PLAN VIEW", loc="left", fontsize=11, color=FG)
    axp.set_xlabel("x downstream (m)"); axp.set_ylabel("y cross-stream (m)")

    # ------------------------------------------------------------- sonar
    axs = fig.add_subplot(gs[0, 1])
    if sonar_img is not None:
        axs.imshow(sonar_img, cmap="inferno", aspect="auto")
    axs.axis("off")
    axs.set_title("FORWARD-LOOKING SONAR", loc="left", fontsize=9.5)

    # ------------------------------------------------------------- map
    axm = fig.add_subplot(gs[0, 2])
    axm.pcolormesh(bm.xs, bm.ys, bm.mean_z, shading="auto", cmap="terrain")
    axm.set_xlim(-14, 58); axm.set_ylim(-26, 26); axm.set_aspect("equal")
    axm.set_title(f"BATHYMETRIC MAP   {bm.n_soundings//1000}k soundings",
                  loc="left", fontsize=9.5)
    axm.tick_params(labelsize=7)

    # ------------------------------------------------------------- profile
    axz = fig.add_subplot(gs[1, 1:])
    w = slice(max(0, k - 2600), k + 1)
    axz.plot(t[w], eta[w, 2], color="#39d3ff", lw=1.5, label="vehicle")
    axz.plot(t[w], bed_under[w], color="#b07d4a", lw=1.5, label="riverbed")
    axz.fill_between(t[w], bed_under[w], eta[w, 2], color="#39d3ff", alpha=0.10)
    axz.set_xlabel("mission time (s)"); axz.set_ylabel("elevation (m)")
    axz.legend(loc="lower left", fontsize=8, ncol=2)
    axz.set_title("VERTICAL PROFILE AND TERRAIN FOLLOWING", loc="left", fontsize=9.5)

    # ------------------------------------------------------------- telemetry
    axt = fig.add_subplot(gs[2, :])
    axt.axis("off")
    alt = eta[k, 2] - bed_under[k]
    tele = [
        ("TIME", f"{now:6.1f} s", FG),
        ("PHASE", str(phase[k]), PH_COL.get(phase[k], FG)),
        ("DEPTH", f"{-eta[k,2]:5.2f} m", FG),
        ("ALTITUDE", f"{alt:4.2f} m", FG),
        ("HEADING", f"{np.degrees(yaw) % 360:5.1f}", FG),
        ("SPEED", f"{np.linalg.norm(d['nu'][k][:3]):4.2f} m/s", FG),
        ("CURRENT", f"{d['current'][k]:4.2f} m/s", "#e3b341"),
        ("NAV ERR", f"{nav_err[k]:4.2f} m", "#3fb950" if nav_err[k] < 1 else "#e3b341"),
        ("DVL", "LOCK" if lock_disp[k] else "NO LOCK",
         "#3fb950" if lock_disp[k] else "#f85149"),
        ("THRUST", f"{d['thrust'][k]:5.1f} N", FG),
    ]
    for i, (kk, vv, col) in enumerate(tele):
        x = 0.010 + i * 0.0995
        axt.text(x, 0.80, kk, fontsize=8, color="#8b98b0", transform=axt.transAxes)
        axt.text(x, 0.44, vv, fontsize=13.5, color=col, transform=axt.transAxes,
                 family="DejaVu Sans Mono")

    # Phase timeline.
    T = t[-1]
    for ph, a, b in spans:
        axt.add_patch(Rectangle((0.010 + 0.978 * a / T, 0.05),
                                0.978 * (b - a) / T, 0.12,
                                transform=axt.transAxes,
                                fc=PH_COL.get(ph, "#555"), alpha=0.85, lw=0))
    axt.plot([0.010 + 0.978 * now / T] * 2, [0.02, 0.20], color="#ffd33d", lw=2.0,
             transform=axt.transAxes)
    for ph in PH_ORDER:
        for p2, a, b in spans:
            if p2 == ph and (b - a) / T > 0.05:
                axt.text(0.010 + 0.978 * (a + b) / 2 / T, 0.19, ph, fontsize=7,
                         color=PH_COL.get(ph, FG), ha="center",
                         transform=axt.transAxes)
                break

    fig.suptitle("VARUNA   autonomous underwater reconnaissance and assessment",
                 x=0.045, ha="left", fontsize=15, color=FG, y=0.965)
    fig.text(0.985, 0.965,
             "Savitri river post-collapse scenario   2.4 m/s current   3.2 g/L sediment",
             ha="right", fontsize=9.5, color="#8b98b0")

    fig.savefig(f"{frames_dir}/f{fi:05d}.png", dpi=args.dpi)
    plt.close(fig)
    if fi % 60 == 0:
        print(f"  frame {fi}/{n_frames}  t={now:.0f}s  {phase[k]}  "
              f"{bm.n_soundings} soundings", flush=True)

out = f"{VID}/varuna_mission.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                "-i", f"{frames_dir}/f%05d.png", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                out], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf",
                "fps=10,scale=1000:-1:flags=lanczos", f"{VID}/varuna_mission.gif"],
               check=False)
print("wrote", out, os.path.getsize(out) // 1024, "kB")
