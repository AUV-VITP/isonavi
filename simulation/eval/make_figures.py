"""Generate the figure set for the technical report from a saved mission run."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from isonavi.scene import DisasterSite
from isonavi.dynamics import (BLUEROV2_HEAVY, isonavi_1, max_holdable_current,
                             vectored_allocation)
from isonavi.acoustics import ForwardLookingSonar, preset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f"{ROOT}/results/figures"
LOG = f"{ROOT}/results/logs"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 145, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
ACC = "#1f6feb"
WARN = "#d1242f"
GOOD = "#1a7f37"

d = np.load(f"{LOG}/mission_isonavi_s1.npz", allow_pickle=True)
summary = json.load(open(f"{LOG}/mission_isonavi_s1.json"))
site = DisasterSite()

t = d["t"]; eta = d["eta"]; est = d["est_pos"]; phase = d["phase"]
nav_err = np.linalg.norm(est - eta[:, :3], axis=1)
PH_COLOURS = {"DEPLOY": "#8250df", "ACQUIRE": "#bf8700", "SEARCH": ACC,
              "INSPECT": GOOD, "RETURN": "#57606a", "REPORT": "#57606a",
              "DONE": "#57606a"}


def phase_spans():
    spans, cur, t0 = [], phase[0], t[0]
    for i in range(1, len(phase)):
        if phase[i] != cur:
            spans.append((cur, t0, t[i]))
            cur, t0 = phase[i], t[i]
    spans.append((cur, t0, t[-1]))
    return spans


def shade_phases(ax):
    for ph, a, b in phase_spans():
        ax.axvspan(a, b, color=PH_COLOURS.get(ph, "#ccc"), alpha=0.08, lw=0)


# =====================================================================  fig 1
fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.6))
im = ax[0].pcolormesh(site.xs, site.ys, site.H, shading="auto", cmap="terrain")
plt.colorbar(im, ax=ax[0], label="bed elevation (m)", fraction=0.035)
for ph, a, b in phase_spans():
    m = (t >= a) & (t <= b)
    ax[0].plot(eta[m, 0], eta[m, 1], lw=1.7,
               color=PH_COLOURS.get(ph, "#333"), label=ph)
for i, p in enumerate(site.cfg.piers):
    ax[0].add_patch(Circle((p.x, p.y), p.radius, fc="0.25", ec="k", zorder=5))
    ax[0].annotate(f"P{i+1}", (p.x, p.y), color="w", ha="center", va="center",
                   fontsize=7, zorder=6)
# Only the operationally significant targets are labelled. Every boulder and
# rubble block is registered as ground truth for the detection evaluation, but
# annotating all of them here would bury the track.
for name, tg in site.targets.items():
    primary = tg["class"] in ("vehicle_large", "vehicle_small", "structure")
    if primary:
        ax[0].plot(tg["centre"][0], tg["centre"][1], marker="s", ms=7,
                   mfc="none", mec=WARN, mew=1.7, zorder=7)
        ax[0].annotate(name, (tg["centre"][0], tg["centre"][1]), fontsize=7,
                       color=WARN, xytext=(5, 5), textcoords="offset points",
                       fontweight="bold")
    else:
        ax[0].plot(tg["centre"][0], tg["centre"][1], marker=".", ms=3.5,
                   color="0.35", zorder=6)
x0, x1, y0, y1 = summary.get("search_box", (-6.0, 52.0, -16.0, 20.0))
ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                              ec="k", ls="--", lw=0.9, alpha=0.6))
ax[0].set_title("Mission track over site bathymetry")
ax[0].set_xlabel("x downstream (m)"); ax[0].set_ylabel("y cross-stream (m)")
ax[0].set_aspect("equal")
h, l = ax[0].get_legend_handles_labels()
uniq = dict(zip(l, h))
ax[0].legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=7, ncol=2)

ax[1].plot(t, eta[:, 2], color=ACC, lw=1.3, label="depth (true)")
bed_under = site.bed_height(eta[:, 0], eta[:, 1])
ax[1].plot(t, bed_under, color="#8b4513", lw=1.2, label="riverbed")
ax[1].fill_between(t, bed_under, eta[:, 2], color=ACC, alpha=0.08)
ax[1].plot(t, d["target"][:, 2], color=GOOD, lw=0.9, ls="--", label="depth setpoint")
shade_phases(ax[1])
ax[1].set_xlabel("mission time (s)"); ax[1].set_ylabel("elevation (m)")
ax[1].set_title("Vertical profile and terrain following")
ax[1].legend(loc="lower left", fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{FIG}/f1_mission_overview.png"); plt.close()

# =====================================================================  fig 2
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
ax[0].plot(t, nav_err, color=ACC, lw=1.2, label="actual error")
ax[0].plot(t, d["sigma"], color=WARN, lw=1.0, ls="--",
           label="filter 1-sigma estimate")
ax[0].set_ylabel("position error (m)")
ax[0].set_title(f"Navigation: mean {nav_err.mean():.2f} m, max {nav_err.max():.2f} m, "
                f"DVL availability {summary['dvl_availability']*100:.1f} %")
ax[0].legend(fontsize=8)

alt = eta[:, 2] - bed_under
ax[1].plot(t, alt, color=GOOD, lw=1.2)
ax[1].axhline(3.2, color="k", ls=":", lw=0.9, label="survey altitude setpoint")
ax[1].axhline(2.6, color="#888", ls=":", lw=0.9, label="inspect altitude setpoint")
ax[1].set_ylabel("altitude above bed (m)")
ax[1].legend(fontsize=8)

ax[2].plot(t, d["current"], color="#8250df", lw=1.1, label="estimated current")
true_c = np.linalg.norm(site.current(eta[:, 0], eta[:, 1], eta[:, 2]), axis=-1)
ax[2].plot(t, true_c, color="k", lw=1.0, ls="--", label="true current")
ax[2].axhline(max_holdable_current(isonavi_1), color=GOOD, lw=0.9,
              label=f"isonavi envelope {max_holdable_current(isonavi_1):.2f} m/s")
ax[2].axhline(max_holdable_current(BLUEROV2_HEAVY), color=WARN, lw=0.9,
              label=f"COTS envelope {max_holdable_current(BLUEROV2_HEAVY):.2f} m/s")
ax[2].set_ylabel("current (m/s)"); ax[2].set_xlabel("mission time (s)")
ax[2].legend(fontsize=7.5, ncol=2)
for a in ax:
    shade_phases(a)
plt.tight_layout(); plt.savefig(f"{FIG}/f2_navigation.png"); plt.close()

# =====================================================================  fig 3
mean_z = np.full_like(d["map_sum"], np.nan)
m = d["map_count"] > 0
mean_z[m] = d["map_sum"][m] / d["map_count"][m]
mx = d["map_x0"] + np.arange(mean_z.shape[1]) * d["map_res"]
my = d["map_y0"] + np.arange(mean_z.shape[0]) * d["map_res"]

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.9))
vmin, vmax = np.nanpercentile(mean_z, [1, 99])
a0 = ax[0].pcolormesh(site.xs, site.ys, site.H, shading="auto",
                      cmap="terrain", vmin=vmin, vmax=vmax)
ax[0].set_title("Ground truth bathymetry")
plt.colorbar(a0, ax=ax[0], fraction=0.04, label="m")
a1 = ax[1].pcolormesh(mx, my, mean_z, shading="auto", cmap="terrain",
                      vmin=vmin, vmax=vmax)
ax[1].set_title(f"Mapped from sonar ({summary['soundings']} soundings)")
plt.colorbar(a1, ax=ax[1], fraction=0.04, label="m")

Xg, Yg = np.meshgrid(mx, my)
truth_interp = site.bed_height(Xg, Yg)
diff = mean_z - truth_interp
a2 = ax[2].pcolormesh(mx, my, diff, shading="auto", cmap="RdBu_r",
                      vmin=-1.0, vmax=1.0)
plt.colorbar(a2, ax=ax[2], fraction=0.04, label="mapped minus truth (m)")
valid = np.isfinite(diff)
rmse = float(np.sqrt(np.nanmean(diff[valid] ** 2)))
bias = float(np.nanmean(diff[valid]))
ax[2].set_title(f"Error: RMSE {rmse:.2f} m, bias {bias:+.2f} m")
for a in ax:
    a.set_aspect("equal"); a.set_xlabel("x (m)")
    a.set_xlim(-20, 60); a.set_ylim(-30, 30)
ax[0].set_ylabel("y (m)")
plt.tight_layout(); plt.savefig(f"{FIG}/f3_mapping.png"); plt.close()

# =====================================================================  fig 4
piers = [p for p in summary["scour"] if summary["scour"][p]]
fig, ax = plt.subplots(1, len(piers) + 1, figsize=(5.0 * (len(piers) + 1), 4.4))
for k, pname in enumerate(piers):
    idx = int(pname[1:]) - 1
    p = site.cfg.piers[idx]
    rr = np.linspace(0, 14, 160)
    prof_t, prof_m = [], []
    for r in rr:
        angs = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        xs = p.x + r * np.cos(angs); ys = p.y + r * np.sin(angs)
        prof_t.append(np.nanmean(site.bed_height(xs, ys)))
        i = np.round((xs - d["map_x0"]) / d["map_res"]).astype(int)
        j = np.round((ys - d["map_y0"]) / d["map_res"]).astype(int)
        ok = ((i >= 0) & (i < mean_z.shape[1]) & (j >= 0) & (j < mean_z.shape[0]))
        vals = mean_z[j[ok], i[ok]] if ok.any() else np.array([np.nan])
        prof_m.append(np.nanmean(vals) if np.isfinite(vals).any() else np.nan)
    ax[k].plot(rr, prof_t, color="k", lw=1.6, label="ground truth")
    ax[k].plot(rr, prof_m, color=ACC, lw=1.6, ls="--", label="mapped")
    ax[k].axvline(p.radius, color="0.5", lw=0.8, ls=":")
    ax[k].axvline(p.scour_radius, color=WARN, lw=0.8, ls=":")
    s = summary["scour"][pname]
    ax[k].set_title(f"{pname} radial profile\nmapped depth {s['max_depth']:.2f} m "
                    f"vs truth {s['truth_depth']:.2f} m")
    ax[k].set_xlabel("radius from pier centre (m)")
    ax[k].set_ylabel("bed elevation (m)")
    ax[k].legend(fontsize=8)

names = list(summary["scour"])
md = [summary["scour"][n]["max_depth"] if summary["scour"][n] else 0 for n in names]
td = [summary["scour"][n]["truth_depth"] if summary["scour"][n] else 0 for n in names]
xpos = np.arange(len(names))
ax[-1].bar(xpos - 0.19, td, 0.36, label="truth", color="0.55")
ax[-1].bar(xpos + 0.19, md, 0.36, label="measured", color=ACC)
for i, (a, b) in enumerate(zip(td, md)):
    ax[-1].annotate(f"{100*b/max(a,1e-9):.0f} %", (i + 0.19, b), ha="center",
                    va="bottom", fontsize=8)
ax[-1].set_xticks(xpos); ax[-1].set_xticklabels(names)
ax[-1].set_ylabel("scour depth (m)")
ax[-1].set_title("Scour depth recovery")
ax[-1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/f4_scour.png"); plt.close()

# =====================================================================  fig 5
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
v = np.linspace(0, 3.6, 250)
for p, col in ((BLUEROV2_HEAVY, WARN), (isonavi_1, GOOD)):
    drag = p.quad_damp[0] * v ** 2 + p.lin_damp[0] * v
    B = vectored_allocation(p.arms)
    f_unit = np.max(np.abs(np.linalg.pinv(B) @ np.array([1.0, 0, 0, 0, 0, 0])))
    fmax = p.max_thrust_n / f_unit
    ax[0].plot(v, drag, color=col, lw=1.8, label=f"{p.name} drag")
    ax[0].axhline(fmax, color=col, ls="--", lw=1.1,
                  label=f"{p.name.split()[0]} max thrust {fmax:.0f} N")
    vm = max_holdable_current(p)
    ax[0].plot([vm], [fmax], "o", color=col, ms=7)
    ax[0].annotate(f"{vm:.2f} m/s", (vm, fmax), xytext=(6, -12),
                   textcoords="offset points", color=col, fontsize=8.5)
ax[0].axvline(site.cfg.surface_current, color="k", ls=":", lw=1.2)
ax[0].annotate("site surface current\n2.4 m/s", (site.cfg.surface_current, 700),
               fontsize=8, ha="center")
ax[0].set_xlabel("current speed (m/s)"); ax[0].set_ylabel("surge force (N)")
ax[0].set_ylim(0, 900)
ax[0].set_title("Thrust envelope against hydrodynamic drag")
ax[0].legend(fontsize=7.5, loc="upper left")

ax[1].plot(t, d["thrust"], color=ACC, lw=0.9)
ax[1].axhline(isonavi_1.max_thrust_n, color=WARN, ls="--", lw=1.0,
              label=f"per-thruster limit {isonavi_1.max_thrust_n:.0f} N")
shade_phases(ax[1])
ax[1].set_xlabel("mission time (s)"); ax[1].set_ylabel("peak thruster force (N)")
ax[1].set_title(f"Thruster utilisation, mean "
                f"{np.mean(d['thrust'])/isonavi_1.max_thrust_n*100:.0f} % of limit")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/f5_vehicle.png"); plt.close()

# =====================================================================  fig 6
cfg = preset("oculus", seed=11, r_max=45.0, ssc_g_per_l=site.cfg.ssc_g_per_l)
fls = ForwardLookingSonar(cfg, site.scene)
poses = d["ping_pose"]; ping_t = d["ping_t"]
sel = [int(len(poses) * f) for f in (0.12, 0.34, 0.56, 0.78, 0.92)]
sel = [min(s, len(poses) - 1) for s in sel]
fig, axes = plt.subplots(2, len(sel), figsize=(3.4 * len(sel), 7.2))
for k, si in enumerate(sel):
    fr = fls.ping(poses[si])
    axes[0, k].imshow(fr.normalised(), aspect="auto", cmap="inferno", origin="lower",
                      extent=[np.degrees(fr.bearings[0]), np.degrees(fr.bearings[-1]),
                              fr.ranges[0], fr.ranges[-1]])
    axes[0, k].set_title(f"t = {ping_t[si]:.0f} s", fontsize=9)
    axes[0, k].set_xlabel("bearing (deg)")
    if k == 0:
        axes[0, k].set_ylabel("slant range (m)")
    axes[1, k].imshow(fr.to_cartesian(400), cmap="inferno")
    axes[1, k].axis("off")
axes[1, 0].set_title("operator fan view", fontsize=9, loc="left")
plt.suptitle("Simulated forward-looking sonar along the mission track", y=1.0)
plt.tight_layout(); plt.savefig(f"{FIG}/f6_sonar_track.png"); plt.close()

print("figures written:")
for f in sorted(os.listdir(FIG)):
    print("  ", f)
print()
print(f"mapping RMSE {rmse:.3f} m, bias {bias:+.3f} m")
json.dump({"map_rmse": rmse, "map_bias": bias},
          open(f"{LOG}/mapping_metrics.json", "w"), indent=1)
