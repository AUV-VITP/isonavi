"""Figures for the hardware-in-the-loop section of the report.

Two panels:
  1. the HIL mission track and estimate against ground truth, proving the loop
     closed on hardware and reproduced the simulation
  2. the board loop timing distribution against the control budget, and the
     compute vs network split
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.expanduser("~/dev/isonavi")
FIG = f"{ROOT}/simulation/results/figures"
HIL = f"{ROOT}/hil/results"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 145, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
ACC = "#1f6feb"
WARN = "#d1242f"
GOOD = "#1a7f37"

d = np.load(f"{HIL}/hil_full.npz", allow_pickle=True)
host = json.load(open(f"{HIL}/hil_full.json"))
board = json.load(open(f"{HIL}/hil_full_board.json"))

true_eta = d["true_eta"]
est = d["est_pos"]
n = min(len(true_eta), len(est))
true_xy = true_eta[:n, :2]
est_xy = est[:n, :2]
nav_err = np.linalg.norm(est[:n] - true_eta[:n, :3], axis=1)
t = np.arange(n) * 0.05

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

# --- track: truth vs on-board estimate
ax[0].plot(true_xy[:, 0], true_xy[:, 1], color="0.4", lw=2.2,
           label="true (host physics)")
ax[0].plot(est_xy[:, 0], est_xy[:, 1], color=ACC, lw=1.0,
           label="estimate (on the board)")
ax[0].plot(true_xy[0, 0], true_xy[0, 1], "o", color=GOOD, ms=7, label="launch")
ax[0].set_aspect("equal")
ax[0].set_xlabel("x downstream (m)"); ax[0].set_ylabel("y cross-stream (m)")
ax[0].set_title("HIL mission: on-board estimate vs ground truth")
ax[0].legend(loc="upper left", fontsize=8)

# --- nav error over the run
ax[1].plot(t, nav_err, color=ACC, lw=0.9)
ax[1].axhline(nav_err.mean(), color=WARN, ls="--", lw=1.0,
              label=f"mean {nav_err.mean():.3f} m")
ax[1].set_xlabel("mission time (s)"); ax[1].set_ylabel("position error (m)")
ax[1].set_title(f"Navigation error on hardware\n"
                f"mean {host['nav_error_mean']:.3f} m, max {host['nav_error_max']:.3f} m")
ax[1].legend(fontsize=8)

# --- loop timing distribution
comp_mean = board["compute_ms_mean"]
comp_p99 = board["compute_ms_p99"]
wait_mean = board["wait_ms_mean"]
budget = board["budget_ms"]
labels = ["compute\n(EKF, control,\nmission)", "network\n(sensor round-trip)"]
vals = [comp_mean, wait_mean]
colors = [ACC, "#8b949e"]
bars = ax[2].bar(labels, vals, color=colors, width=0.6)
ax[2].axhline(budget, color=WARN, ls="--", lw=1.3,
              label=f"control budget {budget:.0f} ms (20 Hz)")
ax[2].bar(labels[0], comp_p99 - comp_mean, bottom=comp_mean, color=ACC,
          alpha=0.35, width=0.6)
for b, v in zip(bars, vals):
    ax[2].annotate(f"{v:.1f} ms", (b.get_x() + b.get_width() / 2, v),
                   ha="center", va="bottom", fontsize=9)
ax[2].annotate(f"p99 {comp_p99:.1f}", (0, comp_p99), ha="center", va="bottom",
               fontsize=8, color=ACC)
ax[2].set_ylabel("time per tick (ms)")
ax[2].set_ylim(0, budget * 1.15)
ax[2].set_title(f"Board timing on RISC-V @ 750 MHz\n"
                f"{budget/comp_p99:.1f}x compute margin at p99")
ax[2].legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(f"{FIG}/f12_hil.png", bbox_inches="tight")
plt.close()

# Equivalence table numbers for the report macros.
ref = json.load(open(f"{ROOT}/simulation/results/logs/mission_isonavi_s1.json"))
out = {
    "hil_nav_mean": host["nav_error_mean"],
    "hil_nav_max": host["nav_error_max"],
    "hil_ticks": host["ticks"],
    "hil_sim_time": host["sim_time"],
    "hil_crc": host["crc_errors"],
    "board_compute_mean": comp_mean,
    "board_compute_p99": comp_p99,
    "board_wait_mean": wait_mean,
    "board_budget": budget,
    "board_margin": budget / comp_p99,
    "sim_nav_mean": ref["nav_error_mean"],
    # Actuator interface, present once the ESP32 is genuinely in the loop.
    "esp_link": board.get("esp_link", "none"),
    "esp_pwm_sent": board.get("esp_pwm_sent", 0),
    "esp_echoes": board.get("esp_echoes", 0),
    "esp_matched": board.get("esp_matched", 0),
    "esp_mismatches": board.get("esp_mismatches", 0),
    "esp_lag_mean": board.get("esp_lag_mean", 0.0),
    "esp_ms_mean": board.get("esp_ms_mean", 0.0),
    "esp_ms_p99": board.get("esp_ms_p99", 0.0),
    "board_loop_p99": board.get("loop_ms_p99", 0.0),
}
json.dump(out, open(f"{ROOT}/simulation/results/logs/hil_metrics.json", "w"),
          indent=1)
print("wrote f12_hil.png and hil_metrics.json")
for k, v in out.items():
    print(f"  {k}: {v}")
