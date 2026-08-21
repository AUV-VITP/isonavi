"""Draw the system architecture and the sonar image-formation pipeline."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
os.makedirs(FIG, exist_ok=True)

INK = "#0d1117"
MUTED = "#57606a"
COLS = {
    "sense": "#1f6feb",
    "estimate": "#8250df",
    "decide": "#1a7f37",
    "act": "#bf8700",
    "world": "#57606a",
    "product": "#d1242f",
}


HEAD = 0.052
LINE = 0.038
PAD = 0.030


def box_height(lines):
    """Height that actually fits the header plus every content line."""
    return HEAD + LINE * len(lines) + PAD


def box(ax, x, y, w, title, lines, kind, fs=8.4, h=None):
    """Draw a titled box anchored at its bottom-left. Height is derived from
    the content unless given explicitly, so text can never overflow."""
    h = box_height(lines) if h is None else h
    c = COLS[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="white", ec=c, lw=1.5, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.052), w, 0.052,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=c, ec=c, lw=1.5, zorder=4))
    ax.text(x + w / 2, y + h - HEAD / 2, title, ha="center", va="center",
            fontsize=fs + 0.6, color="white", fontweight="bold", zorder=5)
    for i, ln in enumerate(lines):
        ax.text(x + 0.014, y + h - HEAD - 0.020 - i * LINE, ln,
                ha="left", va="center", fontsize=fs - 0.6, color=INK, zorder=5)
    return h


def arrow(ax, p0, p1, label=None, style="-|>", col=MUTED, rad=0.0, fs=7.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 lw=1.3, color=col, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.016, label, ha="center", va="bottom", fontsize=fs,
                color=col, style="italic",
                bbox=dict(fc="white", ec="none", pad=0.8))


# ======================================================================
fig, ax = plt.subplots(figsize=(14.5, 8.6))
ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.13); ax.axis("off")

ENV = ["Bathymetry and scour", "Bridge piers, deck span", "Submerged vehicles",
       "Debris field", "Depth-varying current", "Suspended sediment"]
h_env = box_height(ENV)
box(ax, 0.015, 0.50, 0.195, "ENVIRONMENT", ENV, "world")

h_son = box(ax, 0.255, 0.685, 0.185, "SONAR",
            ["Sonar equation per cell", "Ray-cast occlusion",
             "Elevation ambiguity", "Gamma speckle"], "sense")
h_nav = box(ax, 0.255, 0.435, 0.185, "NAV SENSORS",
            ["DVL, bottom lock loss", "IMU, gyro bias walk", "Depth cell"], "sense")

h_map = box(ax, 0.495, 0.685, 0.185, "MAPPING",
            ["Swath soundings", "Bathymetric grid", "Scour estimator",
             "Residual detector"], "estimate")
h_ekf = box(ax, 0.495, 0.415, 0.185, "EKF",
            ["12 states: position,", "velocity, attitude,", "gyro bias",
             "Reports own sigma"], "estimate")

h_mis = box(ax, 0.735, 0.545, 0.20, "MISSION",
            ["State machine", "DEPLOY, ACQUIRE", "SEARCH, INSPECT",
             "RETURN, REPORT", "Lawnmower and orbit", "Heading into flow"], "decide")

h_ctl = box(ax, 0.495, 0.075, 0.185, "CONTROL",
            ["Cascaded pose loop", "Force feedforward", "Current estimator",
             "Prioritised allocation:", "force before torque"], "act")
h_veh = box(ax, 0.255, 0.075, 0.185, "VEHICLE",
            ["6-DOF Fossen model", "Added mass, Coriolis", "Fin stabilisation",
             "8 vectored thrusters", "Thruster lag"], "act")
h_prod = box(ax, 0.735, 0.075, 0.20, "MISSION PRODUCT",
             ["Bathymetric map", "Scour depth, volume", "Target positions",
              "Coverage record", "Timestamped log"], "product")

arrow(ax, (0.210, 0.775), (0.255, 0.800), "returns")
arrow(ax, (0.210, 0.610), (0.255, 0.545), "motion")
arrow(ax, (0.440, 0.800), (0.495, 0.800), "frames")
arrow(ax, (0.440, 0.520), (0.495, 0.545), "meas")
arrow(ax, (0.588, 0.415 + h_ekf), (0.588, 0.685), "pose")
arrow(ax, (0.680, 0.800), (0.735, 0.720), "map")
arrow(ax, (0.680, 0.520), (0.735, 0.640), "pose")
arrow(ax, (0.790, 0.545), (0.680, 0.075 + h_ctl), "waypoint", rad=-0.16)
arrow(ax, (0.495, 0.150), (0.440, 0.150), "wrench")
arrow(ax, (0.300, 0.075 + h_veh), (0.150, 0.50), "thrust", rad=0.22)
arrow(ax, (0.390, 0.075 + h_veh), (0.390, 0.435), "true state")
arrow(ax, (0.835, 0.545), (0.835, 0.075 + h_prod), "report")

ax.text(0.015, 1.085, "VARUNA system architecture", fontsize=15,
        fontweight="bold", color=INK)
ax.text(0.015, 1.045,
        "Closed loop. Everything the vehicle acts on is estimated, never taken "
        "from the simulator ground truth.",
        fontsize=9.2, color=MUTED)

handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=COLS[k], label=v)
           for k, v in (("world", "environment"), ("sense", "sensing"),
                        ("estimate", "estimation"), ("decide", "decision"),
                        ("act", "actuation"), ("product", "output"))]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, -0.03),
          ncol=6, frameon=False, fontsize=8.6)
plt.tight_layout()
plt.savefig(f"{FIG}/f0_architecture.png", bbox_inches="tight")
plt.close()

# ======================================================================
# Keep the same vertical scale as the architecture figure: the rounded box
# geometry is expressed in axis units, so a squashed y-range would compress
# the header band into the first line of text.
fig, ax = plt.subplots(figsize=(16.5, 4.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

stages = [
    ("Ray fan", ["beams x elevation", "jittered per ping"]),
    ("Scene cast", ["nearest hit", "occlusion resolved"]),
    ("Backscatter", ["Lambert + specular", "by material, angle"]),
    ("Transmission", ["spreading 20 log r", "absorption alpha r"]),
    ("Footprint", ["spread over", "r dtheta cot g"]),
    ("Pulse", ["range point", "spread function"]),
    ("Beam pattern", ["bearing", "convolution"]),
    ("Speckle", ["Gamma, L looks", "plus noise floor"]),
    ("Display", ["time-varying gain", "polar or fan"]),
]
n = len(stages)
gap = 0.016
w = (1.0 - (n + 1) * gap) / n
y = 0.34
for i, (title, lines) in enumerate(stages):
    x = gap + i * (w + gap)
    kind = "sense" if i < 2 else ("estimate" if i < 5 else "act")
    h = box(ax, x, y, w, title, lines, kind, fs=7.6)
    if i < n - 1:
        arrow(ax, (x + w, y + h / 2), (x + w + gap, y + h / 2))

ax.text(0.0, 0.70, "Sonar image formation", fontsize=14, fontweight="bold",
        color=INK)
ax.text(0.0, 0.63,
        "Every stage is a physical effect rather than a rendering convenience. "
        "Shadow comes from occlusion, elevation ambiguity from summing "
        "elevation rays, and texture from coherent speckle.",
        fontsize=9, color=MUTED)
plt.tight_layout()
plt.savefig(f"{FIG}/f0b_sonar_pipeline.png", bbox_inches="tight")
plt.close()

print("wrote f0_architecture.png and f0b_sonar_pipeline.png")
