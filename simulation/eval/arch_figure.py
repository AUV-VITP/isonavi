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


FS_ARCH = 10.0

HEAD = 0.052
LINE = 0.038
PAD = 0.030


# The rounded box style paints this far outside the rectangle asked
# for. Connectors have to start and stop clear of it or their heads
# are drawn over by the box next door.
BOXPAD = 0.012


def box_height(lines):
    """Height that actually fits the header plus every content line."""
    return HEAD + LINE * len(lines) + PAD


def box(ax, x, y, w, title, lines, kind, fs=8.4, h=None):
    """Draw a titled box anchored at its bottom-left. Height is derived from
    the content unless given explicitly, so text can never overflow."""
    h = box_height(lines) if h is None else h
    c = COLS[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={BOXPAD},rounding_size=0.02",
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



def wrap_arrow(ax, x_from, y_from, x_to, y_to, band, col=MUTED):
    """Flow connector from the end of one row to the start of the next.

    Drawn as three straight segments through the clear band between the rows,
    because a single arc passes beneath the middle boxes and is hidden by them.
    """
    ax.plot([x_from, x_from], [y_from, band], color=col, lw=1.3, zorder=2)
    ax.plot([x_from, x_to], [band, band], color=col, lw=1.3, zorder=2)
    ax.add_patch(FancyArrowPatch((x_to, band), (x_to, y_to),
                                 arrowstyle="-|>", mutation_scale=13,
                                 lw=1.3, color=col, zorder=2))


def arrow(ax, p0, p1, label=None, style="-|>", col=MUTED, rad=0.0,
          fs=8.6, lab_dx=0.0, lab_dy=0.016):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 lw=1.3, color=col, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        # Above the boxes, not under them. Box bodies sit at zorder 3 and the
        # header bands at 4, so a label left at the text default of 3 was
        # drawn first and then buried by whichever box it ran alongside.
        ax.text(mx + lab_dx, my + lab_dy, label, ha="center", va="bottom",
                fontsize=fs,
                color=col, style="italic", zorder=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                          alpha=1.0))



def on_board(ax, x, y, w, h, pad=0.012):
    """Dashed halo marking a block as running on the flight computer."""
    ax.add_patch(FancyBboxPatch(
        (x - pad, y - pad), w + 2 * pad, h + 2 * pad,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        fc="none", ec="#111827", lw=1.5, ls=(0, (5, 3)), zorder=5))


# ======================================================================
# Drawn near the size it is printed at, so the type survives the
# downscale to the text block. Aspect matches the canvas it replaces.
fig, ax = plt.subplots(figsize=(9.6, 5.81))
ax.set_xlim(0, 1); ax.set_ylim(0.028, 0.975); ax.axis("off")

ENV = ["Bathymetry and scour", "Piers and deck span", "Submerged vehicles",
       "Debris field", "Depth-varying flow", "Suspended sediment"]
h_env = box_height(ENV)
box(ax, 0.015, 0.50, 0.195, "ENVIRONMENT", ENV, "world", fs=FS_ARCH)

h_son = box(ax, 0.255, 0.685, 0.185, "SONAR",
            ["Sonar equation", "Ray-cast occlusion",
             "Elevation ambiguity", "Gamma speckle"], "sense", fs=FS_ARCH)
h_nav = box(ax, 0.255, 0.435, 0.185, "NAV SENSORS",
            ["DVL, lock dropouts", "IMU, gyro bias walk", "Depth cell"], "sense", fs=FS_ARCH)

h_map = box(ax, 0.495, 0.685, 0.185, "MAPPING",
            ["Swath soundings", "Bathymetric grid", "Scour estimator",
             "Residual detector"], "estimate", fs=FS_ARCH)
h_ekf = box(ax, 0.495, 0.415, 0.185, "EKF",
            ["12 states", "position, velocity", "attitude, gyro bias",
             "Reports own sigma"], "estimate", fs=FS_ARCH)

h_mis = box(ax, 0.735, 0.545, 0.20, "MISSION",
            ["State machine", "DEPLOY, ACQUIRE", "SEARCH, INSPECT",
             "RETURN, REPORT", "Lawnmower and orbit", "Heading into flow"], "decide", fs=FS_ARCH)

h_ctl = box(ax, 0.495, 0.075, 0.185, "CONTROL",
            ["Cascaded pose loop", "Force feedforward", "Current estimator",
             "Priority allocation", "Force before torque"], "act", fs=FS_ARCH)
h_veh = box(ax, 0.255, 0.075, 0.185, "VEHICLE",
            ["6-DOF Fossen model", "Added mass, Coriolis", "Fin stabilisation",
             "8 vectored thrusters", "Thruster lag"], "act", fs=FS_ARCH)
h_prod = box(ax, 0.735, 0.075, 0.20, "MISSION PRODUCT",
             ["Bathymetric map", "Scour depth, volume", "Target positions",
              "Coverage record", "Timestamped log"], "product", fs=FS_ARCH)

arrow(ax, (0.210, 0.775), (0.255, 0.800), "returns")
arrow(ax, (0.210, 0.610), (0.255, 0.545), "motion")
arrow(ax, (0.440, 0.800), (0.495, 0.800), "frames")
arrow(ax, (0.440, 0.520), (0.495, 0.545), "meas")
arrow(ax, (0.588, 0.415 + h_ekf), (0.588, 0.685), "pose",
      lab_dx=-0.045, lab_dy=-0.004)
arrow(ax, (0.680, 0.800), (0.735, 0.720), "map")
arrow(ax, (0.680, 0.520), (0.735, 0.640), "pose")
arrow(ax, (0.790, 0.545), (0.680, 0.075 + h_ctl), "waypoint", rad=-0.16)
arrow(ax, (0.495, 0.150), (0.440, 0.150), "wrench")
arrow(ax, (0.300, 0.075 + h_veh), (0.150, 0.50), "thrust", rad=0.22)
arrow(ax, (0.390, 0.075 + h_veh), (0.390, 0.435), "true state")
arrow(ax, (0.835, 0.545), (0.835, 0.075 + h_prod), "report")

# Estimation, control and the mission state machine were executed on the
# RISC-V flight computer for the hardware-in-the-loop run; mapping and
# detection are host side and are not marked.
on_board(ax, 0.495, 0.415, 0.185, h_ekf)
on_board(ax, 0.495, 0.075, 0.185, h_ctl)
on_board(ax, 0.735, 0.545, 0.20, h_mis)
ax.text(0.935, 0.940, "dashed: runs on the RISC-V flight computer",
        fontsize=9.6, color="#111827", ha="right", style="italic",
        va="bottom", zorder=8)

# No title or subtitle here: the figure carries a caption in the document
# that says the same thing, and repeating it wastes the space the boxes need.

handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=COLS[k], label=v)
           for k, v in (("world", "environment"), ("sense", "sensing"),
                        ("estimate", "estimation"), ("decide", "decision"),
                        ("act", "actuation"), ("product", "output"))]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, -0.028),
          ncol=6, frameon=False, fontsize=9.6)
plt.tight_layout()
plt.savefig(f"{FIG}/f0_architecture.png", bbox_inches="tight",
            dpi=210)
plt.close()

# ======================================================================
# Keep the same vertical scale as the architecture figure: the rounded box
# geometry is expressed in axis units, so a squashed y-range would compress
# the header band into the first line of text.
fig, ax = plt.subplots(figsize=(8.6, 5.79))
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

COLS_N, GAP = 3, 0.045
W = (1.0 - (COLS_N + 1) * GAP) / COLS_N
H = box_height(stages[0][1])
ROW_Y = [0.70, 0.38, 0.06]

for i, (title, lines) in enumerate(stages):
    r, c = divmod(i, COLS_N)
    x = GAP + c * (W + GAP)
    y = ROW_Y[r]
    kind = "sense" if i < 2 else ("estimate" if i < 5 else "act")
    box(ax, x, y, W, title, lines, kind, fs=10.4)
    if c < COLS_N - 1:
        arrow(ax, (x + W + BOXPAD, y + H / 2),
              (x + W + GAP - BOXPAD, y + H / 2))
    elif r < len(ROW_Y) - 1:
        top_next = ROW_Y[r + 1] + H
        wrap_arrow(ax, x + W / 2, y - BOXPAD, GAP + W / 2,
                   top_next + BOXPAD, band=(y + top_next) / 2)

# Crop the axes to what is actually drawn in them.
ax.set_ylim(ROW_Y[-1] - 0.035, ROW_Y[0] + H + 0.035)
plt.tight_layout()
plt.savefig(f"{FIG}/f0b_sonar_pipeline.png", bbox_inches="tight",
            dpi=210)
plt.close()

print("wrote f0_architecture.png and f0b_sonar_pipeline.png")
