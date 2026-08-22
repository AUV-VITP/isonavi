"""A single clean three-quarter render of the vehicle for the report, on a
light background with a labelled callout of the major features.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = int.from_bytes(f.read(4), "little")
        tris = np.zeros((n, 3, 3), np.float32)
        for i in range(n):
            f.read(12)
            for j in range(3):
                tris[i, j] = np.frombuffer(f.read(12), np.float32)
            f.read(2)
    return tris


CAD = os.path.expanduser("~/dev/rakshatech/cad")
tris = load_stl(f"{CAD}/varuna_vehicle.stl")

fig = plt.figure(figsize=(13, 7.2), facecolor="white")
ax = fig.add_subplot(111, projection="3d", facecolor="white")

elev, azim = 20, -58
v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
nrm = np.cross(v1 - v0, v2 - v0)
nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
light = np.array([np.cos(np.radians(elev + 25)) * np.cos(np.radians(azim + 35)),
                  np.cos(np.radians(elev + 25)) * np.sin(np.radians(azim + 35)),
                  np.sin(np.radians(elev + 25))])
shade = np.clip(0.45 + 0.55 * np.abs(nrm @ light), 0, 1)
base = np.array(to_rgb("#e08a1e"))
cols = base[None, :] * shade[:, None]

pc = Poly3DCollection(tris, facecolors=cols, edgecolors="none", linewidths=0)
ax.add_collection3d(pc)

allv = tris.reshape(-1, 3)
ctr = allv.mean(0)
rng = (allv.max(0) - allv.min(0)).max() / 2 * 0.92
ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=elev, azim=azim)
ax.set_axis_off()

ax.text2D(0.02, 0.95, "VARUNA-1", transform=ax.transAxes, fontsize=20,
          fontweight="bold", color="#16407A")
ax.text2D(0.02, 0.90,
          "1104 mm x 200 mm, 28 kg, 8 vectored thrusters\n"
          "geometry generated from the simulation parameters",
          transform=ax.transAxes, fontsize=10, color="#444")

plt.tight_layout()
out = f"{CAD}/varuna_hero.png"
plt.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("wrote", out)
