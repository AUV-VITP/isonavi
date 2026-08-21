"""Render the vehicle STL from several angles with matplotlib, no GPU.

Uses flat shading with a simple Lambert light so the form reads clearly. This
is a verification render, not the hero image, but it must be good enough to
judge whether the geometry actually looks like a proper vehicle.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_stl(path):
    """Read a binary STL into an (n, 3, 3) triangle array."""
    with open(path, "rb") as f:
        f.read(80)
        n = int.from_bytes(f.read(4), "little")
        tris = np.zeros((n, 3, 3), np.float32)
        for i in range(n):
            f.read(12)  # normal
            for j in range(3):
                tris[i, j] = np.frombuffer(f.read(12), np.float32)
            f.read(2)
    return tris


def render(tris, ax, elev, azim, color="#d98a1f"):
    # Face normals for shading.
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    nrm = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-9)
    ls = LightSource(azdeg=azim + 40, altdeg=elev + 20)
    light = np.array([np.cos(np.radians(elev + 20)) * np.cos(np.radians(azim + 40)),
                      np.cos(np.radians(elev + 20)) * np.sin(np.radians(azim + 40)),
                      np.sin(np.radians(elev + 20))])
    shade = np.clip(0.35 + 0.65 * np.abs(nrm @ light), 0, 1)
    base = np.array(matplotlib.colors.to_rgb(color))
    cols = base[None, :] * shade[:, None]

    pc = Poly3DCollection(tris, facecolors=cols, edgecolors="none")
    ax.add_collection3d(pc)
    allv = tris.reshape(-1, 3)
    ctr = allv.mean(0)
    rng = (allv.max(0) - allv.min(0)).max() / 2
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


CAD = os.path.expanduser("~/dev/rakshatech/cad")
tris = load_stl(f"{CAD}/varuna_vehicle.stl")
print(f"loaded {len(tris)} triangles")

fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
views = [("side", 8, -90), ("three-quarter", 22, -55),
         ("top", 88, -90), ("front", 2, 0),
         ("bottom-quarter", -18, -50), ("aft-quarter", 16, 130)]
for k, (name, el, az) in enumerate(views):
    ax = fig.add_subplot(2, 3, k + 1, projection="3d", facecolor="#0d1117")
    render(tris, ax, el, az)
    ax.set_title(name, color="#e6edf3", fontsize=11)
fig.suptitle("VARUNA-1  (geometry from simulation parameters)",
             color="#e6edf3", fontsize=15, y=0.97)
plt.tight_layout()
out = f"{CAD}/varuna_render.png"
plt.savefig(out, dpi=120, facecolor="#0d1117")
print("wrote", out)
