"""The vehicle in the site it was designed for.

Ties the two halves of this work together in one image: the CAD model, placed
on the trajectory the simulated mission actually flew, over the bathymetry the
sonar actually mapped. Nothing is arranged by hand. The bed comes from the site
model, the pose comes from the flight log, and the insonified patch is the real
field of view of the modelled head projected onto the real bed.

Everything is drawn as one triangle soup so a single depth sort resolves
occlusion between vehicle, piers, track and bed. Matplotlib sorts within a
collection but not between them, so the track cannot be a separate line artist
or the bed will hide it.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulation"))

import render_views as RV
import varuna_cad as V
from varuna.scene import DisasterSite

OUT = os.path.expanduser("~/dev/rakshatech/cad")
LOG = os.path.expanduser("~/dev/rakshatech/simulation/results/logs")
MM = 1000.0
BLUE = "#16407A"

# Oculus M750d, as configured in the simulation.
FOV_H, FOV_V, RANGE = 130.0, 20.0, 30.0



def trim_png(path, pad=16):
    """Crop a saved figure to its ink, leaving a small margin.

    Matplotlib's bbox_inches only tightens to the axes bounding box, which for
    a 3D axes is the full cubic frame whether or not anything is drawn in it.
    """
    try:
        from PIL import Image
    except ImportError:
        return
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ink = (a < 248).any(axis=2)
    if not ink.any():
        return
    ys, xs = np.where(ink)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, a.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, a.shape[1])
    im.crop((x0, y0, x1, y1)).save(path)


# ------------------------------------------------------------------ meshes
def grid_tris(X, Y, Z):
    P = np.stack([X, Y, Z], -1)
    ny, nx = X.shape
    i = np.arange(ny - 1)[:, None]
    j = np.arange(nx - 1)[None, :]
    a, b, c, d = P[i, j], P[i, j + 1], P[i + 1, j + 1], P[i + 1, j]
    return np.concatenate([np.stack([a, b, c], -2).reshape(-1, 3, 3),
                           np.stack([a, c, d], -2).reshape(-1, 3, 3)])


def bed_mesh(site, x0, x1, y0, y1, step, zex=1.0, lit=None,
             cmap='YlGnBu_r'):
    """Triangulated bed coloured by depth, optionally tinting the patch the
    sonar is insonifying at the given pose."""
    xs = np.arange(x0, x1 + step, step)
    ys = np.arange(y0, y1 + step, step)
    X, Y = np.meshgrid(xs, ys)
    Z = site.bed_height(X, Y)
    tris = grid_tris(X, Y, Z * zex)

    cen = tris.mean(axis=1)
    z = cen[:, 2] / zex
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    cols = plt.get_cmap(cmap)(0.14 + 0.72 * t)[:, :3]

    if lit is not None:
        pos, yaw, rng = lit
        d = cen - np.asarray(pos)
        rho = np.hypot(d[:, 0], d[:, 1])
        az = np.arctan2(d[:, 1], d[:, 0]) - yaw
        az = (az + np.pi) % (2 * np.pi) - np.pi
        inside = (rho < rng) & (np.abs(az) < np.radians(FOV_H / 2))
        # Insonified bed, brightened toward the cyan of the sonar display.
        fall = np.clip(1.0 - rho / rng, 0, 1)[:, None]
        tint = np.array(to_rgb("#28d8f0"))
        cols[inside] = (0.42 * cols[inside]
                        + 0.58 * tint * (0.45 + 0.55 * fall[inside]))
    return tris, cols


def cylinder_mesh(cx, cy, r, z0, z1, n=40, color="#9aa0a6", seg=None):
    """A closed cylinder whose wall is split into short bands.

    The wall used to be one quad per facet spanning the entire height. The
    painter's sort keys on a triangle centroid, so a pier tens of metres
    tall sorted on a point half way up itself and was drawn over terrain
    that stood in front of its base. The buried section reappeared below the
    bed as a floating stub. Short bands put each centroid near the geometry
    it actually belongs to.
    """
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = cx + r * np.cos(ang), cy + r * np.sin(ang)
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    if seg is None:
        # Bands shorter than the terrain grid spacing, so no band can
        # straddle enough depth for its centroid to beat the bed.
        seg = int(np.clip(abs(z1 - z0) / 0.8, 1, 96))
    edges = np.linspace(z0, z1, seg + 1)
    parts = []
    for za, zb in zip(edges[:-1], edges[1:]):
        a, b = np.full(n, za), np.full(n, zb)
        parts.append(np.stack([np.stack([x, y, a], -1),
                               np.stack([xn, yn, a], -1),
                               np.stack([xn, yn, b], -1)], -2))
        parts.append(np.stack([np.stack([x, y, a], -1),
                               np.stack([xn, yn, b], -1),
                               np.stack([x, y, b], -1)], -2))
    t = np.full(n, z1)
    parts.append(np.stack([np.stack([x, y, t], -1),
                           np.stack([xn, yn, t], -1),
                           np.stack([np.full(n, cx), np.full(n, cy), t],
                                    -1)], -2))
    tris = np.concatenate(parts)
    return tris, np.tile(np.array(to_rgb(color)), (len(tris), 1))


def rod(p0, p1, r=0.10, n=7, color="#f97316"):
    """A thin prism along a segment, so lines survive the depth sort."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    ln = np.linalg.norm(d)
    if ln < 1e-9:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    d /= ln
    up = np.array([0.0, 0.0, 1.0])
    if abs(d @ up) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    u = np.cross(d, up)
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(a) * u * r + np.sin(a) * w * r for a in ang])
    A, B = p0 + ring, p1 + ring
    An, Bn = np.roll(A, -1, 0), np.roll(B, -1, 0)
    tris = np.concatenate([np.stack([A, An, Bn], -2),
                           np.stack([A, Bn, B], -2)])
    return tris, np.tile(np.array(to_rgb(color)), (len(tris), 1))


def polyline_rod(pts, r=0.12, color="#f97316", step=1):
    tris, cols = [], []
    pts = np.asarray(pts, float)[::step]
    for a, b in zip(pts[:-1], pts[1:]):
        t, c = rod(a, b, r=r, color=color)
        if len(t):
            tris.append(t)
            cols.append(c)
    if not tris:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    return np.concatenate(tris), np.concatenate(cols)


def vehicle_mesh(pose, yaw, tol=0.7):
    """The CAD model, coarsely tessellated, placed at a pose in site metres."""
    assy, _, _, _, _ = V.build()
    tris, _, cols = RV.tessellate(assy, tol=tol, ang=0.3, skip=RV.INTERNAL)
    tris = tris / MM
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return tris @ R.T + np.asarray(pose), cols



def box_mesh(centre, size, yaw=0.0, color="#5b6472"):
    """An axis aligned box, rotated about z. Used for the submerged vehicles."""
    cx, cy, cz = centre
    lx, ly, lz = np.asarray(size, float) / 2.0
    c = np.array([[sx * lx, sy * ly, sz * lz]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    ca, sa = np.cos(yaw), np.sin(yaw)
    R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    v = c @ R.T + np.array([cx, cy, cz])
    idx = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
           (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    tris = np.array([[v[a], v[b], v[cc]] for a, b, cc in idx])
    return tris, np.tile(np.array(to_rgb(color)), (len(tris), 1))


# ------------------------------------------------------------------ drawing
def render(ax, layers, elev, azim, zoom=1.0, focal=None):
    tris = np.concatenate([t for t, _ in layers if len(t)])
    cols = np.concatenate([c for t, c in layers if len(t)])

    le, la = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(le) * np.cos(la), np.cos(le) * np.sin(la),
                     np.sin(le)])
    order = np.argsort(tris.mean(axis=1) @ view)
    tris, cols = tris[order], cols[order]

    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    sgn = np.sign(n @ view)
    sgn[sgn == 0] = 1
    n *= sgn[:, None]
    key = np.array([0.42, -0.52, 0.75])
    key /= np.linalg.norm(key)
    lam = 0.38 + 0.62 * np.clip(n @ key, 0, 1)
    fc = np.clip(cols * lam[:, None], 0, 1)

    ax.add_collection3d(Poly3DCollection(tris, facecolors=fc,
                                         edgecolors="none", linewidths=0))
    allv = tris.reshape(-1, 3)
    lo, hi = allv.min(0), allv.max(0)
    ctr = (lo + hi) / 2
    # A cube around a scene 116 m long, 88 m wide and a few metres deep
    # leaves the terrain as a sliver in a mostly empty box. That is what
    # letterboxed this figure down to a strip when it was cropped to its
    # ink. Fit each axis to its own extent and give the box the same
    # proportions, so the projection stays undistorted and the content
    # fills the frame it is given.
    span = np.maximum(hi - lo, (hi - lo).max() * 0.04)
    half = span / 2 * zoom
    ax.set_xlim(ctr[0] - half[0], ctr[0] + half[0])
    ax.set_ylim(ctr[1] - half[1], ctr[1] + half[1])
    ax.set_zlim(ctr[2] - half[2], ctr[2] + half[2])
    ax.set_box_aspect(tuple(span / span.max()))
    ax.view_init(elev=elev, azim=azim)
    if focal:
        try:
            ax.set_proj_type("persp", focal_length=focal)
        except TypeError:
            ax.set_proj_type("persp")
    ax.set_axis_off()


def main():
    site = DisasterSite()
    p = f"{LOG}/mission_varuna_s1.npz"
    eta = np.load(p, allow_pickle=True)["eta"] if os.path.exists(p) else None

    # Two panels of very different proportions. Side by side in one figure
    # they share a bounding box, so cropping to ink is governed by whichever
    # is flatter and the other is starved. Each gets its own file and its
    # own crop, and the describing text lives in the caption rather than
    # being drawn at a size that does not survive scaling to the text width.

    # ---------------------------------------------------- wide site view
    fig = plt.figure(figsize=(9.4, 5.0), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection="3d", facecolor="white")
    ZEX = 2.6
    layers = [bed_mesh(site, -58, 58, -44, 44, 1.3, zex=ZEX)]
    for pr in site.cfg.piers:
        z0 = float(site.bed_height(np.array(0.0), np.array(pr.y))) * ZEX
        top = 2.0 * ZEX if pr.intact else z0 + 3.0 * ZEX
        # A token burial below the bed is enough to seat the pier on it.
        layers.append(cylinder_mesh(0.0, pr.y, pr.radius, z0 - 0.4, top,
                                    color="#9aa0a6" if pr.intact
                                    else "#c2410c"))
    if eta is not None:
        trk = eta[::10, :3].copy()
        trk[:, 2] *= ZEX
        layers.append(polyline_rod(trk, r=0.55, color="#f97316"))
    render(ax, layers, elev=27, azim=-62, zoom=1.0)

    plt.tight_layout(pad=0.2)
    out_a = f"{OUT}/varuna_scene.png"
    plt.savefig(out_a, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close()
    trim_png(out_a)
    print("wrote", os.path.basename(out_a))

    # ---------------------------------------------------- close inspection
    fig = plt.figure(figsize=(7.6, 5.6), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection="3d", facecolor="white")
    tgt = site.targets["car_1"]
    tx, ty = float(tgt["centre"][0]), float(tgt["centre"][1])
    tsz = np.asarray(tgt["size"], float)
    tbed = float(site.bed_height(np.array(tx), np.array(ty)))

    PATCH = 7.0
    veh = np.array([tx - 4.6, ty - 3.0, tbed + 2.6])
    yaw = np.arctan2(ty - veh[1], tx - veh[0])

    # Warm bed so the cyan insonified patch reads as a sensor footprint.
    # Draw the bed over the whole insonified fan, not a square centred on
    # the target. The vehicle sits near the corner of that square, so a 130
    # degree aperture at 7 m range reached well outside it and the wedge
    # edges ended in mid air with no terrain under them.
    fan = yaw + np.radians(np.linspace(-FOV_H / 2, FOV_H / 2, 41))
    fx = veh[0] + PATCH * np.cos(fan)
    fy = veh[1] + PATCH * np.sin(fan)
    bx0 = min(fx.min(), veh[0], tx - tsz[0]) - 0.6
    bx1 = max(fx.max(), veh[0], tx + tsz[0]) + 0.6
    by0 = min(fy.min(), veh[1], ty - tsz[1]) - 0.6
    by1 = max(fy.max(), veh[1], ty + tsz[1]) + 0.6
    L2 = [bed_mesh(site, bx0, bx1, by0, by1,
                   0.14, lit=(veh, yaw, PATCH), cmap="copper_r")]
    # The target reads as a car rather than an anonymous slab: a lower body
    # with a cabin set back on it, keeping the overall envelope the site
    # model actually uses so the figure still shows the modelled object.
    CAR_YAW = 0.5
    body_h = tsz[2] * 0.55
    L2.append(box_mesh((tx, ty, tbed + body_h / 2),
                       (tsz[0], tsz[1], body_h), yaw=CAR_YAW,
                       color="#3f4b5b"))
    cab = (tsz[0] * 0.46, tsz[1] * 0.86, tsz[2] * 0.45)
    off = -tsz[0] * 0.08
    L2.append(box_mesh((tx + off * np.cos(CAR_YAW),
                        ty + off * np.sin(CAR_YAW),
                        tbed + body_h + cab[2] / 2), cab, yaw=CAR_YAW,
                       color="#55647a"))

    # Aperture edges, marched onto the bed the same way the DVL beams are.
    # Projecting them a fixed distance left them flying above the terrain and
    # off the top of the frame, because the ray dropped only 0.18 m over
    # seven metres of ground range instead of following the real depression
    # onto the patch.
    for sgn in (-1, 1):
        a = yaw + sgn * np.radians(FOV_H / 2)
        d = np.array([np.cos(a), np.sin(a), -(veh[2] - tbed) / PATCH])
        d /= np.linalg.norm(d)
        hit = veh.copy()
        # Stop at the patch range whether or not the bed is reached. Where
        # the bed falls away from the vehicle an edge never intersects it,
        # and an uncapped march ran the ray straight out of the frame.
        for _ in range(int(PATCH / 0.04)):
            nxt = hit + d * 0.04
            if nxt[2] <= float(site.bed_height(np.array(nxt[0]),
                                               np.array(nxt[1]))):
                break
            hit = nxt
        L2.append(rod(veh, hit, r=0.024, color="#0ea5e9"))

    # The four DVL beams at the 30 degree Janus angle actually modelled.
    for k in range(4):
        b = np.radians(45 + k * 90) + yaw
        d = np.array([np.sin(np.radians(30)) * np.cos(b),
                      np.sin(np.radians(30)) * np.sin(b),
                      -np.cos(np.radians(30))])
        hit = veh.copy()
        for _ in range(200):
            nxt = hit + d * 0.04
            if nxt[2] <= float(site.bed_height(np.array(nxt[0]),
                                               np.array(nxt[1]))):
                break
            hit = nxt
        L2.append(rod(veh, hit, r=0.014, color="#22c55e"))

    L2.append(vehicle_mesh(veh, yaw))

    sx, sy = veh[0] + 0.4, veh[1] - 1.9
    sb = np.array([sx, sy, float(site.bed_height(np.array(sx),
                                                 np.array(sy))) + 0.05])
    L2.append(rod(sb, sb + np.array([2.0, 0, 0]), r=0.04,
                  color="#1f2937"))
    render(ax, L2, elev=24, azim=-62, zoom=1.0, focal=0.5)

    plt.tight_layout(pad=0.2)
    out_b = f"{OUT}/varuna_detection.png"
    plt.savefig(out_b, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close()
    trim_png(out_b)
    print("wrote", os.path.basename(out_b))


if __name__ == "__main__":
    main()
