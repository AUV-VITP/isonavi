"""Renders and drawings of the VARUNA-1 CAD model.

Tessellates the CadQuery assembly directly so every part keeps its own colour,
rather than flattening to a single-colour mesh. Produces the hero view, a four
view general arrangement, a dimensioned drawing, a labelled cutaway of the
internal layout, and an exploded view.

No GPU is involved: triangles are shaded with a Lambert term and drawn by
matplotlib, which is enough for engineering figures and runs anywhere.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import cadquery as cq

import varuna_cad as V
import varuna_layout as L

OUT = os.path.expanduser("~/dev/rakshatech/cad")
MM = 1000.0

INK = "#20242c"
BLUE = "#16407A"
GREY = "#6b7280"

# Items sealed inside the hull. Painter's algorithm over a long thin hull is
# not reliable enough to hide them behind the skin, and physically they should
# not be visible at all, so external views simply leave them out.
# Above this angle between a facet and its smoothed normal, the
# edge is treated as real and left sharp.
COS_CREASE = np.cos(np.radians(38.0))

INTERNAL = ("battery", "electronics", "esc bank", "sonar head", "trim ballast",
            "inertial unit", "ring frame", "rail", "aft closure")



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


# ------------------------------------------------------------------ meshing
def tessellate(assy, tol=0.22, ang=0.09, skip=()):
    """Flatten an assembly to (triangles, shading normals, per-triangle rgb).

    Shading normals are smooth vertex normals, not facet normals: facet
    normals make every band of a revolved surface a separate tone, which is
    what produces the striping on a hull. Facet normals are accumulated at
    shared vertices, and each triangle is shaded by the mean of its three
    vertex normals.
    """
    tris, norms, cols = [], [], []
    for child in assy.children:
        if any(child.name.startswith(sk) for sk in skip):
            continue
        shape = child.obj
        if isinstance(shape, cq.Workplane):
            shape = shape.val()
        if shape is None:
            continue
        try:
            verts, faces = shape.tessellate(tol, ang)
        except Exception:
            continue
        v = np.array([[p.x, p.y, p.z] for p in verts], float)
        f = np.array(faces, int)
        if len(f) == 0:
            continue

        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        # The kernel already winds faces outward consistently: measured, every
        # facet of the hull and of a solid block points out. Re-orienting them
        # against the part centroid, which an earlier version did, corrupts
        # exactly the cases that matter, flipping the inner surface of a duct
        # and flapping unstably on a flat plate where no centroid direction is
        # meaningful. So the winding is trusted as given.

        # Weld coincident vertices so normals blend across face boundaries.
        _, inv = np.unique(np.round(v, 4), axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        vn = np.zeros((int(inv.max()) + 1, 3))
        for k in range(3):
            np.add.at(vn, inv[f[:, k]], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        sn = vn[inv[f]].mean(axis=1)
        sn /= np.maximum(np.linalg.norm(sn, axis=1, keepdims=True), 1e-12)
        # Crease test: where smoothing has swung the normal far from the face
        # it belongs to, the edge is real and the facet normal is kept.
        crease = np.einsum("ij,ij->i", sn, fn) < COS_CREASE
        sn[crease] = fn[crease]

        rgb = child.color.toTuple()[:3] if child.color else (0.75, 0.75, 0.78)
        tris.append(v[f])
        norms.append(sn)
        cols.append(np.tile(np.array(rgb, float), (len(f), 1)))
    return (np.concatenate(tris), np.concatenate(norms),
            np.concatenate(cols))


def shade(n, base, elev, azim, ambient=0.30):
    """Blinn-Phong over smooth vertex normals.

    Ambient comes from above so undersides fall away, and a specular lobe
    gives the shell the gloss of painted composite rather than matte card.
    """
    le, la = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(le) * np.cos(la), np.cos(le) * np.sin(la),
                     np.sin(le)])

    def dirv(d_el, d_az):
        a, b = np.radians(elev + d_el), np.radians(azim + d_az)
        return np.array([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b),
                         np.sin(a)])

    key, fill = dirv(30, 38), dirv(-8, -75)
    amb = ambient * (0.62 + 0.38 * np.clip(n[:, 2], -1, 1))
    dif = 0.72 * np.clip(n @ key, 0, 1) + 0.26 * np.clip(n @ fill, 0, 1)
    half = key + view
    half /= np.linalg.norm(half)
    # A tight highlight bands badly, because PolyCollection fills each
    # triangle with one colour and cannot interpolate across it. A broader
    # lobe spreads the same energy over enough facets to read as smooth.
    spec = 0.26 * np.clip(n @ half, 0, 1) ** 13
    return np.clip(base * (amb + dif)[:, None] + spec[:, None], 0, 1)


def depth_sort(tris, norms, cols, elev, azim):
    """Order triangles far to near, so nearer ones paint over."""
    le, la = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(le) * np.cos(la), np.cos(le) * np.sin(la),
                  np.sin(le)])
    order = np.argsort(tris.mean(axis=1) @ d)
    return tris[order], norms[order], cols[order]


def draw(ax, tris, norms, cols, elev, azim, zoom=0.92):
    tris, norms, cols = depth_sort(tris, norms, cols, elev, azim)
    ax.add_collection3d(Poly3DCollection(
        tris, facecolors=shade(norms, cols, elev, azim),
        edgecolors="none", linewidths=0))
    allv = tris.reshape(-1, 3)
    ctr = allv.mean(0)
    rng = (allv.max(0) - allv.min(0)).max() / 2 * zoom
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def frac(ax, p3):
    """Body point in mm to axes-fraction coordinates for the current view."""
    xs, ys, _ = proj3d.proj_transform(p3[0], p3[1], p3[2], ax.get_proj())
    return ax.transAxes.inverted().transform(ax.transData.transform((xs, ys)))


# ------------------------------------------------------------------ figures
def hero(tris, norms, cols, info):
    fig = plt.figure(figsize=(13.5, 7.6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    draw(ax, tris, norms, cols, 19, -57, zoom=0.86)
    ax.text2D(0.015, 0.95, "VARUNA-1", transform=ax.transAxes, fontsize=25,
              fontweight="bold", color=BLUE)
    ax.text2D(0.015, 0.885,
              f"{info['hull_length_mm']:.0f} x {info['hull_diameter_mm']:.0f}"
              f" mm, {info['mass_kg']:.1f} kg, 8 vectored thrusters\n"
              "every dimension derived from the simulation parameters",
              transform=ax.transAxes, fontsize=11.5, color="#444")
    plt.tight_layout()
    plt.savefig(f"{OUT}/varuna_hero.png", dpi=155, facecolor="white",
                bbox_inches="tight")
    plt.close()
    trim_png(f"{OUT}/varuna_hero.png")
    print("wrote varuna_hero.png")

    # A caption-free version. On the title page the surrounding document
    # already names the vehicle, so the overlay only competes with it.
    fig = plt.figure(figsize=(13.5, 6.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    draw(ax, tris, norms, cols, 19, -57, zoom=0.94)
    plt.tight_layout()
    plt.savefig(f"{OUT}/varuna_hero_plain.png", dpi=155, facecolor="white",
                bbox_inches="tight")
    plt.close()
    trim_png(f"{OUT}/varuna_hero_plain.png")
    print("wrote varuna_hero_plain.png")


def general_arrangement(tris, norms, cols, info):
    views = [("side elevation", 0, -90), ("plan", 89.9, -90),
             ("three quarter", 20, -58), ("head on", 0, 0)]
    fig = plt.figure(figsize=(15, 9.2), facecolor="white")
    for k, (name, el, az) in enumerate(views):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d", facecolor="white")
        draw(ax, tris, norms, cols, el, az)
        ax.set_title(name, color=INK, fontsize=12, pad=-2)
    fig.suptitle("VARUNA-1 general arrangement", fontsize=17, color=BLUE,
                 fontweight="bold", y=0.975)
    fig.text(0.5, 0.935,
             f"{info['hull_length_mm']:.0f} mm overall, "
             f"{info['hull_diameter_mm']:.0f} mm hull diameter, "
             f"L/D {info['hull_slenderness']:.1f}, {info['mass_kg']:.1f} kg, "
             "8 vectored thrusters",
             ha="center", fontsize=10.5, color="#444")
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(f"{OUT}/varuna_ga.png", dpi=145, facecolor="white",
                bbox_inches="tight")
    plt.close()
    trim_png(f"{OUT}/varuna_ga.png")
    print("wrote varuna_ga.png")


LIGHT = np.array([0.36, 0.52, 0.78])
LIGHT /= np.linalg.norm(LIGHT)


def ortho(ax, tris, norms, cols, screen, view, ambient=0.45):
    """Draw a true orthographic projection onto a normal 2D axes.

    Engineering views want an exact parallel projection and a real aspect
    ratio. A 3D axes forces a cubic box, which wastes most of the page on a
    long thin vehicle and makes dimension placement guesswork, so the meridian
    is projected directly instead.
    """
    order = np.argsort(tris.mean(axis=1) @ np.array(view, float))
    t, n, c = tris[order], norms[order], cols[order]
    lam = np.clip(n @ LIGHT, 0, 1)
    amb = ambient * (0.66 + 0.34 * np.clip(n[:, 2], -1, 1))
    fc = np.clip(c * (amb + (1 - ambient) * lam)[:, None], 0, 1)
    polys = t[:, :, screen]
    ax.add_collection(PolyCollection(polys, facecolors=fc, edgecolors="none"))
    lo = polys.reshape(-1, 2).min(axis=0)
    hi = polys.reshape(-1, 2).max(axis=0)
    pad = 0.10 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def dim2(ax, p1, p2, text, off, vertical=False, fs=9.5):
    """Dimension between two points, in drawing millimetres."""
    p1, p2 = np.array(p1, float), np.array(p2, float)
    o = np.array([0.0, off]) if not vertical else np.array([off, 0.0])
    a, b = p1 + o, p2 + o
    ax.annotate("", xy=tuple(a), xytext=tuple(b),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color=INK,
                                shrinkA=0, shrinkB=0))
    for s0, s1 in ((p1, a), (p2, b)):
        ax.plot([s0[0], s1[0]], [s0[1], s1[1]], lw=0.6, color=GREY,
                ls=(0, (3, 3)), zorder=1)
    m = (a + b) / 2
    ax.text(m[0], m[1], text, fontsize=fs, color=INK, ha="center",
            va="center", rotation=90 if vertical else 0,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none"))


def title_block(ax, info):
    """Drawing title block. The vehicle name is a header rather than a row,
    because a long value in a narrow column collides with its own label."""
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.93, fill=False,
                               ec="#c9ccd1", lw=1.0, transform=ax.transAxes))
    ax.text(0.5, 0.93, "VARUNA-1", transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=BLUE, ha="center", va="top")
    ax.text(0.5, 0.875, "RakshaTech Synapse 2026", transform=ax.transAxes,
            fontsize=8, color=GREY, ha="center", va="top")
    ax.annotate("", xy=(0.08, 0.845), xytext=(0.92, 0.845),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#c9ccd1"))
    rows = [("hull", f"{info['hull_length_mm']:.0f} x "
                     f"{info['hull_diameter_mm']:.0f}"),
            ("slenderness", f"L/D {info['hull_slenderness']:.1f}"),
            ("mass", f"{info['mass_kg']:.1f} kg"),
            ("displacement", f"{info['displaced_volume_m3'] * 1000:.1f} L"),
            ("net buoyancy", f"+{info['net_buoyancy_N']:.2f} N"),
            ("BG", f"{info['bg_mm']:.1f} mm"),
            ("trim ballast", f"{info['ballast_kg']:.1f} kg"),
            ("thrusters", f"{info['n_thrusters']} vectored"),
            ("units", "millimetres")]
    y = 0.79
    for k, v in rows:
        ax.text(0.07, y, k, transform=ax.transAxes, fontsize=8.5,
                color=GREY, va="center")
        ax.text(0.93, y, v, transform=ax.transAxes, fontsize=9,
                color=INK, va="center", ha="right", fontweight="bold")
        y -= 0.082


def dimensioned(tris, norms, cols, info, geom):
    """Side elevation and plan, orthographic, with principal dimensions."""
    x_c = L.NOSE_L + geom["l_mid"] / 2
    nose, tail = x_c * MM, (x_c - geom["length"]) * MM
    r = L.HULL_R * MM
    zv = (L.HULL_R + 0.055) * MM + 46

    fig = plt.figure(figsize=(13.6, 8.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[4.6, 1.0],
                          height_ratios=[1, 1], hspace=0.06, wspace=0.04)

    ax = fig.add_subplot(gs[0, 0])
    ortho(ax, tris, norms, cols, screen=[0, 2], view=(0, 1, 0))
    dim2(ax, (tail, -r), (nose, -r), f"{info['hull_length_mm']:.0f}", -150)
    dim2(ax, (tail + 12, -r), (tail + 12, r),
         f"{info['hull_diameter_mm']:.0f}", -95, vertical=True)
    dim2(ax, (-L.ARM_LX * MM, -r), (L.ARM_LX * MM, -r),
         f"{2 * L.ARM_LX * MM:.0f} thruster stations", -74)
    dim2(ax, (nose - 30, 0), (nose - 30, zv), f"{zv:.0f}", 128, vertical=True)
    ax.set_title("side elevation", color=INK, fontsize=12, loc="left")

    ax = fig.add_subplot(gs[1, 0])
    ortho(ax, tris, norms, cols, screen=[0, 1], view=(0, 0, 1))
    dim2(ax, (0, -L.ARM_LY * MM), (0, L.ARM_LY * MM),
         f"{2 * L.ARM_LY * MM:.0f} across thrusters", 470, vertical=True)
    dim2(ax, (tail, L.ARM_LY * MM), (nose, L.ARM_LY * MM),
         f"{info['hull_length_mm']:.0f}", 150)
    ax.annotate("horizontal thrusters canted 45 degrees",
                xy=(L.ARM_LX * MM, L.ARM_LY * MM),
                xytext=(L.ARM_LX * MM - 250, L.ARM_LY * MM + 210),
                fontsize=8.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=GREY))
    ax.set_title("plan", color=INK, fontsize=12, loc="left")

    tb = fig.add_subplot(gs[:, 1])
    title_block(tb, info)

    # The title and its subtitle overlapped here, and both repeated the
    # caption. Removing them fixes the collision and frees the space.
    plt.savefig(f"{OUT}/varuna_dimensioned.png", dpi=155, facecolor="white",
                bbox_inches="tight")
    plt.close()
    trim_png(f"{OUT}/varuna_dimensioned.png")
    print("wrote varuna_dimensioned.png")


CALLOUTS = [
    ("forward looking sonar", (0.360, 0, 0.0)),
    ("electronics stack", (-0.090, 0, 0.020)),
    ("thruster ESC bank", (-0.155, 0, -0.010)),
    ("aft closure", (-0.300, 0, 0.0)),
    ("ring frame", (-0.200, 0, 0.062)),
    ("battery pack, 4.0 kg", (0.080, 0, -0.045)),
    ("trim ballast, 9.0 kg", (-0.023, 0, -0.062)),
    ("equipment rails", (0.180, 0, -0.052)),
    ("doppler velocity log", (0.090, 0, -0.108)),
]


def cutaway():
    """Labelled section. Leaders run to two columns so nothing overlaps."""
    assy, _, _, _, _ = V.build(cutaway=True)
    tris, norms, cols = tessellate(assy)
    fig = plt.figure(figsize=(15.0, 8.8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    draw(ax, tris, norms, cols, 16, -74, zoom=0.70)
    fig.canvas.draw()

    pts = [(lab, frac(ax, (p[0] * MM, p[1] * MM, p[2] * MM)))
           for lab, p in CALLOUTS]
    left = sorted([q for q in pts if q[1][0] < 0.52], key=lambda q: -q[1][1])
    right = sorted([q for q in pts if q[1][0] >= 0.52], key=lambda q: -q[1][1])

    def place(items, col_x, ha):
        if not items:
            return
        top, bot = 0.84, 0.14
        step = (top - bot) / max(len(items) - 1, 1)
        for i, (lab, anchor) in enumerate(items):
            ax.annotate(lab, xy=anchor, xytext=(col_x, top - i * step),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        fontsize=10, color=INK, ha=ha, va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.9, color=GREY,
                                        shrinkA=2, shrinkB=2,
                                        connectionstyle="arc3,rad=0.05"),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#c9ccd1", lw=0.7, alpha=0.96))

    place(left, 0.015, "left")
    place(right, 0.985, "right")

    # Titled by its caption in the document, not here.
    plt.savefig(f"{OUT}/varuna_cutaway.png", dpi=155, facecolor="white",
                bbox_inches="tight")
    plt.close()
    trim_png(f"{OUT}/varuna_cutaway.png")
    print("wrote varuna_cutaway.png")


def exploded():
    assy, _, _, _, _ = V.build(explode=260.0)
    tris, norms, cols = tessellate(assy, 0.25, 0.10)
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    # Titled by its caption in the document, not here.
    draw(ax, tris, norms, cols, 18, -56, zoom=1.0)
    plt.savefig(f"{OUT}/varuna_exploded.png", dpi=150, facecolor="white",
                bbox_inches="tight")
    plt.close()
    trim_png(f"{OUT}/varuna_exploded.png")
    print("wrote varuna_exploded.png")


if __name__ == "__main__":
    info = json.load(open(f"{OUT}/varuna_cad_params.json"))
    assy, _, geom, _, _ = V.build()
    tris, norms, cols = tessellate(assy, skip=INTERNAL)
    print(f"tessellated {len(tris)} triangles from {len(assy.children)} parts")
    hero(tris, norms, cols, info)
    general_arrangement(tris, norms, cols, info)
    dimensioned(tris, norms, cols, info, geom)
    cutaway()
    exploded()
