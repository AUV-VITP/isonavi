"""Parametric CAD model of VARUNA-1, built with CadQuery.

The geometry is driven by two things and invents nothing else:

  simulation/varuna/dynamics.py::VARUNA_1   mass, displacement, thruster arms
  cad/varuna_layout.py                      the solved component layout

The hull size comes from the layout solve, so the modelled hull encloses
exactly the volume the simulation uses for buoyancy, and every component sits
at the station the mass budget put it at. The model is therefore checkable: the
volume it reports back has to agree with the volume it was built from.

Outputs a full assembly, a cutaway for the internal arrangement, and an
exploded view, as STEP and STL.
"""

from __future__ import annotations

import json
import math
import os

import cadquery as cq

import varuna_layout as L

MM = 1000.0
DEG = 180.0 / math.pi

# Colours, kept consistent across every render and drawing.
C_HULL = cq.Color(0.86, 0.55, 0.16, 1.0)
C_WINDOW = cq.Color(0.16, 0.34, 0.52, 1.0)
C_DARK = cq.Color(0.20, 0.21, 0.26, 1.0)
C_ACCENT = cq.Color(0.74, 0.40, 0.10, 1.0)
C_METAL = cq.Color(0.62, 0.64, 0.68, 1.0)
C_BATT = cq.Color(0.18, 0.42, 0.28, 1.0)
C_PCB = cq.Color(0.10, 0.45, 0.35, 1.0)
C_LEAD = cq.Color(0.34, 0.35, 0.40, 1.0)


# ------------------------------------------------------------------ helpers
def align_z(solid, direction, origin):
    """Rotate a solid built along +Z onto `direction`, then move it to origin."""
    dx, dy, dz = direction
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-12:
        return solid.translate(origin)
    dx, dy, dz = dx / n, dy / n, dz / n
    ax, ay, az = -dy, dx, 0.0          # cross((0,0,1), d)
    an = math.sqrt(ax * ax + ay * ay)
    ang = math.acos(max(-1.0, min(1.0, dz))) * DEG
    if an < 1e-12:                      # already along +/-Z
        if dz < 0:
            solid = solid.rotate((0, 0, 0), (1, 0, 0), 180)
        return solid.translate(origin)
    return solid.rotate((0, 0, 0), (ax, ay, 0), ang).translate(origin)


def lens_strut(length, width, thick, taper=0.75):
    """A faired strut of lens section, lofted root to tip along +Z.

    Streamlined rather than a round bar, because these carry the thruster loads
    through the flow and a bluff strut would add drag where it matters least.
    """
    wp = (cq.Workplane()
          .moveTo(-width / 2, 0)
          .threePointArc((0, thick / 2), (width / 2, 0))
          .threePointArc((0, -thick / 2), (-width / 2, 0)).close()
          .workplane(offset=length)
          .moveTo(-width * taper / 2, 0)
          .threePointArc((0, thick * taper / 2), (width * taper / 2, 0))
          .threePointArc((0, -thick * taper / 2), (-width * taper / 2, 0))
          .close())
    return wp.loft(combine=True)


# ------------------------------------------------------------------ hull
def hull_profile(geom, n=72):
    """Body-frame (x, r) points from tail tip to nose tip, in millimetres.

    The layout builds the Myring profile from the nose, but the body frame has
    x forward, so the station is mirrored about the hull mid-length.
    """
    x_c = L.NOSE_L + geom["l_mid"] / 2.0
    pts = []
    # tail tip back to the mid-body
    for i in range(n, -1, -1):
        hx = L.TAIL_L * i / n
        pts.append(((x_c - (geom["x_mid_end"] + hx)) * MM, L.r_tail(hx) * MM))
    # mid-body
    pts.append(((x_c - L.NOSE_L) * MM, L.HULL_R * MM))
    # nose
    for i in range(n, -1, -1):
        hx = L.NOSE_L * i / n
        pts.append(((x_c - hx) * MM, L.r_nose(hx) * MM))
    # strip duplicate stations that would break the wire
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) > 1e-6:
            out.append(p)
    return out


def _revolve_profile(pts):
    """Revolve an (x, r) meridian about the X axis.

    The profile is closed back along the axis, skipping any endpoint that
    already sits on it: the Myring nose comes to a point, so a closing line
    there would be zero length and the kernel rejects it.
    """
    wp = cq.Workplane("XZ").polyline(pts)
    if pts[-1][1] > 1e-6:
        wp = wp.lineTo(pts[-1][0], 0.0)
    if pts[0][1] > 1e-6:
        wp = wp.lineTo(pts[0][0], 0.0)
    return wp.close().revolve(360, (0, 0, 0), (1, 0, 0))


def hull_solid(geom):
    return _revolve_profile(hull_profile(geom))


def hull_inner(geom, t):
    """The internal cavity, built by drawing the meridian in by the skin
    thickness.

    Offsetting the profile is used in preference to a shell operation on the
    solid: shelling a revolved body with a pointed nose is fragile in the
    kernel, whereas an inset meridian is exact and always closes. Stations
    where the inset radius would vanish are dropped, which leaves the nose and
    tail tips solid, as they are in the real structure.
    """
    pts = [(x, r - t) for x, r in hull_profile(geom) if r - t > 1.0]
    if len(pts) < 3:
        return None
    return _revolve_profile(pts)


X_WINDOW = 255.0   # body station where the acoustic window begins, mm


def nose_window(geom):
    """The forward acoustic window, the section the sonar looks through."""
    pts = [p for p in hull_profile(geom) if p[0] >= (0.255 * MM)]
    if len(pts) < 3:
        return None
    return _revolve_profile(pts)


# ------------------------------------------------------------------ parts
def thruster(duct_d=0.100, duct_l=0.088):
    """Ducted thruster: shroud, hub, three stators and a three-blade rotor."""
    rd = duct_d / 2 * MM
    ln = duct_l * MM
    duct = cq.Workplane("YZ").circle(rd).circle(rd - 5).extrude(ln)
    duct = duct.edges("%CIRCLE").fillet(1.5)
    hub = (cq.Workplane("YZ").workplane(offset=ln * 0.28)
           .circle(rd * 0.34).extrude(ln * 0.44))
    body = duct.union(hub)
    for k in range(3):                      # stator vanes
        body = body.union(
            cq.Workplane("YZ").workplane(offset=ln * 0.80)
            .transformed(rotate=(k * 120, 0, 0))
            .rect(2 * (rd - 5), 3.0).extrude(2.0, both=True))
    for k in range(3):                      # rotor blades, pitched
        blade = (cq.Workplane("YZ").workplane(offset=ln * 0.40)
                 .transformed(rotate=(k * 120 + 20, 0, 22))
                 .rect(1.62 * (rd - 5), 12.0).extrude(1.6, both=True))
        body = body.union(blade)
    return body


def propeller_free():
    return None


def fin(le_x, root_c, tip_c, span, sweep, r_le, r_te, thick=6.0):
    """A tail stabiliser, built in the XZ plane and extruded across the flow.

    The vehicle points along +x, so the planform runs aft from the leading edge
    and sweeps further aft as it goes outboard. The root line follows the tail
    cone: the hull is narrowing over the chord, so a straight root would lift
    off the skin at the trailing edge.
    """
    pts = [(le_x, r_le),
           (le_x - root_c, r_te),
           (le_x - sweep - tip_c, r_le + span),
           (le_x - sweep, r_le + span)]
    return (cq.Workplane("XZ").polyline(pts).close()
            .extrude(thick / 2, both=True))


def dvl_head():
    """Four beam Janus head: a body with four transducers at 30 deg."""
    body = (cq.Workplane("XY").circle(38).extrude(-26)
            .edges(">Z or <Z").fillet(3.0))
    for k in range(4):
        d = (math.sin(math.radians(30)) * math.cos(math.radians(45 + k * 90)),
             math.sin(math.radians(30)) * math.sin(math.radians(45 + k * 90)),
             -math.cos(math.radians(30)))
        tx = (d[0] * 20, d[1] * 20, -22 + d[2] * 6)
        body = body.union(align_z(cq.Workplane().circle(11).extrude(20),
                                  d, tx))
    return body


def battery_pack():
    return (cq.Workplane("XY").box(210, 108, 74)
            .edges("|X").fillet(6.0))


def electronics_stack():
    """Three boards on standoffs, the flight computer, the interface and power."""
    a = cq.Workplane("XY")
    stack = None
    for i, z in enumerate((-26, 0, 26)):
        board = cq.Workplane("XY", origin=(0, 0, z)).box(150, 96, 3.0)
        stack = board if stack is None else stack.union(board)
    for sx in (-1, 1):
        for sy in (-1, 1):
            post = (cq.Workplane("XY", origin=(sx * 66, sy * 40, 0))
                    .circle(3.0).extrude(30, both=True))
            stack = stack.union(post)
    return stack


def esc_bank():
    return (cq.Workplane("XY").box(120, 92, 44).edges("|Z").fillet(5.0))


def sonar_head():
    """The forward looking array, a flat faced transducer block."""
    return (cq.Workplane("YZ").circle(52).extrude(56)
            .edges(">X").fillet(6.0))


def ring_frame(radius_mm):
    return (cq.Workplane("YZ").circle(radius_mm).circle(radius_mm - 9)
            .extrude(7, both=True))


def ballast_blocks(mass_kg, x_mm, z_mm):
    """Lead trim, split into blocks bolted along the keel rails."""
    vol_mm3 = mass_kg / L.RHO_LEAD * MM ** 3
    n = 5
    each = vol_mm3 / n
    w, h = 96.0, 30.0
    ln = each / (w * h)
    out = None
    for i in range(n):
        x = x_mm + (i - (n - 1) / 2) * (ln + 6)
        blk = (cq.Workplane("XY", origin=(x, 0, z_mm))
               .box(ln, w, h).edges("|Z").fillet(4.0))
        out = blk if out is None else out.union(blk)
    return out


# ------------------------------------------------------------------ assembly
def build(cutaway=False, explode=0.0):
    parts, geom, v_hull = L.solve_layout()
    pos = {p.name: p.pos for p in parts}
    mass = {p.name: p.mass for p in parts}

    hull = hull_solid(geom)
    assy = cq.Assembly()

    def add(solid, name, color, offset=(0, 0, 0)):
        if solid is None:
            return
        if explode:
            solid = solid.translate(tuple(o * explode for o in offset))
        assy.add(solid, name=name, color=color)

    # -- hull, shown as a shell when cut away so the inside is visible
    if cutaway:
        # Hollow the hull, then take away the near half so the arrangement is
        # visible from the standard viewing side.
        inner = hull_inner(geom, L.SKIN_T * MM)
        shell = hull.cut(inner) if inner is not None else hull
        knife = cq.Workplane("XY").box(2400, 1200, 1200).translate((0, -600, 0))
        add(shell.cut(knife), "hull", C_HULL)
    else:
        # Split the skin at the window station rather than laying a second
        # solid over it: coincident faces render as z-fighting artefacts and
        # would double-count the volume.
        knife = (cq.Workplane("XY").box(4000, 1000, 1000)
                 .translate((X_WINDOW + 2000, 0, 0)))
        add(hull.cut(knife), "hull", C_HULL)
        add(hull.intersect(knife), "acoustic window", C_WINDOW)

    # -- internals
    p = pos["battery pack 14S4P"]
    add(battery_pack().translate((p[0] * MM, p[1] * MM, p[2] * MM)),
        "battery", C_BATT, (0, 0, -1))
    p = pos["electronics stack"]
    add(electronics_stack().translate((p[0] * MM, p[1] * MM, p[2] * MM)),
        "electronics", C_PCB, (0, 0, 1))
    p = pos["thruster ESC bank"]
    add(esc_bank().translate((p[0] * MM, p[1] * MM, p[2] * MM)),
        "esc bank", C_DARK, (0, 0, 1))
    p = pos["forward looking sonar"]
    add(sonar_head().translate((p[0] * MM - 28, p[1] * MM, p[2] * MM)),
        "sonar head", C_METAL, (1, 0, 0))
    p = pos["trim ballast"]
    add(ballast_blocks(mass["trim ballast"], p[0] * MM, p[2] * MM),
        "trim ballast", C_LEAD, (0, 0, -1))
    p = pos["inertial unit"]
    add(cq.Workplane("XY", origin=(p[0] * MM, p[1] * MM, p[2] * MM))
        .box(58, 58, 26).edges("|Z").fillet(4), "inertial unit", C_DARK,
        (0, 0, 1))

    # ring frames at the mid-body quarter points
    for xr in (-0.20, 0.0, 0.20):
        add(ring_frame(L.HULL_R * MM - L.SKIN_T * MM)
            .translate((xr * MM, 0, 0)), f"ring frame {xr}", C_METAL)

    # equipment rails along the keel
    for sy in (-1, 1):
        add(cq.Workplane("XY", origin=(0, sy * 52, -0.052 * MM))
            .box(620, 16, 10), f"rail {sy}", C_METAL)

    p = pos["aft closure and penetrator plate"]
    add(cq.Workplane("YZ", origin=(p[0] * MM, 0, 0)).circle(72).extrude(14),
        "aft closure", C_METAL, (-1, 0, 0))

    # -- external: thrusters on swept pylons rooted on the cylindrical body
    x_mid_fwd = (L.NOSE_L + geom["l_mid"] / 2) - L.NOSE_L      # body x of nose end
    for sx in (+1, -1):
        for sy in (+1, -1):
            tip = (sx * L.ARM_LX * MM, sy * L.ARM_LY * MM, 0.0)
            root = (sx * 0.22 * MM, sy * (L.HULL_R - 0.004) * MM, 0.0)
            d = tuple(t - r for t, r in zip(tip, root))
            ln = math.sqrt(sum(c * c for c in d))
            add(align_z(lens_strut(ln, 52, 17), d, root),
                f"pylon h {sx}{sy}", C_ACCENT, (0, sy, 0))
            cant = 45 * sx * sy if sx > 0 else 135 * (1 if sy > 0 else -1)
            cant = {(1, 1): 45, (1, -1): -45,
                    (-1, 1): 135, (-1, -1): -135}[(sx, sy)]
            add(thruster().rotate((0, 0, 0), (0, 0, 1), cant).translate(tip),
                f"thruster h {sx}{sy}", C_DARK, (sx * 0.4, sy, 0))

    z_v = (L.HULL_R + 0.055) * MM
    for sx in (+1, -1):
        for sy in (+1, -1):
            tip = (sx * L.ARM_VX * MM, sy * L.ARM_VY * MM, z_v)
            root = (sx * 0.20 * MM, sy * 0.045 * MM,
                    (L.HULL_R - 0.004) * MM * 0.80)
            d = tuple(t - r for t, r in zip(tip, root))
            ln = math.sqrt(sum(c * c for c in d))
            add(align_z(lens_strut(ln, 46, 16), d, root),
                f"pylon v {sx}{sy}", C_ACCENT, (0, sy, 0.4))
            add(thruster(0.092, 0.080)
                .rotate((0, 0, 0), (0, 1, 0), 90).translate(tip),
                f"thruster v {sx}{sy}", C_DARK, (0, sy, 1))

    # -- cruciform stabilisers on the tail cone
    x_c = L.NOSE_L + geom["l_mid"] / 2
    le_x, root_c = -0.320, 0.155

    def hull_r_at(bx):
        """Hull radius at a body station, following the tail cone."""
        hx = x_c - bx
        return L.r_tail(max(0.0, hx - geom["x_mid_end"]))

    r_le = hull_r_at(le_x) * MM - 3
    r_te = hull_r_at(le_x - root_c) * MM - 3
    for ang in (0, 90, 180, 270):
        f = fin(le_x * MM, root_c * MM, 72, 104, 54, r_le, r_te)
        add(f.rotate((0, 0, 0), (1, 0, 0), ang), f"fin {ang}", C_ACCENT,
            (0, 0, 0))

    # -- belly acoustics
    p = pos["doppler velocity log"]
    add(dvl_head().translate((p[0] * MM, 0, p[2] * MM + 20)),
        "doppler velocity log", C_METAL, (0, 0, -1))
    p = pos["depth transducer"]
    add(cq.Workplane("XY", origin=(p[0] * MM, 0, p[2] * MM)).circle(13)
        .extrude(16), "depth transducer", C_METAL, (0, 0, -1))

    return assy, hull, geom, v_hull, parts


if __name__ == "__main__":
    out = os.path.expanduser("~/dev/rakshatech/cad")
    os.makedirs(out, exist_ok=True)

    assy, hull, geom, v_hull, parts = build()
    b = L.budget(parts)

    built = hull.val().Volume() / MM ** 3
    err = 100 * (built - v_hull) / v_hull

    cq.exporters.export(assy.toCompound(), f"{out}/varuna_vehicle.step")
    cq.exporters.export(assy.toCompound(), f"{out}/varuna_vehicle.stl",
                        tolerance=0.25, angularTolerance=0.12)

    cut, _, _, _, _ = build(cutaway=True)
    cq.exporters.export(cut.toCompound(), f"{out}/varuna_cutaway.stl",
                        tolerance=0.25, angularTolerance=0.12)

    exp, _, _, _, _ = build(explode=260.0)
    cq.exporters.export(exp.toCompound(), f"{out}/varuna_exploded.stl",
                        tolerance=0.3, angularTolerance=0.15)

    info = {
        "hull_length_mm": round(geom["length"] * MM, 1),
        "hull_diameter_mm": round(L.HULL_D * MM, 1),
        "hull_slenderness": round(geom["length"] / L.HULL_D, 2),
        "midbody_length_mm": round(geom["l_mid"] * MM, 1),
        "wetted_area_m2": round(geom["area"], 4),
        "skin_thickness_mm": L.SKIN_T * MM,
        "hull_volume_m3": round(v_hull, 5),
        "built_hull_volume_m3": round(built, 5),
        "hull_volume_error_pct": round(err, 3),
        "displaced_volume_m3": round(b["volume_m3"], 5),
        "mass_kg": round(b["mass_kg"], 3),
        "net_buoyancy_N": round(b["net_buoyancy_N"], 3),
        "cg_mm": [round(c * MM, 2) for c in b["cg"]],
        "cb_mm": [round(c * MM, 2) for c in b["cb"]],
        "bg_mm": round(b["bg_z"] * MM, 2),
        "trim_offset_mm": round(b["trim_x_offset"] * MM, 4),
        "ballast_kg": round([p.mass for p in parts
                             if p.name == "trim ballast"][0], 3),
        "n_thrusters": 8,
        "n_parts": len(assy.children),
    }
    with open(f"{out}/varuna_cad_params.json", "w") as f:
        json.dump(info, f, indent=1)

    print("built VARUNA-1")
    for k, v in info.items():
        print(f"  {k}: {v}")
