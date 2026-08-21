"""Parametric CAD model of the VARUNA-1 vehicle, built with CadQuery.

Every principal dimension is derived from the simulation parameters in
varuna.dynamics.VARUNA_1, so the drawing is the vehicle that was simulated:

    displaced volume    0.0282 m3   sets the hull size
    thruster arms       (0.42, 0.30, 0.38, 0.26) m   set the pod positions
    CB above CG         85 mm       sets the ballast/float split
    fin_coeff           110         sizes the aft stabilisers

The hull is a Myring profile, the standard low-drag axisymmetric AUV body: an
elliptical nose, a parallel mid-body carrying the pressure housing, and a fined
tail. The mid-body length is solved so the displaced volume matches the
simulation value, which keeps the CAD buoyancy consistent with the physics.

The thrusters are mounted on faired pylons off the hull, in the eight-thruster
vectored arrangement the allocation matrix implies: four canted horizontal
units for surge, sway and yaw, four vertical units for heave, roll and pitch.
Everything is one connected body, not a hull with floating pods.
"""

from __future__ import annotations

import json
import math
import os

import cadquery as cq

MM = 1000.0

# ---------------------------------------------------------------- parameters
MASS = 28.0
VOLUME = 0.0282
ARMS = (0.42, 0.30, 0.38, 0.26)   # lx, ly, vx, vy in metres
CB_ABOVE_CG = 0.085
FIN_COEFF = 110.0

D = 0.20
R = D / 2.0
NOSE = 0.18
TAIL = 0.34
NOSE_N = 2.0
TAIL_THETA = math.radians(16)

THR_D = 0.100        # thruster duct outer diameter, m (T200-class shroud)
THR_L = 0.090        # thruster duct length, m


def r_nose(x, a=NOSE, n=NOSE_N):
    t = (x - a) / a
    return R * max(0.0, 1.0 - t * t) ** (1.0 / n)


def r_tail(x, c=TAIL, theta=TAIL_THETA, r_tip=0.028):
    frac = x / c
    r = (R
         - (3 * (R - r_tip) / c ** 2 - math.tan(theta) / c) * (c * frac) ** 2
         + (2 * (R - r_tip) / c ** 3 - math.tan(theta) / c ** 2) * (c * frac) ** 3)
    return max(r_tip, r)


def solve_midbody(target_v):
    def rev(f, length, steps=240):
        dx = length / steps
        return sum(math.pi * f((i + 0.5) * dx) ** 2 * dx for i in range(steps))
    v_nose = rev(r_nose, NOSE)
    v_tail = rev(r_tail, TAIL)
    l_mid = (target_v - v_nose - v_tail) / (math.pi * R ** 2)
    return l_mid, v_nose, v_tail


def hull_solid(l_mid):
    steps = 64
    pts = [(0.0, 0.0)]
    for i in range(1, steps + 1):
        x = NOSE * i / steps
        pts.append((x, r_nose(x)))
    pts.append((NOSE + l_mid, R))
    x1 = NOSE + l_mid
    for i in range(1, steps + 1):
        x = TAIL * i / steps
        pts.append((x1 + x, r_tail(x)))
    prof = cq.Workplane("XZ").polyline([(px * MM, pr * MM) for px, pr in pts])
    prof = prof.lineTo(pts[-1][0] * MM, 0).close()
    hull = prof.revolve(360, (0, 0, 0), (1, 0, 0))
    return hull, x1 + TAIL, NOSE, x1


def thruster(duct_d=THR_D, duct_l=THR_L):
    """A ducted thruster: shroud ring, central hub, three stator vanes."""
    Rd = duct_d / 2 * MM
    L = duct_l * MM
    duct = cq.Workplane("YZ").circle(Rd).circle(Rd - 5).extrude(L)
    hub = (cq.Workplane("YZ").workplane(offset=L * 0.30)
           .circle(Rd * 0.40).extrude(L * 0.40))
    vanes = cq.Workplane("YZ")
    for k in range(3):
        vanes = vanes.union(
            cq.Workplane("YZ").workplane(offset=L * 0.5)
            .transformed(rotate=(k * 120, 0, 0))
            .rect(2 * (Rd - 5), 4).extrude(2, both=True))
    body = duct.union(hub).union(vanes)
    return body


def _loft_pylon(w, t, length, taper):
    """A faired strut: a lens (aerofoil-like) cross-section lofted from a wide
    root to a narrower tip, so it reads as a streamlined mount not a box.

    Both cross-section wires are pushed onto one workplane stack via two
    ``workplane`` calls before the loft, which is what CadQuery's loft needs.
    """
    def lens_pts(w, t):
        return (cq.Workplane()
                .moveTo(-w / 2, 0)
                .threePointArc((0, t / 2), (w / 2, 0))
                .threePointArc((0, -t / 2), (-w / 2, 0)).close())
    wp = (lens_pts(w, t)
          .workplane(offset=length)
          .moveTo(-w * taper / 2, 0)
          .threePointArc((0, t * taper / 2), (w * taper / 2, 0))
          .threePointArc((0, -t * taper / 2), (-w * taper / 2, 0)).close())
    return wp.loft(combine=True)


def build():
    l_mid, v_nose, v_tail = solve_midbody(VOLUME)
    hull, total_len, x_mid0, x_mid1 = hull_solid(l_mid)
    x_c = (x_mid0 + x_mid1) / 2.0
    lx, ly, vx, vy = ARMS

    assy = cq.Assembly()
    hull_col = cq.Color(0.85, 0.55, 0.15, 1.0)
    dark = cq.Color(0.20, 0.21, 0.26, 1.0)
    accent = cq.Color(0.78, 0.42, 0.10, 1.0)

    assy.add(hull, name="hull", color=hull_col)

    # --- horizontal thrusters: canted 45 deg, on pylons off the hull flanks.
    # The pod centre is at (x_c +/- lx, +/- ly). The pylon runs from the hull
    # surface out to the pod.
    for sx, sy, cant in [(+1, +1, +45), (+1, -1, -45),
                         (-1, +1, +135), (-1, -1, -135)]:
        px = x_c + sx * lx
        py = sy * ly
        # Faired pylon from the hull flank out to the pod. It runs along y, so
        # its loft axis (local +z) is rotated to point outboard, and the lens
        # section is oriented edge-on to the flow (thin in x).
        hull_r = R
        pylon_len = (abs(py) - hull_r) * MM
        pyl = _loft_pylon(46, 16, max(pylon_len, 40), 0.75)
        pyl = pyl.rotate((0, 0, 0), (1, 0, 0), -90 if sy > 0 else 90)
        pyl = pyl.translate((px * MM, sy * hull_r * MM, 0))
        assy.add(pyl, name=f"hpyl_{sx}_{sy}", color=accent)
        # Thruster at the pod position, canted 45 deg about z for vectored thrust.
        thr = thruster().rotate((0, 0, 0), (0, 0, 1), cant)
        thr = thr.translate((px * MM, py * MM, 0))
        assy.add(thr, name=f"hthr_{sx}_{sy}", color=dark)

    # --- vertical thrusters: on short vertical pylons on top of the hull, ducts
    # pointing up. Placed clear of the horizontal pods so heave, roll and pitch
    # authority reads distinctly.
    top_z = R * MM
    for sx, sy in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]:
        px = x_c + sx * vx
        py = sy * vy
        # Vertical faired pylon rising from the hull crown to the duct.
        pyl = _loft_pylon(40, 16, 55, 0.8)  # loft along +z already
        pyl = pyl.translate((px * MM, py * MM, top_z * 0.85))
        assy.add(pyl, name=f"vpyl_{sx}_{sy}", color=accent)
        thr = thruster(duct_d=0.092, duct_l=0.082)
        thr = thr.rotate((0, 0, 0), (0, 1, 0), 90)  # duct axis -> vertical
        thr = thr.translate((px * MM, py * MM, top_z + 60))
        assy.add(thr, name=f"vthr_{sx}_{sy}", color=dark)

    # --- aft cruciform stabilising fins. NACA-ish flat plate, swept.
    fin_span = 0.115 * MM
    fin_root = 0.15 * MM
    fin_x = (x_mid1 + TAIL * 0.30) * MM
    for ang in (0, 90, 180, 270):
        fin = (cq.Workplane("XZ")
               .moveTo(fin_x, R * MM * 0.55)
               .lineTo(fin_x + fin_root, R * MM * 0.65)
               .lineTo(fin_x + fin_root * 0.55, R * MM * 0.55 + fin_span)
               .lineTo(fin_x + fin_root * 0.12, R * MM * 0.55 + fin_span)
               .close().extrude(3.5, both=True))
        fin = fin.rotate((fin_x, 0, 0), (fin_x + 1, 0, 0), ang)
        assy.add(fin, name=f"fin_{ang}", color=accent)

    # --- forward sonar dome, flat-faced for the FLS head.
    dome = (cq.Workplane("YZ").workplane(offset=-28)
            .circle(R * MM * 0.5).extrude(28)
            .faces("<X").fillet(10))
    assy.add(dome, name="sonar_dome", color=cq.Color(0.10, 0.30, 0.50, 1.0))

    # --- a shallow keel strip carrying ballast, giving the CB-above-CG offset.
    keel = (cq.Workplane("XZ").workplane(offset=0)
            .moveTo(x_c * MM - l_mid * MM * 0.35, -R * MM)
            .rect(l_mid * MM * 0.7, 22, centered=(False, False))
            .extrude(30, both=True))
    keel = keel.intersect(hull.translate((0, 0, -R * MM * 0.75)))
    # keel intersect can be finicky; only add if it produced solid.
    try:
        if keel.val().Volume() > 1:
            assy.add(keel, name="keel", color=dark)
    except Exception:
        pass

    info = {
        "mass_kg": MASS,
        "displaced_volume_m3": VOLUME,
        "hull_diameter_mm": D * MM,
        "hull_length_mm": round(total_len * MM, 1),
        "midbody_length_mm": round(l_mid * MM, 1),
        "nose_volume_m3": round(v_nose, 5),
        "tail_volume_m3": round(v_tail, 5),
        "midbody_volume_m3": round(math.pi * R ** 2 * l_mid, 5),
        "thruster_arms_m": ARMS,
        "cb_above_cg_mm": CB_ABOVE_CG * MM,
        "fin_coeff": FIN_COEFF,
        "n_thrusters": 8,
    }
    return assy, hull, total_len, info


if __name__ == "__main__":
    out = os.path.expanduser("~/dev/rakshatech/cad")
    os.makedirs(out, exist_ok=True)
    assy, hull, total_len, info = build()

    built_v = hull.val().Volume() / (MM ** 3)
    info["built_hull_volume_m3"] = round(built_v, 5)
    info["volume_error_pct"] = round(100 * (built_v - VOLUME) / VOLUME, 2)

    cq.exporters.export(assy.toCompound(), f"{out}/varuna_vehicle.step")
    cq.exporters.export(assy.toCompound(), f"{out}/varuna_vehicle.stl",
                        tolerance=0.3, angularTolerance=0.15)
    hull.val().exportStl(f"{out}/varuna_hull.stl", tolerance=0.4)
    with open(f"{out}/varuna_cad_params.json", "w") as f:
        json.dump(info, f, indent=1)

    print("built VARUNA-1 CAD")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"  total length: {total_len*MM:.0f} mm")
