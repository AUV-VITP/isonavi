"""Verification of the CAD model against the layout it was built from.

The layout solves the vehicle analytically; the CAD builds it with a geometry
kernel. They are independent enough that agreement is worth something, and
disagreement has already caught one real error: the layout used to assume the
hull's displaced volume was centred at the hull mid-length, which the Myring
form does not satisfy, and the trim ballast was therefore 105 mm out of place.

Run this after changing either side.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import varuna_cad as V
import varuna_layout as L

MM = 1000.0
FAILURES = []


def check(name, got, want, tol, unit=""):
    ok = abs(got - want) <= tol
    mark = "PASS" if ok else "FAIL"
    if not ok:
        FAILURES.append(name)
    print(f"  {mark}  {name:44s} {got:10.4f} vs {want:10.4f} {unit}"
          f"  (tol {tol:g})")


def main():
    parts, geom, v_hull = L.solve_layout()
    b = L.budget(parts)
    hull = V.hull_solid(geom)
    c = hull.val().Center()
    vol = hull.val().Volume() / MM ** 3

    print("VARUNA-1 CAD verification")
    print("=" * 78)

    print("\n built geometry against the analytic solve")
    check("hull volume (m3)", vol, v_hull, 5e-5, "m3")
    hull_x, shell_x = L.hull_centroids(geom)
    check("hull volume centroid x (mm)", c.x, hull_x * MM, 0.5, "mm")
    check("hull centroid y (mm)", c.y, 0.0, 0.5, "mm")
    check("hull centroid z (mm)", c.z, 0.0, 0.5, "mm")

    print("\n budget against the simulation targets")
    check("total mass (kg)", b["mass_kg"], L.TARGET_MASS, 1e-6, "kg")
    check("displaced volume (m3)", b["volume_m3"], L.TARGET_VOLUME, 1e-9, "m3")
    check("longitudinal trim (mm)", b["trim_x_offset"] * MM, 0.0, 1e-3, "mm")

    print("\n derived quantities")
    check("net buoyancy (N)", b["net_buoyancy_N"], 1.962, 0.01, "N")
    print(f"  ....  {'BG separation (mm)':44s} {b['bg_z'] * MM:10.4f}"
          f"  (dynamics model uses {27.3:.1f})")

    print("\n the hull encloses its own internals")
    r_min = min(abs(p.pos[2]) for p in parts
                if p.name in ("battery pack 14S4P", "trim ballast"))
    inner_r = L.HULL_R - L.SKIN_T
    ok_fit = True
    for p in parts:
        if p.vol > 0 or p.name in ("hull enclosed volume",):
            continue
        if p.name.startswith("thruster") or "pylon" in p.name:
            continue
        rad = (p.pos[1] ** 2 + p.pos[2] ** 2) ** 0.5
        if rad > inner_r:
            ok_fit = False
            print(f"  FAIL  {p.name} sits {rad * MM:.1f} mm off axis, "
                  f"outside the {inner_r * MM:.1f} mm cavity")
    if ok_fit:
        print(f"  PASS  every internal component centroid lies within the "
              f"{inner_r * MM:.0f} mm cavity")
    else:
        FAILURES.append("internal fit")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
