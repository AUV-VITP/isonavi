"""Hydrodynamic coefficients estimated from the CAD geometry.

The dynamics model has been carrying drag and added mass coefficients that
were engineering estimates made before the vehicle had a shape. Now that it
has one, they can be derived from it and checked.

Two things are estimated here:

  drag        bare hull skin friction from the ITTC 1957 correlation line with
              a form factor for a body of revolution, plus an appendage build
              up over the actual ducts, pylons and fins in the model;

  added mass  Lamb's coefficients for the prolate spheroid of the same
              slenderness, plus the water entrained by the ducts.

Neither replaces CFD or a tow tank. What they do is establish whether the
numbers the simulation has been using are the right size, and show where the
drag actually comes from, which turns out not to be the hull.
"""

from __future__ import annotations

import json
import math
import os

import isonavi_layout as L

RHO = 1000.0
NU = 1.0e-6            # kinematic viscosity of water, m2/s
V_REF = 2.0            # m/s, the design survey speed


# ------------------------------------------------------------------ bare hull
def friction_coefficient(v, length):
    """ITTC 1957 correlation line."""
    re = v * length / NU
    return 0.075 / (math.log10(re) - 2.0) ** 2, re


def form_factor(d_over_l):
    """Form factor for a body of revolution.

    The usual Hoerner style fit: pressure drag over and above flat plate
    friction grows with fatness. At the slenderness here it is a small
    correction, which is the point of a Myring hull.
    """
    return 1.0 + 1.5 * d_over_l ** 1.5 + 7.0 * d_over_l ** 3


def bare_hull_drag(geom, v=V_REF):
    cf, re = friction_coefficient(v, geom["length"])
    k = form_factor(L.HULL_D / geom["length"])
    ct = k * cf
    # Force = 0.5 rho Ct S v^2, so the quadratic coefficient is the rest of it.
    return 0.5 * RHO * ct * geom["area"], dict(cf=cf, re=re, form=k, ct=ct)


# ------------------------------------------------------------------ appendages
# Dimensions as modelled in isonavi_cad.py.
DUCT_D, DUCT_L = 0.100, 0.088
DUCT_DV, DUCT_LV = 0.092, 0.080
PYLON_CHORD, PYLON_THICK = 0.052, 0.017
FIN_SPAN, FIN_CHORD, FIN_T = 0.104, 0.155, 0.006

CD_DUCT_AXIAL = 0.90       # short bluff shroud presented at an angle
CD_DUCT_CROSS = 1.00       # shroud broadside, a short cylinder in crossflow
CD_STRUT = 0.10            # streamlined section, on frontal area
CD_FIN = 0.012             # thin plate at zero incidence, on wetted area


def appendage_drag():
    """Quadratic drag coefficient from everything hung off the hull."""
    items = []

    # Four horizontal ducts, canted 45 degrees, so both the annulus and the
    # side of the shroud are presented to the flow.
    a_h = (DUCT_D * DUCT_L * math.cos(math.radians(45))
           + math.pi / 4 * DUCT_D ** 2 * math.sin(math.radians(45)))
    items.append(("horizontal thruster ducts", 4,
                  0.5 * RHO * CD_DUCT_AXIAL * a_h))

    # Four vertical ducts sit axis up, so they are broadside to the flow.
    a_v = DUCT_DV * DUCT_LV
    items.append(("vertical thruster ducts", 4,
                  0.5 * RHO * CD_DUCT_CROSS * a_v))

    # Eight pylons. Frontal area is thickness by span.
    arm_h = L.ARM_LY - L.HULL_R
    arm_v = 0.055
    items.append(("horizontal pylons", 4,
                  0.5 * RHO * CD_STRUT * PYLON_THICK * arm_h))
    items.append(("vertical pylons", 4,
                  0.5 * RHO * CD_STRUT * PYLON_THICK * arm_v))

    # Four fins, skin friction on both faces.
    items.append(("stabilising fins", 4,
                  0.5 * RHO * CD_FIN * 2 * FIN_SPAN * FIN_CHORD))

    total = sum(n * c for _, n, c in items)
    return total, items


# ------------------------------------------------------------------ added mass
def lamb_coefficients(slenderness):
    """Lamb's added mass factors for a prolate spheroid.

    Interpolated over the standard table in the slenderness range of interest.
    k1 acts along the axis, k2 across it.
    """
    table = {2.0: (0.209, 0.702), 3.0: (0.122, 0.803), 4.0: (0.082, 0.860),
             5.0: (0.059, 0.895), 6.0: (0.045, 0.918), 7.0: (0.036, 0.933),
             8.0: (0.029, 0.945), 10.0: (0.021, 0.960)}
    keys = sorted(table)
    s = min(max(slenderness, keys[0]), keys[-1])
    for a, b in zip(keys, keys[1:]):
        if a <= s <= b:
            f = (s - a) / (b - a)
            k1 = table[a][0] + f * (table[b][0] - table[a][0])
            k2 = table[a][1] + f * (table[b][1] - table[a][1])
            return k1, k2
    return table[keys[-1]]


def added_mass(geom):
    sl = geom["length"] / L.HULL_D
    k1, k2 = lamb_coefficients(sl)
    v_hull = L.TARGET_VOLUME
    axial = k1 * RHO * v_hull
    lateral = k2 * RHO * v_hull
    # Water inside and around the eight ducts moves with the vehicle.
    duct_vol = (4 * math.pi / 4 * DUCT_D ** 2 * DUCT_L
                + 4 * math.pi / 4 * DUCT_DV ** 2 * DUCT_LV)
    return axial, lateral, RHO * duct_vol, dict(slenderness=sl, k1=k1, k2=k2)


# ------------------------------------------------------------------ report
def main():
    parts, geom, v_hull = L.solve_layout()
    from isonavi_layout import TARGET_MASS

    hull_x, hull_info = bare_hull_drag(geom)
    app_x, items = appendage_drag()
    total_x = hull_x + app_x

    print("isonavi-1 hydrodynamic coefficients from the CAD geometry")
    print("=" * 72)
    print(f"  reference speed {V_REF:.1f} m/s, "
          f"Re {hull_info['re']:.2e}, Cf {hull_info['cf']:.5f}, "
          f"form factor {hull_info['form']:.3f}")
    print()
    print("  axial quadratic drag build up  (N per (m/s)^2)")
    print(f"    {'bare hull, friction plus form':34s}{hull_x:8.2f}"
          f"{100 * hull_x / total_x:7.1f} %")
    for name, n, c in items:
        print(f"    {name + f' (x{n})':34s}{n * c:8.2f}"
              f"{100 * n * c / total_x:7.1f} %")
    print(f"    {'total':34s}{total_x:8.2f}")
    print()
    sim_xuu = 32.0
    print(f"  simulation uses X_u|u| = {sim_xuu:.0f}, "
          f"CAD estimate {total_x:.1f}, ratio {total_x / sim_xuu:.2f}")
    print()
    print("  The hull is not what makes this vehicle draggy. The eight ducts")
    print(f"  and their pylons account for {100 * app_x / total_x:.0f} percent"
          " of axial drag, which is")
    print("  the price of hover authority on a torpedo hull.")

    ax, lat, duct_am, aminfo = added_mass(geom)
    print()
    print(f"  added mass, prolate spheroid at slenderness "
          f"{aminfo['slenderness']:.2f}")
    print(f"    k1 {aminfo['k1']:.3f}, k2 {aminfo['k2']:.3f}")
    print(f"    {'axial, hull':34s}{ax:8.2f} kg")
    print(f"    {'axial, entrained by ducts':34s}{duct_am:8.2f} kg")
    print(f"    {'axial, total':34s}{ax + duct_am:8.2f} kg"
          f"   (simulation uses 12.0)")
    print(f"    {'lateral, hull':34s}{lat:8.2f} kg"
          f"   (simulation uses 42.0)")

    # The design argument rests on holding station against the site current,
    # so the envelope is re-checked at the higher, CAD derived drag.
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "simulation"))
    from isonavi.dynamics import isonavi_1, vectored_allocation
    import numpy as np
    B = vectored_allocation(isonavi_1.arms)
    f = np.full(8, isonavi_1.max_thrust_n)
    surge_max = float(abs((B @ f)[0]))
    lin = isonavi_1.lin_damp[0]
    site_current = 2.4

    def envelope(quad):
        # surge_max = lin*v + quad*v^2
        return (-lin + math.sqrt(lin ** 2 + 4 * quad * surge_max)) / (2 * quad)

    v_sim = envelope(sim_xuu)
    v_cad = envelope(total_x)
    print()
    print("  station keeping envelope, re-checked at the CAD drag")
    print(f"    saturated surge thrust        {surge_max:.0f} N")
    print(f"    envelope at simulation drag   {v_sim:.2f} m/s")
    print(f"    envelope at CAD drag          {v_cad:.2f} m/s")
    print(f"    site surface current          {site_current:.1f} m/s")
    print(f"    margin retained               {v_cad / site_current:.2f}x")

    out = {
        "surge_thrust_N": surge_max,
        "envelope_sim": v_sim,
        "envelope_cad": v_cad,
        "site_current": site_current,
        "margin_cad": v_cad / site_current,
        "v_ref": V_REF,
        "reynolds": hull_info["re"],
        "cf": hull_info["cf"],
        "form_factor": hull_info["form"],
        "drag_hull": hull_x,
        "drag_appendages": app_x,
        "drag_total": total_x,
        "drag_sim": sim_xuu,
        "drag_ratio": total_x / sim_xuu,
        "appendage_share_pct": 100 * app_x / total_x,
        "added_axial_hull": ax,
        "added_axial_ducts": duct_am,
        "added_axial_total": ax + duct_am,
        "added_lateral_hull": lat,
        "slenderness": aminfo["slenderness"],
    }
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "isonavi_hydro.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n  wrote {os.path.basename(p)}")
    return out


if __name__ == "__main__":
    main()
