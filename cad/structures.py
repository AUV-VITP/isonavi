"""Pressure hull structural check for isonavi-1.

Making the faired hull the pressure boundary is only defensible if the hull
survives the depth, so this sizes it. The governing failure mode for a thin
shell under external pressure is not material strength, it is elastic
instability: the skin buckles long before it is crushed. The two are computed
separately here and the smaller governs.

Everything is closed form. These are preliminary hand calculations of the kind
used to size a shell before analysis, not a substitute for finite element work
or a pressure test, and the report says so.

Geometry and material come from the same place the rest of the CAD does, so
changing the skin thickness or the frame spacing changes this result too.
"""

from __future__ import annotations

import json
import math
import os

import isonavi_layout as L

# ------------------------------------------------------------------ inputs
# Design depth. The modelled site is about 14 m deep. The rating is set well
# beyond it so the vehicle is not depth limited by the first scour hole or
# reservoir it is asked to work in.
DESIGN_DEPTH = 50.0        # m
RHO = 1000.0               # freshwater
G = 9.81

# Glass reinforced laminate, conservative values for a wet layup rather than a
# prepreg. A carbon skin would be stiffer and the margins would grow.
E = 25.0e9                 # Pa, hoop direction modulus
NU = 0.30
SIGMA_C = 250.0e6          # Pa, compressive strength

FRAME_SPACING = 0.200      # m, ring frames as modelled in isonavi_cad.py


def design_pressure(depth):
    return RHO * G * depth


def hoop_stress(p, r_mean, t):
    """Membrane hoop stress in a thin cylinder under external pressure."""
    return p * r_mean / t


def p_collapse_unstiffened(e, nu, t, r_mean):
    """Elastic collapse of a long unstiffened cylinder.

    The classical long-tube result: the shell behaves as a ring of unit length
    with second moment t^3/12, so instability arrives at
    p = E t^3 / (4 (1 - nu^2) r^3).
    """
    return e * t ** 3 / (4 * (1 - nu ** 2) * r_mean ** 3)


def p_collapse_stiffened(e, nu, t, d_mean, frame_spacing):
    """Elastic collapse between ring frames, Windenburg and Trilling form.

    Ring frames shorten the unsupported length, and the critical pressure rises
    steeply as that length falls. This is the standard closed form approximation
    to the von Mises solution and is what makes a 4 mm skin viable at all.
    """
    td = t / d_mean
    ld = frame_spacing / d_mean
    num = 2.42 * e * td ** 2.5
    den = (1 - nu ** 2) ** 0.75 * (ld - 0.45 * math.sqrt(td))
    return num / den


def collapse_depth(p):
    return p / (RHO * G)


# Thruster pylons, as modelled: a lens section cantilever carrying one
# thruster at full thrust. Chord lies along the flow, thickness across it.
PYLON_CHORD = 0.052        # m
PYLON_THICK = 0.017        # m
MAX_THRUST = 120.0         # N, one thruster at the saturation limit


def pylon_bending(chord, thick, arm, force):
    """Root bending stress of a lens section cantilever.

    Treated as a rectangle of the same envelope, which understates the real
    section modulus of a lens and so errs to the safe side.
    """
    i_chord = thick * chord ** 3 / 12.0        # bending in the chord plane
    z_chord = i_chord / (chord / 2)
    i_thick = chord * thick ** 3 / 12.0        # bending across the thickness
    z_thick = i_thick / (thick / 2)
    m = force * arm
    return m / z_chord, m / z_thick


def main():
    parts, geom, v_hull = L.solve_layout()
    t = L.SKIN_T
    r_out = L.HULL_R
    r_mean = r_out - t / 2
    d_mean = 2 * r_mean

    p_d = design_pressure(DESIGN_DEPTH)
    sigma = hoop_stress(p_d, r_mean, t)

    p_un = p_collapse_unstiffened(E, NU, t, r_mean)
    p_st = p_collapse_stiffened(E, NU, t, d_mean, FRAME_SPACING)
    d_un, d_st = collapse_depth(p_un), collapse_depth(p_st)

    sf_yield = SIGMA_C / sigma
    sf_buckle = p_st / p_d

    print("isonavi-1 pressure hull check")
    print("=" * 70)
    print(f"  skin                  {t * 1000:.1f} mm laminate,"
          f" mean radius {r_mean * 1000:.1f} mm")
    print(f"  ring frame spacing    {FRAME_SPACING * 1000:.0f} mm")
    print(f"  design depth          {DESIGN_DEPTH:.0f} m"
          f"   ({p_d / 1e6:.3f} MPa)")
    print()
    print("  membrane strength")
    print(f"    hoop stress         {sigma / 1e6:.1f} MPa compressive")
    print(f"    laminate strength   {SIGMA_C / 1e6:.0f} MPa")
    print(f"    safety factor       {sf_yield:.1f}")
    print()
    print("  elastic stability, the mode that actually governs")
    print(f"    unstiffened shell   collapses at {d_un:.0f} m"
          f"   ({p_un / 1e6:.3f} MPa)")
    print(f"    with ring frames    collapses at {d_st:.0f} m"
          f"   ({p_st / 1e6:.3f} MPa)")
    print(f"    safety factor       {sf_buckle:.1f} at the design depth")
    print()
    print(f"  The frames are load bearing, not tidy: they raise collapse depth")
    print(f"  from {d_un:.0f} m to {d_st:.0f} m, a factor of {d_st / d_un:.1f}.")
    print()
    print("  frame spacing sensitivity")
    print(f"    {'spacing mm':>12}{'collapse m':>13}{'SF':>8}")
    for sp in (0.15, 0.20, 0.30, 0.40, 0.60):
        p = p_collapse_stiffened(E, NU, t, d_mean, sp)
        print(f"    {sp * 1000:>12.0f}{collapse_depth(p):>13.0f}"
              f"{p / p_d:>8.1f}")

    # Pylon check. The horizontal pods stand furthest off the hull, so they
    # set the worst cantilever.
    arm = L.ARM_LY - (L.HULL_R - L.SKIN_T)
    s_chord, s_thick = pylon_bending(PYLON_CHORD, PYLON_THICK, arm, MAX_THRUST)
    worst = max(s_chord, s_thick)
    print()
    print("  thruster pylon, worst case one thruster at full thrust")
    print(f"    cantilever          {arm * 1000:.0f} mm")
    print(f"    root bending        {s_chord / 1e6:.1f} MPa in plane,"
          f" {s_thick / 1e6:.1f} MPa across")
    print(f"    safety factor       {SIGMA_C / worst:.0f}")

    out = {
        "pylon_arm_mm": arm * 1000,
        "pylon_stress_MPa": worst / 1e6,
        "pylon_sf": SIGMA_C / worst,
        "design_depth_m": DESIGN_DEPTH,
        "design_pressure_MPa": p_d / 1e6,
        "skin_mm": t * 1000,
        "frame_spacing_mm": FRAME_SPACING * 1000,
        "hoop_stress_MPa": sigma / 1e6,
        "laminate_strength_MPa": SIGMA_C / 1e6,
        "sf_yield": sf_yield,
        "collapse_depth_unstiffened_m": d_un,
        "collapse_depth_stiffened_m": d_st,
        "sf_buckling": sf_buckle,
        "frame_gain": d_st / d_un,
    }
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "isonavi_structures.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n  wrote {os.path.basename(p)}")

    # The sizing is only defensible if stability, not strength, is what is
    # being designed against; flag it if that ever stops being true.
    if sf_buckle > sf_yield:
        print("  NOTE: strength now governs rather than stability,"
              " revisit the assumptions")
    return out


if __name__ == "__main__":
    main()
