"""Component layout, mass budget and stability budget for isonavi-1.

This is the engineering content behind the CAD. Every part the vehicle carries
is listed with a real mass and a real position, and the free variables are
solved rather than assumed:

  1. the hull size, so displaced volume equals the value the simulation uses
     for buoyancy;
  2. the trim ballast mass, so total dry mass equals the simulated mass;
  3. the ballast longitudinal position, so the vehicle floats level.

The outcome is therefore a prediction, not a decoration. In particular the
separation between centre of buoyancy and centre of gravity, which sets the
passive roll and pitch restoring moment, falls out of the layout and can be
checked against what the dynamics model assumes.

Architecture
------------
The faired hull is itself the pressure boundary, the arrangement used by survey
vehicles of this class, rather than a free flooding fairing wrapped around a
separate cylindrical housing. Two things forced that choice:

  - a separate housing leaves the buoyancy to syntactic foam, and the foam
    volume required here does not fit in the thin annulus between a 150 mm
    housing and a 180 mm skin;
  - trimming that arrangement to a useful stability margin needs ballast in a
    bulb below the hull, and a keel bulb is a snag hazard in the debris field
    this vehicle is built to survey.

With the hull as the pressure vessel the buoyancy is the hull, the ballast sits
inside on the keel line, and nothing protrudes below the skin except the
acoustics.

Body frame is x forward, y to port, z up, origin on the hull axis at hull mid
length. Positions are the centroid of each item in metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ------------------------------------------------------------------ targets
# From simulation/isonavi/dynamics.py::isonavi_1. The layout has to satisfy these.
TARGET_MASS = 28.0        # kg
TARGET_VOLUME = 0.0282    # m3 displaced
RHO_WATER = 1000.0        # freshwater, this is a flood scenario
G = 9.81

# ------------------------------------------------------------------ materials
RHO_AL = 2700.0           # 6061-T6
RHO_LEAD = 11340.0
RHO_COMP = 1800.0         # carbon/glass laminate

# ------------------------------------------------------------------ hull form
# Myring profile, the standard low drag axisymmetric body. Nose and tail are
# held at fixed multiples of the diameter so the form stays similar as the
# mid-body is solved.
HULL_D = 0.180
HULL_R = HULL_D / 2
NOSE_L = 0.90 * HULL_D
TAIL_L = 1.70 * HULL_D
NOSE_N = 2.0
TAIL_THETA = math.radians(16)
SKIN_T = 0.004            # laminate thickness, the pressure boundary
TAIL_TIP_R = 0.030


def r_nose(x):
    t = (x - NOSE_L) / NOSE_L
    return HULL_R * max(0.0, 1.0 - t * t) ** (1.0 / NOSE_N)


def r_tail(x):
    c, rt = TAIL_L, TAIL_TIP_R
    a2 = 3 * (HULL_R - rt) / c ** 2 - math.tan(TAIL_THETA) / c
    a3 = 2 * (HULL_R - rt) / c ** 3 - math.tan(TAIL_THETA) / c ** 2
    return max(rt, HULL_R - a2 * x ** 2 + a3 * x ** 3)


def _revolve(f, length, steps=400, area=False, x_offset=0.0, moment=False):
    """Volume or lateral area of a body of revolution, and its first moment.

    With ``moment`` set, returns the integral of the quantity weighted by the
    axial station, measured from the nose plus ``x_offset``. Dividing the two
    gives the centroid, which the trim solve needs and cannot assume.
    """
    dx = length / steps
    tot = 0.0
    for i in range(steps):
        x0, x1 = i * dx, (i + 1) * dx
        r0, r1 = f(x0), f(x1)
        if area:
            d = math.pi * (r0 + r1) * math.hypot(dx, r1 - r0)
        else:
            d = math.pi / 3 * (r0 * r0 + r0 * r1 + r1 * r1) * dx
        tot += d * (x_offset + 0.5 * (x0 + x1)) if moment else d
    return tot


def hull_centroids(geom):
    """Body-frame x of the hull's volume centroid and of its shell centroid.

    Volume centroid sets where the buoyancy acts; shell centroid sets where the
    skin's own mass acts. Both are aft of the hull mid-length because the tail
    is the longer end.
    """
    x_c = NOSE_L + geom["l_mid"] / 2.0
    x_tail0 = NOSE_L + geom["l_mid"]

    def mid(f):
        return math.pi * HULL_R ** 2 * geom["l_mid"]

    v = (_revolve(r_nose, NOSE_L)
         + math.pi * HULL_R ** 2 * geom["l_mid"]
         + _revolve(r_tail, TAIL_L))
    vm = (_revolve(r_nose, NOSE_L, moment=True)
          + math.pi * HULL_R ** 2 * geom["l_mid"] * (NOSE_L + geom["l_mid"] / 2)
          + _revolve(r_tail, TAIL_L, x_offset=x_tail0, moment=True))

    a = (_revolve(r_nose, NOSE_L, area=True)
         + 2 * math.pi * HULL_R * geom["l_mid"]
         + _revolve(r_tail, TAIL_L, area=True))
    am = (_revolve(r_nose, NOSE_L, area=True, moment=True)
          + 2 * math.pi * HULL_R * geom["l_mid"] * (NOSE_L + geom["l_mid"] / 2)
          + _revolve(r_tail, TAIL_L, area=True, x_offset=x_tail0, moment=True))

    # Hull stations run aft from the nose; the body frame has x forward with
    # its origin at the hull mid-length, so the sense flips.
    return x_c - vm / v, x_c - am / a


def hull_geometry(v_target):
    """Solve the mid-body length so the hull encloses v_target, and report it."""
    v_nose = _revolve(r_nose, NOSE_L)
    v_tail = _revolve(r_tail, TAIL_L)
    l_mid = (v_target - v_nose - v_tail) / (math.pi * HULL_R ** 2)
    a_nose = _revolve(r_nose, NOSE_L, area=True)
    a_tail = _revolve(r_tail, TAIL_L, area=True)
    a_mid = 2 * math.pi * HULL_R * l_mid
    return {
        "l_mid": l_mid,
        "length": NOSE_L + l_mid + TAIL_L,
        "v_nose": v_nose, "v_tail": v_tail,
        "v_mid": math.pi * HULL_R ** 2 * l_mid,
        "area": a_nose + a_mid + a_tail,
        "x_nose_end": NOSE_L,
        "x_mid_end": NOSE_L + l_mid,
    }


@dataclass
class Part:
    """One item in the vehicle.

    mass    kg in air
    pos     (x, y, z) centroid, metres, body frame
    vol     m3 of water displaced: the enclosed volume for the pressure hull
            and for sealed external pods, the material volume for wetted
            structure, and zero for anything already inside the hull
    group   for the reported breakdown
    """
    name: str
    mass: float
    pos: tuple
    vol: float
    group: str


# Thruster stations, from the allocation arms in isonavi_1.
ARM_LX, ARM_LY, ARM_VX, ARM_VY = 0.42, 0.30, 0.38, 0.26
THR_POD_V = 0.00040       # sealed volume of one thruster pod, m3
BALLAST_Z = -0.062        # on the keel line inside the hull, clear of the skin
DROP_WEIGHT = 2.00        # kg, the releasable part of the trim, carried outside
RHO_ZINC = 7140.0


def external_parts():
    """Items outside the pressure hull, which add their own displacement."""
    p = []
    a = p.append
    z_v = HULL_R + 0.055          # vertical pods stand off the hull crown
    for sx in (+1, -1):
        for sy in (+1, -1):
            a(Part(f"thruster h {sx:+d}{sy:+d}", 0.34,
                   (sx * ARM_LX, sy * ARM_LY, 0.0), THR_POD_V, "propulsion"))
            a(Part(f"thruster v {sx:+d}{sy:+d}", 0.34,
                   (sx * ARM_VX, sy * ARM_VY, z_v), THR_POD_V, "propulsion"))
    a(Part("thruster pylons", 1.20, (0.0, 0, 0.030), 1.20 / RHO_COMP,
           "structure"))
    a(Part("stabilising fins", 0.48, (-0.360, 0, 0.0), 0.48 / RHO_COMP,
           "structure"))
    a(Part("doppler velocity log", 0.75, (0.090, 0, -HULL_R - 0.012),
           0.00080, "sensors"))
    a(Part("depth transducer", 0.05, (-0.150, 0, -HULL_R - 0.006), 0.00004,
           "sensors"))

    # -- corrosion protection. Aluminium and stainless in seawater is a
    # galvanic pair, so zinc is fitted to be eaten first. Two, so that one
    # damaged anode does not leave a section unprotected.
    for sx in (+1, -1):
        a(Part(f"sacrificial anode {sx:+d}", 0.15,
               (sx * 0.210, 0, -HULL_R - 0.004), 0.15 / RHO_ZINC,
               "structure"))

    # -- recovery. The lifting eye sits over the centre of gravity so the
    # vehicle comes out of the water level rather than swinging.
    a(Part("lifting eye", 0.16, (-0.035, 0, HULL_R + 0.010), 0.00003,
           "structure"))

    # -- surface location. A vehicle that has surfaced in a flooded river is
    # useless if it cannot be found, so the mast carries GPS, an Iridium
    # transceiver and a strobe.
    a(Part("antenna and strobe mast", 0.28, (-0.255, 0, HULL_R + 0.075),
           0.00012, "avionics"))

    # -- emergency ascent. Part of the trim ballast is carried as a burn wire
    # release on the belly. Dropping it converts the vehicle from very
    # slightly positive to strongly positive, which is what recovers it from
    # entanglement or a flat battery.
    a(Part("drop weight, releasable", DROP_WEIGHT,
           (0.010, 0, -HULL_R - 0.014), DROP_WEIGHT / RHO_LEAD, "trim"))
    return p


def internal_parts(hull_mass, shell_x=0.0):
    """Items inside the pressure hull. They displace nothing extra."""
    p = []
    a = p.append
    a(Part("hull shell, pressure boundary", hull_mass, (shell_x, 0, 0.0), 0.0,
           "structure"))
    a(Part("aft closure and penetrator plate", 0.80, (-0.300, 0, 0.0), 0.0,
           "structure"))
    a(Part("ring frames", 0.90, (0.0, 0, 0.0), 0.0, "structure"))
    a(Part("equipment rails", 0.80, (0.0, 0, -0.030), 0.0, "structure"))
    a(Part("battery pack 14S4P", 4.00, (0.080, 0, -0.045), 0.0, "power"))
    a(Part("electronics stack", 0.85, (-0.090, 0, 0.020), 0.0, "avionics"))
    a(Part("thruster ESC bank", 0.45, (-0.155, 0, -0.010), 0.0, "avionics"))
    a(Part("inertial unit", 0.12, (0.0, 0, 0.005), 0.0, "sensors"))
    a(Part("forward looking sonar", 1.30, (0.360, 0, 0.0), 0.0, "sensors"))
    a(Part("cabling and penetrators", 0.50, (-0.230, 0, 0.0), 0.0, "structure"))
    return p


def solve_layout():
    """Solve hull size, ballast mass and ballast station."""
    ext = external_parts()
    v_ext = sum(q.vol for q in ext)
    v_hull = TARGET_VOLUME - v_ext
    geom = hull_geometry(v_hull)

    hull_mass = geom["area"] * SKIN_T * RHO_COMP
    hull_x, shell_x = hull_centroids(geom)
    parts = ext + internal_parts(hull_mass, shell_x)

    ballast_m = TARGET_MASS - sum(q.mass for q in parts)
    ballast_v = 0.0        # inside the hull, already counted in v_hull

    # Longitudinal trim: with the ballast free in x, put the centre of gravity
    # under the centre of buoyancy so the vehicle floats level.
    hull_part = Part("hull enclosed volume", 0.0, (hull_x, 0, 0.0), v_hull,
                     "structure")
    known = parts + [hull_part]
    M = sum(q.mass for q in known) + ballast_m
    V = sum(q.vol for q in known)
    cb_x = sum(q.vol * q.pos[0] for q in known) / V
    mx = sum(q.mass * q.pos[0] for q in known)
    ballast_x = (cb_x * M - mx) / ballast_m

    all_parts = known + [Part("trim ballast", ballast_m,
                              (ballast_x, 0, BALLAST_Z), ballast_v, "trim")]
    return all_parts, geom, v_hull


def budget(parts):
    """Mass, displacement, centres and the stability separation."""
    M = sum(q.mass for q in parts)
    V = sum(q.vol for q in parts)
    cg = tuple(sum(q.mass * q.pos[i] for q in parts) / M for i in range(3))
    cb = tuple(sum(q.vol * q.pos[i] for q in parts) / V for i in range(3))
    return {
        "mass_kg": M,
        "volume_m3": V,
        "weight_N": M * G,
        "buoyancy_N": V * RHO_WATER * G,
        "net_buoyancy_N": V * RHO_WATER * G - M * G,
        "cg": cg,
        "cb": cb,
        "bg_z": cb[2] - cg[2],
        "trim_x_offset": cb[0] - cg[0],
    }


if __name__ == "__main__":
    parts, geom, v_hull = solve_layout()
    b = budget(parts)

    print("isonavi-1 mass and stability budget")
    print("=" * 64)
    print(f"{'item':36s}{'mass kg':>10s}{'x mm':>9s}{'z mm':>9s}")
    print("-" * 64)
    for q in sorted(parts, key=lambda r: -r.mass):
        if q.mass <= 0:
            continue
        print(f"{q.name:36s}{q.mass:10.3f}{q.pos[0] * 1000:9.1f}"
              f"{q.pos[2] * 1000:9.1f}")
    print("-" * 64)
    groups = {}
    for q in parts:
        groups[q.group] = groups.get(q.group, 0.0) + q.mass
    for g, m in sorted(groups.items(), key=lambda kv: -kv[1]):
        label = "  group " + g
        print(f"{label:36s}{m:10.3f}{'':9s}{100 * m / b['mass_kg']:8.1f}%")
    print("=" * 64)
    print(f"  hull        {geom['length'] * 1000:.0f} mm long,"
          f" {HULL_D * 1000:.0f} mm diameter,"
          f" L/D {geom['length'] / HULL_D:.1f}")
    print(f"  mid-body    {geom['l_mid'] * 1000:.0f} mm,"
          f" wetted area {geom['area']:.3f} m2,"
          f" skin {SKIN_T * 1000:.0f} mm")
    print(f"  hull volume {v_hull:.5f} m3,"
          f" appendages {b['volume_m3'] - v_hull:.5f} m3")
    print()
    print(f"  total mass            {b['mass_kg']:9.3f} kg   "
          f"(target {TARGET_MASS})")
    print(f"  displaced volume      {b['volume_m3']:9.5f} m3  "
          f"(target {TARGET_VOLUME})")
    print(f"  weight                {b['weight_N']:9.2f} N")
    print(f"  buoyancy              {b['buoyancy_N']:9.2f} N")
    print(f"  net buoyancy          {b['net_buoyancy_N']:9.2f} N   "
          f"(positive is float up on failure)")
    print(f"  centre of gravity     x {b['cg'][0] * 1000:7.1f}  "
          f"z {b['cg'][2] * 1000:7.1f} mm")
    print(f"  centre of buoyancy    x {b['cb'][0] * 1000:7.1f}  "
          f"z {b['cb'][2] * 1000:7.1f} mm")
    print(f"  longitudinal trim     {b['trim_x_offset'] * 1000:9.3f} mm  "
          f"(zero is level)")
    print()
    net_dropped = (b["buoyancy_N"] - DROP_WEIGHT * RHO_LEAD
                   / RHO_LEAD * G * 0 - (b["mass_kg"] - DROP_WEIGHT) * G
                   - DROP_WEIGHT / RHO_LEAD * RHO_WATER * G)
    print(f"  emergency ascent")
    print(f"    drop weight         {DROP_WEIGHT:9.2f} kg releasable")
    print(f"    net buoyancy after  {net_dropped:9.2f} N   "
          f"(from {b['net_buoyancy_N']:.2f} N)")
    print()
    print(f"  BG separation         {b['bg_z'] * 1000:9.1f} mm")
    print(f"  simulation assumed    {85.0:9.1f} mm")
    print(f"  righting moment       {b['weight_N'] * b['bg_z']:9.2f} N.m/rad"
          f"   (assumed {b['weight_N'] * 0.085:.2f})")
    print()
    bal = [q for q in parts if q.name == "trim ballast"][0]
    print(f"  trim ballast          {bal.mass:9.2f} kg at x"
          f" {bal.pos[0] * 1000:.0f} mm")
    print(f"  payload growth margin {bal.mass:9.2f} kg   "
          f"(ballast tradeable for payload at fixed displacement)")
