"""How much current can the vehicle actually hold against?

Balances steady surge drag against the maximum surge force the thruster
layout can deliver under per-thruster saturation.
"""
import numpy as np
from isonavi.dynamics import VehicleParams, bluerov_heavy_allocation

B = bluerov_heavy_allocation()
Bp = np.linalg.pinv(B)

def max_surge(p):
    """Largest pure surge wrench achievable before a thruster saturates."""
    f_unit = Bp @ np.array([1.0, 0, 0, 0, 0, 0])
    return p.max_thrust_n / np.max(np.abs(f_unit))

def hold_speed(p):
    """Current speed at which surge drag equals maximum surge thrust."""
    Fmax = max_surge(p)
    Xu, Xuu = p.lin_damp[0], p.quad_damp[0]
    # Xuu v^2 + Xu v - Fmax = 0
    disc = Xu ** 2 + 4 * Xuu * Fmax
    return (-Xu + np.sqrt(disc)) / (2 * Xuu)

p = VehicleParams()
print("BlueROV2-class baseline")
print(f"  per-thruster limit   : {p.max_thrust_n:.0f} N")
print(f"  max surge wrench     : {max_surge(p):.1f} N")
print(f"  surge drag coeffs    : Xu {p.lin_damp[0]:.1f}, Xuu {p.quad_damp[0]:.1f}")
print(f"  max holdable current : {hold_speed(p):.2f} m/s")
print()
print("drag required against current")
for v in (0.5, 1.0, 1.5, 2.0, 2.4, 3.0):
    d = p.quad_damp[0] * v * v + p.lin_damp[0] * v
    print(f"  {v:.1f} m/s -> {d:7.1f} N   "
          f"{'OK' if d <= max_surge(p) else 'EXCEEDS THRUST'}")
print()
print("what would be needed to hold 2.4 m/s")
v = 2.4
for name, Xuu, Xu in (("open frame (as-is)", 141.0, 13.7),
                      ("part faired", 70.0, 18.0),
                      ("fully faired hull", 32.0, 22.0)):
    d = Xuu * v * v + Xu * v
    per = d / np.max(np.abs(Bp @ np.array([1.0, 0, 0, 0, 0, 0]))) ** -1
    f_unit = np.max(np.abs(Bp @ np.array([1.0, 0, 0, 0, 0, 0])))
    per_thruster = d * f_unit
    print(f"  {name:20s}: drag {d:6.1f} N -> {per_thruster:6.1f} N per thruster")
