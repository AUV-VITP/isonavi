"""Smoke test: does the sonar model produce physically sensible imagery?

Checks, in order:
  1. Geometry primitives return correct analytic ranges.
  2. A ping against a flat bed produces a monotonic range/intensity structure.
  3. An object produces both a bright return and a shadow behind it.
  4. Timing is fast enough for closed-loop simulation.
"""
import time
import numpy as np

from varuna.geometry import Scene, Plane, Box, Cylinder, Sphere, Heightfield
from varuna.acoustics import ForwardLookingSonar, preset, MAT_INDEX

ok = lambda c, m: print(("  PASS  " if c else "  FAIL  ") + m)

print("=" * 62)
print("1. GEOMETRY: analytic intersection correctness")
print("=" * 62)

# Ray from origin along +x toward a plane at x = 10.
sc = Scene([Plane([10, 0, 0], [-1, 0, 0], MAT_INDEX["concrete"])])
h = sc.intersect(np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]))
ok(abs(h.t[0] - 10.0) < 1e-9, f"plane at x=10 -> t={h.t[0]:.6f}")

# Sphere of radius 2 centred at x = 20: first hit at 18.
sc = Scene([Sphere([20, 0, 0], 2.0, MAT_INDEX["steel"])])
h = sc.intersect(np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]))
ok(abs(h.t[0] - 18.0) < 1e-9, f"sphere r=2 at x=20 -> t={h.t[0]:.6f}")
ok(abs(h.normal[0, 0] + 1.0) < 1e-9, f"sphere normal faces ray: {h.normal[0]}")

# Cylinder radius 1.5 centred at (15,0), spanning z in [-10, 0].
sc = Scene([Cylinder([15, 0], 1.5, -10, 0, MAT_INDEX["concrete"])])
h = sc.intersect(np.array([[0.0, 0, -5.0]]), np.array([[1.0, 0, 0]]))
ok(abs(h.t[0] - 13.5) < 1e-9, f"cylinder r=1.5 at x=15 -> t={h.t[0]:.6f}")

# Box half-extent 1 centred at x = 8.
sc = Scene([Box([8, 0, 0], [1, 1, 1], MAT_INDEX["steel"])])
h = sc.intersect(np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]))
ok(abs(h.t[0] - 7.0) < 1e-9, f"box half=1 at x=8 -> t={h.t[0]:.6f}")

# Occlusion: nearer primitive must win.
sc = Scene([Sphere([20, 0, 0], 2.0, MAT_INDEX["steel"]),
            Sphere([10, 0, 0], 1.0, MAT_INDEX["concrete"])])
h = sc.intersect(np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]))
ok(abs(h.t[0] - 9.0) < 1e-9, f"nearest wins -> t={h.t[0]:.6f}")

# Heightfield: flat bed at z = -12 sampled by a downward ray from z = -2.
H = np.full((40, 40), -12.0)
hf = Heightfield(-20, -20, 1.0, 1.0, H, MAT_INDEX["silt"], max_range=40)
sc = Scene([hf])
h = sc.intersect(np.array([[0.0, 0, -2.0]]), np.array([[0.0, 0, -1.0]]))
ok(abs(h.t[0] - 10.0) < 0.05, f"flat heightfield -> t={h.t[0]:.4f} (want 10)")
ok(abs(h.normal[0, 2] - 1.0) < 1e-6, f"flat bed normal is +z: {h.normal[0]}")

# Sloped heightfield should give a tilted normal.
xs = np.arange(40) * 1.0 - 20
Hs = np.tile(-12.0 + 0.2 * xs, (40, 1))
hf2 = Heightfield(-20, -20, 1.0, 1.0, Hs, MAT_INDEX["silt"], max_range=40)
n = hf2.normal_at(np.array([0.0]), np.array([0.0]))
ok(abs(n[0, 0] + 0.2 / np.sqrt(1 + 0.04)) < 1e-3,
   f"slope 0.2 -> normal {np.round(n[0], 4)}")

print()
print("=" * 62)
print("2. SONAR: flat-bed ping structure")
print("=" * 62)

H = np.full((120, 120), -12.0)
bed = Heightfield(-60, -60, 1.0, 1.0, H, MAT_INDEX["silt"], max_range=80)
scene = Scene([bed])
cfg = preset("oculus", seed=0, r_max=40.0)
fls = ForwardLookingSonar(cfg, scene)

# Look forward and slightly down from 4 m above the bed.
frame = fls.ping([0, 0, -8.0, 0, np.radians(12), 0])
print(f"  polar shape       : {frame.polar.shape}")
print(f"  dB range          : {frame.polar.min():.1f} .. {frame.polar.max():.1f}")
print(f"  absorption        : {cfg.alpha*1000:.1f} dB/km at {cfg.freq_khz:.0f} kHz")
ok(np.isfinite(frame.polar).all(), "no NaN or Inf in the image")
ok(frame.polar.max() > frame.polar.min() + 10, "image has real dynamic range")
nvalid = np.sum(np.isfinite(frame.hit_range))
ok(nvalid > cfg.n_beams * 0.5, f"bed detected on {nvalid}/{cfg.n_beams} beams")

print()
print("=" * 62)
print("3. SONAR: object return and acoustic shadow")
print("=" * 62)

# Put a steel box on the bed, 18 m ahead, and compare against the empty scene.
scene2 = Scene([bed, Box([18, 0, -11.0], [1.2, 2.0, 1.0], MAT_INDEX["steel"],
                         name="target")])
fls2 = ForwardLookingSonar(preset("oculus", seed=0, r_max=40.0), scene2)
f_empty = ForwardLookingSonar(preset("oculus", seed=0, r_max=40.0), scene).ping(
    [0, 0, -8.0, 0, np.radians(12), 0])
f_obj = fls2.ping([0, 0, -8.0, 0, np.radians(12), 0])

cb = cfg.n_beams // 2
col_e = f_empty.polar[:, cb]
col_o = f_obj.polar[:, cb]
rng = f_obj.ranges
near = (rng > 16) & (rng < 20)
far = (rng > 21) & (rng < 30)
d_near = np.mean(col_o[near]) - np.mean(col_e[near])
d_far = np.mean(col_o[far]) - np.mean(col_e[far])
print(f"  mean dB change on target (16-20 m) : {d_near:+.2f} dB")
print(f"  mean dB change behind  (21-30 m) : {d_far:+.2f} dB")
ok(d_near > 2.0, "target region is brighter than empty bed")
ok(d_far < -2.0, "region behind target is shadowed")

print()
print("=" * 62)
print("4. PERFORMANCE")
print("=" * 62)
for name in ("aris", "oculus"):
    c = preset(name, seed=1)
    f = ForwardLookingSonar(c, scene2)
    f.ping([0, 0, -8.0, 0, np.radians(12), 0])  # warm up
    t0 = time.perf_counter()
    N = 12
    for i in range(N):
        f.ping([i * 0.1, 0, -8.0, 0, np.radians(12), 0])
    dt = (time.perf_counter() - t0) / N
    print(f"  {c.name:22s} {dt*1000:7.1f} ms/ping  ({1/dt:5.1f} Hz)  "
          f"{c.n_beams}x{c.n_elev_rays} rays")

print()
print("smoke test complete")
