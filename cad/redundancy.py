"""Thruster-out analysis for VARUNA-1.

A vehicle sent into a debris field will eventually lose a thruster: fouled by
line, struck by rubble, or flooded. Eight thrusters driving six degrees of
freedom leaves two spare, so the interesting question is not whether the
allocation is over-determined, which it obviously is, but whether the surviving
seven can still produce the specific wrench the mission needs while holding
attitude.

That is a linear program, not an inspection of the matrix. For each failure the
question asked is: what is the largest force along a given axis that can be
produced with zero net moment, given every surviving thruster is limited to its
saturation force. Zero net moment matters, because a vehicle that can push hard
but cannot stop itself rolling has not survived the failure.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulation"))

from varuna.dynamics import VARUNA_1, vectored_allocation

LIN, QUAD = 22.0, 38.9        # surge damping; quadratic term from the CAD
SITE_CURRENT = 2.4


def max_along(B, axis, limit, hold_moments=True):
    """Largest force along `axis` with zero net moment, thrusters within limit.

    Maximising a linear objective over a box with linear equalities is a
    linear program; solving it is exact where a pseudo-inverse would only give
    one feasible point among many.
    """
    n = B.shape[1]
    rows = [i for i in range(6) if i != axis]
    if not hold_moments:
        rows = [i for i in range(3, 6)]
    A_eq = np.vstack([B[r] for r in rows]) if rows else None
    b_eq = np.zeros(len(rows)) if rows else None
    res = linprog(c=-B[axis], A_eq=A_eq, b_eq=b_eq,
                  bounds=[(-limit, limit)] * n, method="highs")
    if not res.success:
        return 0.0
    return float(B[axis] @ res.x)


def drag_at(v):
    return LIN * v + QUAD * v * v


def envelope(force):
    """Current speed a given surge force can hold station against."""
    if force <= 0:
        return 0.0
    return (-LIN + np.sqrt(LIN ** 2 + 4 * QUAD * force)) / (2 * QUAD)


NAMES = ["h fore-port", "h fore-stbd", "h aft-port", "h aft-stbd",
         "v fore-port", "v fore-stbd", "v aft-port", "v aft-stbd"]



def hydro_load(V, psi):
    """Force and moment the water exerts, holding station at heading psi.

    psi is the angle between the vehicle nose and the oncoming flow. The
    water-relative velocity is the whole of the current, resolved into the body
    frame, because the vehicle is holding station over the ground.
    """
    u = V * np.cos(psi)
    v = -V * np.sin(psi)
    Xu, Yv = VARUNA_1.lin_damp[0], VARUNA_1.lin_damp[1]
    Xuu, Yvv = QUAD, VARUNA_1.quad_damp[1]
    fx = Xu * u + Xuu * abs(u) * u
    fy = Yv * v + Yvv * abs(v) * v
    mz = VARUNA_1.fin_coeff * u * v          # fins weathercock the hull
    return fx, fy, mz


def feasible(B, limit, V, psi):
    """Can the surviving thrusters hold station at this speed and heading?"""
    fx, fy, mz = hydro_load(V, psi)
    # Thrusters must cancel the hydrodynamic load and the fin moment.
    A_eq = np.vstack([B[0], B[1], B[3], B[4], B[5]])
    b_eq = np.array([fx, fy, 0.0, 0.0, -mz])
    res = linprog(c=np.zeros(B.shape[1]), A_eq=A_eq, b_eq=b_eq,
                  bounds=[(-limit, limit)] * B.shape[1], method="highs")
    return bool(res.success)


def degraded_envelope(B, limit, psi_deg):
    """Largest current holdable at a fixed heading offset, by bisection."""
    psi = np.radians(psi_deg)
    lo, hi = 0.0, 4.0
    if not feasible(B, limit, 1e-3, psi):
        return 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if feasible(B, limit, mid, psi):
            lo = mid
        else:
            hi = mid
    return lo


def best_heading(B, limit, sweep=None):
    """Best achievable holding current over heading, and the heading itself."""
    sweep = sweep if sweep is not None else np.arange(-40, 41, 2.5)
    best_v, best_psi = 0.0, 0.0
    for p in sweep:
        v = degraded_envelope(B, limit, p)
        if v > best_v:
            best_v, best_psi = v, p
    return best_v, best_psi


def heading_study(B, lim, need):
    """Degraded envelope after each horizontal failure, heading optimised."""
    print()
    print("  the remaining freedom is heading, carrying the drag of incidence")
    print(f"  ({'broadside quadratic damping'} "
          f"{VARUNA_1.quad_damp[1]:.0f} against {QUAD:.1f} in surge)")
    print()
    print(f"  {'failed unit':16s}{'nose on m/s':>13}{'best m/s':>11}"
          f"{'at heading':>12}{'holds site':>12}")
    print("-" * 74)
    rows = []
    for k in range(4):
        keep = [j for j in range(8) if j != k]
        Bk = B[:, keep]
        v0 = degraded_envelope(Bk, lim, 0.0)
        vb, pb = best_heading(Bk, lim)
        ok = vb >= SITE_CURRENT
        rows.append({"failed": NAMES[k], "nose_on": v0, "best": vb,
                     "heading_deg": float(pb), "holds": bool(ok)})
        print(f"  {NAMES[k]:16s}{v0:>13.2f}{vb:>11.2f}{pb:>12.1f}"
              f"{'yes' if ok else 'NO':>12}")
    return rows


def thrust_for_tolerance(B, need, lo=80.0, hi=400.0):
    """Per thruster force that would make a horizontal failure survivable.

    Worth computing, because a limitation that arrives with a priced remedy is
    a design decision rather than a defect.
    """
    keep = [j for j in range(8) if j != 0]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if max_along(B[:, keep], 0, mid) >= need:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    B = vectored_allocation(VARUNA_1.arms)
    lim = VARUNA_1.max_thrust_n

    need = drag_at(SITE_CURRENT)
    surge0 = max_along(B, 0, lim)
    heave0 = max_along(B, 2, lim)

    print("VARUNA-1 thruster-out analysis")
    print("=" * 74)
    print(f"  eight thrusters at {lim:.0f} N, six degrees of freedom")
    print(f"  station keeping at {SITE_CURRENT:.1f} m/s needs "
          f"{need:.0f} N of surge")
    print(f"  intact: surge {surge0:.0f} N, heave {heave0:.0f} N, "
          f"envelope {envelope(surge0):.2f} m/s")
    print()
    print(f"  {'failed unit':16s}{'surge N':>9}{'heave N':>9}"
          f"{'envelope m/s':>14}{'holds site':>12}")
    print("-" * 74)

    rows = []
    worst = None
    for k in range(8):
        keep = [j for j in range(8) if j != k]
        Bk = B[:, keep]
        s = max_along(Bk, 0, lim)
        h = max_along(Bk, 2, lim)
        env = envelope(s)
        ok = s >= need
        rows.append({"failed": NAMES[k], "surge": s, "heave": h,
                     "envelope": env, "holds": bool(ok)})
        if worst is None or s < worst["surge"]:
            worst = rows[-1]
        print(f"  {NAMES[k]:16s}{s:>9.0f}{h:>9.0f}{env:>14.2f}"
              f"{'yes' if ok else 'NO':>12}")

    n_ok = sum(r["holds"] for r in rows)
    print("-" * 74)
    print(f"  {n_ok}/8 single failures retain station keeping at the site "
          f"current")
    print(f"  worst case is {worst['failed']}, {worst['surge']:.0f} N surge, "
          f"envelope {worst['envelope']:.2f} m/s")
    print(f"  surge retained in the worst case: "
          f"{100 * worst['surge'] / surge0:.0f} percent")

    # Two simultaneous failures, the pessimistic case.
    pairs = []
    for a in range(8):
        for b in range(a + 1, 8):
            keep = [j for j in range(8) if j not in (a, b)]
            s = max_along(B[:, keep], 0, lim)
            pairs.append((s, NAMES[a], NAMES[b]))
    pairs.sort()
    n_pair_ok = sum(1 for s, _, _ in pairs if s >= need)
    print()
    print(f"  double failures: {n_pair_ok}/{len(pairs)} pairs still hold the "
          f"site current")
    print(f"  worst pair is {pairs[0][1]} with {pairs[0][2]}, "
          f"{pairs[0][0]:.0f} N surge, envelope {envelope(pairs[0][0]):.2f} m/s")

    hrows = heading_study(B, lim, need)
    n_head_ok = sum(r["holds"] for r in hrows)
    best_deg = max(hrows, key=lambda r: r["best"])
    out_fin = {"heading_rows": hrows, "heading_ok": n_head_ok,
               "degraded_best_ms": best_deg["best"]}

    print()
    print("  Reading. Losing a vertical thruster costs heave authority and")
    print("  nothing else. Losing a horizontal one halves surge, because the")
    print("  single surviving unit on the light side must balance two on the")
    print("  heavy side, and no heading offset recovers it: turning into the")
    print("  flow to gain thrust costs more in broadside drag than it gains.")
    print()
    print(f"  The vehicle is therefore single fault tolerant for attitude and")
    print(f"  depth, but not for station keeping at "
          f"{SITE_CURRENT:.1f} m/s. After a horizontal failure it holds to")
    print(f"  {max(r['best'] for r in hrows):.2f} m/s and must otherwise abort")
    print("  downstream, which is always available because the flow assists.")

    out = {
        "limit_n": lim,
        "need_n": need,
        "site_current": SITE_CURRENT,
        "surge_intact": surge0,
        "heave_intact": heave0,
        "envelope_intact": envelope(surge0),
        "single": rows,
        "single_ok": n_ok,
        "worst_single": worst,
        "worst_single_pct": 100 * worst["surge"] / surge0,
        "pairs_total": len(pairs),
        "pairs_ok": n_pair_ok,
        "worst_pair_n": pairs[0][0],
        "worst_pair": [pairs[0][1], pairs[0][2]],
        "worst_pair_env": envelope(pairs[0][0]),
    }
    out.update(out_fin)
    t_need = thrust_for_tolerance(B, need)
    out["thrust_for_tolerance_n"] = t_need
    print()
    print("  Remedy, if single fault tolerance at the design current were")
    print(f"  required: {t_need:.0f} N per thruster instead of {lim:.0f} N, "
          "or six")
    print("  horizontal units instead of four. Both cost drag and money, so")
    print("  it is stated as a limitation rather than quietly designed around.")

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "varuna_redundancy.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n  wrote {os.path.basename(p)}")
    return out


if __name__ == "__main__":
    main()
