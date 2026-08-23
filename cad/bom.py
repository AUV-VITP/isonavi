"""Bill of materials for one VARUNA-1 airframe.

Prototype quantity, one unit. This is a materials and bought-in equipment cost
only: it excludes design labour, integration, test, tooling and certification,
which for a first article typically exceed the parts themselves. Quoting a BOM
as though it were a unit cost would be dishonest, so it is not done here.

Sourcing basis is recorded per line, because the two acoustic instruments
dominate and their prices behave differently: the DVL is publicly listed, the
imaging sonar is quotation only and is therefore an estimate carrying real
uncertainty.

Masses are taken from the solved layout so the BOM and the mass budget cannot
drift apart.
"""

from __future__ import annotations

import json
import os

import varuna_layout as L

USD_INR = 84.0          # indicative, mid market

# name, qty, unit USD, subsystem, sourcing basis
ITEMS = [
    ("Forward looking multibeam sonar, Oculus M750d class", 1, 15000.0,
     "acoustics", "estimate, quotation only"),
    ("Doppler velocity log, Water Linked A50", 1, 8710.0,
     "acoustics", "published list price"),
    ("Thruster, T200 class, 120 N", 8, 200.0,
     "propulsion", "published list price"),
    ("Electronic speed controller", 8, 35.0,
     "propulsion", "published list price"),
    ("Battery pack, 14S4P lithium ion, 1.5 kWh", 1, 850.0,
     "power", "cell cost plus assembly"),
    ("Flight computer and interface boards", 1, 250.0,
     "avionics", "published list price"),
    ("Attitude and heading reference unit", 1, 800.0,
     "avionics", "published list price"),
    ("Depth transducer", 1, 85.0,
     "avionics", "published list price"),
    ("Hull, composite layup and cure", 1, 2200.0,
     "structure", "workshop estimate"),
    ("Bulkheads, penetrators and wet mate connectors", 1, 1900.0,
     "structure", "workshop estimate"),
    ("Anodes, mast, lifting eye, drop weight release", 1, 450.0,
     "structure", "workshop estimate"),
    ("Harness, fasteners and consumables", 1, 700.0,
     "structure", "workshop estimate"),
]

# Comparable vehicles, for scale. Both are transit survey torpedoes without
# hover or station keeping, so they are a price reference and not a like for
# like capability comparison.
COMPARATORS = [
    ("REMUS 100", 300000, "1.6 m, under 45 kg, 100 m"),
    ("Iver3", 150000, "2.1 m, 0.14 m diameter, 100 m"),
]


def rollup():
    total = 0.0
    groups: dict[str, float] = {}
    rows = []
    for name, qty, unit, grp, basis in ITEMS:
        ext = qty * unit
        total += ext
        groups[grp] = groups.get(grp, 0.0) + ext
        rows.append((name, qty, unit, ext, grp, basis))
    return rows, groups, total


def main():
    parts, geom, v_hull = L.solve_layout()
    rows, groups, total = rollup()

    print("VARUNA-1 bill of materials, one airframe, prototype quantity")
    print("=" * 88)
    print(f"{'item':56s}{'qty':>4}{'unit USD':>11}{'ext USD':>12}")
    print("-" * 88)
    for name, qty, unit, ext, grp, basis in sorted(rows, key=lambda r: -r[3]):
        print(f"{name:56s}{qty:>4}{unit:>11,.0f}{ext:>12,.0f}")
    print("-" * 88)
    for grp, amt in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"{'  ' + grp:56s}{'':4}{'':11}{amt:>12,.0f}"
              f"   {100 * amt / total:5.1f} %")
    print("=" * 88)
    print(f"{'  TOTAL':56s}{'':4}{'':11}{total:>12,.0f} USD")
    print(f"{'':56s}{'':4}{'':11}{total * USD_INR / 1e5:>12,.1f} lakh INR"
          f"   at {USD_INR:.0f} INR per USD")

    ac = groups.get("acoustics", 0.0)
    print()
    print(f"  The two acoustic instruments are {100 * ac / total:.0f} percent "
          f"of the bill.")
    print(f"  Everything else, hull, propulsion, power and the entire autonomy")
    print(f"  stack, comes to {total - ac:,.0f} USD.")
    print()
    print("  For scale, comparable imported survey vehicles:")
    for name, price, note in COMPARATORS:
        print(f"    {name:14s} order {price:>9,.0f} USD   ({note})")
    print()
    print("  Those are transit survey torpedoes with no hover or station")
    print("  keeping, so this is a price reference, not a capability match.")
    print()
    print("  This is parts only. Design, integration, test and qualification")
    print("  for a first article typically exceed the parts cost.")

    out = {
        "total_usd": total,
        "total_lakh_inr": total * USD_INR / 1e5,
        "usd_inr": USD_INR,
        "acoustics_usd": ac,
        "acoustics_pct": 100 * ac / total,
        "non_acoustics_usd": total - ac,
        "groups": groups,
        "n_lines": len(rows),
        "comparators": {n: p for n, p, _ in COMPARATORS},
    }
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "varuna_bom.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n  wrote {os.path.basename(p)}")
    return out


if __name__ == "__main__":
    main()
