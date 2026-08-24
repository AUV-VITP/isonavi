"""Programme budget for isonavi, line by line, with the reason for each line.

Four things are costed separately because they behave differently and a funder
needs to see them apart:

  A  airframe        the parts that fly, one vehicle
  B  capital         equipment bought once and used for every build after
  C  materials       stock and consumables, per airframe plus workshop
  D  validation      the measurements that replace the closed form estimates

Every line carries a sourcing basis, because the confidence is not uniform:

  published   a list price that can be checked today
  quotation   supplier quotes only, so an estimate carrying real uncertainty
  market      a commodity rate, varies with supplier and quantity
  workshop    a fabrication estimate from the drawing
  service     a facility or contractor day rate

Masses come from the solved layout so the budget and the mass budget cannot
disagree about what the vehicle is made of.
"""

from __future__ import annotations

import json
import os

import isonavi_layout as L

# Mid-market open on 24 August 2026 (Moneycontrol / USD-INR spot ~95.64).
USD_INR = 95.64
CONTINGENCY = 0.12      # on the programme, not on the airframe


def inr(usd):
    """Convert a USD line to whole rupees."""
    return int(round(usd * USD_INR))


def inr_fmt(usd):
    """Whole rupees with thousands separators, for tables and macros."""
    return f"{inr(usd):,}"

# category, group, item, qty, unit USD, basis, why the line exists
LINES = [
    # ---------------------------------------------------------- A airframe
    ("A", "acoustics", "Forward looking multibeam sonar, Oculus M750d class",
     1, 15000.0, "quotation",
     "The only sensor that returns anything at 3.2 g/L of suspended sediment. "
     "Its 130 degree by 20 degree fan at 750 kHz is what sets the swath the "
     "mapper integrates, so the survey rate and the whole mission plan follow "
     "from this one choice."),
    ("A", "acoustics", "Doppler velocity log, Water Linked A50",
     1, 8710.0, "published",
     "Bottom lock velocity is the only measurement that bounds dead reckoning "
     "drift with no GPS. Without it position error grows without limit and "
     "the 0.32 m navigation result does not exist."),
    ("A", "sensors", "Attitude and heading reference unit, Xsens MTi-3",
     1, 425.0, "published",
     "Supplies the attitude and gyro bias states of the twelve state filter. "
     "Chosen at the small industrial grade rather than tactical grade because "
     "the DVL, not the gyro, is what bounds the solution here."),
    ("A", "sensors", "Depth transducer, 30 bar",
     1, 85.0, "published",
     "The only directly observed, drift free axis on the vehicle. It anchors "
     "depth absolutely while horizontal position is dead reckoned."),
    ("A", "propulsion", "Thruster, T500 class, 158 N at 24 V",
     8, 790.0, "published",
     "Eight vectored units give six degree of freedom control with two spare. "
     "The 120 N saturation the allocator assumes needs a unit rated above it: "
     "a 51 N T200 class thruster cannot hold the design current and was the "
     "wrong part for this vehicle."),
    ("A", "propulsion", "Electronic speed controller, 50 A",
     8, 121.0, "published",
     "Rated for the T500 peak draw inside a sealed hull with no airflow. A "
     "30 A controller sized for the smaller thruster would run hot at the "
     "duty this mission demands."),
    ("A", "propulsion", "Thruster cable runs and pressure penetrators",
     8, 28.0, "published",
     "Every thruster lead crosses the pressure boundary, so each one is a "
     "potential flood path and is fitted with a compression gland rated past "
     "the depth."),
    ("A", "power", "Lithium ion cells, 21700, 5.0 Ah, 14S6P",
     84, 6.00, "published",
     "84 cells give 1478 Wh, which is the 1.5 kWh the endurance model spends. "
     "At 17.6 Wh and 69 g a cell this is 5.80 kg of cells; the pack cannot be "
     "lighter without breaking the energy density of the chemistry."),
    ("A", "power", "Battery management system, 14S, 60 A",
     1, 140.0, "published",
     "Cell balancing and over discharge cut off on the largest stored energy "
     "hazard aboard. A pack this size is a fire risk if any cell is allowed "
     "to invert."),
    ("A", "power", "Pack assembly, interconnects, holders and potting",
     1, 130.0, "market",
     "Nickel strip, cell holders, barrier paper and potting. This is the "
     "difference between a box of cells and a pack that survives shock and "
     "vibration."),
    ("A", "power", "Main contactor, fusing and service disconnect",
     1, 95.0, "published",
     "A way to make 1.5 kWh safe before anyone opens the hull, and a fault "
     "current path that opens rather than melts."),
    ("A", "avionics", "Flight computer, RISC-V single board",
     1, 25.0, "published",
     "The board the autonomy was actually validated on, running the filter, "
     "the controller and the mission state machine inside a 50 ms budget."),
    ("A", "avionics", "Actuator microcontroller",
     1, 10.0, "published",
     "Generates the eight hardware PWM channels and reports the duty "
     "registers back, so the allocation is verified at the pin rather than "
     "assumed."),
    ("A", "avionics", "Carrier and interface board, four layer, assembled",
     1, 190.0, "quotation",
     "Fans the flight computer out to eight thruster channels, the CRC "
     "checked serial link, the sensor buses and the power rails. Prototype "
     "quantity, so the assembly dominates the board cost."),
    ("A", "avionics", "DC-DC converters, 24 V to 12 V and 5 V",
     1, 85.0, "published",
     "The pack runs at 50 V nominal and nothing else aboard wants that. "
     "Isolated rails also stop thruster switching noise reaching the filter."),
    ("A", "avionics", "Leak detection probes",
     4, 18.0, "published",
     "Four probes at the low points. A flood detected early is an abort and "
     "a recovered vehicle; detected late it is neither."),
    ("A", "structure", "Hull, carbon epoxy layup and cure",
     1, 2200.0, "workshop",
     "The faired hull is the pressure boundary and the drag surface at once, "
     "which is what buys the 2.69 m/s envelope. Cost is the mandrel, the "
     "layup and the cure cycle."),
    ("A", "structure", "Ring frames, machined",
     3, 110.0, "workshop",
     "Three frames at the mid body quarter points. They shorten the "
     "unsupported length and move collapse from 66 m to 482 m, which is what "
     "makes a 4 mm skin viable at all."),
    ("A", "structure", "Bulkheads and aft closure plate, machined aluminium",
     1, 640.0, "workshop",
     "Carries the penetrator field and takes the full end load at depth."),
    ("A", "structure", "Wet mateable connector set",
     1, 1260.0, "published",
     "Lets the payload and thruster looms be broken on a boat deck without "
     "opening the pressure vessel in the field."),
    ("A", "structure", "Acoustic window, cast polyurethane",
     1, 310.0, "workshop",
     "Acoustically matched to water so the sonar sees through the hull "
     "instead of into it."),
    ("A", "structure", "Thruster pylons, machined",
     8, 95.0, "workshop",
     "Set the moment arms the allocator depends on. Sized by stiffness and "
     "by flow rather than by strength, at a factor of 24 on root bending."),
    ("A", "structure", "Stabiliser fins, laid up",
     4, 70.0, "workshop",
     "Not a refinement. A faired hull is directionally unstable under its own "
     "Munk moment, and the fins supply roughly sixty times the restoring "
     "moment the buoyancy offset does at survey speed."),
    ("A", "structure", "Clamp band joints and seal sets",
     2, 145.0, "published",
     "The hull has to open for battery and payload access and close again to "
     "50 m, repeatedly."),
    ("A", "structure", "Sacrificial anodes",
     2, 28.0, "published",
     "Aluminium and stainless in salt or brackish water is a galvanic cell. "
     "Anodes are consumed instead of the structure."),
    ("A", "structure", "Lifting eye, antenna mast and strobe",
     1, 290.0, "published",
     "Recovery in a current needs a lifting point over the centre of gravity "
     "and something visible above the surface."),
    ("A", "structure", "Drop weight release, burn wire",
     1, 180.0, "workshop",
     "Takes the vehicle from 1.96 N positive to 19.85 N positive on command, "
     "which is the recovery path from entanglement or a flat battery."),
    ("A", "structure", "Trim ballast, lead",
     4, 6.0, "market",
     "3.15 kg solved, not chosen, so that displaced volume, dry mass and "
     "level trim all close at once. Priced to the next whole kilogram."),
    ("A", "structure", "Fasteners, harness and assembly consumables",
     1, 420.0, "market",
     "Marine grade fasteners, looms, glands and the small parts that do not "
     "appear on a drawing but do appear on an invoice."),

    # --------------------------------------------------------- B capital
    ("B", "fabrication", "FDM 3D printer, 256 mm class, enclosed",
     1, 1199.0, "published",
     "Prints the internal chassis, sensor mounts, cable guides, layup jigs "
     "and the formers the hull is laid up over. Enclosed and heated because "
     "the structural parts are printed in carbon filled nylon, not PLA."),
    ("B", "fabrication", "MSLA resin printer and wash and cure station",
     1, 400.0, "published",
     "Fine feature parts the filament printer cannot hold tolerance on: "
     "connector shells, sonar mount shims and O-ring test coupons."),
    ("B", "fabrication", "Vacuum pump, bagging kit and consumable set",
     1, 620.0, "published",
     "A wet layup cured without vacuum keeps its voids, and voids are where "
     "a pressure hull fails. This is the difference between a part that holds "
     "50 m and one that looks like it should."),
    ("B", "fabrication", "Composite curing oven, programmable",
     1, 900.0, "published",
     "The epoxy needs a controlled post cure to reach its rated glass "
     "transition. Curing on a bench gives a hull with an unknown strength."),
    ("B", "fabrication", "Machining fixtures and soft jaws",
     1, 300.0, "workshop",
     "Holds the frames and closure plate concentric while they are cut. "
     "Machining is bought in; the fixtures are ours and are reused."),
    ("B", "electronics", "Bench power supply, 0 to 60 V, 20 A",
     1, 420.0, "published",
     "Runs the vehicle at pack voltage on the bench without discharging the "
     "pack, and current limits a fault instead of venting a cell."),
    ("B", "electronics", "Oscilloscope, four channel, 100 MHz",
     1, 680.0, "published",
     "The actuator interface claim is that the pulse widths at the pins match "
     "the allocation. That is measured on a scope or it is an assertion."),
    ("B", "electronics", "Soldering and rework station, hot air",
     1, 260.0, "published",
     "Prototype boards get reworked. Fine pitch parts need controlled heat."),
    ("B", "electronics", "Digital multimeters",
     2, 90.0, "published",
     "Two, because insulation resistance and continuity checks on a sealed "
     "hull are two hands and two instruments."),
    ("B", "test", "Hydrostatic pressure test vessel, 60 bar",
     1, 3200.0, "quotation",
     "The 50 m rating is a closed form buckling calculation with a factor of "
     "9.6. A pressure vessel is the only thing that turns that into a "
     "qualified depth rather than a prediction."),
    ("B", "test", "Ballast, trim and lifting rig",
     1, 450.0, "workshop",
     "Weigh, trim and float the vehicle repeatably. The mass budget closes to "
     "the milligram on paper; this is what checks it in water."),
    ("B", "compute", "Workstation for CFD and network training",
     1, 2400.0, "published",
     "The tow tank campaign is validated against CFD, and the perception "
     "models are retrained as real sonar arrives. Both need a machine with a "
     "GPU and real memory."),
    ("B", "safety", "Charging bunker, extinguisher and personal protection",
     1, 380.0, "published",
     "A 1.5 kWh lithium pack is charged in the same room people work in. "
     "This is the cheapest line here and the one whose absence ends the "
     "programme."),
    ("B", "tooling", "Hand tools, torque wrenches and workshop set",
     1, 540.0, "market",
     "Sealed joints are torque controlled or they are not sealed. Calibrated "
     "torque tools rather than judgement."),

    # ------------------------------------------------------- C materials
    ("C", "composites", "Carbon fibre twill, 200 gsm",
     12, 24.0, "market",
     "Hull, fins and pylon skins, plus coupons for the pressure test and "
     "enough spare for one full remake after the first article teaches us "
     "something."),
    ("C", "composites", "Epoxy laminating resin and hardener",
     6, 28.0, "market",
     "Roughly one part resin to one part fabric by weight, plus ten percent "
     "for losses, which is what a wet layup actually consumes."),
    ("C", "composites", "Peel ply, breather, release film and bagging film",
     1, 210.0, "market",
     "Consumed entirely on every cure. Not reusable, and a bag that leaks "
     "wastes the layup under it."),
    ("C", "stock", "Aluminium 6082 bar and plate stock",
     1, 340.0, "market",
     "Frames, bulkheads, closure plate and pylons are cut from this."),
    ("C", "stock", "Acetal and polyurethane stock",
     1, 120.0, "market",
     "Insulating standoffs, the acoustic window pour and bearing faces."),
    ("C", "stock", "O-ring stock and marine grease",
     1, 95.0, "market",
     "Every seal is remade every time the hull is opened during integration, "
     "and a reused O-ring is the usual cause of a flooded first dive."),
    ("C", "stock", "Marine potting and encapsulation compound",
     1, 140.0, "market",
     "Seals the penetrator field and the pack. Chosen for water absorption "
     "rather than for cost."),
    ("C", "electrical", "Cable, heatshrink, lugs and connectors",
     1, 260.0, "market",
     "Tinned marine cable throughout, because untinned copper in a humid "
     "hull corrodes at the terminations first."),
    ("C", "workshop", "Abrasives, solvents, gloves and shop consumables",
     1, 180.0, "market",
     "Surface preparation is most of the labour in a bonded assembly and all "
     "of the reason bonds fail."),

    # ------------------------------------------------------ D validation
    ("D", "hydrodynamics", "Tow tank campaign, five days",
     1, 4200.0, "service",
     "Replaces the closed form drag and added mass coefficients, which agree "
     "with the CAD derived values to 21 percent and are the least certain "
     "numbers in the report. The station keeping margin of 1.12 rests "
     "directly on them."),
    ("D", "structures", "Hydrostatic proof test, witnessed",
     1, 1100.0, "service",
     "An independent witness on the depth rating, so the qualification is "
     "not self certified."),
    ("D", "field", "Tank and reservoir trials, ten days",
     1, 2600.0, "service",
     "First wet running of the integrated vehicle: buoyancy, trim, thruster "
     "authority and the autonomy closing its own loop against real water."),
    ("D", "field", "Instrumented river trial, boat and crew, six days",
     1, 3900.0, "service",
     "The only way to see the vehicle in the current and turbidity it was "
     "designed for. Everything upstream of this is a prediction about that "
     "day."),
    ("D", "field", "Diver survey for scour ground truth",
     1, 1800.0, "service",
     "The scour measurement is the statutory product. It is only validated "
     "against an independent survey of the same pier."),
    ("D", "calibration", "Doppler log and inertial unit calibration",
     1, 600.0, "service",
     "A navigation claim rests on the sensors being what their datasheets "
     "say. Calibrated once, against a reference."),
]

COMPARATORS = [
    ("REMUS 100", 300000, "1.6 m, under 45 kg, 100 m"),
    ("Iver3", 150000, "2.1 m, 0.14 m diameter, 100 m"),
]

CAT_NAME = {
    "A": "Airframe, one vehicle",
    "B": "Capital equipment and tooling, bought once",
    "C": "Raw materials and consumables",
    "D": "Validation and field trials",
}


def tex_escape(s):
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def rollup():
    rows, cats, groups = [], {}, {}
    for cat, grp, name, qty, unit, basis, why in LINES:
        ext = qty * unit
        rows.append(dict(cat=cat, grp=grp, name=name, qty=qty, unit=unit,
                         ext=ext, basis=basis, why=why))
        cats[cat] = cats.get(cat, 0.0) + ext
        if cat == "A":
            groups[grp] = groups.get(grp, 0.0) + ext
    return rows, cats, groups


def write_tex(rows, cats, groups, air, prog, cont, docs, parts):
    rate_note = (
        f"Amounts in Indian rupees at "
        f"\\rupeesign\\,{USD_INR:.2f} per USD "
        f"(mid-market open, 24 August 2026). Sourced prices are USD "
        f"list or quotation figures converted at that rate."
    )
    out = [
        "% Generated by cad/bom.py. Do not edit: rebuild with docs/build.sh.",
        "",
    ]
    for cat in ("A", "B", "C", "D"):
        sel = [r for r in rows if r["cat"] == cat]
        out += [
            r"\begin{small}",
            r"\begin{longtable}{@{}p{0.26\textwidth}rrr"
            r"p{0.36\textwidth}@{}}",
            r"\caption{" + tex_escape(CAT_NAME[cat]) +
            r". Every line carries the reason it exists and the basis its "
            r"price rests on. " + rate_note +
            r"}\label{tab:budget" + cat + r"}\\",
            r"\toprule",
            r"\textbf{Item} & \textbf{Qty} & \textbf{Unit} & "
            r"\textbf{Ext} & \textbf{Why this line exists} \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"\textbf{Item} & \textbf{Qty} & \textbf{Unit} & "
            r"\textbf{Ext} & \textbf{Why this line exists} \\",
            r"\midrule",
            r"\endhead",
            r"\bottomrule",
            r"\endfoot",
        ]
        for r in sel:
            out.append(
                f"{tex_escape(r['name'])} & {r['qty']} & "
                f"\\INR{{{inr_fmt(r['unit'])}}} & "
                f"\\INR{{{inr_fmt(r['ext'])}}} & "
                f"{tex_escape(r['why'])} \\emph{{({r['basis']})}} \\\\")
        out.append(r"\addlinespace")
        out.append(
            r"\textbf{Subtotal} & & & \textbf{\INR{"
            + inr_fmt(cats[cat]) + r"}} & \\")
        out += [r"\end{longtable}", r"\end{small}", ""]

    out += [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Programme budget by category. The airframe is one vehicle; "
        r"the capital line is bought once and serves every build after it. "
        + rate_note + r"}\label{tab:budgetroll}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"\textbf{Category} & \textbf{Amount} & \textbf{Share} \\",
        r"\midrule",
    ]
    for cat in ("A", "B", "C", "D"):
        out.append(
            f"{tex_escape(CAT_NAME[cat])} & \\INR{{{inr_fmt(cats[cat])}}} & "
            f"{100 * cats[cat] / prog:.0f} \\% \\\\")
    out += [
        r"\midrule",
        f"Contingency at {CONTINGENCY * 100:.0f} percent & "
        f"\\INR{{{inr_fmt(cont)}}} & \\\\",
        r"\midrule",
        r"\textbf{Programme total} & \textbf{\INR{" + inr_fmt(prog) +
        r"}} & \\",
        r"\multicolumn{3}{l}{" + f"{prog * USD_INR / 1e5:.1f}" +
        r"~lakh (\INR{" + inr_fmt(prog) + r"} at "
        r"\rupeesign\," + f"{USD_INR:.2f}" + r" per USD)} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    p = os.path.join(docs, "budget_tables.tex")
    open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")

    # The airframe rolled up by subsystem, for the summary table in the text.
    sub = [
        "% Generated by cad/bom.py. Do not edit.",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"\textbf{Subsystem} & \textbf{Amount} & \textbf{Share} \\",
        r"\midrule",
    ]
    label = {"acoustics": "Acoustics, sonar and doppler log",
             "structure": "Structure, hull and fittings",
             "propulsion": "Propulsion, eight thrusters and drives",
             "avionics": "Avionics and computing",
             "sensors": "Attitude and depth sensing",
             "power": "Power, cells, management and protection"}
    for grp, amt in sorted(groups.items(), key=lambda kv: -kv[1]):
        sub.append(
            f"{label.get(grp, grp)} & \\INR{{{inr_fmt(amt)}}} & "
            f"{100 * amt / air:.0f} \\% \\\\")
    sub += [
        r"\midrule",
        r"Airframe total & \textbf{\INR{" + inr_fmt(air) + r"}} & \\",
        r"\multicolumn{3}{l}{" + f"{air * USD_INR / 1e5:.1f}" +
        r"~lakh (\INR{" + inr_fmt(air) + r"} at "
        r"\rupeesign\," + f"{USD_INR:.2f}" + r" per USD)} \\",
        r"\bottomrule", r"\end{tabular}",
    ]
    # The mass budget table, straight from the layout and the solved
    # budget, group rows plus the footer that reports what the solve
    # closed. Generated so the drawing and the document cannot disagree
    # with the parts list about what the vehicle is made of.
    mg = {}
    for q in parts:
        mg[q.group] = mg.get(q.group, 0.0) + q.mass
    mtot = sum(mg.values())
    mrows = [
        "% Generated by cad/bom.py. Do not edit.",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"\textbf{Group} & \textbf{Mass (kg)} & \textbf{Share} \\",
        r"\midrule",
    ]
    _mlabel = {"trim": "Trim ballast and drop weight"}
    for g, m in sorted(mg.items(), key=lambda kv: -kv[1]):
        mrows.append(f"{_mlabel.get(g, g.capitalize())} & {m:.2f} & "
                     f"{100 * m / mtot:.1f} \\% \\\\")
    mrows += [
        r"\midrule",
        f"Total & {mtot:.1f} & \\\\",
        r"\midrule",
        r"\multicolumn{3}{l}{Displaced volume \SI{\cadVolume}"
        r"{\cubic\metre}, net buoyancy \SI{\cadNetBuoy}{\newton} "
        r"positive} \\",
        r"\multicolumn{3}{l}{Longitudinal trim \SI{0.000}{\milli\metre}, "
        r"hull volume closes to \SI{\cadVolErr}{\percent}} \\",
        r"\bottomrule", r"\end{tabular}",
    ]
    pm = os.path.join(docs, "mass_by_group.tex")
    open(pm, "w", encoding="utf-8").write("\n".join(mrows) + "\n")

    ps = os.path.join(docs, "budget_subsystem.tex")
    open(ps, "w", encoding="utf-8").write("\n".join(sub) + "\n")
    return p


def main():
    parts, geom, v_hull = L.solve_layout()
    rows, cats, groups = rollup()

    air = cats["A"]
    base = sum(cats.values())
    cont = base * CONTINGENCY
    prog = base + cont

    print("isonavi programme budget")
    print("=" * 92)
    for cat in ("A", "B", "C", "D"):
        print(f"\n{cat}  {CAT_NAME[cat]}")
        print("-" * 92)
        for r in [x for x in rows if x["cat"] == cat]:
            print(f"  {r['name']:52s}{r['qty']:>4}"
                  f"{r['unit']:>10,.0f}{r['ext']:>11,.0f}  {r['basis']}")
        print(f"  {'subtotal':52s}{'':4}{'':10}{cats[cat]:>11,.0f}")

    print("\n" + "=" * 92)
    print(f"  {'airframe, one vehicle':52s}{'':4}{'':10}{air:>11,.0f}")
    print(f"  {'programme before contingency':52s}{'':4}{'':10}"
          f"{base:>11,.0f}")
    print(f"  {'contingency at ' + str(int(CONTINGENCY * 100)) + ' percent':52s}"
          f"{'':4}{'':10}{cont:>11,.0f}")
    print(f"  {'PROGRAMME TOTAL':52s}{'':4}{'':10}{prog:>11,.0f} USD")
    print(f"  {'':52s}{'':4}{'':10}{inr(prog):>11,} INR"
          f"  ({prog * USD_INR / 1e5:.1f} lakh at {USD_INR:.2f}/USD)")

    ac = groups.get("acoustics", 0.0)
    print(f"\n  The two acoustic instruments are {100 * ac / air:.0f} percent "
          f"of the airframe.")
    print(f"  Everything else, hull, propulsion, power and the whole autonomy")
    print(f"  stack, comes to {inr_fmt(air - ac)} INR "
          f"({air - ac:,.0f} USD).")

    here = os.path.dirname(os.path.abspath(__file__))
    docs = os.path.join(os.path.dirname(here), "docs")
    tp = write_tex(rows, cats, groups, air, prog, cont, docs, parts)
    print(f"\n  wrote {os.path.relpath(tp, os.path.dirname(here))}")

    out = {
        "total_usd": air,
        "total_inr": inr(air),
        "total_lakh_inr": air * USD_INR / 1e5,
        "usd_inr": USD_INR,
        "acoustics_usd": ac,
        "acoustics_inr": inr(ac),
        "acoustics_pct": 100 * ac / air,
        "non_acoustics_usd": air - ac,
        "non_acoustics_inr": inr(air - ac),
        "groups": groups,
        "n_lines": len(rows),
        "comparators": {n: p for n, p, _ in COMPARATORS},
        "cat_usd": cats,
        "cat_inr": {k: inr(v) for k, v in cats.items()},
        "capital_usd": cats["B"],
        "capital_inr": inr(cats["B"]),
        "materials_usd": cats["C"],
        "materials_inr": inr(cats["C"]),
        "validation_usd": cats["D"],
        "validation_inr": inr(cats["D"]),
        "contingency_pct": CONTINGENCY * 100,
        "contingency_usd": cont,
        "contingency_inr": inr(cont),
        "programme_usd": prog,
        "programme_inr": inr(prog),
        "programme_lakh_inr": prog * USD_INR / 1e5,
    }
    p = os.path.join(here, "isonavi_bom.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"  wrote {os.path.basename(p)}")
    return out


if __name__ == "__main__":
    main()
