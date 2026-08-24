"""Energy and endurance budget for isonavi-1.

Endurance is the question a hovering vehicle answers worst, because holding
station in a current costs full thrust while doing no useful work. It is
therefore worth computing honestly rather than quoting a transit figure.

Propeller power is taken from momentum theory, P = T^(3/2) / sqrt(2 rho A),
divided by a drive efficiency covering motor, controller and propeller. That
relation is superlinear in thrust, so the fleet total cannot be recovered from
the peak thruster alone; the mission log therefore records all eight forces and
this integrates over them.

Two numbers come out: what the reference mission actually consumed, and how
long the vehicle can hold station as a function of the current it is fighting.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import isonavi_layout as L

RHO = 1000.0
BATTERY_WH = 1500.0        # 14S6P lithium ion, 84 cells, as in the bill of materials
USABLE = 0.80              # reserve held back, so 80 percent is available

DUCT_D = 0.100             # thruster disc diameter, m
DISC_A = math.pi * (DUCT_D / 2) ** 2
ETA = 0.60                 # motor, controller and propeller together

# Hotel load. The sonar dominates and is the reason a survey vehicle cannot
# simply be told to loiter cheaply.
HOTEL = {
    "forward looking sonar": 32.0,
    "doppler velocity log": 6.0,
    "flight computer and interface": 9.0,
    "inertial unit and depth": 3.0,
    "losses and standby": 5.0,
}



def trim_png(path, pad=16):
    """Crop a saved figure to its ink, leaving a small margin.

    Matplotlib's bbox_inches only tightens to the axes bounding box, which for
    a 3D axes is the full cubic frame whether or not anything is drawn in it.
    """
    try:
        from PIL import Image
    except ImportError:
        return
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ink = (a < 248).any(axis=2)
    if not ink.any():
        return
    ys, xs = np.where(ink)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, a.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, a.shape[1])
    im.crop((x0, y0, x1, y1)).save(path)


def prop_power(thrust_n):
    """Shaft power for one thruster at a given thrust, momentum theory."""
    t = np.abs(np.asarray(thrust_n, float))
    return t ** 1.5 / math.sqrt(2 * RHO * DISC_A) / ETA


def station_keeping_thrust(v):
    """Total thrust needed to hold station against a current of v."""
    from isonavi_layout import TARGET_MASS  # noqa: F401
    lin, quad = 22.0, 38.9      # surge damping, CAD derived quadratic term
    return lin * v + quad * v * v


def endurance_at(v, hotel_w):
    """Hours the vehicle can hold station against a current of v."""
    total_t = station_keeping_thrust(v)
    per = total_t / 4.0         # the four horizontal units carry surge
    p = 4 * prop_power(per) + hotel_w
    return BATTERY_WH * USABLE / p, p



def figure(out, log):
    """Endurance against current, and the power the reference mission drew."""
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 4.5), facecolor="white")

    vv = np.linspace(0.0, 2.75, 160)
    hotel = out["hotel_w"]
    hrs = np.array([endurance_at(v, hotel)[0] for v in vv])
    ax[0].plot(vv, hrs, color="#1f6feb", lw=2.0)
    ax[0].axvline(2.4, color="#d1242f", ls="--", lw=1.1)
    lab = ("site current 2.4 m/s" + "\n"
           f"{out['hours_at_site_current']:.1f} h")
    ax[0].annotate(lab, xy=(2.4, out["hours_at_site_current"]),
                   xytext=(1.42, 6.4), fontsize=9, color="#d1242f",
                   arrowprops=dict(arrowstyle="->", color="#d1242f",
                                   lw=0.9))
    ax[0].axvline(2.69, color="#6b7280", ls=":", lw=1.0)
    ax[0].annotate("thrust limit", xy=(2.69, 1.0), xytext=(2.2, 13.0),
                   fontsize=8.5, color="#6b7280",
                   arrowprops=dict(arrowstyle="->", color="#6b7280", lw=0.8))
    ax[0].set_yscale("log")
    ax[0].set_xlabel("current held against (m/s)")
    ax[0].set_ylabel("endurance (hours, log scale)")
    ax[0].set_title("Endurance is set by the current, not the survey")

    if os.path.exists(log):
        d = np.load(log, allow_pickle=True)
        t = d["t"]
        p = prop_power(d["thrust_vec"]).sum(axis=1) + hotel
        ax[1].plot(t, p, color="#1f6feb", lw=0.6, alpha=0.85)
        ax[1].axhline(p.mean(), color="#d1242f", ls="--", lw=1.1,
                      label=f"mean {p.mean():.0f} W")
        ax[1].set_xlabel("mission time (s)")
        ax[1].set_ylabel("total electrical draw (W)")
        ax[1].set_title(f"Reference mission drew {out['mission_wh']:.0f} Wh, "
                        f"{out['mission_pct']:.0f} percent of a charge")
        ax[1].legend(fontsize=8.5)

    plt.tight_layout()
    p_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "isonavi_energy.png")
    plt.savefig(p_out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    trim_png(p_out)
    print(f"  wrote {os.path.basename(p_out)}")


def main():
    log = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
        "simulation", "results", "logs", "mission_isonavi_s1.npz")
    hotel_w = sum(HOTEL.values())

    print("isonavi-1 energy and endurance")
    print("=" * 70)
    print(f"  battery {BATTERY_WH:.0f} Wh, {USABLE * 100:.0f} percent usable, "
          f"{BATTERY_WH * USABLE:.0f} Wh available")
    print()
    print("  hotel load")
    for k, v in sorted(HOTEL.items(), key=lambda kv: -kv[1]):
        print(f"    {k:34s}{v:7.1f} W")
    print(f"    {'total':34s}{hotel_w:7.1f} W")

    out = {"battery_wh": BATTERY_WH, "usable": USABLE, "hotel_w": hotel_w,
           "eta": ETA}

    if os.path.exists(log):
        d = np.load(log, allow_pickle=True)
        tv = d["thrust_vec"]
        t = d["t"]
        dt = float(t[1] - t[0])
        p_prop = prop_power(tv).sum(axis=1)
        p_tot = p_prop + hotel_w
        e_wh = float(p_tot.sum() * dt / 3600.0)
        dur = float(t[-1])
        print()
        print("  reference mission, integrated over the logged thruster forces")
        print(f"    duration                      {dur:7.0f} s")
        print(f"    mean propulsion power         {p_prop.mean():7.0f} W")
        print(f"    peak propulsion power         {p_prop.max():7.0f} W")
        print(f"    mean total draw               {p_tot.mean():7.0f} W")
        print(f"    energy consumed               {e_wh:7.1f} Wh"
              f"   ({100 * e_wh / (BATTERY_WH * USABLE):.1f} percent of usable)")
        print(f"    missions per charge           {BATTERY_WH * USABLE / e_wh:7.1f}")
        out.update(mission_s=dur, mission_wh=e_wh,
                   mission_pct=100 * e_wh / (BATTERY_WH * USABLE),
                   missions_per_charge=BATTERY_WH * USABLE / e_wh,
                   p_prop_mean=float(p_prop.mean()),
                   p_prop_peak=float(p_prop.max()),
                   p_total_mean=float(p_tot.mean()))

    print()
    print("  endurance holding station, by current")
    print(f"    {'current m/s':>13}{'thrust N':>11}{'draw W':>10}{'hours':>9}")
    curve = []
    for v in (0.0, 0.5, 1.0, 1.5, 2.0, 2.4, 2.69):
        h, p = endurance_at(v, hotel_w)
        tot = station_keeping_thrust(v)
        print(f"    {v:>13.2f}{tot:>11.0f}{p:>10.0f}{h:>9.2f}")
        curve.append({"v": v, "thrust": tot, "power": p, "hours": h})
    out["curve"] = curve

    h_site, p_site = endurance_at(2.4, hotel_w)
    h_calm, _ = endurance_at(0.4, hotel_w)
    out["hours_at_site_current"] = h_site
    out["hours_calm"] = h_calm
    print()
    print(f"  Holding against the design current of 2.4 m/s costs "
          f"{p_site:.0f} W and")
    print(f"  lasts {h_site:.1f} h. In slack water the same battery lasts "
          f"{h_calm:.1f} h.")
    print("  Endurance is set by the current, not by the survey, which is why")
    print("  the mission plan works the flow rather than ignoring it.")

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "isonavi_energy.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n  wrote {os.path.basename(p)}")
    figure(out, log)
    return out


if __name__ == "__main__":
    main()
