"""Run the reference mission end to end, report it, and persist the result."""
import argparse
import json
import os
import time

import numpy as np

from varuna.scene import DisasterSite
from varuna.mission import MissionRunner, MissionConfig
from varuna.dynamics import BLUEROV2_HEAVY, VARUNA_1, max_holdable_current
from varuna.control import VARUNA_GAINS, ControlGains

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--vehicle", default="varuna", choices=["varuna", "bluerov2"])
ap.add_argument("--tag", default=None)
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()

params = VARUNA_1 if args.vehicle == "varuna" else BLUEROV2_HEAVY
gains = VARUNA_GAINS if args.vehicle == "varuna" else ControlGains()
tag = args.tag or f"{args.vehicle}_s{args.seed}"

site = DisasterSite()
cfg = MissionConfig()
t0 = time.perf_counter()
mr = MissionRunner(site, cfg, params=params, gains=gains, seed=args.seed)
res = mr.run()
wall = time.perf_counter() - t0

if not args.quiet:
    print("=" * 66)
    print(f"MISSION RESULT  [{params.name}]  seed {args.seed}")
    print("=" * 66)
    print(f"  simulated duration    : {res['duration']:.1f} s "
          f"({res['duration']/60:.1f} min)   wall clock {wall:.1f} s")
    print(f"  phases reached        : {' -> '.join(res['phases_reached'])}")
    print(f"  path length           : {res['path_length']:.1f} m")
    print(f"  sonar pings           : {res['pings']}")
    print(f"  soundings mapped      : {res['soundings']}")
    print(f"  search-box coverage   : {res['coverage']*100:.1f} %")
    print(f"  DVL availability      : {res['dvl_availability']*100:.1f} %")
    print(f"  nav error mean/max/end: {res['nav_error_mean']:.2f} / "
          f"{res['nav_error_max']:.2f} / {res['nav_error_final']:.2f} m")
    print(f"  reported horiz sigma  : {res['sigma_final']:.2f} m")
    print()
    print("  SCOUR (mapped vs ground truth)")
    for k, v in res["scour"].items():
        gt = site.scour_truth.get(k)
        if v is None:
            print(f"    {k}: not measurable")
            continue
        print(f"    {k}: depth {v['max_depth']:5.2f} m (truth {gt['depth']:.2f})   "
              f"volume {v['volume']:7.1f} m3 (truth {gt['volume']:.1f})   "
              f"cover {v['coverage']*100:.0f} %")
    print()
    print("  EVENTS")
    for e in res["events"]:
        print(f"    [{e['t']:7.1f}s] {e['msg']}")

# ---------------------------------------------------------------- persist
os.makedirs(f"{OUT}/logs", exist_ok=True)
L = res["log"]
bm = res["map"]
np.savez_compressed(
    f"{OUT}/logs/mission_{tag}.npz",
    t=np.array(L["t"]), eta=np.array(L["eta"]), nu=np.array(L["nu"]),
    est_pos=np.array(L["est_pos"]), est_att=np.array(L["est_att"]),
    sigma=np.array(L["sigma"]), alt=np.array(L["alt"]),
    current=np.array(L["current"]), target=np.array(L["target_pos"]),
    thrust=np.array(L["thrust"]), thrust_vec=np.array(L["thrust_vec"]),
    phase=np.array(L["phase"]),
    dvl_lock=np.array(L["dvl_lock"]),
    map_sum=bm.sum, map_count=bm.count, map_x0=bm.x0, map_y0=bm.y0, map_res=bm.res,
    truth_H=site.H, truth_xs=site.xs, truth_ys=site.ys,
    ping_t=np.array([f["t"] for f in mr.frames]),
    ping_pose=np.array([f["pose"] for f in mr.frames]),
)

summary = {k: res[k] for k in
           ("duration", "path_length", "nav_error_mean", "nav_error_final",
            "nav_error_max", "sigma_final", "dvl_availability", "coverage",
            "soundings", "pings", "phases_reached")}
summary["vehicle"] = params.name
summary["seed"] = args.seed
summary["wall_clock_s"] = wall
summary["max_holdable_current"] = max_holdable_current(params)
summary["site_surface_current"] = site.cfg.surface_current
summary["ssc_g_per_l"] = site.cfg.ssc_g_per_l
summary["scour"] = {
    k: (None if v is None else
        {**{kk: float(vv) for kk, vv in v.items()},
         "truth_depth": site.scour_truth[k]["depth"],
         "truth_volume": site.scour_truth[k]["volume"]})
    for k, v in res["scour"].items()}
summary["events"] = res["events"]
summary["tracks"] = [{"position": list(map(float, t["position"])),
                      "label": t["label"], "hits": int(t["hits"]),
                      "confidence": float(t["confidence"])}
                     for t in res["tracks"]]
with open(f"{OUT}/logs/mission_{tag}.json", "w") as fh:
    json.dump(summary, fh, indent=1)

if not args.quiet:
    print()
    print(f"  saved: results/logs/mission_{tag}.npz / .json")
