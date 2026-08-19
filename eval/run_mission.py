"""Run the reference mission end to end and report the outcome."""
import time
import numpy as np
from varuna.scene import DisasterSite
from varuna.mission import MissionRunner, MissionConfig

site = DisasterSite()
cfg = MissionConfig()
t0 = time.perf_counter()
mr = MissionRunner(site, cfg, seed=1)
res = mr.run()
wall = time.perf_counter() - t0

print("=" * 66)
print("MISSION RESULT")
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
