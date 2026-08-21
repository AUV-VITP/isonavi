"""Run the full host plant and flight computer against each other over
localhost TCP, with no real hardware. This validates the distributed loop
logic and the protocol end to end, and checks that the HIL result reproduces
the pure-simulation result closely enough to be called equivalent.

The flight computer here talks to no ESP32 (esp_dev=None), so the actuator
link is exercised separately on real hardware.
"""
import json
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))    # hil/
sys.path.insert(0, HERE)                             # for common.hil_protocol
sys.path.insert(0, os.path.join(HERE, "host"))
sys.path.insert(0, os.path.join(HERE, "board"))
sys.path.insert(0, os.path.join(HERE, "common"))

# Import both sides.
import host_plant
import flight_computer as fcmod

PORT = 5599
MAX_T = 1000.0

result_holder = {}


def run_host():
    result_holder["host"] = host_plant.serve(PORT, MAX_T, None, seed=0)


def run_board():
    fc = fcmod.FlightComputer("127.0.0.1", PORT, esp_dev=None)
    result_holder["board"] = fc.run(MAX_T)


print("=" * 66)
print("HIL LOOPBACK: host plant  <->  flight computer, no hardware")
print("=" * 66)

th = threading.Thread(target=run_host, daemon=True)
th.start()
import time
time.sleep(1.0)  # let the host bind and listen
run_board()
th.join(timeout=30)

h = result_holder.get("host", {})
b = result_holder.get("board", {})

print()
print("BOARD (flight computer) report:")
for k, v in b.items():
    print(f"    {k}: {v}")
print()
print("HOST (plant) report:")
for k, v in h.items():
    print(f"    {k}: {v}")

print()
print("=" * 66)
print("EQUIVALENCE vs pure simulation")
print("=" * 66)

# Load the pure-sim reference (seed 1 is the committed reference; seed 0 here).
# Re-run pure sim at seed 0 for an apples-to-apples comparison.
from varuna.scene import DisasterSite
from varuna.mission import MissionRunner, MissionConfig
ref = MissionRunner(DisasterSite(), MissionConfig(), seed=0).run(verbose=False)

print(f"  {'metric':<26}{'pure sim':>12}{'HIL':>12}{'delta':>10}")


def row(name, a, bv, fmt="{:.2f}"):
    da = bv - a
    print(f"  {name:<26}{fmt.format(a):>12}{fmt.format(bv):>12}{fmt.format(da):>10}")


row("nav error mean (m)", ref["nav_error_mean"], h.get("nav_error_mean", 0))
row("nav error max (m)", ref["nav_error_max"], h.get("nav_error_max", 0))
row("mission duration (s)", ref["duration"], h.get("sim_time", 0), "{:.0f}")
row("ticks", ref["duration"] / 0.05, h.get("ticks", 0), "{:.0f}")

ok = (h.get("final_phase") == 7 and abs(h.get("sim_time", 0) - ref["duration"]) < 30
      and abs(h.get("nav_error_mean", 9) - ref["nav_error_mean"]) < 0.15)
print()
print("  RESULT:", "EQUIVALENT" if ok else "DIVERGENT (investigate)")
print(f"  loop timing on host: mean {b.get('loop_ms_mean',0):.3f} ms, "
      f"p99 {b.get('loop_ms_p99',0):.3f} ms, budget {b.get('budget_ms',0):.0f} ms")
print(f"  protocol integrity: host crc {h.get('crc_errors',0)}, "
      f"board crc {b.get('host_crc_errors',0)}")
