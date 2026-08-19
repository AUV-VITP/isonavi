# VARUNA

Autonomous underwater reconnaissance and assessment for post-flood disaster
response. Simulation, autonomy stack, and acoustic perception for operating in
turbid, high-current Indian river conditions where divers cannot work.

Submission material for RakshaTech Synapse 2026 (IHFC, IIT Delhi / DST),
broad technology category 1, Autonomous and Robotic Platforms.

## Problem

Post-flood search and structural inspection currently depends on divers.
In monsoon flood conditions the water is opaque with suspended sediment,
the current exceeds what a diver can work in, and submerged debris makes it
unsafe. The 2016 Savitri river bridge collapse at Mahad, which is the scenario
modelled here, took days to search for exactly these reasons.

Optical cameras are useless at these turbidity levels, so the system is built
around acoustic perception.

## What is here

| Module | Purpose |
| --- | --- |
| `varuna/geometry.py` | Vectorised ray casting: analytic primitives and a bathymetric heightfield |
| `varuna/acoustics.py` | Physics-based forward-looking sonar image formation |
| `varuna/scene.py` | Post-collapse river site, bathymetry, scour, and current field |
| `varuna/dynamics.py` | 6-DOF Fossen vehicle dynamics with added mass, fins, and thruster allocation |
| `varuna/sensors.py` | DVL, IMU, and depth cell error models including bottom-lock loss |
| `varuna/estimation.py` | 12-state EKF for GPS-denied navigation |
| `varuna/control.py` | Cascaded pose control with current feedforward and current estimation |
| `varuna/mapping.py` | Bathymetric mapping, scour quantification, target tracking |
| `varuna/mission.py` | Mission state machine and the closed simulation loop |

## Sonar model

The simulator forms images from the active sonar equation applied per
range and bearing cell, rather than texturing a depth buffer. It reproduces
the three properties that FLS interpretation actually depends on:

- acoustic shadow, resolved by ray casting, since shadow length encodes
  target height and is the primary recognition cue
- elevation ambiguity, by casting and summing multiple elevation rays per beam
- speckle, as Gamma distributed intensity from coherent imaging

Verified behaviour on a steel target over silt: +12.7 dB target return and
-43.6 dB shadow behind it, with continuous seabed return falling smoothly with
range.

## Vehicle

The analysis rules out the usual commercial baseline. An open-frame
BlueROV2-class vehicle can hold station against 0.96 m/s. The design current
at the modelled site is 2.4 m/s at the surface. It is swept away.

VARUNA-1 is sized for the mission instead: a faired hull cutting axial drag by
roughly a factor of four, larger thrusters, and fixed stabilising fins. A
faired hull is pitch and yaw unstable under the Munk moment produced by its own
added-mass asymmetry, so the fins are a requirement rather than an addition,
and the vehicle must fly nose-first into the flow.

| | BlueROV2 Heavy | VARUNA-1 |
| --- | --- | --- |
| max holdable current | 0.96 m/s | 2.93 m/s |
| station keeping at 1.96 m/s | swept, 14.0 m error | holds, 0.40 m RMSE |

## Reference mission result

Savitri site, 2.4 m/s surface current, 3.2 g/L suspended sediment.

| Metric | Value |
| --- | --- |
| mission duration | 892 s |
| path length | 573 m |
| search-box coverage | 84.2 % |
| navigation error, mean / max | 0.15 m / 0.29 m |
| DVL availability | 96.4 % |
| soundings mapped | 337942 |
| scour depth, pier P2 | 2.76 m measured against 3.40 m truth |
| scour depth, pier P3 | 1.67 m measured against 2.10 m truth |

## Running it

```bash
source ~/dev/venvs/ml/bin/activate
PYTHONPATH=. python tests/smoke_sonar.py     # sonar physics checks
PYTHONPATH=. python tests/smoke_control.py   # dynamics, EKF, control checks
PYTHONPATH=. python eval/run_mission.py      # full autonomous mission
```

## Status

Simulation, autonomy, and mapping are complete and verified. Acoustic
perception training and the ROS 2 packaging are in progress.
