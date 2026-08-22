# VARUNA

Autonomous underwater reconnaissance and assessment for post-flood disaster
response. Physics-based sonar simulation, a full autonomy stack, and acoustic
perception for turbid, high-current Indian river conditions where divers
cannot work.

Submission material for RakshaTech Synapse 2026 (IHFC, IIT Delhi / DST),
broad technology category 1, Autonomous and Robotic Platforms.

## Problem

Post-flood search and bridge foundation inspection depend on divers. In
monsoon flood the water is opaque with suspended sediment, the current exceeds
what a diver can work in, and submerged debris makes entry unsafe. The
reference scenario is the 2016 Savitri river bridge collapse at Mahad, which
took days to search for exactly these reasons.

Optical imaging is useless at these turbidity levels, so the system is built
around acoustic perception throughout.

## Headline results

Measured against simulation ground truth the vehicle never observes, across
5 independent runs with sensor noise, DVL dropout and speckle re-randomised.

| | |
| --- | --- |
| Maximum holdable current, COTS baseline | 0.96 m/s |
| Maximum holdable current, VARUNA-1 | 2.93 m/s |
| Autonomous mission, path / duration | 573 m / 892 s |
| Missions completed | 5/5 |
| Search-box coverage | 95.3 +/- 0.2 % |
| Navigation error, mean | 0.32 +/- 0.16 m |
| DVL availability | 96.0 +/- 0.3 % |
| Bathymetric map RMSE against truth | 0.286 +/- 0.001 m |
| Scour depth recovery | 86 % |
| Primary targets found, no training data | 4/4 in 4 of 5 runs, 1.79 +/- 0.37 m |
| Sonar detector on real ARIS test split | 99.0 % mAP@0.5 |
| Full mission on RISC-V flight computer | nav error 0.131 m, DONE, 0 CRC errors |
| Board compute per tick | 30.4 ms mean vs 50 ms budget (1.5x margin) |

## What is here

| Module | Purpose |
| --- | --- |
| `simulation/varuna/geometry.py` | Vectorised ray casting: analytic primitives and a bathymetric heightfield |
| `simulation/varuna/acoustics.py` | Physics-based forward-looking sonar image formation |
| `simulation/varuna/scene.py` | Post-collapse river site, bathymetry, scour, current field |
| `simulation/varuna/validation.py` | Water-tank scenes and labelled synthetic frame generation |
| `simulation/varuna/dynamics.py` | 6-DOF Fossen dynamics, added mass, fins, thruster allocation |
| `simulation/varuna/sensors.py` | DVL, IMU, depth cell error models including bottom-lock loss |
| `simulation/varuna/estimation.py` | 12-state EKF for GPS-denied navigation |
| `simulation/varuna/control.py` | Cascaded pose control, force feedforward, current estimation |
| `simulation/varuna/mapping.py` | Bathymetric mapping, scour quantification, target detection |
| `simulation/varuna/mission.py` | Mission state machine and the closed simulation loop |

## Sonar model

Images are formed from the active sonar equation applied per range and bearing
cell, not by texturing a depth buffer. Three properties of real imagery are
reproduced explicitly because they are what perception actually keys on:
acoustic shadow from ray-cast occlusion, elevation ambiguity from summing
elevation rays under the vertical beam pattern, and Gamma-distributed speckle
from coherent imaging.

Verified on a steel target over silt: +12.1 dB target return and -44.5 dB
shadow behind it, with the seabed return falling smoothly with range. One ping
costs about 120 ms on eight CPU cores. No GPU is required.

## Vehicle

The usual commercial baseline cannot do this mission. An open-frame
BlueROV2-class vehicle holds station only to 0.96 m/s against a 2.4 m/s design
current, and no controller can fix that: it is installed thrust against drag.

VARUNA-1 is sized for the job instead. A faired hull cuts axial drag by roughly
four times, larger thrusters and longer moment arms restore authority, and
fixed fins are then mandatory because a faired hull is directionally unstable
under the Munk moment produced by its own added-mass asymmetry. The vehicle
must fly nose-first into the flow, which propagates into the mission planner.

## Perception, including what did not work

Two independent channels, with honest results for both.

**Geometric detection works and needs no training data.** Objects standing
above the riverbed are isolated from the bathymetric residual. This matters
because no public training data exists for submerged Indian buses, which is the
target class that matters most. It finds 4/4 vehicles and structural debris at
1.79 m mean localisation error.

**The learned classifier works only in its own domain.** A YOLOv8s detector
trained on the public marine debris ARIS dataset reaches 99.0 % mAP@0.5 on the
held-out real test split. A detector trained only on simulator output reaches
84.6 % on simulated data. Neither transfers:

| Trained on | Tested on | mAP@0.5 |
| --- | --- | --- |
| real | real | 99.0 % |
| simulated | simulated | 84.6 % |
| real | simulated | 0.8 % |
| simulated | real | 0.4 % |

**Simulator pretraining does not reduce the real-data requirement either.**
Against generic natural-image initialisation it is worse at every dataset size
that matters (-10.5 points at 40 real images, -4.6 at 320) and only draws level
at 640. This is reported rather than hidden. The likely cause is ordinary
negative transfer from a narrow synthetic source domain.

The conclusion drawn in the report is that the simulator's value is closed-loop
autonomy validation, vehicle design decisions, and the geometric detector, not
supplying training data for fine-grained classification.

## Hardware in the loop

The autonomy stack runs on real hardware, driven by the simulator. A LicheeRV
Nano (RISC-V rv64 @ 750 MHz) runs the EKF, controller and mission state machine;
an ESP32 drives the thruster PWM; the host runs the physics and holds ground
truth. The board never sees ground truth, so a matching result means the
autonomy runs on hardware rather than that the hardware was handed the answer.
The full 894 s mission completed on the board at 0.131 m navigation error,
reproducing the pure-simulation result, with 30.4 ms compute per tick against a
50 ms budget and zero protocol errors. See `hil/`.

## Vehicle CAD

`cad/varuna_cad.py` builds the vehicle parametrically from the simulation
parameters. The Myring hull displaces exactly the simulated 0.0282 m3 (matched
to 0.01 percent) and the eight thruster positions come from the allocation
arms, so the drawing is the vehicle that was simulated. See `cad/`.

## Running it

```bash
cd simulation
source ~/dev/venvs/ml/bin/activate
PYTHONPATH=. python tests/smoke_sonar.py      # sonar physics verification
PYTHONPATH=. python tests/smoke_control.py    # dynamics, EKF, control
PYTHONPATH=. python eval/thrust_envelope.py   # vehicle envelope analysis
PYTHONPATH=. python eval/run_mission.py       # full autonomous mission
PYTHONPATH=. python eval/aggregate_seeds.py   # repeatability across seeds
PYTHONPATH=. python eval/make_figures.py      # report figures
PYTHONPATH=. python eval/detect_eval.py       # geometric target detection
PYTHONPATH=. python eval/cross_domain.py      # symmetric transfer evaluation
PYTHONPATH=. python eval/arch_figure.py       # architecture diagrams
PYTHONPATH=. python eval/make_video.py        # operator console video
cd ../docs && pdflatex varuna_report.tex      # technical report
```

GPU training runs as Kaggle kernels in `simulation/ml/kaggle/`. Every number in the report
is emitted by `simulation/eval/make_metrics_tex.py` from the measured result files, so the
document cannot drift out of step with the experiments.

## Status

Simulation, autonomy, mapping and perception are complete and verified in
simulation. No physical vehicle exists yet. Hydrodynamic coefficients for the
faired hull are engineering estimates and need CFD and tow-tank validation.
The site is representative rather than surveyed. These limits, and the negative
perception results above, are stated explicitly in the report.
