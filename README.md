# isonavi

An autonomous underwater vehicle for post-flood search and bridge foundation
inspection in water where divers cannot work. Physics-based sonar simulation, a
full autonomy stack validated on RISC-V flight hardware, and a component-level
vehicle design whose mass and buoyancy budget closes against the simulation
that flies it.

**[Technical report](docs/isonavi_report.pdf)** (29 pages, design and
verification) and **[two page brief](docs/isonavi_brief.pdf)**. Every number in
both is generated from the measured result files by
`simulation/eval/make_metrics_tex.py`, so the documents cannot drift out of
step with the experiments.

## Problem

Post-flood search and bridge foundation inspection depend on divers. In monsoon
flood the water is opaque with suspended sediment, the current exceeds what a
diver can work in, and submerged debris makes entry unsafe. The reference
scenario is the 2016 Savitri river bridge collapse at Mahad, which took days to
search for exactly these reasons.

Optical imaging is not a candidate at these turbidity levels, modelled here at
3.2 g/L of suspended sediment. The system is built around acoustic perception
throughout, and that single decision drives the payload, the hull and the
autonomy.

## Headline results

Measured against simulation ground truth the vehicle never observes, across 5
independent runs with sensor noise, DVL dropout and speckle re-randomised.

| | |
| --- | --- |
| Maximum holdable current, open frame baseline | 0.96 m/s |
| Maximum holdable current, isonavi-1 | 2.69 m/s, on drag derived from its own geometry |
| Design site current | 2.4 m/s |
| Autonomous mission, path and duration | 573 m, 893 s |
| Missions completed | 5/5 |
| Search box coverage | 95.3 +/- 0.2 % |
| Navigation error with no external fix | 0.322 +/- 0.161 m |
| DVL availability | 96.0 +/- 0.3 % |
| Bathymetric map RMSE against truth | 0.285 +/- 0.001 m |
| Target localisation error, no training data | 1.80 +/- 0.40 m |
| Sonar detector on the real ARIS test split | 99.0 % mAP@0.5 |
| Full mission on a RISC-V flight computer | 0.107 m nav error, 17,884 ticks, 0 CRC errors |
| Board compute per tick | 34.6 ms against a 50 ms budget |
| Mass and buoyancy budget | closes at 28.0 kg, 0.0282 m3, level trim, +1.96 N |
| Depth rating | 50 m, factor 9.6 on framed buckling |
| Airframe cost, one vehicle | ₹38.3 lakh (₹3,827,800 at ₹95.64 per USD) |
| Programme cost including capital, materials and trials | ₹72.8 lakh (₹7,278,908 at ₹95.64 per USD) |

## What is here

| Module | Purpose |
| --- | --- |
| `simulation/isonavi/geometry.py` | Vectorised ray casting: analytic primitives and a bathymetric heightfield |
| `simulation/isonavi/acoustics.py` | Physics-based forward-looking sonar image formation |
| `simulation/isonavi/scene.py` | Post-collapse river site, bathymetry, scour, current field |
| `simulation/isonavi/validation.py` | Water-tank scenes and labelled synthetic frame generation |
| `simulation/isonavi/dynamics.py` | 6-DOF Fossen dynamics, added mass, fins, thruster allocation |
| `simulation/isonavi/sensors.py` | DVL, IMU, depth cell error models including bottom-lock loss |
| `simulation/isonavi/estimation.py` | 12-state EKF for GPS-denied navigation |
| `simulation/isonavi/control.py` | Cascaded pose control, force feedforward, current estimation |
| `simulation/isonavi/mapping.py` | Bathymetric mapping, scour quantification, target detection |
| `simulation/isonavi/mission.py` | Mission state machine and the closed simulation loop |
| `cad/` | Parametric vehicle, mass and trim solve, depth rating, drag, energy, bill of materials |
| `hil/` | Hardware in the loop: flight computer, actuator interface, host plant |
| `docs/` | The report and brief, and the script that generates every number in them |

## Sonar model

Images are formed from the active sonar equation applied per range and bearing
cell, not by texturing a depth buffer. Three properties of real imagery are
reproduced explicitly because they are what perception actually keys on:
acoustic shadow from ray-cast occlusion, elevation ambiguity from summing
elevation rays under the vertical beam pattern, and Gamma-distributed speckle
from coherent imaging.

Verified on a steel target over silt: 12.7 dB target return and a 43.6 dB
acoustic shadow behind it, with the seabed return falling smoothly with range.
One ping costs about 150 ms on eight CPU cores, comfortably faster than the
2 Hz the vehicle uses. No GPU is required.

## Vehicle

The usual commercial baseline cannot do this mission. An open frame
BlueROV2-class vehicle holds station only to 0.96 m/s against a 2.4 m/s design
current, and no controller can fix that: it is installed thrust against drag.

isonavi-1 is sized for the job instead. A faired hull cuts axial drag by roughly
four times, larger thrusters and longer moment arms restore authority, and
fixed fins are then mandatory, because a faired hull is directionally unstable
under the Munk moment produced by its own added-mass asymmetry. The vehicle
must fly nose first into the flow, which propagates into the mission planner.

The drag was then rederived from the built geometry rather than assumed. The
bare hull contributes 3 % of axial drag and the eight thruster ducts contribute
93 %, which is the kind of result that only exists once the CAD does. The
station keeping envelope survives that recheck at 2.69 m/s against a 2.4 m/s
site current.

## Perception, including what did not work

Two independent channels, with the results for both stated plainly.

**Geometric detection works and needs no training data.** Objects standing
above the riverbed are isolated from the bathymetric residual. This matters
because no public training data exists for submerged Indian buses, the target
class that matters most. Both submerged vehicles are found in every run at
1.80 +/- 0.40 m mean localisation error, reducing an unsearchable 1.54 hectare
box to roughly 24 georeferenced contacts to dive. It is a screening tool rather
than a classifier, and the flat deck slab is missed in 2 runs of 5.

**The learned detector is strong inside its own domain and does not cross.** A
YOLOv8s detector trained on the public marine debris ARIS dataset reaches
99.0 % mAP@0.5 on the held out real test split. A detector trained only on
simulator output reaches 84.6 % on simulated data. Neither transfers:

| Trained on | Tested on | mAP@0.5 |
| --- | --- | --- |
| real | real | 99.0 % |
| simulated | simulated | 84.6 % |
| real | simulated | 0.8 % |
| simulated | real | 0.4 % |

**Simulator pretraining does not reduce the real-data requirement either.**
Against generic natural-image initialisation it is worse at every dataset size
that matters, 10.5 points down at 40 real images and 4.6 down at 320, and only
draws level at 640. This is reported rather than buried. The likely cause is
ordinary negative transfer from a narrow synthetic source domain.

The conclusion the report draws is that the simulator earns its place through
closed-loop autonomy validation, vehicle design decisions that cannot be made
any other way, and the geometric detector, not through supplying training data
for fine-grained classification.

## Hardware in the loop

The autonomy stack runs on real hardware, driven by the simulator. A LicheeRV
Nano (RISC-V rv64 at 750 MHz) runs the EKF, the controller and the mission
state machine; an ESP32 generates the eight thruster PWM channels; the host
runs the physics and holds ground truth. The board never sees ground truth, so
a matching result means the autonomy runs on hardware rather than that the
hardware was handed the answer.

The full 893 s mission completed on the board at 0.107 m navigation error,
reproducing the pure-simulation result to within centimetres, with 34.6 ms of
compute per tick against a 50 ms budget and zero protocol errors across 17,884
ticks. The same trajectory on two instruction sets agrees to roughly one part
in 10^12, which is the signature of the same arithmetic rather than of two
implementations that happen to behave alike. See `hil/`.

## Vehicle CAD

`cad/isonavi_layout.py` lists every component the vehicle carries with a real
mass at a real station, then solves the hull size, the trim ballast mass and
the ballast station so that displaced volume, dry mass and level trim all match
the simulation. The budget closes: 28.0 kg, 0.0282 m3, trim 0.000 mm, and
+1.96 N net buoyancy so the vehicle surfaces on a power failure.
`cad/isonavi_cad.py` builds the geometry from that layout across 52 parts, and
the volume of the hull it actually built closes against the volume it was asked
for to 0.001 percent.

Because the layout is solved rather than chosen, the stability margin is a
prediction rather than a decoration. It came out at 25.6 mm against the 85 mm
the dynamics model had assumed. 85 mm is not reachable in a 180 mm hull, so the
simulation was corrected and the whole mission and repeatability suite re-run.
The results barely moved, because the fins supply roughly sixty times the
restoring moment the buoyancy offset does at survey speed. Building the vehicle
properly is what caught the error. See `cad/`.

## What it costs

Prices are shown in Indian rupees at ₹95.64 per USD (mid-market open,
24 August 2026). Sourced catalogue prices are USD, converted at that rate.

| Category | Amount | What it buys |
| --- | --- | --- |
| Airframe, one vehicle | ₹3,827,800 | The parts that fly. 59 % of it is the two acoustic instruments |
| Capital equipment and tooling | ₹1,140,890 | Bought once and used for every build after: printers, vacuum and cure kit, pressure test vessel, bench instruments |
| Raw materials and consumables | ₹172,248 | Per airframe, with stock to remake the hull once |
| Validation and field trials | ₹1,358,088 | Tow tank, witnessed pressure test, reservoir and river trials, diver ground truth |
| Contingency at 12 % | ₹779,883 | |
| **Programme total** | **₹7,278,908** | About ₹72.8 lakh |

The report gives all 58 lines with the reason each one exists and the basis
its price rests on, published, quotation, market, workshop or service. The
whole sheet is generated by `cad/bom.py`, so the totals cannot drift from the
parts list.

Two figures are worth pulling out. The airframe is 59 % acoustics, so the
vehicle around the sensors is cheap and the sensors are not. Validation is
19 % of the programme, which is the price of replacing the last of the closed
form analysis with measurement.

## Running it

```bash
cd simulation
PYTHONPATH=. python tests/smoke_sonar.py      # sonar physics verification
PYTHONPATH=. python tests/smoke_control.py    # dynamics, EKF, control
PYTHONPATH=. python eval/thrust_envelope.py   # vehicle envelope analysis
PYTHONPATH=. python eval/run_mission.py       # full autonomous mission
PYTHONPATH=. python eval/aggregate_seeds.py   # repeatability across seeds
PYTHONPATH=. python eval/detect_eval.py       # geometric target detection
PYTHONPATH=. python eval/make_figures.py      # report figures
PYTHONPATH=. python eval/make_video.py        # operator console video

cd ../cad
python isonavi_layout.py                      # mass, buoyancy and trim budget
python isonavi_cad.py                         # geometry, STEP and STL
python verify_cad.py                          # layout against the geometry kernel

cd ../docs && bash build.sh                   # regenerate metrics and both documents
```

GPU training runs as Kaggle kernels in `simulation/ml/kaggle/`.

## Status

Simulation, autonomy, mapping and acoustic perception are complete and
verified. The autonomy and its actuation path are validated on flight hardware,
the perception is validated on real sonar, and the vehicle is validated on
paper against a closed mass, buoyancy, depth and drag analysis.

That ordering is deliberate rather than an accident of what was easy. The
autonomy is the part that is difficult to buy, difficult to copy and specific
to this problem. Hulls, thrusters and pressure housings are none of those
things, and the bill of materials shows they are also the cheap part.

What the work needs next is water: a tank, then a river, then the instrumented
vehicle the analysis already specifies. The hydrodynamic coefficients are
closed-form estimates cross checked against the CAD geometry and agree to 21 %
in axial drag; CFD and a tow tank replace them. The site is representative
rather than surveyed. Those limits are stated in full in the report rather than
left for a reviewer to find.
