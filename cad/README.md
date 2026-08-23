# CAD

A parametric model of isonavi-1 whose job is to check the simulation, not to
illustrate it.

## What is solved rather than assumed

`isonavi_layout.py` lists every component the vehicle carries with a real mass
and a real station, then solves three things:

1. the hull size, so displaced volume equals the value the simulation uses for
   buoyancy;
2. the trim ballast mass, so total dry mass equals the simulated mass;
3. the ballast station, so the vehicle floats level.

Because those are solved and not chosen, the remaining quantities are
predictions. The budget closes exactly: 28.000 kg against a 28.0 kg target,
0.02820 m3 against 0.0282 m3, longitudinal trim 0.000 mm, and +1.96 N of net
buoyancy so the vehicle surfaces on a power failure.

## What it found

The separation between centre of buoyancy and centre of gravity, which sets the
passive roll and pitch restoring moment, falls out of the layout at **27.3 mm**.
The dynamics model had assumed 85 mm.

85 mm is not reachable in a hull this size: it puts the centre of gravity
15 mm off the bottom skin, which no arrangement of a battery and a pressure
housing can achieve. The simulation was corrected to the CAD value and the full
mission and repeatability suite re-run. The result barely moves, because at
survey speed the fins supply roughly fifty times the restoring moment the
buoyancy offset does, and the fixes are documented in the report.

The cross check found a second error. The solve had been placing the hull's
displaced volume at the hull mid-length, which a Myring form does not satisfy:
the tail is nearly twice the nose, so the centroid sits 41 mm aft. The centre
of buoyancy was in the wrong place and the trim ballast the solve returned was
105 mm out, so the budget reported level trim while describing a vehicle that
would have floated nose up. `verify_cad.py` compares the analytic layout
against the geometry kernel, and the two now agree on the centroid to 0.01 mm.

That is the loop worth having: building the vehicle properly is what caught
both errors.

## Architecture

The faired hull is itself the pressure boundary, as on survey vehicles of this
class, rather than a free flooding fairing over a separate cylindrical housing.
Two things forced that choice:

- a separate housing leaves the buoyancy to syntactic foam, and the foam volume
  needed here does not fit in the annulus between a 150 mm housing and the skin;
- trimming that arrangement to a useful stability margin needs ballast in a bulb
  below the hull, and a keel bulb is a snag hazard in the debris field this
  vehicle is meant to survey.

With the hull as the pressure vessel the ballast sits inside on the keel line
and nothing protrudes below the skin except the acoustics.

The 9.0 kg of trim ballast is also the payload growth margin: at fixed
displacement it can be traded for instrumentation without touching the hull.

## Depth rating

`structures.py` sizes the hull. For a thin shell under external pressure the
governing mode is elastic instability, not strength: at the 50 m design depth
the hoop stress carries a factor of 23, but an unstiffened 4 mm skin of this
diameter collapses at 66 m. Ring frames at 200 mm spacing move that to 482 m,
a factor of 7.3, giving 9.6 on the design depth. The frames in the model are
load bearing, not decoration.

The pylons carry a 214 mm cantilever with one thruster at its 120 N limit,
reaching 10.3 MPa at the root, a factor of 24. They are sized by the flow, not
by strength.

These are closed form hand calculations, not a substitute for FEA or a
pressure test.

## Where the drag comes from

`hydrodynamics.py` derives the coefficients the dynamics model uses. Bare hull
friction comes from the ITTC 1957 line with a form factor; the appendages are
built up over the ducts, pylons and fins as modelled.

The total is 38.9 against the 32 the simulation assumes, within 21 percent,
which is close for a number that was previously an assertion. The interesting
part is the split: the bare Myring hull is 3 percent of axial drag and the
eight thruster ducts are 93 percent. Refining the hull form would buy almost
nothing. If this vehicle needs to go further, the thrusters are what to fair or
retract.

Re-checking the station keeping envelope at the higher drag drops it from
2.93 to 2.69 m/s against a 2.4 m/s site current, so the margin is 1.12 rather
than 1.22. The design holds on a thinner margin than the simulation implies.

## Endurance and cost

 integrates propeller power over the eight thruster forces the
mission actually logged, using momentum theory. The reference mission draws
295 Wh, a quarter of a usable charge, so the vehicle carries about four such
missions between charges. Endurance holding station runs from 21.8 h in slack
water to 1.2 h against the 2.4 m/s design current, a factor of eighteen. That
spread is the price of hover authority and has to be quoted against a current,
never as a single number.

 prices one airframe at 32,825 USD, about 27.6 lakh INR, parts only.
The two acoustic instruments are 72 percent of it; the hull, propulsion, power
and the whole autonomy stack together are 9,115 USD. The cheap part is the part
that is ours.

## Losing a thruster

`redundancy.py` asks, for every single and double failure, whether the
surviving thrusters can still make the wrench the mission needs with zero net
moment. That is a linear program, not an inspection of the matrix.

Losing a vertical unit costs heave and nothing else. Losing one of the four
horizontal units halves surge, from 339 N to 170 N, dropping the holdable
current from 2.68 to 1.82 m/s, which is below the 2.4 m/s design condition.
The cause is geometric: the survivor on the light side has to balance two on
the heavy side.

Heading offset does not rescue it, because turning to gain thrust presents the
hull broadside and broadside damping is 210 against 38.9 in surge; the best
degraded envelope over all headings is 1.86 m/s. The fins cannot help either,
because what binds is sway rather than yaw.

So the vehicle is single fault tolerant for attitude and depth, and not for
station keeping at the design current. The remedy, if that were required, is
196 N per thruster instead of 120 N, or six horizontal units instead of four.
It is recorded as a limitation rather than designed around quietly.

## Files

| file | what it is |
| --- | --- |
| `isonavi_layout.py` | component masses, stations, and the budget solve |
| `isonavi_cad.py` | the geometry, driven by the layout |
| `render_views.py` | renders and the dimensioned drawing |
| `verify_cad.py` | cross checks the layout against the geometry kernel |
| `structures.py` | pressure hull and pylon sizing |
| `hydrodynamics.py` | drag and added mass derived from the geometry |
| `energy.py` | power and endurance, integrated over the flight log |
| `bom.py` | bill of materials |
| `redundancy.py` | thruster-out envelope, by linear program |
| `scene_render.py` | the vehicle placed in the modelled site |
| `isonavi_vehicle.step` | parametric assembly for CAD |
| `isonavi_cad_params.json` | derived dimensions and the verification numbers |

Meshes are gitignored because they regenerate in a few seconds.

## Reproducing

```bash
python isonavi_layout.py    # the mass and stability budget
python isonavi_cad.py       # STEP and STL, plus the volume check
python render_views.py     # hero, GA, dimensioned drawing, cutaway, exploded
python verify_cad.py       # cross check the two against each other
python structures.py       # depth rating and pylon loads
python hydrodynamics.py    # drag build up and the current envelope
python energy.py           # endurance against current
python bom.py              # bill of materials
python redundancy.py       # what a lost thruster costs
python scene_render.py     # the vehicle in its site
```

`isonavi_cad.py` reports the volume of the hull it actually built against the
volume it was asked for. That check currently closes to 0.003 percent, and it
is the reason the drawing and the physics cannot quietly drift apart.
