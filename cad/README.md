# CAD

A parametric model of VARUNA-1 whose job is to check the simulation, not to
illustrate it.

## What is solved rather than assumed

`varuna_layout.py` lists every component the vehicle carries with a real mass
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

## Files

| file | what it is |
| --- | --- |
| `varuna_layout.py` | component masses, stations, and the budget solve |
| `varuna_cad.py` | the geometry, driven by the layout |
| `render_views.py` | renders and the dimensioned drawing |
| `verify_cad.py` | cross checks the layout against the geometry kernel |
| `structures.py` | pressure hull and pylon sizing |
| `varuna_vehicle.step` | parametric assembly for CAD |
| `varuna_cad_params.json` | derived dimensions and the verification numbers |

Meshes are gitignored because they regenerate in a few seconds.

## Reproducing

```bash
python varuna_layout.py    # the mass and stability budget
python varuna_cad.py       # STEP and STL, plus the volume check
python render_views.py     # hero, GA, dimensioned drawing, cutaway, exploded
python verify_cad.py       # cross check the two against each other
python structures.py       # depth rating and pylon loads
```

`varuna_cad.py` reports the volume of the hull it actually built against the
volume it was asked for. That check currently closes to 0.003 percent, and it
is the reason the drawing and the physics cannot quietly drift apart.
