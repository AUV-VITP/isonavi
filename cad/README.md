# CAD

Parametric model of the VARUNA-1 vehicle, built from the simulation
parameters so the drawing is the vehicle that was simulated.

`varuna_cad.py` derives every principal dimension from
`simulation/varuna/dynamics.py::VARUNA_1`:

- displaced volume 0.0282 m3 sets the hull size; the built hull matches it to
  within 0.01 percent
- the eight thruster positions come from the allocation arms (0.42, 0.30, 0.38,
  0.26) m
- CB-above-CG 85 mm and fin_coeff 110 size the ballast and the aft fins

The hull is a Myring profile, the standard low-drag AUV body. Thrusters sit on
faired pylons in the vectored eight-thruster arrangement: four canted
horizontal units, four vertical units.

## Outputs

- `varuna_vehicle.step`  parametric assembly for CAD
- `varuna_vehicle.stl`   mesh for rendering and printing
- `varuna_hull.stl`      hull only
- `varuna_cad_params.json` derived dimensions and the volume check

## Reproducing

```bash
python varuna_cad.py     # build STEP + STL from the sim parameters
python render_cad.py     # six-view verification render
```
