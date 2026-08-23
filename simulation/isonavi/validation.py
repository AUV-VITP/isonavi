"""Water-tank scenes used to validate the sonar model against real imagery.

The public FLS dataset was captured with an ARIS Explorer 3000 looking across
the floor of a water tank at small debris objects. To test whether this
simulator produces imagery in the same distribution, we rebuild that setup
geometrically: a flat floor, the same object classes at comparable scale, and
a matching sensor geometry.

The test itself is deliberately zero-shot. A detector trained only on the real
tank imagery is run on simulated frames without any fine tuning or domain
adaptation. If it fires on the simulated objects, the simulator is placing
targets, shadows and speckle inside the distribution the detector learned from
real data, which is a far stronger statement about realism than any pixel
similarity metric.

Objects are built from the analytic primitives rather than meshes. What a
sonar records is the silhouette, the specular highlight, and the shadow, so
matching gross shape and material is what matters.
"""

from __future__ import annotations

import numpy as np

from .geometry import Scene, Box, Cylinder, Sphere, Heightfield
from .acoustics import MAT_INDEX


def _ring_of_spheres(scene, centre, ring_r, tube_r, material, n=20, name="tire"):
    """A torus approximated by a ring of spheres.

    Adequate because the acoustic signature of a small ring target is its
    outline and the shadow it throws, not its exact surface curvature.
    """
    for k in range(n):
        a = 2 * np.pi * k / n
        scene.add(Sphere([centre[0] + ring_r * np.cos(a),
                          centre[1] + ring_r * np.sin(a),
                          centre[2]], tube_r, material, name=name))


def add_object(scene, kind, centre, yaw=0.0):
    """Place one debris object of a named class. Returns its footprint radius."""
    steel = MAT_INDEX["steel"]
    rubble = MAT_INDEX["rubble"]
    x, y, z = centre
    if kind == "tire":
        _ring_of_spheres(scene, (x, y, z + 0.09), 0.28, 0.09, rubble, name="tire")
        return 0.40
    if kind == "can":
        scene.add(Cylinder([x, y], 0.033, z, z + 0.12, steel, name="can"))
        return 0.10
    if kind == "bottle":
        scene.add(Cylinder([x, y], 0.036, z, z + 0.20, steel, name="bottle"))
        return 0.12
    if kind == "standing-bottle":
        scene.add(Cylinder([x, y], 0.036, z, z + 0.26, steel, name="standing-bottle"))
        return 0.12
    if kind == "shampoo-bottle":
        scene.add(Box([x, y, z + 0.10], [0.05, 0.03, 0.10], steel, yaw=yaw,
                      name="shampoo-bottle"))
        return 0.11
    if kind == "drink-carton":
        scene.add(Box([x, y, z + 0.09], [0.035, 0.035, 0.09], rubble, yaw=yaw,
                      name="drink-carton"))
        return 0.10
    if kind == "chain":
        for k in range(14):
            t = (k - 7) * 0.055
            scene.add(Sphere([x + t * np.cos(yaw), y + t * np.sin(yaw), z + 0.03],
                             0.032, steel, name="chain"))
        return 0.45
    if kind == "propeller":
        for k in range(3):
            a = yaw + 2 * np.pi * k / 3
            scene.add(Box([x + 0.13 * np.cos(a), y + 0.13 * np.sin(a), z + 0.03],
                          [0.13, 0.035, 0.02], steel, yaw=a, name="propeller"))
        scene.add(Cylinder([x, y], 0.045, z, z + 0.08, steel, name="propeller"))
        return 0.30
    if kind == "hook":
        _ring_of_spheres(scene, (x, y, z + 0.05), 0.10, 0.022, steel, n=10, name="hook")
        scene.add(Cylinder([x, y - 0.12], 0.02, z, z + 0.16, steel, name="hook"))
        return 0.20
    if kind == "valve":
        scene.add(Cylinder([x, y], 0.075, z, z + 0.09, steel, name="valve"))
        _ring_of_spheres(scene, (x, y, z + 0.11), 0.11, 0.02, steel, n=12, name="valve")
        return 0.20
    if kind == "mine":
        scene.add(Sphere([x, y, z + 0.22], 0.22, steel, name="mine"))
        for k in range(6):
            a = 2 * np.pi * k / 6
            scene.add(Cylinder([x + 0.18 * np.cos(a), y + 0.18 * np.sin(a)],
                               0.022, z + 0.30, z + 0.40, steel, name="mine"))
        return 0.32
    raise KeyError(f"unknown object class {kind!r}")


TANK_CLASSES = ("tire", "can", "bottle", "drink-carton", "chain", "propeller",
                "hook", "valve", "standing-bottle", "shampoo-bottle", "mine")

# Class index order of the public dataset. Synthetic labels must use exactly
# this order, otherwise a model trained on simulated frames and evaluated on
# the real test set would be scored against permuted classes.
DATASET_CLASSES = ("mine", "can", "bottle", "drink-carton", "chain",
                   "propeller", "tire", "hook", "valve", "shampoo-bottle",
                   "standing-bottle")


def tank_scene(objects, floor_z=-4.0, extent=26.0, roughness=0.012, seed=0):
    """Flat tank floor with the given objects placed on it.

    ``objects`` is a list of (class_name, x, y, yaw).
    Returns (scene, ground_truth) where ground_truth lists class and position.
    """
    rng = np.random.default_rng(seed)
    n = int(extent / 0.25) + 1
    H = np.full((n, n), floor_z)
    noise = rng.normal(0, 1, H.shape)
    for _ in range(3):
        noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
                 + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5.0
    H = H + roughness * noise / max(noise.std(), 1e-9)

    bed = Heightfield(-extent / 2, -extent / 2, 0.25, 0.25, H,
                      MAT_INDEX["sand"], name="tank_floor",
                      max_range=40.0, step=0.25, refine=16)
    scene = Scene([bed])
    truth = []
    for kind, x, y, yaw in objects:
        z = float(bed.height(np.array([x]), np.array([y]))[0])
        r = add_object(scene, kind, (x, y, z), yaw)
        truth.append({"class": kind, "x": x, "y": y, "z": z, "radius": r})
    return scene, truth


def random_tank(n_objects=4, seed=0, classes=None, x_range=(2.5, 9.0),
                y_spread=2.2):
    """A tank scene with objects scattered in front of the sensor."""
    rng = np.random.default_rng(seed)
    classes = list(classes or TANK_CLASSES)
    objs = []
    placed = []
    for _ in range(n_objects):
        for _try in range(60):
            x = rng.uniform(*x_range)
            y = rng.uniform(-y_spread, y_spread)
            if all(np.hypot(x - px, y - py) > 1.1 for px, py in placed):
                placed.append((x, y))
                objs.append((classes[rng.integers(len(classes))], x, y,
                             rng.uniform(0, 2 * np.pi)))
                break
    return tank_scene(objs, seed=seed)


def add_tank_clutter(scene, rng, floor_z=-3.0, back_wall_x=11.0, side_y=5.0):
    """Structure that makes a real tank capture look the way it does.

    Real water-tank sonar frames are not clean: the walls return strongly at
    the far edge, and the supporting rig, pipework and cabling fill the scene
    with bright linear features. Without them a simulated frame is far emptier
    than any real one, which is by itself enough to put it outside the
    distribution a detector learned.
    """
    conc = MAT_INDEX["concrete"]
    steel = MAT_INDEX["steel"]
    # Back and side walls.
    scene.add(Box([back_wall_x, 0.0, floor_z + 1.6], [0.25, side_y * 1.6, 1.6],
                  conc, name="tank_wall"))
    for s in (-1, 1):
        scene.add(Box([back_wall_x * 0.5, s * side_y, floor_z + 1.6],
                      [back_wall_x * 0.75, 0.25, 1.6], conc, name="tank_wall"))
    # Pipework and cable runs lying on the floor.
    for _ in range(rng.integers(3, 7)):
        x = rng.uniform(1.5, back_wall_x - 1.0)
        y = rng.uniform(-side_y * 0.8, side_y * 0.8)
        ln = rng.uniform(0.8, 3.0)
        yaw = rng.uniform(0, np.pi)
        scene.add(Box([x, y, floor_z + 0.05], [ln, 0.05, 0.05], steel,
                      yaw=yaw, name="clutter"))
    # Vertical posts of the mounting rig.
    for _ in range(rng.integers(1, 4)):
        x = rng.uniform(2.0, back_wall_x - 1.0)
        y = rng.uniform(-side_y * 0.7, side_y * 0.7)
        scene.add(Cylinder([x, y], rng.uniform(0.04, 0.09),
                           floor_z, floor_z + rng.uniform(0.6, 2.0),
                           steel, name="clutter"))
    return scene


def labelled_tank_frame(seed, sonar_cfg, size=640, n_objects=None,
                        classes=None, clutter=True):
    """Render one simulated tank frame with YOLO bounding boxes.

    Boxes are derived analytically by projecting each object's footprint into
    the fan image, so the labels are exact rather than annotated. Returns
    (image_uint8, boxes) with boxes as (class_index, xc, yc, w, h) normalised.
    """
    from .acoustics import ForwardLookingSonar
    from .dynamics import rot_body_to_world

    rng = np.random.default_rng(seed)
    classes = list(classes or TANK_CLASSES)
    n = n_objects if n_objects is not None else int(rng.integers(2, 6))

    floor_z = -3.0
    objs, placed = [], []
    for _ in range(n):
        for _try in range(80):
            x = rng.uniform(1.6, 7.0)
            y = rng.uniform(-1.9, 1.9)
            if all(np.hypot(x - px, y - py) > 0.95 for px, py in placed):
                placed.append((x, y))
                objs.append((classes[rng.integers(len(classes))], x, y,
                             rng.uniform(0, 2 * np.pi)))
                break
    scene, truth = tank_scene(objs, floor_z=floor_z, extent=30.0, seed=seed)
    if clutter:
        add_tank_clutter(scene, rng, floor_z=floor_z)

    alt = rng.uniform(0.55, 1.15)
    pitch = np.radians(rng.uniform(5.0, 13.0))
    yaw = rng.uniform(-0.05, 0.05)
    pose = np.array([0.0, 0.0, floor_z + alt, 0.0, pitch, yaw])

    # Choose the range window from the geometry so the fan is actually filled
    # with seabed return. With altitude h, depression d and vertical half
    # beamwidth phi, the bed is insonified between h/sin(d+phi) and
    # h/sin(d-phi). A window outside that leaves large dead areas that no real
    # capture would show, and would let a detector separate simulated frames
    # from real ones on layout alone.
    from dataclasses import replace as _replace
    phi = np.radians(sonar_cfg.fov_v_deg) / 2.0
    near = alt / max(np.sin(pitch + phi), 1e-3)
    far = alt / max(np.sin(pitch - phi), np.sin(np.radians(1.5)))
    r_min = max(0.4, near * 0.92)
    r_max = float(np.clip(far * 0.95, r_min + 2.5, sonar_cfg.r_max))
    sonar_cfg = _replace(sonar_cfg, r_min=float(r_min), r_max=r_max)

    fls = ForwardLookingSonar(sonar_cfg, scene)
    fr = fls.ping(pose)
    img = fr.to_cartesian(size)
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    rmax = float(fr.ranges.max())
    rmin_img = float(fr.ranges.min())
    half = float(fr.bearings.max())
    R = rot_body_to_world(pose[3], pose[4], pose[5])
    # Only objects that actually returned an echo may be labelled. An object
    # outside the insonified band, or hidden behind something closer, does not
    # appear in the image, and labelling it would teach the detector to
    # hallucinate targets in empty water.
    #
    # Visibility is tested exactly: a ray is cast from the sensor to the
    # object, and the object counts as seen only if it is inside the vertical
    # beam and nothing intercepts that ray before it.
    phi_v = np.radians(sonar_cfg.fov_v_deg) / 2.0
    origins, dirs, ranges_to, elevs = [], [], [], []
    for t in truth:
        tgt = np.array([t["x"], t["y"], t["z"] + min(0.10, t["radius"])])
        local = R.T @ (tgt - pose[:3])
        rr = float(np.linalg.norm(local))
        origins.append(pose[:3])
        dirs.append((tgt - pose[:3]) / max(rr, 1e-9))
        ranges_to.append(rr)
        elevs.append(np.arcsin(np.clip(local[2] / max(rr, 1e-9), -1, 1)))
    visible = np.zeros(len(truth), dtype=bool)
    if truth:
        hit = scene.intersect(np.array(origins), np.array(dirs))
        for i, t in enumerate(truth):
            inside_beam = abs(elevs[i]) <= phi_v
            unoccluded = hit.t[i] >= ranges_to[i] - max(t["radius"], 0.12)
            visible[i] = bool(inside_beam and unoccluded)

    boxes = []
    for i, t in enumerate(truth):
        if not visible[i]:
            continue
        local = R.T @ (np.array([t["x"], t["y"], t["z"] + 0.08]) - pose[:3])
        r = float(np.linalg.norm(local))
        b = float(np.arctan2(local[1], local[0]))
        if not (fr.ranges.min() < r < rmax) or abs(b) > half * 0.94:
            continue
        X, Y = r * np.sin(b), r * np.cos(b)
        xs1 = rmax * np.sin(half)
        col = (X + xs1) / (2 * xs1) * (size - 1)
        # The fan image spans the range window, not zero to r_max.
        row = (size - 1) - (Y - rmin_img) / max(rmax - rmin_img, 1e-6) * (size - 1)
        px_per_m = (size - 1) / max(rmax - rmin_img, 1e-6)
        # Objects are elongated in range by their acoustic shadow, which is
        # what the real annotations enclose as well.
        w = max(2.1 * t["radius"] * px_per_m, 14.0)
        h = max(2.1 * t["radius"] * px_per_m, 14.0) * 1.35
        row = row + h * 0.18
        if not (0 < col < size and 0 < row < size):
            continue
        # Clip the box to the image before normalising. A target near the edge
        # of the fan otherwise produces coordinates outside [0, 1], which YOLO
        # rejects as a corrupt label and silently drops from the split.
        x0 = max(col - w / 2.0, 0.0)
        x1 = min(col + w / 2.0, size - 1.0)
        y0 = max(row - h / 2.0, 0.0)
        y1 = min(row + h / 2.0, size - 1.0)
        if x1 - x0 < 6.0 or y1 - y0 < 6.0:
            continue
        boxes.append((DATASET_CLASSES.index(t["class"]),
                      ((x0 + x1) / 2.0) / size, ((y0 + y1) / 2.0) / size,
                      (x1 - x0) / size, (y1 - y0) / size))
    return u8, boxes
