"""Post-collapse river scenario used as the reference mission environment.

The geometry is modelled on the collapse of the Savitri river bridge on
NH-66 at Mahad, Maharashtra, on 2 August 2016, when a British-era masonry
arch bridge failed during monsoon flood and carried two state transport buses
and several private vehicles into the river. Recovery took days: the water was
fast, opaque with monsoon sediment, and divers could not work safely in it.
That combination, zero optical visibility plus current plus submerged
structural debris, is the operating envelope this system targets.

Dimensions here are representative rather than surveyed. They are chosen to be
consistent with published descriptions of the site and with IRC:SP:35 bridge
inspection practice, and every figure is stated in the report as a modelling
assumption rather than as measured site data.

Frame convention: x downstream, y cross-stream (left bank positive), z up with
z = 0 at the flood water surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Scene, Box, Cylinder, Sphere, Heightfield
from .acoustics import MAT_INDEX


@dataclass
class Pier:
    """A bridge pier and the scour hole the flood cut around it."""

    y: float
    radius: float = 2.2
    scour_depth: float = 0.0     # metres below the surrounding bed
    scour_radius: float = 0.0
    intact: bool = True

    @property
    def x(self) -> float:
        return 0.0


@dataclass
class SiteConfig:
    """Site geometry and hydraulics."""

    name: str = "Savitri River, Mahad (representative)"
    x_min: float = -70.0
    x_max: float = 70.0
    y_min: float = -55.0
    y_max: float = 55.0
    grid_res: float = 0.5

    channel_half_width: float = 46.0
    bed_centre_depth: float = 13.5      # thalweg depth below surface
    bank_depth: float = 3.0
    downstream_slope: float = 0.004
    roughness: float = 0.16             # bed micro-relief, metres RMS

    surface_current: float = 2.4        # m/s at the surface, mid channel
    roughness_length: float = 0.02      # z0 for the log velocity profile

    ssc_g_per_l: float = 3.2            # suspended sediment, monsoon flood

    piers: tuple = (
        Pier(y=-24.0, scour_depth=1.6, scour_radius=7.0),
        Pier(y=-8.0, scour_depth=3.4, scour_radius=9.5),   # deepest scour
        Pier(y=8.0, scour_depth=2.1, scour_radius=7.5),
        Pier(y=24.0, scour_depth=1.2, scour_radius=6.0, intact=False),
    )
    seed: int = 20160802                # the date of the Mahad collapse


class DisasterSite:
    """Builds the ray-castable Scene, the bathymetry, and the current field."""

    def __init__(self, cfg: SiteConfig | None = None):
        self.cfg = cfg or SiteConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.targets: dict[str, dict] = {}
        self._build_bathymetry()
        self._build_scene()

    # ---------------------------------------------------------------- bathymetry
    def _build_bathymetry(self):
        c = self.cfg
        nx = int((c.x_max - c.x_min) / c.grid_res) + 1
        ny = int((c.y_max - c.y_min) / c.grid_res) + 1
        self.xs = c.x_min + np.arange(nx) * c.grid_res
        self.ys = c.y_min + np.arange(ny) * c.grid_res
        X, Y = np.meshgrid(self.xs, self.ys)

        # Parabolic channel cross-section, deepest at the centreline.
        u = np.clip(np.abs(Y) / c.channel_half_width, 0.0, 1.0)
        depth = c.bed_centre_depth - (c.bed_centre_depth - c.bank_depth) * u ** 2
        H = -depth
        # Gentle downstream fall.
        H = H - c.downstream_slope * (X - c.x_min)

        # Correlated micro-relief so the sonar sees plausible bed texture
        # rather than pure white noise.
        noise = self.rng.normal(0.0, 1.0, H.shape)
        for _ in range(4):
            noise = (noise
                     + np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
                     + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5.0
        noise /= max(noise.std(), 1e-9)
        H = H + c.roughness * noise

        # Scour cones around each pier. These are the quantity the inspection
        # mission is asked to measure, so they are generated from an explicit
        # analytic model and retained as ground truth.
        self.scour_truth = {}
        for i, p in enumerate(c.piers):
            if p.scour_depth <= 0:
                continue
            R = np.hypot(X - p.x, Y - p.y)
            inside = R < p.scour_radius
            # Inverted cone, flat-bottomed at the pier shaft.
            prof = np.clip((p.scour_radius - R) / max(p.scour_radius - p.radius, 1e-6), 0, 1)
            H = H - p.scour_depth * (prof ** 1.6) * inside
            self.scour_truth[f"P{i+1}"] = {
                "y": p.y, "depth": p.scour_depth, "radius": p.scour_radius,
                "volume": float(np.pi * p.scour_radius ** 2 * p.scour_depth / 3.0),
            }

        self.H = H
        self.bed = Heightfield(c.x_min, c.y_min, c.grid_res, c.grid_res, H,
                               MAT_INDEX["silt"], name="riverbed", max_range=90.0,
                               step=0.70, refine=18)

    def bed_height(self, x, y):
        return self.bed.height(np.asarray(x, dtype=float), np.asarray(y, dtype=float))

    # ---------------------------------------------------------------- scene
    def _build_scene(self):
        c = self.cfg
        self.scene = Scene([self.bed])

        # Pier shafts, from below the scoured bed up through the surface.
        for i, p in enumerate(c.piers):
            zb = float(self.bed_height(p.x, p.y)) - 2.0
            ztop = 2.0 if p.intact else -4.5
            self.scene.add(Cylinder([p.x, p.y], p.radius, zb, ztop,
                                    MAT_INDEX["concrete"], name=f"pier_P{i+1}"))

        # Collapsed deck span lying on the bed between P3 and P4.
        zc = float(self.bed_height(7.0, 17.0))
        self.scene.add(Box([7.0, 17.0, zc + 0.7], [7.5, 3.0, 0.7],
                           MAT_INDEX["concrete"], yaw=np.radians(17.0),
                           name="collapsed_span"))
        self._register("collapsed_span", "structure", (7.0, 17.0, zc + 0.7),
                       (15.0, 6.0, 1.4))

        # Fallen masonry blocks along the failure line.
        for k in range(9):
            bx = 2.0 + self.rng.uniform(-4, 12)
            by = 10.0 + self.rng.uniform(-9, 16)
            bz = float(self.bed_height(bx, by))
            s = self.rng.uniform(0.5, 1.3)
            self.scene.add(Box([bx, by, bz + s * 0.5], [s, s * 0.8, s * 0.5],
                               MAT_INDEX["rubble"], yaw=self.rng.uniform(0, np.pi),
                               name=f"rubble_{k}"))

        # The two state transport buses, swept downstream of the failed span.
        bus_specs = [(21.0, 6.5, np.radians(-28.0)), (34.0, -9.0, np.radians(64.0))]
        for k, (bx, by, yaw) in enumerate(bus_specs, start=1):
            bz = float(self.bed_height(bx, by))
            half = np.array([5.5, 1.3, 1.55])
            self.scene.add(Box([bx, by, bz + half[2]], half, MAT_INDEX["steel"],
                               yaw=yaw, name=f"bus_{k}"))
            self._register(f"bus_{k}", "vehicle_large", (bx, by, bz + half[2]),
                           tuple(2 * half))

        # A light passenger vehicle, the hardest target: small and part buried.
        cx, cy = 45.0, 3.0
        cz = float(self.bed_height(cx, cy))
        half = np.array([2.1, 0.9, 0.62])
        self.scene.add(Box([cx, cy, cz + half[2] * 0.75], half, MAT_INDEX["steel"],
                           yaw=np.radians(-12.0), name="car_1"))
        self._register("car_1", "vehicle_small", (cx, cy, cz + half[2] * 0.75),
                       tuple(2 * half))

        # Boulders transported by the flood.
        for k in range(14):
            bx = self.rng.uniform(c.x_min + 12, c.x_max - 12)
            by = self.rng.uniform(-34, 34)
            r = self.rng.uniform(0.35, 1.0)
            bz = float(self.bed_height(bx, by)) + r * 0.6
            self.scene.add(Sphere([bx, by, bz], r, MAT_INDEX["gravel"],
                                  name=f"boulder_{k}"))

    def _register(self, name, cls, centre, size):
        self.targets[name] = {"class": cls, "centre": np.array(centre, float),
                              "size": np.array(size, float)}

    # ---------------------------------------------------------------- hydraulics
    def current(self, x, y, z):
        """Depth-varying current, returned as a velocity vector in m/s.

        Uses the law of the wall for the vertical profile and a cosine lateral
        profile, with local acceleration in the constricted gaps between piers.
        Flow is predominantly downstream (+x).
        """
        c = self.cfg
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        z = np.asarray(z, float)

        bed = self.bed_height(x, y)
        h = np.maximum(-bed, 0.5)                 # local water column depth
        above = np.clip(z - bed, 0.02, None)      # height above bed

        # Log profile normalised so the surface value matches surface_current.
        prof = np.log(above / c.roughness_length) / np.log(h / c.roughness_length)
        prof = np.clip(prof, 0.0, 1.15)

        lateral = np.cos(np.clip(np.abs(y) / c.channel_half_width, 0, 1) * np.pi / 2) ** 0.5

        # Constriction between piers: flow speeds up in the gaps and stalls in
        # the immediate lee of each shaft.
        accel = np.ones_like(np.asarray(x, dtype=float))
        for p in c.piers:
            d = np.hypot(x - p.x, y - p.y)
            gap = np.exp(-((d - p.radius * 2.4) ** 2) / (2 * 3.0 ** 2))
            accel = accel + 0.22 * gap
            wake = (x > p.x) & (np.abs(y - p.y) < p.radius * 1.6) & (x - p.x < 12)
            accel = np.where(wake, accel * 0.35, accel)

        u = c.surface_current * prof * lateral * accel
        v = np.zeros_like(u)
        w = np.zeros_like(u)
        return np.stack([u, v, w], axis=-1)

    def current_at(self, pos):
        p = np.asarray(pos, float)
        return self.current(p[0], p[1], p[2]).reshape(3)

    # ---------------------------------------------------------------- helpers
    def summary(self) -> str:
        c = self.cfg
        lines = [
            f"site            : {c.name}",
            f"extent          : {c.x_max-c.x_min:.0f} m x {c.y_max-c.y_min:.0f} m"
            f" at {c.grid_res:.2f} m grid",
            f"depth           : {-self.H.min():.2f} m max, {-self.H.mean():.2f} m mean",
            f"surface current : {c.surface_current:.2f} m/s",
            f"turbidity       : {c.ssc_g_per_l:.2f} g/L suspended sediment",
            f"primitives      : {len(self.scene.primitives)}",
            f"targets         : {len(self.targets)}",
        ]
        for k, v in self.scour_truth.items():
            lines.append(f"  scour {k}      : {v['depth']:.2f} m deep, "
                         f"{v['radius']:.1f} m radius, {v['volume']:.1f} m3")
        return "\n".join(lines)
