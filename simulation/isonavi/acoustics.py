"""Physics-based forward-looking sonar (FLS) image formation.

The model follows the active sonar equation applied per range/bearing cell.
For a resolution cell at slant range r insonified by beam b, the received
echo level is

    EL(r, b) = SL - 2 TL(r) + BS(r, b) + 10 log10(A_cell)

with

    TL(r)  = 20 log10(r) + alpha r          spherical spreading + absorption
    BS     = mu_material + 10 n log10(cos i) Lambert-like angular dependence
    alpha  = alpha_water(f) + alpha_sediment(C)

Three properties of real FLS imagery are reproduced explicitly, because they
are what a perception model actually keys on:

1. Acoustic shadow. Ray casting resolves occlusion, so every object casts a
   dark shadow whose length encodes its height. Shadow geometry is the single
   most informative cue in FLS target recognition.
2. Elevation ambiguity. An FLS collapses the vertical aperture: all scatterers
   within the vertical beamwidth at a given slant range fold into one pixel.
   We reproduce this by casting several elevation rays per beam and summing.
3. Speckle. Coherent imaging gives Rayleigh-distributed amplitude, so the
   intensity is Gamma distributed with shape equal to the number of looks.

References for the functional forms are given in docs/. Absorption uses the
pure-water term of the Francois and Garrison formulation; the sediment term is
an engineering approximation appropriate to monsoon flood turbidity and is
flagged as such in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .geometry import Scene

# --------------------------------------------------------------------------
# Acoustic materials
# --------------------------------------------------------------------------
# mu     : Lambert backscatter coefficient (dB), the strength at normal incidence
# n_exp  : angular exponent, higher means more directional (smoother) scattering
# spec   : specular fraction, raises return near normal incidence for hard flat faces

@dataclass(frozen=True)
class Material:
    name: str
    mu: float
    n_exp: float = 1.0
    spec: float = 0.0


MATERIALS = (
    Material("water", -99.0, 1.0, 0.0),
    Material("silt", -29.0, 1.0, 0.0),          # soft mud, weak diffuse scatterer
    Material("sand", -25.0, 1.0, 0.0),
    Material("gravel", -19.0, 1.2, 0.05),
    Material("concrete", -13.0, 1.5, 0.35),     # pier shafts, deck slabs
    Material("steel", -8.0, 2.0, 0.65),         # vehicle bodies, rebar
    Material("rubble", -16.0, 1.1, 0.10),
    Material("vegetation", -24.0, 0.8, 0.0),
)
MAT_INDEX = {m.name: i for i, m in enumerate(MATERIALS)}
MU = np.array([m.mu for m in MATERIALS])
NEXP = np.array([m.n_exp for m in MATERIALS])
SPEC = np.array([m.spec for m in MATERIALS])


def absorption_db_per_m(freq_khz: float, temp_c: float = 25.0,
                        depth_m: float = 10.0, ssc_g_per_l: float = 0.0) -> float:
    """Acoustic absorption in dB/m.

    Pure-water viscous term from Francois and Garrison, valid into the MHz
    band that imaging sonars occupy. The sediment term models excess
    attenuation from suspended solids in flood water and is a linear
    engineering fit, not a first-principles result.
    """
    f = float(freq_khz)
    alpha_w_db_km = 4.9e-4 * f * f * np.exp(-(temp_c / 27.0 + depth_m / 17.0))
    alpha_w = alpha_w_db_km / 1000.0
    # Suspended sediment: scattering loss grows with concentration and frequency.
    alpha_s = 0.014 * ssc_g_per_l * (f / 1000.0) ** 0.5
    return float(alpha_w + alpha_s)


# --------------------------------------------------------------------------
# Sonar configuration
# --------------------------------------------------------------------------

@dataclass
class SonarConfig:
    """Geometry and signal parameters of an imaging sonar head."""

    name: str = "generic"
    n_beams: int = 128
    fov_h_deg: float = 30.0
    fov_v_deg: float = 14.0
    n_elev_rays: int = 48          # elevation samples folded into each beam
    r_min: float = 0.7
    r_max: float = 15.0
    n_range_bins: int = 512
    freq_khz: float = 1800.0
    source_level_db: float = 210.0
    noise_floor_db: float = 35.0
    beamwidth_h_deg: float = 0.25  # -3 dB horizontal beamwidth
    sidelobe_db: float = -25.0
    n_looks: float = 1.6           # speckle averaging, higher is smoother
    temp_c: float = 25.0
    ssc_g_per_l: float = 0.0       # suspended sediment concentration
    dynamic_range_db: float = 45.0
    range_resolution_m: float = 0.04   # c*tau/2, sets the range point spread
    seed: int | None = None

    @property
    def alpha(self) -> float:
        return absorption_db_per_m(self.freq_khz, self.temp_c,
                                   10.0, self.ssc_g_per_l)

    @property
    def range_bins(self):
        return np.linspace(self.r_min, self.r_max, self.n_range_bins)

    @property
    def bearings(self):
        half = np.radians(self.fov_h_deg) / 2.0
        return np.linspace(-half, half, self.n_beams)


# Presets for two real heads. ARIS matches the domain of the public FLS
# training data; Oculus is the wide-swath head used for area search.
ARIS_EXPLORER_3000 = SonarConfig(
    name="ARIS Explorer 3000", n_beams=128, fov_h_deg=30.0, fov_v_deg=14.0,
    r_min=0.7, r_max=15.0, n_range_bins=512, freq_khz=1800.0,
    beamwidth_h_deg=0.25, n_looks=1.6, n_elev_rays=56,
    range_resolution_m=0.02,
)

OCULUS_M750D = SonarConfig(
    name="Oculus M750d", n_beams=256, fov_h_deg=130.0, fov_v_deg=20.0,
    r_min=1.0, r_max=60.0, n_range_bins=640, freq_khz=750.0,
    beamwidth_h_deg=0.6, n_looks=1.8, n_elev_rays=44,
    range_resolution_m=0.09,
)


def preset(name: str, **overrides) -> SonarConfig:
    table = {"aris": ARIS_EXPLORER_3000, "oculus": OCULUS_M750D}
    key = name.lower().split()[0]
    if key not in table:
        raise KeyError(f"unknown sonar preset {name!r}, have {sorted(table)}")
    return replace(table[key], **overrides) if overrides else replace(table[key])


# --------------------------------------------------------------------------
# Image formation
# --------------------------------------------------------------------------

@dataclass
class SonarFrame:
    """One sonar ping."""

    polar: np.ndarray                    # (n_range_bins, n_beams), dB
    ranges: np.ndarray
    bearings: np.ndarray
    pose: np.ndarray                     # (x, y, z, roll, pitch, yaw)
    t: float = 0.0
    config: SonarConfig | None = None
    hit_range: np.ndarray | None = None   # (n_beams,) nearest geometric range
    hit_material: np.ndarray | None = None
    hit_point: np.ndarray | None = None   # (n_beams, 3) world coordinates
    hit_incidence: np.ndarray | None = None
    # Every ray intersection in the fan, not just the nearest return per beam.
    # A bottom-detection stage on a real multibeam extracts returns across the
    # whole swath, so this is what the mapper should consume: the nearest
    # return alone samples only the leading edge of the insonified band.
    all_point: np.ndarray | None = None    # (n_valid, 3)
    all_incidence: np.ndarray | None = None
    all_material: np.ndarray | None = None

    def normalised(self) -> np.ndarray:
        """Map dB to [0, 1] over the configured dynamic range."""
        dr = self.config.dynamic_range_db if self.config else 45.0
        top = np.percentile(self.polar, 99.5)
        lo = top - dr
        return np.clip((self.polar - lo) / max(dr, 1e-6), 0.0, 1.0)

    def to_cartesian(self, size=512):
        """Fan-shaped Cartesian rendering, the usual operator view.

        The vertical axis spans the actual range window rather than starting
        at zero. Including the blind zone in front of the head would leave a
        large empty wedge that carries no information and, when these frames
        are used as training data, would let a network separate them from real
        captures on layout alone.
        """
        img = self.normalised()
        half = self.bearings.max()
        r0, r1 = float(self.ranges[0]), float(self.ranges[-1])
        xs = np.linspace(-r1 * np.sin(half), r1 * np.sin(half), size)
        ys = np.linspace(r0, r1, size)
        X, Y = np.meshgrid(xs, ys)
        R = np.hypot(X, Y)
        B = np.arctan2(X, Y)
        ri = (R - self.ranges[0]) / (self.ranges[-1] - self.ranges[0]) * (len(self.ranges) - 1)
        bi = (B - self.bearings[0]) / (self.bearings[-1] - self.bearings[0]) * (len(self.bearings) - 1)
        ok = ((ri >= 0) & (ri <= len(self.ranges) - 1)
              & (bi >= 0) & (bi <= len(self.bearings) - 1))
        out = np.zeros_like(R)
        rr = np.clip(ri.astype(int), 0, len(self.ranges) - 1)
        bb = np.clip(bi.astype(int), 0, len(self.bearings) - 1)
        out[ok] = img[rr[ok], bb[ok]]
        return np.flipud(out)


def _euler_to_R(roll, pitch, yaw):
    """ZYX Euler angles to a rotation matrix, ROS REP-103 convention.

    Body frame is x forward, y left, z up. Rotations follow the right-hand
    rule, so a positive pitch about +y tilts the nose down. Callers wanting a
    downward-looking sonar therefore pass a positive pitch.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


class ForwardLookingSonar:
    """Renders sonar frames by ray casting a Scene.

    One ping costs n_beams * n_elev_rays ray casts. At the default ARIS
    settings that is 128 * 9 = 1152 rays, which NumPy evaluates in a few
    milliseconds against analytic primitives.
    """

    def __init__(self, config: SonarConfig, scene: Scene):
        self.cfg = config
        self.scene = scene
        self.rng = np.random.default_rng(config.seed)
        self._build_fan()

    def _build_fan(self, jitter=True):
        """Lay out the ray fan.

        Elevation samples are jittered by up to half a sample spacing on every
        ping. With a deterministic fan the discrete elevation samples produce
        visible terracing in the seabed return, because each ray paints a fixed
        range interval. Re-jittering decorrelates that structure between pings
        and lets it average out, which is what makes the imagery resemble a
        continuously insonified surface.
        """
        cfg = self.cfg
        self.bearings = cfg.bearings
        half_v = np.radians(cfg.fov_v_deg) / 2.0
        base = np.linspace(-half_v, half_v, cfg.n_elev_rays)
        if jitter and cfg.n_elev_rays > 1:
            spacing = base[1] - base[0]
            base = base + self.rng.uniform(-0.5, 0.5, base.shape) * spacing
            base = np.clip(np.sort(base), -half_v, half_v)
        self.elevs = base
        # Vertical beam pattern weight, parabolic in dB with a sidelobe floor.
        rel = self.elevs / max(half_v, 1e-9)
        w_db = np.maximum(-12.0 * rel ** 2, cfg.sidelobe_db)
        self.elev_w = 10 ** (w_db / 10.0)
        self.elev_w /= self.elev_w.sum()
        B, E = np.meshgrid(self.bearings, self.elevs, indexing="ij")
        self.fan_b = B.ravel()
        self.fan_e = E.ravel()
        # Ray directions in the sensor frame: x forward, y port, z up.
        self.dirs_local = np.stack([
            np.cos(self.fan_e) * np.cos(self.fan_b),
            np.cos(self.fan_e) * np.sin(self.fan_b),
            np.sin(self.fan_e),
        ], axis=1)
        self.ranges = cfg.range_bins
        self._dr = self.ranges[1] - self.ranges[0]

    def ping(self, pose, t: float = 0.0) -> SonarFrame:
        """Render one ping from a 6-DOF pose (x, y, z, roll, pitch, yaw)."""
        cfg = self.cfg
        self._build_fan(jitter=True)
        pose = np.asarray(pose, dtype=np.float64)
        R = _euler_to_R(pose[3], pose[4], pose[5])
        dirs = self.dirs_local @ R.T
        origins = np.broadcast_to(pose[:3], dirs.shape).copy()

        hit = self.scene.intersect(origins, dirs)
        r = hit.t
        valid = np.isfinite(r) & (r >= cfg.r_min) & (r <= cfg.r_max)

        # Incidence angle between the ray and the surface normal.
        cos_i = np.zeros_like(r)
        if np.any(valid):
            cos_i[valid] = np.abs(np.einsum(
                "ij,ij->i", dirs[valid], hit.normal[valid]))
            cos_i = np.clip(cos_i, 1e-4, 1.0)

        mat = np.clip(hit.material, 0, len(MATERIALS) - 1)
        mu = MU[mat]
        nexp = NEXP[mat]
        spec = SPEC[mat]

        # Backscattering strength: Lambert diffuse plus a specular lobe.
        bs = mu + 10.0 * nexp * np.log10(np.maximum(cos_i, 1e-4))
        specular = 10.0 * np.log10(1.0 + spec * cos_i ** 12 * 40.0)
        bs = bs + specular

        # Spatially coherent variation in bed composition. This is a property
        # of the seabed rather than of the sensor, so it is a deterministic
        # function of world position: the same patch looks the same from any
        # viewpoint, which matters once frames are fused into a map.
        pts = origins + np.where(valid, r, 0.0)[:, None] * dirs
        bs = bs + np.where(valid, self._patchiness(pts), 0.0)

        alpha = cfg.alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            tl = 20.0 * np.log10(np.maximum(r, 1e-3)) + alpha * r
        el = cfg.source_level_db - 2.0 * tl + bs

        # Accumulate into (range, bearing) bins in the linear domain, weighting
        # each elevation ray by the vertical beam pattern. This is where the
        # elevation ambiguity of a real FLS is introduced.
        lin = np.where(valid, 10 ** (el / 10.0), 0.0)
        w = np.tile(self.elev_w, cfg.n_beams)
        acc = self._scatter_footprint(r, lin * w, valid, cos_i)

        # Finite transmit pulse: the range response is the scene convolved with
        # the pulse envelope, which sets the true range resolution.
        acc = self._apply_pulse(acc)

        # Horizontal beam spreading: neighbouring beams leak into each other.
        acc = self._apply_beam_pattern(acc)

        # Volume reverberation from suspended sediment, range dependent.
        acc += self._volume_reverberation()

        # Speckle: Gamma(L, 1/L) multiplicative, mean preserving.
        L = max(cfg.n_looks, 1.0)
        acc *= self.rng.gamma(shape=L, scale=1.0 / L, size=acc.shape)

        # Additive receiver noise, then time-varying gain.
        noise = 10 ** (cfg.noise_floor_db / 10.0)
        acc += self.rng.exponential(noise, size=acc.shape)
        db = 10.0 * np.log10(np.maximum(acc, 1e-12))
        db += self._tvg()

        # Per-beam nearest return, used by the mapping and scour estimator.
        vb = valid.reshape(cfg.n_beams, cfg.n_elev_rays)
        rr = np.where(vb, r.reshape(cfg.n_beams, cfg.n_elev_rays), np.inf)
        mm = hit.material.reshape(cfg.n_beams, cfg.n_elev_rays)
        k = np.argmin(rr, axis=1)
        rows = np.arange(cfg.n_beams)
        best = rr[rows, k]
        good = np.isfinite(best)
        hr = np.where(good, best, np.nan)
        hm = np.where(good, mm[rows, k], -1).astype(np.int32)

        # World coordinates of the nearest return on each beam. These are the
        # soundings the mapping layer accumulates into a bathymetric grid.
        flat = rows * cfg.n_elev_rays + k
        hp = np.full((cfg.n_beams, 3), np.nan)
        hp[good] = (origins[flat[good]]
                    + best[good, None] * dirs[flat[good]])
        hi_ang = np.full(cfg.n_beams, np.nan)
        hi_ang[good] = np.degrees(np.arccos(np.clip(cos_i[flat[good]], 0, 1)))

        pts_all = origins[valid] + r[valid, None] * dirs[valid]
        inc_all = np.degrees(np.arccos(np.clip(cos_i[valid], 0, 1)))
        return SonarFrame(polar=db, ranges=self.ranges, bearings=self.bearings,
                          pose=pose.copy(), t=t, config=cfg,
                          hit_range=hr, hit_material=hm,
                          hit_point=hp, hit_incidence=hi_ang,
                          all_point=pts_all, all_incidence=inc_all,
                          all_material=hit.material[valid])

    # Octaves for the seabed patchiness field: (wavelength m, amplitude dB).
    _PATCH_OCTAVES = ((11.0, 2.6), (3.7, 1.7), (1.3, 1.1), (0.55, 0.7))

    def _patchiness(self, pts):
        """Spatially correlated backscatter anomaly in dB.

        Value noise built from a small sum of sinusoids at decreasing
        wavelength. Cheap, smooth, and a pure function of world position.
        """
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        out = np.zeros(len(pts))
        for k, (lam, amp) in enumerate(self._PATCH_OCTAVES):
            w = 2.0 * np.pi / lam
            ph = 1.7 * k + 0.9
            out += amp * (np.sin(w * x + ph) * np.cos(w * 0.83 * y - ph)
                          + 0.6 * np.sin(w * 0.61 * (x + y) + 2.1 * ph)
                          + 0.4 * np.cos(w * 0.47 * z + ph))
        return out * 0.5

    def _scatter_footprint(self, r, energy, valid, cos_i):
        """Deposit each ray's energy across the range extent of its footprint.

        A single elevation ray does not sample a point: it represents a solid
        angle whose intersection with a surface spans a finite range interval.
        Depositing into one bin would leave the seabed return sparse and let
        the noise floor dominate between samples, which destroys the shadow
        contrast that FLS interpretation depends on.

        The footprint length is computed from local geometry alone. For an
        elevation beam of angular width dtheta striking a surface at grazing
        angle g, the along-range extent is

            L = r dtheta cot(g),    sin(g) = |d . n| = cos_i

        Deriving L locally rather than from neighbouring rays matters at object
        edges: there the neighbouring ray may strike a surface tens of metres
        further away, and a midpoint rule would smear that ray's energy across
        the very shadow the object is casting.
        """
        cfg = self.cfg
        nb, ne = cfg.n_beams, cfg.n_elev_rays
        dtheta = np.radians(cfg.fov_v_deg) / max(ne - 1, 1)

        # Analytic footprint from local surface geometry. On a locally planar
        # surface this equals the range spacing between adjacent elevation
        # rays, so the fan tiles the range axis exactly.
        c = np.clip(cos_i, 1e-3, 1.0)
        cot_g = np.sqrt(np.maximum(1.0 - c * c, 0.0)) / c
        L_geom = np.nan_to_num(r * dtheta * cot_g, nan=self._dr, posinf=self._dr)
        L_geom = np.clip(L_geom, self._dr, 8.0).reshape(nb, ne)

        E = energy.reshape(nb, ne)
        V = valid.reshape(nb, ne)
        R = r.reshape(nb, ne)
        Rf = np.where(V, R, np.nan)

        # Observed spacing to the elevation neighbours. This tracks the actual
        # sampling, including the per-ping jitter, so it leaves no gaps.
        span_obs = np.full_like(L_geom, np.nan)
        if ne > 1:
            d = np.abs(np.diff(Rf, axis=1))
            span_obs[:, 1:-1] = 0.5 * (d[:, :-1] + d[:, 1:])
            span_obs[:, 0] = d[:, 0]
            span_obs[:, -1] = d[:, -1]
        span_obs = np.nan_to_num(span_obs, nan=0.0, posinf=0.0)

        # Trust the observed spacing, but never beyond a few times what the
        # local geometry allows. That cap is what prevents a neighbour lying on
        # a far surface, across an object edge, from smearing this ray's energy
        # into the shadow the object is casting.
        span = np.clip(span_obs, self._dr, 3.0 * L_geom + self._dr)
        span = np.maximum(span, 0.8 * L_geom) * 1.15  # slight overlap
        # Keep the deposit centred on the true hit range, not on the midpoints.
        r_safe = np.where(V & np.isfinite(R), R, cfg.r_min)
        lo = r_safe - 0.5 * span
        hi = r_safe + 0.5 * span

        i0 = np.ceil((lo - cfg.r_min) / self._dr).astype(int)
        i1 = np.floor((hi - cfg.r_min) / self._dr).astype(int)
        i0 = np.clip(i0, 0, cfg.n_range_bins - 1)
        i1 = np.clip(i1, 0, cfg.n_range_bins - 1)
        nbins = np.maximum(i1 - i0 + 1, 1)

        good = V & np.isfinite(Rf) & (E > 0)
        per_bin = np.where(good, E / nbins, 0.0)

        # Interval scatter via a difference array plus a cumulative sum.
        diff = np.zeros((cfg.n_range_bins + 1, nb))
        beam_col = np.repeat(np.arange(nb), ne)
        np.add.at(diff, (i0.ravel()[good.ravel()], beam_col[good.ravel()]),
                  per_bin.ravel()[good.ravel()])
        np.add.at(diff, (i1.ravel()[good.ravel()] + 1, beam_col[good.ravel()]),
                  -per_bin.ravel()[good.ravel()])
        acc = np.cumsum(diff[:-1], axis=0)
        return np.maximum(acc, 0.0)

    def _apply_pulse(self, acc):
        """Convolve along range with the transmit pulse envelope.

        A sonar cannot resolve range more finely than c*tau/2 for a pulse of
        duration tau. Applying that envelope is both the correct image
        formation step and what removes the residual banding left by discrete
        elevation sampling.
        """
        cfg = self.cfg
        sigma_bins = max(cfg.range_resolution_m / max(self._dr, 1e-9), 0.6)
        half = int(np.ceil(3.0 * sigma_bins))
        if half < 1:
            return acc
        k = np.arange(-half, half + 1)
        w = np.exp(-0.5 * (k / sigma_bins) ** 2)
        w /= w.sum()
        padded = np.pad(acc, ((half, half), (0, 0)), mode="edge")
        out = np.empty_like(acc)
        for b in range(acc.shape[1]):
            out[:, b] = np.convolve(padded[:, b], w, mode="valid")
        return out

    def _apply_beam_pattern(self, acc):
        """Convolve across bearing with the horizontal beam response."""
        cfg = self.cfg
        dtheta = np.degrees(self.bearings[1] - self.bearings[0]) if cfg.n_beams > 1 else 1.0
        half_w = max(int(round(2.0 * cfg.beamwidth_h_deg / max(dtheta, 1e-6))), 1)
        k = np.arange(-half_w, half_w + 1) * dtheta
        w_db = np.maximum(-12.0 * (k / max(cfg.beamwidth_h_deg, 1e-6)) ** 2,
                          cfg.sidelobe_db)
        w = 10 ** (w_db / 10.0)
        w /= w.sum()
        if len(w) <= 1:
            return acc
        pad = len(w) // 2
        padded = np.pad(acc, ((0, 0), (pad, pad)), mode="edge")
        out = np.empty_like(acc)
        for i in range(acc.shape[0]):
            out[i] = np.convolve(padded[i], w, mode="valid")
        return out

    def _volume_reverberation(self):
        """Backscatter from suspended sediment in the water column."""
        cfg = self.cfg
        if cfg.ssc_g_per_l <= 0:
            return 0.0
        r = self.ranges[:, None]
        sv = -70.0 + 10.0 * np.log10(max(cfg.ssc_g_per_l, 1e-6))
        tl = 20.0 * np.log10(np.maximum(r, 1e-3)) + cfg.alpha * r
        rev_db = cfg.source_level_db - 2.0 * tl + sv + 10.0 * np.log10(self._dr)
        return 10 ** (rev_db / 10.0)

    def _tvg(self):
        """Time-varying gain, compensating spreading and absorption."""
        r = self.ranges[:, None]
        return 20.0 * np.log10(np.maximum(r, 1e-3)) + 2.0 * self.cfg.alpha * r
