"""Navigation sensor models.

Underwater there is no GPS, so the pose estimate is built from a Doppler
velocity log, an inertial unit, and a pressure depth cell. Each model here
reproduces the error behaviour that actually limits the resulting navigation
solution:

  DVL     scale-factor error, per-beam noise, and loss of bottom lock outside
          its altitude envelope or over soft, absorbing sediment
  IMU     white noise plus a slowly drifting gyro bias, which is what makes
          heading drift unbounded without an external reference
  Depth   near-exact in absolute terms, so it fully observes the z channel

The DVL dropout model matters for this scenario specifically. The research
identifies loss of bottom lock over silt as a principal failure mode, and the
mission logic has to cope with it rather than assume continuous lock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import rot_body_to_world


@dataclass
class DVLConfig:
    """Doppler velocity log, Water Linked A50 class."""

    name: str = "DVL-A50 class"
    rate_hz: float = 8.0
    min_altitude: float = 0.05
    max_altitude: float = 50.0
    velocity_sigma: float = 0.004       # m/s per axis, at 1 sigma
    scale_error: float = 0.005          # 0.5 percent long-term scale error
    altitude_sigma: float = 0.01
    # Probability of losing bottom lock per sample over highly absorbing silt.
    silt_dropout_prob: float = 0.04
    beam_angle_deg: float = 22.5


@dataclass
class IMUConfig:
    rate_hz: float = 100.0
    gyro_sigma: float = 0.0017          # rad/s
    gyro_bias_walk: float = 2.0e-5      # rad/s per sqrt(s)
    accel_sigma: float = 0.02           # m/s^2
    attitude_sigma: float = 0.0035      # rad, roll and pitch from gravity
    heading_sigma: float = 0.025        # rad, magnetic heading


@dataclass
class DepthConfig:
    rate_hz: float = 20.0
    sigma: float = 0.012                # m


class DVL:
    """Bottom-lock velocity and altitude."""

    def __init__(self, cfg: DVLConfig | None = None, site=None, rng=None):
        self.cfg = cfg or DVLConfig()
        self.site = site
        self.rng = rng or np.random.default_rng(0)
        self.scale = 1.0 + self.rng.normal(0.0, self.cfg.scale_error)
        self.locked = True

    def altitude(self, eta):
        if self.site is None:
            return 5.0
        bed = float(self.site.bed_height(eta[0], eta[1]))
        return float(eta[2] - bed)

    def measure(self, eta, nu):
        """Returns (velocity_body, altitude, locked)."""
        c = self.cfg
        alt = self.altitude(eta)
        in_envelope = c.min_altitude <= alt <= c.max_altitude
        # Bottom lock is probabilistic over soft sediment even inside envelope.
        drop = self.rng.random() < c.silt_dropout_prob
        # Extreme attitude tilts the beams off the bed.
        tilted = abs(eta[3]) > np.radians(35) or abs(eta[4]) > np.radians(35)
        self.locked = bool(in_envelope and not drop and not tilted)
        if not self.locked:
            return None, alt, False
        v = nu[:3] * self.scale + self.rng.normal(0.0, c.velocity_sigma, 3)
        a = alt + self.rng.normal(0.0, c.altitude_sigma)
        return v, a, True


class IMU:
    """Rate gyro, accelerometer, and derived attitude."""

    def __init__(self, cfg: IMUConfig | None = None, rng=None):
        self.cfg = cfg or IMUConfig()
        self.rng = rng or np.random.default_rng(1)
        self.bias = self.rng.normal(0.0, 0.002, 3)

    def measure(self, eta, nu, dt):
        c = self.cfg
        self.bias = self.bias + self.rng.normal(
            0.0, c.gyro_bias_walk * np.sqrt(max(dt, 1e-9)), 3)
        gyro = nu[3:] + self.bias + self.rng.normal(0.0, c.gyro_sigma, 3)
        att = eta[3:].copy()
        att[0] += self.rng.normal(0.0, c.attitude_sigma)
        att[1] += self.rng.normal(0.0, c.attitude_sigma)
        att[2] += self.rng.normal(0.0, c.heading_sigma)
        return gyro, att


class DepthCell:
    def __init__(self, cfg: DepthConfig | None = None, rng=None):
        self.cfg = cfg or DepthConfig()
        self.rng = rng or np.random.default_rng(2)

    def measure(self, eta):
        return eta[2] + self.rng.normal(0.0, self.cfg.sigma)


class SensorSuite:
    """Bundles the navigation sensors and handles their differing rates."""

    def __init__(self, site=None, seed=0):
        rng = np.random.default_rng(seed)
        self.dvl = DVL(site=site, rng=rng)
        self.imu = IMU(rng=rng)
        self.depth = DepthCell(rng=rng)
        self._next = {"dvl": 0.0, "imu": 0.0, "depth": 0.0}
        self.dropout_samples = 0
        self.total_samples = 0

    def sample(self, t, eta, nu, dt):
        """Returns whichever measurements are due at time t."""
        out = {}
        if t >= self._next["imu"]:
            out["imu"] = self.imu.measure(eta, nu, dt)
            self._next["imu"] = t + 1.0 / self.imu.cfg.rate_hz
        if t >= self._next["dvl"]:
            v, alt, lock = self.dvl.measure(eta, nu)
            out["dvl"] = (v, alt, lock)
            self.total_samples += 1
            if not lock:
                self.dropout_samples += 1
            self._next["dvl"] = t + 1.0 / self.dvl.cfg.rate_hz
        if t >= self._next["depth"]:
            out["depth"] = self.depth.measure(eta)
            self._next["depth"] = t + 1.0 / self.depth.cfg.rate_hz
        return out

    @property
    def dvl_availability(self):
        if self.total_samples == 0:
            return 1.0
        return 1.0 - self.dropout_samples / self.total_samples
