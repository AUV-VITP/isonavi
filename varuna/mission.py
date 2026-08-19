"""Autonomous mission execution: state machine, path planning, and the
closed simulation loop that drives every other module.

Mission phases

    DEPLOY      descend from the surface to survey altitude
    ACQUIRE     wait for DVL bottom lock and for the filter to settle
    TRANSIT     run upstream to the start of the survey box
    SEARCH      lawnmower the box, sounding the bed and screening for targets
    INSPECT     orbit each pier at fixed standoff to measure scour
    RETURN      transit back to the launch point
    REPORT      emit the mission product

Two constraints from the vehicle analysis shape the planner:

1. The hull is only directionally stable flying nose-first, and at the design
   current a broadside turn is unrecoverable. Every commanded heading
   therefore points into the flow, and survey legs run along the flow axis
   rather than across it.

2. Altitude above the bed, not depth below the surface, is what fixes the
   sonar grazing geometry. Legs are flown on altitude hold, which also keeps
   the vehicle inside the DVL bottom-lock envelope over uneven bathymetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import VehicleDynamics, VARUNA_1, rot_body_to_world
from .sensors import SensorSuite
from .estimation import NavigationEKF
from .control import PoseController, VARUNA_GAINS, CurrentEstimator
from .acoustics import ForwardLookingSonar, preset
from .mapping import BathymetryMap, TargetTracker, estimate_scour

PHASES = ("DEPLOY", "ACQUIRE", "TRANSIT", "SEARCH", "INSPECT", "RETURN", "REPORT", "DONE")


def lawnmower(x0, x1, y0, y1, spacing, along_flow=True):
    """Boustrophedon coverage path.

    Legs run along x, the flow axis, so the vehicle never has to hold a
    broadside attitude in the current for longer than the turn at each end.
    """
    wps = []
    ys = np.arange(y0, y1 + 1e-9, spacing)
    for k, y in enumerate(ys):
        leg = [(x1, y), (x0, y)] if k % 2 else [(x0, y), (x1, y)]
        wps.extend(leg)
    return [np.array([p[0], p[1]]) for p in wps]


def orbit(centre, radius, n=16, start_angle=0.0):
    """Circular standoff path around a structure."""
    a = start_angle + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [np.array([centre[0] + radius * np.cos(t),
                      centre[1] + radius * np.sin(t)]) for t in a]


@dataclass
class MissionConfig:
    survey_altitude: float = 3.2       # m above bed
    inspect_altitude: float = 2.6
    lawnmower_spacing: float = 9.0
    search_box: tuple = (-6.0, 52.0, -16.0, 20.0)   # x0, x1, y0, y1
    pier_standoff: float = 8.0
    orbit_points: int = 14
    waypoint_radius: float = 2.0
    sonar_rate_hz: float = 2.0
    dt: float = 0.05
    max_time: float = 1500.0
    launch: tuple = (-46.0, -2.0, -3.0)
    detect_range: float = 34.0
    piers_to_inspect: tuple = ("P2", "P3")


class MissionRunner:
    """Runs the full closed loop and records everything needed for evaluation."""

    def __init__(self, site, cfg: MissionConfig | None = None,
                 params=VARUNA_1, gains=VARUNA_GAINS, sonar=None,
                 detector=None, seed=0):
        self.site = site
        self.cfg = cfg or MissionConfig()
        self.params = params
        self.detector = detector
        self.rng = np.random.default_rng(seed)

        self.dyn = VehicleDynamics(params, current_fn=site.current_at)
        launch = np.array(self.cfg.launch, float)
        yaw0 = CurrentEstimator.heading_into_flow(site.current_at(launch)) or np.pi
        self.dyn.reset(eta=[*launch, 0, 0, yaw0])

        self.sens = SensorSuite(site=site, seed=seed + 11)
        self.ekf = NavigationEKF(p0=launch)
        self.ekf.x[6:9] = np.array([0.0, 0.0, yaw0])
        self.ctl = PoseController(gains, params=params)
        self.cest = CurrentEstimator(params.quad_damp, params.lin_damp)

        self.sonar_cfg = sonar or preset("oculus", seed=seed + 3, r_max=45.0,
                                         ssc_g_per_l=site.cfg.ssc_g_per_l)
        self.fls = ForwardLookingSonar(self.sonar_cfg, site.scene)

        c = self.cfg
        self.bmap = BathymetryMap(site.cfg.x_min, site.cfg.x_max,
                                  site.cfg.y_min, site.cfg.y_max, res=0.5)
        self.tracker = TargetTracker()

        self.phase = "DEPLOY"
        self.t = 0.0
        self.wps: list = []
        self.wi = 0
        self.pier_queue = list(c.piers_to_inspect)
        self.current_pier = None
        self.log = {k: [] for k in
                    ("t", "phase", "eta", "nu", "est_pos", "est_att", "sigma",
                     "alt", "dvl_lock", "current", "target_pos", "thrust")}
        self.frames = []
        self.events = []
        self.scour_results = {}
        self._next_ping = 0.0
        self._phase_t0 = 0.0

    # ------------------------------------------------------------------ helpers
    def _pier_xy(self, name):
        idx = int(name[1:]) - 1
        p = self.site.cfg.piers[idx]
        return np.array([p.x, p.y])

    def _alt_target_z(self, xy, altitude):
        """Depth setpoint for a given altitude above the bed.

        The bed is sampled beneath the vehicle, not beneath the distant
        waypoint, so the vehicle terrain-follows instead of holding a fixed
        depth that slowly drifts out of the DVL envelope as the bathymetry
        changes. On the real vehicle this comes from the DVL altimeter; here
        it is read from the site, which is equivalent given that altitude is
        directly measured rather than estimated.
        """
        return float(self.site.bed_height(xy[0], xy[1])) + altitude

    def _depth_setpoint(self, altitude):
        here = self.ekf.position[:2]
        return self._alt_target_z(here, altitude)

    def _event(self, msg):
        self.events.append({"t": round(self.t, 2), "phase": self.phase, "msg": msg})

    def _set_phase(self, ph):
        self.phase = ph
        self._phase_t0 = self.t
        self._event(f"phase -> {ph}")

    # ------------------------------------------------------------------ planning
    def _plan_search(self):
        c = self.cfg
        x0, x1, y0, y1 = c.search_box
        pts = lawnmower(x0, x1, y0, y1, c.lawnmower_spacing)
        self.wps = pts
        self.wi = 0
        self._event(f"search plan: {len(pts)} waypoints, "
                    f"{c.lawnmower_spacing:.1f} m line spacing")

    def _plan_orbit(self, pier):
        c = self.cfg
        centre = self._pier_xy(pier)
        self.wps = orbit(centre, c.pier_standoff, c.orbit_points)
        self.wi = 0
        self.current_pier = pier
        self._event(f"inspect {pier}: orbit r={c.pier_standoff:.1f} m, "
                    f"{c.orbit_points} stations")

    # ------------------------------------------------------------------ perception
    def _perceive(self, frame):
        """Turn a sonar frame into world-frame detections."""
        dets = []
        if self.detector is not None:
            dets = self.detector(frame, self.ekf)
        return dets

    # ------------------------------------------------------------------ step
    def _target_for_phase(self):
        c = self.cfg
        est = self.ekf.position
        if self.phase in ("DEPLOY", "ACQUIRE"):
            xy = est[:2]
            return np.array([xy[0], xy[1], self._depth_setpoint(c.survey_altitude)])
        if self.phase in ("SEARCH", "INSPECT", "TRANSIT"):
            if self.wi < len(self.wps):
                xy = self.wps[self.wi]
            else:
                xy = est[:2]
            alt = c.inspect_altitude if self.phase == "INSPECT" else c.survey_altitude
            return np.array([xy[0], xy[1], self._depth_setpoint(alt)])
        if self.phase == "RETURN":
            xy = np.array(c.launch[:2])
            return np.array([xy[0], xy[1], self._depth_setpoint(c.survey_altitude)])
        return est

    def _advance_phase(self, tgt):
        c = self.cfg
        est = self.ekf.position
        d = float(np.linalg.norm(est - tgt))
        dt_phase = self.t - self._phase_t0

        if self.phase == "DEPLOY":
            if abs(est[2] - tgt[2]) < 1.0:
                self._set_phase("ACQUIRE")
        elif self.phase == "ACQUIRE":
            if dt_phase > 6.0 and self.sens.dvl.locked:
                self._event(f"DVL bottom lock, horiz sigma "
                            f"{self.ekf.horizontal_uncertainty:.2f} m")
                self._plan_search()
                self._set_phase("SEARCH")
        elif self.phase in ("SEARCH", "INSPECT"):
            if d < c.waypoint_radius or dt_phase > 400:
                self.wi += 1
                self._phase_t0 = self.t
            if self.wi >= len(self.wps):
                if self.phase == "SEARCH":
                    self._event(f"search complete, coverage "
                                f"{self.bmap.coverage(c.search_box)*100:.1f} %")
                    if self.pier_queue:
                        self._plan_orbit(self.pier_queue.pop(0))
                        self._set_phase("INSPECT")
                    else:
                        self._set_phase("RETURN")
                else:
                    self._finish_pier()
                    if self.pier_queue:
                        self._plan_orbit(self.pier_queue.pop(0))
                        self._phase_t0 = self.t
                    else:
                        self._set_phase("RETURN")
        elif self.phase == "RETURN":
            if d < 3.0 or dt_phase > 260:
                self._set_phase("REPORT")
        elif self.phase == "REPORT":
            self._set_phase("DONE")

    def _finish_pier(self):
        pier = self.current_pier
        xy = self._pier_xy(pier)
        idx = int(pier[1:]) - 1
        pr = self.site.cfg.piers[idx].radius
        res = estimate_scour(self.bmap, xy, pr,
                             search_radius=self.site.cfg.piers[idx].scour_radius + 1.5)
        self.scour_results[pier] = res
        if res:
            self._event(f"{pier} scour: depth {res['max_depth']:.2f} m, "
                        f"volume {res['volume']:.1f} m3")
        else:
            self._event(f"{pier} scour: insufficient coverage")

    def step(self):
        c = self.cfg
        dt = c.dt
        dyn, ekf, ctl = self.dyn, self.ekf, self.ctl

        m = self.sens.sample(self.t, dyn.eta, dyn.nu, dt)
        if "imu" in m:
            gyro, att = m["imu"]
            ekf.predict(gyro, dt)
            ekf.update_attitude(att)
        else:
            ekf.predict(ekf.last_gyro, dt)
        alt, lock = np.nan, False
        if "dvl" in m:
            v, alt, lock = m["dvl"]
            if lock:
                ekf.update_dvl(v)
        if "depth" in m:
            ekf.update_depth(m["depth"])

        tgt = self._target_for_phase()
        yaw = CurrentEstimator.heading_into_flow(self.cest.v)
        if yaw is None:
            yaw = dyn.eta[5]
        tau = ctl.compute(ekf.position, ekf.attitude, ekf.velocity_body,
                          tgt, yaw, dt, current_ff=self.cest.v)
        dyn.step(tau, dt)
        self.cest.update(tau[:3], dyn.nu[:3], dyn.eta[3:])

        # Sonar at its own rate, using the estimated pose as a real system would.
        if self.t >= self._next_ping and self.phase in ("SEARCH", "INSPECT", "TRANSIT"):
            est_pose = np.concatenate([ekf.position, ekf.attitude])
            # The head is tilted down so the swath lands ahead of the vehicle.
            ping_pose = est_pose.copy()
            ping_pose[4] += np.radians(14.0)
            fr = self.fls.ping(ping_pose, t=self.t)
            self.bmap.add(fr.hit_point, max_incidence_deg=78.0,
                          incidence=fr.hit_incidence)
            dets = self._perceive(fr)
            if dets:
                self.tracker.update(dets, self.t)
            self.frames.append({"t": self.t, "phase": self.phase,
                                "pose": ping_pose.copy(),
                                "n_soundings": int(np.isfinite(fr.hit_point).all(1).sum())})
            self._next_ping = self.t + 1.0 / c.sonar_rate_hz

        self._advance_phase(tgt)

        L = self.log
        L["t"].append(self.t)
        L["phase"].append(self.phase)
        L["eta"].append(dyn.eta.copy())
        L["nu"].append(dyn.nu.copy())
        L["est_pos"].append(ekf.position)
        L["est_att"].append(ekf.attitude)
        L["sigma"].append(ekf.horizontal_uncertainty)
        L["alt"].append(alt)
        L["dvl_lock"].append(bool(lock))
        L["current"].append(self.cest.speed)
        L["target_pos"].append(tgt.copy())
        L["thrust"].append(float(np.max(np.abs(dyn.thrust))))
        self.t += dt

    def run(self, verbose=True):
        c = self.cfg
        while self.phase != "DONE" and self.t < c.max_time:
            self.step()
        if verbose:
            self._event(f"mission ended in phase {self.phase} at t={self.t:.1f} s")
        return self.results()

    # ------------------------------------------------------------------ results
    def results(self):
        L = {k: np.array(v) for k, v in self.log.items()
             if k not in ("phase", "dvl_lock")}
        phases = np.array(self.log["phase"])
        lock = np.array(self.log["dvl_lock"])
        eta = L["eta"]
        est = L["est_pos"]
        nav_err = np.linalg.norm(est - eta[:, :3], axis=1)
        path = float(np.sum(np.linalg.norm(np.diff(eta[:, :3], axis=0), axis=1)))
        return {
            "duration": float(self.t),
            "phases_reached": list(dict.fromkeys(self.log["phase"])),
            "path_length": path,
            "nav_error_mean": float(nav_err.mean()),
            "nav_error_final": float(nav_err[-1]),
            "nav_error_max": float(nav_err.max()),
            "sigma_final": float(L["sigma"][-1]),
            # Taken from the sensor itself: the log runs at the control rate
            # while the DVL samples at 8 Hz, so counting log rows would report
            # the rate ratio rather than the dropout rate.
            "dvl_availability": float(self.sens.dvl_availability),
            "dvl_duty_in_log": float(lock.mean()),
            "coverage": self.bmap.coverage(self.cfg.search_box),
            "soundings": int(self.bmap.n_soundings),
            "pings": len(self.frames),
            "scour": self.scour_results,
            "tracks": self.tracker.confirmed,
            "events": self.events,
            "log": self.log,
            "map": self.bmap,
        }
