"""Cascaded pose controller for a hovering underwater vehicle.

Structure, outermost loop first:

    position error  ->  desired world velocity   (P, saturated)
    world velocity  ->  desired body velocity    (rotate by attitude)
    body velocity   ->  body wrench              (PI with feedforward)

The inner velocity loop carries integral action. That integrator is what
absorbs the steady hydrodynamic load from the river current: with a 2 m/s
flow the vehicle needs a permanent upstream thrust bias simply to hold
station, and a purely proportional loop would sit at a standing offset.

Attitude is regulated with an independent PD loop on roll, pitch and yaw.
Roll and pitch are naturally stabilised by the buoyancy restoring moment, so
those gains only need to damp; yaw is unrestrained and needs real authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import rot_body_to_world


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


@dataclass
class ControlGains:
    # Outer position loop, metres to metres per second.
    kp_pos_xy: float = 0.85
    kp_pos_z: float = 1.10
    max_speed_xy: float = 0.75
    max_speed_z: float = 0.45

    # Inner body-velocity loop, metres per second to newtons.
    kp_vel: tuple = (95.0, 105.0, 130.0)
    ki_vel: tuple = (34.0, 36.0, 40.0)
    i_limit_n: float = 55.0

    # Attitude loop.
    kp_att: tuple = (18.0, 18.0, 26.0)
    kd_att: tuple = (7.0, 7.0, 9.0)
    max_yaw_rate: float = 0.7

    # Wrench limits.
    max_force_n: float = 110.0
    max_torque_nm: float = 22.0


isonavi_GAINS = ControlGains(
    kp_pos_xy=0.95, kp_pos_z=1.20,
    max_speed_xy=1.10, max_speed_z=0.55,
    kp_vel=(185.0, 265.0, 285.0),
    ki_vel=(72.0, 92.0, 98.0),
    i_limit_n=210.0,
    kp_att=(42.0, 58.0, 72.0),
    kd_att=(19.0, 26.0, 31.0),
    max_yaw_rate=0.8,
    max_force_n=340.0, max_torque_nm=60.0,
)


class PoseController:
    """Drives the vehicle to a commanded pose or velocity."""

    def __init__(self, gains: ControlGains | None = None, params=None):
        self.g = gains or ControlGains()
        # Vehicle damping, used to convert a current estimate into the thrust
        # required to oppose it. Without it the feedforward channel is disabled.
        self.lin = np.asarray(params.lin_damp, float)[:3] if params is not None else None
        self.quad = np.asarray(params.quad_damp, float)[:3] if params is not None else None
        self.reset()

    def reset(self):
        self.i_vel = np.zeros(3)
        self.last_att_err = np.zeros(3)
        self.f_ff = np.zeros(3)

    def drag_force(self, v_rel_body):
        """Steady drag on the hull for a given through-water velocity."""
        v = np.asarray(v_rel_body, float)
        return self.lin * v + self.quad * v * np.abs(v)

    def compute(self, est_pos, est_att, est_vel_body, target_pos,
                target_yaw, dt, current_ff=None, vel_cmd_world=None):
        """Returns the commanded 6-DOF body wrench.

        ``current_ff`` is an optional estimate of the water velocity in world
        coordinates. It is applied as a *force* feedforward: the thrust needed
        to balance hull drag at that flow speed. Adding it to the velocity
        setpoint instead would be wrong, because the inner loop regulates
        ground-referenced velocity from the DVL, and inflating that setpoint
        would command the vehicle to fly downstream rather than hold position.
        """
        g = self.g
        R = rot_body_to_world(est_att[0], est_att[1], est_att[2])

        # ---- outer loop: position to desired ground velocity
        if vel_cmd_world is not None:
            v_des_world = np.asarray(vel_cmd_world, float).copy()
        else:
            e = np.asarray(target_pos, float) - np.asarray(est_pos, float)
            v_des_world = np.array([g.kp_pos_xy * e[0],
                                    g.kp_pos_xy * e[1],
                                    g.kp_pos_z * e[2]])
        hxy = np.hypot(v_des_world[0], v_des_world[1])
        if hxy > g.max_speed_xy:
            v_des_world[:2] *= g.max_speed_xy / hxy
        v_des_world[2] = np.clip(v_des_world[2], -g.max_speed_z, g.max_speed_z)

        # ---- inner loop: body velocity to force
        v_des_body = R.T @ v_des_world
        ev = v_des_body - np.asarray(est_vel_body, float)

        # Force feedforward: oppose the drag the current exerts while the
        # vehicle holds the commanded ground velocity.
        self.f_ff = np.zeros(3)
        if current_ff is not None and self.quad is not None:
            v_rel = R.T @ (np.asarray(current_ff, float)) - v_des_body
            self.f_ff = -self.drag_force(v_rel)

        # Integral with anti-windup applied to the integral *contribution*,
        # back-calculating the state so it cannot accumulate past the limit.
        ki = np.asarray(g.ki_vel, float)
        self.i_vel = self.i_vel + ev * dt
        i_term = np.clip(ki * self.i_vel, -g.i_limit_n, g.i_limit_n)
        self.i_vel = i_term / np.maximum(ki, 1e-9)

        f = np.asarray(g.kp_vel, float) * ev + i_term + self.f_ff
        f = np.clip(f, -g.max_force_n, g.max_force_n)

        # ---- attitude loop
        att_err = np.array([wrap(0.0 - est_att[0]),
                            wrap(0.0 - est_att[1]),
                            wrap(target_yaw - est_att[2])])
        d_err = (att_err - self.last_att_err) / max(dt, 1e-6)
        self.last_att_err = att_err
        m = np.array(g.kp_att) * att_err + np.array(g.kd_att) * np.clip(d_err, -3, 3)
        m = np.clip(m, -g.max_torque_nm, g.max_torque_nm)

        return np.concatenate([f, m])

    def heading_to(self, frm, to):
        d = np.asarray(to, float)[:2] - np.asarray(frm, float)[:2]
        if np.linalg.norm(d) < 1e-6:
            return None
        return float(np.arctan2(d[1], d[0]))


class CurrentEstimator:
    """Estimates the ambient water velocity from the controller integrator.

    The vehicle has no flow sensor. However, in steady state the integral term
    of the velocity loop is exactly the thrust needed to oppose the drag from
    the water moving past the hull, so the current can be inferred from it by
    inverting the drag model. This gives the mission layer a usable current
    readout without additional hardware.
    """

    def __init__(self, quad_damp=(141.0, 217.0, 190.0), lin_damp=(13.7, 0.0, 33.0),
                 alpha=0.02):
        # Accepts either the three translational coefficients or the full
        # six element damping tuple from VehicleParams.
        self.cd = np.asarray(quad_damp, float).ravel()[:3]
        self.cl = np.asarray(lin_damp, float).ravel()[:3]
        self.alpha = alpha
        self.v = np.zeros(3)

    def _invert_drag(self, f):
        """Solve cl v + cd v|v| = f for v, taking the physical root."""
        cd = np.maximum(self.cd, 1e-6)
        cl = self.cl
        disc = cl ** 2 + 4.0 * cd * np.abs(f)
        return np.sign(f) * (-cl + np.sqrt(disc)) / (2.0 * cd)

    def update(self, f_body, v_body_ground, est_att):
        """Infer the ambient current from the thrust and drag balance.

        In quasi-steady flight the applied thrust balances hull drag, which
        depends on the through-water velocity. Inverting the drag model gives
        that relative velocity, and subtracting it from the DVL ground
        velocity leaves the water velocity itself.
        """
        v_rel = -self._invert_drag(np.asarray(f_body, float))
        v_c_body = np.asarray(v_body_ground, float) - v_rel
        R = rot_body_to_world(est_att[0], est_att[1], est_att[2])
        v_world = R @ v_c_body
        self.v = (1 - self.alpha) * self.v + self.alpha * v_world
        return self.v.copy()

    @staticmethod
    def heading_into_flow(v_current_world):
        """Yaw that points the low-drag axis into the flow.

        A faired hull has several times more drag broadside than along its
        axis, so holding station costs far less power when the vehicle is
        turned to face the oncoming water.
        """
        v = np.asarray(v_current_world, float)
        if np.hypot(v[0], v[1]) < 1e-3:
            return None
        return float(np.arctan2(-v[1], -v[0]))

    @property
    def speed(self):
        return float(np.linalg.norm(self.v))
