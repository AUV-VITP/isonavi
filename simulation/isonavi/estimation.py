"""Extended Kalman filter for GPS-denied underwater navigation.

State vector, 12 elements:

    x = [ p_world(3), v_body(3), euler(3), gyro_bias(3) ]

Prediction propagates position from body velocity through the attitude, and
attitude from bias-corrected gyro rates. Corrections come from the DVL
(body velocity), the depth cell (world z), and the IMU attitude solution.

The horizontal position channel is never directly observed: it is dead
reckoned from DVL velocity, so its error grows without bound. That is the
fundamental limitation of underwater navigation and the reason the filter also
carries and reports its own position covariance, which the mission layer uses
to decide when a position reset against a mapped structure is required.
"""

from __future__ import annotations

import numpy as np

from .dynamics import rot_body_to_world, euler_rate_matrix


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class NavigationEKF:
    IP = slice(0, 3)     # position
    IV = slice(3, 6)     # body velocity
    IA = slice(6, 9)     # euler angles
    IB = slice(9, 12)    # gyro bias

    def __init__(self, p0=None, sigma_p0=0.1, seed=0):
        self.x = np.zeros(12)
        if p0 is not None:
            self.x[self.IP] = np.asarray(p0, float)
        self.P = np.diag([
            sigma_p0 ** 2, sigma_p0 ** 2, 0.02 ** 2,
            0.05 ** 2, 0.05 ** 2, 0.05 ** 2,
            0.02 ** 2, 0.02 ** 2, 0.05 ** 2,
            1e-4, 1e-4, 1e-4,
        ])
        # Process noise densities.
        self.q_vel = 0.08 ** 2       # unmodelled acceleration on body velocity
        self.q_att = 0.004 ** 2
        self.q_bias = (2.0e-5) ** 2
        self.q_pos = 0.002 ** 2
        self.last_gyro = np.zeros(3)

    # ------------------------------------------------------------------ predict
    def predict(self, gyro, dt):
        x = self.x
        att = x[self.IA]
        vb = x[self.IV]
        bias = x[self.IB]
        w = np.asarray(gyro, float) - bias
        self.last_gyro = w

        R = rot_body_to_world(att[0], att[1], att[2])
        T = euler_rate_matrix(att[0], att[1])

        x[self.IP] = x[self.IP] + R @ vb * dt
        x[self.IA] = wrap(att + T @ w * dt)

        # Jacobian. Position depends on attitude through R and on body velocity;
        # attitude depends on bias through T.
        F = np.eye(12)
        F[self.IP, self.IV] = R * dt
        F[self.IP, self.IA] = self._dR_dEuler(att, vb) * dt
        F[self.IA, self.IB] = -T * dt

        Q = np.zeros((12, 12))
        Q[self.IP, self.IP] = np.eye(3) * self.q_pos * dt
        Q[self.IV, self.IV] = np.eye(3) * self.q_vel * dt
        Q[self.IA, self.IA] = np.eye(3) * self.q_att * dt
        Q[self.IB, self.IB] = np.eye(3) * self.q_bias * dt

        self.P = F @ self.P @ F.T + Q
        return self.x.copy()

    @staticmethod
    def _dR_dEuler(att, vb, eps=1e-6):
        """Numerical Jacobian of R(att) @ vb with respect to the Euler angles."""
        J = np.zeros((3, 3))
        base = rot_body_to_world(att[0], att[1], att[2]) @ vb
        for k in range(3):
            a = att.copy()
            a[k] += eps
            J[:, k] = (rot_body_to_world(a[0], a[1], a[2]) @ vb - base) / eps
        return J

    # ------------------------------------------------------------------ update
    def _update(self, H, z, h, R):
        y = z - h
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[self.IA] = wrap(self.x[self.IA])
        I_KH = np.eye(12) - K @ H
        # Joseph form keeps the covariance symmetric and positive definite.
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return y

    def update_dvl(self, v_body, sigma=0.006):
        H = np.zeros((3, 12))
        H[:, self.IV] = np.eye(3)
        return self._update(H, np.asarray(v_body, float), self.x[self.IV],
                            np.eye(3) * sigma ** 2)

    def update_depth(self, z_meas, sigma=0.012):
        H = np.zeros((1, 12))
        H[0, 2] = 1.0
        return self._update(H, np.array([z_meas]), np.array([self.x[2]]),
                            np.array([[sigma ** 2]]))

    def update_attitude(self, att, sigma_rp=0.004, sigma_y=0.03):
        H = np.zeros((3, 12))
        H[:, self.IA] = np.eye(3)
        z = np.asarray(att, float)
        h = self.x[self.IA]
        innov = wrap(z - h)
        R = np.diag([sigma_rp ** 2, sigma_rp ** 2, sigma_y ** 2])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.x[self.IA] = wrap(self.x[self.IA])
        I_KH = np.eye(12) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return innov

    def update_position_fix(self, p_xy, sigma=0.35):
        """Absolute horizontal fix from re-observing a mapped structure.

        This is the only mechanism that bounds horizontal drift. It is applied
        when the perception layer recognises a previously mapped pier and can
        resolve the vehicle position against its surveyed location.
        """
        H = np.zeros((2, 12))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return self._update(H, np.asarray(p_xy, float), self.x[:2],
                            np.eye(2) * sigma ** 2)

    # ------------------------------------------------------------------ access
    @property
    def position(self):
        return self.x[self.IP].copy()

    @property
    def attitude(self):
        return self.x[self.IA].copy()

    @property
    def velocity_body(self):
        return self.x[self.IV].copy()

    @property
    def pose(self):
        return np.concatenate([self.x[self.IP], self.x[self.IA]])

    @property
    def position_sigma(self):
        return np.sqrt(np.clip(np.diag(self.P)[self.IP], 0, None))

    @property
    def horizontal_uncertainty(self):
        s = self.position_sigma
        return float(np.hypot(s[0], s[1]))
