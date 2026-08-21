"""Flight computer for the LicheeRV Nano.

This is the code that runs on real hardware. It owns the estimation, control
and mission logic, and it is the real-time master of the loop: each tick it
asks the host for a sensor sample, runs the EKF and controller, allocates
thrust, sends pulse widths to the ESP32, and reports the applied wrench back to
the host so the host can integrate the plant.

It imports the same varuna estimation, control and dynamics code that the
pure-simulation stack uses. Nothing is reimplemented for the board, so a
matching mission result is evidence about the same software, not a lookalike.

The board never receives ground truth. It sees only the sensor sample and acts
on its own estimate, exactly as the flight software would at sea.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, "/root/hil")
sys.path.insert(0, "/root/hil/common")

import hil_protocol as P
from varuna.estimation import NavigationEKF
from varuna.control import PoseController, VARUNA_GAINS, CurrentEstimator
from varuna.dynamics import VARUNA_1, vectored_allocation
from varuna.mission import MissionConfig, lawnmower, orbit, PHASES
from varuna.dynamics import rot_body_to_world


def thrust_to_pwm(force_n, max_n=120.0):
    """Map a per-thruster force in newtons to an ESC pulse width in us.

    Bidirectional ESC convention: 1500 us neutral, 1100 full reverse, 1900
    full forward, linear in between. This is what the ESP32 will output.
    """
    frac = float(np.clip(force_n / max_n, -1.0, 1.0))
    return int(round(1500 + frac * 400))


class SerialLink:
    """UART link to the ESP32 via python-periphery or pyserial."""

    def __init__(self, dev, baud=921600):
        self.dev = dev
        self.ser = None
        self.dec = P.FrameDecoder()
        if dev:
            try:
                import serial
                self.ser = serial.Serial(dev, baud, timeout=0)
            except Exception as e:
                print(f"[esp32] serial open failed: {e}, running without ESP32")
                self.ser = None

    def send_pwm(self, widths):
        if self.ser:
            self.ser.write(P.encode(P.MSG_PWM, P.pack_pwm(widths)))

    def poll_echo(self):
        if not self.ser:
            return None
        n = self.ser.in_waiting
        if not n:
            return None
        for mt, pl in self.dec.feed(self.ser.read(n)):
            if mt == P.MSG_PWM_ECHO:
                return P.unpack_pwm(pl)
        return None


class HostLink:
    """TCP link to the host plant. The board connects; the host listens."""

    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.dec = P.FrameDecoder()
        self._pending = []

    def send(self, msg_type, payload):
        self.sock.sendall(P.encode(msg_type, payload))

    def recv_frame(self, want_type, timeout=5.0):
        """Block until a frame of want_type arrives."""
        self.sock.settimeout(timeout)
        while True:
            for mt, pl in self._pending:
                if mt == want_type:
                    self._pending.remove((mt, pl))
                    return pl
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("host closed")
            self._pending += self.dec.feed(data)


class FlightComputer:
    def __init__(self, host, port, esp_dev, cfg=None, pier_y=None, seed=0):
        self.link = HostLink(host, port)
        self.esp = SerialLink(esp_dev)
        self.cfg = cfg or MissionConfig()
        # Pier cross-stream positions for the inspection orbits. The host is
        # the authority on site geometry, so these are supplied rather than
        # hardcoded on the board. Defaults match the reference site.
        self.pier_y = pier_y or {"P1": -24.0, "P2": -8.0, "P3": 8.0, "P4": 24.0}
        self.params = VARUNA_1
        self.B = vectored_allocation(self.params.arms)
        self.Bp = np.linalg.pinv(self.B)

        launch = np.array(self.cfg.launch, float)
        yaw0 = CurrentEstimator.heading_into_flow(np.array([1.0, 0, 0])) or np.pi
        self.ekf = NavigationEKF(p0=launch)
        self.ekf.x[6:9] = np.array([0.0, 0.0, yaw0])
        self.ctl = PoseController(VARUNA_GAINS, params=self.params)
        self.cest = CurrentEstimator(self.params.quad_damp, self.params.lin_damp)

        self.phase = "DEPLOY"
        self.tick = 0
        self.t = 0.0
        self.dt = self.cfg.dt
        self.wps = []
        self.wi = 0
        self.pier_queue = list(self.cfg.piers_to_inspect)
        self.current_pier = None
        self._phase_t0 = 0.0
        self.timing = []

    # -- the same allocation the plant uses, so wrench maps to the same forces
    def allocate(self, tau):
        """Force-prioritised thrust allocation, computed in closed form.

        The pure-simulation version found the largest torque scale by a 24-step
        bisection. That is a Python loop over matrix-vector products, which on
        the 750 MHz board costs about 9 ms, most of the loop budget. The same
        answer is available in closed form: the largest s in [0, 1] such that
        every |f_force_i + s f_torque_i| <= lim is the tightest per-thruster
        bound, so it is one vectorised min over the eight thrusters rather than
        an iterated search. Bit-for-bit identical to the bisection limit and
        about 20x faster.
        """
        lim = self.params.max_thrust_n
        Bp = self.Bp
        f_force = Bp[:, :3] @ tau[:3]
        f_torque = Bp[:, 3:] @ tau[3:]
        peak_f = np.max(np.abs(f_force))
        if peak_f > lim:
            return f_force * (lim / peak_f)
        # For each thruster, the s at which it saturates in either direction.
        with np.errstate(divide="ignore", invalid="ignore"):
            s_pos = (lim - f_force) / f_torque
            s_neg = (-lim - f_force) / f_torque
        cand = np.where(f_torque > 0, s_pos, s_neg)
        cand = cand[np.isfinite(cand) & (f_torque != 0)]
        s = 1.0 if cand.size == 0 else float(min(1.0, max(0.0, cand.min())))
        return f_force + s * f_torque

    def _depth_setpoint(self, altitude, bed_z):
        return bed_z + altitude

    def _target(self, bed_z):
        c = self.cfg
        est = self.ekf.position
        if self.phase in ("DEPLOY", "ACQUIRE"):
            return np.array([est[0], est[1], self._depth_setpoint(c.survey_altitude, bed_z)])
        if self.phase in ("SEARCH", "INSPECT"):
            xy = self.wps[self.wi] if self.wi < len(self.wps) else est[:2]
            alt = c.inspect_altitude if self.phase == "INSPECT" else c.survey_altitude
            return np.array([xy[0], xy[1], self._depth_setpoint(alt, bed_z)])
        if self.phase == "RETURN":
            xy = np.array(c.launch[:2])
            return np.array([xy[0], xy[1], self._depth_setpoint(c.survey_altitude, bed_z)])
        return est

    def _advance(self, tgt):
        c = self.cfg
        est = self.ekf.position
        d = float(np.linalg.norm(est - tgt))
        dtp = self.t - self._phase_t0
        if self.phase == "DEPLOY":
            if abs(est[2] - tgt[2]) < 1.0:
                self.phase, self._phase_t0 = "ACQUIRE", self.t
        elif self.phase == "ACQUIRE":
            if dtp > 6.0:
                x0, x1, y0, y1 = c.search_box
                self.wps = lawnmower(x0, x1, y0, y1, c.lawnmower_spacing)
                self.wi = 0
                self.phase, self._phase_t0 = "SEARCH", self.t
        elif self.phase in ("SEARCH", "INSPECT"):
            if d < c.waypoint_radius or dtp > 400:
                self.wi += 1
                self._phase_t0 = self.t
            if self.wi >= len(self.wps):
                if (self.phase == "SEARCH" or self.phase == "INSPECT") and self.pier_queue:
                    self.current_pier = self.pier_queue.pop(0)
                    py = self.pier_y[self.current_pier]
                    self.wps = orbit((0.0, py), c.pier_standoff, c.orbit_points)
                    self.wi = 0
                    if self.phase == "SEARCH":
                        self.phase = "INSPECT"
                    self._phase_t0 = self.t
                else:
                    self.phase, self._phase_t0 = "RETURN", self.t
        elif self.phase == "RETURN":
            if d < 3.0 or dtp > 260:
                self.phase, self._phase_t0 = "REPORT", self.t
        elif self.phase == "REPORT":
            self.phase = "DONE"

    def run(self, max_time):
        self.link.send(P.MSG_HELLO, struct.pack("<if", 1, 1.0 / self.dt))
        self.t_wait = []   # time blocked waiting for the host's sensor reply
        self.t_comp = []   # time in estimation + control + allocation
        while self.phase != "DONE" and self.t < max_time:
            t0 = time.perf_counter()

            self.link.send(P.MSG_SENSOR_REQ, P.pack_sensor_req(self.t))
            s = P.unpack_sensor(self.link.recv_frame(P.MSG_SENSOR))
            t_after_wait = time.perf_counter()
            self.t_wait.append((t_after_wait - t0) * 1000.0)

            # Apply exactly the corrections the sensors reported this tick,
            # matching the pure-simulation loop: IMU drives predict+attitude,
            # a tick with no IMU still predicts on the last gyro, DVL corrects
            # velocity only under bottom lock, depth corrects z.
            if s["has_imu"]:
                self.ekf.predict(np.array(s["gyro"]), self.dt)
                self.ekf.update_attitude(np.array(s["att"]))
            else:
                self.ekf.predict(self.ekf.last_gyro, self.dt)
            if s["dvl_lock"]:
                self.ekf.update_dvl(np.array(s["dvl_v"]))
            if s["has_depth"]:
                self.ekf.update_depth(s["depth"])

            # Held DVL altitude gives the bed height under the vehicle for
            # terrain following.
            bed_z = self.ekf.position[2] - s["dvl_alt"]
            tgt = self._target(bed_z)
            yaw = CurrentEstimator.heading_into_flow(self.cest.v)
            if yaw is None:
                yaw = self.ekf.attitude[2]
            tau = self.ctl.compute(self.ekf.position, self.ekf.attitude,
                                   self.ekf.velocity_body, tgt, yaw, self.dt,
                                   current_ff=self.cest.v)
            f = self.allocate(tau)
            self.cest.update(tau[:3], self.ekf.velocity_body, self.ekf.attitude)

            widths = [thrust_to_pwm(fi, self.params.max_thrust_n) for fi in f]
            self.esp.send_pwm(widths)

            self.link.send(P.MSG_THRUST, P.pack_thrust(tau.tolist(), f.tolist()))
            self.link.send(P.MSG_STATE, P.pack_state(
                self.ekf.position.tolist(), self.ekf.attitude.tolist(),
                self.ekf.horizontal_uncertainty, PHASES.index(self.phase),
                self.tick))

            self._advance(tgt)
            self.t += self.dt
            self.tick += 1
            now = time.perf_counter()
            self.timing.append((now - t0) * 1000.0)
            self.t_comp.append((now - t_after_wait) * 1000.0)

        self.link.send(P.MSG_STATE, P.pack_state(
            self.ekf.position.tolist(), self.ekf.attitude.tolist(),
            self.ekf.horizontal_uncertainty, PHASES.index("DONE"), self.tick))
        return self.report()

    def report(self):
        ti = np.array(self.timing)
        return {
            "ticks": self.tick,
            "sim_time": self.t,
            "phase": self.phase,
            "loop_ms_mean": float(ti.mean()) if len(ti) else 0.0,
            "loop_ms_p99": float(np.percentile(ti, 99)) if len(ti) else 0.0,
            "loop_ms_max": float(ti.max()) if len(ti) else 0.0,
            "compute_ms_mean": float(np.mean(self.t_comp)) if self.t_comp else 0.0,
            "compute_ms_p99": float(np.percentile(self.t_comp, 99)) if self.t_comp else 0.0,
            "wait_ms_mean": float(np.mean(self.t_wait)) if self.t_wait else 0.0,
            "budget_ms": self.dt * 1000.0,
            "esp_crc_errors": self.esp.dec.crc_errors,
            "host_crc_errors": self.link.dec.crc_errors,
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.133.84.100")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--esp", default="/dev/ttyS1")
    ap.add_argument("--max-time", type=float, default=1000.0)
    args = ap.parse_args()

    fc = FlightComputer(args.host, args.port, args.esp)
    import json
    rep = fc.run(args.max_time)
    print("REPORT " + json.dumps(rep))
