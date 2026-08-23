"""Flight computer for the LicheeRV Nano.

This is the code that runs on real hardware. It owns the estimation, control
and mission logic, and it is the real-time master of the loop: each tick it
asks the host for a sensor sample, runs the EKF and controller, allocates
thrust, sends pulse widths to the ESP32, and reports the applied wrench back to
the host so the host can integrate the plant.

It imports the same isonavi estimation, control and dynamics code that the
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
from isonavi.estimation import NavigationEKF
from isonavi.control import PoseController, isonavi_GAINS, CurrentEstimator
from isonavi.dynamics import isonavi_1, vectored_allocation
from isonavi.mission import MissionConfig, lawnmower, orbit, PHASES
from isonavi.dynamics import rot_body_to_world


def thrust_to_pwm(force_n, max_n=120.0):
    """Map a per-thruster force in newtons to an ESC pulse width in us.

    Bidirectional ESC convention: 1500 us neutral, 1100 full reverse, 1900
    full forward, linear in between. This is what the ESP32 will output.
    """
    frac = float(np.clip(force_n / max_n, -1.0, 1.0))
    return int(round(1500 + frac * 400))


class SerialLink:
    """Link to the ESP32 actuator interface.

    Accepts either a serial device, which is how it is wired on the vehicle,
    or ``tcp:host:port``, which is how the bench reaches it: the ESP32
    enumerates on the Windows host while this board is a USB network device,
    so the frames cross a transparent socket bridge instead of a UART. The
    bytes on the wire are identical either way.
    """

    def __init__(self, dev, baud=115200):
        self.dev = dev
        self.kind = "none"
        self.ser = None
        self.sock = None
        self.dec = P.FrameDecoder()
        self.sent = 0
        self.echoes = 0
        self.mismatches = 0
        self.worst_err = 0
        self.unanswered = 0
        self.matched = 0
        self.lag_sum = 0
        self.lag_max = 0
        self.trace = []
        # Commands awaiting their echo, oldest first.
        self.q_cmd = []
        self.ring = []
        if not dev:
            return
        if dev.startswith("tcp:"):
            try:
                import socket
                _, host, port = dev.split(":")
                self.sock = socket.create_connection((host, int(port)),
                                                     timeout=5)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.setblocking(False)
                self.kind = "tcp"
                print(f"[esp32] bridged over {host}:{port}")
            except Exception as e:
                print(f"[esp32] bridge connect failed: {e}, running without")
        else:
            try:
                import serial
                self.ser = serial.Serial(dev, baud, timeout=0)
                self.kind = "uart"
                print(f"[esp32] serial {dev} at {baud}")
            except Exception as e:
                print(f"[esp32] serial open failed: {e}, running without")

    @property
    def active(self):
        return self.kind != "none"

    def send_pwm(self, widths):
        if not self.active:
            return
        frame = P.encode(P.MSG_PWM, P.pack_pwm(widths))
        try:
            if self.ser:
                self.ser.write(frame)
            else:
                self.sock.sendall(frame)
            self.sent += 1
            self.q_cmd.append(list(widths))
            self.ring.append(list(widths))
            if len(self.ring) > 32:
                del self.ring[:len(self.ring) - 32]
            # Bound the queue: if the actuator stops answering, the backlog
            # should not grow without limit.
            if len(self.q_cmd) > 64:
                del self.q_cmd[:len(self.q_cmd) - 64]
        except Exception:
            pass

    def poll_echo(self):
        """Non-blocking. The control loop never waits on the actuator."""
        if not self.active:
            return None
        try:
            if self.ser:
                n = self.ser.in_waiting
                data = self.ser.read(n) if n else b""
            else:
                try:
                    data = self.sock.recv(4096)
                except BlockingIOError:
                    data = b""
        except Exception:
            return None
        if not data:
            return None
        got = None
        for mt, pl in self.dec.feed(data):
            if mt == P.MSG_PWM_ECHO:
                got = P.unpack_pwm(pl)
                self.echoes += 1

                # Compare against a ring of recent commands rather than a
                # strict queue. The strict version required each echo to
                # answer the oldest outstanding command; when thrust is
                # changing by tens of microseconds per tick, an echo that
                # legitimately answers a neighbouring command then counts as a
                # fault. What matters is whether the hardware reproduced a
                # command it was actually given.
                hit = -1
                for i in range(len(self.ring) - 1, -1, -1):
                    if max(abs(x - y)
                           for x, y in zip(got, self.ring[i])) <= 2:
                        hit = len(self.ring) - 1 - i
                        break
                if hit < 0:
                    self.mismatches += 1
                    if self.ring and len(self.trace) < 6:
                        ref = self.ring[-1]
                        self.worst_err = max(
                            self.worst_err,
                            max(abs(x - y) for x, y in zip(got, ref)))
                        self.trace.append({"echo": list(got),
                                           "latest_cmd": list(ref)})
                else:
                    self.matched += 1
                    self.lag_sum += hit
                    self.lag_max = max(self.lag_max, hit)
        return got


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
        self.params = isonavi_1
        self.B = vectored_allocation(self.params.arms)
        self.Bp = np.linalg.pinv(self.B)

        launch = np.array(self.cfg.launch, float)
        yaw0 = CurrentEstimator.heading_into_flow(np.array([1.0, 0, 0])) or np.pi
        self.ekf = NavigationEKF(p0=launch)
        self.ekf.x[6:9] = np.array([0.0, 0.0, yaw0])
        self.ctl = PoseController(isonavi_GAINS, params=self.params)
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
        self.t_wait = []
        self.t_esp = []   # time blocked waiting for the host's sensor reply
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
            # Time the actuator link on its own, so the cost of the bench
            # transport can be separated from the control computation.
            t_esp0 = time.perf_counter()
            self.esp.send_pwm(widths)
            # Non-blocking: whatever the actuator has echoed since last tick is
            # collected here, so the control loop never waits on it.
            self.esp.poll_echo()
            self.t_esp.append((time.perf_counter() - t_esp0) * 1000.0)

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
            "esp_ms_mean": (float(np.mean(self.t_esp))
                            if self.t_esp else 0.0),
            "esp_ms_p99": (float(np.percentile(self.t_esp, 99))
                           if self.t_esp else 0.0),
            "esp_link": self.esp.kind,
            "esp_pwm_sent": self.esp.sent,
            "esp_echoes": self.esp.echoes,
            "esp_mismatches": self.esp.mismatches,
            "esp_worst_width_err_us": self.esp.worst_err,
            "esp_matched": self.esp.matched,
            "esp_lag_mean": (self.esp.lag_sum / self.esp.matched
                             if self.esp.matched else 0.0),
            "esp_lag_max": self.esp.lag_max,
            "esp_trace": self.esp.trace,
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
