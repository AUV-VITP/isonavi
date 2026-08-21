"""Host-side plant for the hardware-in-the-loop bench.

The host owns everything physical: the disaster site, the true 6-DOF vehicle
dynamics, and the sensor models. It listens for a board to connect, then serves
the loop the board drives:

    board asks for the sensor sample at tick time t  ->  host samples its
    sensor models from the true state and replies

    board sends the wrench it allocated  ->  host integrates the true
    dynamics one step under that wrench

Ground truth never leaves the host. The board's estimate is compared against
the host's truth only after the run, to measure navigation error, exactly as
the pure-simulation evaluation does. This is what makes a matching result mean
"the autonomy ran on the board", not "the board was handed the answer".
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common.hil_protocol as P  # noqa: E402
from varuna.scene import DisasterSite  # noqa: E402
from varuna.dynamics import VehicleDynamics, VARUNA_1  # noqa: E402
from varuna.sensors import SensorSuite  # noqa: E402
from varuna.control import CurrentEstimator  # noqa: E402
from varuna.mapping import BathymetryMap  # noqa: E402


class HostPlant:
    def __init__(self, seed=0):
        self.site = DisasterSite()
        launch = np.array((-46.0, -2.0, -3.0), float)
        yaw0 = CurrentEstimator.heading_into_flow(
            self.site.current_at(launch)) or np.pi
        self.dyn = VehicleDynamics(VARUNA_1, current_fn=self.site.current_at)
        self.dyn.reset(eta=[*launch, 0, 0, yaw0])
        self.sens = SensorSuite(site=self.site, seed=seed + 11)
        self.dt = 0.05
        self.t = 0.0
        self._last_alt = float(launch[2] - self.site.bed_height(launch[0], launch[1]))
        # Logs for the post-run comparison.
        self.true_eta = []
        self.true_t = []
        self.est_pos = []
        self.est_att = []
        self.est_sigma = []
        self.phase_id = []
        self.thrust_peak = []
        self.current_est = []
        # Reuse the same bathymetric map the pure-sim mission builds, so the
        # HIL run produces a comparable coverage / map product.
        c = self.site.cfg
        self.bmap = BathymetryMap(c.x_min, c.x_max, c.y_min, c.y_max, 0.5)

    def sample_sensors(self, t):
        """Sample the sensor models at the current true state for tick t.

        The SensorSuite runs its own internal rates (IMU 100 Hz, DVL 8 Hz,
        depth 20 Hz), so a given tick may carry any subset. A validity mask
        tells the board exactly which sensors reported, so it applies the same
        corrections as the pure-simulation loop rather than acting on a
        fabricated sample when a sensor was not actually due.

        The last DVL altitude is held between updates so the board always has a
        terrain reference, matching how the real altimeter reading persists.
        """
        m = self.sens.sample(self.t, self.dyn.eta, self.dyn.nu, self.dt)
        valid = 0
        gyro = (0.0, 0.0, 0.0)
        att = (0.0, 0.0, 0.0)
        if "imu" in m:
            valid |= P.V_IMU
            gyro, att = tuple(m["imu"][0]), tuple(m["imu"][1])
        v = (0.0, 0.0, 0.0)
        alt = self._last_alt
        if "dvl" in m:
            valid |= P.V_DVL
            dv, dalt, lock = m["dvl"]
            if lock:
                valid |= P.V_LOCK
                v = tuple(dv)
                alt = float(dalt)
                self._last_alt = alt
        depth = self.dyn.eta[2]
        if "depth" in m:
            valid |= P.V_DEPTH
            depth = float(m["depth"])
        return P.pack_sensor(t, valid, gyro, att, v, float(alt), float(depth))

    def step(self, wrench):
        """Integrate the true dynamics one step under the board's wrench."""
        self.dyn.step(np.array(wrench, float), self.dt)
        self.t += self.dt


def serve(port, max_time, out_path, seed=0):
    plant = HostPlant(seed=seed)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"[host] waiting for flight computer on :{port} ...")
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[host] board connected from {addr}")
    dec = P.FrameDecoder()
    pending = []
    conn.settimeout(15.0)

    def next_frame():
        while True:
            if pending:
                return pending.pop(0)
            data = conn.recv(4096)
            if not data:
                raise ConnectionError("board closed")
            pending.extend(dec.feed(data))

    t_wall0 = time.perf_counter()
    board_state = None
    last_thrust = None
    ticks = 0
    try:
        while plant.t < max_time:
            mt, pl = next_frame()
            if mt == P.MSG_HELLO:
                ver, rate = struct.unpack("<if", pl)
                print(f"[host] board hello: proto {ver}, {rate:.0f} Hz")
            elif mt == P.MSG_SENSOR_REQ:
                t_req = P.unpack_sensor_req(pl)
                conn.sendall(P.encode(P.MSG_SENSOR, plant.sample_sensors(t_req)))
            elif mt == P.MSG_THRUST:
                d = P.unpack_thrust(pl)
                last_thrust = d
                # Log truth at this tick before integrating.
                plant.true_eta.append(self_copy(plant.dyn.eta))
                plant.true_t.append(plant.t)
                plant.thrust_peak.append(float(np.max(np.abs(d["thrusters"]))))
                plant.step(d["wrench"])
                ticks += 1
            elif mt == P.MSG_STATE:
                st = P.unpack_state(pl)
                board_state = st
                plant.est_pos.append(list(st["pos"]))
                plant.est_att.append(list(st["att"]))
                plant.est_sigma.append(st["sigma"])
                plant.phase_id.append(st["phase_id"])
                if st["phase_id"] == P.PHASES.index("DONE"):
                    print("[host] board reported DONE")
                    break
    except (ConnectionError, socket.timeout) as e:
        print(f"[host] link ended: {e}")

    wall = time.perf_counter() - t_wall0
    conn.close()
    srv.close()

    # Post-run comparison against truth.
    n = min(len(plant.est_pos), len(plant.true_eta))
    est = np.array(plant.est_pos[:n])
    tru = np.array(plant.true_eta[:n])[:, :3]
    nav_err = np.linalg.norm(est - tru, axis=1) if n else np.array([0.0])
    result = {
        "ticks": ticks,
        "sim_time": plant.t,
        "wall_time": wall,
        "realtime_factor": plant.t / wall if wall else 0.0,
        "nav_error_mean": float(nav_err.mean()),
        "nav_error_max": float(nav_err.max()),
        "nav_error_final": float(nav_err[-1]),
        "thrust_peak_mean": float(np.mean(plant.thrust_peak)) if plant.thrust_peak else 0.0,
        "crc_errors": dec.crc_errors,
        "resyncs": dec.resyncs,
        "final_phase": int(plant.phase_id[-1]) if plant.phase_id else -1,
    }
    print("[host] " + json.dumps(result, indent=1))
    if out_path:
        np.savez_compressed(out_path,
                            true_eta=np.array(plant.true_eta),
                            true_t=np.array(plant.true_t),
                            est_pos=np.array(plant.est_pos),
                            est_att=np.array(plant.est_att),
                            est_sigma=np.array(plant.est_sigma),
                            phase_id=np.array(plant.phase_id),
                            thrust_peak=np.array(plant.thrust_peak))
        with open(out_path.replace(".npz", ".json"), "w") as f:
            json.dump(result, f, indent=1)
        print(f"[host] wrote {out_path}")
    return result


def self_copy(a):
    return np.array(a, float).copy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--max-time", type=float, default=1000.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    serve(args.port, args.max_time, args.out, args.seed)
