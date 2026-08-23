"""Verify dynamics, estimation and control against physical expectations."""
import numpy as np

from isonavi.dynamics import (VehicleDynamics, VehicleParams, vectored_allocation,
                             BLUEROV2_HEAVY, isonavi_1, max_holdable_current)
from isonavi.sensors import SensorSuite
from isonavi.estimation import NavigationEKF
from isonavi.control import PoseController, ControlGains, isonavi_GAINS, CurrentEstimator
from isonavi.scene import DisasterSite

ok = lambda c, m: print(("  PASS  " if c else "  FAIL  ") + m)
print("=" * 64)
print("1. THRUST ALLOCATION")
print("=" * 64)
B = vectored_allocation()
print(f"  allocation matrix : {B.shape[0]} DOF x {B.shape[1]} thrusters")
rank = np.linalg.matrix_rank(B)
print(f"  rank              : {rank}")
ok(rank == 6, "all six degrees of freedom are actuated")
Bp = np.linalg.pinv(B)
for k, nm in enumerate(["surge", "sway", "heave", "roll", "pitch", "yaw"]):
    tau = np.zeros(6); tau[k] = 10.0
    f = Bp @ tau
    got = B @ f
    err = np.linalg.norm(got - tau)
    print(f"  {nm:6s}: |f|max {np.max(np.abs(f)):6.2f} N   reconstruction err {err:.2e}")

print()
print("=" * 64)
print("2. OPEN-LOOP PHYSICS")
print("=" * 64)
p = VehicleParams()
print(f"  weight   {p.weight:7.2f} N")
print(f"  buoyancy {p.buoyancy:7.2f} N   net {p.buoyancy-p.weight:+.2f} N")
dyn = VehicleDynamics(p)
dyn.reset(eta=[0, 0, -8, 0, 0, 0])
for _ in range(400):
    dyn.step(np.zeros(6), 0.01)
print(f"  after 4 s with no thrust: z = {dyn.eta[2]:.3f} m, w = {dyn.nu[2]:+.3f} m/s")
ok(dyn.eta[2] > -8.0, "slightly positive buoyancy makes it rise")
ok(abs(dyn.nu[2]) < 0.6, "terminal rise speed is bounded by drag")

# Drift in current with no thrust.
site = DisasterSite()
dyn2 = VehicleDynamics(p, current_fn=site.current_at)
dyn2.reset(eta=[-20, -8, -9, 0, 0, 0])
for _ in range(600):
    dyn2.step(np.zeros(6), 0.01)
drift = dyn2.eta[0] - (-20)
cur = site.current_at([-20, -8, -9])
print(f"  ambient current at start : {np.linalg.norm(cur):.2f} m/s")
print(f"  drift after 6 s, no thrust: {drift:+.2f} m downstream")
ok(drift > 2.0, "vehicle is swept downstream when unactuated")

print()
print("=" * 64)
print("3. STATION KEEPING IN CURRENT: COTS baseline vs purpose-designed")
print("=" * 64)
site = site if "site" in dir() else DisasterSite()
station = np.array([-14.0, -8.0, -9.0])
truth_current = float(np.linalg.norm(site.current_at(station)))
print(f"  ambient current at station : {truth_current:.2f} m/s")
print()
# Deploy already oriented into the flow. A finned hull is only stable flying
# nose-first, and at this current a 180 degree turn is broadside long enough to
# be swept, so orientation is set at launch rather than corrected later.
yaw_into_flow = CurrentEstimator.heading_into_flow(site.current_at(station)) or 0.0
print(f"  commanded heading          : {np.degrees(yaw_into_flow):.0f} deg (into flow)")
print()
for params, gains in ((BLUEROV2_HEAVY, ControlGains()), (isonavi_1, isonavi_GAINS)):
    envelope = max_holdable_current(params)
    dyn = VehicleDynamics(params, current_fn=site.current_at)
    dyn.reset(eta=[*station, 0, 0, yaw_into_flow])
    ctl = PoseController(gains, params=params)
    cest = CurrentEstimator(params.quad_damp, params.lin_damp)
    dt, errs = 0.02, []
    for i in range(int(80 / dt)):
        tau = ctl.compute(dyn.eta[:3], dyn.eta[3:], dyn.nu[:3],
                          station, yaw_into_flow, dt, current_ff=cest.v)
        dyn.step(tau, dt)
        cest.update(tau[:3], dyn.nu[:3], dyn.eta[3:])
        if i * dt > 35:
            errs.append(np.linalg.norm(dyn.eta[:3] - station))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    print(f"  {params.name}")
    print(f"    thrust envelope        : {envelope:.2f} m/s max holdable current")
    print(f"    station-keeping RMSE   : {rmse:.3f} m")
    print(f"    inferred current       : {cest.speed:.2f} m/s (true {truth_current:.2f})")
    holds = rmse < 0.5
    expect = envelope > truth_current
    ok(holds == expect,
       f"{'holds station' if expect else 'is swept away'} as the thrust envelope predicts")
    print()

print()
print("=" * 64)
print("4. EKF TRACKING (closed loop, real sensors)")
print("=" * 64)
dyn = VehicleDynamics(isonavi_1, current_fn=site.current_at)
dyn.reset(eta=[-30.0, -8.0, -9.0, 0, 0, np.pi])
sens = SensorSuite(site=site, seed=5)
ekf = NavigationEKF(p0=[-30.0, -8.0, -9.0])
ctl = PoseController(isonavi_GAINS, params=isonavi_1)
dt = 0.02
t = 0.0
wps = [np.array([-30.0, -8.0, -9.0]), np.array([-10.0, -8.0, -9.5]),
       np.array([-10.0, 8.0, -9.5]), np.array([10.0, 8.0, -9.0])]
wi = 0
perr, herr = [], []
for i in range(int(140 / dt)):
    m = sens.sample(t, dyn.eta, dyn.nu, dt)
    if "imu" in m:
        gyro, att = m["imu"]
        ekf.predict(gyro, dt)
        ekf.update_attitude(att)
    else:
        ekf.predict(ekf.last_gyro, dt)
    if "dvl" in m:
        v, alt, lock = m["dvl"]
        if lock:
            ekf.update_dvl(v)
    if "depth" in m:
        ekf.update_depth(m["depth"])

    tgt = wps[wi]
    if np.linalg.norm(ekf.position - tgt) < 1.2 and wi < len(wps) - 1:
        wi += 1
    yaw = ctl.heading_to(ekf.position, tgt) or 0.0
    tau = ctl.compute(ekf.position, ekf.attitude, ekf.velocity_body,
                      tgt, yaw, dt)
    dyn.step(tau, dt)
    t += dt
    perr.append(np.linalg.norm(ekf.position - dyn.eta[:3]))
    herr.append(abs(np.arctan2(np.sin(ekf.attitude[2] - dyn.eta[5]),
                               np.cos(ekf.attitude[2] - dyn.eta[5]))))

perr = np.array(perr); herr = np.array(herr)
dist = 0.0
print(f"  DVL availability      : {sens.dvl_availability*100:.1f} %")
print(f"  final EKF pos error   : {perr[-1]:.3f} m")
print(f"  mean EKF pos error    : {perr.mean():.3f} m")
print(f"  mean heading error    : {np.degrees(herr.mean()):.2f} deg")
print(f"  reported horiz sigma  : {ekf.horizontal_uncertainty:.3f} m")
print(f"  waypoints reached     : {wi+1}/{len(wps)}")
ok(perr[-1] < 2.5, "EKF position error stays bounded over the run")
ok(np.degrees(herr.mean()) < 5.0, "heading estimate tracks truth")
ok(wi >= 2, "vehicle progresses through the waypoint list")

print()
print("smoke test complete")
