"""Flight state machine test: enrol, follow, and each kind of loss (decision 36).

Scenario, with a camera that sees 70 degrees and nothing closer than 1 m:
  1. the person stands still and is enrolled                  -> FOLLOW
  2. walks, then passes behind an obstacle for 3 s             -> SEARCH (mid-frame): hold heading, re-lock
  3. turns back and walks past the drone                       -> SEARCH (bottom): turn about 180, re-lock
  4. darts off to the right faster than the drone can yaw      -> SEARCH (right): yaw right, re-lock
  5. disappears for good                                       -> SEARCH (mid-frame): hold, full turn, LAND at 30 s

    python sim/flight_test.py --offline     # built-in vehicle model, runs in about a second
    python sim/flight_test.py               # against SITL (start sim_vehicle.py first)
"""
import argparse
import math
import pathlib
import sys
import time

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "control"))
sys.path.insert(0, str(REPO / "sim"))
from follow_controller import FollowController            # noqa: E402
from flight_state import FlightStateMachine, LAND         # noqa: E402
from fake_target import FakeTarget                        # noqa: E402
from fake_perception import FakePerception                # noqa: E402

SCRIPT = [
    (8.0, 0.0, 0.0),                      # 1 stand: enrolled after the 5 s hold
    (12.0, 1.4, 0.0),                     #   walk
    (3.0, 1.4, 0.0, ("hidden",)),         # 2 behind an obstacle
    (8.0, 1.4, 0.0),                      #   walk on
    (22.0, 1.4, 180.0),                   # 3 turn back and walk past the drone
    (2.0, 6.0, 90.0, ("instant",)),       # 4 dart to the right
    (12.0, 0.0, 0.0, ("instant",)),       #   stand
    (45.0, 0.0, 0.0, ("hidden",)),        # 5 gone
]
NAMES = ["stand", "walk", "obstacle", "walk on", "walk back past", "dart right", "stand", "gone"]


class KinematicVehicle:
    """Simple stand-in for the copter: velocity and yaw rate follow commands with a lag."""

    def __init__(self):
        self.n = self.e = self.yaw = 0.0
        self.v = self.r = 0.0

    def poll(self):
        pass

    @property
    def forward_speed(self):
        return self.v

    def send(self, cmd, dt):
        for _ in range(5):
            h = dt / 5
            a = max(-2.5, min(2.5, (cmd.vx - self.v) / 1.0))
            self.v += a * h
            self.r += (cmd.yaw_rate - self.r) / 0.3 * h
            self.yaw += self.r * h
            self.n += self.v * math.cos(self.yaw) * h
            self.e += self.v * math.sin(self.yaw) * h

    def land(self):
        pass


class SitlVehicle:
    def __init__(self, conn, params, alt):
        from pymavlink import mavutil
        import hold_test as ht
        import velocity_test as vt
        self.mavutil, self.mask = mavutil, ht.MASK_VEL_YAWRATE
        self.vt = vt
        m = self.m = mavutil.mavlink_connection(conn)
        vt.wait_heartbeat(m)
        for name, val in (params.get("mavlink", {}).get("autopilot_params") or {}).items():
            ht.set_param(m, name, val)
        ht.request_rate(m, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 20)
        ht.request_rate(m, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 20)
        vt.set_mode(m, "GUIDED")
        vt.arm(m)
        vt.takeoff(m, alt)
        time.sleep(2)
        self.tel = ht.Telemetry(m)
        t0 = time.time()
        while not (self.tel.have_pos and self.tel.have_att):
            self.tel.poll()
            if time.time() - t0 > 5:
                raise SystemExit("no position or attitude telemetry")
            time.sleep(0.05)

    def poll(self):
        self.tel.poll()

    n = property(lambda self: self.tel.n)
    e = property(lambda self: self.tel.e)
    yaw = property(lambda self: self.tel.yaw)
    forward_speed = property(lambda self: self.tel.forward_speed)

    def send(self, cmd, dt):
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            self.mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, self.mask,
            0, 0, 0, cmd.vx, cmd.vy, cmd.vz, 0, 0, 0, 0, cmd.yaw_rate)

    def land(self):
        self.vt.set_mode(self.m, "LAND")


def run(vehicle, params, realtime):
    rate = float(params["limits"]["command_rate_hz"])
    dt = 1.0 / rate
    vehicle.poll()
    tgt = FakeTarget(vehicle.n + 4.0 * math.cos(vehicle.yaw), vehicle.e + 4.0 * math.sin(vehicle.yaw),
                     vehicle.yaw, SCRIPT)
    per = FakePerception(params, tgt)
    fsm = FlightStateMachine(params, FollowController(params))
    t, last_seg, enrolled_at, landed_at = 0.0, -1, None, None
    lost_at, last_search_start = None, None
    tick = time.time()
    while True:
        vehicle.poll()
        tgt.step(dt)
        if tgt.segment != last_seg:
            last_seg = tgt.segment
            print(f"t={t:6.1f}s  person: {NAMES[last_seg]}")
        obs = per.observe(vehicle.n, vehicle.e, vehicle.yaw, t)
        cmd, events = fsm.update(obs, dt, vehicle.forward_speed)
        for ev in events:
            print(f"t={t:6.1f}s    {ev}")
            if "person enrolled" in ev:
                enrolled_at = t
            if "-> SEARCH" in ev:
                last_search_start = t
            if "-> LAND" in ev:
                landed_at = t
        vehicle.send(cmd, dt)
        t += dt
        if fsm.state == LAND or tgt.done:
            break
        if realtime:
            tick += dt
            time.sleep(max(0.0, tick - time.time()))
    vehicle.land()

    print()
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    check("enrolment", enrolled_at is not None and 4.5 <= enrolled_at <= 6.5,
          f"locked at {enrolled_at if enrolled_at is None else round(enrolled_at, 1)} s (5 s still hold)")
    log = fsm.search_log
    want = [("mid", "relock"), ("bottom", "relock"), ("right", "relock"), ("mid", "land")]
    check("search count", len(log) == 4, f"{len(log)} searches, expected 4")
    labels = ["obstacle", "walk past", "dart right", "gone"]
    for i, (edge, outcome) in enumerate(want):
        if i >= len(log):
            break
        s = log[i]
        turned = math.degrees(s["turned"]) * s["dir"]
        ok = s["edge"] == edge and s["outcome"] == outcome
        if i == 0:
            ok &= abs(turned) < 10
        elif i == 1:
            ok &= 90 <= abs(turned) <= 300
        elif i == 2:
            ok &= turned > 0
        else:
            ok &= abs(turned) >= 350 and 29.0 <= s["elapsed"] <= 31.5
        check(labels[i], ok, f"lost at {s['edge']}, turned {turned:+.0f} degrees, {s['elapsed']:.1f} s, outcome {s['outcome']}")
    print("\nRESULT:", "PASS" if all(checks) else "FAIL")
    return all(checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="built-in vehicle model, no SITL")
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    ap.add_argument("--alt", type=float, default=3.0)
    a = ap.parse_args()
    with open(REPO / "config" / "params.yaml") as f:
        params = yaml.safe_load(f)
    vehicle = KinematicVehicle() if a.offline else SitlVehicle(a.conn, params, a.alt)
    sys.exit(0 if run(vehicle, params, realtime=not a.offline) else 1)


if __name__ == "__main__":
    main()
