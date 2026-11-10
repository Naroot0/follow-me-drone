"""Station-hold test in SITL: the follow controller against a scripted fake person.

The fake person is a point in the simulator's local frame. Each cycle the script
converts it to what the camera would report (range, bearing), runs the controller,
and sends the body-frame velocity and yaw-rate command to ArduPilot.

Script: stand 10 s, walk away 20 s at 1.4 m/s, turn right 90 degrees and walk 15 s,
stop 10 s, turn around and walk 1.4 m toward the drone, stand 8 s.

Pass criteria (measured from 3 s into each segment, against true positions):
  walk, turn+walk : range inside station distance +/- band for at least 95 % of the
                    time (85 % with --noisy), bearing never beyond 10 degrees
  approach        : the drone moves less than 0.5 m while the person walks toward it (hover, no retreat)
  stop            : after braking, the drone backs up and is inside the band by the end of the segment
  stand2          : after the person has approached and stopped, the drone holds position (no retreat)

SITL (ArduPilot venv, own terminal):  sim_vehicle.py --console --map
The script connects straight to SITL's second serial port (tcp 5762), not through
MAVProxy: MAVProxy's forwarded telemetry arrives at about 4 Hz, which is too stale
to stand in for a camera. QGroundControl keeps working through MAVProxy as before.
Run (project venv):
    python sim/hold_test.py
    python sim/hold_test.py --noisy      # 10 % range noise, 2 deg bearing noise, 100 ms delay
"""
import argparse
import math
import pathlib
import random
import sys
import time

import yaml
from pymavlink import mavutil

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "control"))
sys.path.insert(0, str(REPO / "sim"))
from follow_controller import FollowController          # noqa: E402
from fake_target import FakeTarget, SEGMENT_NAMES       # noqa: E402
import velocity_test as vt                              # noqa: E402

MASK_VEL_YAWRATE = 0x05C7   # ignore position, acceleration, yaw; use velocity and yaw rate


class Telemetry:
    """Latest position, velocity and heading; refreshed by reading everything pending."""

    def __init__(self, m):
        self.m = m
        self.n = self.e = self.d = self.vn = self.ve = 0.0
        self.yaw = 0.0
        self.pos_count = 0
        self.have_pos = self.have_att = False

    def poll(self):
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                return
            t = msg.get_type()
            if t == "LOCAL_POSITION_NED":
                self.n, self.e, self.d, self.vn, self.ve = msg.x, msg.y, msg.z, msg.vx, msg.vy
                self.pos_count += 1
                self.have_pos = True
            elif t == "ATTITUDE":
                self.yaw = msg.yaw
                self.have_att = True

    @property
    def forward_speed(self):
        return self.vn * math.cos(self.yaw) + self.ve * math.sin(self.yaw)


def set_param(m, name, value):
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(),
                         float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(0.5)
    print(f"param {name} = {value}")


def request_rate(m, msg_id, hz):
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    ap.add_argument("--alt", type=float, default=3.0)
    ap.add_argument("--noisy", action="store_true")
    ap.add_argument("--fov", type=float, default=70.0, help="camera horizontal field of view, degrees")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    random.seed(a.seed)

    with open(REPO / "config" / "params.yaml") as f:
        params = yaml.safe_load(f)
    mav = params.get("mavlink", {})
    rate = float(params["limits"]["command_rate_hz"])
    dist = float(params["station"]["distance_m"])
    band = float(params["station"]["distance_band_m"])
    ctl = FollowController(params)
    c = params["control"]
    print(f"gains: kp_distance {c['kp_distance']}, kp_bearing {c['kp_bearing']}, "
          f"feedforward {c.get('feedforward_gain', 0)}, max reverse {params['limits'].get('max_reverse_mps', 0)} m/s"
          f"{'  [noisy]' if a.noisy else ''}")

    m = mavutil.mavlink_connection(a.conn)
    vt.wait_heartbeat(m)
    for name, val in (mav.get("autopilot_params") or {}).items():
        set_param(m, name, val)
    request_rate(m, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 20)
    request_rate(m, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 20)

    vt.set_mode(m, "GUIDED")
    vt.arm(m)
    vt.takeoff(m, a.alt)
    time.sleep(2)

    tel = Telemetry(m)
    t_wait = time.time()
    while not (tel.have_pos and tel.have_att):
        tel.poll()
        if time.time() - t_wait > 5:
            raise SystemExit("no position or attitude telemetry")
        time.sleep(0.05)

    # person starts 4 m ahead of the drone, facing away from it
    tgt = FakeTarget(tel.n + 4.0 * math.cos(tel.yaw), tel.e + 4.0 * math.sin(tel.yaw), tel.yaw)
    dt = 1.0 / rate
    stats, seg_t0, seg_pos0, seg_pos1 = {}, {}, {}, {}
    prev_meas = (None, None)
    peak, last_range = {}, {}
    lost = 0
    t0 = time.time()
    count0 = tel.pos_count
    next_tick = t0
    last_seg = -1
    while not tgt.done:
        tel.poll()
        tgt.step(dt)
        s = tgt.segment
        now = time.time() - t0
        if s != last_seg:
            print(f"t={now:5.1f}s  segment: {SEGMENT_NAMES[s]}")
            seg_t0[s], seg_pos0[s], last_seg = now, (tel.n, tel.e), s
        seg_pos1[s] = (tel.n, tel.e)

        rng, brg = tgt.seen_from(tel.n, tel.e, tel.yaw, a.fov)
        if rng is None:
            lost += 1
        meas = (rng, brg)
        if a.noisy:
            if rng is not None:
                meas = (rng * (1 + random.gauss(0, 0.10)), brg + random.gauss(0, math.radians(2)))
            use, prev_meas = prev_meas, meas          # one cycle of delay
        else:
            use = meas
        cmd = ctl.update(use[0], use[1], dt, own_speed=tel.forward_speed)
        m.mav.set_position_target_local_ned_send(
            0, m.target_system, m.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, MASK_VEL_YAWRATE,
            0, 0, 0, cmd.vx, cmd.vy, cmd.vz, 0, 0, 0, 0, cmd.yaw_rate)

        true_r = math.hypot(tgt.n - tel.n, tgt.e - tel.e)
        true_b = (math.atan2(tgt.e - tel.e, tgt.n - tel.n) - tel.yaw + math.pi) % (2 * math.pi) - math.pi
        peak[s] = max(peak.get(s, 0.0), abs(math.degrees(true_b)))
        last_range[s] = true_r
        if now - seg_t0[s] >= 3.0:
            stats.setdefault(s, []).append((true_r, abs(math.degrees(true_b))))

        next_tick += dt
        time.sleep(max(0.0, next_tick - time.time()))

    elapsed = time.time() - t0
    print(f"\ntelemetry rate {(tel.pos_count - count0) / elapsed:.1f} Hz, cycles with the person out of view: {lost}")
    need = 0.85 if a.noisy else 0.95
    ok_all = True
    for s, name in enumerate(SEGMENT_NAMES):
        d = stats.get(s)
        if not d:
            continue
        rs = [x[0] for x in d]
        bs = [x[1] for x in d]
        inband = sum(1 for x in rs if dist - band <= x <= dist + band) / len(rs)
        moved = math.hypot(seg_pos1[s][0] - seg_pos0[s][0], seg_pos1[s][1] - seg_pos0[s][1])
        verdict = ""
        if name in ("walk", "turn+walk"):
            ok = inband >= need and max(bs) <= 10.0
            verdict = "PASS" if ok else "FAIL"
            ok_all &= ok
        elif name == "stop":
            ok = dist - band <= last_range[s] <= dist + band
            verdict = "PASS (recovered to station)" if ok else "FAIL (not back in band)"
            ok_all &= ok
        elif name == "stand2":
            ok = moved < 0.5
            verdict = "PASS (held position)" if ok else "FAIL (backed away)"
            ok_all &= ok
        print(f"{name:10s} range min {min(rs):4.2f} mean {sum(rs)/len(rs):4.2f} max {max(rs):4.2f} | "
              f"in band {inband*100:5.1f}% | bearing max {max(bs):4.1f} (peak {peak[s]:4.1f}) deg | "
              f"drone moved {moved:5.2f} m  {verdict}")
    # the approach segment is shorter than the 3 s settling time, so it is judged on movement alone
    ai = SEGMENT_NAMES.index("approach")
    if ai in seg_pos0:
        moved = math.hypot(seg_pos1[ai][0] - seg_pos0[ai][0], seg_pos1[ai][1] - seg_pos0[ai][1])
        ok = moved < 0.5
        ok_all &= ok
        print(f"approach   drone moved {moved:5.2f} m while the person walked toward it  "
              f"{'PASS (hovered)' if ok else 'FAIL (moved)'}")
    print("\nRESULT:", "PASS" if ok_all else "FAIL")
    vt.set_mode(m, "LAND")
    print("landing")


if __name__ == "__main__":
    main()
