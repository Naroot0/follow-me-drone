"""Velocity command test against ArduPilot SITL (Stage 1).

Proves three things, each of which fails silently if wrong:
  1. the type mask makes ArduPilot use the velocity fields and ignore the rest,
  2. the frame is body-relative (x forward, y right, z down),
  3. commands at 10 Hz keep Guided mode alive (it times out below about 3 Hz).

Method: Guided, arm, take off to 5 m, then fly a square at 1 m/s, 3 s per leg,
measuring the copter's own LOCAL_POSITION_NED after each leg. Each leg should
move about 3 m in the commanded direction. Then land.

SITL side (one terminal):
    source ~/.venvs/ardupilot/bin/activate
    cd /Volumes/ArduPilot/ardupilot/ArduCopter && sim_vehicle.py --console --map
MAVProxy forwards telemetry to 127.0.0.1:14550 (QGroundControl) and 14551.
This script listens on 14551.

Run (project venv, second terminal):
    python src/control/velocity_test.py
    python src/control/velocity_test.py --conn udpin:0.0.0.0:14551 --speed 1.0 --leg 3
"""
import argparse
import math
import time

from pymavlink import mavutil

# SET_POSITION_TARGET_LOCAL_NED type_mask: a set bit means "ignore this field".
# bits 0-2 position, 3-5 velocity, 6-8 acceleration, 9 force, 10 yaw, 11 yaw rate.
IGNORE_POS = 0b0000_0000_0111
IGNORE_VEL = 0b0000_0011_1000
IGNORE_ACC = 0b0001_1100_0000
IGNORE_FORCE = 0b0010_0000_0000
IGNORE_YAW = 0b0100_0000_0000
IGNORE_YAW_RATE = 0b1000_0000_0000
# ArduPilot rejects the message if the force bit (9) is SET; leave it clear. Wiki value 0x0DC7 = 3527.
MASK_VELOCITY_ONLY = IGNORE_POS | IGNORE_ACC | IGNORE_YAW | IGNORE_YAW_RATE  # 0x0DC7 = 3527
MASK = MASK_VELOCITY_ONLY  # overridable with --mask


def wait_heartbeat(m, timeout=30):
    print("waiting for heartbeat ...")
    hb = m.wait_heartbeat(timeout=timeout)
    if hb is None:
        raise SystemExit("no heartbeat; is SITL running and forwarding to this port?")
    print(f"heartbeat from system {m.target_system} component {m.target_component}")
    # address the autopilot explicitly; the first heartbeat may come from another component
    m.target_component = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
    print(f"addressing system {m.target_system} component {m.target_component}")


def set_mode(m, name):
    mode_id = m.mode_mapping()[name]
    m.mav.set_mode_send(m.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
    t0 = time.time()
    while time.time() - t0 < 10:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb and mavutil.mode_string_v10(hb) == name:
            print(f"mode {name}")
            return
    raise SystemExit(f"mode change to {name} not confirmed")


def arm(m):
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    t0 = time.time()
    while time.time() - t0 < 30:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb and hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            print("armed")
            return
        st = m.recv_match(type="STATUSTEXT", blocking=False)
        if st:
            print("  ", st.text)
    raise SystemExit("arm not confirmed (pre-arm checks? wait for EKF to settle after SITL start)")


def takeoff(m, alt):
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt)
    t0 = time.time()
    while time.time() - t0 < 30:
        drain(m)
        p = m.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=1)
        if p and -p.z >= alt * 0.95:
            print(f"at {-p.z:.1f} m")
            return
    raise SystemExit("takeoff did not reach altitude")


def drain(m):
    """Discard every buffered message so the next read is live, not seconds old."""
    n = 0
    while m.recv_match(blocking=False) is not None:
        n += 1
    return n


def position(m, timeout=2):
    """Fresh LOCAL_POSITION_NED as (north, east, down) in metres."""
    drain(m)
    p = m.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=timeout)
    if p is None:
        raise SystemExit("no LOCAL_POSITION_NED")
    return p.x, p.y, p.z


def heading_deg(m):
    drain(m)
    att = m.recv_match(type="ATTITUDE", blocking=True, timeout=2)
    return math.degrees(att.yaw) % 360 if att else float("nan")


def send_body_velocity(m, vx, vy, vz, frame=mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED):
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component, frame, MASK,
        0, 0, 0,          # position, ignored
        vx, vy, vz,       # velocity m/s
        0, 0, 0,          # acceleration, ignored
        0, 0)             # yaw, yaw rate, ignored


def fly_leg(m, name, vx, vy, speed, leg_s, rate_hz, frame):
    yaw = heading_deg(m)
    x0, y0, z0 = position(m)
    t0 = time.time()
    n = 0
    while time.time() - t0 < leg_s:
        send_body_velocity(m, vx * speed, vy * speed, 0.0, frame)
        n += 1
        drain(m)  # keep the receive buffer live
        time.sleep(1.0 / rate_hz)
    # brake
    for _ in range(int(rate_hz)):
        send_body_velocity(m, 0, 0, 0, frame)
        drain(m)
        time.sleep(1.0 / rate_hz)
    x1, y1, z1 = position(m)
    dn, de = x1 - x0, y1 - y0
    # expected displacement in NED from a body-frame command, given heading
    yr = math.radians(yaw)
    exp_n = speed * leg_s * (vx * math.cos(yr) - vy * math.sin(yr))
    exp_e = speed * leg_s * (vx * math.sin(yr) + vy * math.cos(yr))
    err = math.hypot(dn - exp_n, de - exp_e)
    dist = math.hypot(dn, de)
    ok = dist > 0.5 * speed * leg_s and err < 0.5 * speed * leg_s
    print(f"{name:8s} sent {n:3d} msgs | moved N {dn:+5.2f} E {de:+5.2f} ({dist:4.2f} m) | "
          f"expected N {exp_n:+5.2f} E {exp_e:+5.2f} | err {err:4.2f} m | alt change {-(z1 - z0):+.2f} m | "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="udpin:0.0.0.0:14551")
    ap.add_argument("--alt", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--leg", type=float, default=3.0, help="seconds per leg")
    ap.add_argument("--rate", type=float, default=10.0, help="command rate Hz")
    ap.add_argument("--frame", default="body", choices=["body", "local"],
                    help="body = BODY_OFFSET_NED (x forward); local = LOCAL_NED (x north)")
    ap.add_argument("--no-land", action="store_true")
    ap.add_argument("--mask", type=lambda v: int(v, 0), default=None,
                    help="override type_mask, e.g. 0x0DC7 or 3527")
    a = ap.parse_args()
    global MASK
    if a.mask is not None:
        MASK = a.mask
    frame = (mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED if a.frame == "body"
             else mavutil.mavlink.MAV_FRAME_LOCAL_NED)

    m = mavutil.mavlink_connection(a.conn)
    wait_heartbeat(m)
    print(f"type_mask 0x{MASK:04X} ({MASK}), frame {a.frame}, {a.rate:.0f} Hz")

    set_mode(m, "GUIDED")
    arm(m)
    takeoff(m, a.alt)
    time.sleep(2)
    print(f"heading {heading_deg(m):.0f} deg; flying square at {a.speed} m/s, {a.leg} s per leg")

    results = [
        fly_leg(m, "forward", +1, 0, a.speed, a.leg, a.rate, frame),
        fly_leg(m, "right", 0, +1, a.speed, a.leg, a.rate, frame),
        fly_leg(m, "back", -1, 0, a.speed, a.leg, a.rate, frame),
        fly_leg(m, "left", 0, -1, a.speed, a.leg, a.rate, frame),
    ]
    print(f"\n{sum(results)}/4 legs passed")
    if sum(results) == 0:
        print("nothing moved: the type_mask or frame is being ignored, or Guided timed out")
    if not a.no_land:
        set_mode(m, "LAND")
        print("landing")


if __name__ == "__main__":
    main()
