"""Autopilot-side protections in decision 46, tested in SITL without a compass or GPS.

Three questions, five cases, each in its own SITL instance, all at the same time:

  silent     Does the autopilot land by itself when the Jetson goes silent?
             Heartbeats and commands are sent as the Jetson would send them, then
             both stop. Pass: Land mode within 6 s, a descent no faster than
             1.5 m/s, on the ground and disarmed within 60 s.
  handover   Is the watchdog's handover accepted? The aircraft flies forward in
             Guided, is commanded to stop, then Loiter is requested with the
             sticks centred. Pass: Loiter accepted, 30 s hold within 1.5 m and
             0.5 m of height, full forward stick gives no more than the Loiter
             speed limit, full throttle stick gives no more than the climb limit.
  cap_calm   Does the autopilot limit Guided speed over the ground when the
  cap_head   command asks for more? Calm air, a 4 m/s headwind and a 4 m/s
  cap_tail   tailwind. A 2.0 m/s command must give 2.0 m/s over the ground in
             all three (pass: 1.8 to 2.2). A 5.0 m/s command and 3.0 m/s climb
             and descent commands show whether the autopilot's own limits apply
             (pass: no more than 2.3 m/s horizontal, 1.3 m/s vertical).

Speeds over the ground come from the simulator's true position, not from the
autopilot's estimate. All cases use the compass-free configuration of decision 42
(route 1): optical flow and rangefinder only, estimator heading wrong by 120 degrees.

The script starts and stops SITL itself. Do not run sim_vehicle.py alongside it.
Run (ArduPilot environment active, from the repository root):
    python sim/decision46_test.py --ardupilot /path/to/ardupilot
    python sim/decision46_test.py --case handover        # one case only
Logs for each case are written to sim/out/decision46_<case>/.
"""
import argparse
import collections
import concurrent.futures
import json
import math
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
import nocompass_test as nc   # noqa: E402

OUT = REPO / "sim" / "out"
GUIDED, LOITER, LAND = 4, 5, 9
ALT_M = 5.0
H_LIMIT, V_LIMIT = 2.0, 1.0          # decision 46, metres per second

# Both spellings where ArduPilot renamed a parameter; SITL ignores names it does not know.
LIMITS = {
    "WP_SPD": H_LIMIT, "WPNAV_SPEED": H_LIMIT * 100,              # Guided and Auto horizontal speed
    "WP_SPD_UP": V_LIMIT, "WPNAV_SPEED_UP": V_LIMIT * 100,
    "WP_SPD_DN": V_LIMIT, "WPNAV_SPEED_DN": V_LIMIT * 100,
    "LOIT_SPEED_MS": H_LIMIT, "LOIT_SPEED": H_LIMIT * 100,        # Loiter horizontal speed
    "PILOT_SPD_UP": V_LIMIT, "PILOT_SPEED_UP": V_LIMIT * 100,     # pilot climb and descent in Loiter and AltHold
    "PILOT_SPD_DN": V_LIMIT, "PILOT_SPEED_DN": V_LIMIT * 100,
}
BASE = dict(nc.COMMON, **{
    "EK3_SRC1_YAW": 0,                                            # route 1: no heading source
    "RNGFND1_MAX": 30, "RNGFND1_MAX_CM": 3000,                    # room for the climb test
    "FLTMODE_CH": 0,                                              # no mode switch: modes come from the test only
}, **LIMITS)
HEADING = nc.HOME[3]                                              # the aircraft truly faces this way
CASES = {
    "silent": ("Jetson goes silent: the autopilot lands", {"FS_GCS_ENABLE": 5, "FS_GCS_TIMEOUT": 3}),
    "handover": ("Handover to Loiter after a stick movement", {}),
    "cap_calm": ("Guided speed limits, calm air", {}),
    "cap_head": ("Guided speed limits, 4 m/s headwind", {"SIM_WIND_SPD": 4, "SIM_WIND_DIR": HEADING}),
    "cap_tail": ("Guided speed limits, 4 m/s tailwind", {"SIM_WIND_SPD": 4, "SIM_WIND_DIR": (HEADING + 180) % 360}),
}
REPORT = sorted(set(LIMITS) | {"FS_GCS_ENABLE", "FS_GCS_TIMEOUT", "FS_OPTIONS", "GUID_TIMEOUT", "EK3_SRC1_YAW",
                               "ANGLE_MAX", "ATC_ANGLE_MAX", "LOIT_ANG_MAX"})


class Link(nc.Link):
    """nocompass_test.Link plus: heartbeats as the Jetson, stick input, and speed histories."""

    def __init__(self, m, log):
        super().__init__(m, log, feed_heading=False)
        self.heartbeat, self._last_hb = False, 0.0
        self.sticks, self._last_rc = None, 0.0           # (roll, pitch, throttle, yaw) pulse widths
        self.track = collections.deque(maxlen=6000)      # (time, north, east), simulator truth
        self.vert = collections.deque(maxlen=6000)       # (time, down speed, height), autopilot estimate
        self.version = None

    def _handle(self, msg):
        super()._handle(msg)
        t = msg.get_type()
        if t == "SIMSTATE" and self.truth is not None:
            self.track.append((time.time(), self.truth[0], self.truth[1]))
        elif t == "LOCAL_POSITION_NED":
            self.vert.append((time.time(), msg.vz, -msg.z))
        elif t == "AUTOPILOT_VERSION":
            v = msg.flight_sw_version
            git = bytes(msg.flight_custom_version).split(b"\0")[0].decode(errors="replace")
            self.version = f"{(v >> 24) & 255}.{(v >> 16) & 255}.{(v >> 8) & 255} ({git})"

    def pump(self, seconds=0.0, until=None):
        end = time.time() + seconds
        while True:
            msg = self.m.recv_match(blocking=True, timeout=0.05)
            while msg is not None:
                self._handle(msg)
                msg = self.m.recv_match(blocking=False)
            now = time.time()
            if self.heartbeat and now - self._last_hb >= 1.0:
                self.m.mav.heartbeat_send(6, 8, 0, 0, 0)          # type ground station, no autopilot
                self._last_hb = now
            if self.sticks is not None and now - self._last_rc >= 0.1:
                r, p, th, y = self.sticks
                self.m.mav.rc_channels_override_send(self.m.target_system, 1, r, p, th, y,
                                                     65535, 65535, 65535, 65535)
                self._last_rc = now
            if self.cmd is not None and now - self._last_cmd >= 0.1:
                vx, vy, vz, yr = self.cmd
                self.m.mav.set_position_target_local_ned_send(
                    0, self.m.target_system, 1, nc.FRAME_BODY_OFFSET_NED, nc.MASK_VEL_YAWRATE,
                    0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yr)
                self._last_cmd = now
            if until is not None and until():
                return True
            if now >= end:
                return False

    # measurements ---------------------------------------------------------
    def ground_speed(self, window=1.0):
        """True speed over the ground across the last `window` seconds."""
        if len(self.track) < 2:
            return 0.0
        t1, n1, e1 = self.track[-1]
        for t0, n0, e0 in reversed(self.track):
            if t1 - t0 >= window:
                return math.hypot(n1 - n0, e1 - e0) / (t1 - t0)
        return 0.0

    def along_nose(self, since, yaw):
        """True distance travelled along `yaw` and sideways since time `since`."""
        pts = [(n, e) for t, n, e in self.track if t >= since]
        if len(pts) < 2:
            return 0.0, 0.0
        dn, de = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
        return dn * math.cos(yaw) + de * math.sin(yaw), -dn * math.sin(yaw) + de * math.cos(yaw)

    def vertical_peak(self, since):
        """(fastest climb, fastest descent) in m/s since time `since`, from the autopilot's estimate."""
        vz = [v for t, v, _ in self.vert if t >= since]
        return (max(0.0, -min(vz)), max(0.0, max(vz))) if vz else (0.0, 0.0)

    def height(self):
        return self.vert[-1][2] if self.vert else float("nan")

    def pos(self):
        return self.truth[0], self.truth[1]


def bring_up(link, m, params, step, numbers):
    """Boot, estimator origin, Guided, arm and take off. Raises StopIteration on a failed step."""
    for msg_id, hz in nc.INTERVALS_HZ.items():
        m.mav.command_long_send(m.target_system, 1, 511, 0, msg_id, 1e6 / hz, 0, 0, 0, 0, 0)
    link.pump(12)
    m.mav.command_long_send(m.target_system, 1, 512, 0, 148, 0, 0, 0, 0, 0, 0)   # ask for the firmware version
    link.pump(1, lambda: link.version is not None)
    numbers["firmware"] = link.version

    applied = {}
    for name in REPORT:
        m.mav.param_request_read_send(m.target_system, 1, name.encode(), -1)
        got = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.6)
        if got is not None and got.param_id == name:
            applied[name] = round(got.param_value, 3)
    link.say("parameters in effect: " + json.dumps(applied))
    numbers["parameters"] = applied
    if abs(applied.get("EK3_SRC1_YAW", -1)) > 0.1:
        raise RuntimeError("EK3_SRC1_YAW did not take effect")

    def send_origin():
        m.mav.set_gps_global_origin_send(m.target_system, int(nc.HOME[0] * 1e7),
                                         int(nc.HOME[1] * 1e7), int(nc.HOME[2] * 1000))
    for _ in range(10):
        send_origin()
        if link.pump(2, lambda: link.origin_set):
            break
    rel = lambda: bool(link.ekf_flags & (0x008 | 0x100))
    link.pump(45, rel)
    if not step("estimator ready without a compass or GPS", link.origin_set and rel(), f"flags 0x{link.ekf_flags:04x}"):
        raise StopIteration
    ack = link.command(176, 1, GUIDED)
    link.pump(2, lambda: link.mode == GUIDED)
    for _ in range(4):
        if link.mode == GUIDED:
            link.command(400, 1)
            if link.pump(3, lambda: link.armed):
                break
        link.pump(5)
    link.command(22, 0, 0, 0, 0, 0, 0, ALT_M)
    up = lambda: link.height() >= ALT_M - 0.5
    link.pump(40, up)
    if not step("Guided, armed and at height", link.mode == GUIDED and link.armed and up(),
                f"mode {link.mode}, armed {link.armed}, height {link.height():.1f} m (ack {ack})"):
        raise StopIteration
    if link.truth is None:
        step("simulator truth available", False, "no SIMSTATE messages")
        raise StopIteration
    link.cmd = (0, 0, 0, 0)
    link.pump(8)


def steady_speed(link, cmd, seconds=9.0, measure=4.0):
    """Stream `cmd`, return (true ground speed over the last `measure` seconds, along, sideways)."""
    yaw = link.truth[2]
    t0 = time.time()
    link.cmd = cmd
    link.pump(seconds)
    speed = link.ground_speed(measure)
    along, cross = link.along_nose(t0, yaw)
    return speed, along, cross


def stop(link, limit_s=10.0):
    """Command zero velocity; return seconds until the true speed is below 0.3 m/s."""
    t0 = time.time()
    link.cmd = (0, 0, 0, 0)
    link.pump(1.0)
    link.pump(limit_s, lambda: link.ground_speed(0.5) < 0.3)
    return time.time() - t0


def case_silent(link, step, numbers):
    link.heartbeat = True
    link.pump(15)
    step("no failsafe while the Jetson's heartbeat arrives", link.mode == GUIDED and link.armed,
         f"mode {link.mode} after 15 s of heartbeats")
    link.cmd = (1.0, 0, 0, 0)
    link.pump(5)
    n0, e0 = link.pos()
    h0 = link.height()
    link.heartbeat, link.cmd = False, None                 # the Jetson dies: no heartbeats, no commands
    t0 = time.time()
    link.say("heartbeats and commands stopped")
    link.pump(15, lambda: link.mode == LAND)
    t_land = time.time() - t0
    numbers["seconds_to_land_mode"] = round(t_land, 1)
    step("autopilot switches to Land by itself", link.mode == LAND and t_land <= 6.0,
         f"mode {link.mode} after {t_land:.1f} s of silence (FS_GCS_TIMEOUT 3 s)")
    t1 = time.time()
    link.pump(60, lambda: not link.armed)
    _, descent = link.vertical_peak(t1)
    n1, e1 = link.pos()
    numbers["landing"] = {"from_height_m": round(h0, 1), "seconds": round(time.time() - t1, 1),
                          "fastest_descent_mps": round(descent, 2),
                          "drift_m": round(math.hypot(n1 - n0, e1 - e0), 2)}
    step("controlled descent, on the ground and disarmed", (not link.armed) and descent <= 1.5,
         f"from {h0:.1f} m in {time.time() - t1:.0f} s, fastest descent {descent:.2f} m/s, "
         f"{math.hypot(n1 - n0, e1 - e0):.1f} m from where the Jetson went silent")


def case_handover(link, step, numbers):
    link.sticks = (1500, 1500, 1500, 1500)                 # pilot's sticks centred, throttle in the middle
    speed, along, _ = steady_speed(link, (H_LIMIT, 0, 0, 0), seconds=6.0, measure=2.0)
    numbers["guided_speed_before_mps"] = round(speed, 2)
    t_stop = stop(link)
    numbers["seconds_to_stop"] = round(t_stop, 1)
    step("hover command stops the aircraft", link.ground_speed(0.5) < 0.3,
         f"from {speed:.1f} m/s to below 0.3 m/s in {t_stop:.1f} s")
    link.cmd = None
    ack = link.command(176, 1, LOITER)
    link.pump(3, lambda: link.mode == LOITER)
    if not step("Loiter accepted without a compass or GPS", link.mode == LOITER, f"mode {link.mode}, ack {ack}"):
        raise StopIteration

    n0, e0 = link.pos()
    h0, worst, worst_h = link.height(), 0.0, 0.0
    t_end = time.time() + 30
    while time.time() < t_end and link.armed:
        link.pump(1)
        n, e = link.pos()
        worst = max(worst, math.hypot(n - n0, e - e0))
        worst_h = max(worst_h, abs(link.height() - h0))
    numbers["loiter_hold"] = {"worst_m": round(worst, 2), "worst_height_change_m": round(worst_h, 2)}
    step("30 s hold in Loiter, sticks centred", link.armed and link.mode == LOITER and worst <= 1.5 and worst_h <= 0.5,
         f"worst {worst:.2f} m from the handover point, height within {worst_h:.2f} m")

    yaw, t0 = link.truth[2], time.time()
    link.sticks = (1500, 1000, 1500, 1500)                 # full forward stick
    link.pump(9)
    speed = link.ground_speed(4.0)
    along, cross = link.along_nose(t0, yaw)
    link.sticks = (1500, 1500, 1500, 1500)
    link.pump(6)
    numbers["loiter_full_stick"] = {"speed_mps": round(speed, 2), "along_nose_m": round(along, 1),
                                    "sideways_m": round(cross, 1)}
    step("full forward stick in Loiter stays at the speed limit", 1.2 <= speed <= H_LIMIT + 0.3,
         f"{speed:.2f} m/s over the ground (limit {H_LIMIT}), {along:+.1f} m along the nose, {cross:+.1f} m sideways")
    step("stick forward moves the aircraft along its nose", along >= 5.0 and abs(cross) <= 0.3 * abs(along),
         f"{along:+.1f} m along, {cross:+.1f} m sideways, with the estimator heading wrong by 120 degrees")

    t0 = time.time()
    link.sticks = (1500, 1500, 2000, 1500)                 # full throttle stick
    link.pump(3)
    link.sticks = (1500, 1500, 1500, 1500)
    link.pump(3)
    climb, _ = link.vertical_peak(t0)
    t0 = time.time()
    link.sticks = (1500, 1500, 1000, 1500)                 # throttle stick at the bottom
    link.pump(2.5, lambda: link.height() < 3.0)
    link.sticks = (1500, 1500, 1500, 1500)
    link.pump(3)
    _, descent = link.vertical_peak(t0)
    numbers["loiter_vertical"] = {"climb_mps": round(climb, 2), "descent_mps": round(descent, 2)}
    step("throttle stick in Loiter stays at the climb and descent limit",
         climb <= V_LIMIT + 0.3 and descent <= V_LIMIT + 0.3,
         f"fastest climb {climb:.2f} m/s, fastest descent {descent:.2f} m/s (limit {V_LIMIT})")
    link.sticks = None


def case_cap(link, step, numbers):
    speed, along, cross = steady_speed(link, (H_LIMIT, 0, 0, 0))
    numbers["command_2_mps"] = {"speed_mps": round(speed, 2), "along_nose_m": round(along, 1),
                                "sideways_m": round(cross, 1)}
    step("a 2.0 m/s command gives 2.0 m/s over the ground", 1.8 <= speed <= 2.2 and along > 0,
         f"{speed:.2f} m/s, {along:+.1f} m along the nose, {cross:+.1f} m sideways")
    stop(link)
    speed, along, cross = steady_speed(link, (5.0, 0, 0, 0))
    numbers["command_5_mps"] = {"speed_mps": round(speed, 2), "along_nose_m": round(along, 1)}
    step("a 5.0 m/s command is held to the autopilot's limit", speed <= H_LIMIT + 0.3,
         f"{speed:.2f} m/s over the ground (limit {H_LIMIT})")
    stop(link)

    t0 = time.time()
    link.cmd = (0, 0, -3.0, 0)                             # climb at 3 m/s
    link.pump(2.5, lambda: link.height() > 20.0)
    link.cmd = (0, 0, 0, 0)
    link.pump(4)
    climb, _ = link.vertical_peak(t0)
    t0 = time.time()
    link.cmd = (0, 0, 3.0, 0)                              # descend at 3 m/s
    link.pump(2.0, lambda: link.height() < 3.5)
    link.cmd = (0, 0, 0, 0)
    link.pump(4)
    _, descent = link.vertical_peak(t0)
    numbers["command_3_mps_vertical"] = {"climb_mps": round(climb, 2), "descent_mps": round(descent, 2)}
    step("3.0 m/s climb and descent commands are held to the autopilot's limit",
         climb <= V_LIMIT + 0.3 and descent <= V_LIMIT + 0.3,
         f"fastest climb {climb:.2f} m/s, fastest descent {descent:.2f} m/s (limit {V_LIMIT})")


RUNNERS = {"silent": case_silent, "handover": case_handover,
           "cap_calm": case_cap, "cap_head": case_cap, "cap_tail": case_cap}


def run_case(case, instance, root):
    import subprocess
    title = CASES[case][0]
    outdir = OUT / f"decision46_{case}"
    steps, numbers = [], {}
    proc = log = link = None

    def step(name, ok, detail=""):
        steps.append((name, bool(ok), detail))
        link.say(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        return bool(ok)

    try:
        params = dict(BASE, **CASES[case][1])
        proc = nc.start_sitl(pathlib.Path(root), params, instance, outdir)
        log = open(outdir / "test.log", "w")
        m = nc.connect(5760 + 10 * instance, log)
        link = Link(m, log)
        link.say(f"{title}: connected")
        bring_up(link, m, params, step, numbers)
        RUNNERS[case](link, step, numbers)
    except StopIteration:
        pass
    except Exception as exc:
        steps.append(("test ran to completion", False, f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            if link is not None and link.armed:
                link.cmd, link.sticks = None, None
                link.command(176, 1, LAND)
                link.pump(40, lambda: not link.armed)
        except Exception:
            pass
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log is not None:
            log.close()

    passed = bool(steps) and all(ok for _, ok, _ in steps)
    texts = link.texts[-12:] if link is not None else []
    result = {"case": case, "title": title, "passed": passed, "steps": steps,
              "numbers": numbers, "last_autopilot_messages": texts}
    (outdir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=list(CASES), action="append", help="run one case (default: all, in parallel)")
    ap.add_argument("--ardupilot", help="ArduPilot source directory (default: found from sim_vehicle.py on PATH)")
    args = ap.parse_args()
    root = nc.find_ardupilot(args.ardupilot)
    cases = args.case or list(CASES)
    print(f"ArduPilot: {root}")
    print(f"Running {len(cases)} case(s) in parallel, about 4 minutes. Logs: {OUT}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(run_case, c, i, str(root)) for i, c in enumerate(cases)]
        results = [f.result() for f in futures]
    for r in results:
        print(f"\n{r['title']}")
        for name, ok, detail in r["steps"]:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
        for key, value in r["numbers"].items():
            print(f"        {key}: {value}")
        if not r["passed"] and r["last_autopilot_messages"]:
            print("        last autopilot messages:")
            for text in r["last_autopilot_messages"]:
                print(f"          {text}")
        print(f"  RESULT: {'PASS' if r['passed'] else 'FAIL'}")
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
