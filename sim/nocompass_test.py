"""Compass-free heading test in SITL (decision 42, routes 1 and 2).

Question: with no compass and no GPS, will ArduPilot accept the project's
body-frame velocity commands in Guided mode and hold position on optical flow?

Two cases run at the same time, each in its own SITL instance:
  none    route 1: no heading source (EK3_SRC1_YAW 0)
  jetson  route 2: heading supplied by the companion computer (EK3_SRC1_YAW 6).
          The script integrates the gyro rates from telemetry, starting from an
          arbitrary heading of zero, and sends the result back as an external
          navigation heading, as the Jetson would.

Both cases: simulated GPS off, compass off, simulated optical flow and downward
rangefinder on, position estimate from optical flow only.

Each case passes if all of these hold, measured against the simulator's true
position and heading (not the autopilot's estimate):
  - Guided mode is accepted, the aircraft arms and takes off
  - a forward command moves the aircraft along its true nose direction
    (at least 2.5 m along, at most 1.0 m sideways), before and after a turn
  - a yaw-rate command turns the aircraft about 90 degrees
  - the aircraft stays within 2.0 m of its start point over a two-minute hold

The script starts and stops SITL itself. Do not run sim_vehicle.py alongside it.
Run (ArduPilot environment active, from the repository root):
    python sim/nocompass_test.py
    python sim/nocompass_test.py --case none        # one case only
    python sim/nocompass_test.py --ardupilot /path/to/ardupilot
Logs for each case are written to sim/out/nocompass_<case>/.
"""
import argparse
import concurrent.futures
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time

from pymavlink import mavutil

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "sim" / "out"

HOME = (-35.363262, 149.165237, 584.0, 120.0)   # SITL default field. The aircraft truly faces 120 degrees
                                                # while the estimator starts from 0, so its heading is wrong by 120.
MASK_VEL_YAWRATE = 0x05C7
FRAME_BODY_OFFSET_NED = 9
GUIDED, LAND = 4, 9
TAKEOFF_M = 3.0

# Parameter names differ between ArduPilot versions. Unknown names in a defaults
# file are ignored by SITL, so both spellings are listed where a name has changed.
COMMON = {
    "SIM_GPS1_ENABLE": 0, "SIM_GPS_DISABLE": 1,      # simulated GPS off
    "GPS1_TYPE": 0, "GPS_TYPE": 0,                   # GPS driver off
    "COMPASS_ENABLE": 0, "COMPASS_USE": 0, "COMPASS_USE2": 0, "COMPASS_USE3": 0,
    "SIM_FLOW_ENABLE": 1, "FLOW_TYPE": 10,           # simulated optical flow
    "RNGFND1_TYPE": 1, "RNGFND1_PIN": 0, "RNGFND1_SCALING": 12.12,   # simulated downward rangefinder
    "RNGFND1_ORIENT": 25, "RNGFND1_MIN": 0, "RNGFND1_MIN_CM": 0,     # (the settings ArduPilot's own flow tests use)
    "RNGFND1_MAX": 8, "RNGFND1_MAX_CM": 800,
    "EK3_SRC1_POSXY": 0, "EK3_SRC1_VELXY": 5, "EK3_SRC1_POSZ": 1, "EK3_SRC1_VELZ": 0,
    "EK3_SRC_OPTIONS": 0,
    "ARMING_SKIPCHK": 4108, "ARMING_CHECK": 1048534,  # skip the compass and GPS arming checks only
    "WP_YAW_BEHAVIOR": 0,
}
CASES = {
    "none": ("Route 1: no heading source", {"EK3_SRC1_YAW": 0}),
    "jetson": ("Route 2: heading supplied by the companion computer",
               {"EK3_SRC1_YAW": 6, "VISO_TYPE": 1}),
}
INTERVALS_HZ = {30: 50, 32: 10, 193: 2, 164: 10, 108: 10, 49: 1}   # ATTITUDE, LOCAL_POSITION_NED, EKF_STATUS_REPORT, SIMSTATE, SIM_STATE, GPS_GLOBAL_ORIGIN


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Link:
    """One MAVLink connection: keeps the latest state and streams commands while pumping."""

    def __init__(self, m, log, feed_heading):
        self.m, self.log, self.feed_heading = m, log, feed_heading
        self.truth = None            # (north, east, yaw) from the simulator
        self.truth0 = None
        self.est = None              # (north, east, down) from the autopilot
        self.est_yaw = 0.0
        self.ekf_flags = 0
        self.mode, self.armed = None, False
        self.origin_set = False
        self.texts = []
        self.acks = {}
        self.cmd = None              # (vx, vy, vz, yaw_rate) streamed at 10 Hz when set
        self._last_cmd = 0.0
        self.fed_yaw, self._att_ms, self.fed_count = 0.0, None, 0

    def say(self, text):
        line = f"{time.strftime('%H:%M:%S')} {text}"
        self.log.write(line + "\n")
        self.log.flush()

    def _truth(self, lat, lon, yaw):
        if abs(lat) > 1000:          # some versions send degrees * 1e7
            lat, lon = lat / 1e7, lon / 1e7
        if self.truth0 is None:
            self.truth0 = (lat, lon)
        n = (lat - self.truth0[0]) * 111319.5
        e = (lon - self.truth0[1]) * 111319.5 * math.cos(math.radians(self.truth0[0]))
        self.truth = (n, e, yaw)

    def _handle(self, msg):
        t = msg.get_type()
        if t == "ATTITUDE":
            self.est_yaw = msg.yaw
            if self.feed_heading:
                if self._att_ms is not None:
                    dt = (msg.time_boot_ms - self._att_ms) / 1000.0
                    if 0 < dt < 0.5:
                        rate = (msg.pitchspeed * math.sin(msg.roll)
                                + msg.yawspeed * math.cos(msg.roll)) / max(0.2, math.cos(msg.pitch))
                        self.fed_yaw = wrap(self.fed_yaw + rate * dt)
                self._att_ms = msg.time_boot_ms
                self.m.mav.vision_position_estimate_send(
                    int(time.time() * 1e6), 0.0, 0.0, 0.0, msg.roll, msg.pitch, self.fed_yaw)
                self.fed_count += 1
        elif t == "SIMSTATE":
            self._truth(msg.lat, msg.lng, msg.yaw)
        elif t == "SIM_STATE" and self.truth is None:
            self._truth(msg.lat, msg.lon, msg.yaw)
        elif t == "LOCAL_POSITION_NED":
            self.est = (msg.x, msg.y, msg.z)
        elif t == "EKF_STATUS_REPORT":
            self.ekf_flags = msg.flags
        elif t == "HEARTBEAT" and msg.get_srcComponent() == 1:
            self.mode = msg.custom_mode
            self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        elif t == "GPS_GLOBAL_ORIGIN":
            self.origin_set = True
        elif t == "STATUSTEXT":
            self.texts.append(msg.text)
            self.say(f"  autopilot: {msg.text}")
        elif t == "COMMAND_ACK":
            self.acks[msg.command] = msg.result

    def pump(self, seconds=0.0, until=None):
        """Read messages for `seconds`, or until `until()` is true. Returns until()'s value."""
        end = time.time() + seconds
        while True:
            msg = self.m.recv_match(blocking=True, timeout=0.05)
            while msg is not None:
                self._handle(msg)
                msg = self.m.recv_match(blocking=False)
            now = time.time()
            if self.cmd is not None and now - self._last_cmd >= 0.1:
                vx, vy, vz, yr = self.cmd
                self.m.mav.set_position_target_local_ned_send(
                    0, self.m.target_system, 1, FRAME_BODY_OFFSET_NED, MASK_VEL_YAWRATE,
                    0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yr)
                self._last_cmd = now
            if until is not None and until():
                return True
            if now >= end:
                return False

    def command(self, cmd, *p, wait=3.0):
        p = list(p) + [0] * (7 - len(p))
        self.acks.pop(cmd, None)
        self.m.mav.command_long_send(self.m.target_system, 1, cmd, 0, *p)
        self.pump(wait, lambda: cmd in self.acks)
        return self.acks.get(cmd)


def start_sitl(root, params, instance, outdir):
    binary = root / "build" / "sitl" / "bin" / "arducopter"
    if not binary.exists():
        raise RuntimeError(f"SITL binary not found at {binary}. Run sim_vehicle.py once to build it.")
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    parm = outdir / "case.parm"
    parm.write_text("".join(f"{k} {v}\n" for k, v in params.items()))
    defaults = root / "Tools" / "autotest" / "default_params" / "copter.parm"
    args = [str(binary), "-w", "--model", "+", "--speedup", "1", f"-I{instance}",
            "--home", ",".join(str(v) for v in HOME),
            "--defaults", f"{defaults},{parm}", "--sim-address=127.0.0.1"]
    out = open(outdir / "sitl.log", "w")
    return subprocess.Popen(args, cwd=outdir, stdout=out, stderr=subprocess.STDOUT)


def connect(port, log):
    last = None
    for _ in range(40):
        try:
            m = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}", source_system=255)
            m.wait_heartbeat(timeout=10)
            if m.target_system:
                return m
        except Exception as exc:      # SITL not listening yet
            last = exc
        time.sleep(0.5)
    raise RuntimeError(f"could not connect to SITL on port {port}: {last}")


def run_case(case, instance, root, hold_s=120.0, wind=0.0):
    title = CASES[case][0]
    outdir = OUT / f"nocompass_{case}"
    steps, numbers = [], {}
    proc = None
    log = None

    def step(name, ok, detail=""):
        steps.append((name, bool(ok), detail))
        link.say(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        return bool(ok)

    try:
        params = dict(COMMON, **CASES[case][1])
        if wind:
            params.update({"SIM_WIND_SPD": wind, "SIM_WIND_DIR": 0})
        proc = start_sitl(pathlib.Path(root), params, instance, outdir)
        log = open(outdir / "test.log", "w")
        m = connect(5760 + 10 * instance, log)
        link = Link(m, log, feed_heading=(case == "jetson"))
        link.say(f"{title}: connected")
        for msg_id, hz in INTERVALS_HZ.items():
            m.mav.command_long_send(m.target_system, 1, 511, 0, msg_id, 1e6 / hz, 0, 0, 0, 0, 0)
        link.pump(12)                                   # boot, gyro calibration, tilt alignment

        # which of the requested parameters this ArduPilot version actually has
        applied = {}
        for name, want in params.items():
            m.mav.param_request_read_send(m.target_system, 1, name.encode(), -1)
            got = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.6)
            if got is not None and got.param_id == name:
                applied[name] = got.param_value
        link.say("parameters in effect: " + json.dumps(applied))
        numbers["parameters"] = applied
        if abs(applied.get("EK3_SRC1_YAW", -1) - params["EK3_SRC1_YAW"]) > 0.1:
            raise RuntimeError("EK3_SRC1_YAW did not take effect")

        # the estimator needs an origin when there is no GPS to give it one
        def send_origin():
            m.mav.set_gps_global_origin_send(m.target_system, int(HOME[0] * 1e7),
                                             int(HOME[1] * 1e7), int(HOME[2] * 1000))
        for _ in range(10):
            send_origin()
            if link.pump(2, lambda: link.origin_set):
                break
        step("estimator origin set", link.origin_set)

        rel = lambda: bool(link.ekf_flags & (0x008 | 0x100))   # horizontal position, actual or predicted
        link.pump(45, rel)
        step("estimator reports a horizontal position estimate", rel(), f"flags 0x{link.ekf_flags:04x}")

        ack = link.command(176, 1, GUIDED)
        link.pump(2, lambda: link.mode == GUIDED)
        if not step("Guided mode accepted", link.mode == GUIDED, f"ack {ack}"):
            raise StopIteration

        for attempt in range(4):
            ack = link.command(400, 1)
            if link.pump(3, lambda: link.armed):
                break
            link.pump(5)
        if not step("armed", link.armed, f"ack {ack}"):
            raise StopIteration

        ack = link.command(22, 0, 0, 0, 0, 0, 0, TAKEOFF_M)
        up = lambda: link.est is not None and -link.est[2] >= TAKEOFF_M - 0.5
        link.pump(30, up)
        if not step("takeoff", up(), f"ack {ack}, height {-link.est[2] if link.est else float('nan'):.1f} m"):
            raise StopIteration
        if link.truth is None:
            step("simulator truth available", False, "no SIMSTATE or SIM_STATE messages")
            raise StopIteration
        link.cmd = (0, 0, 0, 0)
        link.pump(8)                                    # settle

        numbers["estimated_minus_true_heading_deg_at_start"] = round(
            math.degrees(wrap(link.est_yaw - link.truth[2])), 1)

        def forward_leg(name):
            n0, e0, yaw0 = link.truth
            link.cmd = (1.0, 0, 0, 0)
            link.pump(4)
            link.cmd = (0, 0, 0, 0)
            link.pump(4)
            dn, de = link.truth[0] - n0, link.truth[1] - e0
            along = dn * math.cos(yaw0) + de * math.sin(yaw0)
            cross = -dn * math.sin(yaw0) + de * math.cos(yaw0)
            numbers[name] = {"along_m": round(along, 2), "sideways_m": round(cross, 2)}
            return step(name, along >= 2.5 and abs(cross) <= 1.0,
                        f"{along:.2f} m along the nose, {cross:+.2f} m sideways")

        forward_leg("forward command, first leg")

        yaw0 = link.truth[2]
        link.cmd = (0, 0, 0, math.radians(30))
        link.pump(3)
        link.cmd = (0, 0, 0, 0)
        link.pump(4)
        turned = math.degrees(wrap(link.truth[2] - yaw0))
        numbers["turn_deg"] = round(turned, 1)
        step("yaw-rate command turns the aircraft", 70 <= turned <= 110, f"{turned:.0f} degrees")

        forward_leg("forward command after the turn")

        n0, e0, _ = link.truth
        worst, t_end = 0.0, time.time() + hold_s
        while time.time() < t_end:
            link.pump(1)
            worst = max(worst, math.hypot(link.truth[0] - n0, link.truth[1] - e0))
            if not link.armed:
                break
        final = math.hypot(link.truth[0] - n0, link.truth[1] - e0)
        numbers["hold"] = {"worst_m": round(worst, 2), "final_m": round(final, 2)}
        step(f"{hold_s:.0f} s hold on optical flow", link.armed and worst <= 2.0,
             f"worst {worst:.2f} m, final {final:.2f} m from the start point")
        numbers["estimated_minus_true_heading_deg_at_end"] = round(
            math.degrees(wrap(link.est_yaw - link.truth[2])), 1)
        if case == "jetson":
            numbers["heading_messages_sent"] = link.fed_count
    except StopIteration:
        pass
    except Exception as exc:
        steps.append(("test ran to completion", False, f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            link.cmd = None
            link.command(176, 1, LAND)
            link.pump(25, lambda: not link.armed)
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
    texts = link.texts[-12:] if "link" in locals() else []
    result = {"case": case, "title": title, "passed": passed,
              "steps": steps, "numbers": numbers, "last_autopilot_messages": texts}
    (outdir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def find_ardupilot(arg):
    if arg:
        return pathlib.Path(arg).expanduser()
    if os.environ.get("ARDUPILOT_HOME"):
        return pathlib.Path(os.environ["ARDUPILOT_HOME"]).expanduser()
    sim_vehicle = shutil.which("sim_vehicle.py")
    if sim_vehicle:
        return pathlib.Path(sim_vehicle).resolve().parents[2]
    sys.exit("ArduPilot not found. Pass --ardupilot, set ARDUPILOT_HOME, or activate the ArduPilot environment.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=list(CASES), action="append", help="run one case (default: all, in parallel)")
    ap.add_argument("--wind", type=float, default=0.0, help="simulated wind speed in m/s (default 0)")
    ap.add_argument("--hold-seconds", type=float, default=120.0, help="length of the position hold (default 120)")
    ap.add_argument("--ardupilot", help="ArduPilot source directory (default: found from sim_vehicle.py on PATH)")
    args = ap.parse_args()
    root = find_ardupilot(args.ardupilot)
    cases = args.case or list(CASES)
    print(f"ArduPilot: {root}")
    print(f"Running {len(cases)} case(s) in parallel, about {(args.hold_seconds + 150) / 60:.1f} minutes. Logs: {OUT}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(run_case, c, i, str(root), args.hold_seconds, args.wind) for i, c in enumerate(cases)]
        results = [f.result() for f in futures]

    for r in results:
        print(f"\n{r['title']}")
        for name, ok, detail in r["steps"]:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
        for key, value in r["numbers"].items():
            if key != "parameters":
                print(f"        {key}: {value}")
        if not r["passed"] and r["last_autopilot_messages"]:
            print("        last autopilot messages:")
            for text in r["last_autopilot_messages"]:
                print(f"          {text}")
        print(f"  RESULT: {'PASS' if r['passed'] else 'FAIL'}")
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
