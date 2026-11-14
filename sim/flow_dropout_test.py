"""Optical-flow dropout test in SITL, compass-free configuration (decision 42, route 1).

Question: with no compass and no GPS, what does the aircraft do when the optical
flow data stops mid-hover? Flow is its only horizontal reference, so this is the
case where it cannot tell where it is.

Four cases run at the same time, each in its own SITL instance:
  short dropout (3 s) and long dropout (15 s), each in calm air and in wind.

Each case: Guided takeoff to 3 m, hover on a zero-velocity command, switch the
simulated flow sensor off for the dropout time, switch it back on, watch for 30 s.
The EKF failsafe action is Land (FS_EKF_ACTION 1), so losing the position estimate
should end in a landing where the aircraft is.

A case passes if the outcome is safe, measured against the simulator's true position:
  - it ends either holding position in Guided (within 2 m of the start point) or
    landed and disarmed
  - it never moves more than 5 m from the start point
  - it never tilts more than 30 degrees
  - if it lands, it touches down at 1.0 m/s or slower
  - the autopilot reports no crash

The script starts and stops SITL itself. Do not run sim_vehicle.py alongside it.
Run (ArduPilot environment active, from the repository root):
    python sim/flow_dropout_test.py --ardupilot /path/to/ardupilot
    python sim/flow_dropout_test.py --case long_wind      # one case only
Logs for each case are written to sim/out/dropout_<case>/.
"""
import argparse
import concurrent.futures
import json
import math
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nocompass_test as nc                                   # noqa: E402

MODE_NAMES = {0: "Stabilize", 2: "AltHold", 4: "Guided", 5: "Loiter", 9: "Land", 20: "Guided_NoGPS"}
WIND_MPS = 4.0
CASES = {                      # name: (title, dropout seconds, wind m/s)
    "short_calm": ("3 s dropout, calm", 3.0, 0.0),
    "short_wind": (f"3 s dropout, {WIND_MPS:.0f} m/s wind", 3.0, WIND_MPS),
    "long_calm": ("15 s dropout, calm", 15.0, 0.0),
    "long_wind": (f"15 s dropout, {WIND_MPS:.0f} m/s wind", 15.0, WIND_MPS),
}


class Watch(nc.Link):
    """Link that also records mode changes, tilt, height and descent speed."""

    def __init__(self, m, log):
        super().__init__(m, log, feed_heading=False)
        self.t0 = None                 # time the dropout started
        self.events = []
        self.max_tilt = 0.0
        self.height, self.vz = 0.0, 0.0
        self.touchdown_speed = None
        self.sim_touchdown = None
        self._mode_seen, self._armed_seen = None, False

    def event(self, text):
        t = time.time() - self.t0 if self.t0 else 0.0
        self.events.append((round(t, 1), text))
        self.say(f"  event t={t:+.1f} s: {text}")

    def _handle(self, msg):
        super()._handle(msg)
        t = msg.get_type()
        if t == "ATTITUDE" and self.t0:
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(msg.roll) * math.cos(msg.pitch)))))
            self.max_tilt = max(self.max_tilt, tilt)
        elif t == "LOCAL_POSITION_NED":
            self.height, self.vz = -msg.z, msg.vz
            if self.height > 0.3:
                self.touchdown_speed = msg.vz          # last descent speed seen above 0.3 m
        elif t == "HEARTBEAT" and msg.get_srcComponent() == 1:
            if self.mode != self._mode_seen:
                if self._mode_seen is not None and self.t0:
                    self.event(f"mode changed to {MODE_NAMES.get(self.mode, self.mode)}")
                self._mode_seen = self.mode
            if self._armed_seen and not self.armed and self.t0:
                self.event("disarmed")
            self._armed_seen = self.armed
        elif t == "STATUSTEXT" and self.t0:
            self.events.append((round(time.time() - self.t0, 1), f"autopilot: {msg.text}"))
            if "Hit ground at" in msg.text:            # the simulator's own touchdown speed
                try:
                    self.sim_touchdown = float(msg.text.split("Hit ground at")[1].split()[0])
                except ValueError:
                    pass


def set_param(link, name, value):
    link.m.mav.param_set_send(link.m.target_system, 1, name.encode(), float(value), 9)
    link.pump(0.3)


def run_case(case, instance, root, land_speed=None):
    title, dropout_s, wind = CASES[case]
    outdir = nc.OUT / f"dropout_{case}"
    steps, numbers = [], {}
    proc = log = link = None

    def step(name, ok, detail=""):
        steps.append((name, bool(ok), detail))
        link.say(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        return bool(ok)

    try:
        params = dict(nc.COMMON, EK3_SRC1_YAW=0, FS_EKF_ACTION=1)
        if wind:
            params.update({"SIM_WIND_SPD": wind, "SIM_WIND_DIR": 0})
        if land_speed:                                  # both spellings: m/s (newer) and cm/s (older)
            params.update({"LAND_SPD_MS": land_speed, "LAND_SPEED": land_speed * 100})
        proc = nc.start_sitl(pathlib.Path(root), params, instance, outdir)
        log = open(outdir / "test.log", "w")
        m = nc.connect(5760 + 10 * instance, log)
        link = Watch(m, log)
        link.say(f"{title}: connected")
        for msg_id, hz in nc.INTERVALS_HZ.items():
            m.mav.command_long_send(m.target_system, 1, 511, 0, msg_id, 1e6 / hz, 0, 0, 0, 0, 0)
        link.pump(12)
        for _ in range(10):
            m.mav.set_gps_global_origin_send(m.target_system, int(nc.HOME[0] * 1e7),
                                             int(nc.HOME[1] * 1e7), int(nc.HOME[2] * 1000))
            if link.pump(2, lambda: link.origin_set):
                break
        link.pump(45, lambda: bool(link.ekf_flags & (0x008 | 0x100)))
        link.command(176, 1, nc.GUIDED)
        link.pump(2, lambda: link.mode == nc.GUIDED)
        for _ in range(4):
            link.command(400, 1)
            if link.pump(3, lambda: link.armed):
                break
            link.pump(5)
        link.command(22, 0, 0, 0, 0, 0, 0, nc.TAKEOFF_M)
        up = lambda: link.est is not None and -link.est[2] >= nc.TAKEOFF_M - 0.5
        link.pump(30, up)
        if not step("hovering in Guided before the dropout",
                    link.mode == nc.GUIDED and link.armed and up() and link.truth is not None):
            raise StopIteration
        link.cmd = (0, 0, 0, 0)
        link.pump(10)

        n0, e0, _ = link.truth
        drift = lambda: math.hypot(link.truth[0] - n0, link.truth[1] - e0)
        worst = 0.0

        def watch(seconds):
            nonlocal worst
            end = time.time() + seconds
            while time.time() < end:
                link.pump(0.2)
                worst = max(worst, drift())
                if not link.armed:
                    return

        link.t0 = time.time()
        set_param(link, "SIM_FLOW_ENABLE", 0)
        link.event("flow sensor off")
        watch(dropout_s)
        numbers["drift_at_end_of_dropout_m"] = round(drift(), 2)
        set_param(link, "SIM_FLOW_ENABLE", 1)
        link.event("flow sensor back on")
        watch(30)
        link.pump(2)

        landed = not link.armed
        holding = link.armed and link.mode == nc.GUIDED and drift() <= 2.0
        outcome = "landed and disarmed" if landed else ("holding position in Guided" if holding
                  else f"armed, mode {MODE_NAMES.get(link.mode, link.mode)}, {drift():.1f} m from start")
        numbers.update({"outcome": outcome, "worst_drift_m": round(worst, 2),
                        "final_drift_m": round(drift(), 2), "max_tilt_deg": round(link.max_tilt, 1)})
        step("ends holding position or landed", landed or holding, outcome)
        step("stays within 5 m of the start point", worst <= 5.0, f"worst {worst:.2f} m")
        step("tilt stays under 30 degrees", link.max_tilt <= 30.0, f"max {link.max_tilt:.1f} degrees")
        if landed:
            speed = link.sim_touchdown if link.sim_touchdown is not None else link.touchdown_speed
            speed = float("nan") if speed is None else speed
            numbers["touchdown_speed_mps"] = round(speed, 2)
            step("touchdown at 1.0 m/s or slower", speed <= 1.0, f"{speed:.2f} m/s")
        crashed = any("crash" in text.lower() for text in link.texts)
        step("no crash reported", not crashed)
        numbers["events"] = link.events
    except StopIteration:
        pass
    except Exception as exc:
        steps.append(("test ran to completion", False, f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            link.cmd = None
            if link.armed:
                link.command(176, 1, nc.LAND)
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

    result = {"case": case, "title": title, "passed": bool(steps) and all(ok for _, ok, _ in steps),
              "steps": steps, "numbers": numbers}
    (outdir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=list(CASES), action="append", help="run one case (default: all, in parallel)")
    ap.add_argument("--land-speed", type=float, help="final landing descent speed in m/s (default: ArduPilot's 0.5)")
    ap.add_argument("--ardupilot", help="ArduPilot source directory (default: found from sim_vehicle.py on PATH)")
    args = ap.parse_args()
    root = nc.find_ardupilot(args.ardupilot)
    cases = args.case or list(CASES)
    print(f"ArduPilot: {root}")
    print(f"Running {len(cases)} case(s) in parallel, about 2.5 minutes. Logs: {nc.OUT}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(run_case, c, i, str(root), args.land_speed) for i, c in enumerate(cases)]
        results = [f.result() for f in futures]

    for r in results:
        print(f"\n{r['title']}")
        for name, ok, detail in r["steps"]:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
        for t, text in r["numbers"].get("events", []):
            print(f"        t={t:+6.1f} s  {text}")
        for key in ("drift_at_end_of_dropout_m", "worst_drift_m", "final_drift_m"):
            if key in r["numbers"]:
                print(f"        {key}: {r['numbers'][key]}")
        print(f"  RESULT: {'PASS' if r['passed'] else 'FAIL'}")
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
