"""Watchdog test: every rule in src/safety/watchdog.py, with no autopilot needed.

  1. normal flight   the follow scenario runs through the watchdog untouched
  2. limits          over-speed, reverse, sideways, climb and yaw commands are cut to the limits
  3. short gap       commands stop for 1 s: hover, then pass again when they resume
  4. long gap        commands stop for good: hover, then land, and it stays landed
  5. bad command     a not-a-number value: land
  6. fence           off by default: 500 m from the start passes; with a 50 m radius set, 51 m lands
  7. ceiling         above the altitude limit: land
  8. mode switch     the pilot changes mode: release
  9. stick           the pilot moves a stick: hover, hand over once stopped with the throttle
                     stick in the middle, release when the autopilot changes mode; a lost
                     heartbeat during the hover still lands
 10. disarm          release
 11. heartbeat       the autopilot goes quiet: land
 12. position        position stops arriving: land
 13. speed           time per check

    python sim/watchdog_test.py
"""
import math
import pathlib
import sys
import time

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "control"))
sys.path.insert(0, str(REPO / "src" / "safety"))
sys.path.insert(0, str(REPO / "sim"))
from watchdog import Watchdog, Command, Status, PASS, HOVER, HANDOVER, LAND, RELEASE   # noqa: E402
from follow_controller import FollowController            # noqa: E402
from flight_state import FlightStateMachine               # noqa: E402
from fake_target import FakeTarget                        # noqa: E402
from fake_perception import FakePerception                # noqa: E402
from flight_test import KinematicVehicle                  # noqa: E402

DT = 0.1
STICKS = (1500, 1500, 1500, 1500)
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def status(t, **kw):
    base = dict(mode="GUIDED", armed=True, heartbeat_time=t, position_time=t,
                north_m=0.0, east_m=0.0, altitude_m=3.0, speed_mps=0.0, sticks_us=STICKS)
    base.update(kw)
    return Status(**base)


def fresh(params):
    wd = Watchdog(params)
    wd.engage(status(0.0))
    return wd


def run(wd, steps):
    """steps: list of (now, command, command_time, status). Returns the verdicts."""
    return [wd.check(now, c, ct, st) for now, c, ct, st in steps]


def normal_flight(params):
    veh = KinematicVehicle()
    script = [(8.0, 0.0, 0.0), (20.0, 1.4, 0.0), (10.0, 1.4, 60.0), (5.0, 0.0, 0.0)]
    tgt = FakeTarget(4.0, 0.0, 0.0, script)
    per = FakePerception(params, tgt)
    fsm = FlightStateMachine(params, FollowController(params))
    wd = Watchdog(params)
    t, actions, changed, top = 0.0, set(), 0, 0.0
    while not tgt.done:
        tgt.step(DT)
        obs = per.observe(veh.n, veh.e, veh.yaw, t)
        cmd, _ = fsm.update(obs, DT, veh.forward_speed)
        v = wd.check(t, Command(cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate), t,
                     status(t, north_m=veh.n, east_m=veh.e))
        actions.add(v.action)
        changed += v.limited
        top = max(top, abs(cmd.vx))
        if v.command is not None:
            veh.send(v.command, DT)
        t += DT
    away = math.hypot(veh.n, veh.e)
    check("normal flight", actions == {PASS} and changed == 0,
          f"{t:.0f} s of follow, {away:.0f} m travelled, top commanded speed {top:.1f} m/s, "
          f"actions {sorted(actions)}, commands changed {changed}")


def limits(params):
    lim = params["limits"]
    wd = fresh(params)
    cases = [("over-speed", Command(vx=5.0), "vx", lim["max_speed_mps"]),
             ("reverse", Command(vx=-2.0), "vx", -lim["max_reverse_mps"]),
             ("climb", Command(vz=-3.0), "vz", -lim["max_vertical_mps"]),
             ("yaw", Command(yaw_rate=math.radians(300)), "yaw_rate", math.radians(lim["max_yaw_rate_dps"]))]
    ok, parts = True, []
    for name, c, attr, want in cases:
        v = wd.check(1.0, c, 1.0, status(1.0))
        got = getattr(v.command, attr)
        ok &= v.action == PASS and v.limited and abs(got - want) < 1e-9
        parts.append(f"{name} {getattr(c, attr):.1f} -> {got:.2f}")
    v = wd.check(1.0, Command(vx=2.0, vy=2.0), 1.0, status(1.0))
    h = math.hypot(v.command.vx, v.command.vy)
    ok &= abs(h - lim["max_speed_mps"]) < 1e-9
    parts.append(f"diagonal 2.8 -> {h:.2f}")
    v = wd.check(1.0, Command(vx=1.0, yaw_rate=0.2), 1.0, status(1.0))
    ok &= not v.limited and v.command == Command(vx=1.0, yaw_rate=0.2)
    check("limits", ok, "; ".join(parts) + "; in-limit command unchanged")


def gaps(params):
    wd = fresh(params)
    steps = []
    for i in range(1, 41):                      # 4 s at 10 Hz
        now = i * DT
        last = now if now <= 1.0 or now > 2.0 else 1.0   # commands stop from 1.0 to 2.0 s
        steps.append((now, Command(vx=1.0), last, status(now)))
    v = run(wd, steps)
    acts = [x.action for x in v]
    first_hover = next(i for i, a in enumerate(acts) if a == HOVER)
    t_hover = steps[first_hover][0] - 1.0
    resumed = acts[-1] == PASS and v[-1].command.vx == 1.0
    zero = all(x.command == Command() for x in v if x.action == HOVER)
    check("short gap", abs(t_hover - params["limits"]["watchdog_timeout_s"]) < 1e-6 and resumed and zero
          and LAND not in acts,
          f"hover {t_hover:.1f} s after the last command, zero-velocity command sent, passed again when commands resumed")

    wd = fresh(params)
    steps = [(i * DT, Command(vx=1.0), min(i * DT, 1.0), status(i * DT)) for i in range(1, 51)]
    steps.append((5.1, Command(vx=1.0), 5.1, status(5.1)))       # a fresh command after the landing
    v = run(wd, steps)
    acts = [x.action for x in v]
    t_land = steps[acts.index(LAND)][0] - 1.0
    check("long gap", abs(t_land - params["watchdog"]["land_after_s"]) < 1e-6 and acts[-1] == LAND
          and v[-1].command is None and HOVER in acts[:acts.index(LAND)],
          f"hover first, land {t_land:.1f} s after the last command, still landing when commands return")


def one_shot(params, name, want, detail, **kw):
    wd = fresh(params)
    c = kw.pop("command", Command(vx=1.0))
    ok0 = wd.check(1.0, Command(vx=1.0), 1.0, status(1.0)).action == PASS
    v = wd.check(1.1, c, 1.1, status(1.1, **kw))
    v2 = wd.check(1.2, Command(vx=1.0), 1.2, status(1.2))        # everything normal again
    latched = v2.action == want
    check(name, ok0 and v.action == want and latched and (v.command is None) == (want in (LAND, RELEASE)),
          f"{detail}: {v.events[0] if v.events else 'no event'}")


def fence(params):
    wd = fresh(params)
    far = wd.check(1.0, Command(vx=1.0), 1.0, status(1.0, north_m=500.0))
    import copy
    p2 = copy.deepcopy(params)
    p2["limits"]["geofence_radius_m"] = 50
    wd2 = fresh(p2)
    inside = wd2.check(1.0, Command(vx=1.0), 1.0, status(1.0, north_m=49.0))
    out = wd2.check(1.1, Command(vx=1.0), 1.1, status(1.1, north_m=51.0))
    check("fence", params["limits"]["geofence_radius_m"] == 0 and far.action == PASS
          and inside.action == PASS and out.action == LAND,
          f"default setting is off and 500 m from the start passes; with a 50 m radius set, 49 m passes and "
          f"51 m gives: {out.events[0] if out.events else 'no event'}")


def stick(params):
    wd = fresh(params)
    c = Command(vx=1.5)
    moved, low = (1650, 1500, 1500, 1500), (1650, 1500, 1100, 1500)
    small = wd.check(1.0, c, 1.0, status(1.0, speed_mps=1.5, sticks_us=(1560, 1500, 1500, 1500)))
    first = wd.check(1.1, c, 1.1, status(1.1, speed_mps=1.5, sticks_us=moved))       # stick moved while flying
    slowing = wd.check(1.5, c, 1.5, status(1.5, speed_mps=0.8, sticks_us=moved))
    stopped = wd.check(2.0, c, 2.0, status(2.0, speed_mps=0.2, sticks_us=moved))      # stopped, throttle in the middle
    held = wd.check(2.1, c, 2.1, status(2.1, speed_mps=0.1))                          # stick released: still handing over
    rel = wd.check(2.2, c, 2.2, status(2.2, mode="LOITER"))
    ok = (small.action == PASS and first.action == HOVER and first.command == Command()
          and slowing.action == HOVER and stopped.action == HANDOVER and stopped.mode == "LOITER"
          and stopped.command == Command() and held.action == HANDOVER and rel.action == RELEASE and rel.command is None)

    wd = fresh(params)                                                                # throttle stick low
    wd.check(1.0, c, 1.0, status(1.0, sticks_us=low[:2] + (1500,) + low[3:]))
    w1 = wd.check(1.2, c, 1.2, status(1.2, sticks_us=low))
    w2 = wd.check(4.0, c, 4.0, status(4.0, sticks_us=low))
    w3 = wd.check(4.1, c, 4.1, status(4.1, sticks_us=(1650, 1500, 1480, 1500)))
    ok &= w1.action == HOVER and w2.action == HOVER and w3.action == HANDOVER

    wd = fresh(params)                                                                # still moving after the wait
    wd.check(1.0, c, 1.0, status(1.0, speed_mps=1.5, sticks_us=moved))
    m1 = wd.check(2.9, c, 2.9, status(2.9, speed_mps=1.0, sticks_us=moved))
    m2 = wd.check(3.0, c, 3.0, status(3.0, speed_mps=1.0, sticks_us=moved))
    ok &= m1.action == HOVER and m2.action == HANDOVER

    wd = fresh(params)                                                                # heartbeat lost while hovering for the pilot
    wd.check(1.0, c, 1.0, status(1.0, sticks_us=low))
    h = wd.check(4.5, c, 4.5, status(4.5, sticks_us=low, heartbeat_time=1.0))
    ok &= h.action == LAND
    check("stick", ok,
          "60 us ignored; 150 us stops the autonomy in a hover; handover to LOITER once stopped with the throttle "
          "in the middle; hover continues while the throttle stick is low; handover after 2.0 s if still moving; "
          "release when the autopilot reports the new mode; lost heartbeat during the hover lands")


def stale(params):
    wd = fresh(params)
    acts = []
    for i in range(1, 51):
        now = i * DT
        acts.append((now, wd.check(now, Command(vx=1.0), now, status(now, heartbeat_time=min(now, 1.0))).action))
    t = next(now for now, a in acts if a == LAND) - 1.0
    check("heartbeat", abs(t - params["watchdog"]["heartbeat_timeout_s"]) < 1e-6,
          f"land {t:.1f} s after the last heartbeat")
    wd = fresh(params)
    acts = []
    for i in range(1, 51):
        now = i * DT
        acts.append((now, wd.check(now, Command(vx=1.0), now, status(now, position_time=min(now, 1.0))).action))
    t = next(now for now, a in acts if a == LAND) - 1.0
    check("position", abs(t - params["watchdog"]["land_after_s"]) < 1e-6, f"land {t:.1f} s after the last position")


def speed(params):
    wd = fresh(params)
    c, st = Command(vx=1.0, yaw_rate=0.1), status(1.0)
    n = 200_000
    t0 = time.perf_counter()
    for _ in range(n):
        wd.check(1.0, c, 1.0, st)
    us = (time.perf_counter() - t0) / n * 1e6
    period_us = 1e6 / params["limits"]["command_rate_hz"]
    check("speed", us < period_us / 100, f"{us:.1f} microseconds per check, {us / period_us * 100:.3f} percent of a "
          f"{params['limits']['command_rate_hz']} Hz cycle")


def main():
    with open(REPO / "config" / "params.yaml") as f:
        params = yaml.safe_load(f)
    lim = params["limits"]
    normal_flight(params)
    limits(params)
    gaps(params)
    one_shot(params, "bad command", LAND, "not-a-number forward speed", command=Command(vx=float("nan")))
    fence(params)
    one_shot(params, "ceiling", LAND, f"{lim['max_altitude_agl_m'] + 0.5} m up", altitude_m=lim["max_altitude_agl_m"] + 0.5)
    one_shot(params, "mode switch", RELEASE, "pilot selects ALT_HOLD", mode="ALT_HOLD")
    stick(params)
    one_shot(params, "disarm", RELEASE, "disarmed", armed=False)
    stale(params)
    speed(params)
    print("\nRESULT:", "PASS" if all(results) else "FAIL", f"({sum(results)} of {len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
