"""Watchdog: the last check between the flight software and the autopilot.

Stage 1 task "Watchdog on command rate, speed limit, geofence, override check".
Pure logic, no MAVLink, so the same class runs in simulation and on the aircraft.
The process that owns the MAVLink link calls check() once per cycle with the
newest command from the flight state machine and the newest autopilot telemetry,
and sends what check() returns.

Rules (every number is in config/params.yaml). Each is one if-then sentence:

  Pilot first
   1. If the autopilot is not in the autonomy mode (the pilot switched mode, or
      an autopilot failsafe did), release: send nothing, for the rest of the flight.
   2. If the aircraft is disarmed, release.
   3. If a stick moves more than stick_deadband_us from where it was when the
      autonomy engaged, stop the autonomy and hover. Once the aircraft has
      stopped (slower than handover_speed_mps, or handover_max_wait_s have
      passed) and the throttle stick is within throttle_mid_band_us of the
      middle, ask the autopilot for handover_mode so the pilot's sticks fly the
      aircraft. Until the throttle stick is in the middle the aircraft keeps
      hovering, because the pilot modes read a low throttle stick as "descend".

  Land in place when the next action cannot be resolved (decision 44)
   4. If the command contains a value that is not a finite number, land.
   5. If no fresh command has arrived for land_after_s, land.
   6. If the autopilot heartbeat is older than heartbeat_timeout_s, land.
   7. If the position is older than land_after_s, land.
   8. If geofence_radius_m is above zero and the aircraft is farther than that
      from where the autonomy engaged, land. The setting is zero (off) for
      follow flights: the person leads the aircraft away from its start, and
      losing the person already ends in a landing (decision 46). A radius is
      set only for tests flown in one spot.
   9. If the aircraft is higher than max_altitude_agl_m, land.

  Hover through a short gap
  10. If no fresh command has arrived for watchdog_timeout_s, send a hover
      command until a fresh one arrives.

  Limit
  11. Forward speed is limited to [-max_reverse_mps, max_speed_mps], total
      horizontal speed to max_speed_mps, vertical speed to max_vertical_mps and
      yaw rate to max_yaw_rate_dps. These are speeds over the ground. The same
      limits are set in the autopilot, which enforces them whatever this
      software sends (decision 46).

Release, land and the pilot takeover are latched: once taken, the autonomy does
not resume until the software is restarted on the ground. During a pilot
takeover rules 6 to 9 still apply. The rules are checked in the order above, so
the pilot always outranks the software.

The autopilot keeps its own protections underneath this module: the radio
failsafe, the position-estimate failsafe (FS_EKF_ACTION) and the Guided command
timeout all act without the Jetson.
"""
import math
from dataclasses import dataclass, field

PASS, HOVER, HANDOVER, LAND, RELEASE = "PASS", "HOVER", "HANDOVER", "LAND", "RELEASE"


@dataclass
class Command:
    """Body-frame command, the same fields as follow_controller.FollowCommand."""
    vx: float = 0.0        # forward, m/s
    vy: float = 0.0        # right, m/s
    vz: float = 0.0        # down, m/s
    yaw_rate: float = 0.0  # rad/s


@dataclass
class Status:
    """Newest autopilot telemetry. Times are seconds on the same clock as check(now)."""
    mode: str = "GUIDED"
    armed: bool = True
    heartbeat_time: float = 0.0                 # when the last heartbeat arrived
    position_time: float = 0.0                  # when the last position arrived
    north_m: float = 0.0                        # local position
    east_m: float = 0.0
    altitude_m: float = 0.0                     # height above the ground or the takeoff point
    speed_mps: float = 0.0                      # horizontal speed
    sticks_us: tuple | None = None              # (roll, pitch, throttle, yaw) pulse widths, None if not received


@dataclass
class Verdict:
    action: str                                 # PASS, HOVER, HANDOVER, LAND or RELEASE
    command: Command | None                     # what to send; None for LAND and RELEASE
    events: list = field(default_factory=list)  # one line per change, for the flight log
    limited: bool = False                       # rule 11 changed the command this cycle
    mode: str | None = None                     # HANDOVER only: the autopilot mode to request


def _finite(*values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


class Watchdog:
    def __init__(self, params):
        lim = params["limits"]
        wd = params.get("watchdog", {})
        self.max_speed = float(lim["max_speed_mps"])
        self.max_reverse = float(lim.get("max_reverse_mps", 0.0))
        self.max_vertical = float(lim.get("max_vertical_mps", 1.0))
        self.max_yaw_rate = math.radians(float(lim.get("max_yaw_rate_dps", 90)))
        self.max_altitude = float(lim["max_altitude_agl_m"])
        self.fence = float(lim.get("geofence_radius_m", 0.0))   # 0 = off
        self.hover_after = float(lim["watchdog_timeout_s"])
        self.land_after = float(wd.get("land_after_s", 2.0))
        self.heartbeat_timeout = float(wd.get("heartbeat_timeout_s", 3.0))
        self.deadband = float(wd.get("stick_deadband_us", 100))
        self.autonomy_mode = str(wd.get("autonomy_mode", "GUIDED"))
        self.handover_mode = str(wd.get("handover_mode", "LOITER"))
        self.handover_speed = float(wd.get("handover_speed_mps", 0.3))
        self.handover_wait = float(wd.get("handover_max_wait_s", 2.0))
        self.thr_mid = float(wd.get("throttle_mid_us", 1500))
        self.thr_band = float(wd.get("throttle_mid_band_us", 100))
        self.pilot_t0 = None         # when the pilot's stick movement was seen
        self.pilot_stage = None      # "stopping", "throttle" or "handover"
        self.latched = None          # (action, reason) once a latched rule has fired
        self.origin = None           # (north, east) where the autonomy engaged
        self.sticks0 = None          # stick positions when the autonomy engaged
        self.hovering = False        # rule 10 active
        self.max_gap = 0.0           # longest command gap seen, seconds
        self.limit_count = 0

    def engage(self, status):
        """Call once when the autonomy takes control: records the fence centre and the stick positions."""
        self.origin = (status.north_m, status.east_m)
        self.sticks0 = tuple(status.sticks_us) if status.sticks_us else None

    def _latch(self, action, reason, events):
        self.latched = (action, reason)
        events.append(f"watchdog -> {action} ({reason})")
        return self._latched_verdict(events)

    def _latched_verdict(self, events):
        return Verdict(self.latched[0], None, events)

    def _pilot(self, now, status, events):
        """Rule 3 after the stick movement: hover, then hand over."""
        stopped = status.speed_mps <= self.handover_speed or now - self.pilot_t0 >= self.handover_wait
        thr = status.sticks_us[2] if status.sticks_us else None
        mid = thr is not None and abs(thr - self.thr_mid) <= self.thr_band
        if stopped and mid:
            if self.pilot_stage != "handover":
                self.pilot_stage = "handover"
                events.append(f"watchdog -> HANDOVER (stopped, throttle stick at {thr:.0f}; requesting {self.handover_mode})")
            return Verdict(HANDOVER, Command(), events, mode=self.handover_mode)
        if stopped and self.pilot_stage != "throttle":
            self.pilot_stage = "throttle"
            where = "not received" if thr is None else f"at {thr:.0f}"
            events.append(f"watchdog: hovering until the throttle stick is in the middle (now {where})")
        return Verdict(HOVER, Command(), events)

    def _limit(self, c):
        vx = max(-self.max_reverse, min(self.max_speed, c.vx))
        vy = max(-self.max_speed, min(self.max_speed, c.vy))
        h = math.hypot(vx, vy)
        if h > self.max_speed:
            vx, vy = vx * self.max_speed / h, vy * self.max_speed / h
        vz = max(-self.max_vertical, min(self.max_vertical, c.vz))
        yr = max(-self.max_yaw_rate, min(self.max_yaw_rate, c.yaw_rate))
        out = Command(vx, vy, vz, yr)
        return out, (out.vx, out.vy, out.vz, out.yaw_rate) != (c.vx, c.vy, c.vz, c.yaw_rate)

    def check(self, now, command, command_time, status):
        """command is the newest output of the flight state machine and command_time
        is when it was produced. Returns a Verdict."""
        events = []
        if self.latched:
            return self._latched_verdict(events)
        if self.origin is None:
            self.engage(status)

        # pilot first
        if status.mode != self.autonomy_mode:
            return self._latch(RELEASE, f"autopilot mode is {status.mode}, not {self.autonomy_mode}", events)
        if not status.armed:
            return self._latch(RELEASE, "disarmed", events)
        if self.sticks0 is None and status.sticks_us:
            self.sticks0 = tuple(status.sticks_us)
        if self.pilot_t0 is None and self.sticks0 and status.sticks_us:
            moved = max(abs(a - b) for a, b in zip(status.sticks_us, self.sticks0))
            if moved > self.deadband:
                self.pilot_t0, self.pilot_stage = now, "stopping"
                events.append(f"watchdog -> HOVER (pilot stick moved {moved:.0f} us; autonomy stopped)")
        pilot = self.pilot_t0 is not None

        # land in place when the next action cannot be resolved
        gap = 0.0
        if not pilot:
            if command is not None and not _finite(command.vx, command.vy, command.vz, command.yaw_rate):
                return self._latch(LAND, "command is not a finite number", events)
            gap = now - command_time if command is not None else float("inf")
            if math.isfinite(gap):
                self.max_gap = max(self.max_gap, gap)
            if gap >= self.land_after:
                return self._latch(LAND, f"no fresh command for {self.land_after:.1f} s", events)
        if now - status.heartbeat_time >= self.heartbeat_timeout:
            return self._latch(LAND, f"no autopilot heartbeat for {self.heartbeat_timeout:.1f} s", events)
        if now - status.position_time >= self.land_after:
            return self._latch(LAND, f"no position for {self.land_after:.1f} s", events)
        if not _finite(status.north_m, status.east_m, status.altitude_m):
            return self._latch(LAND, "position is not a finite number", events)
        away = math.hypot(status.north_m - self.origin[0], status.east_m - self.origin[1])
        if self.fence > 0 and away > self.fence:
            return self._latch(LAND, f"{away:.1f} m from the start, fence is {self.fence:.0f} m", events)
        if status.altitude_m > self.max_altitude:
            return self._latch(LAND, f"altitude {status.altitude_m:.1f} m, limit is {self.max_altitude:.0f} m", events)

        if pilot:
            return self._pilot(now, status, events)

        # hover through a short gap
        if gap >= self.hover_after:
            if not self.hovering:
                self.hovering = True
                events.append(f"watchdog -> HOVER (no fresh command for {self.hover_after:.1f} s)")
            return Verdict(HOVER, Command(), events)
        if self.hovering:
            self.hovering = False
            events.append("watchdog -> PASS (commands resumed)")

        # limit
        out, limited = self._limit(command)
        if limited:
            self.limit_count += 1
        return Verdict(PASS, out, events, limited)
