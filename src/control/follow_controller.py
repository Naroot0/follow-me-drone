"""Follow controller: range and bearing in, body-frame velocity and yaw rate out.

Stage 1 task "Proportional hold on distance and bearing". Pure logic, no MAVLink,
so the same class runs against SITL and on the aircraft.

Inputs each cycle (from perception, or from the fake target in sim):
    range_m      horizontal distance to the person, metres; None when not visible
    bearing_rad  angle of the person off the nose, radians, positive to the right

Rules (every number is in config/params.yaml):
  1. Yaw rate = kp_bearing x bearing, limited. The camera stays on the person.
  2. Forward speed = kp_distance x (range - station distance)
                     + feedforward_gain x (person's speed away from the drone),
     limited to [-max_reverse, max_speed]. The feedforward removes the lag a
     proportional term alone has behind a walking person. The estimate of the
     person's speed is refreshed every cycle, so it follows a changing pace.
     Reverse (up to max_reverse, for at most max_reverse_distance_m) is allowed only
     to recover the distance lost while braking after the person stops. If the
     person walks toward a hovering drone it hovers and yaws (decisions 32 and 33).
  3. Forward speed is scaled by cos(bearing), so the drone turns toward the
     person before closing on them.
  4. No measurement -> hold the last command for coast_s, then zero velocity and
     zero yaw rate (hover). Searching is the flight state machine's job.
  5. Sideways and vertical speed are zero in this version (trailing follow,
     fixed altitude).
"""
import math
from dataclasses import dataclass


@dataclass
class FollowCommand:
    vx: float = 0.0        # forward, m/s (body frame)
    vy: float = 0.0        # right, m/s
    vz: float = 0.0        # down, m/s
    yaw_rate: float = 0.0  # rad/s, positive clockwise seen from above


class FollowController:
    def __init__(self, params):
        st, lim, c = params["station"], params["limits"], params["control"]
        self.distance = float(st["distance_m"])
        self.max_speed = float(lim["max_speed_mps"])
        self.max_yaw_rate = math.radians(float(lim.get("max_yaw_rate_dps", 60)))
        self.kp_d = float(c["kp_distance"])
        self.kp_b = float(c["kp_bearing"])
        self.ff = float(c.get("feedforward_gain", 0.0))        # 0 = pure proportional
        self.fa = float(c.get("range_filter_alpha", 1.0))      # alpha-beta filter on range
        self.fb = float(c.get("range_filter_beta", 0.0))
        self.max_reverse = float(lim.get("max_reverse_mps", 0.0))  # 0 = never backward
        self.approach_speed = float(params.get("approach", {}).get("speed_threshold_mps", 0.3))
        self.max_rev_dist = float(lim.get("max_reverse_distance_m", 0.0))
        self.rev_window = float(c.get("reverse_window_s", 2.0))
        self.since_fwd = 1e9      # seconds since the drone last flew forward
        self.rev_used = 0.0       # metres of reverse used in the current recovery
        self.recovering = False
        self.coast_s = float(c.get("coast_s", 0.0))   # hold the last command this long through a missed frame
        self.last = None
        self.miss_t = 0.0
        self.reset()

    def reset(self):
        self.r = None        # filtered range, m
        self.rr = 0.0        # filtered range rate, m/s (positive = person pulling away)
        self.person_speed = 0.0

    def update(self, range_m, bearing_rad, dt, own_speed=0.0):
        """own_speed: the drone's current forward speed, m/s (telemetry)."""
        if range_m is None or bearing_rad is None:
            # A missed frame or two is routine for a detector. Hold the last command
            # briefly so one dropout does not stop the turn that would bring the
            # person back into view; after coast_s, hover.
            self.miss_t += dt
            if self.last is not None and self.miss_t <= self.coast_s:
                return self.last
            self.reset()
            self.last = None
            return FollowCommand()
        self.miss_t = 0.0

        if self.r is None:
            self.r = range_m
        else:
            pred = self.r + self.rr * dt
            resid = range_m - pred
            self.r = pred + self.fa * resid
            self.rr += self.fb / dt * resid
        err = self.r - self.distance

        # Person's speed along the line of sight = how fast the gap opens + our own speed.
        self.person_speed = self.rr + own_speed * math.cos(bearing_rad)
        ff = self.ff * max(0.0, self.person_speed)

        vx = self.kp_d * err + ff
        vx *= max(0.0, math.cos(bearing_rad))
        # Reverse only undoes the drone's own overshoot: the person is standing and the
        # gap is too small. If the person is walking toward the drone, it hovers (decision 32).
        # The rear rangefinder stop is applied by the obstacle layer, not here.
        # Reverse exists for one purpose: recovering the distance lost while braking
        # after the person stops. It may start only within reverse_window_s of the
        # drone's own forward flight, covers at most max_reverse_distance_m, and never
        # runs while the person is walking toward the drone. A drone that has been
        # hovering does not back away from an approaching person (decision 32).
        if own_speed > 0.3:
            self.since_fwd, self.rev_used = 0.0, 0.0
        else:
            self.since_fwd += dt
        approaching = self.person_speed < -self.approach_speed
        allowed = (vx < 0 and not approaching and self.rev_used < self.max_rev_dist
                   and (self.recovering or self.since_fwd <= self.rev_window))
        self.recovering = allowed
        vx = min(max(vx, -self.max_reverse if allowed else 0.0), self.max_speed)
        if vx < 0:
            self.rev_used += -vx * dt

        yaw_rate = self.kp_b * bearing_rad
        yaw_rate = min(max(yaw_rate, -self.max_yaw_rate), self.max_yaw_rate)
        self.last = FollowCommand(vx=vx, yaw_rate=yaw_rate)
        return self.last
