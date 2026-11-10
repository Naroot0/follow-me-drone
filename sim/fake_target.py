"""Scripted fake person for SITL and offline tests.

The person is a point in the local north-east plane. Segments are
(duration_s, speed_mps, turn_deg): the turn is applied at the start of the segment.
Speed changes ramp at `accel` rather than jumping, as a walking person's do.
"""
import math

DEFAULT_SCRIPT = [
    (10.0, 0.0, 0.0),     # stand still
    (20.0, 1.4, 0.0),     # walk straight away
    (15.0, 1.4, 90.0),    # turn right 90 degrees and walk
    (10.0, 0.0, 0.0),     # stop
    (1.0, 1.4, 180.0),    # turn around, walk 1.4 m toward the drone
    (8.0, 0.0, 0.0),      # stand
]
SEGMENT_NAMES = ["stand", "walk", "turn+walk", "stop", "approach", "stand2"]


class FakeTarget:
    def __init__(self, north, east, heading_rad, script=DEFAULT_SCRIPT, accel=1.5):
        self.accel = accel       # m/s^2; a person takes about a second to reach or leave walking pace
        self.goal_speed = 0.0
        self.n, self.e, self.h = north, east, heading_rad
        self.script = script
        self.t = 0.0
        self.i = -1
        self.seg_end = 0.0
        self.speed = 0.0

    @property
    def done(self):
        return self.i >= len(self.script) - 1 and self.t >= self.seg_end

    @property
    def segment(self):
        return max(self.i, 0)

    def step(self, dt):
        while self.t >= self.seg_end and self.i < len(self.script) - 1:
            self.i += 1
            dur, self.goal_speed, turn = self.script[self.i]
            self.h += math.radians(turn)
            self.seg_end += dur
        step = self.accel * dt
        self.speed += max(-step, min(step, self.goal_speed - self.speed))
        self.n += self.speed * math.cos(self.h) * dt
        self.e += self.speed * math.sin(self.h) * dt
        self.t += dt

    def seen_from(self, north, east, yaw_rad, fov_deg=70.0):
        """(range_m, bearing_rad) as a forward camera would report, or (None, None) outside the view."""
        dn, de = self.n - north, self.e - east
        rng = math.hypot(dn, de)
        brg = (math.atan2(de, dn) - yaw_rad + math.pi) % (2 * math.pi) - math.pi
        if abs(brg) > math.radians(fov_deg) / 2:
            return None, None
        return rng, brg
