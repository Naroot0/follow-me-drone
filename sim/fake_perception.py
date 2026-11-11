"""Stand-in for the camera, detector, tracker and target selector, for simulator tests.

Produces the same Observation the real perception chain will: the target selector's
state, the person's range and bearing while locked, and which frame edge they are at.
Timing comes from config/params.yaml, the same numbers the real selector uses.
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "control"))
from flight_state import Observation  # noqa: E402


class FakePerception:
    def __init__(self, params, target, fov_deg=70.0, min_range_m=1.0):
        acq, re = params["acquisition"], params["reacquisition"]
        self.hold_s = float(acq.get("hold_s", 5))
        self.timeout_s = float(acq.get("timeout_s", 30))
        self.occlusion_s = float(re["occlusion_max_s"])
        self.hover_s = float(params.get("loss", {}).get("hover_then_land_after_s", 25))
        self.reacquire = int(re.get("reacquire_frames", 5))
        self.edge_fraction = float(params.get("search", {}).get("edge_fraction", 0.15))
        self.tgt = target
        self.half = math.radians(fov_deg) / 2
        self.min_range = min_range_m          # closer than this the person is under the camera's view
        self.state = "ENROLLING"
        self.t0 = None
        self.still_since = None
        self.lost_since = None
        self.seen = 0

    def observe(self, north, east, yaw, t):
        if self.t0 is None:
            self.t0 = t
        dn, de = self.tgt.n - north, self.tgt.e - east
        rng = math.hypot(dn, de)
        brg = (math.atan2(de, dn) - yaw + math.pi) % (2 * math.pi) - math.pi
        visible = (not self.tgt.hidden) and rng >= self.min_range and abs(brg) <= self.half
        edge = None
        if visible:
            inner = self.half * (1 - 2 * self.edge_fraction)
            if rng < self.min_range + 0.5:
                edge = "bottom"
            elif brg > inner:
                edge = "right"
            elif brg < -inner:
                edge = "left"

        if self.state == "ENROLLING":
            if visible and self.tgt.speed < 0.1:
                if self.still_since is None:
                    self.still_since = t
                if t - self.still_since >= self.hold_s:
                    self.state = "LOCKED"
            else:
                self.still_since = None
            if self.state == "ENROLLING" and t - self.t0 > self.timeout_s:
                self.state = "FAILED"
        elif self.state == "LOCKED":
            if not visible:
                self.state, self.lost_since, self.seen = "LOST", t, 0
        elif self.state in ("LOST", "DROPPED"):
            self.seen = self.seen + 1 if visible else 0
            gone = t - self.lost_since
            if self.seen >= self.reacquire:
                self.state = "LOCKED"
            elif gone > self.occlusion_s + self.hover_s:
                self.state = "LAND"
            elif gone > self.occlusion_s:
                self.state = "DROPPED"

        if self.state == "LOCKED" and visible:
            return Observation("LOCKED", rng, brg, edge)
        return Observation(self.state)
