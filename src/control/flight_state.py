"""Flight state machine: ENROLL, FOLLOW, SEARCH, LAND (decision 36).

Joins the target selector (who is the person, are they visible) to the follow
controller (how to fly), and adds the search turn used while the person is lost.

Each cycle it takes an Observation and returns a body-frame command.

  ENROLL  hover while the target selector enrols the person.
  FOLLOW  run the follow controller on the person's range and bearing.
  SEARCH  the person is lost. Hold the last command for the controller's coast
          time, then hover and yaw by how the person was lost:
              lost mid-frame (occlusion)   hold heading for the first phase,
                                           then one full turn
              left by a side edge          yaw toward that side, up to one full turn
              left by the bottom, close    yaw 180 degrees in the first phase,
                                           continuing to one full turn in the second
          "First phase" is the target selector's LOST state (occlusion_max_s),
          "second phase" its DROPPED state (the hover before landing). After the
          turn the drone holds heading until the selector calls LAND.
  LAND    terminal; the caller switches the autopilot to Land.

Timing lives in the target selector, not here: this module follows its state.
"""
import math
from dataclasses import dataclass

from follow_controller import FollowCommand

ENROLL, FOLLOW, SEARCH, LAND = "ENROLL", "FOLLOW", "SEARCH", "LAND"


@dataclass
class Observation:
    target_state: str                 # ENROLLING, LOCKED, LOST, DROPPED, LAND or FAILED
    range_m: float | None = None      # when LOCKED
    bearing_rad: float | None = None  # when LOCKED, positive to the right
    edge: str | None = None           # "left", "right" or "bottom" when the person's box is at that frame edge


class FlightStateMachine:
    def __init__(self, params, controller):
        s = params.get("search", {})
        self.rate = math.radians(float(s.get("yaw_rate_dps", 30)))
        self.ctl = controller
        self.state = ENROLL
        self.last_edge = None
        self.last_bearing = 0.0
        self.search_log = []          # one dict per search, for tests and the flight log
        self._s = None

    def _begin_search(self, events):
        edge = self.last_edge or "mid"
        if edge == "left":
            direction = -1.0
        elif edge == "right":
            direction = 1.0
        else:
            direction = 1.0 if self.last_bearing >= 0 else -1.0
        self._s = {"edge": edge, "dir": direction, "turned": 0.0, "elapsed": 0.0, "outcome": None}
        self.search_log.append(self._s)
        plan = {"mid": "hold heading, then one full turn",
                "bottom": "turn 180 degrees, then on to a full turn"}.get(
                    edge, f"yaw {edge}, up to one full turn")
        events.append(f"{self.state} -> SEARCH (person lost at {edge}; {plan})")
        self.state = SEARCH

    def _turn_goal(self, phase_two):
        edge = self._s["edge"]
        if edge == "mid":
            return 2 * math.pi if phase_two else 0.0
        if edge == "bottom":
            return 2 * math.pi if phase_two else math.pi
        return 2 * math.pi

    def update(self, obs, dt, own_speed=0.0):
        """Returns (FollowCommand, list of event strings)."""
        events = []
        ts = obs.target_state
        if self.state == LAND:
            return FollowCommand(), events

        if ts in ("LAND", "FAILED"):
            why = "no enrolment" if ts == "FAILED" else "person not found"
            if self._s is not None and self._s["outcome"] is None:
                self._s["outcome"] = "land"
            events.append(f"{self.state} -> LAND ({why})")
            self.state = LAND
            return FollowCommand(), events

        if ts == "ENROLLING":
            self.state = ENROLL
            return FollowCommand(), events

        if ts == "LOCKED":
            if self.state != FOLLOW:
                if self.state == SEARCH:
                    self._s["outcome"] = "relock"
                    events.append(f"SEARCH -> FOLLOW (re-locked after {self._s['elapsed']:.1f} s, "
                                  f"turned {math.degrees(self._s['turned']) * self._s['dir']:+.0f} degrees)")
                else:
                    events.append(f"{self.state} -> FOLLOW (person enrolled)")
                self.ctl.reset()
                self.state = FOLLOW
            self.last_edge = obs.edge
            if obs.bearing_rad is not None:
                self.last_bearing = obs.bearing_rad
            return self.ctl.update(obs.range_m, obs.bearing_rad, dt, own_speed), events

        # LOST or DROPPED
        if self.state != SEARCH:
            self._begin_search(events)
        s = self._s
        s["elapsed"] += dt
        coast = self.ctl.update(None, None, dt, own_speed)   # holds the last command for coast_s, then zero
        if s["elapsed"] <= self.ctl.coast_s:
            return coast, events
        goal = self._turn_goal(phase_two=(ts == "DROPPED"))
        if s["turned"] < goal:
            s["turned"] += self.rate * dt
            return FollowCommand(yaw_rate=s["dir"] * self.rate), events
        return FollowCommand(), events
