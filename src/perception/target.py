"""Target selection: enrollment, lock, loss, and reacquisition (decision 29).

Takes the tracker's boxes and IDs plus a re-ID embedding per box, and decides
which track, if any, is the person being followed.

States
  ENROLLING  no target yet; waiting for one person to be the largest box, and
             nearly still, for acquisition.hold_s. Times out after
             acquisition.timeout_s -> FAILED (caller lands).
  LOCKED     target track ID visible this frame; gallery learns new views slowly.
  LOST       target track ID missing; every visible track is a candidate.
             Re-lock rule (per frame):
               s(t) = max similarity of track t to any gallery embedding
               best, second = highest and second-highest s (second = 0 if alone)
               provisional if s(best) >= reid_similarity_threshold
                           and s(best) - s(second) >= reid_margin
               counter increments while the SAME track is provisional on
               consecutive frames, else resets; LOCKED when counter reaches
               reacquire_frames. Gallery is frozen in this state.
             Longer than occlusion_max_s -> DROPPED.
  DROPPED    caller hovers in place. The same re-lock rule keeps running, so a
             late reappearance still re-locks. After loss.hover_then_land_after_s
             more seconds -> LAND.
  LAND, FAILED  terminal; caller lands.

Library use:
    sel = TargetSelector(params, reid)
    out = sel.update(frame, boxes_xyxy, track_ids, t_now)
    out.state, out.target_id, out.target_box, out.events (list of strings)

Bench:
    python src/perception/target.py --camera 0
Prints every state transition with the similarity values behind it.
"""
import argparse
import pathlib
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reid import ReID  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

ENROLLING, LOCKED, LOST, DROPPED, LAND, FAILED = "ENROLLING", "LOCKED", "LOST", "DROPPED", "LAND", "FAILED"


@dataclass
class TargetOutput:
    state: str
    target_id: int | None
    target_box: np.ndarray | None
    scores: dict = field(default_factory=dict)   # track_id -> similarity (LOST and LOCKED)
    events: list = field(default_factory=list)   # human-readable transitions this frame


class TargetSelector:
    def __init__(self, params, reid: ReID, t0=None):
        acq, re, rid = params["acquisition"], params["reacquisition"], params.get("reid", {})
        self.hold_s = float(acq.get("hold_s", 5))
        self.timeout_s = float(acq.get("timeout_s", 10))
        self.still_max = float(acq.get("still_max_motion", 0.15))   # box heights per second
        self.thresh = float(re["reid_similarity_threshold"])
        self.margin = float(re.get("reid_margin", 0.05))
        self.reacq_frames = int(re.get("reacquire_frames", 5))
        self.occlusion_max_s = float(re["occlusion_max_s"])
        self.hover_s = float(params.get("loss", {}).get("hover_then_land_after_s", 15))
        self.still_smooth = int(acq.get("still_smooth_frames", 10))
        self.gallery_max = int(rid.get("gallery_max", 30))
        self.learn_every = int(rid.get("learn_every_frames", 15))
        self.learn_hi = float(rid.get("learn_max_similarity", 0.95))
        self.reid = reid

        self.state = ENROLLING
        self.t_start = t0
        self.target_id = None
        self.gallery = []
        self.frame = 0
        self.last_learn = -10**9
        # enrollment bookkeeping
        self.cand_id = None
        self.cand_since = None
        self.cand_path = []          # (t, cx, cy, h) of the candidate while held
        # loss bookkeeping
        self.lost_since = None
        self.prov_id = None
        self.prov_count = 0

    # ------------------------------------------------------------------ helpers
    def _reset_candidate(self):
        self.cand_id, self.cand_since, self.cand_path = None, None, []

    def _candidate_motion(self):
        """Path length of the smoothed box centre over the hold, in box heights per second.

        The centre is averaged over still_smooth_frames before differencing so that
        detector jitter on a standing person does not count as motion, while pacing
        back and forth (which returns to the start) still does.
        """
        if len(self.cand_path) < 2:
            return 0.0
        p = np.array(self.cand_path)
        dt = p[-1, 0] - p[0, 0]
        if dt <= 0:
            return 0.0
        k = max(1, min(self.still_smooth, len(p) // 2))
        kern = np.ones(k) / k
        cx = np.convolve(p[:, 1], kern, mode="valid")
        cy = np.convolve(p[:, 2], kern, mode="valid")
        if len(cx) < 2:
            return 0.0
        d = np.hypot(np.diff(cx), np.diff(cy)).sum()
        return float(d / dt / max(p[:, 3].mean(), 1.0))

    def _learn(self, f, s):
        if (len(self.gallery) < self.gallery_max and self.frame - self.last_learn >= self.learn_every
                and self.thresh <= s < self.learn_hi):
            self.gallery.append(f.copy())
            self.last_learn = self.frame

    # ------------------------------------------------------------------ update
    def update(self, frame, boxes, ids, t):
        self.frame += 1
        if self.t_start is None:
            self.t_start = t
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        ids = list(ids)
        out = TargetOutput(self.state, self.target_id, None)

        if self.state in (FAILED, LAND):
            return out

        if self.state == ENROLLING:
            if t - self.t_start > self.timeout_s:
                self.state = FAILED
                out.events.append(f"t={t - self.t_start:5.1f}s ENROLLING -> FAILED (no stable person within {self.timeout_s:.0f} s)")
                out.state = self.state
                return out
            if len(boxes) == 0:
                if self.cand_id is not None:
                    out.events.append(f"t={t - self.t_start:5.1f}s enrol candidate {self.cand_id} lost (nobody visible)")
                self._reset_candidate()
                return out
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            i = int(np.argmax(areas))
            tid = ids[i]
            if tid != self.cand_id:
                if self.cand_id is not None:
                    out.events.append(f"t={t - self.t_start:5.1f}s enrol candidate changed {self.cand_id} -> {tid}, hold restarted")
                self._reset_candidate()
                self.cand_id, self.cand_since = tid, t
            b = boxes[i]
            self.cand_path.append((t, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1]))
            held = t - self.cand_since
            motion = self._candidate_motion()
            if held >= self.hold_s:
                if motion <= self.still_max:
                    f = self.reid.embed(frame, b)
                    if not np.isnan(f).any():
                        self.target_id = tid
                        self.gallery = [f]
                        self.last_learn = self.frame
                        self.state = LOCKED
                        out.events.append(f"t={t - self.t_start:5.1f}s ENROLLING -> LOCKED on track {tid} "
                                          f"(held {held:.1f} s, motion {motion:.2f} box-heights/s)")
                        self._reset_candidate()
                else:
                    # moving too much: slide the window forward rather than restart from zero
                    self.cand_path = self.cand_path[len(self.cand_path) // 2:]
                    self.cand_since = self.cand_path[0][0]
                    out.events.append(f"t={t - self.t_start:5.1f}s candidate {tid} held {held:.1f} s but moving "
                                      f"{motion:.2f} > {self.still_max} box-heights/s, waiting")
            out.state, out.target_id = self.state, self.target_id
            if self.state == LOCKED:
                out.target_box = b
            return out

        # LOCKED or LOST: need embeddings of everything visible
        feats = self.reid.embed_many(frame, boxes) if len(boxes) else np.zeros((0, 512), np.float32)
        scores = {}
        for tid, f in zip(ids, feats):
            if not np.isnan(f).any():
                scores[tid] = ReID.best_match(self.gallery, f)
        out.scores = scores

        if self.state == LOCKED:
            if self.target_id in ids:
                k = ids.index(self.target_id)
                out.target_box = boxes[k]
                if self.target_id in scores:
                    self._learn(feats[k], scores[self.target_id])
            else:
                self.state = LOST
                self.lost_since = t
                self.prov_id, self.prov_count = None, 0
                out.events.append(f"t={t - self.t_start:5.1f}s LOCKED -> LOST (track {self.target_id} missing, "
                                  f"gallery {len(self.gallery)} views frozen)")
            out.state = self.state
            return out

        # LOST or DROPPED: both scan for the target; DROPPED means the caller is hovering
        gone = t - self.lost_since
        if self.state == LOST and gone > self.occlusion_max_s:
            self.state = DROPPED
            out.events.append(f"t={t - self.t_start:5.1f}s LOST -> DROPPED (no re-lock within {self.occlusion_max_s:.0f} s; "
                              f"hover, keep scanning {self.hover_s:.0f} s more)")
        if self.state == DROPPED and gone > self.occlusion_max_s + self.hover_s:
            self.state = LAND
            out.events.append(f"t={t - self.t_start:5.1f}s DROPPED -> LAND (no re-lock within {gone:.1f} s)")
            out.state = self.state
            return out
        if scores:
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            best_id, best_s = ranked[0]
            second_s = ranked[1][1] if len(ranked) > 1 else 0.0
            provisional = best_s >= self.thresh and (best_s - second_s) >= self.margin
            if provisional and best_id == self.prov_id:
                self.prov_count += 1
            elif provisional:
                self.prov_id, self.prov_count = best_id, 1
            else:
                if self.prov_id is not None:
                    out.events.append(f"t={t - self.t_start:5.1f}s provisional {self.prov_id} reset "
                                      f"(best {best_id} s={best_s:.2f}, second s={second_s:.2f})")
                self.prov_id, self.prov_count = None, 0
            if self.prov_count >= self.reacq_frames:
                old = self.target_id
                was = self.state
                self.target_id = best_id
                self.state = LOCKED
                out.target_box = boxes[ids.index(best_id)]
                out.events.append(f"t={t - self.t_start:5.1f}s {was} -> LOCKED on track {best_id} (was {old}); "
                                  f"s={best_s:.2f}, second={second_s:.2f}, margin={best_s - second_s:.2f}, "
                                  f"after {t - self.lost_since:.2f} s")
                self.prov_id, self.prov_count = None, 0
        out.state, out.target_id = self.state, self.target_id
        return out


# ---------------------------------------------------------------------- bench
def _bench(camera):
    from ultralytics import YOLO

    with open(REPO / "config" / "params.yaml") as f:
        params = yaml.safe_load(f)
    det, trk = params["detector"], params.get("tracker", {})
    tracker_cfg = trk.get("config", "bytetrack.yaml")
    if (REPO / tracker_cfg).exists():
        tracker_cfg = str(REPO / tracker_cfg)

    reid = ReID(weights=params.get("reid", {}).get("weights", "osnet_x0_25_msmt17.pt"))
    yolo = YOLO(det.get("weights", "yolo11n.pt"))
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, det["input_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, det["input_height"])
    if not cap.isOpened():
        raise SystemExit(f"camera {camera} did not open")

    sel = TargetSelector(params, reid)
    print(f"hold {sel.hold_s}s, timeout {sel.timeout_s}s, still<= {sel.still_max}, thresh {sel.thresh}, "
          f"margin {sel.margin}, reacquire {sel.reacq_frames} frames, occlusion max {sel.occlusion_max_s}s")
    print("keys: r = restart enrollment, q = quit")
    colors = {ENROLLING: (0, 255, 255), LOCKED: (0, 255, 0), LOST: (0, 165, 255),
              DROPPED: (0, 90, 255), LAND: (0, 0, 255), FAILED: (0, 0, 255)}
    t0 = time.perf_counter()
    transitions = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter() - t0
        res = yolo.track(frame, persist=True, tracker=tracker_cfg, classes=[0],
                         conf=float(trk.get("detector_conf", 0.25)),
                         device=str(reid.device), verbose=False)[0]
        boxes, ids = [], []
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().tolist()
        out = sel.update(frame, boxes, ids, t)
        for e in out.events:
            print(e)
            transitions.append(e)

        for b, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, b)
            is_t = tid == out.target_id and out.state == LOCKED
            col = (0, 255, 0) if is_t else (160, 160, 160)
            label = f"ID {tid}" + ("  TARGET" if is_t else "")
            if tid in out.scores:
                label += f"  s={out.scores[tid]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if is_t else 1)
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        hud = f"{out.state}"
        if out.state == ENROLLING and sel.cand_id is not None:
            hud += f"  candidate {sel.cand_id} held {t - sel.cand_since:.1f}/{sel.hold_s:.0f}s motion {sel._candidate_motion():.2f}"
        if out.state in (LOST, DROPPED):
            limit = sel.occlusion_max_s if out.state == LOST else sel.occlusion_max_s + sel.hover_s
            hud += f"  {t - sel.lost_since:.1f}/{limit:.0f}s  provisional {sel.prov_id} x{sel.prov_count}"
        if out.state == LOCKED:
            hud += f"  target {out.target_id}  gallery {len(sel.gallery)}"
        cv2.putText(frame, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors[out.state], 2)
        cv2.imshow("target selector (r restart, q quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            sel = TargetSelector(params, reid, t0=t)
            print(f"t={t:5.1f}s restarted enrollment")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{len(transitions)} events; final state {sel.state}, target {sel.target_id}, gallery {len(sel.gallery)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    _bench(ap.parse_args().camera)
