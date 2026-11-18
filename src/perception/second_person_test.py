"""Second-person test with no display: can re-ID tell the enrolled person from someone else?

Two people are needed: the TARGET (the person to be followed) and a HELPER.
The script enrolls the target, then walks both through timed phases and measures
how the similarity scores of the two people compare, and whether the target
selector ever locks onto the helper.

Phases after enrollment (each starts with a few seconds to get into place):
  1. both in view, walking around and crossing paths
  2. target leaves, helper stays         -> the selector must NOT lock
  3. target returns, helper stays        -> the selector must lock again
  4. both leave
  5. helper returns alone                -> the selector must NOT lock
  6. target returns as well              -> the selector must lock again

Lines that start with "SAY:" are instructions for the two people. Run it from the
Mac with the command in the project notes and the Mac reads those lines aloud.

Usage (on the Jetson):
    python3 -u src/perception/second_person_test.py
    python3 -u src/perception/second_person_test.py --label similar-clothes

Writes logs/bench/second_person_<label>.csv (one row per person per frame) and
annotated frames for every lock event and for the helper-only phases. Neither is
committed. The landing timeout is disabled for this test so a long absence does
not end the run; every other setting comes from config/params.yaml.
"""
import argparse
import csv
import pathlib
import sys
import time

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reid import ReID                                                    # noqa: E402
from target import TargetSelector, ENROLLING, LOCKED, FAILED            # noqa: E402
from pipeline_bench import open_camera, pick_weights, show_temperature  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "logs" / "bench"
SETTLE_S = 5.0
clock = time.perf_counter                # replaced in tests

# name, seconds, instruction, expectation: "scores", "no_lock", "lock" or None
PHASES = [
    ("both", 45, "Helper, come into view. Both of you walk around and cross paths.", "scores"),
    ("helper_stays", 25, "Target, leave the view. Helper, stay in view and keep moving.", "no_lock"),
    ("target_returns", 30, "Target, come back into view. Helper, stay in view.", "lock"),
    ("empty", 10, "Both of you, leave the view.", None),
    ("helper_alone", 25, "Helper only, come back into view and keep moving.", "no_lock"),
    ("target_returns_again", 30, "Target, come back into view as well.", "lock"),
]


def say(text):
    print(f"SAY: {text}", flush=True)


def describe(name, values):
    if not len(values):
        return f"  {name:34s} no samples"
    v = np.array(values)
    return (f"  {name:34s} n={len(v):5d}  lowest {v.min():.2f}  5th pct {np.percentile(v, 5):.2f}  "
            f"median {np.median(v):.2f}  95th pct {np.percentile(v, 95):.2f}  highest {v.max():.2f}")


def longest_run(flags):
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--weights", help="detector weights or TensorRT engine")
    ap.add_argument("--label", default="run", help="name for this run's output files, e.g. similar-clothes")
    args = ap.parse_args()

    with open(REPO / "config" / "params.yaml") as f:
        params = yaml.safe_load(f)
    params.setdefault("loss", {})["hover_then_land_after_s"] = 3600      # no landing during the test
    params["acquisition"]["timeout_s"] = 3600                            # wait as long as enrollment takes
    det, trk = params["detector"], params.get("tracker", {})
    width, height = int(det["input_width"]), int(det["input_height"])
    tracker_cfg = trk.get("config", "bytetrack.yaml")
    if (REPO / tracker_cfg).exists():
        tracker_cfg = str(REPO / tracker_cfg)
    weights = pick_weights(args.weights, params)

    reid = ReID(weights=params.get("reid", {}).get("weights", "osnet_x0_25_msmt17.pt"))
    yolo = YOLO(weights, task="detect")
    cap = open_camera(args.camera, width, height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera opened but returned no frame")

    def track(image):
        return yolo.track(image, persist=True, tracker=tracker_cfg, classes=[0],
                          conf=float(trk.get("detector_conf", 0.25)), imgsz=(height, width),
                          device=str(reid.device), verbose=False)[0]

    for _ in range(20):
        ok, frame = cap.read()
        track(frame)

    sel = TargetSelector(params, reid)
    thresh, margin, need = sel.thresh, sel.margin, sel.reacq_frames
    print(f"detector {pathlib.Path(weights).name}, re-ID {reid.name} ({reid.backend}), threshold {thresh}, "
          f"margin {margin}, {need} frames to re-lock, label '{args.label}'", flush=True)
    show_temperature("at start")
    OUT.mkdir(parents=True, exist_ok=True)
    log_file = open(OUT / f"second_person_{args.label}.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["t", "phase", "settled", "state", "target_id", "track_id", "score", "is_target",
                  "x1", "y1", "x2", "y2"])

    def save(tag, t, boxes, ids, scores, out):
        shot = frame.copy()
        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            target = tid == out.target_id and out.state == LOCKED
            colour = (0, 255, 0) if target else (0, 165, 255)
            label = f"ID {tid}" + ("  TARGET" if target else "")
            if tid in scores:
                label += f"  s={scores[tid]:.2f}"
            cv2.rectangle(shot, (x1, y1), (x2, y2), colour, 3 if target else 2)
            cv2.putText(shot, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.putText(shot, f"{tag}  {out.state}  t={t:.1f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f"second_person_{args.label}_{t:06.1f}s_{tag}.jpg"), shot)

    say("Target only, stand still in view. Helper, stay out of view.")
    phase_index, phase_start = -1, None                # -1 = enrolling
    phase_name, expectation = "enroll", None
    stats = {name: {"lock_events": [], "locks_any": [], "locked_frames": 0, "frames": 0, "scores": [],
                    "last_shot": -99.0}
             for name, *_ in PHASES}
    self_scores, other_scores, margins = [], [], []
    frames = 0
    t0 = clock()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = track(frame)
        boxes, ids = np.zeros((0, 4), np.float32), []
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().tolist()
        t = clock() - t0
        out = sel.update(frame, boxes, ids, t)
        frames += 1
        for event in out.events:
            print(f"[{phase_name}] {event}", flush=True)

        # this test scores every visible person on every frame, whatever the selector needed
        scores = {}
        if sel.gallery and len(boxes):
            feats = reid.embed_many(frame, boxes)
            for tid, f in zip(ids, feats):
                if not np.isnan(f).any():
                    scores[tid] = ReID.best_match(sel.gallery, f)

        if phase_index == -1:
            if out.state == FAILED:
                raise SystemExit("enrollment failed")
            if out.state == LOCKED:
                save("enrolled", t, boxes, ids, scores, out)
                phase_index, phase_start = 0, t
            else:
                continue
        elif t - phase_start >= PHASES[phase_index][1] + SETTLE_S:
            phase_index, phase_start = phase_index + 1, t
            if phase_index >= len(PHASES):
                break
        if phase_start == t:                           # a phase just began
            phase_name, _, instruction, expectation = PHASES[phase_index]
            print(f"--- phase {phase_index + 1} of {len(PHASES)}: {phase_name}", flush=True)
            say(instruction)

        settled = (t - phase_start) >= SETTLE_S
        st = stats[phase_name]
        locked_now = out.state == LOCKED
        new_lock = any("-> LOCKED" in e and "ENROLLING" not in e for e in out.events)
        if new_lock:
            save(f"{phase_name}_lock", t, boxes, ids, scores, out)
            st["locks_any"].append(round(t - phase_start, 1))
        if settled:
            st["frames"] += 1
            st["locked_frames"] += int(locked_now)
            if new_lock:
                st["lock_events"].append(round(t - phase_start - SETTLE_S, 1))
            if expectation == "no_lock" and not locked_now:   # still locked means the target has not left yet
                st["scores"].append(max(scores.values()) if scores else float("nan"))
                if t - st["last_shot"] >= 5.0:
                    save(phase_name, t, boxes, ids, scores, out)
                    st["last_shot"] = t
            if expectation == "scores" and locked_now and out.target_id in scores:
                own = scores[out.target_id]
                others = [v for k, v in scores.items() if k != out.target_id]
                self_scores.append(own)
                if others:
                    other_scores.extend(others)
                    margins.append(own - max(others))
        for box, tid in zip(boxes, ids):
            log.writerow([f"{t:.3f}", phase_name, int(settled), out.state, out.target_id, tid,
                          f"{scores.get(tid, float('nan')):.4f}", int(tid == out.target_id and locked_now),
                          *[f"{v:.0f}" for v in box]])

    cap.release()
    log_file.close()
    say("Test finished. Thank you.")
    elapsed = clock() - t0
    print(f"\n{frames} frames in {elapsed:.0f} s = {frames / max(elapsed, 1e-6):.1f} frames per second")

    print("\nPhase 1, both in view (similarity to the enrolled gallery, target assumed to be the locked track):")
    print(describe("target", self_scores))
    print(describe("helper, same frames", other_scores))
    if margins:
        m = np.array(margins)
        print(f"  target scored higher than the helper in {np.mean(m > 0) * 100:.1f}% of {len(m)} frames; "
              f"margin median {np.median(m):.2f}, 5th pct {np.percentile(m, 5):.2f}")

    alone = np.array([s for name in ("helper_stays", "helper_alone") for s in stats[name]["scores"]], dtype=float)
    seen = alone[~np.isnan(alone)]
    print("\nHelper in view without the target (the wrong-person case):")
    print(describe("helper, best score per frame", seen))

    print(f"\nThreshold table (current threshold {thresh}, {need} frames in a row needed to re-lock):")
    print("  threshold   keeps target   admits helper (both in view)   admits helper (alone)   longest run alone")
    filled = np.nan_to_num(alone, nan=0.0)
    for th in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        keep = np.mean(np.array(self_scores) >= th) * 100 if self_scores else float("nan")
        adm_both = np.mean(np.array(other_scores) >= th) * 100 if other_scores else float("nan")
        adm_alone = np.mean(seen >= th) * 100 if len(seen) else float("nan")
        run = longest_run(filled >= th)
        flag = "  <- would lock on the helper" if run >= need else ""
        print(f"    {th:.2f}       {keep:5.1f}%            {adm_both:5.1f}%                     "
              f"{adm_alone:5.1f}%               {run:3d} frames{flag}")

    print("\nSelector behaviour:")
    verdicts = []
    for name, seconds, _, expect in PHASES:
        st = stats[name]
        if expect == "no_lock":
            passed = not st["lock_events"]
            detail = "no lock" if passed else f"LOCKED at {st['lock_events']} s into the phase"
        elif expect == "lock":
            passed = bool(st["locks_any"])
            detail = f"locked {st['locks_any'][0]} s after the instruction" if passed else "did not lock"
        else:
            continue
        verdicts.append(passed)
        want = "must not lock" if expect == "no_lock" else "must lock"
        print(f"  {'PASS' if passed else 'FAIL'}  {name:22s} ({want}): {detail}; "
              f"locked on {st['locked_frames']} of {st['frames']} frames")
    print("  Check the saved '_lock' pictures: each lock should be on the target, not the helper.")
    print(f"\nfiles: {OUT}/second_person_{args.label}*")
    show_temperature("at end")
    print(f"RESULT: {'PASS' if verdicts and all(verdicts) else 'FAIL'} (selector behaviour; confirm with the pictures)")
    print("DONE")


if __name__ == "__main__":
    main()
