"""Person tracker: detector boxes in, persistent per-person IDs out.

Stage 1 task "Tracker added, stable IDs confirmed". Uses Ultralytics' built-in
ByteTrack. A stable tracker assigns one person one ID for as long as they are
visible; every new ID on the same person is an "ID switch" and counts against
track continuity (metric 2).

Usage:
    python src/perception/tracker.py --camera 0
Press q to quit. On exit prints frames per second, the number of distinct IDs
assigned, and the longest unbroken track in frames and seconds.
"""
import argparse
import pathlib
import time
from collections import defaultdict

import cv2
import yaml
from ultralytics import YOLO

REPO = pathlib.Path(__file__).resolve().parents[2]
PERSON_CLASS = 0


def load_params():
    with open(REPO / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def color_for(track_id):
    # Deterministic distinct colour per ID so a switch is visible on screen.
    import colorsys
    h = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.8, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    params = load_params()
    det = params["detector"]
    trk = params.get("tracker", {})
    device = pick_device()
    model = YOLO(det.get("weights", "yolo11n.pt"))
    tracker_cfg = trk.get("config", "bytetrack.yaml")
    if not pathlib.Path(tracker_cfg).is_absolute() and (REPO / tracker_cfg).exists():
        tracker_cfg = str(REPO / tracker_cfg)
    track_conf = float(trk.get("detector_conf", 0.1))
    print(f"model {det.get('weights', 'yolo11n.pt')} on {device}, tracker {tracker_cfg}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, det["input_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, det["input_height"])
    if not cap.isOpened():
        raise SystemExit(f"camera {args.camera} did not open")

    frames_per_id = defaultdict(int)       # total frames each ID was present
    current_run = {}                        # id -> consecutive frames so far
    longest_run = {}                        # id -> best consecutive run
    seen_order = []                         # IDs in first-seen order
    t_start = time.perf_counter()
    t_prev = t_start
    fps = 0.0
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, tracker=tracker_cfg,
                          classes=[PERSON_CLASS], conf=track_conf,
                          device=device, verbose=False)[0]

        ids_now = set()
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid in zip(res.boxes, res.boxes.id.int().tolist()):
                ids_now.add(tid)
                if tid not in frames_per_id:
                    seen_order.append(tid)
                frames_per_id[tid] += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                col = color_for(tid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"ID {tid}  {float(box.conf[0]):.2f}",
                            (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        for tid in ids_now:
            current_run[tid] = current_run.get(tid, 0) + 1
            longest_run[tid] = max(longest_run.get(tid, 0), current_run[tid])
        for tid in list(current_run):
            if tid not in ids_now:
                current_run[tid] = 0

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        n += 1
        cv2.putText(frame, f"{fps:4.1f} fps  ids assigned so far: {len(seen_order)}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("tracker (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start
    mean_fps = n / elapsed if elapsed else 0
    print(f"{n} frames in {elapsed:.1f} s = {mean_fps:.1f} fps end to end")
    print(f"distinct IDs assigned: {len(seen_order)}  (order: {seen_order})")
    if longest_run:
        best_id = max(longest_run, key=longest_run.get)
        best = longest_run[best_id]
        print(f"longest unbroken track: ID {best_id}, {best} frames "
              f"= {best / mean_fps:.1f} s at the measured frame rate")
        for tid in seen_order:
            print(f"  ID {tid}: {frames_per_id[tid]} frames total, "
                  f"longest run {longest_run[tid]}")


if __name__ == "__main__":
    main()
