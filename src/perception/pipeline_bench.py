"""End-to-end perception speed test with no display: camera, detector, tracker, re-ID, target selection.

Runs the same chain as target.py's bench, but prints numbers and saves a few
annotated frames instead of opening a window, so it works over a remote login
on the Jetson. It also runs on the Mac.

What to do while it runs: stand still in view until it prints LOCKED (about
5 s), step out of view, wait for LOST, step back in and wait for LOCKED again.

Usage:
    python src/perception/pipeline_bench.py                 # 90 s on camera 0
    python src/perception/pipeline_bench.py --seconds 120 --camera 0
    python src/perception/pipeline_bench.py --weights yolo11n.pt

Detector weights: --weights if given; otherwise a TensorRT engine named
yolo11n.engine in the repository folder or the home folder if one exists;
otherwise the weights named in config/params.yaml.

Prints every target-selection event as it happens, then end-to-end frames per
second and the time spent in each stage. Annotated frames for the first events
are saved under logs/bench/ (not committed).
"""
import argparse
import glob
import pathlib
import platform
import sys
import time

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reid import ReID                                                   # noqa: E402
from target import TargetSelector, LOCKED, LAND, FAILED                 # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "logs" / "bench"
MAX_SAVED = 12


def temperatures():
    """Board temperature sensors in degrees C (Jetson and other Linux boards); empty elsewhere."""
    out = {}
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            name = open(zone + "/type").read().strip().replace("-thermal", "")
            out[name] = int(open(zone + "/temp").read()) / 1000.0
        except Exception:
            pass                                                        # some sensors do not answer
    return out


def show_temperature(label):
    t = temperatures()
    if t:
        print(f"temperature {label}: hottest {max(t.values()):.1f} C  ("
              + "  ".join(f"{k} {v:.1f}" for k, v in t.items()) + ")", flush=True)


def pick_weights(arg, params):
    if arg:
        return arg
    for folder in (REPO, pathlib.Path.home()):
        engine = folder / "yolo11n.engine"
        if engine.exists():
            return str(engine)
    return params["detector"].get("weights", "yolo11n.pt")


def open_camera(index, width, height):
    if platform.system() == "Linux":
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # compressed stream, allows 30 fps and more
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} did not open")
    return cap


def summary(name, values):
    if not values:
        return f"  {name:28s} no samples"
    v = np.array(values)
    return f"  {name:28s} median {np.median(v):6.1f} ms   95th percentile {np.percentile(v, 95):6.1f} ms"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--weights", help="detector weights or TensorRT engine (default: see module text)")
    args = ap.parse_args()

    with open(REPO / "config" / "params.yaml") as f:
        params = yaml.safe_load(f)
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
    print(f"detector {pathlib.Path(weights).name}, re-ID {reid.name} ({reid.backend}) on {reid.device}, "
          f"frame {frame.shape[1]}x{frame.shape[0]}, running {args.seconds:.0f} s", flush=True)
    show_temperature("at start")
    OUT.mkdir(parents=True, exist_ok=True)

    def track(image):
        return yolo.track(image, persist=True, tracker=tracker_cfg, classes=[0],
                          conf=float(trk.get("detector_conf", 0.25)), imgsz=(height, width),
                          device=str(reid.device), verbose=False)[0]

    for _ in range(20):                                                 # warm-up, not timed
        ok, frame = cap.read()
        track(frame)

    sel = TargetSelector(params, reid)
    print("stand still in view until LOCKED, step out, wait for LOST, step back in", flush=True)
    t_capture, t_track, t_select, t_reid, t_total = [], [], [], [], []
    people_max, frames, saved, events_total = 0, 0, 0, 0
    last_state = None
    t0 = time.perf_counter()
    while True:
        a = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        b = time.perf_counter()
        res = track(frame)
        boxes, ids = [], []
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().tolist()
        c = time.perf_counter()
        t = c - t0
        out = sel.update(frame, boxes, ids, t)
        d = time.perf_counter()

        frames += 1
        people_max = max(people_max, len(ids))
        t_capture.append((b - a) * 1000)
        t_track.append((c - b) * 1000)
        t_select.append((d - c) * 1000)
        t_total.append((d - a) * 1000)
        if len(ids):
            t_reid.append(reid.last_ms)

        for event in out.events:
            events_total += 1
            print(event, flush=True)
        if out.state != last_state and saved < MAX_SAVED:
            shot = frame.copy()
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                target = tid == out.target_id and out.state == LOCKED
                colour = (0, 255, 0) if target else (160, 160, 160)
                cv2.rectangle(shot, (x1, y1), (x2, y2), colour, 3 if target else 1)
                cv2.putText(shot, f"ID {tid}" + ("  TARGET" if target else ""), (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            cv2.putText(shot, f"{out.state}  t={t:.1f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imwrite(str(OUT / f"{saved:02d}_{t:05.1f}s_{out.state}.jpg"), shot)
            saved += 1
        last_state = out.state

        if out.state in (LAND, FAILED):                                 # keep measuring: start a new enrollment
            print(f"t={t:5.1f}s {out.state} reached; restarting enrollment", flush=True)
            sel = TargetSelector(params, reid, t0=t)
            last_state = None
        if t >= args.seconds:
            break

    cap.release()
    elapsed = time.perf_counter() - t0
    print(f"\n{frames} frames in {elapsed:.1f} s = {frames / max(elapsed, 1e-6):.1f} frames per second end to end")
    print(summary("camera read (waits for frame)", t_capture))
    print(summary("detector and tracker", t_track))
    print(summary("target selection with re-ID", t_select))
    print(summary("re-ID model alone", t_reid))
    print(summary("whole frame", t_total))
    work = np.median(np.array(t_track) + np.array(t_select)) if frames else float("nan")
    print(f"  processing without the camera wait: median {work:.1f} ms, "
          f"which would allow {1000 / work:.0f} frames per second")
    print(f"most people in view at once: {people_max}; events: {events_total}; "
          f"final state {sel.state}; gallery {len(sel.gallery)}; {saved} frames saved in {OUT}")
    show_temperature("at end")
    print("DONE")


if __name__ == "__main__":
    main()
