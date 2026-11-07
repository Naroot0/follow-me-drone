"""Person detector: webcam in, person boxes out, frames-per-second on screen.

Stage 1 task "Detector running on the Mac with a webcam". No tracker yet.
Reads detector settings from config/params.yaml. Press q to quit.

Usage:
    python src/perception/detector.py                # camera index 0
    python src/perception/detector.py --camera 1     # external webcam
    python src/perception/detector.py --list         # probe camera indices 0 to 4
"""
import argparse
import pathlib
import time

import cv2
import yaml
from ultralytics import YOLO

REPO = pathlib.Path(__file__).resolve().parents[2]
PERSON_CLASS = 0  # COCO class index for "person"


def load_params():
    with open(REPO / "config" / "params.yaml") as f:
        return yaml.safe_load(f)["detector"]


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def list_cameras():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"camera {i}: open, {int(w)}x{int(h)}")
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        list_cameras()
        return

    p = load_params()
    device = pick_device()
    model = YOLO(p.get("weights", "yolo11n.pt"))
    print(f"model {p.get('weights', 'yolo11n.pt')} on {device}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, p["input_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, p["input_height"])
    if not cap.isOpened():
        raise SystemExit(f"camera {args.camera} did not open; try --list")

    t_prev = time.perf_counter()
    fps = 0.0
    n = 0
    t_start = t_prev
    infer_ms = []
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame read failed")
            break
        t0 = time.perf_counter()
        res = model.predict(frame, classes=[PERSON_CLASS], conf=p["confidence_min"],
                            device=device, verbose=False)[0]
        infer_ms.append((time.perf_counter() - t0) * 1000)

        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            c = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{c:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        n += 1
        cv2.putText(frame, f"{fps:4.1f} fps  {len(res.boxes)} person(s)  {device}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("detector (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start
    infer_ms.sort()
    if n:
        print(f"{n} frames in {elapsed:.1f} s = {n / elapsed:.1f} fps end to end")
        print(f"inference ms: median {infer_ms[len(infer_ms)//2]:.1f}, "
              f"p95 {infer_ms[int(len(infer_ms)*0.95)]:.1f}")


if __name__ == "__main__":
    main()
