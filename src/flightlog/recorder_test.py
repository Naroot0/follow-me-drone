"""Flight recorder test with a made-up flight: no camera, autopilot or model needed.

  1. cost        the main loop's cost per frame while recording at 30 frames per second
  2. nothing lost  every frame, person, event and embedding is read back
  3. pictures    saved frames decode and match what was recorded
  4. overload    the main loop is not delayed when the writer cannot keep up; pictures are
                 dropped and counted, data rows are all kept
  5. crash       a copy of the folder taken mid-flight, with no closing step, still reads back

    python src/flightlog/recorder_test.py
Leaves one example flight in logs/flights/ for trying the archive tool on.
"""
import pathlib
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "flightlog"))
from recorder import FlightRecorder, FlightLog   # noqa: E402

W, H, FPS = 640, 480, 30
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


@dataclass
class Out:                       # stands in for target.TargetOutput
    state: str
    target_id: int | None
    scores: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    feats: dict = field(default_factory=dict)
    learned: bool = False


@dataclass
class Cmd:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class Verd:
    action: str = "PASS"
    limited: bool = False
    events: list = field(default_factory=list)


class Scene:
    """A textured background with two moving 'people', and a scripted lock, loss and re-lock."""

    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        base = rng.integers(0, 255, (H // 4, W // 4, 3), dtype=np.uint8)
        self.bg = cv2.GaussianBlur(cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 3)
        self.tex = [cv2.GaussianBlur(rng.integers(0, 255, (260, 110, 3), dtype=np.uint8), (0, 0), 2) for _ in range(2)]
        self.rng = rng
        self.vec = [rng.standard_normal(512).astype(np.float32) for _ in range(2)]

    def frame(self, i):
        t = i / FPS
        img = self.bg.copy()
        img = cv2.add(img, self.rng.integers(0, 6, img.shape, dtype=np.uint8))       # sensor noise
        hidden = 10.0 <= t < 14.0                                                    # person 1 out of view
        boxes, ids = [], []
        for k in range(2):
            if k == 0 and hidden:
                continue
            x = int(120 + 300 * k + 60 * np.sin(t * (0.7 + 0.3 * k)))
            y = int(150 + 20 * np.cos(t))
            img[y:y + 260, x:x + 110] = self.tex[k]
            boxes.append([x, y, x + 110, y + 260])
            ids.append(k + 1)
        if t < 5.0:
            out = Out("ENROLLING", None)
        elif t < 10.0:
            out = Out("LOCKED", 1)
            if i == 5 * FPS:
                out.events.append("ENROLLING -> LOCKED on track 1")
                out.feats, out.learned = {1: self.vec[0]}, True
        elif t < 14.5:
            out = Out("LOST", 1)
            if i == 10 * FPS:
                out.events.append("LOCKED -> LOST (track 1 missing)")
            for tid in ids:
                out.scores[tid] = 0.82 if tid == 1 else 0.58 + 0.05 * np.sin(t * 3)
                out.feats[tid] = self.vec[tid - 1] + 0.01 * i
        else:
            out = Out("LOCKED", 1)
            if i == int(14.5 * FPS):
                out.events.append("LOST -> LOCKED on track 1 (was 1); s=0.82")
        cmd = Cmd(vx=1.2 if out.state == "LOCKED" else 0.0, yaw_rate=0.05)
        return img, t, np.array(boxes, np.float32), ids, [0.9] * len(ids), out, (3.1, 0.02), cmd, Verd()


def record(root, params, frames, paced, label, snapshot_at=None, snapshot_to=None):
    scene = Scene()
    rec = FlightRecorder(root, label, params)
    call_ms, keep = [], {}
    tick = time.perf_counter()
    ready = None if paced else [scene.frame(i) for i in range(30)]    # overload: frames prepared in advance
    for i in range(frames):
        img, t, boxes, ids, confs, out, est, cmd, verd = scene.frame(i) if paced else ready[i % 30]
        if not paced:
            t = i / FPS
        if i % 97 == 0:
            keep[i] = img.copy()
        a = time.perf_counter()
        rec.frame(img, t, boxes, ids, confs, target=out, estimate=est, command=cmd, verdict=verd,
                  flight_state="FOLLOW", loop_ms=33.0)
        if i == 20 * FPS:
            rec.event(t, "pilot", "mode switch to LOITER")
        call_ms.append((time.perf_counter() - a) * 1000)
        if snapshot_at is not None and i == snapshot_at:
            time.sleep(1.3)                                   # one flush interval
            shutil.copytree(rec.folder, snapshot_to)
        if paced:
            tick += 1.0 / FPS
            time.sleep(max(0.0, tick - time.perf_counter()))
    summary = rec.close()
    return rec.folder, summary, np.array(call_ms), keep


def main():
    with open(REPO / "config" / "params.yaml") as f:
        params = yaml.safe_load(f)
    root = REPO / "logs" / "flights"
    root.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="recorder_test_"))

    n = 25 * FPS
    folder, s, call_ms, keep = record(root, params, n, True, "recorder-test", snapshot_at=12 * FPS,
                                      snapshot_to=tmp / "crash")
    check("cost", np.percentile(call_ms, 95) < 1.0 and s["pictures_dropped"] == 0,
          f"main loop pays {np.median(call_ms) * 1000:.0f} microseconds per frame (95th percentile "
          f"{np.percentile(call_ms, 95) * 1000:.0f}); the writer thread compresses a frame in "
          f"{s['compress_ms_median']} ms; at most {s['most_pictures_waiting']} pictures waiting; "
          f"{s['megabytes']} MB for {s['seconds']} s, which is {s['megabytes_per_20_min']:.0f} MB per 20 minutes "
          f"for this made-up scene")

    log = FlightLog(folder)
    rows, vecs = log.embeddings()
    people = sum(len(v) for v in log.tracks.values())
    scene = Scene()
    want_people = sum(len(scene.frame(i)[3]) for i in range(n))
    kinds = sorted({r["kind"] for r in rows})
    check("nothing lost", len(log.frames) == n and people == want_people and len(log.events) == 4
          and len(rows) == vecs.shape[0] > 100 and kinds == ["candidate", "gallery"]
          and log.meta["totals"]["frames"] == n and log.frames[200]["cmd_vx"] == "1.2"
          and log.frames[200]["watchdog"] == "PASS",
          f"{len(log.frames)} frames, {people} person rows, {len(log.events)} events, {len(rows)} embeddings "
          f"({', '.join(kinds)}), settings and totals in meta.json")

    worst = 99.0
    for i, original in keep.items():
        back = log.image(i)
        mse = np.mean((back.astype(np.float32) - original.astype(np.float32)) ** 2)
        worst = min(worst, 10 * np.log10(255 ** 2 / max(mse, 1e-9)))
    check("pictures", worst > 30 and back.shape == (H, W, 3),
          f"{len(keep)} frames compared with the originals, closest match no worse than {worst:.0f} dB "
          f"(above 30 dB is hard to tell apart by eye)")

    slow = dict(params, logging=dict(params.get("logging", {}), queue_frames=5))
    folder2, s2, call2, _ = record(tmp, slow, 300, False, "overload")
    log2 = FlightLog(folder2)
    with_picture = sum(1 for r in log2.frames if r["jpeg_bytes"] not in ("", "0"))
    check("overload", s2["pictures_dropped"] > 0 and len(log2.frames) == 300
          and with_picture == 300 - s2["pictures_dropped"] and np.percentile(call2, 95) < 1.0
          and log2.image(next(i for i, r in enumerate(log2.frames) if r["jpeg_bytes"] not in ("", "0"))) is not None,
          f"frames fed with no pause and room for 5 waiting pictures: {s2['pictures_dropped']} pictures dropped "
          f"and counted, all 300 data rows kept, main loop still pays {np.percentile(call2, 95) * 1000:.0f} "
          f"microseconds at the 95th percentile")

    crash = FlightLog(tmp / "crash")
    last = len(crash.frames) - 1
    ok = last > 9 * FPS and crash.image(last) is not None and "totals" not in crash.meta
    check("crash", ok, f"folder copied mid-flight at frame {12 * FPS} with no closing step: {len(crash.frames)} "
          f"frames read back, the last one decodes, {len(crash.events)} events present")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nexample flight left in {folder}")
    print("RESULT:", "PASS" if all(results) else "FAIL", f"({sum(results)} of {len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
