"""Flight recorder: every camera frame and every per-frame result, one folder per flight.

Written for the Jetson's main loop, which must not wait on the SD card: frame()
only hands the data to a background thread. That thread compresses the picture
and writes the files. If it falls behind, pictures are dropped (and counted)
before the main loop is ever delayed; the data rows are always written.

One folder per flight, logs/flights/<date>_<time>_<label>/ (not committed):
  frames.mjpg      every camera frame as a JPEG picture, one after another
  frames.csv       one row per frame: time, where its picture is in frames.mjpg,
                   target state, range and bearing, the command sent, the watchdog's verdict
  tracks.csv       one row per person per frame: box, confidence, similarity score
  events.csv       lock, loss, re-lock, landing and watchdog events
  embeddings.bin   every re-ID embedding the pipeline computed (512 numbers each)
  embeddings.csv   which frame and person each embedding belongs to
  meta.json        settings in force, software version, totals

The files are plain rows appended as the flight goes and flushed every second,
so a crash or a pulled battery loses at most the last second. frames.mjpg needs
no closing step: each picture is found from its offset in frames.csv.

Use:
    rec = FlightRecorder(REPO / "logs" / "flights", "trail-1", params)
    rec.frame(image, t, boxes, ids, confs, target=out, command=cmd, verdict=v)
    rec.event(t, "watchdog", "watchdog -> LAND (...)")
    summary = rec.close()

    log = FlightLog(folder)          # read a flight back
    log.frames[i], log.image(i), log.tracks[i], log.events, log.embeddings()
"""
import collections
import csv
import json
import os
import pathlib
import queue
import subprocess
import threading
import time

import cv2
import numpy as np

FRAME_COLS = ["frame", "t_s", "unix_time", "jpeg_offset", "jpeg_bytes", "people", "target_state", "target_id",
              "range_m", "bearing_rad", "flight_state", "cmd_vx", "cmd_vy", "cmd_vz", "cmd_yaw_rate",
              "watchdog", "limited", "loop_ms"]
TRACK_COLS = ["frame", "t_s", "track_id", "x1", "y1", "x2", "y2", "confidence", "is_target", "score"]
EVENT_COLS = ["frame", "t_s", "source", "text"]
EMB_COLS = ["row", "frame", "t_s", "track_id", "kind"]
EMB_DIM = 512


def _num(v, digits=3):
    return "" if v is None else round(float(v), digits)


class FlightRecorder:
    def __init__(self, root, label="", params=None, meta=None):
        cfg = (params or {}).get("logging", {})
        self.quality = int(cfg.get("jpeg_quality", 85))
        self.max_pending = int(cfg.get("queue_frames", 90))
        self.flush_s = float(cfg.get("flush_s", 1.0))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.folder = pathlib.Path(root) / (stamp + (f"_{label}" if label else ""))
        self.folder.mkdir(parents=True, exist_ok=False)
        self.meta = {"label": label, "started": time.strftime("%Y-%m-%d %H:%M:%S %z"), "software": _git_version(),
                     "jpeg_quality": self.quality, "params": params or {}}
        self.meta.update(meta or {})
        self._write_meta()

        self._q = queue.Queue()
        self._pending = 0                      # pictures waiting to be compressed
        self._lock = threading.Lock()
        self.n = 0                             # frames handed in
        self.dropped = 0                       # pictures dropped because the writer was behind
        self.max_pending_seen = 0
        self.encode_ms = collections.deque(maxlen=20000)
        self.bytes = 0
        self._t_first = self._t_last = None
        self._thread = threading.Thread(target=self._run, name="flight-recorder", daemon=True)
        self._thread.start()

    # called from the main loop: must return at once ------------------------
    def frame(self, image, t, boxes=(), ids=(), confs=None, target=None, estimate=None, command=None,
              verdict=None, flight_state="", loop_ms=None):
        """image: the camera frame (not modified, not copied; do not draw on it afterwards).
        target: target.TargetOutput. estimate: (range_m, bearing_rad). command: object with vx, vy, vz,
        yaw_rate. verdict: watchdog.Verdict."""
        i = self.n
        self.n += 1
        with self._lock:
            if self._pending >= self.max_pending:
                image = None
                self.dropped += 1
            else:
                self._pending += 1
                self.max_pending_seen = max(self.max_pending_seen, self._pending)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        self._q.put(("frame", i, float(t), time.time(), image, boxes, list(ids),
                     None if confs is None else list(confs), target, estimate, command, verdict,
                     flight_state, loop_ms))
        return i

    def event(self, t, source, text):
        self._q.put(("event", max(self.n - 1, 0), float(t), source, text))

    def close(self):
        """Write everything still waiting, close the files and return the totals."""
        self._q.put(("stop",))
        self._thread.join()
        return self.summary()

    def summary(self):
        enc = np.array(self.encode_ms) if self.encode_ms else np.array([0.0])
        dur = (self._t_last - self._t_first) if self.n > 1 and self._t_first is not None else 0.0
        return {"folder": str(self.folder), "frames": self.n, "pictures_dropped": self.dropped,
                "seconds": round(dur, 1), "megabytes": round(self.bytes / 1e6, 2),
                "megabytes_per_20_min": round(self.bytes / 1e6 / dur * 1200, 0) if dur > 0 else None,
                "compress_ms_median": round(float(np.median(enc)), 2),
                "compress_ms_95th": round(float(np.percentile(enc, 95)), 2),
                "most_pictures_waiting": self.max_pending_seen}

    # background thread ----------------------------------------------------
    def _write_meta(self):
        tmp = self.folder / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=1, default=str))
        os.replace(tmp, self.folder / "meta.json")

    def _run(self):
        f = self.folder
        mjpg = open(f / "frames.mjpg", "wb")
        emb = open(f / "embeddings.bin", "wb")
        files = [mjpg, emb]
        writers = {}
        for name, cols in (("frames", FRAME_COLS), ("tracks", TRACK_COLS), ("events", EVENT_COLS),
                           ("embeddings", EMB_COLS)):
            fh = open(f / f"{name}.csv", "w", newline="")
            files.append(fh)
            writers[name] = csv.writer(fh)
            writers[name].writerow(cols)
        offset, emb_row = 0, 0
        last_flush = last_sync = time.time()
        jpeg_opts = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        while True:
            try:
                item = self._q.get(timeout=self.flush_s)
            except queue.Empty:
                item = None
            if item is not None and item[0] == "stop":
                break
            if item is not None and item[0] == "event":
                _, i, t, source, text = item
                writers["events"].writerow([i, round(t, 3), source, text])
            elif item is not None:
                (_, i, t, wall, image, boxes, ids, confs, target, estimate, command, verdict,
                 flight_state, loop_ms) = item
                if self._t_first is None:
                    self._t_first = t
                    if image is not None:
                        self.meta["frame_height"], self.meta["frame_width"] = image.shape[:2]
                self._t_last = t
                size = 0
                if image is not None:
                    a = time.perf_counter()
                    ok, buf = cv2.imencode(".jpg", image, jpeg_opts)
                    if ok:
                        mjpg.write(buf.tobytes())
                        size = len(buf)
                    self.encode_ms.append((time.perf_counter() - a) * 1000)
                    with self._lock:
                        self._pending -= 1
                state = getattr(target, "state", "")
                tid = getattr(target, "target_id", None)
                scores = getattr(target, "scores", None) or {}
                rng, brg = estimate if estimate is not None else (None, None)
                c = command
                writers["frames"].writerow([
                    i, round(t, 3), round(wall, 3), offset if size else "", size, len(ids), state,
                    "" if tid is None else tid, _num(rng), _num(brg, 4), flight_state,
                    _num(getattr(c, "vx", None)), _num(getattr(c, "vy", None)), _num(getattr(c, "vz", None)),
                    _num(getattr(c, "yaw_rate", None), 4), getattr(verdict, "action", ""),
                    "" if verdict is None else int(bool(getattr(verdict, "limited", False))), _num(loop_ms, 2)])
                offset += size
                self.bytes += size
                locked = state == "LOCKED"
                for k, track in enumerate(ids):
                    x1, y1, x2, y2 = boxes[k]
                    writers["tracks"].writerow([
                        i, round(t, 3), track, round(float(x1), 1), round(float(y1), 1), round(float(x2), 1),
                        round(float(y2), 1), "" if confs is None else round(float(confs[k]), 3),
                        int(locked and track == tid), _num(scores.get(track), 4)])
                for text in getattr(target, "events", None) or []:
                    writers["events"].writerow([i, round(t, 3), "target", text])
                for text in getattr(verdict, "events", None) or []:
                    writers["events"].writerow([i, round(t, 3), "watchdog", text])
                feats = getattr(target, "feats", None) or {}
                for track, vec in feats.items():
                    v = np.asarray(vec, dtype=np.float32).reshape(-1)
                    if v.size != EMB_DIM:
                        continue
                    emb.write(v.tobytes())
                    kind = ("gallery" if getattr(target, "learned", False) and track == tid
                            else "target" if locked else "candidate")
                    writers["embeddings"].writerow([emb_row, i, round(t, 3), track, kind])
                    emb_row += 1
                    self.bytes += v.nbytes
            now = time.time()
            if now - last_flush >= self.flush_s:
                for fh in files:
                    fh.flush()
                last_flush = now
                if now - last_sync >= 5.0:
                    for fh in files:
                        os.fsync(fh.fileno())
                    last_sync = now
        for fh in files:
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()
        self.meta["totals"] = self.summary()
        self.meta["ended"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        self._write_meta()


def _git_version():
    try:
        repo = pathlib.Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(repo), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class FlightLog:
    """Reads a flight folder, complete or cut short by a crash."""

    def __init__(self, folder):
        self.folder = pathlib.Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text())
        self.frames = self._rows("frames")
        self.events = self._rows("events")
        self.tracks = collections.defaultdict(list)          # frame number -> list of track rows
        for row in self._rows("tracks"):
            self.tracks[int(row["frame"])].append(row)
        self.mjpg_bytes = (self.folder / "frames.mjpg").stat().st_size
        # a crash can leave the last picture half written: keep only rows whose picture is all there
        self.frames = [r for r in self.frames
                       if not r["jpeg_bytes"] or r["jpeg_bytes"] == "0"
                       or int(r["jpeg_offset"]) + int(r["jpeg_bytes"]) <= self.mjpg_bytes]
        self._fh = None

    def _rows(self, name):
        path = self.folder / f"{name}.csv"
        if not path.exists():
            return []
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        # a crash can leave the last row half written
        want = {"frames": FRAME_COLS, "tracks": TRACK_COLS, "events": EVENT_COLS, "embeddings": EMB_COLS}[name]
        return [r for r in rows if None not in r.values() and len(r) == len(want)]

    def image(self, index):
        """The camera frame for self.frames[index], or None if its picture was dropped."""
        row = self.frames[index]
        if not row["jpeg_bytes"] or row["jpeg_bytes"] == "0":
            return None
        if self._fh is None:
            self._fh = open(self.folder / "frames.mjpg", "rb")
        self._fh.seek(int(row["jpeg_offset"]))
        data = self._fh.read(int(row["jpeg_bytes"]))
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def embeddings(self):
        """(index rows, array of shape rows x 512)."""
        rows = self._rows("embeddings")
        data = np.fromfile(self.folder / "embeddings.bin", dtype=np.float32)
        n = min(len(rows), data.size // EMB_DIM)
        return rows[:n], data[:n * EMB_DIM].reshape(n, EMB_DIM)

    def fps(self):
        if len(self.frames) < 2:
            return 30.0
        span = float(self.frames[-1]["t_s"]) - float(self.frames[0]["t_s"])
        return (len(self.frames) - 1) / span if span > 0 else 30.0
