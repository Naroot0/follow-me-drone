"""After a session, on the Mac: cut the useful crops out of a recorded flight and shrink the video.

For each flight folder written by recorder.py this tool:
  1. picks which person crops to keep (rules below) and saves them under crops/
     in three folders: target, other, unlabeled (the person was not locked at
     that moment, so the label needs a human)
  2. saves the original camera frames around every lock, loss and re-lock
     under hard_frames/
  3. re-encodes all frames as one video file, video.mp4, about a tenth the size
  4. with --delete-frames, removes the large frames.mjpg once 1 to 3 succeeded

Which crops are kept (numbers in config/params.yaml, logging.cull):
  - every crop within event_window_s of a lock, loss or re-lock
  - every crop whose similarity score is within close_score of the re-lock threshold
  - otherwise one crop per person every interval_s
  - between those, every novelty_every_frames frames, a crop that the re-ID model
    scores as less than novelty_max_similarity like the last one kept (a new view)

All processor cores are used for decoding and cutting. The new-view rule needs
PyTorch and the re-ID weights; without them it is skipped and the tool says so.

    python src/flightlog/archive.py logs/flights/20261125_101500_trail-1
    python src/flightlog/archive.py --all                # every flight not yet archived
    python src/flightlog/archive.py <folder> --delete-frames
"""
import argparse
import concurrent.futures
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "flightlog"))
sys.path.insert(0, str(REPO / "src" / "perception"))
from recorder import FlightLog   # noqa: E402

DEFAULT_CULL = {"event_window_s": 1.0, "close_score": 0.10, "interval_s": 1.0,
                "novelty_every_frames": 6, "novelty_max_similarity": 0.95}
CROP_COLS = ["file", "frame", "t_s", "track_id", "label", "reason", "score", "x1", "y1", "x2", "y2"]


def has_picture(row):
    return row["jpeg_bytes"] not in ("", "0")


def plan(log, cull, threshold):
    """Returns (keep, candidates, hard): keep maps (frame index, track) -> reason;
    candidates is the list of (frame index, track) for the new-view rule; hard is the set of
    frame indexes whose full picture is kept."""
    times = np.array([float(r["t_s"]) for r in log.frames])
    hard = set()
    for ev in log.events:
        if "->" in ev["text"] or ev["source"] != "target":
            t = float(ev["t_s"])
            lo, hi = np.searchsorted(times, [t - cull["event_window_s"], t + cull["event_window_s"]])
            hard.update(range(int(lo), int(hi)))
    reason_hard = {i: "event" for i in hard}
    for i, row in enumerate(log.frames):
        target = row["target_id"]
        for tr in log.tracks.get(int(row["frame"]), []):
            if tr["score"] == "":
                continue
            s = float(tr["score"])
            mine = tr["track_id"] == target
            if (mine and s <= threshold + cull["close_score"]) or (not mine and s >= threshold - cull["close_score"]):
                reason_hard.setdefault(i, "close score")
    keep, candidates, last = {}, [], {}
    for i, row in enumerate(log.frames):
        if not has_picture(row):
            continue
        t = times[i]
        for tr in log.tracks.get(int(row["frame"]), []):
            track = tr["track_id"]
            if i in reason_hard:
                keep[(i, track)] = reason_hard[i]
            elif t - last.get(track, -1e9) >= cull["interval_s"]:
                keep[(i, track)] = "interval"
                last[track] = t
            elif int(row["frame"]) % int(cull["novelty_every_frames"]) == 0:
                candidates.append((i, track))
    return keep, candidates, hard


def box_of(tr):
    return [float(tr[k]) for k in ("x1", "y1", "x2", "y2")]


def new_views(log, keep, candidates, cull, workers):
    """Adds candidates that look new to the re-ID model. Returns how many, or None if the model is unavailable."""
    try:
        from reid import ReID
        weights = log.meta.get("params", {}).get("reid", {}).get("weights", "osnet_x0_25_msmt17.pt")
        reid = ReID(weights=weights)
    except Exception as exc:
        print(f"  new-view rule skipped: the re-ID model could not be loaded ({type(exc).__name__}: {exc})")
        return None
    need = {}                                                   # frame index -> list of tracks
    for (i, track), reason in keep.items():
        if reason == "interval":
            need.setdefault(i, []).append(track)
    for i, track in candidates:
        need.setdefault(i, []).append(track)
    cand = set(candidates)
    order = sorted(need)
    rows = {i: {tr["track_id"]: tr for tr in log.tracks.get(int(log.frames[i]["frame"]), [])} for i in order}
    data = (log.folder / "frames.mjpg").read_bytes()

    def decode(i):
        r = log.frames[i]
        o, n = int(r["jpeg_offset"]), int(r["jpeg_bytes"])
        return cv2.imdecode(np.frombuffer(data, np.uint8, n, o), cv2.IMREAD_COLOR)

    added, last = 0, {}
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:    # decoding releases the interpreter lock
        for i, image in zip(order, pool.map(decode, order)):
            tracks = need[i]
            feats = reid.embed_many(image, np.array([box_of(rows[i][t]) for t in tracks], np.float32))
            for track, f in zip(tracks, feats):
                if np.isnan(f).any():
                    continue
                if (i, track) in cand:
                    ref = last.get(track)
                    if ref is not None and float(np.dot(ref, f) / (np.linalg.norm(ref) * np.linalg.norm(f) + 1e-9)) \
                            < cull["novelty_max_similarity"]:
                        keep[(i, track)] = "new view"
                        last[track] = f
                        added += 1
                else:
                    last[track] = f
    return added


def cut(job):
    """Worker: decode the listed frames, write crops and hard frames. Returns the crop rows."""
    folder, items = job
    folder = pathlib.Path(folder)
    out = []
    with open(folder / "frames.mjpg", "rb") as fh:
        for frame_no, t, offset, size, save_frame, crops in items:
            fh.seek(offset)
            data = fh.read(size)
            if save_frame:
                (folder / "hard_frames" / f"f{frame_no:06d}.jpg").write_bytes(data)
            if not crops:
                continue
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            for track, box, label, reason, score in crops:
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                name = f"{label}/f{frame_no:06d}_t{track}.jpg"
                cv2.imwrite(str(folder / "crops" / name), image[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
                out.append([name, frame_no, t, track, label, reason, score, x1, y1, x2, y2])
    return out


def encoder():
    if not shutil.which("ffmpeg"):
        return None
    listed = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "hevc_videotoolbox" in listed:                           # the Mac's hardware encoder
        return ["-c:v", "hevc_videotoolbox", "-q:v", "55", "-tag:v", "hvc1"]
    if "libx265" in listed:
        return ["-c:v", "libx265", "-crf", "24", "-preset", "medium", "-tag:v", "hvc1", "-loglevel", "error"]
    if "libx264" in listed:
        return ["-c:v", "libx264", "-crf", "22", "-preset", "medium"]
    return None


def make_video(log):
    enc = encoder()
    if enc is None:
        print("  video skipped: ffmpeg was not found (install it with: brew install ffmpeg)")
        return None
    out = log.folder / "video.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "mjpeg", "-framerate", f"{log.fps():.3f}",
           "-i", str(log.folder / "frames.mjpg"), *enc, "-pix_fmt", "yuv420p", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        print(f"  video failed: {done.stderr.strip()[-300:]}")
        return None
    return out


def mb(n):
    return f"{n / 1e6:.1f} MB"


def archive(folder, args, workers):
    t0 = time.time()
    log = FlightLog(folder)
    if not (log.folder / "frames.mjpg").exists() or log.mjpg_bytes == 0:
        print(f"{log.folder.name}: no frames.mjpg, nothing to do")
        return False
    params = log.meta.get("params", {})
    cull = dict(DEFAULT_CULL, **(params.get("logging", {}).get("cull", {}) or {}))
    threshold = float(params.get("reacquisition", {}).get("reid_similarity_threshold", 0.70))
    print(f"{log.folder.name}: {len(log.frames)} frames, {float(log.frames[-1]['t_s']) - float(log.frames[0]['t_s']):.0f} s, "
          f"frames.mjpg {mb(log.mjpg_bytes)}")

    keep, candidates, hard = plan(log, cull, threshold)
    added = None if args.no_new_views else new_views(log, keep, candidates, cull, workers)

    for sub in ("crops/target", "crops/other", "crops/unlabeled", "hard_frames"):
        (log.folder / sub).mkdir(parents=True, exist_ok=True)
    per_frame = {}
    for (i, track), reason in keep.items():
        per_frame.setdefault(i, []).append((track, reason))
    items = []
    for i in sorted(set(per_frame) | {h for h in hard if has_picture(log.frames[h])}):
        row = log.frames[i]
        tracks = {tr["track_id"]: tr for tr in log.tracks.get(int(row["frame"]), [])}
        crops = []
        for track, reason in per_frame.get(i, []):
            tr = tracks[track]
            label = ("target" if tr["is_target"] == "1"
                     else "other" if row["target_state"] == "LOCKED" else "unlabeled")
            crops.append((track, box_of(tr), label, reason, tr["score"]))
        items.append((int(row["frame"]), row["t_s"], int(row["jpeg_offset"]), int(row["jpeg_bytes"]), i in hard, crops))
    chunk = max(1, len(items) // (workers * 4) + 1)
    jobs = [(str(log.folder), items[k:k + chunk]) for k in range(0, len(items), chunk)]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        for part in pool.map(cut, jobs):
            rows.extend(part)
    rows.sort(key=lambda r: (r[1], str(r[3])))
    with open(log.folder / "crops.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CROP_COLS)
        w.writerows(rows)

    video = None if args.no_video else make_video(log)

    people = sum(len(log.tracks.get(int(r["frame"]), [])) for r in log.frames if has_picture(r))
    by_label, by_reason = {}, {}
    for r in rows:
        by_label[r[4]] = by_label.get(r[4], 0) + 1
        by_reason[r[5]] = by_reason.get(r[5], 0) + 1
    size = lambda d: sum(f.stat().st_size for f in (log.folder / d).rglob("*") if f.is_file())
    crops_b, hard_b = size("crops"), size("hard_frames")
    small = sum(f.stat().st_size for f in log.folder.glob("*.csv")) + (log.folder / "embeddings.bin").stat().st_size
    print(f"  crops kept: {len(rows)} of {people} ({mb(crops_b)}): "
          + ", ".join(f"{v} {k}" for k, v in sorted(by_label.items())))
    print("  kept because: " + ", ".join(f"{v} {k}" for k, v in sorted(by_reason.items()))
          + ("" if added is not None or args.no_new_views else "  (new-view rule not applied)"))
    print(f"  original frames kept around events: {len([h for h in hard if has_picture(log.frames[h])])} ({mb(hard_b)})")
    if video:
        print(f"  video.mp4: {mb(video.stat().st_size)} ({log.mjpg_bytes / video.stat().st_size:.0f} times smaller than frames.mjpg)")
    total = crops_b + hard_b + small + (video.stat().st_size if video else 0)
    print(f"  folder without frames.mjpg: {mb(total)}   (with it: {mb(total + log.mjpg_bytes)})   "
          f"done in {time.time() - t0:.1f} s on {workers} cores")
    meta = dict(log.meta, archive={"crops": len(rows), "by_label": by_label, "by_reason": by_reason,
                                   "new_view_rule": added is not None, "video": bool(video),
                                   "when": time.strftime("%Y-%m-%d %H:%M:%S")})
    if args.delete_frames:
        if video and rows:
            os.remove(log.folder / "frames.mjpg")
            meta["archive"]["frames_removed"] = True
            print("  frames.mjpg removed")
        else:
            print("  frames.mjpg kept: the video or the crops were not produced")
    (log.folder / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="*", help="flight folders under logs/flights/")
    ap.add_argument("--all", action="store_true", help="every flight in logs/flights/ that has not been archived")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-new-views", action="store_true", help="skip the rule that needs the re-ID model")
    ap.add_argument("--delete-frames", action="store_true", help="remove frames.mjpg after a successful archive")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    folders = [pathlib.Path(f) for f in args.folders]
    if args.all:
        root = REPO / "logs" / "flights"
        folders += [d for d in sorted(root.iterdir()) if (d / "frames.mjpg").exists() and not (d / "crops.csv").exists()]
    if not folders:
        ap.error("give a flight folder or --all")
    for folder in folders:
        archive(folder, args, args.workers)


if __name__ == "__main__":
    main()
