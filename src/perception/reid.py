"""Person re-identification: a person crop in, a 512-d appearance vector out.

Stage 1 task "Re-ID model installed, embeddings on crops, latency measured".

Model: OSNet x0.25 trained on MSMT17 (torchreid model zoo weights). The model
definition is vendored in osnet.py. Weights are downloaded once into
weights/ (gitignored) from the torchreid model zoo.

Library use:
    from perception.reid import ReID
    reid = ReID(device="mps")
    vec = reid.embed(frame, [x1, y1, x2, y2])        # one crop -> (512,) unit vector
    vecs = reid.embed_many(frame, xyxy_array)         # N crops -> (N, 512)
    sim = ReID.cosine(vec_a, vec_b)                    # -1..1, higher = more alike

Bench (webcam, uses the tracker for boxes):
    python src/perception/reid.py --camera 0
    keys:  e  enroll the largest tracked person as the target
           q  quit and print latency and similarity statistics
On screen each track shows its ID and its cosine similarity to the enrolled
embedding. Similarities above reid_similarity_threshold (params.yaml) draw green.
"""
import argparse
import pathlib
import sys
import time

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from osnet import osnet_x0_25, osnet_x0_5, osnet_x0_75, osnet_x1_0  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO / "weights"

# torchreid model zoo, OSNet x0.25 trained on MSMT17 (the largest public re-ID set)
MODEL_ZOO = {
    "osnet_x0_25_msmt17.pt": "1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
    "osnet_x0_5_msmt17.pt": "1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv",
    "osnet_x0_75_msmt17.pt": "1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc",
    "osnet_x1_0_msmt17.pt": "112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
}
CONSTRUCTORS = {
    "osnet_x0_25": osnet_x0_25, "osnet_x0_5": osnet_x0_5,
    "osnet_x0_75": osnet_x0_75, "osnet_x1_0": osnet_x1_0,
}
INPUT_H, INPUT_W = 256, 128   # OSNet training resolution (height, width)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_weights(name="osnet_x0_25_msmt17.pt"):
    path = WEIGHTS_DIR / name
    if path.exists():
        return path
    WEIGHTS_DIR.mkdir(exist_ok=True)
    try:
        import gdown
    except ImportError:
        raise SystemExit("pip install gdown   (needed once, to fetch the re-ID weights)")
    print(f"downloading {name} to {path}")
    gdown.download(id=MODEL_ZOO[name], output=str(path), quiet=False)
    if not path.exists():
        raise SystemExit(f"download failed; fetch {name} manually into {WEIGHTS_DIR}")
    return path


def load_state_dict(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for k, v in sd.items():
        k = k[7:] if k.startswith("module.") else k
        if k.startswith("classifier."):
            continue  # dataset-specific ID head, not needed for embeddings
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    missing = [m for m in missing if not m.startswith("classifier.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")


class ReID:
    def __init__(self, device=None, weights="osnet_x0_25_msmt17.pt"):
        if device is None:
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        variant = weights.rsplit("_msmt17", 1)[0].rsplit("_market1501", 1)[0]
        self.name = variant
        self.model = CONSTRUCTORS[variant](num_classes=1, pretrained=False)
        load_state_dict(self.model, ensure_weights(weights))
        self.model.eval().to(self.device)
        self.dim = 512
        self.last_ms = 0.0
        with torch.no_grad():  # warm up: the first call on MPS compiles kernels
            for _ in range(3):
                self.model(torch.zeros(2, 3, INPUT_H, INPUT_W, device=self.device))
            if self.device.type == "mps":
                torch.mps.synchronize()

    def _crop(self, frame, xyxy):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - MEAN) / STD
        return crop.transpose(2, 0, 1)  # HWC -> CHW

    @torch.no_grad()
    def embed_many(self, frame, xyxys):
        """N boxes -> (N, 512) float32 L2-normalised rows. Rows for unusable boxes are NaN."""
        xyxys = np.asarray(xyxys, dtype=np.float32).reshape(-1, 4)
        out = np.full((len(xyxys), self.dim), np.nan, dtype=np.float32)
        crops, idx = [], []
        for i, b in enumerate(xyxys):
            c = self._crop(frame, b)
            if c is not None:
                crops.append(c)
                idx.append(i)
        if not crops:
            self.last_ms = 0.0
            return out
        t0 = time.perf_counter()
        x = torch.from_numpy(np.stack(crops)).to(self.device)
        f = self.model(x)
        f = torch.nn.functional.normalize(f, dim=1).cpu().numpy()
        if self.device.type == "mps":
            torch.mps.synchronize()
        self.last_ms = (time.perf_counter() - t0) * 1000
        out[idx] = f
        return out

    def embed(self, frame, xyxy):
        return self.embed_many(frame, [xyxy])[0]

    @staticmethod
    def best_match(gallery, f):
        """Highest cosine similarity between f and any gallery embedding (rows are unit vectors)."""
        g = np.asarray(gallery, dtype=np.float32)
        return float(np.max(g @ (f / (np.linalg.norm(f) + 1e-8))))

    @staticmethod
    def cosine(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ----------------------------------------------------------------------------
# Bench mode
# ----------------------------------------------------------------------------

def _bench(camera, weights):
    from ultralytics import YOLO

    with open(REPO / "config" / "params.yaml") as f:
        params = yaml.safe_load(f)
    det, trk = params["detector"], params.get("tracker", {})
    thresh = float(params["reacquisition"]["reid_similarity_threshold"])
    gallery_max = int(params.get("reid", {}).get("gallery_max", 30))
    tracker_cfg = trk.get("config", "bytetrack.yaml")
    if (REPO / tracker_cfg).exists():
        tracker_cfg = str(REPO / tracker_cfg)

    reid = ReID(weights=weights)
    print(f"re-ID {reid.name} on {reid.device}, threshold {thresh}, gallery max {gallery_max}")
    yolo = YOLO(det.get("weights", "yolo11n.pt"))
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, det["input_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, det["input_height"])
    if not cap.isOpened():
        raise SystemExit(f"camera {camera} did not open")

    enrolled_id = None
    gallery = []             # diverse set of the enrolled person's embeddings
    last_added = -999
    embed_ms = []
    sims_self, sims_other = [], []   # "other" = a track visible in the SAME frame as the enrolled track
    rank_ok, rank_n = 0, 0           # frames where the enrolled track scored highest of all visible
    margins = []                     # self score minus best other, per frame with both present
    n, t_start = 0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = yolo.track(frame, persist=True, tracker=tracker_cfg, classes=[0],
                         conf=float(trk.get("detector_conf", 0.25)),
                         device=str(reid.device), verbose=False)[0]
        boxes, ids = [], []
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().tolist()

        feats = reid.embed_many(frame, boxes) if len(boxes) else np.zeros((0, 512))
        if len(boxes):
            embed_ms.append(reid.last_ms)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("e") and len(boxes):
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            i = int(np.argmax(areas))
            if not np.isnan(feats[i]).any():
                enrolled_id = ids[i]
                gallery = [feats[i].copy()]
                last_added = n
                print(f"enrolled track {ids[i]} at frame {n}")
        if key == ord("q"):
            break

        scores = {}
        for tid, f in zip(ids, feats):
            if gallery and not np.isnan(f).any():
                scores[tid] = ReID.best_match(gallery, f)

        if enrolled_id in scores:
            s_self = scores[enrolled_id]
            sims_self.append(s_self)
            others = [v for k, v in scores.items() if k != enrolled_id]
            if others:
                sims_other.extend(others)
                rank_n += 1
                rank_ok += int(s_self > max(others))
                margins.append(s_self - max(others))
            # grow the gallery slowly with views that are new but still clearly the same person
            if (len(gallery) < gallery_max and n - last_added >= 15
                    and thresh <= s_self < 0.95):
                f_self = feats[ids.index(enrolled_id)]
                gallery.append(f_self.copy())
                last_added = n

        for b, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, b)
            label = f"ID {tid}"
            col = (200, 200, 200)
            if tid in scores:
                label += f"  sim {scores[tid]:.2f}"
                col = (0, 255, 0) if scores[tid] >= thresh else (0, 0, 255)
            if tid == enrolled_id:
                label += "  [enrolled]"
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        n += 1
        hud = f"embed {reid.last_ms:4.1f} ms  {len(boxes)} crop(s)  gallery {len(gallery)}"
        hud += f"  enrolled: track {enrolled_id}" if enrolled_id is not None else "  press e to enroll"
        cv2.putText(frame, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("re-ID bench (e enroll, q quit)", frame)

    cap.release()
    cv2.destroyAllWindows()
    el = time.perf_counter() - t_start
    print(f"model {reid.name}; {n} frames in {el:.1f} s = {n / max(el, 1e-6):.1f} fps end to end")
    if embed_ms:
        e = sorted(embed_ms)
        print(f"re-ID embed ms per frame: median {e[len(e)//2]:.1f}, p95 {e[int(len(e)*0.95)]:.1f}, "
              f"max {e[-1]:.1f}")
    print(f"gallery size at exit: {len(gallery)}")

    def stats(name, xs):
        if not xs:
            print(f"{name}: no samples")
            return
        xs = np.array(xs)
        print(f"{name}: n={len(xs)} min {xs.min():.2f} p5 {np.percentile(xs, 5):.2f} "
              f"median {np.median(xs):.2f} p95 {np.percentile(xs, 95):.2f} max {xs.max():.2f}")
    stats("similarity, enrolled track          ", sims_self)
    stats("similarity, others in the same frame", sims_other)
    if sims_self and sims_other:
        same, other = np.array(sims_self), np.array(sims_other)
        for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
            print(f"  threshold {t:.2f}: keeps {np.mean(same >= t)*100:5.1f}% of self, "
                  f"admits {np.mean(other >= t)*100:5.1f}% of others")
    if rank_n:
        m = np.array(margins)
        print(f"ranking: enrolled track scored highest in {rank_ok}/{rank_n} frames "
              f"({rank_ok / rank_n * 100:.1f}%); margin median {np.median(m):.2f}, p5 {np.percentile(m, 5):.2f}")
        if len(m) >= 10:
            avg = np.convolve(m, np.ones(10) / 10, mode="valid")
            print(f"ranking with 10-frame averaging: correct in {np.mean(avg > 0) * 100:.1f}% of windows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="osnet_x0_25_msmt17.pt",
                    help="one of: " + ", ".join(MODEL_ZOO))
    a = ap.parse_args()
    _bench(a.camera, a.model)
