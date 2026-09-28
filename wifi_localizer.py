#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wi-Fi RSSI Localization System - offline pipeline + data store.

Block diagram implemented by this file:

  Device B (authorized Wi-Fi target / transmitter)
        |  Wi-Fi signal
        v
  Device A (scanner) - RSSI / channel / timestamp collection
        v
  Signal Filter        (median + EMA + outlier rejection)      [Layer 1]
        v
  Feature Extraction   (RSSI statistics, channel, temporal)     [Layer 2]
        v
  Localization Engine
    1. Fingerprint matching       (kNN vs calibration database)
    2. Signal propagation model    (log-distance path loss -> range rings)
    3. Temporal tracking           (Kalman filter, constant velocity)
    4. Confidence estimation
        v
  Position Estimator -> X, Y, error radius, confidence
        v
  STORAGE  ->  data/*.json + data/*.csv

Data storage: every run still writes JSON/CSV into ./data/ — the
combined single-file UI (index.html) can load those offline through
its file picker, and a built-in JS demo runs with no Python at all.

Connection: `serve` additionally starts a tiny local engine on
http://127.0.0.1:8765 that index.html auto-connects to. The page is
served by it (or opened from file:// and cross-connects), and
POST /api/scan pushes each scan burst through the real pipeline.

Usage:
  python wifi_localizer.py demo                  # end-to-end synthetic run -> data/
  python wifi_localizer.py serve                  # local engine the HTML UI connects to
  python wifi_localizer.py calibrate mycal.json  # store a calibration fingerprint DB
  python wifi_localizer.py track mylog.csv       # process an RSSI log -> data/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Layer 1 - SIGNAL FILTER: raw RSSI -> median -> EMA -> outlier rejection
# ---------------------------------------------------------------------------
class SignalFilter:
    """Streaming filter: sliding-window median, then EMA, with MAD-based
    outlier rejection so a single spiked sample cannot yank the output."""

    def __init__(self, window: int = 7, alpha: float = 0.35, z: float = 3.0):
        self.window = window
        self.alpha = alpha
        self.z = z
        self.buf: list[float] = []
        self.ema: float | None = None

    def reset(self) -> None:
        self.buf = []
        self.ema = None

    def step(self, rssi: float) -> float:
        self.buf.append(rssi)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        med = statistics.median(self.buf)
        if self.ema is None:
            self.ema = med
            return self.ema
        mad = statistics.median(abs(v - med) for v in self.buf)
        sigma = 1.4826 * mad
        if sigma > 1e-6 and abs(med - self.ema) > self.z * sigma:
            # outlier rejected: absorb only 5% of the jump
            self.ema += 0.05 * (med - self.ema)
        else:
            self.ema = self.alpha * med + (1.0 - self.alpha) * self.ema
        return self.ema

    def filter_batch(self, samples: list[float]) -> list[float]:
        self.reset()
        return [round(self.step(s), 2) for s in samples]


# ---------------------------------------------------------------------------
# Layer 2 - FEATURE EXTRACTION: RSSI statistics / channel / temporal variation
# ---------------------------------------------------------------------------
class FeatureExtractor:
    @staticmethod
    def extract(rssis: list[float], channel: int | None = None) -> dict:
        n = len(rssis)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "slope": 0.0,
                    "temporal_variation": 0.0, "channel": channel, "n": 0}
        mean = statistics.fmean(rssis)
        std = statistics.pstdev(rssis) if n > 1 else 0.0
        half = n // 2
        if half > 0 and n - half > 0:
            slope = statistics.fmean(rssis[half:]) - statistics.fmean(rssis[:half])
        else:
            slope = 0.0
        return {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "slope": round(slope, 2),              # temporal trend of the burst
            "temporal_variation": round(std, 2),   # short-term fading measure
            "channel": channel,
            "n": n,
        }


# ---------------------------------------------------------------------------
# Localization 2/4 pieces - SIGNAL PROPAGATION MODEL (log-distance path loss)
# ---------------------------------------------------------------------------
class PropagationModel:
    """RSSI(d) = A - 10*n*log10(d/d0)   (A = RSSI at 1 m, n = path-loss exponent)"""

    def __init__(self, a: float = -32.0, n: float = 2.6, d0: float = 1.0):
        self.a = a
        self.n = n
        self.d0 = d0

    def rssi(self, d: float) -> float:
        d = max(d, 0.5)
        return self.a - 10.0 * self.n * math.log10(d / self.d0)

    def distance(self, rssi: float) -> float:
        return self.d0 * 10.0 ** ((self.a - rssi) / (10.0 * self.n))

    def fit(self, pairs: list[tuple[float, float]]) -> None:
        """Least-squares fit of (A, n) from (distance, rssi) calibration pairs."""
        if len(pairs) < 3:
            return
        xs = [math.log10(max(d, 0.5) / self.d0) for d, _ in pairs]
        ys = [r for _, r in pairs]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx < 1e-9:
            return
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        self.n = min(4.0, max(1.5, -b / 10.0))
        self.a = my - b * mx


# ---------------------------------------------------------------------------
# Localization 1 - FINGERPRINT DATABASE (calibration map + kNN matching)
# ---------------------------------------------------------------------------
class FingerprintDatabase:
    """Calibration entries: 'transmitter at position P produces RSSI vector V
    when Device A sits at each of the M scan points.' Matching a (possibly
    partial) observed vector against the DB yields candidate locations."""

    def __init__(self, entries: list | None = None, scan_points: list | None = None):
        self.entries: list[dict] = entries or []
        self.scan_points: list[dict] = scan_points or []

    def add(self, position: dict, vector: list[float]) -> None:
        self.entries.append({
            "position": {"x": round(position["x"], 2), "y": round(position["y"], 2)},
            "vector": [round(v, 2) for v in vector],
        })

    def match(self, observed: dict[int, float], k: int = 6) -> dict | None:
        """kNN over the observed subset of the RSSI vector.
        observed: {scan_index: filtered_rssi}. Returns weighted-mean candidate
        position, spread of the top-k, and best RMS match."""
        if not self.entries or not observed:
            return None
        idxs = sorted(i for i in observed if i >= 0)
        if not idxs:
            return None
        need = max(idxs) + 1
        scored = []
        for e in self.entries:
            v = e["vector"]
            if len(v) < need:
                continue
            se = sum((observed[i] - v[i]) ** 2 for i in idxs)
            scored.append((math.sqrt(se / len(idxs)), e["position"]))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0])
        top = scored[:k]
        wsum = cx = cy = 0.0
        for s, p in top:
            w = 1.0 / ((s + 2.0) ** 2)  # squared inverse match error -> sharp weighting
            wsum += w
            cx += w * p["x"]
            cy += w * p["y"]
        cx, cy = cx / wsum, cy / wsum
        spread = statistics.fmean(
            math.hypot(p["x"] - cx, p["y"] - cy) for _, p in top)
        return {
            "x": round(cx, 2), "y": round(cy, 2),
            "spread": round(spread, 2), "rms": round(top[0][0], 2),
            "candidates": [p for _, p in top],
        }

    # -- storage ------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "scan_points": self.scan_points, "entries": self.entries,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "FingerprintDatabase":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(raw.get("entries", []), raw.get("scan_points", []))


# ---------------------------------------------------------------------------
# Localization 3 - range-ring solver (fuses path-loss constraints)
# ---------------------------------------------------------------------------
def solve_rings(constraints: list[tuple[float, float, float]],
                init_points: list[tuple[float, float]] | None = None,
                iters: int = 120, huber: float = 1.2) -> dict:
    """Robust multi-start solver for  sum_i w_i * (|p - s_i| - d_i)^2.

    v3 precision upgrades:
      * Huber-style iterative reweighting down-weights bad rings (deep
        shadowing / multipath) instead of letting them drag the solution;
      * multiple start points (previous fix, fingerprint, centroid
        heuristics) avoid gradient-descent local minima;
      * 120 decaying-normalized-gradient iterations, best robust cost wins.
    """
    if not constraints:
        return {"x": 0.0, "y": 0.0, "residual_rms": 99.0}
    n = len(constraints)
    base = [max(0.1, 0.82 ** (n - 1 - i)) for i in range(n)]  # recency weights
    ax = statistics.fmean(c[0] for c in constraints)
    ay = statistics.fmean(c[1] for c in constraints)

    starts = [tuple(p) for p in (init_points or []) if p]
    s0 = constraints[0]
    dx, dy = ax - s0[0], ay - s0[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        dx, dy, norm = 1.0, 0.0, 1.0
    starts.append((s0[0] + s0[2] * dx / norm, s0[1] + s0[2] * dy / norm))
    starts.append((ax, ay))

    best: tuple[float, float, float] | None = None
    for px, py in starts:
        for it in range(iters):
            gx = gy = 0.0
            for (sx, sy, d), w in zip(constraints, base):
                distc = math.hypot(px - sx, py - sy)
                if distc < 0.25:
                    distc = 0.25
                r = distc - d
                hw = 1.0 if abs(r) <= huber else huber / abs(r)
                g = 2.0 * w * hw * r
                gx += g * (px - sx) / distc
                gy += g * (py - sy) / distc
            gnorm = math.hypot(gx, gy)
            if gnorm < 1e-9:
                break
            step = min(0.8, 0.6 * (0.97 ** it))
            px -= step * gx / gnorm
            py -= step * gy / gnorm
        cost = 0.0
        for (sx, sy, d), w in zip(constraints, base):
            r = abs(math.hypot(px - sx, py - sy) - d)
            cost += w * (r * r if r <= huber else 2.0 * huber * r - huber * huber)
        if best is None or cost < best[0]:
            best = (cost, px, py)
    _, px, py = best
    se = wsum = 0.0
    for (sx, sy, d), w in zip(constraints, base):
        se += w * (math.hypot(px - sx, py - sy) - d) ** 2
        wsum += w
    return {"x": round(px, 2), "y": round(py, 2),
            "residual_rms": round(math.sqrt(se / wsum), 2)}


# ---------------------------------------------------------------------------
# Localization 3 - TEMPORAL TRACKING: 2-D constant-velocity Kalman filter
# ---------------------------------------------------------------------------
def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mat_mul(A: list, B: list) -> list:
    inner, cols = len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(cols)]
            for i in range(len(A))]


def _mat_vec(A: list, v: list) -> list:
    return [sum(a * b for a, b in zip(row, v)) for row in A]


def _transpose(A: list) -> list:
    return [list(col) for col in zip(*A)]


def _mat_add(A: list, B: list) -> list:
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def _mat_sub(A: list, B: list) -> list:
    return [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


class KalmanTracker:
    """State [x, y, vx, vy]; measurement = fused position fix.
    Prevents the marker jumping on every RSSI change."""

    def __init__(self, dt: float = 1.0, q: float = 0.03, r: float = 1.0):
        self.dt, self.q, self.r = dt, q, r
        self.x = [0.0, 0.0, 0.0, 0.0]
        self.P = [[25.0, 0, 0, 0], [0, 25.0, 0, 0],
                  [0, 0, 9.0, 0], [0, 0, 0, 9.0]]

    def update(self, zx: float, zy: float) -> float:
        dt = self.dt
        F = [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.x = _mat_vec(F, self.x)
        self.P = _mat_add(_mat_mul(_mat_mul(F, self.P), _transpose(F)),
                          [[self.q * dt * dt, 0, 0, 0], [0, self.q * dt * dt, 0, 0],
                           [0, 0, self.q, 0], [0, 0, 0, self.q]])
        H = [[1, 0, 0, 0], [0, 1, 0, 0]]
        PHt = _mat_mul(self.P, _transpose(H))
        S = _mat_add(_mat_mul(H, PHt), [[self.r, 0], [0, self.r]])
        det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
        Sinv = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]]
        K = _mat_mul(PHt, Sinv)
        y = [zx - self.x[0], zy - self.x[1]]
        Ky = _mat_vec(K, y)
        self.x = [xi + ki for xi, ki in zip(self.x, Ky)]
        self.P = _mat_mul(_mat_sub(_identity(4), _mat_mul(K, H)), self.P)
        return math.hypot(*y)  # innovation

    def position(self) -> tuple[float, float]:
        return self.x[0], self.x[1]

    def position_sigma(self) -> float:
        return math.sqrt(self.P[0][0] + self.P[1][1])


# ---------------------------------------------------------------------------
# Localization 4 - CONFIDENCE ESTIMATION
# ---------------------------------------------------------------------------
class ConfidenceEstimator:
    @staticmethod
    def estimate(residual_rms: float, fp_spread: float | None,
                 innovation: float, n_constraints: int, coverage: float) -> float:
        c_ring = math.exp(-0.18 * max(0.0, residual_rms - 0.5))
        c_fp = 0.5 if fp_spread is None else math.exp(-0.06 * max(0.0, fp_spread - 1.0))
        c_inn = math.exp(-0.12 * max(0.0, innovation - 1.0))
        c_n = min(1.0, n_constraints / 6.0)
        c_cov = min(1.0, coverage)
        conf = 0.35 * c_ring + 0.25 * c_fp + 0.15 * c_inn + 0.15 * c_n + 0.10 * c_cov
        return round(max(0.02, min(0.99, conf)), 3)


# ---------------------------------------------------------------------------
# THE LOCALIZATION ENGINE - glues every layer together
# ---------------------------------------------------------------------------
class WifiLocalizer:
    def __init__(self, prop: PropagationModel, db: FingerprintDatabase,
                 floor: dict, k: int = 6):
        self.prop = prop
        self.db = db
        self.floor = floor
        self.k = k
        self.floor_center = (floor["width"] / 2.0, floor["height"] / 2.0)
        self.n_scan_points = max(1, len(db.scan_points))
        self.reset()

    def reset(self) -> None:
        self.constraints: list[tuple[float, float, float]] = []
        self.observed: dict[int, float] = {}
        self.kalman = KalmanTracker()
        self.last_fused: dict | None = None

    def process_step(self, step: dict) -> dict:
        """step = {t, anchor, scanner{x,y}, samples[raw rssi...], channel}"""
        anchor = step["anchor"]
        sf = SignalFilter()
        filtered = sf.filter_batch(step["samples"])                 # Layer 1
        feats = FeatureExtractor.extract(filtered, step.get("channel"))  # Layer 2

        observed_rssi = statistics.fmean(filtered)
        self.observed[anchor] = observed_rssi

        # --- 2. signal propagation model: one range ring per scan ---------
        d = self.prop.distance(observed_rssi)
        self.constraints.append((step["scanner"]["x"], step["scanner"]["y"], d))

        # --- 1. fingerprint matching on the partial RSSI vector -----------
        fp = self.db.match(self.observed, self.k)

        # --- fuse the rings (multi-start robust solve) --------------------
        starts = []
        if self.last_fused:
            starts.append((self.last_fused["x"], self.last_fused["y"]))
        if fp:
            starts.append((fp["x"], fp["y"]))
        rings = solve_rings(self.constraints, starts)
        self.last_fused = {"x": rings["x"], "y": rings["y"]}

        # --- 3. temporal tracking ------------------------------------------
        innovation = self.kalman.update(rings["x"], rings["y"])
        ex, ey = self.kalman.position()
        err_radius = round(min(12.0, self.kalman.position_sigma()
                               + rings["residual_rms"]), 2)

        # --- 4. confidence --------------------------------------------------
        coverage = len(self.observed) / self.n_scan_points
        conf = ConfidenceEstimator.estimate(
            rings["residual_rms"], (fp or {}).get("spread"),
            innovation, len(self.constraints), coverage)

        return {
            "t": step["t"], "anchor": step["anchor"], "scanner": step["scanner"],
            "channel": step.get("channel"),
            "raw_rssi": step["samples"],
            "filtered_series": filtered,
            "filtered_rssi": round(observed_rssi, 2),
            "features": feats,
            "pathloss_distance": round(d, 2),
            "fingerprint": fp,
            "fused_measurement": {"x": rings["x"], "y": rings["y"],
                                   "residual_rms": rings["residual_rms"]},
            "estimate": {"x": round(ex, 2), "y": round(ey, 2),
                         "error_radius": err_radius, "confidence": conf},
        }


# ---------------------------------------------------------------------------
# Synthetic demo scenario (two devices, one floor)
# ---------------------------------------------------------------------------
FLOOR = {"width": 20.0, "height": 15.0}
SCAN_POINTS = [  # Device A reference positions (metres) - 10 for better geometry
    {"x": 2.0, "y": 2.0}, {"x": 10.0, "y": 1.8}, {"x": 18.0, "y": 2.0},
    {"x": 18.2, "y": 7.5}, {"x": 18.0, "y": 13.0}, {"x": 10.0, "y": 13.2},
    {"x": 2.0, "y": 13.0}, {"x": 1.8, "y": 7.5}, {"x": 10.0, "y": 7.5},
    {"x": 5.0, "y": 7.5},
]
GRID_X = [3.0, 6.5, 10.0, 13.5, 17.0]   # finer calibration grid (5x4 = 20 points)
GRID_Y = [3.5, 6.0, 8.5, 11.0]
TRUE_TX = {"x": 13.2, "y": 9.0}   # hidden transmitter position for the demo


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def rolling_accuracy(steps: list[dict], threshold: float = 2.5,
                     warmup: int | None = None) -> float | None:
    """Indoor accuracy metric: share of post-warm-up estimates whose
    distance error is within `threshold` metres of the true position.
    Warm-up = one full sweep of the scan points (before Device A has
    visited every reference point the geometry is incomplete)."""
    if warmup is None:
        warmup = len(SCAN_POINTS)
    vals = [s["distance_error"] for s in steps[warmup:] if "distance_error" in s]
    if not vals:
        return None
    return round(100.0 * sum(1 for v in vals if v <= threshold) / len(vals), 1)


def cmd_demo(seed: int) -> None:
    rng = random.Random(seed)
    prop_true = PropagationModel(a=-32.0, n=2.6)

    # ---- calibration walk: build the fingerprint database ----------------
    db, cal_positions = build_synthetic_calibration(rng)

    # ---- fit the propagation model from the calibration data -------------
    prop_fit = fit_propagation(db)

    # ---- tracking walk: Device A visits the scan points (twice) ----------
    order = list(range(len(SCAN_POINTS))) + list(range(len(SCAN_POINTS) - 1, -1, -1))
    steps, t = [], 0.0
    for anchor in order:
        sp = SCAN_POINTS[anchor]
        d = _dist(TRUE_TX["x"], TRUE_TX["y"], sp["x"], sp["y"])
        samples = []
        for _ in range(12):  # 12 samples per burst -> less filtered-RSSI noise
            r = prop_true.rssi(d) + rng.gauss(0, 3.0)
            if rng.random() < 0.12:  # occasional spike outlier
                r += rng.choice([-1, 1]) * rng.uniform(8, 15)
            samples.append(round(r, 1))
        steps.append({"t": round(t, 1), "anchor": anchor, "scanner": sp,
                      "samples": samples, "channel": [1, 6, 11][anchor % 3]})
        t += 2.0

    # ---- run the pipeline --------------------------------------------------
    loc = WifiLocalizer(prop_fit, db, FLOOR)
    out_steps = []
    for s in steps:
        rec = loc.process_step(s)
        rec["true_position"] = TRUE_TX
        rec["distance_error"] = round(_dist(rec["estimate"]["x"], rec["estimate"]["y"],
                                           TRUE_TX["x"], TRUE_TX["y"]), 2)
        out_steps.append(rec)
        rec["accuracy"] = rolling_accuracy(out_steps)

    session = {
        "meta": {
            "mode": "demo", "seed": seed,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "floor": FLOOR, "transmitter_true": TRUE_TX,
            "scan_points": SCAN_POINTS,
            "calibration_positions": cal_positions,
            "propagation_fit": {"a": round(prop_fit.a, 2), "n": round(prop_fit.n, 2)},
            "indoor_accuracy": rolling_accuracy(out_steps),
            "note": ("Stored by wifi_localizer.py. No server/host is involved: "
                     "open index.html and load this file with the file picker."),
        },
        "steps": out_steps,
    }

    DATA_DIR.mkdir(exist_ok=True)
    db.save(DATA_DIR / "calibration_db.json")
    (DATA_DIR / "tracking_session.json").write_text(json.dumps(session, indent=2))
    with open(DATA_DIR / "rssi_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "anchor", "scanner_x", "scanner_y", "channel", "rssi"])
        for rec in out_steps:
            for r in rec["raw_rssi"]:
                w.writerow([rec["t"], rec["anchor"], rec["scanner"]["x"],
                            rec["scanner"]["y"], rec["channel"], r])

    final = out_steps[-1]
    avg_conf = statistics.fmean(o["estimate"]["confidence"] for o in out_steps)
    print("Demo session stored in", DATA_DIR)
    print(f"  steps                : {len(out_steps)}")
    print(f"  final estimate       : X={final['estimate']['x']} m, "
          f"Y={final['estimate']['y']} m")
    print(f"  true position        : X={TRUE_TX['x']} m, Y={TRUE_TX['y']} m")
    print(f"  final distance error : {final['distance_error']} m")
    print(f"  final confidence     : {final['estimate']['confidence']}")
    print(f"  mean confidence      : {avg_conf:.3f}")
    print(f"  indoor accuracy      : {rolling_accuracy(out_steps)}% of steady-state steps within 2.5 m")
    print("Next: open index.html in a browser and load "
          "data/tracking_session.json with the file picker.")


# ---------------------------------------------------------------------------
# Shared helpers + the live engine that the combined HTML UI connects to
# ---------------------------------------------------------------------------
def build_synthetic_calibration(rng: random.Random) -> tuple[FingerprintDatabase, list]:
    """Device A walks the scan points with Device B at every calibration
    grid position; each entry maps a transmitter position to its RSSI
    vector (one value per scan point)."""
    prop_true = PropagationModel(a=-32.0, n=2.6)
    db = FingerprintDatabase(scan_points=SCAN_POINTS)
    cal_positions = []
    for gx in GRID_X:
        for gy in GRID_Y:
            pos = {"x": gx, "y": gy}
            cal_positions.append(pos)
            vector = [prop_true.rssi(_dist(gx, gy, sp["x"], sp["y"])) + rng.gauss(0, 1.2)
                      for sp in SCAN_POINTS]
            db.add(pos, vector)
    return db, cal_positions


def fit_propagation(db: FingerprintDatabase) -> PropagationModel:
    """Least-squares fit of the path-loss model from the calibration DB."""
    pairs = []
    for e in db.entries:
        for sp, v in zip(db.scan_points, e["vector"]):
            pairs.append((_dist(e["position"]["x"], e["position"]["y"],
                               sp["x"], sp["y"]), v))
    prop = PropagationModel()
    prop.fit(pairs)
    return prop


class LiveEngine:
    """In-memory localization engine behind the `serve` API. Each
    POST /api/scan simulates one Device A measurement burst at a scan
    point and runs it through the full pipeline, returning the estimate."""

    def __init__(self):
        stored = FingerprintDatabase.load(DATA_DIR / "calibration_db.json")
        if stored.entries and stored.scan_points and len(stored.scan_points) == len(SCAN_POINTS):
            self.db = stored
            self.cal_positions = [e["position"] for e in stored.entries]
        else:
            self.db, self.cal_positions = build_synthetic_calibration(random.Random(7))
        self.prop = fit_propagation(self.db)
        self.prop_true = PropagationModel(a=-32.0, n=2.6)
        self.reset()

    def reset(self) -> None:
        """New hidden transmitter position + fresh localizer state."""
        self.rng = random.Random()
        self.true = {
            "x": round(self.rng.uniform(4.0, FLOOR["width"] - 4.0), 1),
            "y": round(self.rng.uniform(4.0, FLOOR["height"] - 4.0), 1),
        }
        self.localizer = WifiLocalizer(self.prop, self.db, FLOOR)
        self.steps: list[dict] = []
        self.t = 0.0

    def meta(self) -> dict:
        return {
            "mode": "live", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "floor": FLOOR, "scan_points": SCAN_POINTS,
            "calibration_positions": self.cal_positions,
            "propagation_fit": {"a": round(self.prop.a, 2), "n": round(self.prop.n, 2)},
            "note": "Live session driven by wifi_localizer.py serve.",
        }

    def scan(self, anchor: int) -> dict:
        anchor = int(anchor)
        if not 0 <= anchor < len(SCAN_POINTS):
            raise ValueError(f"anchor must be 0..{len(SCAN_POINTS) - 1}")
        sp = SCAN_POINTS[anchor]
        d = _dist(self.true["x"], self.true["y"], sp["x"], sp["y"])
        samples = []
        for _ in range(12):  # 12 samples per burst -> less filtered-RSSI noise
            r = self.prop_true.rssi(d) + self.rng.gauss(0, 3.0)
            if self.rng.random() < 0.12:  # occasional spike outlier
                r += self.rng.choice([-1, 1]) * self.rng.uniform(8, 15)
            samples.append(round(r, 1))
        step = {"t": round(self.t, 1), "anchor": anchor, "scanner": sp,
                "samples": samples, "channel": [1, 6, 11][anchor % 3]}
        self.t += 2.0
        rec = self.localizer.process_step(step)
        rec["true_position"] = self.true
        rec["distance_error"] = round(_dist(rec["estimate"]["x"], rec["estimate"]["y"],
                                           self.true["x"], self.true["y"]), 2)
        self.steps.append(rec)
        rec["accuracy"] = rolling_accuracy(self.steps)
        return rec


def cmd_serve(host: str, port: int) -> None:
    """Tiny stdlib-only HTTP server: serves index.html and exposes the
    localization engine at /api/* so the combined UI can connect to it."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    engine = LiveEngine()
    index_html = Path(__file__).resolve().parent / "index.html"

    class Handler(BaseHTTPRequestHandler):
        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, text: str) -> None:
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # CORS preflight
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/api/status":
                self._json({"connected": True, "mode": "live",
                            "floor": FLOOR, "scan_points": SCAN_POINTS,
                            "calibration_positions": engine.cal_positions,
                            "live_steps": len(engine.steps)})
            elif path == "/api/session":
                f = DATA_DIR / "tracking_session.json"
                if f.exists():
                    self._json(json.loads(f.read_text()))
                else:
                    self._json({"meta": engine.meta(), "steps": []})
            elif path in ("/", "/index.html"):
                if index_html.exists():
                    self._html(index_html.read_text())
                else:
                    self._html("<h1>index.html not found next to wifi_localizer.py</h1>")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            path = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            if path == "/api/reset":
                engine.reset()
                return self._json(engine.meta())
            if path == "/api/scan":
                try:
                    return self._json(engine.scan(payload.get("anchor", 0)))
                except (ValueError, TypeError) as exc:
                    return self._json({"error": str(exc)}, 400)
            self._json({"error": "unknown endpoint"}, 404)

        def log_message(self, fmt, *args) -> None:
            pass  # keep the console quiet

    srv = HTTPServer((host, port), Handler)
    print(f"Wi-Fi localization engine listening on http://{host}:{port}")
    print("  - open http://" + host + ":" + str(port) + "/ in a browser, or")
    print("  - open index.html directly (file://): it auto-connects to this engine.")
    print("  Endpoints: GET /api/status, GET /api/session, POST /api/reset, POST /api/scan")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


# ---------------------------------------------------------------------------
# CLI: calibrate / track against real measurement files
# ---------------------------------------------------------------------------
def cmd_calibrate(path: str) -> None:
    """Input JSON: {"scan_points":[{"x":..,"y":..},..],
                     "fingerprints":[{"position":{"x":..,"y":..},"vector":[..]}]}"""
    raw = json.loads(Path(path).read_text())
    db = FingerprintDatabase(scan_points=raw.get("scan_points", []))
    for fp in raw.get("fingerprints", []):
        db.add(fp["position"], fp["vector"])
    DATA_DIR.mkdir(exist_ok=True)
    db.save(DATA_DIR / "calibration_db.json")
    print(f"Stored {len(db.entries)} fingerprints "
          f"({len(db.scan_points)} scan points) in {DATA_DIR / 'calibration_db.json'}")


def cmd_track(path: str) -> None:
    """Input CSV columns: t,anchor,scanner_x,scanner_y,channel,rssi
    (one row per raw RSSI sample; consecutive rows with the same anchor form
    one measurement burst). Uses the stored calibration DB when present."""
    db = FingerprintDatabase.load(DATA_DIR / "calibration_db.json")
    if not db.entries:
        print("No calibration DB found - run `calibrate` or `demo` first.")
        return

    pairs = []
    for e in db.entries:
        for sp, v in zip(db.scan_points, e["vector"]):
            pairs.append((_dist(e["position"]["x"], e["position"]["y"],
                               sp["x"], sp["y"]), v))
    prop = PropagationModel()
    prop.fit(pairs)

    steps, rows = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    cur = None
    for row in rows:
        anchor = int(float(row["anchor"]))
        if cur is None or cur["anchor"] != anchor:
            cur = {"t": float(row["t"]), "anchor": anchor,
                   "scanner": {"x": float(row["scanner_x"]), "y": float(row["scanner_y"])},
                   "channel": int(float(row.get("channel") or 0)) or None,
                   "samples": []}
            steps.append(cur)
        cur["samples"].append(float(row["rssi"]))

    loc = WifiLocalizer(prop, db, {"width": 25.0, "height": 20.0})
    out = [loc.process_step(s) for s in steps]
    session = {
        "meta": {"mode": "track", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "floor": {"width": 25.0, "height": 20.0},
                 "scan_points": db.scan_points,
                 "calibration_positions": [e["position"] for e in db.entries],
                 "propagation_fit": {"a": round(prop.a, 2), "n": round(prop.n, 2)},
                 "note": "Stored by wifi_localizer.py; load into index.html via file picker."},
        "steps": out,
    }
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "tracking_session.json").write_text(json.dumps(session, indent=2))
    print(f"Stored {len(out)} estimates in {DATA_DIR / 'tracking_session.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Wi-Fi RSSI localization (offline, storage only).")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run the synthetic end-to-end demo and store results")
    d.add_argument("--seed", type=int, default=7)
    sv = sub.add_parser("serve", help="run the local engine the HTML UI connects to")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sub.add_parser("calibrate", help="store a calibration fingerprint DB").add_argument(
        "input", help="calibration JSON file")
    sub.add_parser("track", help="process an RSSI log CSV").add_argument(
        "log", help="RSSI log CSV file")
    args = p.parse_args()

    if args.cmd == "demo":
        cmd_demo(args.seed)
    elif args.cmd == "serve":
        cmd_serve(args.host, args.port)
    elif args.cmd == "calibrate":
        cmd_calibrate(args.input)
    elif args.cmd == "track":
        cmd_track(args.log)


if __name__ == "__main__":
    main()
