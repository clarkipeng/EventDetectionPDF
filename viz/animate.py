"""Render scrolling animations of held-out sleep nights from saved model outputs.

Reads the out-of-fold predictions that train.py saves under
experiments/sleep/<model>/<objective>/predictions, the accelerometer signal from
data/processed, and the true events from data/train_events.csv. Each shot is a
PNG sequence rendered at a fixed timestep (frame i shows time t0 + i * speed / fps),
headless via matplotlib's Agg backend.

Picture, top to bottom: the anglez signal as a per-pixel min/max envelope, then the
model's onset (blue) and wakeup (orange) curves. Dashed lines are true events; a dot on
a stem is a predicted event. Curves draw in up to a leading edge at 78% of the width,
and a predicted event pops in when the edge reaches it.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import get_candidates  # noqa: E402
from src.sleep import get_sleep_dataclass  # noqa: E402

BG = "#161a22"
BAND = "#1f2633"  # true sleep window
SIGNAL = "#aeb8c8"
ONSET = "#3987e5"
WAKE = "#e0742f"
W, H, DPI = 2400, 1600, 100
STEP_S = 5  # seconds per sample
PER_HOUR = 3600 // STEP_S
EDGE = 0.78  # leading edge position, fraction of window

# plot geometry in figure fractions (keeps everything >= 8% from every edge)
X0, X1 = 0.08, 0.92
SIG = (0.53, 0.88)  # bottom, top of the signal panel
PRB = (0.12, 0.49)  # bottom, top of the probability panel


class Night:
    def __init__(self, series_id, pred_dir, data_dir, params, score_floor):
        dc = get_sleep_dataclass()
        out = Path(data_dir) / "processed"
        ids = list(np.load(out / "ids.npy", allow_pickle=True))
        b = np.load(out / "bounds.npy")
        j = ids.index(series_id)
        feats = np.load(out / "feats.npy", mmap_mode="r")
        self.anglez = np.asarray(feats[b[j] : b[j + 1], 0], dtype=np.float32)
        self.steps = np.asarray(np.load(out / "step.npy", mmap_mode="r")[b[j] : b[j + 1]])
        n = len(self.anglez)

        pred = np.load(Path(pred_dir) / f"{series_id}.npy")
        pred = np.repeat(pred, 10, 0)[:n].astype(np.float64)
        if len(pred) < n:
            pred = np.pad(pred, ((0, n - len(pred)), (0, 0)))
        # same post-processing the evaluation uses (smooths pred in place)
        (on, off), (on_s, off_s) = get_candidates(dc, pred, "hard", dict(params))
        self.curve = pred
        self.pred_on = on[on_s >= score_floor]
        self.pred_off = off[off_s >= score_floor]

        ev = pd.read_csv(Path(data_dir) / "train_events.csv").dropna(subset=["step"])
        ev = ev[ev.series_id == series_id]
        pos = {s: i for i, s in enumerate(self.steps)}
        self.true_on = np.array([pos[s] for s in ev[ev.event == "onset"].step.astype(int) if s in pos])
        self.true_off = np.array([pos[s] for s in ev[ev.event == "wakeup"].step.astype(int) if s in pos])
        self.windows = list(zip(np.sort(self.true_on), np.sort(self.true_off)))


def envelope(x, i0, i1, bins):
    """min/max of x per bin on an absolute grid, so content doesn't shimmer while scrolling."""
    bw = max(1, (i1 - i0) // bins)
    a = (i0 // bw) * bw
    b = min(len(x), (i1 // bw + 1) * bw)
    a = max(0, a)
    seg = x[a : a + ((b - a) // bw) * bw].reshape(-1, bw)
    centers = a + bw * (np.arange(len(seg)) + 0.5)
    lo, hi = seg.min(1), seg.max(1)
    mid = (lo + hi) / 2
    return centers, np.minimum(lo, mid - 2.5), np.maximum(hi, mid + 2.5)


def render_shot(night, start_h, window_h, speed_h, seconds, fps, outdir, scale, warm_s=0.0):
    outdir.mkdir(parents=True, exist_ok=True)
    nframes = int(round(seconds * fps))
    win = window_h * PER_HOUR
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    # background layer for true sleep windows, spanning both panels
    ax_b = fig.add_axes([X0, PRB[0], X1 - X0, SIG[1] - PRB[0]], zorder=0)
    ax_s = fig.add_axes([X0, SIG[0], X1 - X0, SIG[1] - SIG[0]], zorder=1)
    # true-event lines sit between the signal and the curves
    ax_e = fig.add_axes([X0, PRB[0], X1 - X0, SIG[1] - PRB[0]], zorder=2)
    ax_p = fig.add_axes([X0, PRB[0], X1 - X0, PRB[1] - PRB[0]], zorder=3)
    for a in (ax_b, ax_s, ax_p, ax_e):
        a.set_axis_off()
        a.patch.set_alpha(0)

    pop = 0.25 * fps * speed_h * PER_HOUR / fps  # samples covered in 0.25 s
    for f in range(nframes):
        t = (start_h + (f / fps + warm_s) * speed_h) * PER_HOUR
        i0, i1 = int(t), int(t + win)
        edge = i0 + EDGE * win
        for a in (ax_b, ax_s, ax_p, ax_e):
            a.cla()
            a.set_axis_off()
            a.set_xlim(i0, i1)
        ax_s.set_ylim(-95, 95)
        ax_p.set_ylim(0, 1.08)
        ax_e.set_ylim(0, 1)
        ax_b.set_ylim(0, 1)

        for s, e in night.windows:
            if e > i0 and s < i1:
                ax_b.axvspan(s, e, color=BAND, lw=0, zorder=0)

        c, lo, hi = envelope(night.anglez, max(0, i0 - win // 50), i1 + win // 50, 700)
        ax_s.fill_between(c, lo, hi, color=SIGNAL, lw=0, alpha=0.8, zorder=2)

        lo_i, hi_i = max(0, i0 - 10), min(len(night.curve), int(edge))
        if hi_i > lo_i:
            k = max(1, (hi_i - lo_i) // 1500)
            xs = np.arange(lo_i, hi_i, k)
            for col, color in ((0, ONSET), (1, WAKE)):
                ys = np.clip(night.curve[xs, col] / scale, 0, 1.05)
                ax_p.fill_between(xs, 0, ys, color=color, alpha=0.28, lw=0, zorder=3)
                ax_p.plot(xs, ys, color=color, lw=9, solid_capstyle="round", zorder=4)
                ax_p.scatter([xs[-1]], [ys[-1]], s=420, color=color, zorder=5, edgecolors=BG, linewidths=4)

        for locs, color in ((night.true_on, ONSET), (night.true_off, WAKE)):
            for x in locs:
                if i0 - win * 0.05 < x < i1 + win * 0.05:
                    ax_e.axvline(x, color=color, lw=5, ls=(0, (3, 2.2)), alpha=0.95, zorder=6)

        for locs, col, color in ((night.pred_on, 0, ONSET), (night.pred_off, 1, WAKE)):
            for x in locs:
                if x < edge and i0 - win * 0.05 < x < i1 + win * 0.05:
                    grow = min(1.0, (edge - x) / pop)
                    grow = 1 - (1 - grow) ** 3
                    y = min(1.0, night.curve[x, col] / scale)
                    ax_p.plot([x, x], [0, y], color=color, lw=6, zorder=7)
                    ax_p.scatter([x], [y], s=1500 * grow, color=color, edgecolors=BG, linewidths=6, zorder=8)

        fig.savefig(outdir / f"{f:05d}.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    return nframes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", required=True, help="json list of shots")
    ap.add_argument("--pred_dir", default=str(ROOT / "experiments/sleep/gru/hard/predictions"))
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default='{"cutoff": 0.0, "smooth": 100}')
    ap.add_argument("--score_floor", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    params = json.loads(args.params)
    cache = {}
    for shot in json.load(open(args.shots)):
        sid = shot["series"]
        if sid not in cache:
            cache[sid] = Night(sid, args.pred_dir, args.data_dir, params, args.score_floor)
        n = render_shot(
            cache[sid],
            shot["start_h"],
            shot["window_h"],
            shot["speed_h"],
            shot["seconds"],
            args.fps,
            Path(args.out) / shot["name"],
            shot.get("scale", 1.0),
        )
        print(shot["name"], n, "frames")


if __name__ == "__main__":
    main()
