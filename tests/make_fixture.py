"""
Synthetic neuro-align output with known ground truth, for the tests and for `review.generate`.

  sub-enc01_ses-track   400 s on a 200 cm linear track (25 ms bins): 30 place cells (Gaussian fields,
                        10 of them firing in one running direction only), 8 speed cells (rate rises
                        with running speed), 12 untuned cells.
  sub-enc02_ses-arena   400 s in a 1 m open field: 10 2D place cells, 10 head-direction cells (von Mises
                        tuning), 10 untuned cells.
  sub-enc03_ses-vr      400 s of calcium-like ROI activity (15.5 Hz) on a one-way 400 cm virtual track:
                        15 place ROIs whose activity sits on an offset outside the field, 10 untuned.

Written exactly as neuro-align writes them: neuro-align/<session_id>/{aligned.npz, spikes.npz,
manifest.json, events.parquet}. Layouts: linked/aligned/neuro-align/... (a linked port) and
flat/neuro-align/...; ground_truth.json next to them.

    python -m tests.make_fixture OUT [--layout linked|flat|both]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

BIN = 0.025
DUR = 400.0


def smooth_walk(n, rng, sigma, lo, hi, step):
    v = np.zeros(n)
    x = (lo + hi) / 2
    vel = 0.0
    for i in range(n):
        vel = 0.98 * vel + rng.normal(0, step)
        x += vel
        if x < lo or x > hi:
            vel = -vel
            x = np.clip(x, lo, hi)
        v[i] = x
    return v


def track(rng):
    t = np.arange(0, DUR, BIN)
    n = len(t)
    pos = np.zeros(n)
    x, d, i = 0.0, 1, 0
    while i < n:
        pause = int(rng.uniform(1, 2.5) / BIN)
        pos[i:i + pause] = x
        i += pause
        v = rng.uniform(20, 50)
        m = int(200 / v / BIN)
        seg = x + d * np.linspace(0, 200, m)
        k = min(m, n - i)
        if k <= 0:
            break
        pos[i:i + k] = seg[:k]
        i += k
        x = seg[k - 1]
        d = -d
    vel = np.gradient(pos, BIN)
    speed = np.abs(vel)
    rd = np.where(speed >= 3, np.sign(vel), 0.0)
    return t, pos, speed, rd


def write_session(root: Path, sid: str, t, X, spikes, names, kinds, units, meta_regions, extra_manifest,
                  neural_kind: str = "units", bin_size: float = BIN):
    d = root / "neuro-align" / sid
    d.mkdir(parents=True, exist_ok=True)
    N = X.shape[1]
    ids = np.array([str(i) for i in range(N)], dtype="U64")
    meta_dt = np.dtype([("id", "U64"), ("region", "U64"), ("label", "U32"), ("cell_type", "U32"), ("depth", "f8"),
                        ("x", "f8"), ("y", "f8"), ("rate_hz", "f8")])
    meta = np.array([(ids[j], meta_regions[j], "good", "", float(j), np.nan, np.nan, X[:, j].sum() / DUR)
                     for j in range(N)], dtype=meta_dt)
    Y = np.stack([v for v in extra_manifest.pop("_Y")], axis=1).astype(np.float32)
    np.savez_compressed(d / "aligned.npz", t=t, X=X.astype(np.float32), Y=Y, neuron_ids=ids, neuron_meta=meta,
                        behavior_names=np.array(names, dtype="U64"), behavior_units=np.array(units, dtype="U64"),
                        behavior_kinds=np.array(kinds, dtype="U32"), epoch_mask=np.ones(len(t), bool),
                        bin_size=np.float64(bin_size), session_id=np.array(sid), neural_kind=np.array(neural_kind))
    if spikes is not None:
        lens = np.array([len(s) for s in spikes])
        np.savez_compressed(d / "spikes.npz", times=np.concatenate(spikes), offsets=np.r_[0, np.cumsum(lens)],
                            neuron_ids=ids)
    manifest = {"session_id": sid, "app": "neuro-align", "bin_size_s": bin_size, "n_bins": len(t),
                "neural": {"kind": neural_kind, "n_neurons": N},
                "behavior": {"variables": [{"name": n, "kind": k, "unit": u} for n, k, u in zip(names, kinds, units)]},
                "epoch": {"name": "all"}, "sheet": {"subject_id": sid.split("_")[0][4:], "condition": "synthetic"}}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))


def spikes_from_rate(t, rate, rng):
    k = rng.poisson(rate * BIN)
    s = np.repeat(t, k) + rng.uniform(0, BIN, k.sum())
    return np.sort(s), k


def make(out: Path, layout: str = "both", seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    src = out / "_src"
    gt = {}

    # --- linear track
    t, pos, speed, rd = track(rng)
    rates, gtt = [], {"place": {}, "place_direction": {}, "speed": [], "untuned": []}
    for j, c in enumerate(np.linspace(15, 185, 30)):
        gate = (speed > 3) & ((rd > 0) if j % 3 == 0 else (rd < 0) if j % 3 == 1 else True)
        if j >= 10:
            gate = speed > 3
        rates.append(0.1 + 12 * np.exp(-0.5 * ((pos - c) / 10) ** 2) * gate)
        gtt["place"][str(j)] = float(c)
        gtt["place_direction"][str(j)] = ("fwd" if j % 3 == 0 else "bwd" if j % 3 == 1 else "both") if j < 10 else "both"
    for j in range(30, 38):
        rates.append(0.5 + 0.25 * speed)
        gtt["speed"].append(str(j))
    for j in range(38, 50):
        rates.append(np.full(len(t), 2.0))
        gtt["untuned"].append(str(j))
    sp, cols = zip(*(spikes_from_rate(t, r, rng) for r in rates))
    X = np.stack(cols, axis=1)
    regions = ["CA1"] * 30 + ["MEC"] * 8 + ["CA1"] * 6 + ["MEC"] * 6
    write_session(src, "sub-enc01_ses-track", t + BIN / 2, X, list(sp), ["position", "speed", "run_direction"],
                  ["position", "speed", "direction"], ["cm", "cm/s", "sign"], regions,
                  {"_Y": [pos, speed, rd]})
    gt["sub-enc01_ses-track"] = gtt

    # --- open field with head direction
    n = len(t)
    x = smooth_walk(n, rng, 0, 0, 100, 0.08)
    y = smooth_walk(n, rng, 0, 0, 100, 0.08)
    hd = np.mod(np.cumsum(rng.normal(0, 0.05, n)), 2 * np.pi)
    sp2 = np.hypot(np.gradient(x, BIN), np.gradient(y, BIN))
    rates, gta = [], {"place2d": {}, "hd": {}, "untuned": []}
    centres = rng.uniform(20, 80, (10, 2))
    for j, (cx, cy) in enumerate(centres):
        rates.append(0.1 + 10 * np.exp(-0.5 * (((x - cx) ** 2 + (y - cy) ** 2) / 12 ** 2)))
        gta["place2d"][str(j)] = [float(cx), float(cy)]
    for j, pd in enumerate(np.linspace(0, 2 * np.pi, 10, endpoint=False)):
        rates.append(0.2 + 8 * np.exp(3 * (np.cos(hd - pd) - 1)))
        gta["hd"][str(10 + j)] = float(np.rad2deg(pd))
    for j in range(20, 30):
        rates.append(np.full(n, 2.0))
        gta["untuned"].append(str(j))
    sp, cols = zip(*(spikes_from_rate(t, r, rng) for r in rates))
    X = np.stack(cols, axis=1)
    write_session(src, "sub-enc02_ses-arena", t + BIN / 2, X, list(sp), ["x", "y", "speed", "head_direction"],
                  ["position_x", "position_y", "speed", "angle"], ["cm", "cm", "cm/s", "rad"],
                  ["CA1"] * 10 + ["PoS"] * 10 + ["CA1"] * 5 + ["PoS"] * 5, {"_Y": [x, y, sp2, hd]})
    gt["sub-enc02_ses-arena"] = gta

    # --- calcium-like ROIs on a one-way virtual track: deconvolved activity on an offset (as in real data,
    # it does not fall to zero outside the field), 15 place ROIs, 10 untuned
    fb = 1 / 15.5
    tf = np.arange(0, DUR, fb)
    nf = len(tf)
    # laps of varying speed (a perfectly periodic track would make circular shifts by whole laps no null)
    v_lap = np.repeat(rng.uniform(30, 80, int(DUR / 4) + 2), int(round(4 / fb)))[:nf]
    vpos = np.mod(np.cumsum(v_lap) * fb, 400.0)
    vspeed = v_lap
    rd3 = np.ones(nf)
    cols, gtr = [], {"place": {}, "untuned": []}
    for j, c in enumerate(np.linspace(30, 370, 15)):
        drive = 0.4 + 2.0 * np.exp(-0.5 * ((vpos - c) / 25.0) ** 2)
        cols.append(np.clip(rng.gamma(2.0, drive / 2.0) + rng.normal(0, 0.1, nf), 0, None))
        gtr["place"][str(j)] = float(c)
    for j in range(15, 25):
        cols.append(np.clip(rng.gamma(2.0, 0.6 / 2.0, nf) + rng.normal(0, 0.1, nf), 0, None))
        gtr["untuned"].append(str(j))
    write_session(src, "sub-enc03_ses-vr", tf + fb / 2, np.stack(cols, axis=1), None,
                  ["position", "speed", "run_direction"], ["position", "speed", "direction"], ["cm", "cm/s", "sign"],
                  ["CA1"] * 25, {"_Y": [vpos, vspeed, rd3]}, neural_kind="roi", bin_size=fb)
    gt["sub-enc03_ses-vr"] = gtr

    if layout in ("linked", "both"):
        dst = out / "linked" / "aligned"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src / "neuro-align", dst / "neuro-align", dirs_exist_ok=True)
    if layout in ("flat", "both"):
        dst = out / "flat"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src / "neuro-align", dst / "neuro-align", dirs_exist_ok=True)
    shutil.rmtree(src)
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=1))
    return gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layout", choices=("linked", "flat", "both"), default="both")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    make(a.out, a.layout, a.seed)
    print(f"wrote {a.out}")
