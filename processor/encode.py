"""One session: tuning curves, information, place fields, speed scores, head-direction tuning, GLMs."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from processor import stats as S

log = logging.getLogger("neuro-encoding")

ANALYSED = ("position", "speed", "angle", "wheel_velocity", "pupil", "keypoint_speed", "other")
SPEED_LIKE = ("speed", "keypoint_speed")
CM_PER = {"cm": 1.0, "m": 100.0, "mm": 0.1, "meters": 100.0, "centimeters": 1.0}
MIN_DIR_FRACTION = 0.2


@dataclass
class Session:
    sid: str
    t: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    names: list[str]
    kinds: list[str]
    units: list[str]
    epoch: np.ndarray
    bin_size: float
    ids: list[str]
    meta: np.ndarray
    neural_kind: str
    manifest: dict = field(default_factory=dict)

    def col(self, name):
        return self.Y[:, self.names.index(name)].astype(float) if name in self.names else None

    def kind(self, name):
        return self.kinds[self.names.index(name)]

    def unit(self, name):
        return self.units[self.names.index(name)]


def load_session(path, manifest: dict) -> Session:
    d = np.load(path, allow_pickle=False)
    names = [str(x) for x in d["behavior_names"]]
    kinds = [str(x) for x in d["behavior_kinds"]] if "behavior_kinds" in d.files else \
        [next((v["kind"] for v in manifest.get("behavior", {}).get("variables", []) if v["name"] == n), "other") for n in names]
    units = [str(x) for x in d["behavior_units"]] if "behavior_units" in d.files else [""] * len(names)
    sid = str(d["session_id"]) if "session_id" in d.files else manifest.get("session_id") or path.parent.name
    nk = str(d["neural_kind"]) if "neural_kind" in d.files else manifest.get("neural", {}).get("kind", "units")
    bs = float(d["bin_size"]) if "bin_size" in d.files else float(np.median(np.diff(d["t"])))
    return Session(sid, d["t"], d["X"], d["Y"], names, kinds, units, d["epoch_mask"].astype(bool), bs,
                   [str(x) for x in d["neuron_ids"]], d["neuron_meta"], nk, manifest)


def run_mask(s: Session, params: dict) -> tuple[np.ndarray | None, str]:
    rd = s.col("run_direction")
    if rd is not None:
        return np.isfinite(rd) & (rd != 0), "run_direction != 0 (neuro-align's running threshold)"
    sp = s.col("speed")
    if sp is None:
        return None, "no speed: all epoch bins"
    thr = params["RUN_SPEED"]
    if thr is None:
        f = CM_PER.get(s.unit("speed").replace("/s", "").strip().lower())
        thr = 3.0 / f if f else 0.1 * np.nanpercentile(sp, 95)
    return np.isfinite(sp) & (sp >= thr), f"speed >= {thr:g} {s.unit('speed')}"


def edges_for(v: np.ndarray, kind: str, params: dict) -> np.ndarray:
    f = v[np.isfinite(v)]
    if kind == "angle":
        return np.linspace(0, 2 * np.pi, params["ANGLE_BINS"] + 1)
    if kind == "position":
        lo, hi = np.percentile(f, [0.5, 99.5])
        return np.linspace(lo, hi, params["POSITION_BINS"] + 1)
    lo, hi = np.percentile(f, [1, 99])
    if not hi > lo:
        lo, hi = f.min(), f.max() + 1e-9
    return np.linspace(lo, hi, params["N_BINS"] + 1)


def digitize(v: np.ndarray, edges: np.ndarray, circular: bool) -> np.ndarray:
    K = len(edges) - 1
    fb = np.full(len(v), -1, dtype=np.int64)
    ok = np.isfinite(v)
    if circular:
        x = np.mod(v[ok], 2 * np.pi)
        fb[ok] = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, K - 1)
    else:
        # values outside the edges (the extreme percentiles) are left out, as pynapple does
        x = v[ok]
        k = np.searchsorted(edges, x, side="right") - 1
        k[x == edges[-1]] = K - 1
        fb[ok] = np.where((x >= edges[0]) & (x <= edges[-1]), k, -1)
    return fb


def pynapple_curves(Xv: np.ndarray, tv: np.ndarray, v: np.ndarray, edges: np.ndarray, bin_size: float):
    """Observed tuning curves with pynapple (mean activity per feature bin, per second for spike counts)."""
    import pynapple as nap

    ok = np.isfinite(v)
    tc = nap.compute_tuning_curves(nap.TsdFrame(t=tv[ok], d=Xv[ok]), nap.Tsd(t=tv[ok], d=v[ok]),
                                   bins=[edges], fs=1.0 / bin_size)
    return np.asarray(tc.values, dtype=float).T / bin_size   # K x N, per second


def analyse(s: Session, params: dict, rng, progress) -> dict:
    t0 = time.monotonic()
    N = len(s.ids)
    bs = s.bin_size
    min_shift = int(round(params["MIN_SHIFT_S"] / bs))
    X = s.X.astype(np.float64)
    if s.neural_kind != "units":
        X = np.clip(np.nan_to_num(X), 0, None)   # information needs non-negative activity
    ep = s.epoch
    rows: dict[str, dict] = {}
    curves = []
    variables = []
    run, run_desc = run_mask(s, params)
    warnings = []

    wanted = params["VARIABLES"]
    cand = [(n, k) for n, k in zip(s.names, s.kinds) if k in ANALYSED]
    if "position" not in s.names and "x" in s.names and "y" in s.names:
        cand.insert(0, ("position_2d", "position2d"))
    if wanted is not None:
        cand = [(n, k) for n, k in cand if n in wanted]
        missing = [w for w in wanted if w not in [c[0] for c in cand]]
        if missing:
            warnings.append(f"VARIABLES not in this session: {', '.join(missing)}")

    jobs = []
    for name, kind in cand:
        if kind == "position":
            v = s.col(name)
            m = ep & np.isfinite(v)
            if params["PLACE_RUN_ONLY"] and run is not None:
                m &= run
            jobs.append((name, "position", v, m, name))
            rd = s.col("run_direction")
            if rd is not None and params["SPLIT_DIRECTION"]:
                fwd, bwd = m & (rd > 0), m & (rd < 0)
                tot = fwd.sum() + bwd.sum()
                if tot and min(fwd.sum(), bwd.sum()) >= MIN_DIR_FRACTION * tot:
                    jobs.append((f"{name}_fwd", "position", v, fwd, name))
                    jobs.append((f"{name}_bwd", "position", v, bwd, name))
        elif kind == "position2d":
            xx, yy = s.col("x"), s.col("y")
            m = ep & np.isfinite(xx) & np.isfinite(yy)
            if params["PLACE_RUN_ONLY"] and run is not None:
                m &= run
            jobs.append((name, "position2d", np.c_[xx, yy], m, "x,y"))
        else:
            v = s.col(name)
            jobs.append((name, kind, v, ep & np.isfinite(v), name))

    for name, kind, v, m, src in jobs:
        sel = np.flatnonzero(m)
        if len(sel) * bs < params["MIN_DURATION_S"]:
            warnings.append(f"{name}: only {len(sel) * bs:.1f} s of data in the epoch; not analysed")
            continue
        tv = time.monotonic()
        Xv = X[sel]
        sp = S.Sparse(Xv)
        circular = kind == "angle"
        if kind == "position2d":
            vv = v[sel]
            ex = edges_for(vv[:, 0], "position", {"POSITION_BINS": params["POSITION_BINS_2D"]})
            ey = edges_for(vv[:, 1], "position", {"POSITION_BINS": params["POSITION_BINS_2D"]})
            fx, fy = digitize(vv[:, 0], ex, False), digitize(vv[:, 1], ey, False)
            fb = np.where((fx >= 0) & (fy >= 0), fx * (len(ey) - 1) + fy, -1)
            K = (len(ex) - 1) * (len(ey) - 1)
            centres = None
        else:
            vv = v[sel]
            edges = edges_for(vv, kind, params)
            fb = digitize(vv, edges, circular)
            K = len(edges) - 1
            centres = (edges[:-1] + edges[1:]) / 2
        occ = S.occupancy(fb, K, bs)
        R = S.rates(S.sums(sp, fb, K), occ, params["MIN_OCCUPANCY_S"])
        if kind not in ("position2d",):
            try:
                Rp = pynapple_curves(Xv, s.t[sel], vv, edges, bs)
                Rp[occ < params["MIN_OCCUPANCY_S"]] = np.nan
                if Rp.shape == R.shape and np.nanmax(np.abs(Rp - R)) > 1e-6 * max(1.0, np.nanmax(np.abs(R))):
                    warnings.append(f"{name}: pynapple and binned tuning curves differ (max {np.nanmax(np.abs(Rp - R)):.3g})")
            except Exception as exc:
                warnings.append(f"{name}: pynapple tuning curves failed ({exc}); used the binned estimate")
        info = S.skaggs(R, occ)
        sh = S.shifts(len(sel), params["N_SHUFFLES"], min_shift, rng)
        if len(sel) < 2 * min_shift:
            warnings.append(f"{name}: {len(sel) * bs:.0f} s of data is short for MIN_SHIFT_S={params['MIN_SHIFT_S']}; shifts of at least a quarter of it used")
        route = S.null_cost(sp, K, len(sh))
        if route == "fft":
            null_i, null_r = S.info_null_fft(Xv, fb, K, occ, params["MIN_OCCUPANCY_S"], sh, centres if circular else None)
        else:
            null_i, null_r = S.info_null(sp, fb, K, occ, params["MIN_OCCUPANCY_S"], sh, centres if circular else None)
        p_i, z_i = S.empirical_p(info, null_i)
        res = {"info": info, "info_p": p_i, "info_z": z_i,
               "info_null95": np.percentile(null_i, 95, axis=0)}
        Rs = None
        if kind == "position":
            Rs = np.stack([S.nan_smooth(R[:, j], params["SMOOTH_BINS"]) for j in range(N)], axis=1)
            mean_rate = np.nansum(R * occ[:, None], axis=0) / max(occ[occ >= params["MIN_OCCUPANCY_S"]].sum(), 1e-9)
            fields = [S.place_fields(Rs[:, j], centres, mean_rate[j], params["FIELD_THRESHOLD"],
                                     params["FIELD_MIN_PEAK_HZ"] if s.neural_kind == "units" else 0.0,
                                     params["FIELD_MIN_BINS"], params["FIELD_MAX_FRACTION"]) for j in range(N)]
            half = len(sel) // 2
            sp1, sp2 = S.Sparse(Xv[:half]), S.Sparse(Xv[half:])
            stab = S.split_half(sp1, sp2, fb[:half], fb[half:], K, bs, params["MIN_OCCUPANCY_S"] / 2)
            # held-out halves for the cross-validated sequence figure
            R1 = S.rates(S.sums(sp1, fb[:half], K), S.occupancy(fb[:half], K, bs), params["MIN_OCCUPANCY_S"] / 2)
            R2 = S.rates(S.sums(sp2, fb[half:], K), S.occupancy(fb[half:], K, bs), params["MIN_OCCUPANCY_S"] / 2)
            res.update(fields=fields, stability=stab, halves=(R1, R2))
        if kind == "position2d":
            res["fields"] = None
        if circular:
            mrl, pref = S.resultant(R, centres)
            p_r, z_r = S.empirical_p(mrl, null_r)
            res.update(mrl=mrl, mrl_p=p_r, pref=pref)
        if kind in SPEED_LIKE:
            r, null = S.speed_score(Xv, vv, bs, params["SPEED_SMOOTH_S"], sh)
            res.update(score=r, score_p=S.two_sided(r, null))
        variables.append({"name": name, "kind": kind, "source": src, "K": K, "edges": None if kind == "position2d" else edges,
                          "seconds": round(len(sel) * bs, 2), "unit": s.unit(src) if src in s.names else s.unit("x"),
                          "mask": run_desc if kind.startswith("position") and params["PLACE_RUN_ONLY"] else "epoch",
                          "R": R, "Rs": Rs, "occ": occ, "centres": centres, "res": res,
                          "shape2d": ((len(ex) - 1), (len(ey) - 1)) if kind == "position2d" else None})
        if kind != "position2d":
            for j in range(N):
                for k in range(K):
                    if np.isfinite(R[k, j]):
                        curves.append((s.ids[j], name, float(centres[k]), float(R[k, j]), float(occ[k])))
        progress(f"{s.sid}: {name} ({kind}, {len(sel) * bs:.0f} s, {K} bins, {params['N_SHUFFLES']} shuffles by {route}) "
                 f"{time.monotonic() - tv:.1f}s")

    # FDR per statistic across neurons, then the per-variable verdict
    for var in variables:
        r = var["res"]
        r["info_q"] = S.bh_fdr(r["info_p"])
        if "mrl" in r:
            r["mrl_q"] = S.bh_fdr(r["mrl_p"])
        if "score" in r:
            r["score_q"] = S.bh_fdr(r["score_p"])
        fdr = params["FDR"]
        if var["kind"] == "position":
            nf = np.array([len(f) for f in r["fields"]])
            r["tuned"] = (r["info_q"] < fdr) & (nf > 0)
        elif var["kind"] == "angle":
            r["tuned"] = r["mrl_q"] < fdr
        elif var["kind"] in SPEED_LIKE:
            r["tuned"] = r["score_q"] < fdr
        else:
            r["tuned"] = r["info_q"] < fdr

    out = {"variables": variables, "curves": curves, "warnings": warnings, "glm": None, "run_mask": run_desc}
    if params["GLM"]:
        try:
            out["glm"] = glm_session(s, X, variables, params, rng, progress)
        except Exception as exc:
            log.exception("%s: GLM failed: %s", s.sid, exc)
            warnings.append(f"GLM failed: {exc}")
    out["table"] = table(s, out, params)
    out["seconds"] = round(time.monotonic() - t0, 2)
    return out


def glm_session(s: Session, X: np.ndarray, variables: list[dict], params: dict, rng, progress) -> dict | None:
    from processor.glm import build_covariates, run_glm

    names = [v["name"] for v in variables if not v["name"].endswith(("_fwd", "_bwd"))]
    cols, kinds = {}, {}
    for n in names:
        if n == "position_2d":
            cols[n] = np.c_[s.col("x"), s.col("y")]
            kinds[n] = "position2d"
        else:
            cols[n] = s.col(n)
            kinds[n] = s.kind(n)
    # keypoint speeds: at most GLM_MAX_KEYPOINTS of them (they are strongly correlated with each other)
    kp = [n for n in cols if kinds[n] == "keypoint_speed"]
    for n in kp[params["GLM_MAX_KEYPOINTS"]:]:
        cols.pop(n)
    if not cols:
        return None
    g = max(1, int(round(params["GLM_BIN_S"] / s.bin_size)))
    valid = s.epoch.copy()
    for v in cols.values():
        valid &= np.all(np.isfinite(v.reshape(len(v), -1)), axis=1)
    nb = len(valid) // g
    vb = valid[: nb * g].reshape(nb, g).all(axis=1)
    if vb.sum() < 300:
        return {"skipped": f"only {vb.sum()} valid {g * s.bin_size:.3g} s bins in the epoch"}
    Yn = X[: nb * g].reshape(nb, g, -1).sum(axis=1)[vb]
    if s.neural_kind != "units":
        Yn = Yn / g   # ROI activity: mean per GLM bin (quasi-Poisson on non-negative activity)
    rebinned = {}
    for n, v in cols.items():
        if kinds[n] == "angle":
            c = np.cos(v[: nb * g]).reshape(nb, g).mean(axis=1)
            si = np.sin(v[: nb * g]).reshape(nb, g).mean(axis=1)
            rebinned[n] = np.mod(np.arctan2(si, c), 2 * np.pi)[vb]
        elif v.ndim == 2:
            rebinned[n] = v[: nb * g].reshape(nb, g, 2).mean(axis=1)[vb]
        else:
            rebinned[n] = v[: nb * g].reshape(nb, g).mean(axis=1)[vb]
    covs = build_covariates(rebinned, kinds, params["GLM_N_BASIS"])
    tg = time.monotonic()
    res = run_glm(Yn, covs, params["GLM_FOLDS"], int(round(params["GLM_GAP_S"] / (g * s.bin_size))),
                  params["GLM_RIDGE"], params["N_SHUFFLES_GLM"], int(round(params["MIN_SHIFT_S"] / (g * s.bin_size))),
                  rng, log_every=progress, maxiter=params["GLM_MAX_ITER"])
    for c in res["covariates"].values():
        c["p"] = S.gaussian_p_upper(c["delta_d2"], c["null"])
        c["p_empirical"] = S.empirical_p(c["delta_d2"], c["null"])[0]
        c["q"] = S.bh_fdr(c["p"])
        c["significant"] = (c["q"] < params["FDR"]) & (c["delta_d2"] > 0)
    res.update(bin_s=g * s.bin_size, n_bins=int(vb.sum()), covariate_names=[c.name for c in covs],
               n_features=int(sum(c.n_basis for c in covs)), seconds=round(time.monotonic() - tg, 1))
    progress(f"{s.sid}: GLM {len(covs)} covariates, {res['n_features']} features, {res['n_bins']} bins of "
             f"{res['bin_s']:.3g} s, {len(s.ids)} neurons, {res['seconds']}s")
    return res


def table(s: Session, out: dict, params: dict) -> pd.DataFrame:
    meta = s.meta
    N = len(s.ids)
    df = pd.DataFrame({"session_id": s.sid, "neuron_id": s.ids})
    for col in ("region", "label", "cell_type", "depth", "x", "y", "rate_hz"):
        if meta.dtype.names and col in meta.dtype.names:
            df[col if col != "rate_hz" else "mean_rate"] = meta[col]
    ep = s.epoch
    df["epoch_mean_rate"] = np.nanmean(s.X[ep], axis=0) / (s.bin_size if s.neural_kind == "units" else 1.0)
    place_any = np.zeros(N, bool)
    place_dir = np.array([""] * N, dtype=object)
    for var in out["variables"]:
        n, r = var["name"], var["res"]
        df[f"{n}_info"] = r["info"]
        df[f"{n}_info_z"] = r["info_z"]
        df[f"{n}_info_p"] = r["info_p"]
        df[f"{n}_info_q"] = r["info_q"]
        R = var["Rs"] if var["Rs"] is not None else var["R"]
        if var["centres"] is not None:
            with np.errstate(invalid="ignore"):
                okc = np.isfinite(R).any(axis=0)
                k = np.where(okc, np.nanargmax(np.where(np.isfinite(R), R, -np.inf), axis=0), 0)
            df[f"{n}_peak"] = np.where(okc, var["centres"][k], np.nan)
            df[f"{n}_peak_rate"] = np.where(okc, R[k, np.arange(N)], np.nan)
        if "mrl" in r:
            df[f"{n}_mrl"] = r["mrl"]
            df[f"{n}_mrl_p"] = r["mrl_p"]
            df[f"{n}_mrl_q"] = r["mrl_q"]
            df[f"{n}_preferred_deg"] = np.rad2deg(r["pref"])
        if "score" in r:
            df[f"{n}_speed_score"] = r["score"]
            df[f"{n}_speed_score_p"] = r["score_p"]
            df[f"{n}_speed_score_q"] = r["score_q"]
        if var["kind"] == "position":
            df[f"{n}_n_fields"] = [len(f) for f in r["fields"]]
            df[f"{n}_field_centres"] = [";".join(f"{x['centre']:.4g}" for x in f) for f in r["fields"]]
            df[f"{n}_field_widths"] = [";".join(f"{x['width']:.4g}" for x in f) for f in r["fields"]]
            df[f"{n}_stability"] = r["stability"]
            place_any |= r["tuned"]
            if n.endswith(("_fwd", "_bwd")):
                for j in np.flatnonzero(r["tuned"]):
                    place_dir[j] = ";".join(filter(None, [place_dir[j], n[-3:]]))
        df[f"{n}_tuned"] = r["tuned"]
    if any(v["kind"] == "position" for v in out["variables"]):
        df["place_cell"] = place_any
        if any(v["name"].endswith("_fwd") for v in out["variables"]):
            df["place_direction"] = [d or ("any" if pa else "") for d, pa in zip(place_dir, place_any)]
    g = out.get("glm")
    if g and "d2" in g:
        df["glm_d2"] = g["d2"]
        for c, v in g["covariates"].items():
            df[f"glm_{c}_delta_d2"] = v["delta_d2"]
            df[f"glm_{c}_bits_per_spike"] = v["bits_per_spike"]
            df[f"glm_{c}_p"] = v["p"]
            df[f"glm_{c}_q"] = v["q"]
            df[f"glm_{c}_significant"] = v["significant"]
    return df
