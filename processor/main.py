"""
neuro-encoding: what does each neuron encode?

Reads neuro-align's output (port `aligned`: neuro-align/<session_id>/aligned.npz + manifest.json) and
writes, per session, under neuro-encoding/<session_id>/:

  <session_id>_encoding.csv        one row per neuron: tuning information per variable with shift-null
                                   p / q, place fields and stability, speed scores, head-direction
                                   tuning, GLM deviance explained and drop-one contributions
  <session_id>_tuning_curves.csv   neuron_id, variable, bin_centre, rate, occupancy_s
  <session_id>_place_sequence.png, _tuned_fraction.png, _tuning_examples.png, _glm.png, (_place_maps.png)
  summary.json (per session)

and at the top: encoding.csv (all sessions), summary.json, provenance.json.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from processor.common import app_version, peak_rss_mb, rel, runtime_env, set_threads, tool_versions, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("neuro-encoding")
log.setLevel(logging.INFO)
for noisy in ("matplotlib", "PIL", "fontTools", "jax", "absl"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "neuro-encoding"
DEFAULTS = {
    "VARIABLES": "auto",
    "N_SHUFFLES": "500",
    "MIN_SHIFT_S": "20",
    "FDR": "0.05",
    "POSITION_BINS": "40",
    "POSITION_BINS_2D": "20",
    "N_BINS": "12",
    "ANGLE_BINS": "36",
    "MIN_OCCUPANCY_S": "0.2",
    "MIN_DURATION_S": "60",
    "PLACE_RUN_ONLY": "true",
    "SPLIT_DIRECTION": "true",
    "RUN_SPEED": "auto",
    "SMOOTH_BINS": "1.5",
    "FIELD_THRESHOLD": "0.2",
    "FIELD_MIN_PEAK_HZ": "1.0",
    "FIELD_MIN_BINS": "3",
    "FIELD_MAX_FRACTION": "0.8",
    "FIELD_BASELINE_PCT": "20",
    "SPEED_SMOOTH_S": "0.25",
    "GLM": "true",
    "N_SHUFFLES_GLM": "19",
    "GLM_BIN_S": "0.1",
    "GLM_N_BASIS": "8",
    "GLM_FOLDS": "3",
    "GLM_GAP_S": "2",
    "GLM_RIDGE": "0.001",
    "GLM_MAX_KEYPOINTS": "3",
    "GLM_MAX_ITER": "1000",
    "SEED": "42",
    "THREADS": "0",
}
PARAMETERS = tuple(DEFAULTS)
CITATIONS = ["10.7554/eLife.85786", "10.5281/zenodo.17553286", "10.1038/nature14622",
             "10.1111/j.2517-6161.1995.tb02031.x"]


class ParameterError(ValueError):
    pass


def _bool(v: str, name: str) -> bool:
    s = v.strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise ParameterError(f"{name} must be true or false")


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k).strip() if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {}
    v = [x.strip() for x in raw["VARIABLES"].split(",") if x.strip()]
    p["VARIABLES"] = None if not v or "auto" in [x.lower() for x in v] else v
    try:
        for k in ("N_SHUFFLES", "POSITION_BINS", "POSITION_BINS_2D", "N_BINS", "ANGLE_BINS", "FIELD_MIN_BINS",
                  "N_SHUFFLES_GLM", "GLM_N_BASIS", "GLM_FOLDS", "GLM_MAX_KEYPOINTS", "GLM_MAX_ITER", "SEED", "THREADS"):
            p[k] = int(raw[k])
        for k in ("MIN_SHIFT_S", "FDR", "MIN_OCCUPANCY_S", "MIN_DURATION_S", "SMOOTH_BINS", "FIELD_THRESHOLD",
                  "FIELD_MIN_PEAK_HZ", "FIELD_MAX_FRACTION", "FIELD_BASELINE_PCT", "SPEED_SMOOTH_S", "GLM_BIN_S", "GLM_GAP_S", "GLM_RIDGE"):
            p[k] = float(raw[k])
        p["RUN_SPEED"] = None if raw["RUN_SPEED"].lower() == "auto" else float(raw["RUN_SPEED"])
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    for k in ("PLACE_RUN_ONLY", "SPLIT_DIRECTION", "GLM"):
        p[k] = _bool(raw[k], k)
    if p["N_SHUFFLES"] < 20:
        raise ParameterError("N_SHUFFLES must be at least 20")
    if p["N_SHUFFLES_GLM"] < 2 and p["GLM"]:
        raise ParameterError("N_SHUFFLES_GLM must be at least 2")
    if not 0 <= p["FIELD_BASELINE_PCT"] < 100:
        raise ParameterError("FIELD_BASELINE_PCT must be in [0, 100)")
    if not 0 < p["FDR"] < 1 or not 0 < p["FIELD_THRESHOLD"] < 1 or not 0 < p["FIELD_MAX_FRACTION"] <= 1:
        raise ParameterError("FDR and FIELD_THRESHOLD must be in (0, 1), FIELD_MAX_FRACTION in (0, 1]")
    if min(p["POSITION_BINS"], p["POSITION_BINS_2D"], p["N_BINS"], p["ANGLE_BINS"]) < 3 or p["GLM_FOLDS"] < 2 \
            or p["GLM_N_BASIS"] < 3:
        raise ParameterError("bin counts must be >= 3, GLM_FOLDS >= 2, GLM_N_BASIS >= 3")
    if min(p["MIN_SHIFT_S"], p["MIN_OCCUPANCY_S"], p["MIN_DURATION_S"], p["SMOOTH_BINS"], p["SPEED_SMOOTH_S"],
           p["GLM_GAP_S"], p["GLM_RIDGE"], p["THREADS"]) < 0 or p["GLM_BIN_S"] <= 0:
        raise ParameterError("durations, smoothing, GLM_RIDGE and THREADS must be >= 0 (GLM_BIN_S > 0)")
    return p


def discover(root: Path) -> list[tuple[Path, dict]]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if "aligned.npz" in filenames:
            p = Path(dirpath) / "aligned.npz"
            m = {}
            mp = Path(dirpath) / "manifest.json"
            if mp.exists():
                try:
                    m = json.loads(mp.read_text())
                except (OSError, ValueError):
                    pass
            found.append((p, m))
    return found


def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 1
    runtime_env(APP_NAME)
    try:
        params = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    threads = set_threads(params["THREADS"])
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                               f" --xla_cpu_multi_thread_eigen={'true' if threads > 1 else 'false'}"
                               f" intra_op_parallelism_threads={threads}").strip()

    import numpy as np
    import pandas as pd

    import jax

    jax.config.update("jax_enable_x64", True)
    from processor import encode as E
    from processor import figures as F

    found = discover(input_dir)
    if not found:
        log.error("no neuro-align output (aligned.npz) under %s", input_dir)
        return 1
    log.info("%d sessions; %d shuffles (GLM %s, %d shuffles), FDR %g, seed %d, %d threads", len(found),
             params["N_SHUFFLES"], "on" if params["GLM"] else "off", params["N_SHUFFLES_GLM"], params["FDR"],
             params["SEED"], threads)
    top = output_dir / APP_NAME
    top.mkdir(parents=True, exist_ok=True)
    statuses, tables, figures, all_warn = [], [], [], []
    last = [time.monotonic()]

    def progress(msg):
        log.info("%s", msg)
        last[0] = time.monotonic()

    for path, manifest in found:
        t = time.monotonic()
        sid = manifest.get("session_id") or path.parent.name
        st = {"session_id": sid, "input": rel(path, input_dir)}
        try:
            s = E.load_session(path, manifest)
            sid = s.sid
            st["session_id"] = sid
            if s.X.shape[1] == 0:
                raise ValueError("no neurons")
            # one generator per session, so a session's results do not depend on the others
            rng = np.random.default_rng(params["SEED"])
            out = E.analyse(s, params, rng, progress)
            d = top / sid
            d.mkdir(parents=True, exist_ok=True)
            df = out["table"]
            df.to_csv(d / f"{sid}_encoding.csv", index=False, float_format="%.6g")
            pd.DataFrame(out["curves"], columns=["neuron_id", "variable", "bin_centre", "rate", "occupancy_s"]) \
                .to_csv(d / f"{sid}_tuning_curves.csv", index=False, float_format="%.6g")
            tables.append(df)
            unit_word = "firing rate (Hz)" if s.neural_kind == "units" else "activity"
            sfig = []
            for fn in (lambda: F.place_sequence(d / f"{sid}_place_sequence.png", sid, out["variables"], unit_word),
                       lambda: next((F.place_maps_2d(d / f"{sid}_place_maps.png", sid, v, s.ids)
                                     for v in out["variables"] if v["kind"] == "position2d"), None),
                       lambda: F.tuned_fraction(d / f"{sid}_tuned_fraction.png", sid, df, out["variables"], params["FDR"]),
                       lambda: F.tuning_examples(d / f"{sid}_tuning_examples.png", sid, out["variables"], s.ids, unit_word),
                       lambda: F.glm_contributions(d / f"{sid}_glm.png", sid, out["glm"], params["FDR"])
                       if out["glm"] and "covariates" in out["glm"] else None):
                try:
                    r = fn()
                    if r:
                        sfig.append({"path": rel(r[0], top), "caption": r[1]})
                except Exception as exc:
                    log.exception("%s: figure failed: %s", sid, exc)
                    out["warnings"].append(f"figure failed: {exc}")
            figures += sfig
            head = headline(s, out, df, params)
            write_json(d / "summary.json", {"app": APP_NAME, "version": app_version(), "sessions": {sid: head},
                                            "figures": [{"path": f["path"].split("/", 1)[-1], "caption": f["caption"]} for f in sfig],
                                            "tables": [f"{sid}_encoding.csv", f"{sid}_tuning_curves.csv"],
                                            "warnings": out["warnings"]})
            st.update(status="ok", headline=head, warnings=out["warnings"], seconds=round(time.monotonic() - t, 2))
            all_warn += [f"{sid}: {w}" for w in out["warnings"]]
            for w in out["warnings"]:
                log.warning("%s: %s", sid, w)
            log.info("%s: %d neurons, %s; %.1fs", sid, head["n_neurons"],
                     ", ".join(f"{k} {v['n_tuned']}/{head['n_neurons']}" for k, v in head["variables"].items()),
                     time.monotonic() - t)
        except Exception as exc:
            log.exception("%s: failed: %s", sid, exc)
            st.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=round(time.monotonic() - t, 2))
            all_warn.append(f"{sid}: failed: {exc}")
        statuses.append(st)
    ok = [s for s in statuses if s["status"] == "ok"]
    if tables:
        pd.concat(tables, ignore_index=True, sort=False).to_csv(top / "encoding.csv", index=False, float_format="%.6g")
    write_json(top / "summary.json", {
        "app": APP_NAME, "version": app_version(),
        "sessions": {s["session_id"]: s.get("headline") or {"status": s["status"], "error": s.get("error")} for s in statuses},
        "n_sessions": len(ok), "n_failed": len(statuses) - len(ok),
        "figures": figures,
        "tables": (["encoding.csv"] if tables else []) + [f"{s['session_id']}/{s['session_id']}_encoding.csv" for s in ok],
        "warnings": all_warn,
    })
    write_json(top / "provenance.json", {
        "app": APP_NAME, "version": app_version(), "variant": None,
        "tools": tool_versions(("pynapple", "nemos", "jax", "jaxlib", "numpy", "scipy", "pandas", "matplotlib")),
        "parameters": params | {"VARIABLES": "auto" if params["VARIABLES"] is None else ",".join(params["VARIABLES"]),
                                "RUN_SPEED": "auto" if params["RUN_SPEED"] is None else params["RUN_SPEED"]},
        "inputs": [rel(p, input_dir) for p, _ in found],
        "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.monotonic() - t0, 2), "peak_rss_mb": peak_rss_mb(), "gpu": None,
        "sessions": {s["session_id"]: {"status": s["status"], "seconds": s.get("seconds")} |
                     ({"error": s["error"]} if s.get("error") else {}) for s in statuses},
        "citations": CITATIONS,
    })
    if not ok:
        log.error("every session failed")
        return 1
    log.info("done: %d sessions ok, %d failed, %.1fs, peak RSS %s MB -> %s", len(ok), len(statuses) - len(ok),
             time.monotonic() - t0, peak_rss_mb(), top)
    return 0


def headline(s, out: dict, df, params) -> dict:
    import numpy as np

    h = {"n_neurons": len(s.ids), "neural_kind": s.neural_kind, "bin_size_s": s.bin_size,
         "epoch_seconds": round(float(s.epoch.sum() * s.bin_size), 2), "position_mask": out["run_mask"],
         "subject_id": (s.manifest.get("sheet") or {}).get("subject_id"),
         "condition": (s.manifest.get("sheet") or {}).get("condition"),
         "group": (s.manifest.get("sheet") or {}).get("group"), "variables": {}}
    for v in out["variables"]:
        r = v["res"]
        e = {"kind": v["kind"], "seconds": v["seconds"], "n_tuned": int(r["tuned"].sum()),
             "fraction_tuned": round(float(r["tuned"].mean()), 4),
             "median_info_bits": float(np.nanmedian(r["info"]))}
        if v["kind"] == "position":
            e["median_info_tuned_bits"] = float(np.nanmedian(r["info"][r["tuned"]])) if r["tuned"].any() else None
            e["median_stability_tuned"] = float(np.nanmedian(r["stability"][r["tuned"]])) if r["tuned"].any() else None
        if "score" in r:
            e["n_positive_speed_cells"] = int((r["tuned"] & (r["score"] > 0)).sum())
        h["variables"][v["name"]] = e
    if "place_cell" in df:
        h["n_place_cells"] = int(df["place_cell"].sum())
    g = out.get("glm")
    if g and "d2" in g:
        h["glm"] = {"median_d2": float(np.nanmedian(g["d2"])), "bin_s": g["bin_s"], "n_features": g["n_features"],
                    "n_significant": {c: int(v["significant"].sum()) for c, v in g["covariates"].items()}}
    elif g:
        h["glm"] = g
    return h


if __name__ == "__main__":
    sys.exit(main())
