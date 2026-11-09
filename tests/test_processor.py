"""neuro-encoding on synthetic neuro-align output with known tuning, without Docker."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.make_fixture import make

FAST = {"N_SHUFFLES": 200, "N_SHUFFLES_GLM": 4, "GLM_FOLDS": 2, "GLM_MAX_ITER": 50}


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fixture")
    make(d, "both")
    return d


def run(input_dir: Path, output_dir: Path, **params) -> int:
    from processor import main as app

    env = {"INPUT_DIR": str(input_dir), "OUTPUT_DIR": str(output_dir)} | {k: str(v) for k, v in params.items()}
    old = {k: os.environ.get(k) for k in list(env) + list(app.PARAMETERS)}
    for k in app.PARAMETERS:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        return app.main()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="session")
def out(fixture_dir, tmp_path_factory):
    o = tmp_path_factory.mktemp("out")
    assert run(fixture_dir / "linked", o, **FAST) == 0
    return o / "neuro-encoding"


def table(out, sid):
    return pd.read_csv(out / sid / f"{sid}_encoding.csv", dtype={"neuron_id": str}).set_index("neuron_id")


def test_track(out, fixture_dir):
    gt = json.loads((fixture_dir / "ground_truth.json").read_text())["sub-enc01_ses-track"]
    df = table(out, "sub-enc01_ses-track")
    place = list(gt["place"])
    assert df.loc[place, "place_cell"].sum() >= 28
    assert df.loc[gt["untuned"], "place_cell"].sum() == 0
    peak_err = np.abs(df.loc[place, "position_peak"] - pd.Series(gt["place"]).loc[place])
    assert peak_err.median() < 8
    # one-direction cells are found in their direction
    for u, d in gt["place_direction"].items():
        if d in ("fwd", "bwd") and df.loc[u, "place_cell"]:
            assert d in str(df.loc[u, "place_direction"]), (u, d, df.loc[u, "place_direction"])
    # speed cells: significant positive speed score; untuned cells not
    assert df.loc[gt["speed"], "speed_tuned"].all()
    assert (df.loc[gt["speed"], "speed_speed_score"] > 0.2).all()
    assert df.loc[gt["untuned"], "speed_tuned"].sum() <= 1
    # GLM: position needed for place cells, speed for speed cells
    assert df.loc[place, "glm_position_significant"].mean() > 0.8
    assert df.loc[gt["speed"], "glm_speed_significant"].mean() > 0.7
    assert df.loc[gt["untuned"], "glm_d2"].median() < 0.02
    for f in ("place_sequence", "tuned_fraction", "tuning_examples", "glm"):
        assert (out / "sub-enc01_ses-track" / f"sub-enc01_ses-track_{f}.png").exists()


def test_arena(out, fixture_dir):
    gt = json.loads((fixture_dir / "ground_truth.json").read_text())["sub-enc02_ses-arena"]
    df = table(out, "sub-enc02_ses-arena")
    assert df.loc[list(gt["place2d"]), "position_2d_tuned"].sum() >= 8
    hd = list(gt["hd"])
    assert df.loc[hd, "head_direction_tuned"].all()
    err = np.abs(((df.loc[hd, "head_direction_preferred_deg"] - pd.Series(gt["hd"]).loc[hd]) + 180) % 360 - 180)
    assert err.max() < 20
    assert df.loc[gt["untuned"], "head_direction_tuned"].sum() <= 1
    assert (out / "sub-enc02_ses-arena" / "sub-enc02_ses-arena_place_maps.png").exists()


def test_outputs(out):
    s = json.loads((out / "summary.json").read_text())
    p = json.loads((out / "provenance.json").read_text())
    assert set(p) == {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                      "wall_seconds", "peak_rss_mb", "gpu", "sessions", "citations"}
    assert s["n_sessions"] == 2 and all((out / f["path"]).exists() and f["caption"] for f in s["figures"])
    allrows = pd.read_csv(out / "encoding.csv")
    assert set(allrows["session_id"]) == {"sub-enc01_ses-track", "sub-enc02_ses-arena"}
    tc = pd.read_csv(out / "sub-enc01_ses-track" / "sub-enc01_ses-track_tuning_curves.csv")
    assert {"neuron_id", "variable", "bin_centre", "rate", "occupancy_s"} <= set(tc.columns)


def test_flat_layout_and_determinism(fixture_dir, tmp_path):
    kw = dict(FAST, GLM="false", VARIABLES="speed")
    assert run(fixture_dir / "flat", tmp_path / "a", **kw) == 0
    assert run(fixture_dir / "flat", tmp_path / "b", **kw) == 0
    a = (tmp_path / "a" / "neuro-encoding" / "encoding.csv").read_text()
    b = (tmp_path / "b" / "neuro-encoding" / "encoding.csv").read_text()
    assert a == b


def test_bad_parameters_and_no_input(fixture_dir, tmp_path):
    assert run(fixture_dir / "linked", tmp_path, N_SHUFFLES="5") == 2
    assert run(fixture_dir / "linked", tmp_path, FDR="2") == 2
    assert run(fixture_dir / "linked", tmp_path, GLM="maybe") == 2
    (tmp_path / "empty").mkdir()
    assert run(tmp_path / "empty", tmp_path / "o") == 1


def test_null_routes_agree_and_pynapple():
    from processor import encode as E
    from processor import stats as S

    rng = np.random.default_rng(0)
    T, N, K = 4000, 6, 10
    X = rng.poisson(0.5, (T, N)).astype(float)
    v = rng.uniform(0, 10, T)
    edges = np.linspace(0, 10, K + 1)
    fb = E.digitize(v, edges, False)
    occ = S.occupancy(fb, K, 0.025)
    sh = S.shifts(T, 30, 100, rng)
    a = S.info_null(S.Sparse(X), fb, K, occ, 0.0, sh)[0]
    b = S.info_null_fft(X, fb, K, occ, 0.0, sh)[0]
    assert np.allclose(a, b)
    R = S.rates(S.sums(S.Sparse(X), fb, K), occ, 0.0)
    Rp = E.pynapple_curves(X, np.arange(T) * 0.025, v, edges, 0.025)
    assert np.allclose(R, Rp)
    # Skaggs information of a flat rate is 0, of a single-bin rate log2(1/p)
    occ1 = np.full(4, 1.0)
    assert abs(S.skaggs(np.full((4, 1), 3.0), occ1)[0]) < 1e-12
    assert abs(S.skaggs(np.array([[4.0], [0], [0], [0]]), occ1)[0] - 2.0) < 1e-12


def test_bh_and_fields():
    from processor import stats as S

    q = S.bh_fdr(np.array([0.01, 0.04, 0.03, 0.5, np.nan]))
    assert np.allclose(q[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert np.isnan(q[4])
    c = np.r_[np.zeros(10), np.hanning(8) * 10, np.zeros(22)]
    f = S.place_fields(c, np.arange(40) * 5.0, 1.0, 0.2, 1.0, 3, 0.5)
    assert len(f) == 1 and 50 <= f[0]["centre"] <= 85
