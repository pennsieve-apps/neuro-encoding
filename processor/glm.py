"""
Poisson GLM per neuron with NeMoS (all neurons fitted together as a PopulationGLM, which shares the
design matrix): each behaviour covariate enters through a basis (raised cosines for linear variables,
cyclic B-splines for angles, a product of raised cosines for 2D position). Cross-validated on
contiguous time blocks; the contribution of covariate c is the deviance the model loses when c is
dropped, as a fraction of the mean-rate model's deviance (Delta D^2), tested against fits in which only
c is circularly shifted.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

log = logging.getLogger("neuro-encoding")


@dataclass
class Covariate:
    name: str
    columns: np.ndarray      # (T, n_features) basis features
    n_basis: int


def build_covariates(cols: dict, kinds: dict, n_basis: int) -> list[Covariate]:
    """cols: name -> values (T,) or (T, 2) for 2D position; returns basis-expanded covariates."""
    import nemos as nmo

    out = []
    for name, v in cols.items():
        kind = kinds[name]
        if kind == "angle":
            b = nmo.basis.CyclicBSplineEval(n_basis, order=3, bounds=(0, 2 * np.pi))
            F = b.compute_features(np.mod(v, 2 * np.pi))
        elif v.ndim == 2:
            lo = np.nanpercentile(v, 1, axis=0)
            hi = np.nanpercentile(v, 99, axis=0)
            nb = max(4, int(round(np.sqrt(n_basis * 4))))
            b = nmo.basis.RaisedCosineLinearEval(nb, bounds=(lo[0], hi[0])) * \
                nmo.basis.RaisedCosineLinearEval(nb, bounds=(lo[1], hi[1]))
            F = b.compute_features(np.clip(v[:, 0], lo[0], hi[0]), np.clip(v[:, 1], lo[1], hi[1]))
        else:
            lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
            if not hi > lo:
                continue
            b = nmo.basis.RaisedCosineLinearEval(n_basis, bounds=(lo, hi))
            F = b.compute_features(np.clip(v, lo, hi))
        out.append(Covariate(name, np.asarray(F, dtype=np.float64), F.shape[1]))
    return out


def poisson_ll(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu = np.clip(mu, 1e-12, None)
    return (y * np.log(mu) - mu - gammaln(y + 1)).sum(axis=0)


def folds(T: int, k: int, gap: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """k contiguous test blocks, training on the rest minus a gap on each side. Every fold's training (and test)
    set is cut to the same length, so JAX compiles the fit once per design width instead of once per fold."""
    edges = np.linspace(0, T, k + 1).astype(int)
    out = []
    for i in range(k):
        test = np.arange(edges[i], edges[i + 1])
        train = np.r_[np.arange(0, max(edges[i] - gap, 0)), np.arange(min(edges[i + 1] + gap, T), T)]
        out.append((train, test))
    n_tr = min(len(tr) for tr, _ in out)
    n_te = min(len(te) for _, te in out)
    return [(tr[:n_tr], te[:n_te]) for tr, te in out]


class Fitter:
    def __init__(self, ridge: float, maxiter: int = 1000):
        import nemos as nmo

        self.nmo = nmo
        self.ridge = ridge
        self.maxiter = maxiter
        self.models: dict = {}
        self.n_fits = 0
        self.n_unconverged = 0
        self.unconverged_keys: set = set()

    def model(self, width: int):
        """One model per design width, reused: a new instance (or a new data shape) makes JAX compile again,
        which costs seconds and memory that is not given back."""
        if width not in self.models:
            self.models[width] = self.nmo.glm.PopulationGLM(
                observation_model="Poisson", regularizer="Ridge", regularizer_strength=self.ridge,
                solver_name="LBFGS", solver_kwargs={"maxiter": self.maxiter})
        return self.models[width]

    def cv_ll(self, design: np.ndarray, Y: np.ndarray, fl, key: str) -> np.ndarray:
        """Held-out Poisson log-likelihood per neuron, summed over folds. Every fit starts cold from zero
        weights and the log mean rate (a neuron silent in the training blocks gets a very low rate), so the
        observed and the shifted fits are treated alike; fits that stop at GLM_MAX_ITER are counted."""
        tot = np.zeros(Y.shape[1])
        for i, (tr, te) in enumerate(fl):
            m = self.model(design.shape[1])
            init = (np.zeros((design.shape[1], Y.shape[1])), np.log(Y[tr].mean(axis=0) + 1e-6))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m.fit(design[tr], Y[tr], init_params=init)
            self.n_fits += 1
            if any("converge" in str(w.message).lower() or "number of iterations" in str(w.message).lower()
                   for w in caught):
                self.n_unconverged += 1
                self.unconverged_keys.add(key.split(":")[0])
            mu = np.asarray(m.predict(design[te]))
            tot += poisson_ll(Y[te], mu)
        return tot


def run_glm(Y: np.ndarray, covs: list[Covariate], fold_k: int, gap_bins: int, ridge: float, n_shuffles: int,
            min_shift: int, rng, log_every=None, maxiter: int = 1000) -> dict:
    """Cross-validated deviance explained (D^2) of the full model and Delta D^2 per covariate with shift nulls."""
    T, N = Y.shape
    fl = folds(T, fold_k, gap_bins)
    F = Fitter(ridge, maxiter)
    test_idx = np.concatenate([te for _, te in fl])
    ll_sat = poisson_ll(Y[test_idx], np.maximum(Y[test_idx], 1e-12))
    # mean-rate (intercept-only) model from the training folds
    ll_null = np.zeros(N)
    for tr, te in fl:
        ll_null += poisson_ll(Y[te], np.broadcast_to(Y[tr].mean(axis=0), Y[te].shape))
    dev_null = 2 * (ll_sat - ll_null)
    dev_null = np.where(dev_null > 1e-9, dev_null, np.nan)   # a neuron silent in the test blocks: no D^2
    full = np.concatenate([c.columns for c in covs], axis=1)
    ll_full = F.cv_ll(full, Y, fl, "full")
    dev_full = 2 * (ll_sat - ll_full)
    with np.errstate(divide="ignore", invalid="ignore"):
        d2 = 1 - dev_full / dev_null
    res = {"d2": d2, "ll_full": ll_full, "ll_null": ll_null, "covariates": {}}
    for ci, c in enumerate(covs):
        others = [x.columns for j, x in enumerate(covs) if j != ci]
        if others:
            ll_drop = F.cv_ll(np.concatenate(others, axis=1), Y, fl, f"drop:{c.name}")
        else:
            ll_drop = ll_null
        dev_drop = 2 * (ll_sat - ll_drop)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = (dev_drop - dev_full) / dev_null
        null = np.empty((n_shuffles, N))
        lo = min(min_shift, T // 4)
        sh = rng.integers(lo, max(T - lo, lo + 1), size=n_shuffles)
        for s_i, s in enumerate(sh):
            shifted = [x.columns if j != ci else np.roll(x.columns, int(s), axis=0) for j, x in enumerate(covs)]
            ll_s = F.cv_ll(np.concatenate(shifted, axis=1), Y, fl, f"shift:{c.name}")
            dev_s = 2 * (ll_sat - ll_s)
            with np.errstate(divide="ignore", invalid="ignore"):
                null[s_i] = (dev_drop - dev_s) / dev_null
        if log_every:
            log_every(f"GLM: {c.name} done ({ci + 1}/{len(covs)})")
        res["covariates"][c.name] = {"delta_d2": delta, "null": null,
                                     "bits_per_spike": (ll_full - ll_drop) / np.maximum(Y[test_idx].sum(axis=0), 1) / np.log(2)}
    res["fits"] = {"n": F.n_fits, "stopped_at_maxiter": F.n_unconverged, "kinds": sorted(F.unconverged_keys)}
    import jax

    jax.clear_caches()   # compiled fits of this session are not needed again
    return res
