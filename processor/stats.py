"""
Single-neuron tuning statistics and their circular-shift nulls.

Everything works on the epoch bins of aligned.npz: X (bins x neurons, spike counts or ROI activity) and
one behaviour variable at a time, discretised into feature bins. A circular shift moves the behaviour
relative to the neural data by a random offset of at least MIN_SHIFT_S, which keeps both
autocorrelations intact and breaks only their relation; occupancy is unchanged by the shift, so only
the activity sums are recomputed (one bincount over the non-zero entries of X per shuffle).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q); NaN stays NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return q
    pv = p[ok]
    o = np.argsort(pv)
    ranked = pv[o] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def shifts(n: int, n_shuffles: int, min_shift: int, rng) -> np.ndarray:
    lo = min(max(min_shift, 1), max(n // 4, 1))
    hi = max(n - lo, lo + 1)
    return rng.integers(lo, hi, size=n_shuffles)


def empirical_p(obs: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-sided p = (1 + #null >= obs) / (1 + n) per neuron, and the z-score of obs against the null."""
    n = null.shape[0]
    ge = (null >= obs[None, :] - 1e-12).sum(axis=0)
    p = (1 + ge) / (1 + n)
    sd = null.std(axis=0)
    z = np.where(sd > 0, (obs - null.mean(axis=0)) / np.where(sd > 0, sd, 1), np.nan)
    # Beyond every shuffle the empirical p is only bounded by 1/(n+1); there a normal fitted to the null gives the
    # tail probability (never more than 1/(n+1)), so FDR over hundreds of neurons stays possible with 500 shuffles.
    beyond = (ge == 0) & np.isfinite(z)
    p[beyond] = np.minimum(1.0 / (1 + n), norm.sf(z[beyond]))
    p[~np.isfinite(obs)] = np.nan
    return p, z


class Sparse:
    """The non-zero entries of X (bins x neurons) restricted to the analysed bins, for fast re-binning."""

    def __init__(self, X: np.ndarray):
        Xc = np.nan_to_num(np.clip(X, 0, None))
        self.T, self.N = Xc.shape
        self.rows, self.cols = np.nonzero(Xc)
        self.vals = Xc[self.rows, self.cols].astype(np.float64)


def sums(sp: Sparse, fb: np.ndarray, K: int) -> np.ndarray:
    """Activity summed per feature bin (K x N); fb = feature bin per time bin, -1 = no value."""
    k = fb[sp.rows]
    ok = k >= 0
    flat = np.bincount(k[ok] * sp.N + sp.cols[ok], weights=sp.vals[ok], minlength=K * sp.N)
    return flat.reshape(K, sp.N)


def occupancy(fb: np.ndarray, K: int, bin_size: float) -> np.ndarray:
    return np.bincount(fb[fb >= 0], minlength=K).astype(float) * bin_size


def rates(S: np.ndarray, occ: np.ndarray, min_occ: float) -> np.ndarray:
    good = occ >= min_occ
    R = np.full(S.shape, np.nan)
    R[good] = S[good] / occ[good, None]
    return R


def skaggs(R: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """Skaggs information in bits per spike (per unit activity for ROIs) of each column of R (K x N)."""
    good = np.isfinite(R[:, 0]) if R.ndim == 2 and R.shape[1] else np.zeros(len(occ), bool)
    p = np.where(good, occ, 0.0)
    if p.sum() <= 0:
        return np.full(R.shape[1], np.nan)
    p = p / p.sum()
    Rg = np.where(good[:, None], np.nan_to_num(R), 0.0)
    lam = p @ Rg
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = Rg / lam[None, :]
        term = np.where(ratio > 0, ratio * np.log2(ratio), 0.0)
    info = p @ term
    info[~(lam > 0)] = np.nan
    return info


def resultant(R: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean resultant length and preferred angle (rad) of angular tuning curves (K x N)."""
    Rg = np.nan_to_num(R)
    z = (Rg * np.exp(1j * centres)[:, None]).sum(axis=0)
    tot = Rg.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mrl = np.abs(z) / tot
    return np.where(tot > 0, mrl, np.nan), np.mod(np.angle(z), 2 * np.pi)


def info_null(sp: Sparse, fb: np.ndarray, K: int, occ: np.ndarray, min_occ: float, sh: np.ndarray,
              centres: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Null distributions (n_shuffles x N) of Skaggs information (and of the resultant length for angles)."""
    null_i = np.empty((len(sh), sp.N))
    null_r = np.empty((len(sh), sp.N)) if centres is not None else None
    for j, s in enumerate(sh):
        R = rates(sums(sp, np.roll(fb, int(s)), K), occ, min_occ)
        null_i[j] = skaggs(R, occ)
        if centres is not None:
            null_r[j] = resultant(R, centres)[0]
    return null_i, null_r


def shifted_sums_fft(X: np.ndarray, fb: np.ndarray, K: int, sh: np.ndarray, chunk_bytes: float = 2e8) -> np.ndarray:
    """Activity sums per feature bin for every circular shift in sh at once (n_shuffles x K x N), by FFT
    cross-correlation of each feature-bin indicator with X. Same result as sums(sp, np.roll(fb, s), K)."""
    T, N = X.shape
    Xc = np.nan_to_num(np.clip(X, 0, None))
    out = np.empty((len(sh), K, N))
    lags = (-np.asarray(sh)) % T
    FI = np.fft.rfft(np.stack([(fb == k).astype(float) for k in range(K)], axis=1), axis=0)   # (F, K)
    nc = max(1, int(chunk_bytes / (16 * T)))
    for a in range(0, N, nc):
        FX = np.conj(np.fft.rfft(Xc[:, a:a + nc], axis=0))                                       # (F, nc)
        for k in range(K):
            c = np.fft.irfft(FX * FI[:, k:k + 1], n=T, axis=0)                                   # c[l] = sum_t X[t] I[t+l]
            out[:, k, a:a + nc] = c[lags]
    return out


def info_null_fft(X, fb, K, occ, min_occ, sh, centres=None, max_bytes: float = 2e8):
    """As info_null, through shifted_sums_fft (cheaper for dense activity or many shuffles); neurons in chunks so
    the (shuffles x K x neurons) sums stay under max_bytes."""
    N = X.shape[1]
    null_i = np.empty((len(sh), N))
    null_r = np.empty((len(sh), N)) if centres is not None else None
    nc = max(1, int(max_bytes / (8 * len(sh) * K)))
    for a in range(0, N, nc):
        SS = shifted_sums_fft(X[:, a:a + nc], fb, K, sh)
        for j in range(len(sh)):
            R = rates(SS[j], occ, min_occ)
            null_i[j, a:a + nc] = skaggs(R, occ)
            if centres is not None:
                null_r[j, a:a + nc] = resultant(R, centres)[0]
        del SS
    return null_i, null_r


def null_cost(sp: "Sparse", K: int, n_shuffles: int) -> str:
    """'fft' when the FFT route is cheaper than re-binning the non-zero entries once per shuffle."""
    bincount = sp.rows.size * n_shuffles * 3.0
    fft = K * sp.N * sp.T * max(np.log2(max(sp.T, 2)), 1) * 1.0
    return "fft" if fft < bincount else "bincount"


def speed_score(X: np.ndarray, v: np.ndarray, bin_size: float, smooth_s: float, sh: np.ndarray):
    """Kropff et al. (2015) speed score: Pearson r between the smoothed rate and speed, with the
    circular-shift null computed for every shift at once by FFT. Returns r, null (n x N)."""
    ok = np.isfinite(v)
    sig = smooth_s / bin_size
    Xs = gaussian_filter1d(np.nan_to_num(X.astype(np.float64)), sig, axis=0, mode="nearest") if sig > 0 else X.astype(np.float64)
    Xs = np.where(ok[:, None], Xs, np.nan)
    mu, sd = np.nanmean(Xs, axis=0), np.nanstd(Xs, axis=0)
    Z = np.where(ok[:, None], (Xs - mu) / np.where(sd > 0, sd, 1), 0.0)
    vz = np.where(ok, (v - np.nanmean(v)) / (np.nanstd(v) or 1), 0.0)
    n = ok.sum()
    r = (Z * vz[:, None]).sum(axis=0) / n
    T = len(v)
    # circular cross-correlation over the analysed bins: c[s] = sum_t Z[t] vz[(t + s) mod T]
    fz = np.fft.rfft(Z, n=T, axis=0)
    fv = np.fft.rfft(vz, n=T)
    cc = np.fft.irfft(np.conj(fz) * fv[:, None], n=T, axis=0)
    null = cc[np.asarray(sh) % T] / n
    r[sd == 0] = np.nan
    return r, null


def two_sided(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    return empirical_p(np.abs(obs), np.abs(null))[0]


def nan_smooth(y: np.ndarray, sigma: float, circular: bool = False) -> np.ndarray:
    if sigma <= 0:
        return y.copy()
    m = np.isfinite(y)
    mode = "wrap" if circular else "nearest"
    num = gaussian_filter1d(np.where(m, y, 0.0), sigma, mode=mode)
    den = gaussian_filter1d(m.astype(float), sigma, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~m] = np.nan
    return out


def place_fields(curve: np.ndarray, centres: np.ndarray, mean_rate: float, threshold: float, min_peak: float,
                 min_bins: int, max_fraction: float) -> list[dict]:
    """Contiguous runs of a (smoothed) 1D tuning curve above threshold x its peak, each with a peak at least
    min_peak and twice the mean rate, at least min_bins wide and at most max_fraction of the track."""
    c = np.nan_to_num(curve, nan=0.0)
    peak = c.max()
    if not np.isfinite(peak) or peak <= 0:
        return []
    above = c >= threshold * peak
    fields = []
    i, K = 0, len(c)
    width = centres[1] - centres[0] if K > 1 else 1.0
    while i < K:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < K and above[j]:
            j += 1
        seg = slice(i, j)
        pk = c[seg].max()
        n = j - i
        if pk >= max(min_peak, 2 * mean_rate) and n >= min_bins and n <= max_fraction * K:
            k = i + int(np.argmax(c[seg]))
            fields.append({"centre": float(centres[k]), "width": float(n * width), "peak_rate": float(pk),
                           "start": float(centres[i] - width / 2), "end": float(centres[j - 1] + width / 2)})
        i = j
    return fields


def split_half(sp_first: Sparse, sp_second: Sparse, fb1, fb2, K, bin_size, min_occ) -> np.ndarray:
    o1, o2 = occupancy(fb1, K, bin_size), occupancy(fb2, K, bin_size)
    R1 = rates(sums(sp_first, fb1, K), o1, min_occ)
    R2 = rates(sums(sp_second, fb2, K), o2, min_occ)
    ok = np.isfinite(R1[:, 0]) & np.isfinite(R2[:, 0]) if R1.shape[1] else np.zeros(K, bool)
    out = np.full(R1.shape[1], np.nan)
    if ok.sum() < 3:
        return out
    a, b = R1[ok], R2[ok]
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a * b).sum(axis=0) / den
    return out


def gaussian_p_upper(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-sided p of obs under a normal fitted to the null (for small shuffle counts)."""
    mu, sd = null.mean(axis=0), null.std(axis=0, ddof=1) if null.shape[0] > 1 else np.zeros(null.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (obs - mu) / sd
    p = norm.sf(z)
    p[~np.isfinite(z)] = np.nan
    return p
