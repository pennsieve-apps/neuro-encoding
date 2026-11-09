"""Figures: place-cell sequence, fraction of tuned neurons per variable and region, example tuning curves,
2D place maps, GLM covariate contributions."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

DPI = 150
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = "Blues"

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def _norm_rows(M: np.ndarray) -> np.ndarray:
    mx = np.nanmax(M, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mx > 0, M / mx, 0.0)


def _smooth(M, sigma=1.5):
    from processor.stats import nan_smooth

    return np.stack([nan_smooth(r, sigma) for r in M])


def place_sequence(path: Path, sid: str, variables: list[dict], unit_word: str) -> tuple[Path, str] | None:
    pos = [v for v in variables if v["kind"] == "position" and v["name"].endswith(("_fwd", "_bwd"))] or \
          [v for v in variables if v["kind"] == "position"]
    if not pos:
        return None
    fig, axes = plt.subplots(len(pos), 2, figsize=(8, 2.2 + 2.6 * len(pos)), squeeze=False)
    counts = []
    for row, v in enumerate(pos):
        r = v["res"]
        tuned = np.flatnonzero(r["tuned"])
        counts.append((v["name"], len(tuned)))
        c = v["centres"]
        ext = [v["edges"][0], v["edges"][-1]]
        Rs = v["Rs"].T[tuned]
        R1, R2 = (h.T[tuned] for h in r["halves"])
        for col, (M, sorter, title) in enumerate((
                (Rs, Rs, "all data, sorted by peak"),
                (_smooth(R2), _smooth(R1), "cross-validated: second half, sorted by first half"))):
            ax = axes[row, col]
            if len(tuned) == 0:
                ax.text(0.5, 0.5, "no significant place cells", ha="center", va="center", transform=ax.transAxes,
                        color=INK2)
                ax.set_axis_off()
                continue
            peak = np.nanargmax(np.where(np.isfinite(sorter), sorter, -np.inf), axis=1)
            order = np.argsort(peak, kind="stable")
            im = ax.imshow(_norm_rows(np.nan_to_num(M[order])), aspect="auto", cmap=SEQ, vmin=0, vmax=1,
                           extent=[ext[0], ext[1], len(order), 0], interpolation="nearest")
            dir_word = {"_fwd": "runs to higher position\n", "_bwd": "runs to lower position\n"}
            prefix = next((w for k, w in dir_word.items() if v["name"].endswith(k)), "")
            if row == 0:
                ax.set_title(title, loc="left", fontsize=8)
            ax.set_ylabel(f"{prefix}place cells ({len(order)})" if col == 0 else "")
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.set_xlabel(f"position ({v['unit']})" if row == len(pos) - 1 else "")
        del c
    fig.suptitle(f"{sid}: place-cell sequence", x=0.01, ha="left", fontweight="bold", fontsize=10)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01) if counts and any(n for _, n in counts) else None
    if cb:
        cb.set_label("rate / peak rate")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    n_txt = ", ".join(f"{n}: {k}" for n, k in counts)
    cap = (f"{sid}: every significant place cell ({n_txt}) as one row, its {unit_word} along the track scaled to "
           f"its own peak, rows sorted by where the peak is. The diagonal band is the classic place-cell sequence: "
           f"together the cells tile the track. Right: the same cells on the second half of the session, sorted "
           f"by the first half; a diagonal that survives this cross-validation shows the fields are stable and not "
           f"an artefact of sorting noise.")
    return path, cap


def tuned_fraction(path: Path, sid: str, df, variables: list[dict], fdr: float) -> tuple[Path, str] | None:
    names = [v["name"] for v in variables]
    if not names:
        return None
    regions = df["region"].astype(str).fillna("unknown") if "region" in df else np.array(["all"] * len(df))
    u, c = np.unique(regions, return_counts=True)
    order = [r for _, r in sorted(zip(-c, u))]
    shown = order[:7] if len(order) > 8 else order
    groups = shown + (["other"] if len(order) > len(shown) else [])
    reg = np.where(np.isin(regions, shown), regions, "other")
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(names) * max(1, len(groups)) ** 0.5 + 2), 3.4))
    w = 0.8 / len(groups)
    x = np.arange(len(names))
    for gi, g in enumerate(groups):
        m = reg == g
        fr = [df.loc[m, f"{n}_tuned"].astype(bool).mean() if m.any() else 0 for n in names]
        col = SERIES[gi] if gi < len(SERIES) and g != "other" else OTHER
        ax.bar(x + (gi - (len(groups) - 1) / 2) * w, fr, width=w * 0.92, color=col,
               label=f"{g} (n={int(m.sum())})")
    ax.set_xticks(x, names, rotation=30 if len(names) > 4 else 0, ha="right" if len(names) > 4 else "center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of neurons tuned")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="region", title_fontsize=7)
    ax.set_title(f"{sid}: neurons significantly tuned to each variable", loc="left")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    cap = (f"{sid}: fraction of neurons in each brain region whose activity carries significant information about "
           f"each behaviour variable (against circular-shift nulls, Benjamini-Hochberg FDR < {fdr:g}). Place cells "
           f"also need a place field; speed variables use the speed score, angles the mean resultant length. "
           f"Compare regions: which areas encode what.")
    return path, cap


def tuning_examples(path: Path, sid: str, variables: list[dict], ids: list[str], unit_word: str,
                    n_cols: int = 5) -> tuple[Path, str] | None:
    vs = [v for v in variables if v["kind"] != "position2d" and not v["name"].endswith(("_fwd", "_bwd"))][:6]
    if not vs:
        return None
    fig, axes = plt.subplots(len(vs), n_cols, figsize=(1.8 * n_cols, 1.5 * len(vs) + 0.6), squeeze=False)
    for row, v in enumerate(vs):
        r = v["res"]
        z = np.nan_to_num(r["info_z"], nan=-np.inf)
        if "score" in r:
            z = np.nan_to_num(np.abs(r["score"]), nan=-np.inf)
        if "mrl" in r:
            z = np.nan_to_num(r["mrl"], nan=-np.inf)
        top = np.argsort(-z)[:n_cols]
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(top):
                ax.set_axis_off()
                continue
            j = top[col]
            y = v["R"][:, j]
            xs = np.rad2deg(v["centres"]) if v["kind"] == "angle" else v["centres"]
            ax.plot(xs, y, color=SERIES[0] if r["tuned"][j] else INK2, linewidth=1.2, marker="o", markersize=2)
            ax.set_title(f"{ids[j]}" + (" *" if r["tuned"][j] else ""), fontsize=7, loc="left")
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"{v['name']}\n{unit_word}", fontsize=7)
            if row == len(vs) - 1 or True:
                unit = "deg" if v["kind"] == "angle" else v["unit"]
                ax.set_xlabel(unit, fontsize=6)
    fig.suptitle(f"{sid}: strongest tuning curves per variable", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    cap = (f"{sid}: for each behaviour variable (rows), the {n_cols} neurons with the strongest tuning "
           f"({unit_word} against the variable's value, occupancy-normalised). Blue and * = significant after "
           f"FDR; grey = not significant (the strongest of a weak lot).")
    return path, cap


def place_maps_2d(path: Path, sid: str, var: dict, ids: list[str], n: int = 12) -> tuple[Path, str] | None:
    r = var["res"]
    order = np.argsort(-np.nan_to_num(r["info"], nan=-1))[:n]
    nx, ny = var["shape2d"]
    cols = 6
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.7 * rows + 0.4), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.set_axis_off()
        if i >= len(order):
            continue
        j = order[i]
        M = var["R"][:, j].reshape(nx, ny).T
        ax.imshow(M, origin="lower", cmap=SEQ, interpolation="nearest")
        ax.set_title(f"{ids[j]} {r['info'][j]:.2f} b" + (" *" if r["tuned"][j] else ""), fontsize=7)
    fig.suptitle(f"{sid}: rate maps of the most spatially informative neurons", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    cap = (f"{sid}: occupancy-normalised 2D rate maps (dark = high rate, white = unvisited or low) of the {len(order)} "
           f"neurons with the highest spatial information (bits per spike in the title; * = significant).")
    return path, cap


def glm_contributions(path: Path, sid: str, glm: dict, fdr: float) -> tuple[Path, str] | None:
    if not glm or "covariates" not in glm:
        return None
    names = list(glm["covariates"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    fr = [glm["covariates"][n]["significant"].mean() for n in names]
    axes[0].bar(names, fr, color=SERIES[0], width=0.6)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("fraction of neurons")
    axes[0].set_title("covariate needed (drop-one test)", loc="left", fontsize=8)
    data = [np.clip(glm["covariates"][n]["delta_d2"][np.isfinite(glm["covariates"][n]["delta_d2"])], -0.05, None)
            for n in names]
    bp = axes[1].boxplot(data, tick_labels=names, showfliers=False, patch_artist=True)
    for b in bp["boxes"]:
        b.set_facecolor(GRID)
        b.set_edgecolor(INK2)
    axes[1].axhline(0, color=INK2, linewidth=0.6)
    axes[1].set_ylabel("Δ deviance explained")
    axes[1].set_title("contribution per neuron", loc="left", fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle(f"{sid}: what the GLM needs to predict each neuron", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    d2 = np.nanmedian(glm["d2"])
    cap = (f"{sid}: Poisson GLMs (NeMoS) predicting each neuron from all behaviour variables together, "
           f"cross-validated on held-out time blocks (median deviance explained {d2:.3f}). Left: fraction of "
           f"neurons for which dropping a variable makes the prediction significantly worse than dropping a "
           f"circularly shifted copy of it (FDR < {fdr:g}); this separates variables that are only correlated "
           f"with each other (e.g. position and speed on a track). Right: how much each variable adds per neuron.")
    return path, cap
