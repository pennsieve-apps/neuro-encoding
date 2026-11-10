# neuro-encoding

neuro-encoding is built on **[NeMoS](https://github.com/flatironinstitute/nemos)**, the GLM toolbox
developed by Edoardo Balzani, Guillaume Viejo, Alex Williams and colleagues at the Flatiron
Institute's Center for Computational Neuroscience ([software DOI 10.5281/zenodo.17553286](https://doi.org/10.5281/zenodo.17553286)),
on **[pynapple](https://github.com/pynapple-org/pynapple)** by Guillaume Viejo, Adrien Peyrache and
colleagues (Peyrache lab, Montreal Neurological Institute, McGill University, and the Flatiron
Institute; [Viejo et al. 2023, eLife](https://doi.org/10.7554/eLife.85786)), and on
**[JAX](https://github.com/jax-ml/jax)** (Bradbury, Frostig, Hawkins and colleagues, Google). The
measures it reports were introduced by others: spatial information (Skaggs, McNaughton and
colleagues; as used by [Markus et al. 1994](https://doi.org/10.1002/hipo.450040404)), place cells
([O'Keefe & Dostrovsky 1971](https://doi.org/10.1016/0006-8993(71)90358-1)), head-direction cells
([Taube, Muller & Ranck 1990](https://doi.org/10.1523/JNEUROSCI.10-02-00420.1990)), the speed score
([Kropff et al. 2015](https://doi.org/10.1038/nature14622)), Poisson GLMs of single neurons
([Pillow et al. 2008](https://doi.org/10.1038/nature07140)) and the false discovery rate
([Benjamini & Hochberg 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)). Pennsieve
packages these methods as a workflow step; it did not develop them. The demo data are DANDI
datasets by Grosmark & Buzsáki (000044), the International Brain Laboratory (000409) and Plusnin,
Pospelov et al. (002047); see [Citations](#citations).

A step of the **Neural Population Dynamics & Decoding** workflow, after `neuro-align`:

- **Neural Population Dynamics & Decoding:** `neuro-align → {neuro-encoding, neuro-embedding, neuro-decoding} → neuro-report`

**For a neuroscientist.** Which neurons care about which part of behaviour? For every neuron and
every behaviour variable neuro-align found (position, speed, head direction, wheel, pupil, paw and
nose speeds, ...), the app draws the tuning curve (how the firing rate changes with the variable)
and asks whether the neuron carries more information about the variable than chance. Chance is
measured, not assumed: behaviour is slid in time against the spikes by at least 20 s, 500 times,
which keeps the neuron's own burstiness and the animal's own movement statistics and destroys only
their relation. On position it finds place fields and draws the classic place-cell sequence (the
cells sorted by where they fire tile the track), including a version sorted on the first half of
the session and shown on the second, so you can see the sequence is real. Because variables are
correlated (on a linear track, speed depends on position), it also fits a model that uses all
variables at once and asks, per neuron, which variable the model cannot do without.

**For a computational analyst.** Per session and variable: occupancy-normalised tuning curves
(pynapple `compute_tuning_curves` on the binned activity, minimum occupancy `MIN_OCCUPANCY_S`;
cross-checked against our own binning), Skaggs information in bits per spike, its circular-shift
null (`N_SHUFFLES`, minimum shift `MIN_SHIFT_S`, seeded; computed by re-binning the non-zero
entries per shift or, when cheaper, for all shifts at once by FFT cross-correlation), one-sided
p = (1 + #null ≥ observed)/(1 + n); when the observed value beats every shuffle, p is the tail of a
normal fitted to the null (never more than 1/(n + 1)), so that with 500 shuffles the FDR over
hundreds of neurons can still reject; z-score; Benjamini-Hochberg q across neurons. Position: 1D (per running direction on a linear track
when both directions have ≥ 20% of the running time; running bins only) or 2D (`position_2d`)
rate maps; place fields as contiguous runs of the smoothed curve above baseline +
`FIELD_THRESHOLD` × (peak − baseline), 3 bins to 80% of the track, whose peak rises above the
baseline at least twice as much as the mean rate does (and is ≥ `FIELD_MIN_PEAK_HZ` for units).
For spikes the baseline is 0 (the classic rule: > 20% of peak, peak ≥ 2 × mean rate); for ROI
activity it is the curve's `FIELD_BASELINE_PCT` percentile, because deconvolved activity does not
fall to zero outside a field (on Sosa et al. it stays at 0.2-0.4 × peak along the whole track, so
the classic rule found no field at all); split-half stability (r between the tuning curves of the
first and second half). Place cell: significant spatial information and a field; for ROI data also
split-half stability ≥ `MIN_STABILITY`, because there the field rule finds a bump in almost every
ROI. Speed variables: Kropff speed score (Pearson r of the 250 ms-smoothed rate with speed),
two-sided shift null by FFT. Angles: mean resultant length of the tuning curve and preferred
direction, shift null. GLM: NeMoS `PopulationGLM` (Poisson, exp link, ridge `GLM_RIDGE`, LBFGS)
on counts summed into `GLM_BIN_S` bins; each covariate through `GLM_N_BASIS` raised-cosine basis
functions over its 1st-99th percentile (cyclic B-splines for angles, a product of raised cosines
for 2D position); 3 contiguous folds with a 2 s gap; cross-validated deviance explained D² of the
full model against the training-mean model; per covariate ΔD² = (dev(without c) − dev(full)) /
dev(mean), tested against `N_SHUFFLES_GLM` refits with only c circularly shifted. Every fit starts
from zero weights and the log mean rate (no warm starts), and fits that stop at `GLM_MAX_ITER` are
counted and reported as a warning. Checked on Grosmark: with 100 or 300 iterations every fit
stopped before NeMoS' convergence criterion, with 1000 all 123 converged (in the same wall time:
JAX compilation dominates there), and the significance calls at 100 and 1000 iterations differed
for 2 of 116 neuron × covariate pairs (both speed). On Sosa's deconvolved calcium the quasi-Poisson
fits do not meet the criterion even at 1000 iterations (118 of 123 report it), so treat ROI GLM numbers
as approximate. Large sessions pay for the iterations: lower `GLM_MAX_ITER` or set `GLM=false`
when time matters. p is empirical, floored at 1/(N + 1) (no fitted tail: with ~20
refits that would be extrapolation); `glm_<c>_significant` means ΔD² > 0 and larger than every
shifted refit, i.e. p = 1/(N + 1) = 0.05 at the default N = 19. That is a per-neuron test without
multiple-comparison correction, and no q is given (with 19 refits p cannot go below 0.05, so BH
could not reject anything). The FDR-controlled statements are the tuning statistics above. When
fewer than 90% of a session's fits converge, `summary.json` marks its GLM `approximate` (with
`converged_fraction`) and a warning says so.

## Ports

| Port | Direction | Content |
|---|---|---|
| `aligned` | in | neuro-align's output: `neuro-align/<session_id>/aligned.npz` + `manifest.json` (found anywhere under `$INPUT_DIR`) |
| `encoding` | out | `neuro-encoding/<session_id>/{<session_id>_encoding.csv, <session_id>_tuning_curves.csv, figures, summary.json}`, `neuro-encoding/{encoding.csv, summary.json, provenance.json}` |

## Parameters

| Name | Default | What it does |
|---|---|---|
| `VARIABLES` | `auto` | variables to analyse (`auto` = all but wheel position, licks, run direction; `position_2d` = x and y) |
| `N_SHUFFLES` | `500` | circular shifts for the tuning statistics |
| `MIN_SHIFT_S` | `20` | smallest shift (s) |
| `FDR` | `0.05` | Benjamini-Hochberg level, per statistic across the neurons of a session |
| `POSITION_BINS` / `POSITION_BINS_2D` / `N_BINS` / `ANGLE_BINS` | `40` / `20` / `12` / `36` | feature bins (1D position, 2D maps per side, other variables, angles) |
| `MIN_OCCUPANCY_S` | `0.2` | feature bins visited less are left out |
| `MIN_DURATION_S` | `60` | skip variables with less data |
| `PLACE_RUN_ONLY` | `true` | position analyses only while running |
| `SPLIT_DIRECTION` | `true` | linear track: also each running direction |
| `RUN_SPEED` | `auto` | running threshold when neuro-align gave no run direction (3 cm/s) |
| `SMOOTH_BINS` | `1.5` | smoothing of 1D curves for fields and figures (bins) |
| `FIELD_THRESHOLD`, `FIELD_MIN_PEAK_HZ`, `FIELD_MIN_BINS`, `FIELD_MAX_FRACTION` | `0.2`, `1.0`, `3`, `0.8` | place-field criteria |
| `FIELD_BASELINE_PCT` | `20` | ROI data: percentile of the tuning curve taken as its baseline for field detection |
| `MIN_STABILITY` | `0.5` | ROI data: a place cell also needs split-half stability (r between the halves' tuning curves) ≥ this |
| `SPEED_SMOOTH_S` | `0.25` | rate smoothing for the speed score (s) |
| `GLM` | `true` | fit the GLMs |
| `N_SHUFFLES_GLM` | `19` | shifted refits per covariate for the drop-one test (significance at p = 1/(N + 1)) |
| `GLM_BIN_S` | `0.1` | GLM bin (s) |
| `GLM_N_BASIS` | `8` | basis functions per covariate |
| `GLM_FOLDS` / `GLM_GAP_S` | `3` / `2` | cross-validation blocks and the gap around test blocks (s) |
| `GLM_RIDGE` | `0.001` | ridge strength |
| `GLM_MAX_KEYPOINTS` | `3` | at most this many keypoint speeds in the GLM |
| `GLM_MAX_ITER` | `1000` | LBFGS iterations per fit at most; fits stopped by it are counted in a warning |
| `GLM_MAX_NEURONS` | `150` | GLM only on the most active neurons (0 = all); the skipped ones are listed in `summary.json` and get empty GLM columns |
| `SEED` | `42` | seed of all shifts; results are reproducible |
| `THREADS` | `0` | BLAS/XLA threads; 0 = the container's CPU quota |

## Outputs

`<session_id>_encoding.csv`, one row per neuron: `session_id, neuron_id, region, label, cell_type,
depth, x, y, mean_rate` (from neuro-align), `epoch_mean_rate`; per variable `<v>`: `<v>_info`
(bits/spike; per unit activity for ROIs), `<v>_info_z`, `<v>_info_p`, `<v>_info_q`, `<v>_peak`
(variable value at the curve's maximum), `<v>_peak_rate`, `<v>_tuned`; position variables also
`<v>_n_fields`, `<v>_field_centres`, `<v>_field_widths` (`;`-separated), `<v>_stability`, and
`place_cell`, `place_direction` (`fwd`, `bwd`, `fwd;bwd` or `any`); speed variables
`<v>_speed_score`, `_p`, `_q`; angles `<v>_mrl`, `_p`, `_q`, `<v>_preferred_deg`; GLM `glm_d2`
and per covariate `glm_<c>_delta_d2`, `glm_<c>_bits_per_spike`, `glm_<c>_z` (against the shifted
refits), `glm_<c>_p`, `glm_<c>_significant` (see Method). Position variables also have
`<v>_informative` (significant spatial information, BH across neurons). `<v>_tuned`: position,
significant information and at least one field, and for ROI data also split-half stability ≥
`MIN_STABILITY`;
speed variables, significant speed score; angles, significant resultant length; others,
significant information. `<session_id>_tuning_curves.csv`: `neuron_id, variable, bin_centre, rate,
occupancy_s` (rate in Hz for units). `place_cell` is the OR of up to three separately
FDR-corrected tests (`position`, and on a linear track `position_fwd` and `position_bwd`), so the
family-wise false discovery rate of `place_cell` is up to about three times `FDR`. For ROI data
`<v>_n_fields` is ≥ 1 on almost every ROI (168 of 168 on Sosa, 10 of 10 untuned fixture ROIs:
noise bumps above the baseline), so the field rule does not separate place cells there; an ROI is
a place cell when its spatial information is significant and its tuning curves of the first and
second half of the session correlate at ≥ `MIN_STABILITY` (default 0.5: on the fixture the planted
place ROIs reach ≥ 0.97 and untuned ROIs ≤ 0.22; on Sosa the ROIs without significant information
have a median of 0.05). `summary.json` gives both counts (`n_spatially_informative` and
`n_tuned`). Figures (listed with captions in `summary.json`):

- `<session_id>_place_sequence.png`: place cells sorted by peak position, per running direction;
  right, the cross-validated version.
- `<session_id>_place_maps.png` (2D position): rate maps of the most informative neurons.
- `<session_id>_tuned_fraction.png`: fraction of significantly tuned neurons per variable and region.
- `<session_id>_tuning_examples.png`: the strongest tuning curves per variable.
- `<session_id>_glm.png`: per covariate, the fraction of neurons that need it and ΔD² per neuron.

## Demo results

Run on neuro-align's output of the demo data with the defaults, natively (arm64, 4-6 threads).
Grosmark, Sosa and Steinmetz were run with this version (Grosmark and Sosa on neuro-align 0.1.1
output). For IBL and the 002047 sessions the tuning results are from the previous version (the
changes since — ROI field rule, GLM p-values, iterations and `GLM_MAX_NEURONS` — do not touch them
for spike data, and the 002047 tuning test is information-based); their GLM results with the new
settings are not in yet, so they are left out here.

| Session | Neurons | Result |
|---|---|---|
| Grosmark & Buzsáki `sub-Buddy` (linear track, neuro-align 0.1.1 `EPOCH=run`, 57 laps, 267 s running) | 58 CA1 units (38 pyramidal, 20 interneurons) | **29 place cells, all pyramidal (29/38 = 76%)**: 15 with fields in both running directions, 10 in the runs towards lower position only, 4 towards higher only; median spatial information of place cells 1.15 bits/spike, median split-half stability 0.58; the place-cell sequence tiles the track in both directions and survives cross-validation (`_place_sequence.png`). Position is in the authors' track coordinate (neuro-align maps its linearisation onto it): the analysed running range is −0.13 to 1.83 m, i.e. the 1.6 m track plus the edges of the end platforms, and place-cell peaks lie between −0.13 and 1.82 m. GLM (all 123 fits converged): position needed (ΔD² beats all 19 shifted refits) for 28 pyramidal cells and 11 interneurons, speed for 19 neurons; median deviance explained 0.11 (pyramidal), 0.016 (interneurons). |
| Sosa, Plitt & Giocomo `sub-m11_ses-04` (DANDI 001361; 2p CA1, virtual linear track, neuro-align `EPOCH=run`, 80 laps, 702 s) | 168 accepted ROIs (deconvolved) | **130 spatially informative ROIs** (significant spatial information, BH; median q 1.8e-5) and **36 place cells (21%)**, the informative ROIs whose tuning is stable across the two halves of the session (split-half r ≥ 0.5; median 0.67; median field width 77 cm of the 450 cm track; 10 of the 36 peak in the first 50 cm). Field detection is relative to the curve's baseline, because deconvolved activity stays at 0.2-0.4 × peak along the whole track; for ROIs it finds a bump almost everywhere (168/168), so stability makes the call. Speed tuning in 101 ROIs (28 positive, 73 negative). The GLM fits (150 most active ROIs, quasi-Poisson) mostly did not converge (118 of 123), so their numbers are approximate and not reported here (they are in the table, marked in summary.json). |
| IBL `sub-NR-0031_ses-caa5dddc` (whole session) | 186 good units in 13 regions | Tuned to wheel velocity: 94 (61% in the anterior pretectal nucleus, 66% in the midbrain reticular nucleus, 42% in CA1); to pupil diameter 55; to paw or nose speed (Lightning Pose) 71-80. |
| Steinmetz et al. `sub-Richards` (DANDI 000017, whole session) | 490 units (ORB 257, root 134, MOs 92, OLF 7) | Tuned to wheel velocity: 309 (74% of MOs, 63% of ORB units); to pupil area 32. GLM on the 150 most active units (340 skipped, listed in summary.json; all 123 fits converged): wheel velocity needed for 54, pupil for 17; median cross-validated deviance explained −0.015, i.e. for the typical unit wheel and pupil predict held-out blocks no better than its mean rate (slow drift across the 50-minute session dominates). |
| CA1 1p calcium, WT and 5xFAD, 6 sessions (DANDI 002047; DRIADA calcium events) | 42-264 accepted ROIs per session | While running (speed ≥ 3 cm/s, 380-460 s per session), few ROIs carry significant 2D spatial information: 1, 4, 1, 2, 4 and 39 of 42-264 ROIs; speed-tuned 0-137; head direction 0-4. Without the running filter (`PLACE_RUN_ONLY=false`) 141 of 264 ROIs in sub-F14 are "spatial", which we take to be confounded by where the mouse rests; the defaults keep the conservative answer. Calcium events are sparse, so 6-8 minutes of running give little power per ROI. |

Synthetic fixture (`make test`, ground truth planted): on the linear track 30 of 30 place cells
found, peaks within 8 cm of the planted centres (median), one-direction cells in the right
direction, 8 of 8 speed cells with a significant positive speed score, no untuned cell called a
place cell; the GLM needs position for > 80% of place cells and speed for > 70% of speed cells. In
the arena, ≥ 8 of 10 2D place cells and 10 of 10 head-direction cells (preferred direction within
20°) are found. Calcium-like ROIs on a virtual track (activity on an offset outside the field):
15 of 15 place ROIs found, at most 1 of 10 untuned ROIs called a place cell.

## Runtime

`cpu: 4096` (4 vCPU), `memory: 8192` MiB, `timeoutSeconds: 10800`, `standard`. The timeout is
3× the slowest native run with the defaults (the six 002047 sessions, about 10 min each, about
60 min in all); memory leaves room above Steinmetz's 4.7 GB peak. Measured on 2026-11-09/10:
"native" rows ran on arm64 with 4-6 threads, "Docker" rows ran the image (linux/amd64 under
emulation on an M-series Mac, read-only root, `/tmp` tmpfs) while other jobs shared the host, so
they overstate real runtimes several-fold. Rows marked *previous version* used 10 GLM shifts and at
most 100 LBFGS iterations, without `GLM_MAX_NEURONS`. The real numbers will come from Pennsieve runs.

| Data | Where | Wall time | Peak memory |
|---|---|---|---|
| Grosmark `sub-Buddy` (58 units, 266 s running in a 2328 s window; GLM 2 covariates, 2572 bins, 123 fits, all converged) | Docker, `--cpus 4 --memory 6g` | 444 s (GLM 421 s) | 2.0 GB (cgroup `memory.peak`) |
| Grosmark, as above | native | 160 s (GLM 151 s) | 1.6 GB (process RSS) |
| Grosmark, `GLM=false` | Docker, `--cpus 4 --memory 6g` | 36 s | 0.84 GB |
| Sosa `sub-m11_ses-04` (168 ROIs; GLM on 150, 2 covariates, 5351 bins, 118 of 123 fits stopping at 1000 iterations) | native, 4 threads | 10.5 min (GLM 10.4 min) | 1.7 GB (process RSS) |
| IBL `sub-NR-0031_ses-caa5dddc` (186 units, 2002 s; GLM 5 covariates, 19,110 bins) | native, *previous version* | 18 min (GLM 16 min) | 1.4 GB (process RSS) |
| Steinmetz `sub-Richards` (490 units, 2955 s; GLM on 150, 2 covariates, 29,400 bins, 123 fits, all converged) | native, 4 threads | 20.1 min (GLM 19.7 min) | 4.7 GB (process RSS) |
| 002047, 6 calcium sessions (42-264 ROIs, 719 s each; GLM on ≤ 150, 6 covariates, 76 features) | native, 4 threads | 9-10.5 min per session for the first three (the run of six is about 60 min) | not yet measured with this version; 2.8 GB with the previous one |

An emulated Docker run of the IBL session was stopped after more than an hour (about 33 min per
GLM covariate under emulation); Docker runs of Steinmetz were killed when the shared Docker VM (7.7 GB
for several builders' containers) ran out of memory, so the Steinmetz and IBL figures here are
native. `GLM_MAX_ITER=100` or `GLM=false` keep large sessions short.

The tuning statistics take seconds to a minute per session; the GLM refits dominate (each of
`N_SHUFFLES_GLM` × `GLM_FOLDS` × covariates is a full NeMoS fit; under amd64 emulation JAX is
several times slower than on a native amd64 node). A multi-session run or a large session exceeds
the 15-minute Lambda limit, so this is a container step; with `GLM=false` a session takes under a
minute. It is Lambda-safe all the same: it writes only to `OUTPUT_DIR` and `$TMPDIR` (matplotlib
and numba caches under `$TMPDIR`, the font cache baked into `/opt/neuro-encoding/mpl`) and runs
with `docker run --read-only --user 65534:65534 --tmpfs /tmp` and `HOME`, `TMPDIR`, `INPUT_DIR`,
`OUTPUT_DIR` under `/tmp`. Image: 1.68 GB.

## Limitations

- Tuning is not causation, and a neuron "tuned" to a variable may be tuned to a correlated one.
  The GLM drop-one test separates variables only as far as they are not collinear, and only for the
  variables given; unmeasured ones (theta phase, reward expectation, ...) are invisible.
- The shift null keeps autocorrelation but assumes stationarity: slow drift in firing and in
  behaviour (e.g. the animal runs more at the start) can still make a neuron look tuned.
- With 500 shuffles the empirical p-value cannot go below 1/501; beyond the last shuffle a normal
  fitted to the null is used, which is an approximation in the far tail.
- Calcium data: information is computed on deconvolved activity (or dF/F clipped at 0) as if it
  were a rate; the GLM is a quasi-Poisson fit to non-negative activity, not a calcium model.
- Place-field criteria (thresholds, widths) are conventions; fields near the track ends, and in
  sparse 2D maps, depend on them. No spike history in the GLM.
- The GLM refits dominate the run time (each shifted covariate is a full refit, `N_SHUFFLES_GLM`
  × folds × covariates); `GLM=false` gives the tuning analyses in minutes.

## Running it locally

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python      # pytest on the synthetic fixture (no Docker)
make run PYTHON=.venv/bin/python       # build the image and run it over data/fixture/linked/
```

## Licences

Our code is Apache-2.0 (`LICENSE`). NeMoS and pynapple are MIT, JAX/jaxlib (CPU wheels) and the JAX
libraries under NeMoS Apache-2.0, NumPy, SciPy and pandas BSD-3-Clause, Matplotlib its PSF-based
licence. Nothing in the image is GPL. Details in `THIRD_PARTY.md`. The demo datasets are CC-BY-4.0.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in `CITATION.cff`.

- **NeMoS** (software): Balzani E, Bagi B, Biswas A, Broderick WF, Venditto SJ, Lewis C, Maura C,
  Crespo Darriba P, Viejo G, Modumudi P, Thomson E, Williams A. NeMoS: Neural ModelS. Zenodo 2026.
  doi:10.5281/zenodo.17553286.
- **pynapple**: Viejo G, Levenstein D, Skromne Carrasco S, Mehrotra D, Mahallati S, Vite GR,
  Denny H, Sjulson L, Battaglia FP, Peyrache A. Pynapple, a toolbox for data analysis in
  neuroscience. *eLife* 2023;12:e85786. doi:10.7554/eLife.85786, PMID 37843985.
- **JAX** (software, the citation its README asks for): Bradbury J, Frostig R, Hawkins P, Johnson
  MJ, Katariya Y, Leary C, Maclaurin D, Necula G, Paszke A, VanderPlas J, Wanderman-Milne S,
  Zhang Q. JAX: composable transformations of Python+NumPy programs. 2018.
  https://github.com/jax-ml/jax.
- **Spatial information**: Markus EJ, Barnes CA, McNaughton BL, Gladden VL, Skaggs WE. Spatial
  information content and reliability of hippocampal CA1 neurons: effects of visual input.
  *Hippocampus* 1994;4(4):410-421. doi:10.1002/hipo.450040404, PMID 7874233.
- **Speed score**: Kropff E, Carmichael JE, Moser MB, Moser EI. Speed cells in the medial
  entorhinal cortex. *Nature* 2015;523(7561):419-424. doi:10.1038/nature14622, PMID 26176924.
- **Head-direction cells**: Taube JS, Muller RU, Ranck JB Jr. Head-direction cells recorded from
  the postsubiculum in freely moving rats. I. Description and quantitative analysis. *J Neurosci*
  1990;10(2):420-435. doi:10.1523/JNEUROSCI.10-02-00420.1990, PMID 2303851.
- **Place cells**: O'Keefe J, Dostrovsky J. The hippocampus as a spatial map. Preliminary evidence
  from unit activity in the freely-moving rat. *Brain Res* 1971;34(1):171-175.
  doi:10.1016/0006-8993(71)90358-1, PMID 5124915.
- **GLMs of single neurons**: Pillow JW, Shlens J, Paninski L, Sher A, Litke AM, Chichilnisky EJ,
  Simoncelli EP. Spatio-temporal correlations and visual signalling in a complete neuronal
  population. *Nature* 2008;454(7207):995-999. doi:10.1038/nature07140, PMID 18650810.
- **FDR**: Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful
  approach to multiple testing. *J R Stat Soc Series B* 1995;57(1):289-300.
  doi:10.1111/j.2517-6161.1995.tb02031.x.
- **Demo data, linear track**: Grosmark AD, Buzsáki G. Diversity in neural firing dynamics supports
  both rigid and learned hippocampal sequences. *Science* 2016;351(6280):1440-1443.
  doi:10.1126/science.aad1935, PMID 27013730. Dataset: DANDI 000044,
  doi:10.48324/dandi.000044/0.261106.0236.
- **Demo data, IBL**: International Brain Laboratory et al. A brain-wide map of neural activity
  during complex behaviour. *Nature* 2025;645:177-191. doi:10.1038/s41586-025-09235-0,
  PMID 40903598. Dataset: DANDI 000409, doi:10.48324/dandi.000409/0.260309.1324.
- **Demo data, CA1 calcium in 5xFAD and WT mice**: Plusnin V, Pospelov N, DANDI 002047,
  doi:10.48324/dandi.002047/0.261106.0903.

## Acknowledgements

We thank the NeMoS and pynapple teams at the Flatiron Institute's Center for Computational
Neuroscience and the Peyrache lab, the JAX team, and the DANDI Archive. NeMoS is supported by the
Center for Computational Neuroscience in the Flatiron Institute of the Simons Foundation and by the
NIH BRAIN Initiative (1RF1MH133778), as its README asks us to acknowledge. The demo data were shared
by Grosmark and Buzsáki, the International Brain Laboratory, and Plusnin, Pospelov and colleagues,
under CC-BY-4.0.
