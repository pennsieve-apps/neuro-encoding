# Third-party software

This app is Apache-2.0 (see `LICENSE`). The image it builds contains the following third-party
components, installed from PyPI at the pinned versions in `requirements.txt`; their licences apply
to the image. Nothing here is GPL. No model weights or data are bundled. jax and jaxlib are the
CPU builds that PyPI ships (no CUDA libraries in the image).

| Component | What | Authors / copyright | Licence | Source |
|---|---|---|---|---|
| NeMoS 0.2.9 | Poisson GLMs (`PopulationGLM`), basis functions (`RaisedCosineLinearEval`, `CyclicBSplineEval`, product bases). Called as a library. | Edoardo Balzani, Bence Bagi, Arnab Biswas, William F. Broderick, Sarah Jo Venditto, Caitlin Lewis, Camila Maura, Wolf De Wulf, Pablo Crespo Darriba, Guillaume Viejo, Pranati Modumudi, Eric Thomson, Alex Williams (Flatiron Institute, Center for Computational Neuroscience) | MIT | https://github.com/flatironinstitute/nemos |
| pynapple 0.11.4 | Occupancy-normalised tuning curves (`compute_tuning_curves`). | Guillaume Viejo, Adrien Peyrache and contributors (Peyrache lab, McGill University; Flatiron Institute) | MIT | https://github.com/pynapple-org/pynapple |
| JAX 0.12.0.1, jaxlib 0.12.0.1 (CPU) | Array computation and compilation (XLA) under NeMoS. | Google LLC and the JAX authors | Apache-2.0 | https://github.com/jax-ml/jax |
| optax 0.2.8, equinox 0.13.8, lineax 0.1.1, optimistix 0.1.0, jaxtyping 0.3.11, ml_dtypes 0.6.0, opt_einsum 3.4.0, absl-py, wadler_lindig | JAX libraries used by NeMoS (solvers, pytrees, typing). | Google DeepMind; Patrick Kidger and contributors; others | Apache-2.0 (optax, equinox, lineax, optimistix, ml_dtypes, absl-py), MIT (jaxtyping, opt_einsum, wadler_lindig) | PyPI |
| NumPy 2.5.4, SciPy 1.18.1 | Arrays, FFT shuffle nulls, statistics. | NumPy and SciPy developers | BSD-3-Clause | https://numpy.org, https://scipy.org |
| pandas 3.1.1 | Tables. | pandas developers | BSD-3-Clause | https://pandas.pydata.org |
| Matplotlib 3.11.2, Pillow 13.0.0 | Figures. | Matplotlib development team; Jeffrey A. Clark and contributors | Matplotlib licence (PSF-based); MIT-CMU | https://matplotlib.org, https://python-pillow.org |
| numba 0.68.1, llvmlite 0.50.0, xarray 2026.9.0, scikit-learn 1.9.1, pynwb 4.2.0, hdmf 6.2.0, h5py 3.16.0, neo 0.14.6, quantities 0.16.4, joblib, threadpoolctl, rich, tabulate, narwhals | Dependencies of pynapple (pynwb is imported by pynapple, not used here). | Their developers | BSD-2/3-Clause; Apache-2.0 (xarray); MIT (rich, tabulate, narwhals) | PyPI |
| jsonschema, referencing, rpds-py, ruamel.yaml, attrs, cloudpickle, contourpy, cycler, fonttools, kiwisolver, lazy-loader, packaging, pyparsing, python-dateutil, typing_extensions, tzdata, platformdirs, Pygments, markdown-it-py, mdurl | Dependencies of the above. | Their developers | MIT / BSD / Apache-2.0 / PSF-2.0 | PyPI |
| python:3.12.11-slim-bookworm | Base image (Debian 12 + CPython 3.12). | Python Software Foundation; Debian | PSF (Python); Debian packages under their own free licences | https://hub.docker.com/_/python |
