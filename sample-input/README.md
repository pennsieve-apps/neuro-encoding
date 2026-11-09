# sample-input

neuro-encoding reads neuro-align's output (port `aligned`): `neuro-align/<session_id>/aligned.npz`
with its `manifest.json`, found anywhere under `$INPUT_DIR`.

## Synthetic fixture (no download)

```sh
make fixture PYTHON=.venv/bin/python   # data/fixture/{linked,flat}/ + data/fixture/ground_truth.json
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked/ -> data/output/
```

The fixture is written in neuro-align's format directly (no NWB needed):

- `sub-enc01_ses-track`: 400 s on a 200 cm linear track; 30 place cells (10 of them fire in one
  running direction only), 8 speed cells, 12 untuned cells. Expected: about 30 place cells with
  peaks near the planted centres and the one-direction cells found in their direction; the 8 speed
  cells with a significant positive speed score; the GLM needs position for place cells and speed
  for speed cells; untuned cells not flagged.
- `sub-enc02_ses-arena`: 400 s in a 1 m open field; 10 2D place cells, 10 head-direction cells,
  10 untuned. Expected: the place cells significant in `position_2d`, the head-direction cells
  significant with preferred directions within 20 degrees of the planted ones.

## Real data

Run neuro-align first (see its `sample-input/README.md`) on any NWB session with behaviour, e.g. the
Grosmark & Buzsáki 2016 linear track (DANDI 000044, the place-cell sequence demo), the IBL Brain Wide
Map session (DANDI 000409), or the CA1 1p calcium open-field sessions (DANDI 002047), then:

```sh
docker run --rm --platform linux/amd64 --network none --memory 8g --cpus 4 \
  -e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output \
  -v "$PWD/align-out:/data/input/aligned:ro" -v "$PWD/out:/data/output" neuro-encoding
```
