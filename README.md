# PathSnap map matching pipeline

This repository contains a modular pipeline for matching noisy GPS observations
to the road network supplied for each PathSnap trip. It reads `.track`, `.nodes`,
`.arcs`, and `.route` files, creates candidate road segments, trains candidate
rankers, and emits an observation-level CSV of matched arcs.

## Dataset layout

Point `--raw-root` at the directory with `train/` and `student_test/` children.
Each trip directory is expected to contain one file of each extension:

* `.track`: longitude, latitude, timestamp in seconds.
* `.nodes`: longitude, latitude; row number is the zero-based node ID.
* `.arcs`: from-node ID, to-node ID; row number is the arc ID.
* `.route`: chronological arc IDs. It is used to make weak training labels and,
  for permitted development evaluation, reference labels.

The supplied route is a reference, not an independently verified label. For
each GPS point this baseline assigns the closest route arc within 75 m, then
labels generated nearby candidates against that arc. Candidate-miss rate and
distance filtering are recorded; interpret scores with this weak-supervision
limitation in mind. Route-derived distance is excluded from model features.

## Setup and commands

Python 3.10+ is recommended. Install the core packages with
`python -m pip install -r requirements.txt`. Torch and PyTorch Geometric are
only needed for the optional GraphSAGE architecture in `src/model.py`; install
`requirements-advanced.txt` for those optional experiments. The default final
fit uses scikit-learn's Random Forest; `--compare-models` enables the full
five-model comparison.

```text
python main.py eda --raw-root "dataset_raw/Student folder"
python main.py features --raw-root "dataset_raw/Student folder"
python main.py train --raw-root "dataset_raw/Student folder" --selected-model random_forest
python main.py evaluate --raw-root "dataset_raw/Student folder"
```

Pass `--training-csv "C:/path/to/PathSnap_fast_training_data.csv"` to train
directly from the compact candidate table. It normalizes the three available
features (point-to-segment distance, road length, and heading difference) to
the same columns used during API inference. The default `train` command fits
the selected Random Forest after the model comparison. Use `--compare-models`
to run all five candidates through trip-grouped CV and select by CV macro F1.

`all` runs the EDA inventory, training, and student-test prediction flow.
Evaluation defaults to `student_test`; evaluator-test data must remain separate
and must never be used for fitting or model selection. Cross-validation groups
by trip to avoid splitting observations from the same trip across folds.

## Outputs

* `data/processed/*_candidates.csv`: candidate-level engineered features and
  weak labels.
* `experiments/model_comparison.csv`: CV and student-test model metrics. This
  file is generated from actual runs; no metrics are prefilled.
* `models/pathsnap_model.joblib` and `models/model_metadata.json`: selected
  fitted model and its feature/label definition.
* `submissions/submission.csv`: concatenated matched-arc predictions, with one
  output row per GPS observation. Per-trip CSVs are also written.

The submission includes trip/observation IDs, timestamps, raw GPS coordinates,
matched arc ID, confidence, and coordinates for both endpoints of the selected
road segment. The Viterbi decoder uses classifier probabilities as emissions
and directed, distance-weighted shortest paths through the road graph as
transitions. It also writes each trip's expanded reconstructed arc sequence to
a separate `*_route.csv` file.

## Modules

* `src/parser.py`: validated whitespace table readers.
* `src/features.py`: metric projection, candidate generation, weak labeling,
  and trip-level graph/trajectory summaries.
* `src/model.py`: five scikit-learn candidate classifiers and optional PyG
  GraphSAGE scorer.
* `src/train.py`: grouped cross-validation, development comparison, and model
  checkpointing. The default CLI fits the selected Random Forest; use
  `--compare-models` to rerun the five classifiers and select by grouped CV
  macro F1.
* `src/evaluate.py`: inference, sequence decoding, and submission construction.
* `notebooks/01_eda.ipynb`: starter notebook for inventory and visualization.

The optional GraphSAGE module is an architecture scaffold; the default CLI does
not train it. A full comparison selects by grouped CV candidate macro F1 and
records log-loss, observation-level top-1, coverage, and student-test metrics.
After a comparison, the selected model can be fitted directly without
repeating CV.

## API and browser interface

Train a checkpoint once, then launch the API and its built-in frontend from the
repository root:

```text
python -m pip install -r requirements.txt
python main.py train --training-csv "C:/path/to/PathSnap_fast_training_data.csv"
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/`. The interface lists `student_test` trips, draws
the GPS trace, and displays the reconstructed connected road path and per-point
confidence after you select **Reconstruct route**. It also shows per-trip
candidate macro-F1, observation top-1 accuracy, log loss, candidate miss rate,
and a 0.5-threshold confusion matrix. These development metrics are calculated
after inference against the student-test route-derived weak labels; the route
is not passed into model prediction. The API documentation is at
`http://127.0.0.1:8000/docs`.

The service reads student-test trips from
`dataset_raw/Student folder/student_test` and loads
`models/pathsnap_model.joblib`. Set `PATHSNAP_RAW_ROOT`, `PATHSNAP_MODEL_PATH`,
or `PATHSNAP_OUTPUT_DIR` to override those locations. Endpoints include
`GET /api/health`, `GET /api/trips/`, `GET /api/trips/{trip_id}`, and
`POST /api/trips/{trip_id}/match`.

### Two-laptop setup

Keep the dataset and trained model on Laptop 1. Find its LAN IPv4 address with
`ipconfig`, then allow the frontend's exact origin and bind the API to the LAN
interface (replace the sample address with Laptop 2's IPv4 address):

```powershell
$env:PATHSNAP_CORS_ORIGINS = "http://192.168.1.25:5500"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

On Laptop 2, serve the `frontend` folder from the project checkout:

```powershell
python -m http.server 5500 --bind 0.0.0.0 --directory frontend
```

Open `http://127.0.0.1:5500/`, expand **Connection settings**, enter
`http://<Laptop-1-IPv4>:8000`, and press **Connect**. Both machines should be
on the same reachable network. The browser UI stores that API address locally.

Local datasets, trained checkpoints, generated predictions, and virtual
environments are excluded from Git. Copy the data to the expected local folder
after cloning, then train a checkpoint before using the interface.
