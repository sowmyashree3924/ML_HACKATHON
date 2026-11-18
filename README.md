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
`requirements-advanced.txt` for those optional experiments. The default
five-model comparison uses scikit-learn.

```text
python main.py eda --raw-root "dataset_raw/Student folder"
python main.py features --raw-root "dataset_raw/Student folder"
python main.py train --raw-root "dataset_raw/Student folder" --folds 5
python main.py evaluate --raw-root "dataset_raw/Student folder"
```

`all` runs the EDA inventory, training, and student-test prediction flow.
Evaluation defaults to `student_test`; evaluator-test data must remain separate
and must never be used for fitting or model selection. Training uses grouped
cross-validation by trip to avoid splitting observations from the same trip
across folds.

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
road segment. The sequence decoder uses classifier probabilities and endpoint
continuity as a compact Viterbi-style smoother. It is not a replacement for a
full shortest-path HMM and should be validated against route continuity before
deployment.

## Modules

* `src/parser.py`: validated whitespace table readers.
* `src/features.py`: metric projection, candidate generation, weak labeling,
  and trip-level graph/trajectory summaries.
* `src/model.py`: five scikit-learn candidate classifiers and optional PyG
  GraphSAGE scorer.
* `src/train.py`: grouped cross-validation, development comparison, and model
  checkpointing.
* `src/evaluate.py`: inference, sequence decoding, and submission construction.
* `notebooks/01_eda.ipynb`: starter notebook for inventory and visualization.

The optional GraphSAGE module is an architecture scaffold; the default CLI does
not train it. Results from the default trainer cover the five tabular models
only. Model selection is based on grouped CV observation-level top-1 accuracy,
with additional classification, log-loss, coverage, and student-test columns
recorded for review.

## API and browser interface

Train a checkpoint once, then launch the API and its built-in frontend from the
repository root:

```text
python -m pip install -r requirements.txt
python main.py train
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/`. The interface lists `student_test` trips, draws
the GPS trace, and displays the matched road segments and confidence after you
select **Reconstruct route**. The API documentation is at
`http://127.0.0.1:8000/docs`.

The service reads student-test trips from
`dataset_raw/Student folder/student_test` and loads
`models/pathsnap_model.joblib`. Set `PATHSNAP_RAW_ROOT`, `PATHSNAP_MODEL_PATH`,
or `PATHSNAP_OUTPUT_DIR` to override those locations. Endpoints include
`GET /api/health`, `GET /api/trips/`, `GET /api/trips/{trip_id}`, and
`POST /api/trips/{trip_id}/match`.

Local datasets, trained checkpoints, generated predictions, and virtual
environments are excluded from Git. Copy the data to the expected local folder
after cloning, then train a checkpoint before using the interface.
