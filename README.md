# BrainFMBench

A living benchmark for **brain MRI foundation models**. Contributors submit a
model; its frozen features are extracted on neuroimaging datasets (HBN, NKI) and scored
on downstream tasks; the leaderboard updates automatically.

The evaluation follows a simple principle: extract frozen features, then probe them on downstream tasks (sex, age, BMI) with no fine-tuning, measuring how well a model's representations transfer.

## Leaderboard

See [`LEADERBOARD.md`](LEADERBOARD.md) for the current ranking and the figure. Scores are downstream RandomForest probes of frozen
features (5-fold CV across 5 seeds for the box, held-out test for the points).

## Tasks & data

Models are evaluated on two cohorts from the Reproducible Brain Charts (RBC)
initiative:

- **NKI** — Nathan Kline Institute Rockland Sample (~958 subjects)
- **HBN** — Healthy Brain Network (~1000 subjects)

across three tasks: **sex** classification (balanced accuracy) and **age** / **BMI**
regression (MAE). Preprocessing is turboprep by default (cat12 also available).

## How it works

BrainFMBench splits the work between the cluster and CI:

```
  contributor PR                     Compute Canada (rorqual)          GitHub CI
  ─────────────                      ────────────────────────         ─────────
  model.yaml                                                          validate PR
  extract.py        ── merge ──▶     extract frozen features    ──▶   score features
  weights.txt                        (on preprocessed data)               │
                                                                       leaderboard
                                                                       updates
```

- **Feature extraction runs on the cluster**, against preprocessed data. Only the resulting feature vectors come back.
- **Scoring runs in CI**, publicly and reproducibly, so anyone can verify how a
  leaderboard number was produced.

The cluster is reached through a fixed-IP jump server and a constrained
automation key (Compute Canada's supported automation-node path), so CI can
submit and retrieve jobs.

## Repository layout

```
models/<name>/          submitted models (model.yaml + features, or + extract.py)
labels/<DS>.csv         shared task labels per dataset (subject_id, sex, age, bmi)
eval/scorer.py          the downstream probing protocol
scripts/                scoring, leaderboard, and submission validation
cluster/                the async cluster runner (reap/sow) + sbatch template
example-submission/     a minimal, working submission you can copy
.github/workflows/      validation, scoring, and cluster-extraction workflows
```

## Contributing a model

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide. In short, open a PR
adding `models/<your-model>/` with:

- **`model.yaml`** — metadata (name, preprocessing, embedding_dim, datasets, tasks)
- **`extract.py`** — defines `extract(input_dir, output_csv, weights_dir)`
- **`weights.txt`** — direct-download URL(s) to your checkpoint(s) (HuggingFace,
  Zenodo, etc.)

Copy [`example-submission/`](example-submission/) as a starting point. A PR
validation check runs your `extract.py` on a synthetic volume before anything
touches the cluster. Because extraction runs on our allocation, only
maintainer-reviewed submissions are executed.

## Scoring protocol

For each model / dataset / task, at the full training size:

- **Box** = 5-fold cross-validation across 5 seeds (25 values)
- **Points** = held-out test result for the same 5 seeds
- RandomForest probe (200 trees, depth 6) on standardized features; balanced
  accuracy for sex, MAE for age / BMI.

Dependencies are pinned (scikit-learn 1.7.1, numpy 2.4.2, pandas 2.2.3) so the
leaderboard is reproducible run to run.
