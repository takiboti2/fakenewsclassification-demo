# Fake news headline classification

Compare TF-IDF with logistic regression, BERT-base-uncased, and DistilBERT-base-uncased
on English headlines from WELFake. Models use headlines only; article bodies are
used for exploratory analysis. Labels are `0 = real` and `1 = fake`. Predictions
reflect dataset patterns and are not independent fact-checks.

## Data and preprocessing

`utils.py` handles loading, English-language filtering, headline cleanup,
deduplication, tokenization, and metrics. Cleanup removes configured source names,
publisher suffixes, media markers such as `(VIDEO)` and `(TWEETS)`, `Factbox`, and
URL-only headlines.

Cleaned headlines are deduplicated using Unicode, whitespace, and case normalization.
Conflicting labels are excluded. Stratified splits use seed `42`, with normalized
headline overlap checked within and across splits:

| Training (70%) | Validation (15%) | Test (15%) |
|---:|---:|---:|
| 42,914 | 9,196 | 9,196 |

Split CSVs contain `title` and `labels`. Exact-duplicate checks do not detect
semantic near-duplicates or guarantee generalization to unseen publishers.

## Setup and workflow

Use Python 3.13 or later. From the repository root, install the locked environment:

```bash
uv sync
```

Select that environment as the notebook kernel. Place the source dataset at
`data/WELFake_Dataset.csv`, then generate splits:

```bash
uv run python -c "from utils import split_data; split_data()"
```

`load_splits()` reuses existing CSVs and validates headline overlap. Use the same
splits for all experiments, and run the notebooks in this order:

1. **`eda.ipynb`** — inspect data quality, distributions, duplicates, and artefacts.
2. **`baseline_model.ipynb`** — tune the TF-IDF/logistic regression pipeline with
   cross-validation and save the estimator and validation metrics.
3. **`transformer_models.ipynb`** — configure BERT or DistilBERT, learning rates,
   batch size, and gradient accumulation; train and export epoch logs in the final cell.
4. **`evaluation.ipynb`** — compare validation metrics, select checkpoints by macro F1,
   evaluate the test set, and fit temperature scaling on validation predictions.

Model artifacts are saved under `models/`; validation metrics are saved under
`logs/`. Transformer logs combine training and validation metrics per epoch, while
the baseline log contains a single metric dictionary. Evaluation scores and plots
are available in `evaluation.ipynb`.

Source data, split CSVs, model weights, and the local environment are ignored by Git
and must be supplied or generated locally. The Gradio demo has a separate
[README](fake-news-demo/README.md).

## Tests

The tests use `unittest` and cover:

- **`test_review_fixes.py`** — headline cleanup, deduplication, split overlap,
  classification metrics, log loss, and label mappings.
- **`test_training_notebook.py`** — seed initialization, state saving, cleanup on
  failure or interruption, and epoch-log generation using mocked training.
- **`test_evaluation_notebook.py`** — loading baseline and transformer metrics into
  the validation dataframe without accessing model checkpoints.

Run them from the repository root:

```bash
uv run python -m unittest discover -s tests -v
```
