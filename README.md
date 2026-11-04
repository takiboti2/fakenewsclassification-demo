# Fake news headline classification

The notebooks train a TF-IDF baseline and transformer headline classifiers on WELFake.
Run them from the repository root using the environment in `pyproject.toml` / `uv.lock`.

The corrected pipeline normalizes headlines with Unicode NFKC, case folding, and whitespace
normalization, excludes headlines with conflicting labels, and keeps one row per headline
before a stratified 70/15/15 split. Bracketed video/videos markers (case-insensitive,
including parentheses, square/curly brackets, and whitespace variations) are removed
before language filtering and deduplication. Ordinary uses of the word video are kept. This prevents normalized duplicates crossing splits;
it does not detect semantic near-duplicates or guarantee source-independent evaluation.
Cached splits are checked for duplicate headlines on load.

The local splits were rebuilt from the union of the existing English-filtered splits:
42,914 training, 9,196 validation, and 9,196 test rows after marker and source cleanup. Conflicting headline groups were excluded during each cleanup. For a fresh dataset, put `WELFake_Dataset.csv` in `data/` and call
`utils.split_data()` to run preprocessing and generate splits.

## Reproduce results after the split correction

1. Run `baseline_model.ipynb` to retrain and overwrite the baseline model and metrics.
2. Run `DistilBERT_model.ipynb` for each desired backbone and learning rate. The training loop saves its final `trainer_state.json` in the run directory.
   Run the final notebook cell to export per-epoch lists to `logs/`, merging training
   losses and validation metrics and recording the configured learning rate. The final cell loops over `model_names` and `learning_rates`, loading the root
   `trainer_state.json` in each run directory and skipping missing files.
   Existing checkpoints from before data cleanup remain historical results; retrain
   before using their exported logs to compare corrected-split performance.
3. Run `evaluation.ipynb`. It selects the best run per backbone by validation macro F1
   and loads the recorded checkpoint and its tokenizer. Legacy logs without selection
   metadata are excluded. Train both BERT and DistilBERT to run all comparison cells.
4. Refit calibration and export the chosen model before updating the demo.

Existing model weights, legacy logs, and the demo were trained using the old overlapping
splits. They are not valid results for the corrected splits. Existing local checkpoint
and demo configuration label mappings have been repaired; weights were not retrained.
Notebook execution outputs were cleared to avoid presenting obsolete scores.
Data and model artifacts are ignored by Git and must be regenerated locally.

Run focused checks with `python -m unittest discover -s tests -v`.
See `fake-news-demo/README.md` for demo launch instructions.

Source names are removed case-insensitively using `SOURCE_IDENTIFIERS` in `utils.py`,
including Reuters, The New York Times, InfoWars, Associated Press, and other listed outlets.
Whitespace variants and empty attribution brackets are cleaned up. Ambiguous abbreviations
such as AP are intentionally excluded. Names are removed even when the outlet is the
subject of a headline; this is heuristic cleanup, not a guarantee of source independence.

Bracketed tweet/tweets and live video/videos markers are also removed, including
case, whitespace, hyphen/dash, and full-width variants. Unbracketed sentence words remain.

Bracketed AUDIO markers are removed with the same case, spacing, and bracket normalization.

Bracketed IMAGE/IMAGES markers are removed with the same case, spacing, and bracket normalization.

Bracketed DETAIL/DETAILS markers are removed with the same case, spacing, and bracket normalization.

High-priority artefact cleanup also removes bracketed SCREENSHOT/SCREENSHOTS and WATCH,
optional leading + markers, and allowlisted media combinations separated by /, +, &, or commas.
Unknown bracketed prose is preserved. `PUBLISHER_SUFFIXES` lists outlets removed only
as trailing attributions after a pipe or dash (for example `| Daily Wire` or `– TruthFeed`).
Arbitrary dash-separated headline content is retained. URL-only headlines beginning
with http://, https://, or www. are excluded; URLs embedded in prose are preserved.
These rules run before deduplication and split validation. Existing weights need retraining.

The standalone word Factbox is removed case-insensitively before deduplication.
