# TP53-MUTATIONS

Reproducible baseline for predicting binary TP53 mutation status from CCLE gene expression.

## Project Layout

- `src/tp53_baseline/`: shared Python package for data loading, label creation, feature filtering, and ranking
- `scripts/run_baseline.py`: command-line entrypoint for the baseline feature-selection workflow
- `notebooks/tp53_baseline_exploration.ipynb`: local notebook version of the original Colab exploration
- `tests/`: targeted tests for data preparation and ranking logic
- `outputs/`: generated reports and ranked feature tables, kept out of git

## Setup

The repository already includes a local `venv/`. If you prefer a fresh environment, install the dependencies from `pyproject.toml` or `requirements.txt`.

Example editable install:

```bash
./venv/bin/pip install -e .
```

## Configure Local Data Paths

Keep the raw CCLE files outside git and point the pipeline at their local paths with environment variables:

```bash
export TP53_EXPRESSION_CSV="/Users/v_angelov/Downloads/CCLE_expression_full.csv"
export TP53_MUTATION_CSV="/Users/v_angelov/Downloads/CCLE_mutations.csv"
export TP53_OUTPUT_DIR="/Users/v_angelov/GroupProj/TP53-MUTATIONS/outputs/baseline"
export TP53_TOP_N="500"
export TP53_EXCLUDE_PREFIX="TP53"
```

You can also override each of these on the command line.

## Run The Baseline

```bash
./venv/bin/python scripts/run_baseline.py
```

This workflow will:

- load CCLE expression and mutation data
- derive binary TP53 mutation labels
- remove TP53-related expression columns before ranking
- compute univariate ranking metrics across all remaining genes
- save a dataset summary, the full ranking table, and the top-500 feature table

## Expected Baseline Counts

With the current CCLE files referenced above, the pipeline should reproduce:

- `1406` expression samples
- `882` TP53-mutant samples
- `524` TP53-wild-type samples

## Train The First Model

After generating `outputs/baseline/top_500_features.csv`, train the first exploratory model:

```bash
./venv/bin/python scripts/train_top500.py
```

This writes model artifacts to `outputs/models/top500_logistic/`:

- `model_metrics.json`: stratified dummy baseline and top-500 logistic regression metrics
- `test_predictions.csv`: held-out predictions and probabilities
- `logistic_coefficients.csv`: logistic regression coefficients ranked by absolute magnitude

The first model uses a stratified train/test split, a `DummyClassifier(strategy="stratified")`, and scaled logistic regression with balanced class weights. Because the top-500 features were selected before the train/test split, treat these first metrics as exploratory signal-checking rather than final unbiased performance.

## Notebook

The notebook at `notebooks/tp53_baseline_exploration.ipynb` mirrors the original Colab flow, but imports the shared project modules instead of duplicating logic.
