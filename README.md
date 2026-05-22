# Predicting TP53 Mutation Status from Gene Expression Profiles

An end-to-end machine learning project predicting TP53 mutation status — and mutation type — from RNA-seq gene expression data across two independent datasets: **CCLE** (cancer cell lines) and **TCGA** (primary tumours).

---

## Biological Background

TP53 is the most frequently mutated gene in human cancer (~50% of all tumours). As a master transcription factor, it controls hundreds of downstream genes involved in cell cycle arrest, apoptosis, and DNA repair. When TP53 is mutated, this transcriptional programme is disrupted, leaving a measurable fingerprint in RNA-seq data. This project asks whether that fingerprint is strong enough for a classifier to learn — and whether the *type* of mutation can also be distinguished.

---

## Research Questions

1. Can TP53 mutant vs wild-type status be predicted from expression profiles?
2. Can the *type* of mutation (Missense, Nonsense, Frameshift, Other) be distinguished?
3. Do curated p53 pathway gene sets outperform genome-wide feature selection?
4. Does a larger, noisier pan-cancer dataset (TCGA) improve or hurt prediction compared to cell lines (CCLE)?

---

## Key Results

| Task | Dataset | Best model | Feature set | Metric |
|---|---|---|---|---|
| Binary (mutant vs WT) | CCLE | Random Forest | p53_pathway | ROC-AUC **0.906** |
| Binary (mutant vs WT) | TCGA | Hist. Gradient Boosting | top_3000_variable | ROC-AUC **0.921** |
| Multiclass (mutation type) | CCLE | Hist. Gradient Boosting | tp53_targets | Macro F1 **0.467** |
| Multiclass (mutation type) | TCGA | Linear SVM | p53_pathway | Macro F1 **~0.37** |

**Main finding:** binary classification is strongly solvable on both datasets (ROC-AUC > 0.90). Multiclass mutation-type prediction is fundamentally limited by biology — all mutation types disrupt the same p53 transcriptional programme, making their expression signatures near-indistinguishable (~0.37–0.47 macro F1 ceiling regardless of model complexity).

---

## Repository Structure

```
.
├── config/
│   └── project.yaml                   # Dataset paths, split sizes, feature config
├── data/
│   ├── raw/                           # Downloaded source files (git-ignored)
│   └── processed/
│       ├── expression_matched.csv.gz  # CCLE expression matrix
│       ├── tp53_labels.csv            # CCLE mutation labels
│       ├── sample_metadata.csv        # CCLE sample metadata
│       └── tcga/                      # TCGA processed files (same structure)
├── notebooks/
│   ├── 0_tcga_00_pipeline.ipynb       # Run this first — trains all models
│   ├── MAIN_NOTEBOOK.ipynb            # Results verification and cross-dataset comparison
│   ├── ccle_main_notebook.ipynb       # CCLE deep-dive (EDA, binary, multiclass, interpretation)
│   ├── tcga_main_notebook.ipynb       # TCGA deep-dive (EDA, binary, multiclass)
│   └── ...                            # Additional EDA and QC notebooks
├── scripts/
│   ├── 00_download_data.py            # Download CCLE/DepMap data
│   ├── 00_download_tcga.py            # Download TCGA data from UCSC Xena
│   ├── 01_build_dataset.py            # Build matched CCLE dataset
│   ├── 01_build_tcga_dataset.py       # Build matched TCGA dataset
│   ├── 02_eda.py                      # Exploratory data analysis
│   ├── 03_train_binary.py             # Binary classification + hill-climb tuning
│   ├── 04_train_multiclass.py         # Multiclass classification + hill-climb tuning
│   ├── 05_interpret.py                # Feature importance + differential expression
│   └── run_pipeline.py                # Run full CCLE pipeline end-to-end
├── src/tp53_ml/
│   ├── config.py                      # YAML config loader
│   ├── data.py                        # DepMap download + label construction
│   ├── evaluation.py                  # Metrics, ROC/PR curves, confusion matrices
│   ├── genes.py                       # Curated TP53 targets and p53 pathway gene sets
│   ├── models.py                      # Classifier factories + sklearn pipeline builders
│   └── preprocessing.py              # TopVarianceSelector, GeneSetSelector
├── reports/
│   ├── figures/                       # All saved plots
│   └── tables/                        # All saved CSV result tables
├── requirements.txt
└── pyproject.toml
```

---

## Setup

**Python 3.10+ required.**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

---

## Running the Pipeline

Everything is orchestrated through `notebooks/0_tcga_00_pipeline.ipynb`. Open it in Jupyter and run cells top to bottom. The steps are:

### 1. Download Data

**CCLE** is downloaded automatically:
```bash
python scripts/00_download_data.py
```

**TCGA** is downloaded automatically:
```bash
python scripts/00_download_tcga.py
```

### 2. Build Datasets

```bash
python scripts/01_build_dataset.py          # CCLE
python scripts/01_build_tcga_dataset.py     # TCGA
```

### 3. Train Models

```bash
# CCLE — binary
python scripts/03_train_binary.py

# TCGA — binary
python scripts/03_train_binary.py --data-dir data/processed/tcga --tag tcga

# CCLE — multiclass
python scripts/04_train_multiclass.py

# TCGA — multiclass (Splice merged into Other for comparability with CCLE)
python scripts/04_train_multiclass.py --data-dir data/processed/tcga --tag tcga_nosplice --merge-splice
```

Both training scripts apply **cancer type filtering** and **hill-climb hyperparameter tuning** automatically — no extra flags needed.

### 4. View Results

Open `notebooks/MAIN_NOTEBOOK.ipynb` to see:
- Verification that cancer type filtering worked correctly
- Hyperparameter tuning gain (baseline vs tuned)
- Cross-dataset results comparison (CCLE vs TCGA)

---

## Methods

### Datasets

| Dataset | Source | Samples | Expression | Notes |
|---|---|---|---|---|
| CCLE / DepMap 26Q1 | [DepMap](https://depmap.org) | 1,719 | log2(TPM+1) | Cancer cell lines |
| TCGA Pan-Cancer | [UCSC Xena](https://xena.ucsc.edu) | 9,701 | RSEM | Primary tumours only |

The two datasets use different normalisation scales and cannot be mixed without batch correction.

### Feature Sets

| Name | Size | Description |
|---|---|---|
| `top_3000_variable` | 3,000 genes | Top most variable genes across training samples |
| `tp53_targets` | 33 genes | Known direct p53 transcriptional targets (MDM2, CDKN1A, BAX, ...) |
| `p53_pathway` | ~60 genes | Checkpoint, apoptosis, and DNA repair pathway genes |

### Models

Logistic Regression (L2, Elastic Net), Linear SVM, Random Forest, Extra Trees, Histogram Gradient Boosting, MLP.

### Design Principles

**Cancer type filtering:** Before splitting, cancer types with fewer than 7 samples are removed. The threshold of 7 is derived from the split sizes — with a 70/15/15 split, at least `ceil(1/0.15) = 7` samples per stratum are needed to guarantee representation in both validation and test sets. This also prevents models from exploiting cancer type identity as a shortcut, since TP53 mutation rates vary heavily by lineage.

**Hyperparameter tuning:** A coordinate hill-climb is run on the validation set (up to 3 rounds, minimum improvement of 1e-4). For each model, the two neighbouring values of each hyperparameter are evaluated and the best accepted. This is far more tractable than grid search on a 9,700-sample dataset while still capturing meaningful gains.

**No data leakage:** Feature selection and scaling are fitted inside sklearn pipelines on training data only. The test set is touched exactly once, after model selection on the validation set.

**Class imbalance:** All models use class-weighted loss. Macro F1 and balanced accuracy are the primary evaluation metrics — not overall accuracy, which is misleading on imbalanced data.

**Split:** Stratified 70% / 15% / 15% train / validation / test, stratified by cancer type × TP53 status × mutation type to ensure balanced representation across all splits.
