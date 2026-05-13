# Predicting TP53 Mutation Status from Gene Expression Profiles

An end-to-end machine learning project predicting TP53 mutation status — and mutation type — from RNA-seq gene expression data across two datasets: **CCLE** (cancer cell lines) and **TCGA** (primary tumours).

## Biological Background

TP53 is the most frequently mutated gene in human cancer (~50% of all tumours). As a master transcription factor, it controls hundreds of downstream genes involved in cell cycle arrest, apoptosis, and DNA repair. When TP53 is mutated, this transcriptional programme is disrupted, leaving a measurable fingerprint in RNA-seq data. This project asks whether that fingerprint is strong enough for a classifier to learn.

## Research Questions

1. Can TP53 mutant vs wild-type status be predicted from expression profiles?
2. Can the *type* of mutation (Missense, Nonsense, Frameshift, Splice, Other) be distinguished?
3. Do curated p53 pathway gene sets outperform genome-wide feature selection?
4. Does a larger, noisier pan-cancer dataset (TCGA) improve or hurt prediction?
5. Are the genes the model relies on biologically coherent with known p53 targets?

## Key Results

| Task | Dataset | Best model | Key metric |
|---|---|---|---|
| Binary (mutant vs WT) | CCLE | Random Forest / p53_pathway | ROC-AUC **0.906** |
| Binary (mutant vs WT) | TCGA | Hist. Gradient Boosting / top_3000 | ROC-AUC **0.921** |
| Multiclass (mutation type) | CCLE | Hist. Gradient Boosting / tp53_targets | Macro F1 **0.467** |
| Multiclass (mutation type) | TCGA | Linear SVM / p53_pathway | Macro F1 **0.368** |

**Main finding:** binary classification is strongly solvable (ROC-AUC 0.92). Multiclass mutation-type prediction is fundamentally limited by biology — all mutation types disrupt the same p53 transcriptional programme, making their expression signatures near-indistinguishable (~0.37 macro F1 ceiling regardless of model complexity).

## Repository Structure

```
.
├── config/
│   └── project.yaml              # Dataset paths, split sizes, feature config
├── data/
│   ├── raw/                      # Downloaded source files (not tracked)
│   └── processed/
│       ├── expression_matched.csv.gz   # CCLE matched expression matrix
│       ├── tp53_labels.csv             # CCLE mutation labels
│       └── tcga/                       # TCGA processed files
├── notebooks/
│   ├── tcga_main_notebook.ipynb  # Main report notebook (TCGA, primary deliverable)
│   ├── ccle_main_notebook.ipynb  # Main report notebook (CCLE)
│   ├── tcga_01_eda.ipynb         # TCGA raw data EDA
│   ├── tcga_02_eda_processed.ipynb  # TCGA processed data EDA + quality checks
│   ├── tcga_03_results.ipynb     # TCGA results summary
│   └── 03_advanced_eda_qc.ipynb  # Advanced QC: sparsity, outliers, p53 targets
├── scripts/
│   ├── 00_download_data.py       # Download CCLE/DepMap data
│   ├── 00_download_tcga.py       # Download TCGA data from UCSC Xena
│   ├── 01_build_dataset.py       # Build matched CCLE dataset
│   ├── 01_build_tcga_dataset.py  # Build matched TCGA dataset
│   ├── 02_eda.py                 # Exploratory data analysis
│   ├── 03_train_binary.py        # Binary classification (supports --data-dir, --tag)
│   ├── 04_train_multiclass.py    # Multiclass classification (supports --data-dir, --tag)
│   ├── 05_interpret.py           # Feature importance + differential expression
│   ├── 06_train_advanced_multiclass.py  # LightGBM + OvR (supports --feature-set)
│   ├── 07_train_twostage_multiclass.py  # Two-stage: binary then mutants-only multiclass
│   └── run_pipeline.py           # Run full CCLE pipeline
├── src/tp53_ml/
│   ├── config.py                 # YAML config loader
│   ├── data.py                   # DepMap download + label construction
│   ├── evaluation.py             # Metrics, ROC/PR curves, confusion matrices
│   ├── genes.py                  # Curated TP53 targets, p53 pathway, hotspot residues
│   ├── models.py                 # Classifier factories + sklearn pipeline builders
│   └── preprocessing.py         # TopVarianceSelector, GeneSetSelector
├── reports/
│   ├── figures/                  # All saved plots
│   ├── tables/                   # All saved CSV result tables
│   └── final_report.md           # Written report
├── requirements.txt
└── pyproject.toml
```

## Setup

**Python 3.10+ required.**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install lightgbm>=4.0   # for advanced multiclass models
```

## Running the Pipeline

### CCLE / DepMap (cancer cell lines)

```bash
# Download data (requires internet)
python scripts/00_download_data.py

# Build matched dataset
python scripts/01_build_dataset.py

# Binary classification
python scripts/03_train_binary.py

# Multiclass classification
python scripts/04_train_multiclass.py

# Advanced models (LightGBM)
python scripts/06_train_advanced_multiclass.py --feature-set p53_pathway

# Feature interpretation
python scripts/05_interpret.py
```

### TCGA (primary tumours)

Download the three files from [UCSC Xena](https://xena.ucsc.edu/) and place them in `data/raw/tcga/`:
- `tcga_RSEM_gene_tpm.gz` — expression
- `mc3.v0.2.8.PUBLIC.maf.gz` — somatic mutations
- `TCGA_phenotype_denseDataOnlyDownload.tsv.gz` — sample metadata

```bash
# Build matched TCGA dataset
python scripts/01_build_tcga_dataset.py

# Binary classification
python scripts/03_train_binary.py --data-dir data/processed/tcga --tag tcga

# Standard multiclass
python scripts/04_train_multiclass.py --data-dir data/processed/tcga --tag tcga

# Advanced models — test both feature sets
python scripts/06_train_advanced_multiclass.py \
    --data-dir data/processed/tcga --tag tcga --feature-set top_k_variable
python scripts/06_train_advanced_multiclass.py \
    --data-dir data/processed/tcga --tag tcga --feature-set p53_pathway

# Two-stage multiclass (mutants only)
python scripts/07_train_twostage_multiclass.py \
    --data-dir data/processed/tcga --tag tcga
```

## Notebooks

The primary deliverable is **`notebooks/tcga_main_notebook.ipynb`** — a self-contained report covering:

1. Dataset selection (CCLE vs TCGA) with side-by-side comparison
2. Data quality analysis — sparsity, outlier detection, canonical p53 targets
3. Class balance and expression distributions
4. PCA visualisation (TP53 status, mutation type, cancer type)
5. Cancer type analysis — TP53 mutation rates and mutation type profiles
6. Binary model training and evaluation (ROC-AUC 0.921)
7. Per-cancer-type performance breakdown
8. Multiclass model comparison — baseline, LightGBM variants, two-stage approach

To export to HTML:

```bash
jupyter nbconvert --to html notebooks/tcga_main_notebook.ipynb
```

## Methods

**Feature sets compared:**
- `top_3000_variable` — top 3,000 most variable genes across training samples
- `tp53_targets` — 33 known direct p53 transcriptional targets (MDM2, CDKN1A, BAX, ...)
- `p53_pathway` — ~60 genes including checkpoint, apoptosis, and DNA repair genes

**Models:** Logistic regression (L2, elastic net), Linear SVM, Random Forest, Extra Trees, Histogram Gradient Boosting, MLP, LightGBM (multiclass and one-vs-rest).

**Design principles:**
- Stratified 70% / 15% / 15% train / validation / test split
- Feature selection and scaling fitted inside sklearn pipelines on training data only (no data leakage)
- Model selection on validation set; test set touched exactly once
- Class-weighted loss for all models to handle imbalance
- Macro F1 and balanced accuracy as primary multiclass metrics (not overall accuracy)

## Datasets

| Dataset | Source | Samples | Expression | Notes |
|---|---|---|---|---|
| CCLE / DepMap 26Q1 | [DepMap](https://depmap.org) | 1,719 | log2(TPM+1) | Cancer cell lines |
| TCGA Pan-Cancer | [UCSC Xena](https://xena.ucsc.edu) | 9,701 | RSEM | Primary tumours only |

The two datasets use different normalisation scales and **cannot be mixed** without batch correction.
