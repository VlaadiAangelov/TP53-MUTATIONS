# Predicting TP53 Mutation Status from Gene Expression

This repository implements an end-to-end ML Lab project for predicting TP53
mutation status from cancer cell-line gene expression profiles.

The default workflow uses DepMap/CCLE public data:

- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` for log2(TPM + 1) RNA-seq expression
- `OmicsSomaticMutations.csv` or `CCLE_mutations.csv` for TP53 mutation labels
- `Models.csv` or equivalent metadata for cancer lineage information

## Project Questions

1. Can TP53 mutant and wild-type samples be predicted from expression profiles?
2. Which classical ML models perform best?
3. Are curated TP53 target and p53 pathway gene sets informative?
4. Can expression distinguish broad mutation classes?
5. Do model-selected genes connect to p53 biology, apoptosis, DNA repair, or cell cycle control?

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

If you do not want a virtual environment, run:

```bash
python3 -m pip install -r requirements.txt
```

## Full Pipeline

```bash
python3 scripts/run_pipeline.py
```

Or step by step:

```bash
python3 scripts/00_download_data.py
python3 scripts/01_build_dataset.py
python3 scripts/02_eda.py
python3 scripts/03_train_binary.py
python3 scripts/04_train_multiclass.py
python3 scripts/05_interpret.py
```

Notebook versions are available in `notebooks/`:

- `main_notebook.ipynb` - the main step-by-step report notebook
- `00_run_pipeline.ipynb`
- `01_data_and_eda.ipynb`
- `02_model_results_and_interpretation.ipynb`

The DepMap release and filenames are configurable in `config/project.yaml`.
Older releases may use the previous expression filename
`OmicsExpressionProteinCodingGenesTPMLogp1.csv`.

## Outputs

- Processed matched expression matrix: `data/processed/expression_matched.csv.gz`
- TP53 labels: `data/processed/tp53_labels.csv`
- EDA figures: `reports/figures/`
- Model performance tables: `reports/tables/`
- Best fitted models: `models/`
- Report draft: `reports/final_report.md`

## Method Summary

The binary task labels every matched sample as TP53 mutant or wild-type. The
mutation-type task uses hotspot-aware broad classes: WT, Hotspot Missense, Other
Missense, Truncating, Other, and any in-frame indel class retained only if it
meets the minimum class-count threshold. Multiple TP53 mutations in a sample are
first resolved by severity:

1. Frameshift or nonsense
2. Splice, start lost, or stop lost
3. Missense
4. In-frame insertion/deletion
5. Synonymous
6. Other

Hotspot missense samples are defined using TP53 residues R175, G245, R248, R249,
R273, and R282. Nonsense and frameshift mutations are grouped as Truncating.

Feature selection and scaling are fitted only on training folds through sklearn
pipelines to avoid data leakage.
