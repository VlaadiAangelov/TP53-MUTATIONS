# Predicting TP53 Mutation Status from Gene Expression Profiles Using Machine Learning

## 1. Introduction

TP53 is a tumor suppressor and transcription factor that regulates cell cycle
arrest, apoptosis, DNA repair, and cellular stress responses. It is one of the
most frequently mutated genes in cancer. Because transcription factors influence
the expression of downstream genes, TP53 mutation may leave a measurable
transcriptional signature in RNA-seq data.

This project asks whether gene expression profiles can predict TP53 mutation
status. The main supervised learning task is binary classification of TP53
mutant versus wild-type samples. A secondary task classifies broad TP53 mutation
types.

## 2. Data

The planned primary dataset is CCLE/DepMap cancer cell-line data. Expression
features come from protein-coding gene RNA-seq values in log2(TPM + 1) units.
Mutation labels are derived from DepMap somatic mutation calls.

The pipeline matches samples by DepMap model/sample identifier and keeps samples
with both expression measurements and mutation labels. Samples with at least one
TP53 mutation are labeled mutant; samples without a TP53 mutation entry are
labeled wild-type.

Pipeline output:

- `reports/tables/binary_class_balance.csv`
- `reports/tables/mutation_type_balance.csv`
- `reports/tables/expression_summary.csv`

The matched 26Q1 CCLE/DepMap dataset contains 1,719 cell-line samples and
19,215 protein-coding expression features. The binary label distribution is 997
TP53-mutant samples and 722 TP53-wild-type samples.

## 3. Methods

The workflow uses a stratified train/validation/test design. Preprocessing is
performed inside sklearn pipelines to prevent leakage: feature selection and
standardization are fitted on training data and applied to validation or test
data.

Feature sets:

- Top variable expression genes
- Curated TP53 target genes
- Curated p53 pathway genes

Binary models:

- Majority-class baseline
- L2 logistic regression
- Elastic-net logistic regression
- Linear SVM
- Random forest
- Extra Trees
- Histogram gradient boosting
- Shallow multilayer perceptron

Mutation-type models:

- Majority-class baseline
- L2 logistic regression
- Random forest
- Extra Trees

Binary evaluation metrics include accuracy, balanced accuracy, precision,
recall, F1, ROC-AUC, PR-AUC, and confusion matrix. Multi-class evaluation uses
macro F1, weighted F1, balanced accuracy, and confusion matrix.

## 4. Results

The binary model comparison selected a random forest trained on curated p53
pathway genes as the best validation model. On the held-out test set it reached:

- Accuracy: 0.829
- Balanced accuracy: 0.812
- Precision: 0.812
- Recall: 0.920
- F1-score: 0.863
- ROC-AUC: 0.906
- PR-AUC: 0.920

The mutation-type task was harder, as expected from class imbalance and the
biological heterogeneity of TP53 variants. The best validation model was a
random forest trained on TP53 target genes. On the held-out test set it reached:

- Accuracy: 0.715
- Balanced accuracy: 0.449
- Macro F1: 0.448
- Weighted F1: 0.690

Generated result files:

- `reports/tables/binary_validation_performance.csv`
- `reports/tables/binary_test_performance.csv`
- `reports/tables/multiclass_validation_performance.csv`
- `reports/tables/multiclass_test_performance.csv`
- `reports/figures/binary_best_model_roc.png`
- `reports/figures/binary_best_model_pr.png`
- `reports/figures/binary_confusion_matrix.png`
- `reports/figures/multiclass_confusion_matrix.png`

The strong performance of the pathway-based binary model supports the biological
hypothesis that TP53 mutation status is reflected in downstream transcriptional
patterns.

## 5. Biological Interpretation

The interpretation step performs differential expression between TP53 mutant and
wild-type samples and compares top differential genes with curated TP53 target
and p53 pathway genes. When the best binary model exposes coefficients or feature
importance values, the top model genes are also compared with these biological
gene sets.

The top genes from the best binary model were strongly p53-related. The highest
importance genes included `MDM2`, `ZMAT3`, `CDKN1A`, `BAX`, `SESN1`, `RRM2B`,
`SFN`, `TNFRSF10B`, `PHLDA3`, and `DRAM1`. Differential expression analysis also
found overlap with canonical TP53 targets, including `AEN`, `BAX`, `BBC3`,
`CDKN1A`, `DDB2`, `DRAM1`, `FAS`, `FDXR`, `MDM2`, `RRM2B`, `SESN1`, `SFN`,
`TNFRSF10B`, `TRIAP1`, and `ZMAT3`.

Generated interpretation outputs:

- `reports/tables/binary_best_model_top_genes.csv`
- `reports/tables/differential_expression_tp53_mutant_vs_wt.csv`
- `reports/tables/biological_overlap_summary.csv`
- `reports/figures/binary_top_genes.png`

## 6. Limitations

Cell lines are experimentally convenient but are simplified models of tumors.
Pan-cancer modeling may partly learn tissue or lineage differences rather than
only TP53 biology. Mutation labels may also be noisy because mutation consequence
does not always equal functional effect. RNA-seq is an indirect readout of TP53
activity, and downstream genes can be regulated by multiple transcription
factors.

## 7. Conclusion

TP53 mutation status was predictable from CCLE/DepMap expression profiles, with
the best binary model reaching ROC-AUC 0.906 on held-out samples. The strongest
features were biologically coherent p53 target and pathway genes, suggesting
that the model is capturing a real transcriptional signal rather than only a
generic high-dimensional pattern. Mutation-type prediction was feasible but much
weaker, indicating that expression more clearly separates mutant from wild-type
than it separates detailed TP53 variant classes.
