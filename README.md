# AEGIS SOC — Hybrid Autoencoder + LightGBM APT Detection Framework

Companion code repository for the paper *"A Hybrid Two-Stage Autoencoder and LightGBM Machine Learning Framework for APT Detection in IIoT Networks"* (Rangan & Mascrenghe).

This repository contains the training, retraining, and evaluation scripts used to produce the results reported in the paper, including both ablation studies (fused anomaly-score removal, and port-identity feature removal).

## Overview

The detection pipeline is a two-stage hybrid architecture:

1. **Stage 1 — Autoencoder.** A dense, symmetric encoder-decoder trained exclusively on benign traffic, producing a reconstruction-error anomaly score for each network flow.
2. **Stage 2 — LightGBM.** A gradient-boosted classifier that receives the Stage 1 anomaly score as an explicit fused input feature, alongside the raw flow features, rather than using it as a pre-filter.

Full architectural and methodological details are in the paper's Methods section.

## Dataset

This work uses **CICAPT-IIoT Phase 2**, created by the Canadian Institute for Cybersecurity, University of New Brunswick:

> Ghiasvand, E. et al. (2026) "Resilience against APTs: a provenance-based IIoT dataset for cybersecurity research," in *Mobile and Ubiquitous Systems: Computing, Networking and Services (MobiQuitous 2024)*. Springer Nature Switzerland, Cham, pp. 121–144. https://doi.org/10.1007/978-3-032-10554-7_7

The dataset is not redistributed in this repository. Obtain it from the original source and place the CSV at the path referenced in each script's `CSV` constant before running.

## Repository Structure

06_training/ Original 64-feature (with-ports) pipeline
01_train_autoencoder.py Stage 1 Autoencoder training (benign-only)
02_generate_stage2_features.py Feature fusion + stratified 60/20/20 split
03_train_lgbm.py Stage 2 LightGBM training (50:1 undersampling)
04_evaluate.py Held-out test set evaluation

07_retrain_noports/scripts/ Port-identity ablation (62-feature, no-ports) pipeline
01_train_autoencoder.py Stage 1 retrain, ports excluded from raw features
02_generate_stage2_features.py
split_memsafe.py Memory-safe stratified split (fallback for large datasets)
03_train_lgbm.py
04_evaluate.py

03_evaluation/ Analysis and comparison scripts
evaluate_ablation_models.py Fused-anomaly-score ablation (Table: feature fusion)
baseline_comparison.py Standard classifier comparison (Random Forest, XGBoost, etc.)
generate_plots.py Confusion matrix, ROC, and Precision-Recall curve generation
phase_breakdown_drift_latency.py Per-attack-phase error analysis, latency benchmarking


## Reproducing the Results

1. Obtain the CICAPT-IIoT Phase 2 dataset and update the `CSV` path constant in each script.
2. Run the pipeline in order within either `06_training/` (original, with ports) or `07_retrain_noports/scripts/` (ablated, without ports):

python3 01_train_autoencoder.py
python3 02_generate_stage2_features.py
python3 03_train_lgbm.py
python3 04_evaluate.py

3. Run the analysis scripts in `03_evaluation/` against the trained model artefacts to reproduce the ablation tables, baseline comparison, and figures reported in the paper.

## Key Results

| Configuration | Features | Recall | FPR | MCC |
|---|---|---|---|---|
| With ports (original) | 64 | 85.57% | 0.0025% | 0.8198 |
| Without ports (this work) | 62 | 87.88% | 0.0022% | 0.8453 |

Full results, including the fused-anomaly-score ablation and full-stream production evaluation, are reported in the paper.

## Requirements

- Python 3.9+
- TensorFlow/Keras 2.20.0
- LightGBM 4.3.0
- scikit-learn 1.4.2
- pandas, numpy, joblib

## Citation

If you use this code, please cite the associated paper (citation details to be added upon publication).

## License

License to be finalised prior to publication.
