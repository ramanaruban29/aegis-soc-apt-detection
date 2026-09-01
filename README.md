# aegis-soc-apt-detection
Hybrid Autoencoder + LightGBM APT detection framework for IIoT networks

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
