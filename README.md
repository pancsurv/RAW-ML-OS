# RAW-ML-OS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20724219.svg)](https://doi.org/10.5281/zenodo.20724219)

Machine learning survival prediction models for pancreatic ductal adenocarcinoma (PDAC) following pancreatoduodenectomy.

This repository contains the analysis code accompanying the manuscript *"Development and External Validation of Machine Learning-Based Survival Prediction Models for Pancreatic Ductal Adenocarcinoma Following Pancreatoduodenectomy."* The models were developed on the international, multicentre Recurrence After Whipple's (RAW) study cohort (ClinicalTrials.gov NCT04596865) and externally validated on an independent single-centre cohort.

## Overview

Three survival models (Cox proportional hazards, random survival forest, and DeepSurv) are trained on a common set of 16 clinicopathological features and benchmarked against the AJCC TNM 8th edition and the Amsterdam prognostic model. A simplified integer risk score (the Pancreatic Survival Index, PSI) and a stacked ensemble meta-learner are also included, together with a risk-stratified CT surveillance cost analysis.

## Repository structure

```
internal/    Model development and internal validation (80/20 split of the RAW cohort)
  cox_internal.py            Cox PH with MICE imputation and Rubin's rules pooling
  rsf_internal.py            Random survival forest with MICE imputation and SHAP
  deepsurv_internal.py       DeepSurv neural network

external/    External / temporal validation on the independent cohort
  cox_external.py
  rsf_external.py
  deepsurv_external.py

shared/      Cross-model analyses and derived scores
  amsterdam_tnm_comparison.py   AJCC TNM and Amsterdam model benchmarking
  model_comparison_auc.py       Time-dependent AUC comparison with DeLong tests
  psi.py                        Pancreatic Survival Index and surveillance cost analysis
  stacked_ensemble.py           Random survival forest meta-learner over base models

sensitivity/
  sensitivity_no_chemo.py       Sensitivity analysis excluding adjuvant chemotherapy features
```

## Requirements

- Python 3.9
- Install dependencies with:

```
pip install -r requirements.txt
```

Key libraries: scikit-learn, scikit-survival, lifelines, statsmodels, PyTorch, torchtuples, pycox, SHAP.

## Configuration and run order

Input data are located via the `RAW_DATA_DIR` environment variable (default: the current directory). The scripts expect two CSVs in that directory: `Cleaned_Dataset_for_Analysis.csv` (development cohort) and `cambook_cleaned.csv` (external cohort).

```
set RAW_DATA_DIR=C:\path\to\data        # Windows
export RAW_DATA_DIR=/path/to/data       # Linux/macOS
```

The internal scripts must be run before the external ones: each `internal/` script fits the M imputation models and writes the fitted models, imputers, scalers, and training survival reference as artefacts in the working directory; the matching `external/` script then loads those artefacts. Run everything from a single working directory, e.g.:

```
python internal/cox_internal.py
python external/cox_external.py
```

## Data availability

The patient-level datasets are not included in this repository because they contain confidential clinical information and are governed by the respective ethics approvals (Greater Manchester South REC 20/NW/0397; Cambridge University Hospitals NHS Foundation Trust R&D A097432). Data may be available from the corresponding author on reasonable request and subject to the relevant data governance approvals.

## Reproducibility

All scripts use a fixed random seed (42). Missing data are handled with multiple imputation by chained equations (MICE, M = 10). The training-derived imputers and scalers are saved and applied to the test and external sets without refitting to prevent data leakage. Internal and external validation both pool predictions across all M imputations (risks are averaged; survival probabilities are averaged across the common training time grid).

## Citation

If you use this code, please cite the archived release:

> RAW-ML-OS: Machine learning survival prediction models for pancreatic ductal adenocarcinoma following pancreatoduodenectomy. Zenodo. https://doi.org/10.5281/zenodo.20724219

## License

Released under the MIT License. See [LICENSE](LICENSE).
