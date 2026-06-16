# RAW-ML-OS

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

## Data availability

The patient-level datasets are not included in this repository because they contain confidential clinical information and are governed by the respective ethics approvals (Greater Manchester South REC 20/NW/0397; Cambridge University Hospitals NHS Foundation Trust R&D A097432). Each script expects a local CSV path defined at the top of the file (e.g. `DATA_PATH` / `INTERNAL_CSV` / `EXTERNAL_CSV`); update these to point to your own data. Data may be available from the corresponding author on reasonable request and subject to the relevant data governance approvals.

## Reproducibility

All scripts use a fixed random seed (42). Missing data are handled with multiple imputation by chained equations (MICE, M = 10); the training-derived imputer and scaler are saved and applied to the test and external sets without refitting to prevent data leakage.

## License

Released under the MIT License. See [LICENSE](LICENSE).
