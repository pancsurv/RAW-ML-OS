import pandas as pd
import numpy as np
import torch
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss
from torchtuples.practical import MLPVanilla
from torchtuples.optim import Adam
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score
from sksurv.util import Surv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_PATH = os.path.join(os.environ.get("RAW_DATA_DIR", "."), "Cleaned_Dataset_for_Analysis.csv")
M         = 10
MAX_ITER  = 10
BASE_SEED = 42
EPOCHS    = 50

SELECTED_FEATURES = [
    'age', 'bmi', 'asa', 'adjchemo',
    'log_tumoursize', 'histoT', 'histoN', 'differentiation',
    'LNR', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR',
    'asa_age', 'adjchemo_LNR'
]
RAW_COLS = ['age', 'bmi', 'asa', 'adjchemo',
            'histotumoursize', 'histoT', 'histoN', 'differentiation',
            'totnodes', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR']

BEST_PARAMS = {
    "lr": 0.0001839, "num_nodes": [128, 64],
    "dropout": 0.4485, "weight_decay": 0.0002721, "batch_size": 32
}

print("=" * 70)
print("DEEPSURV - MICE IMPUTATION (M=10) + AVERAGED PREDICTIONS")
print("=" * 70)

data = pd.read_csv(DATA_PATH, na_values=['#NUM!'])
data['event'] = data['alive_dead'].map({1: 0, 2: 1, 3: 0})
data['adjchemo'] = data['adjchemo'].map({1: 1, 2: 0, 3: np.nan})
for _col in ['pancfatinvasion', 'perineuralinvasion', 'microvascinvasion',
             'lymphinvasion', 'adjrad', 'recurrence_yn',
             'recurrence_within_6_months']:
    if _col in data.columns:
        data[_col] = pd.to_numeric(data[_col], errors='coerce')
        data[_col] = data[_col].map({1: 1, 2: 0, 3: float('nan')})
if 'differentiation' in data.columns:
    data['differentiation'] = pd.to_numeric(data['differentiation'], errors='coerce')
    data.loc[data['differentiation'] == 6, 'differentiation'] = float('nan')
if 'histoT' in data.columns:
    data['histoT'] = pd.to_numeric(data['histoT'], errors='coerce')
    data.loc[data['histoT'] == 5, 'histoT'] = float('nan')
if 'histoN' in data.columns:
    data['histoN'] = pd.to_numeric(data['histoN'], errors='coerce')
    data.loc[data['histoN'] == 2, 'histoN'] = float('nan')
if 'histoM' in data.columns:
    data['histoM'] = pd.to_numeric(data['histoM'], errors='coerce')
    data.loc[data['histoM'] == 2, 'histoM'] = float('nan')

MAX_FUP = 60
data.loc[data['OS_months'].isna() & (data['event'] == 0), 'OS_months'] = MAX_FUP
data = data.dropna(subset=['OS_months', 'event'])

over_fup = data['OS_months'] > MAX_FUP
data.loc[over_fup, 'event']     = 0
data.loc[over_fup, 'OS_months'] = MAX_FUP
initial_n = len(data)
data = data[~((data['event'] == 1) & (data['OS_months'] <= 3))]
print(f"Excluded {initial_n - len(data)} patients (died within 90 days).")
print(f"Final analytic N: {len(data)}, events: {int(data['event'].sum())}")

def engineer_features(df):
    d = df.copy()
    for col in ['posnodes', 'totnodes', 'histotumoursize', 'asa', 'age', 'adjchemo']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d['LNR'] = np.where(
        d['totnodes'].fillna(0) > 0,
        d['posnodes'] / d['totnodes'],
        0.0
    )
    d['log_tumoursize'] = np.log1p(d['histotumoursize'])
    d['asa_age']      = d['asa'] * d['age']
    d['adjchemo_LNR'] = d['adjchemo'] * d['LNR']
    return d

for col in RAW_COLS:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
data = engineer_features(data)

X_raw = data[SELECTED_FEATURES].copy()
y     = data[['OS_months', 'event']].copy()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y['event']
)
print(f"\nTrain: {len(X_train_raw)} | Test: {len(X_test_raw)}")
print(f"Train events: {int(y_train['event'].sum())} | "
      f"Test events: {int(y_test['event'].sum())}")

y_train_sksurv = Surv.from_arrays(
    y_train['event'].astype(bool).values, y_train['OS_months'].values
)
y_test_sksurv = Surv.from_arrays(
    y_test['event'].astype(bool).values, y_test['OS_months'].values
)

print(f"\n{'='*70}")
print(f"MICE IMPUTATION (M={M})")
print(f"{'='*70}")

imputers     = []
scalers      = []
X_train_list = []
X_test_list  = []

for m in range(M):
    seed_m = BASE_SEED + m
    imp = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=MAX_ITER,
        random_state=seed_m,
        sample_posterior=True,
        tol=1e-3
    )
    X_tr_imp = imp.fit_transform(X_train_raw)
    X_te_imp = imp.transform(X_test_raw)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_imp)
    X_te_sc = sc.transform(X_te_imp)

    imputers.append(imp)
    scalers.append(sc)
    X_train_list.append(X_tr_sc)
    X_test_list.append(X_te_sc)
    print(f"  Imputation {m+1}/{M} complete (seed={seed_m})")

joblib.dump(imputers, 'imputers_ds_mice.joblib')
joblib.dump(scalers,  'scalers_ds_mice.joblib')
print("Saved: imputers_ds_mice.joblib, scalers_ds_mice.joblib")

print(f"\n{'='*70}")
print(f"TRAINING {M} DEEPSURV MODELS ({EPOCHS} epochs each)")
print(f"{'='*70}")

def build_model(n_features):
    net = MLPVanilla(
        in_features=n_features,
        num_nodes=BEST_PARAMS["num_nodes"],
        out_features=1,
        batch_norm=True,
        dropout=BEST_PARAMS["dropout"]
    ).to(device)
    optimizer = Adam(lr=BEST_PARAMS["lr"],
                     weight_decay=BEST_PARAMS["weight_decay"])
    model = CoxPH(net, optimizer)
    model.loss = CoxPHLoss()
    return model

ds_models         = []
baseline_hazards  = []
test_risks_all    = np.zeros((M, len(X_test_raw)))

y_tr_dur = torch.tensor(y_train['OS_months'].values, dtype=torch.float32).to(device)
y_tr_evt = torch.tensor(y_train['event'].values,    dtype=torch.float32).to(device)

for m in range(M):
    print(f"\n--- DeepSurv model {m+1}/{M} ---")
    torch.manual_seed(BASE_SEED + m)
    np.random.seed(BASE_SEED + m)

    X_tr_tensor = torch.tensor(X_train_list[m], dtype=torch.float32).to(device)
    X_te_tensor = torch.tensor(X_test_list[m],  dtype=torch.float32).to(device)

    model_m = build_model(len(SELECTED_FEATURES))
    model_m.fit(
        input=X_tr_tensor,
        target=(y_tr_dur, y_tr_evt),
        batch_size=BEST_PARAMS["batch_size"],
        epochs=EPOCHS,
        verbose=True
    )

    model_m.compute_baseline_hazards(
        input=X_tr_tensor, target=(y_tr_dur, y_tr_evt)
    )

    blh = model_m.baseline_hazards_
    baseline_hazards.append(blh)

    with torch.no_grad():
        risks_m = model_m.predict(X_te_tensor).squeeze().cpu().numpy()
    test_risks_all[m] = risks_m

    c_m = concordance_index(
        y_test['OS_months'].values, -risks_m, y_test['event'].values
    )
    print(f"  Model {m+1} C-index: {c_m:.3f}")

    weights_path = f'ds_weights_mice_m{m+1}.pt'
    torch.save(model_m.net.state_dict(), weights_path)
    ds_models.append(model_m)

joblib.dump(baseline_hazards, 'ds_baseline_hazards_mice.joblib')
print("\nSaved: ds_baseline_hazards_mice.joblib")
print(f"Saved: ds_weights_mice_m1.pt ... ds_weights_mice_m{M}.pt")

torch.save(ds_models[0].net.state_dict(), 'model_weights_blh.pickle')


joblib.dump(imputers[0], 'imputer_ds.joblib')
joblib.dump(scalers[0],  'scaler_ds.joblib')
pd.DataFrame({'time': baseline_hazards[0].index,
              'baseline_hazard': baseline_hazards[0].values}
             ).to_csv('baseline_hazards.csv', index=False)
print("Saved: model_weights_blh.pickle, imputer_ds.joblib, scaler_ds.joblib "
      "(imputation 1)")

pooled_test_risks = test_risks_all.mean(axis=0)

def pooled_surv_at_times(times):
    surv_all = np.zeros((M, len(X_test_raw), len(times)))
    for m in range(M):
        X_te_tensor = torch.tensor(X_test_list[m], dtype=torch.float32).to(device)
        with torch.no_grad():
            exp_risk = torch.exp(
                ds_models[m].predict(X_te_tensor)
            ).squeeze().cpu().numpy()
        blh   = baseline_hazards[m]
        H0    = np.cumsum(blh.values)
        H0_t  = blh.index.values
        for j, t in enumerate(times):
            idx_t = np.searchsorted(H0_t, t, side='right') - 1
            if idx_t < 0: idx_t = 0
            H0_at_t = H0[idx_t]
            surv_all[m, :, j] = np.exp(-exp_risk * H0_at_t)
    return surv_all.mean(axis=0)

print(f"\n{'='*70}")
print(f"TEST SET PERFORMANCE (pooled predictions)")
print(f"{'='*70}")

c_index = concordance_index(
    y_test['OS_months'], -pooled_test_risks, y_test['event']
)
print(f"\nC-index (pooled): {c_index:.3f}")

np.random.seed(42)
boot_ci = []
for _ in range(1000):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    try:
        c = concordance_index(
            y_test['OS_months'].values[idx],
            -pooled_test_risks[idx],
            y_test['event'].values[idx]
        )
        boot_ci.append(c)
    except Exception:
        pass
ci_lo, ci_hi = np.percentile(boot_ci, [2.5, 97.5])
print(f"Bootstrapped C-index (n=1000): {np.mean(boot_ci):.3f} "
      f"(95% CI: {ci_lo:.3f}-{ci_hi:.3f})")

valid_times = np.array([t for t in [12, 36, 48] if t < y_test['OS_months'].max()])
if len(valid_times):
    auc_vals, mean_auc = cumulative_dynamic_auc(
        y_train_sksurv, y_test_sksurv, pooled_test_risks, valid_times
    )
    print(f"\nTime-dependent AUC:")
    for t, a in zip(valid_times, auc_vals):
        print(f"  {t}m: {a:.3f}")
    print(f"  Mean AUC: {mean_auc:.3f}")

brier_times = np.array([t for t in [12, 36, 48] if t < y_test['OS_months'].max()])
surv_probs  = pooled_surv_at_times(brier_times)
_, brier_vals = brier_score(y_train_sksurv, y_test_sksurv, surv_probs, brier_times)
print(f"\nBrier Scores:")
for t, b in zip(brier_times, brier_vals):
    print(f"  {t}m: {b:.3f}")

ibs_times = np.linspace(
    y_test['OS_months'].min() + 0.1,
    y_test['OS_months'].max() - 0.5, 50
)
surv_ibs = pooled_surv_at_times(ibs_times)
ibs = integrated_brier_score(y_train_sksurv, y_test_sksurv, surv_ibs, ibs_times)
print(f"  IBS: {ibs:.3f}")

print(f"\n{'='*70}")
print("SHAP ANALYSIS - DeepSurv INTERNAL (imputation 1)")
print(f"{'='*70}")

try:
    import shap

    net_shap = ds_models[0].net.cpu()
    net_shap.eval()

    scaler_shap  = scalers[0]
    imputer_shap = imputers[0]

    X_test_imp = imputer_shap.transform(X_test_raw)
    X_test_sc  = scaler_shap.transform(X_test_imp)
    X_test_df  = pd.DataFrame(X_test_sc, columns=SELECTED_FEATURES)

    X_train_imp = imputer_shap.transform(X_train_raw)
    X_train_sc  = scaler_shap.transform(X_train_imp)

    def ds_predict_shap(x):
        t = torch.tensor(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            return net_shap(t).numpy().ravel()

    bg_idx  = np.random.default_rng(42).choice(len(X_train_sc),
                                                min(100, len(X_train_sc)),
                                                replace=False)
    bg_data = X_train_sc[bg_idx]

    explainer = shap.explainers.Permutation(
        ds_predict_shap,
        bg_data,
        feature_names=SELECTED_FEATURES
    )
    n_shap   = min(len(X_test_sc), 99)
    shap_vals = explainer(X_test_sc[:n_shap],
                          max_evals=2 * len(SELECTED_FEATURES) + 1)
    shap_vals.feature_names = SELECTED_FEATURES

    mean_shap = np.abs(shap_vals.values).mean(axis=0)
    shap_df   = pd.DataFrame({
        'Feature':     SELECTED_FEATURES,
        'Mean_|SHAP|': mean_shap
    }).sort_values('Mean_|SHAP|', ascending=False)

    print(f"\n{'='*50}")
    print("DeepSurv INTERNAL - Mean |SHAP| values (ranked)")
    print(f"{'='*50}")
    print(f"{'Rank':<6}{'Feature':<22}{'Mean |SHAP|':>12}")
    print("-"*40)
    for rank, (_, row) in enumerate(shap_df.iterrows(), 1):
        print(f"{rank:<6}{row['Feature']:<22}{row['Mean_|SHAP|']:>12.4f}")
    print("="*50)
    shap_df.to_csv('ds_mice_shap.csv', index=False)

    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    shap_sorted = shap_df.sort_values('Mean_|SHAP|')
    ax_bar.barh(shap_sorted['Feature'], shap_sorted['Mean_|SHAP|'],
                color='#884EA0', alpha=0.8)
    ax_bar.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax_bar.set_title("DeepSurv Internal - Mean |SHAP| Feature Importance",
                      fontsize=12, fontweight='bold')
    ax_bar.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('ds_mice_shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: ds_mice_shap_bar.png")

    try:
        plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(shap_vals, max_display=len(SELECTED_FEATURES), show=False)
        plt.title("DeepSurv Internal - SHAP Beeswarm", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('ds_mice_shap_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: ds_mice_shap_beeswarm.png")
    except Exception as e:
        print(f"Beeswarm skipped: {e}")

except Exception as e:
    print(f"SHAP analysis failed: {e}")

print(f"\n{'='*70}")
print(f"SUMMARY - DEEPSURV MICE PRIMARY ANALYSIS")
print(f"{'='*70}")
print(f"Imputations (M):          {M}")
print(f"C-index (pooled):         {c_index:.3f} (95% CI: {ci_lo:.3f}-{ci_hi:.3f})")
if len(valid_times):
    print(f"Mean AUC:                 {mean_auc:.3f}")
print(f"IBS:                      {ibs:.3f}")
print(f"\nFiles saved:")
print(f"  ds_weights_mice_m1-{M}.pt       - {M} DeepSurv weight files")
print(f"  imputers_ds_mice.joblib         - {M} fitted MICE imputers")
print(f"  scalers_ds_mice.joblib          - {M} fitted scalers")
print(f"  ds_baseline_hazards_mice.joblib - {M} baseline hazard DataFrames")
print(f"  model_weights_blh.pickle        - imputation 1 weights (compat.)")
print(f"\nNext: run sensitivity_analysis.py (complete case + median imputation)")
print("=" * 70)
