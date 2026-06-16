import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss
from torchtuples.practical import MLPVanilla
from torchtuples.optim import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = os.path.join(os.environ.get("RAW_DATA_DIR", "."), "Cleaned_Dataset_for_Analysis.csv")

SELECTED_FEATURES = [
    'age', 'bmi', 'asa',
    'log_tumoursize', 'histoT', 'histoN', 'differentiation',
    'LNR', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR',
    'asa_age'
]
RAW_COLS = [
    'age', 'bmi', 'asa',
    'histotumoursize', 'histoT', 'histoN', 'differentiation',
    'totnodes', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR'
]

MICE_M = 10
RSF_PARAMS = dict(
    n_estimators=300, min_samples_split=10,
    min_samples_leaf=15, max_features='sqrt',
    n_jobs=-1, random_state=42
)
DS_PARAMS = {
    "lr": 0.0001839, "num_nodes": [128, 64],
    "dropout": 0.4485, "weight_decay": 0.0002721,
    "batch_size": 32, "epochs": 50
}

print("=" * 70)
print("SENSITIVITY ANALYSIS - NO CHEMOTHERAPY (14 features)")
print("=" * 70)

data = pd.read_csv(DATA_PATH, na_values=['#NUM!'])
data['event'] = data['alive_dead'].map({1: 0, 2: 1, 3: 0})

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
data['OS_months'] = pd.to_numeric(data['OS_months'], errors='coerce')
data.loc[data['OS_months'].isna() & (data['event'] == 0), 'OS_months'] = MAX_FUP
data = data.dropna(subset=['OS_months', 'event'])
over_fup = data['OS_months'] > MAX_FUP
data.loc[over_fup, 'event']     = 0
data.loc[over_fup, 'OS_months'] = MAX_FUP
data = data[~((data['event'] == 1) & (data['OS_months'] <= 3))]

def engineer_features(df):
    d = df.copy()
    for col in ['posnodes', 'totnodes', 'histotumoursize', 'asa', 'age']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d['LNR'] = np.where(
        d['totnodes'].fillna(0) > 0,
        d['posnodes'] / d['totnodes'],
        0.0
    )
    d['log_tumoursize'] = np.log1p(d['histotumoursize'])
    d['asa_age'] = d['asa'] * d['age']
    return d

for col in RAW_COLS:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
data = engineer_features(data)

X_raw = data[SELECTED_FEATURES].copy()
y     = data[['OS_months', 'event']].copy()

print(f"Full cohort: N={len(data)}, events={int(data['event'].sum())}")
print(f"Missing per feature:")
for col, pct in (X_raw.isna().mean() * 100).sort_values(ascending=False).items():
    if pct > 0:
        print(f"  {col:<25} {pct:.1f}%")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y['event']
)
print(f"\nTrain: {len(X_train_raw)} | Test: {len(X_test_raw)}")

y_train_sksurv = Surv.from_arrays(
    y_train['event'].astype(bool).values, y_train['OS_months'].values
)
y_test_sksurv = Surv.from_arrays(
    y_test['event'].astype(bool).values, y_test['OS_months'].values
)

def compute_metrics(risks, label, y_tr_s, y_te_s, y_te_df, n_boot=500):
    c = concordance_index(y_te_df['OS_months'], -risks, y_te_df['event'])
    np.random.seed(42)
    boots = []
    for _ in range(n_boot):
        idx = np.random.choice(len(y_te_df), len(y_te_df), replace=True)
        try:
            boots.append(concordance_index(
                y_te_df['OS_months'].values[idx],
                -risks[idx], y_te_df['event'].values[idx]
            ))
        except Exception:
            pass
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    valid_t = np.array([t for t in [12, 36] if t < y_te_df['OS_months'].max()])
    aucs = {}
    if len(valid_t):
        try:
            auc_vals, _ = cumulative_dynamic_auc(y_tr_s, y_te_s, risks, valid_t)
            for t, a in zip(valid_t, auc_vals):
                aucs[t] = round(a, 3)
        except Exception:
            pass

    print(f"  {label:<35} C={c:.3f} (95% CI: {ci_lo:.3f}-{ci_hi:.3f}) "
          + "  ".join([f"AUC@{t}m={v:.3f}" for t, v in aucs.items()]))
    return {'label': label, 'c_index': c, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            **{f'auc_{t}m': v for t, v in aucs.items()}}

print(f"\n{'='*70}")
print("STRATEGY 1: COMPLETE CASE (no imputation)")
print("=" * 70)

train_cc_mask = ~X_train_raw.isna().any(axis=1)
test_cc_mask  = ~X_test_raw.isna().any(axis=1)

X_tr_cc = X_train_raw[train_cc_mask]
X_te_cc = X_test_raw[test_cc_mask]
y_tr_cc = y_train[train_cc_mask]
y_te_cc = y_test[test_cc_mask]

print(f"Training complete cases: {len(X_tr_cc)} / {len(X_train_raw)} "
      f"({len(X_tr_cc)/len(X_train_raw)*100:.1f}%)")
print(f"Test complete cases:     {len(X_te_cc)} / {len(X_test_raw)} "
      f"({len(X_te_cc)/len(X_test_raw)*100:.1f}%)")

sc_cc = StandardScaler()
X_tr_cc_sc = sc_cc.fit_transform(X_tr_cc)
X_te_cc_sc = sc_cc.transform(X_te_cc)

y_tr_cc_s = Surv.from_arrays(y_tr_cc['event'].astype(bool).values, y_tr_cc['OS_months'].values)
y_te_cc_s = Surv.from_arrays(y_te_cc['event'].astype(bool).values, y_te_cc['OS_months'].values)

cc_results = []

tr_df_cc = pd.DataFrame(X_tr_cc_sc, columns=SELECTED_FEATURES)
tr_df_cc['OS_months'] = y_tr_cc['OS_months'].values
tr_df_cc['event']     = y_tr_cc['event'].values
te_df_cc = pd.DataFrame(X_te_cc_sc, columns=SELECTED_FEATURES)
te_df_cc['OS_months'] = y_te_cc['OS_months'].values
te_df_cc['event']     = y_te_cc['event'].values

cph_cc = CoxPHFitter()
cph_cc.fit(tr_df_cc, duration_col='OS_months', event_col='event', show_progress=False)
cox_cc_risks = cph_cc.predict_partial_hazard(te_df_cc).values
cc_results.append(compute_metrics(cox_cc_risks, 'Cox - Complete Case',
                                  y_tr_cc_s, y_te_cc_s, y_te_cc))

rsf_cc = RandomSurvivalForest(**RSF_PARAMS)
rsf_cc.fit(X_tr_cc_sc, y_tr_cc_s)
rsf_cc_risks = rsf_cc.predict(X_te_cc_sc)
cc_results.append(compute_metrics(rsf_cc_risks, 'RSF - Complete Case',
                                  y_tr_cc_s, y_te_cc_s, y_te_cc))

torch.manual_seed(42)
n_feat = len(SELECTED_FEATURES)
X_tr_cc_t = torch.tensor(X_tr_cc_sc, dtype=torch.float32).to(device)
X_te_cc_t = torch.tensor(X_te_cc_sc, dtype=torch.float32).to(device)
y_tr_dur_cc = torch.tensor(y_tr_cc['OS_months'].values, dtype=torch.float32).to(device)
y_tr_evt_cc = torch.tensor(y_tr_cc['event'].values,    dtype=torch.float32).to(device)

net_cc = MLPVanilla(n_feat, DS_PARAMS["num_nodes"], 1, True, DS_PARAMS["dropout"]).to(device)
opt_cc = Adam(lr=DS_PARAMS["lr"], weight_decay=DS_PARAMS["weight_decay"])
ds_cc  = CoxPH(net_cc, opt_cc); ds_cc.loss = CoxPHLoss()
ds_cc.fit(X_tr_cc_t, (y_tr_dur_cc, y_tr_evt_cc),
          batch_size=DS_PARAMS["batch_size"], epochs=DS_PARAMS["epochs"], verbose=False)
with torch.no_grad():
    ds_cc_risks = ds_cc.predict(X_te_cc_t).squeeze().cpu().numpy()
cc_results.append(compute_metrics(ds_cc_risks, 'DeepSurv - Complete Case',
                                  y_tr_cc_s, y_te_cc_s, y_te_cc))

print(f"\n{'='*70}")
print("STRATEGY 2: MEDIAN IMPUTATION")
print("=" * 70)

imp_med = SimpleImputer(strategy='median')
X_tr_med = imp_med.fit_transform(X_train_raw)
X_te_med = imp_med.transform(X_test_raw)

sc_med = StandardScaler()
X_tr_med_sc = sc_med.fit_transform(X_tr_med)
X_te_med_sc = sc_med.transform(X_te_med)

med_results = []

tr_df_med = pd.DataFrame(X_tr_med_sc, columns=SELECTED_FEATURES)
tr_df_med['OS_months'] = y_train['OS_months'].values
tr_df_med['event']     = y_train['event'].values
te_df_med = pd.DataFrame(X_te_med_sc, columns=SELECTED_FEATURES)
te_df_med['OS_months'] = y_test['OS_months'].values
te_df_med['event']     = y_test['event'].values

cph_med = CoxPHFitter()
cph_med.fit(tr_df_med, duration_col='OS_months', event_col='event', show_progress=False)
cox_med_risks = cph_med.predict_partial_hazard(te_df_med).values
med_results.append(compute_metrics(cox_med_risks, 'Cox - Median Imputation',
                                   y_train_sksurv, y_test_sksurv, y_test))

rsf_med = RandomSurvivalForest(**RSF_PARAMS)
rsf_med.fit(X_tr_med_sc, y_train_sksurv)
rsf_med_risks = rsf_med.predict(X_te_med_sc)
med_results.append(compute_metrics(rsf_med_risks, 'RSF - Median Imputation',
                                   y_train_sksurv, y_test_sksurv, y_test))

torch.manual_seed(42)
X_tr_med_t = torch.tensor(X_tr_med_sc, dtype=torch.float32).to(device)
X_te_med_t = torch.tensor(X_te_med_sc, dtype=torch.float32).to(device)
y_tr_dur_m = torch.tensor(y_train['OS_months'].values, dtype=torch.float32).to(device)
y_tr_evt_m = torch.tensor(y_train['event'].values,    dtype=torch.float32).to(device)

net_med = MLPVanilla(n_feat, DS_PARAMS["num_nodes"], 1, True, DS_PARAMS["dropout"]).to(device)
opt_med = Adam(lr=DS_PARAMS["lr"], weight_decay=DS_PARAMS["weight_decay"])
ds_med  = CoxPH(net_med, opt_med); ds_med.loss = CoxPHLoss()
ds_med.fit(X_tr_med_t, (y_tr_dur_m, y_tr_evt_m),
           batch_size=DS_PARAMS["batch_size"], epochs=DS_PARAMS["epochs"], verbose=False)
with torch.no_grad():
    ds_med_risks = ds_med.predict(X_te_med_t).squeeze().cpu().numpy()
med_results.append(compute_metrics(ds_med_risks, 'DeepSurv - Median Imputation',
                                   y_train_sksurv, y_test_sksurv, y_test))

print(f"\n{'='*70}")
print(f"STRATEGY 3: MICE (M={MICE_M}) - in-script, 14-feature set")
print("=" * 70)

mice_results = []
cox_mice_risks_all = np.zeros((MICE_M, len(X_test_raw)))
rsf_mice_risks_all = np.zeros((MICE_M, len(X_test_raw)))
ds_mice_risks_all  = np.zeros((MICE_M, len(X_test_raw)))

for m in range(MICE_M):
    print(f"  Imputation {m+1}/{MICE_M} ...", end=' ', flush=True)
    mice = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,
        sample_posterior=True,
        random_state=m,
        min_value=0
    )
    X_tr_imp = mice.fit_transform(X_train_raw)
    X_te_imp = mice.transform(X_test_raw)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_imp)
    X_te_sc = sc.transform(X_te_imp)

    tr_df_m = pd.DataFrame(X_tr_sc, columns=SELECTED_FEATURES)
    tr_df_m['OS_months'] = y_train['OS_months'].values
    tr_df_m['event']     = y_train['event'].values
    te_df_m = pd.DataFrame(X_te_sc, columns=SELECTED_FEATURES)
    te_df_m['OS_months'] = y_test['OS_months'].values
    te_df_m['event']     = y_test['event'].values
    cph_m = CoxPHFitter()
    cph_m.fit(tr_df_m, duration_col='OS_months', event_col='event', show_progress=False)
    cox_mice_risks_all[m] = cph_m.predict_partial_hazard(te_df_m).values

    rsf_m = RandomSurvivalForest(**RSF_PARAMS)
    rsf_m.fit(X_tr_sc, y_train_sksurv)
    rsf_mice_risks_all[m] = rsf_m.predict(X_te_sc)

    torch.manual_seed(m)
    X_tr_t = torch.tensor(X_tr_sc, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
    y_tr_dur_t = torch.tensor(y_train['OS_months'].values, dtype=torch.float32).to(device)
    y_tr_evt_t = torch.tensor(y_train['event'].values,    dtype=torch.float32).to(device)
    net_m = MLPVanilla(n_feat, DS_PARAMS["num_nodes"], 1, True, DS_PARAMS["dropout"]).to(device)
    opt_m = Adam(lr=DS_PARAMS["lr"], weight_decay=DS_PARAMS["weight_decay"])
    ds_m  = CoxPH(net_m, opt_m); ds_m.loss = CoxPHLoss()
    ds_m.fit(X_tr_t, (y_tr_dur_t, y_tr_evt_t),
             batch_size=DS_PARAMS["batch_size"], epochs=DS_PARAMS["epochs"], verbose=False)
    with torch.no_grad():
        ds_mice_risks_all[m] = ds_m.predict(X_te_t).squeeze().cpu().numpy()

    print("done")

mice_results.append(compute_metrics(
    cox_mice_risks_all.mean(axis=0), 'Cox - MICE (M=10)',
    y_train_sksurv, y_test_sksurv, y_test
))
mice_results.append(compute_metrics(
    rsf_mice_risks_all.mean(axis=0), 'RSF - MICE (M=10)',
    y_train_sksurv, y_test_sksurv, y_test
))
mice_results.append(compute_metrics(
    ds_mice_risks_all.mean(axis=0), 'DeepSurv - MICE (M=10)',
    y_train_sksurv, y_test_sksurv, y_test
))

print(f"\n{'='*70}")
print("SENSITIVITY ANALYSIS SUMMARY - NO CHEMOTHERAPY (14 features)")
print("=" * 70)

all_results = mice_results + med_results + cc_results
df_res = pd.DataFrame(all_results)
df_res['strategy'] = df_res['label'].str.extract(r'- (.+)$')
df_res['model']    = df_res['label'].str.extract(r'^(\w+)')

print(f"\n{'Label':<40} {'C-index':>8} {'95% CI':>18} {'AUC@12m':>9} {'AUC@36m':>9}")
print("-" * 90)
for _, row in df_res.iterrows():
    auc12 = f"{row.get('auc_12m', float('nan')):.3f}"            if not pd.isna(row.get('auc_12m', float('nan'))) else "  n/a "
    auc36 = f"{row.get('auc_36m', float('nan')):.3f}"            if not pd.isna(row.get('auc_36m', float('nan'))) else "  n/a "
    print(f"  {row['label']:<38} {row['c_index']:>8.3f} "
          f"({row['ci_lo']:.3f}-{row['ci_hi']:.3f}) "
          f"{auc12:>9} {auc36:>9}")

df_res.to_csv('sensitivity_analysis_no_chemo_results.csv', index=False)
print("\nSaved: sensitivity_analysis_no_chemo_results.csv")

strategies  = df_res['strategy'].unique()
models_list = ['Cox', 'RSF', 'DeepSurv']
colors  = {'MICE (M=10)': '#1A5276', 'Median Imputation': '#E67E22',
           'Complete Case': '#7D3C98'}
offsets = {'MICE (M=10)': -0.22, 'Median Imputation': 0, 'Complete Case': 0.22}
markers = {'MICE (M=10)': 's', 'Median Imputation': 'o', 'Complete Case': '^'}

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(models_list))

for strat in strategies:
    sub = df_res[df_res['strategy'] == strat].copy()
    sub['model_ord'] = sub['model'].map({m: i for i, m in enumerate(models_list)})
    sub = sub.dropna(subset=['model_ord'])
    sub['model_ord'] = sub['model_ord'].astype(int)

    xp    = sub['model_ord'].values + offsets.get(strat, 0)
    ci    = sub['c_index'].values
    lo    = sub['ci_lo'].values
    hi    = sub['ci_hi'].values
    color = colors.get(strat, 'grey')

    ax.errorbar(xp, ci, yerr=[ci - lo, hi - ci],
                fmt=markers.get(strat, 'o'), color=color,
                markersize=8, capsize=4, capthick=1.5,
                linewidth=1.5, label=strat,
                markeredgecolor='white', markeredgewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(models_list, fontsize=12)
ax.set_ylabel("C-index (95% CI)", fontsize=11)
ax.set_title("Sensitivity Analysis - Imputation Strategy Comparison\n"
             "Without chemotherapy features (adjchemo, adjchemo_LNR excluded)",
             fontsize=12, fontweight='bold')
ax.axhline(0.5, color='grey', linestyle=':', linewidth=1, alpha=0.5)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.45, 0.80)

plt.tight_layout()
plt.savefig('sensitivity_analysis_no_chemo_imputation.png', dpi=150, bbox_inches='tight')
print("Saved: sensitivity_analysis_no_chemo_imputation.png")

print(f"\n{'='*70}")
print("SENSITIVITY ANALYSIS (NO CHEMOTHERAPY) COMPLETE")
print("=" * 70)
