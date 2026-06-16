import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score
from sksurv.util import Surv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy import stats

DATA_PATH  = r"D:\Users\Downloads\Cleaned_Dataset_for_Analysis.csv"
M          = 10
MAX_ITER   = 10
BASE_SEED  = 42

SELECTED_FEATURES = [
    'age', 'bmi', 'asa', 'adjchemo',
    'log_tumoursize', 'histoT', 'histoN', 'differentiation',
    'LNR', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR',
    'asa_age', 'adjchemo_LNR'
]
RAW_COLS = ['age', 'bmi', 'asa', 'adjchemo',
            'histotumoursize', 'histoT', 'histoN', 'differentiation',
            'totnodes', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR']

print("=" * 70)
print("COX MODEL - MICE IMPUTATION (M=10) + RUBIN'S RULES")
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

y_train[['OS_months', 'event']].to_csv('y_train_cox.csv', index=False)
print("Saved: y_train_cox.csv")

print(f"\nMissing data in training features:")
miss = X_train_raw.isna().mean() * 100
for col, pct in miss[miss > 0].sort_values(ascending=False).items():
    print(f"  {col:<25} {pct:.1f}%")

print(f"\n{'='*70}")
print(f"MICE IMPUTATION (M={M}, max_iter={MAX_ITER}, estimator=BayesianRidge)")
print(f"{'='*70}")

imputers  = []
scalers   = []
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

joblib.dump(imputers, 'imputers_cox_mice.joblib')
joblib.dump(scalers,  'scalers_cox_mice.joblib')
print(f"\nSaved: imputers_cox_mice.joblib, scalers_cox_mice.joblib")

print(f"\n{'='*70}")
print(f"FITTING {M} COX MODELS")
print(f"{'='*70}")

cox_models = []
coefs_list  = []
vars_list   = []

for m in range(M):
    train_df_m = pd.DataFrame(X_train_list[m], columns=SELECTED_FEATURES)
    train_df_m['OS_months'] = y_train['OS_months'].values
    train_df_m['event']     = y_train['event'].values

    cph_m = CoxPHFitter()
    cph_m.fit(train_df_m, duration_col='OS_months', event_col='event',
              show_progress=False)
    cox_models.append(cph_m)

    coefs_list.append(cph_m.summary['coef'].values)
    vars_list.append(cph_m.summary['se(coef)'].values ** 2)

    print(f"  Model {m+1}/{M}: C-index={cph_m.concordance_index_:.3f}")

joblib.dump(cox_models, 'cox_models_mice.joblib')

joblib.dump(cox_models[0], 'cox_model.pkl')
joblib.dump(imputers[0],   'imputer_cox.joblib')
joblib.dump(scalers[0],    'scaler_cox.save')
print(f"\nSaved: cox_models_mice.joblib")
print(f"Saved: cox_model.pkl, imputer_cox.joblib, scaler_cox.save (imputation 1, backward compat.)")

coefs_arr = np.array(coefs_list)
vars_arr  = np.array(vars_list)

print(f"\n{'='*70}")
print(f"RUBIN'S RULES POOLING")
print(f"{'='*70}")

Q_bar = coefs_arr.mean(axis=0)
W     = vars_arr.mean(axis=0)
B     = coefs_arr.var(axis=0, ddof=1)
T     = W + (1 + 1/M) * B
SE    = np.sqrt(T)

lambda_m = (1 + 1/M) * B / T
nu_old   = (M - 1) / (lambda_m ** 2)
nu_obs   = (len(X_train_raw) - len(SELECTED_FEATURES) + 1) /           (len(X_train_raw) - len(SELECTED_FEATURES) + 3) *           (len(X_train_raw) - len(SELECTED_FEATURES)) * (1 - lambda_m)
nu       = nu_old * nu_obs / (nu_old + nu_obs)

t_stat  = Q_bar / SE
p_vals  = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=nu))
z_crit  = 1.96

HR      = np.exp(Q_bar)
CI_lo   = np.exp(Q_bar - z_crit * SE)
CI_hi   = np.exp(Q_bar + z_crit * SE)

pooled_summary = pd.DataFrame({
    'feature':    SELECTED_FEATURES,
    'coef':       Q_bar,
    'SE':         SE,
    'HR':         HR,
    'CI_lo_95':   CI_lo,
    'CI_hi_95':   CI_hi,
    't_stat':     t_stat,
    'p_value':    p_vals,
    'W_var':      W,
    'B_var':      B,
    'lambda_m':   lambda_m,
    'frac_miss_info': lambda_m,
})
pooled_summary['sig'] = pooled_summary['p_value'].apply(
    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
)

print("\nPooled Cox Model (Rubin's Rules):")
print(f"{'Feature':<20} {'HR':>6} {'95% CI':>18} {'p':>8} {'FMI':>6} {'Sig':>4}")
print("-" * 68)
for _, row in pooled_summary.iterrows():
    print(f"  {row['feature']:<18} {row['HR']:>6.3f} "
          f"({row['CI_lo_95']:.3f}-{row['CI_hi_95']:.3f}) "
          f"{row['p_value']:>8.4f} {row['frac_miss_info']:>6.3f} {row['sig']:>4}")

pooled_summary.to_csv('cox_mice_pooled_summary.csv', index=False)
print("\nSaved: cox_mice_pooled_summary.csv")

X_vif = add_constant(pd.DataFrame(X_train_list[0], columns=SELECTED_FEATURES))
vif_data = pd.DataFrame({
    'Feature': X_vif.columns,
    'VIF':     [variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])]
})
print(f"\nVIF (imputation 1):\n{vif_data.to_string(index=False)}")
if (vif_data['VIF'] > 5).any():
    print("WARNING: High collinearity detected (VIF > 5)")
else:
    print("Collinearity check passed (all VIF < 5).")

test_risks_all = np.zeros((M, len(X_test_raw)))
for m in range(M):
    test_df_m = pd.DataFrame(X_test_list[m], columns=SELECTED_FEATURES)
    test_df_m['OS_months'] = y_test['OS_months'].values
    test_df_m['event']     = y_test['event'].values
    test_risks_all[m] = cox_models[m].predict_partial_hazard(test_df_m).values

pooled_test_risks = test_risks_all.mean(axis=0)

def pooled_survival_at_times(times):
    surv_all = np.zeros((M, len(X_test_raw), len(times)))
    for m in range(M):
        test_df_m = pd.DataFrame(X_test_list[m], columns=SELECTED_FEATURES)
        test_df_m['OS_months'] = y_test['OS_months'].values
        test_df_m['event']     = y_test['event'].values
        sf = cox_models[m].predict_survival_function(test_df_m, times=times)
        surv_all[m] = sf.values.T
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

y_train_sksurv = Surv.from_arrays(
    y_train['event'].astype(bool).values,
    y_train['OS_months'].values
)
y_test_sksurv = Surv.from_arrays(
    y_test['event'].astype(bool).values,
    y_test['OS_months'].values
)
valid_times = np.array([t for t in [12, 36, 58] if t < y_test['OS_months'].max()])
if len(valid_times):
    auc_vals, mean_auc = cumulative_dynamic_auc(
        y_train_sksurv, y_test_sksurv, pooled_test_risks, valid_times
    )
    print(f"\nTime-dependent AUC:")
    for t, a in zip(valid_times, auc_vals):
        print(f"  {t}m: {a:.3f}")
    print(f"  Mean AUC: {mean_auc:.3f}")

print(f"\nBrier Scores (pooled survival functions):")
brier_times = np.array([t for t in [12, 36, 58] if t < y_test['OS_months'].max()])
surv_probs = pooled_survival_at_times(brier_times)
_, brier_vals = brier_score(
    y_train_sksurv, y_test_sksurv, surv_probs, brier_times
)
for t, b in zip(brier_times, brier_vals):
    print(f"  {t}m: {b:.3f}")

all_times = np.unique(np.concatenate([
    cox_models[m].baseline_survival_.index.values for m in range(M)
]))
all_times = all_times[(all_times > 0) & (all_times < y_test['OS_months'].max())]
all_surv  = pooled_survival_at_times(all_times)
ibs = integrated_brier_score(y_train_sksurv, y_test_sksurv, all_surv, all_times)
print(f"  IBS: {ibs:.3f}")

print(f"\n{'='*70}")
print(f"PROPORTIONAL HAZARDS TEST (imputation 1)")
print(f"{'='*70}")
train_df_1 = pd.DataFrame(X_train_list[0], columns=SELECTED_FEATURES)
train_df_1['OS_months'] = y_train['OS_months'].values
train_df_1['event']     = y_train['event'].values
cox_models[0].check_assumptions(train_df_1, p_value_threshold=0.05,
                                 show_plots=False)

print(f"\n{'='*70}")
print(f"GENERATING FOREST PLOT")
print(f"{'='*70}")

LABEL_MAP = {
    'age':              'Age (years)',
    'bmi':              'BMI (kg/m²)',
    'asa':              'ASA score',
    'adjchemo_LNR':     'Adjuvant chemo × LNR',
    'asa_age':          'ASA × Age (frailty interaction)',
    'adjchemo':         'Adjuvant chemotherapy',
    'log_tumoursize':   'Log tumour size (log mm)',
    'histoT':           'Pathological T stage',
    'histoN':           'Pathological N stage',
    'differentiation':  'Tumour differentiation',
    'LNR':              'Lymph node ratio (LNR)',
    'posnodes':         'Positive lymph nodes',
    'rstatus':          'Resection margin (R-status)',
    'albumin':          'Albumin (g/L)',
    'bilirubin':        'Bilirubin (µmol/L)',
    'NLR':              'Neutrophil-lymphocyte ratio',
}

ps_sorted = pooled_summary.sort_values('HR').copy()
ps_sorted['label'] = [LABEL_MAP.get(f, f) for f in ps_sorted['feature']]
ps_sorted['label_disp'] = ps_sorted.apply(
    lambda r: f"{r['label']} *" if r['p_value'] < 0.05 else r['label'], axis=1
)

n_f = len(ps_sorted)
fig, ax = plt.subplots(figsize=(10, max(4, n_f * 0.65 + 2)))
y_pos = np.arange(n_f)
hrs   = ps_sorted['HR'].values
lo    = ps_sorted['CI_lo_95'].values
hi    = ps_sorted['CI_hi_95'].values
ps    = ps_sorted['p_value'].values

for i in range(n_f):
    ax.axhspan(i - 0.4, i + 0.4,
               color='#F4F6F7' if i % 2 == 0 else 'white', zorder=0)

ax.errorbar(hrs, y_pos, xerr=[hrs - lo, hi - hrs],
            fmt='none', ecolor='#555555', capsize=4,
            capthick=1.5, linewidth=1.5, zorder=2)
for i, (hr, p) in enumerate(zip(hrs, ps)):
    color = ('#C0392B' if hr >= 1 else '#1A5276') if p < 0.05 else '#888888'
    ax.plot(hr, i, 's', color=color, markersize=8, zorder=3,
            markeredgecolor='white', markeredgewidth=0.5)

ax.axvline(x=1, color='black', linestyle='--', linewidth=1.2)

for i, (hr, lo_i, hi_i, p_i) in enumerate(zip(hrs, lo, hi, ps)):
    p_str = f"{p_i:.3f}" if p_i >= 0.001 else "<0.001"
    sig_str = ' *' if p_i < 0.05 else ''
    ax.text(max(hi) * 1.02 + 0.03, i,
            f"{hr:.2f} ({lo_i:.2f}-{hi_i:.2f})  p={p_str}{sig_str}",
            va='center', fontsize=8.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(ps_sorted['label_disp'].values, fontsize=10)
ax.set_xlabel("Hazard Ratio (95% CI, pooled)", fontsize=11)
ax.set_title(f"Cox Model - Pooled Hazard Ratios (MICE, M={M}, Rubin's Rules)\n"
             "* p < 0.05  |  Red = risk-increasing  |  Blue = protective",
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(max(0.1, min(lo) * 0.80),
            max(hi) * 1.02 + (max(hi) - min(lo)) * 0.55)

ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(y_pos)
ax2.set_yticklabels(
    [f"FMI={r['frac_miss_info']:.2f}" for _, r in ps_sorted.iterrows()],
    fontsize=7.5, color='grey'
)
ax2.set_ylabel("Fraction of Missing Information", fontsize=8, color='grey')

plt.tight_layout()
plt.savefig('cox_mice_hr_forest.png', dpi=150, bbox_inches='tight')
print("Saved: cox_mice_hr_forest.png")

print(f"\n{'='*70}")
print(f"SUMMARY - COX MICE PRIMARY ANALYSIS")
print(f"{'='*70}")
print(f"Imputations (M):         {M}")
print(f"MICE estimator:          BayesianRidge (sample_posterior=True)")
print(f"C-index (pooled):        {c_index:.3f} (95% CI: {ci_lo:.3f}-{ci_hi:.3f})")
if len(valid_times):
    print(f"Mean AUC:                {mean_auc:.3f}")
print(f"IBS:                     {ibs:.3f}")
print(f"\nSignificant features (p < 0.05):")
for _, row in pooled_summary[pooled_summary['p_value'] < 0.05].iterrows():
    print(f"  {row['feature']:<20} HR={row['HR']:.3f} "
          f"({row['CI_lo_95']:.3f}-{row['CI_hi_95']:.3f})  p={row['p_value']:.4f}")
print(f"\nFiles saved:")
print(f"  cox_models_mice.joblib          - {M} fitted Cox models")
print(f"  imputers_cox_mice.joblib        - {M} fitted MICE imputers")
print(f"  scalers_cox_mice.joblib         - {M} fitted scalers")
print(f"  cox_mice_pooled_summary.csv     - pooled HRs + Rubin's rules stats")
print(f"  y_train_cox.csv                 - training survival reference")
print(f"  cox_mice_hr_forest.png          - forest plot")
print(f"\nNext: run rsfnewwv_mice.py (RSF with MICE)")
print("=" * 70)
