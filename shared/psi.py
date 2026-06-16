import os
import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import roc_curve, roc_auc_score
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index

warnings.filterwarnings('ignore')
sns.set_style("white")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'

try:
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH
    PYCOX_AVAILABLE = True
except ImportError:
    PYCOX_AVAILABLE = False
    print("WARNING: pycox not available - DeepSurv comparison disabled.")

INTERNAL_CSV  = os.path.join(os.environ.get("RAW_DATA_DIR", "."), "Cleaned_Dataset_for_Analysis.csv")
EXTERNAL_CSV  = os.path.join(os.environ.get("RAW_DATA_DIR", "."), "cambook_cleaned.csv")
WEIGHTS_PATH  = "model_weights_blh.pickle"

SCALER_PATH   = "scaler_ds.joblib"
IMPUTER_PATH  = "imputer_ds.joblib"
BASELINE_PATH = "baseline_hazards.csv"

PSI2_COX_HRS = {
    'LNR':            1.343,
    'differentiation': 1.266,
    'histoN':          1.235,
    'histoT':          1.117,
    'adjchemo':        0.869,
}

PSI2_WEIGHTS = {
    'LNR':             2,
    'differentiation': 2,
    'histoN':          1,
    'histoT':          1,
    'adjchemo':       -1,
}

DEEPSURV_FEATURES = [
    'age', 'bmi', 'asa', 'adjchemo',
    'log_tumoursize', 'histoT', 'histoN', 'differentiation',
    'LNR', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR',
    'asa_age', 'adjchemo_LNR'
]

RSF_FEATURES = DEEPSURV_FEATURES

COSTS = {'ct_scan': 250}
SURVEILLANCE_PROTOCOLS = {
    'low':              {'scans_2yr': 2, 'name': 'Annual (1 scan/yr)'},
    'intermediate':     {'scans_2yr': 4, 'name': 'q6-monthly'},
    'high':             {'scans_2yr': 6, 'name': 'q4-monthly'},
    'current_standard': {'scans_2yr': 4, 'name': 'Standard (q6mo for all)'},
}
CFG = {"nodes": [128, 64], "dropout": 0.4485, "max_followup": 60}

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
    d['asa_age']        = d['asa'] * d['age']
    d['adjchemo_LNR']   = d['adjchemo'] * d['LNR']
    return d

def load_internal_data(path):
    print("LOADING INTERNAL DATASET")
    df = pd.read_csv(path, na_values=['#NUM!', 'NA', 'NaN', '#DIV/0!'], encoding='latin1')
    df.columns = df.columns.str.strip()
    df['event'] = df['alive_dead'].map({1: 0, 2: 1, 3: 0})
    if 'adjchemo' in df.columns:
        df['adjchemo'] = pd.to_numeric(df['adjchemo'], errors='coerce').map(
            {1.0: 1, 2.0: 0, 3.0: float('nan')}
        )
    for _col in ['pancfatinvasion', 'perineuralinvasion', 'microvascinvasion',
                 'lymphinvasion', 'adjrad', 'recurrence_yn',
                 'recurrence_within_6_months']:
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors='coerce')
            df[_col] = df[_col].map({1: 1, 2: 0, 3: float('nan')})
    if 'differentiation' in df.columns:
        df['differentiation'] = pd.to_numeric(df['differentiation'], errors='coerce')
        df.loc[df['differentiation'] == 6, 'differentiation'] = float('nan')
    if 'histoT' in df.columns:
        df['histoT'] = pd.to_numeric(df['histoT'], errors='coerce')
        df.loc[df['histoT'] == 5, 'histoT'] = float('nan')
    if 'histoN' in df.columns:
        df['histoN'] = pd.to_numeric(df['histoN'], errors='coerce')
        df.loc[df['histoN'] == 2, 'histoN'] = float('nan')
    if 'histoM' in df.columns:
        df['histoM'] = pd.to_numeric(df['histoM'], errors='coerce')
        df.loc[df['histoM'] == 2, 'histoM'] = float('nan')
    df['OS_months'] = pd.to_numeric(df['OS_months'], errors='coerce')

    df.loc[df['OS_months'].isna() & (df['event'] == 0), 'OS_months'] = CFG['max_followup']
    df = df.dropna(subset=['OS_months', 'event'])


    over_fup = df['OS_months'] > CFG['max_followup']
    df.loc[over_fup, 'event']     = 0
    df.loc[over_fup, 'OS_months'] = CFG['max_followup']

    n0 = len(df)
    df = df[~((df['event'] == 1) & (df['OS_months'] <= 3))]
    print(f"Excluded {n0 - len(df)} patients who died within 90 days.")
    print(f"Internal cohort: N={len(df)}, Events={int(df['event'].sum())} "
          f"({df['event'].mean()*100:.1f}%)")
    return df

def load_external_data(path):
    print("LOADING EXTERNAL DATASET")
    df = pd.read_csv(path, encoding='latin1',
                     na_values=['#DIV/0!', 'NA', 'N/A', 'NaN', '#NUM!'])
    df.columns = df.columns.str.strip()




    if 'adjchemo' in df.columns:
        df['adjchemo'] = pd.to_numeric(df['adjchemo'], errors='coerce').map(
            {1.0: 1, 2.0: 0, 3.0: float('nan')}
        )
    for _col in ['pancfatinvasion', 'perineuralinvasion', 'microvascinvasion',
                 'lymphinvasion', 'adjrad', 'recurrence_yn',
                 'recurrence_within_6_months']:
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors='coerce')
            df[_col] = df[_col].map({1: 1, 2: 0, 3: float('nan')})
    if 'differentiation' in df.columns:
        df['differentiation'] = pd.to_numeric(df['differentiation'], errors='coerce')
        df.loc[df['differentiation'] == 6, 'differentiation'] = float('nan')
    if 'histoT' in df.columns:
        df['histoT'] = pd.to_numeric(df['histoT'], errors='coerce')
        df.loc[df['histoT'] == 5, 'histoT'] = float('nan')
    if 'histoN' in df.columns:
        df['histoN'] = pd.to_numeric(df['histoN'], errors='coerce')
        df.loc[df['histoN'] == 2, 'histoN'] = float('nan')
    if 'histoM' in df.columns:
        df['histoM'] = pd.to_numeric(df['histoM'], errors='coerce')
        df.loc[df['histoM'] == 2, 'histoM'] = float('nan')

    df['OS_months']  = pd.to_numeric(df['OS_months'],  errors='coerce')
    df['alive_dead'] = pd.to_numeric(df['alive_dead'], errors='coerce')

    nan_os = df['OS_months'].isna()
    df.loc[nan_os & (df['alive_dead'] == 1), 'OS_months'] = CFG['max_followup']
    df = df.dropna(subset=['OS_months', 'alive_dead'])
    df['event'] = (df['alive_dead'] == 2).astype(int)
    n0 = len(df)
    df = df[~((df['event'] == 1) & (df['OS_months'] <= 3))]
    print(f"Excluded {n0 - len(df)} patients who died within 90 days.")


    over_fup = df['OS_months'] > CFG['max_followup']
    df.loc[over_fup, 'OS_months'] = CFG['max_followup']
    df.loc[over_fup, 'event']     = 0

    print(f"External cohort: N={len(df)}, Events={int(df['event'].sum())} "
          f"({df['event'].mean()*100:.1f}%)")
    return df

def derive_psi2_lnr_cutoff(df_train, time_train, event_train, target_months=24):
    print(f"\n  Deriving optimal LNR cutoff (Youden index, {target_months}-month mortality)...")
    binary_outcome = ((event_train == 1) & (time_train <= target_months)).astype(int)

    if binary_outcome.sum() < 10:
        print(f"  WARNING: <10 events at {target_months}m - using default LNR cutoff 0.30")
        return 0.30

    lnr_vals = df_train['LNR'].fillna(0).values
    fpr, tpr, thresholds = roc_curve(binary_outcome, lnr_vals)
    youden      = tpr - fpr
    optimal_idx = int(np.argmax(youden))
    cutoff      = float(thresholds[optimal_idx])
    auc_val     = roc_auc_score(binary_outcome, lnr_vals)

    print(f"    Optimal LNR cutoff  : {cutoff:.3f}")
    print(f"    Youden index        : {youden[optimal_idx]:.3f}")
    print(f"    Sensitivity         : {tpr[optimal_idx]:.3f}")
    print(f"    Specificity         : {1 - fpr[optimal_idx]:.3f}")
    print(f"    AUC (LNR alone)     : {auc_val:.3f}")
    return cutoff

def calculate_psi2_score(df, lnr_cutoff):
    d = df.copy()
    for col in ['LNR', 'differentiation', 'histoN', 'histoT', 'adjchemo']:
        d[col] = pd.to_numeric(d[col], errors='coerce')

    lnr_high      = (d['LNR'].fillna(0) > lnr_cutoff).astype(int)
    poor_diff     = (d['differentiation'].fillna(2) >= 3).astype(int)
    node_positive = (d['histoN'].fillna(0) > 0).astype(int)
    t3_plus       = (d['histoT'].fillna(2) >= 3).astype(int)
    adj_chemo     = (d['adjchemo'].fillna(0) == 1).astype(int)

    score = (
        PSI2_WEIGHTS['LNR']            * lnr_high      +
        PSI2_WEIGHTS['differentiation'] * poor_diff     +
        PSI2_WEIGHTS['histoN']          * node_positive +
        PSI2_WEIGHTS['histoT']          * t3_plus       +
        PSI2_WEIGHTS['adjchemo']        * adj_chemo
    )
    return score

def derive_psi2_tertile_thresholds(train_scores):
    low_thresh  = float(np.percentile(train_scores, 33.33))
    high_thresh = float(np.percentile(train_scores, 66.67))
    return low_thresh, high_thresh

def apply_psi2_tertiles(scores, low_thresh, high_thresh):
    return np.where(scores <= low_thresh, 'Low',
           np.where(scores <= high_thresh, 'Intermediate', 'High'))

def print_psi2_derivation_report(lnr_cutoff, low_thresh, high_thresh):
    print("PSI-2 DERIVATION REPORT")
    print("\nVariable selection: features significant (p<0.05) in MICE-pooled Cox")
    print("model (M=10, Rubin's Rules). Adjuvant chemotherapy included at p=0.052")
    print("as clinically essential protective factor.\n")
    print(f"{'Variable':<20} {'HR':>8} {'log(HR)':>10} {'|log(HR)|/min':>15} {'Weight':>8}")

    min_log_hr = min(abs(np.log(hr)) for hr in PSI2_COX_HRS.values())
    for var, hr in PSI2_COX_HRS.items():
        log_hr   = np.log(hr)
        relative = log_hr / min_log_hr
        weight   = PSI2_WEIGHTS[var]
        print(f"  {var:<18} {hr:>8.3f} {log_hr:>10.3f} {relative:>15.2f} {weight:>8}")

    print(f"\n  Reference: min |log(HR)| = {min_log_hr:.3f} (adjchemo)")
    print(f"\nOptimal LNR cutoff (Youden index, 24m mortality): {lnr_cutoff:.3f}")
    print(f"Tertile thresholds (training 33rd/67th percentiles):")
    print(f"  Low <= {low_thresh:.2f} | Intermediate {low_thresh:.2f}-{high_thresh:.2f} | High > {high_thresh:.2f}")
    print(f"\nPSI-2 Formula:")
    print(f"  PSI-2 = 2*(LNR > {lnr_cutoff:.3f})")
    print(f"        + 2*(differentiation == poor)")
    print(f"        + 1*(histoN > 0)")
    print(f"        + 1*(histoT >= 3)")
    print(f"        - 1*(adjuvant chemotherapy received)")
    print(f"  Score range: -1 (lowest risk) to +6 (highest risk)")

def load_pretrained_deepsurv():
    if not PYCOX_AVAILABLE:
        raise ImportError("pycox required for DeepSurv loading.")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler  = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    net = tt.practical.MLPVanilla(
        in_features=len(DEEPSURV_FEATURES),
        num_nodes=CFG["nodes"],
        out_features=1,
        batch_norm=True,
        dropout=CFG["dropout"]
    )
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device).eval()
    return net, scaler, imputer, device

def train_rsf_on_split(X_train, y_train):
    rsf = RandomSurvivalForest(
        n_estimators=100, min_samples_split=10, min_samples_leaf=5,
        max_features='sqrt', n_jobs=-1, random_state=42
    )
    rsf.fit(X_train, y_train)
    return rsf

def bootstrap_c_index(scores, time, event, n_boot=1000):
    boot_c = []
    for i in range(n_boot):
        idx = resample(np.arange(len(time)), replace=True, random_state=i)
        if event[idx].sum() < 2:
            continue
        try:
            c = concordance_index(time[idx], scores[idx], event[idx])
            if not np.isnan(c):
                boot_c.append(c)
        except Exception:
            pass
    if boot_c:
        return np.percentile(boot_c, [2.5, 97.5])
    return np.nan, np.nan

def compute_auc_ibs(risk_scores, time, event, y_train_ref,
                    eval_times=(12, 36, 58), label='',
                    train_risk_scores=None):
    time        = np.asarray(time,        dtype=float)
    event       = np.asarray(event,       dtype=float)
    risk_scores = np.asarray(risk_scores, dtype=float)
    results     = {'auc_times': [], 'auc_vals': [], 'mean_auc': np.nan, 'ibs': np.nan}

    max_train_time   = float(y_train_ref['time'].max())
    train_times      = y_train_ref['time']
    train_events     = y_train_ref['event'].astype(int)
    cens_is_event    = (train_events == 0)
    sort_idx         = np.argsort(train_times)
    st               = train_times[sort_idx]
    se               = cens_is_event[sort_idx]
    n_at_risk        = np.arange(len(st), 0, -1)
    km_cens          = np.cumprod(np.where(se, 1 - 1 / n_at_risk, 1.0))
    pos_mask         = km_cens > 0
    last_pos_cens_time = float(st[pos_mask].max()) if pos_mask.any() else float(st.max()) * 0.9
    safe_cap         = min(max_train_time, last_pos_cens_time) * 0.99

    time_capped  = np.minimum(time, safe_cap)
    event_capped = event.copy()
    event_capped[time > safe_cap] = 0
    y_test_capped = Surv.from_arrays(event_capped.astype(bool), time_capped)

    safe_times = np.array(
        [t for t in eval_times if t < safe_cap and t < time_capped.max()],
        dtype=float
    )

    if len(safe_times):
        working_times, working_aucs = [], []
        for t in safe_times:
            try:
                a_vals, _ = cumulative_dynamic_auc(
                    y_train_ref, y_test_capped,
                    risk_scores, np.array([t], dtype=float)
                )
                working_times.append(int(t))
                working_aucs.append(round(float(a_vals[0]), 3))
            except Exception as e:
                print(f"  {label} AUC@{int(t)}m skipped: {e}")
        if working_times:
            results['auc_times'] = working_times
            results['auc_vals']  = working_aucs
            results['mean_auc']  = round(float(np.mean(working_aucs)), 3)
            if label:
                print(f"  {label} AUC:")
                for t, a in zip(working_times, working_aucs):
                    print(f"    {t}m: {a:.3f}")
                print(f"    Mean AUC: {results['mean_auc']:.3f}")
    else:
        print(f"  {label} AUC: no safe eval times within cap ({safe_cap:.1f}m)")

    if train_risk_scores is None:
        print(f"  {label} IBS: skipped - train_risk_scores not provided "
              f"(previously this fell back to within-evaluation KM bins, which "
              f"gave optimistic, non-comparable IBS values).")
        return results

    try:
        train_risk_scores = np.asarray(train_risk_scores, dtype=float)
        train_time_arr    = np.asarray(y_train_ref['time'],  dtype=float)
        train_event_arr   = np.asarray(y_train_ref['event'], dtype=int)

        n_bins    = min(10, len(train_risk_scores) // 5)
        if n_bins < 2:
            print(f"  {label} IBS: skipped - too few training rows to bin.")
            return results

        quantiles = np.unique(np.percentile(train_risk_scores,
                                            np.linspace(0, 100, n_bins + 1)))
        if len(quantiles) < 3:
            print(f"  {label} IBS: skipped - degenerate training risk-score quantiles.")
            return results

        eval_bin_idx = np.clip(
            np.digitize(risk_scores, quantiles[1:-1]),
            0, len(quantiles) - 2
        )

        ibs_times   = np.linspace(time.min() + 0.1,
                                  min(time.max(), safe_cap) - 0.5, 50)
        surv_matrix = np.zeros((len(risk_scores), len(ibs_times)))

        kmf_all = KaplanMeierFitter()
        kmf_all.fit(train_time_arr, event_observed=train_event_arr)
        km_t_all = kmf_all.survival_function_.index.values
        km_s_all = kmf_all.survival_function_.values.flatten()
        overall_row = np.array([float(np.interp(t, km_t_all, km_s_all))
                                for t in ibs_times])

        for b in range(len(quantiles) - 1):
            train_in_bin = (
                (train_risk_scores >= quantiles[b]) &
                (train_risk_scores <= quantiles[b + 1] if b == len(quantiles) - 2
                 else train_risk_scores < quantiles[b + 1])
            )
            eval_in_bin = (eval_bin_idx == b)
            if not eval_in_bin.any():
                continue
            if train_in_bin.sum() < 5:
                surv_matrix[eval_in_bin] = overall_row
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(train_time_arr[train_in_bin],
                    event_observed=train_event_arr[train_in_bin])
            km_t = kmf.survival_function_.index.values
            km_s = kmf.survival_function_.values.flatten()
            bin_row = np.array([float(np.interp(t, km_t, km_s))
                                for t in ibs_times])
            surv_matrix[eval_in_bin] = bin_row

        zero_rows = (surv_matrix == 0).all(axis=1)
        if zero_rows.any():
            surv_matrix[zero_rows] = overall_row

        ibs = integrated_brier_score(y_train_ref, y_test_capped,
                                     surv_matrix, ibs_times)
        results['ibs'] = round(float(ibs), 3)
        if label:
            print(f"  {label} IBS: {ibs:.3f}")
    except Exception as e:
        print(f"  {label} IBS failed: {e}")

    return results

def calculate_cost_savings(risk_groups, n_simulations=1000):
    cohort_size    = len(risk_groups)
    scans_standard = SURVEILLANCE_PROTOCOLS['current_standard']['scans_2yr']
    total_standard = cohort_size * scans_standard * COSTS['ct_scan']
    stratified_cost = sum(
        (risk_groups == g).sum() * SURVEILLANCE_PROTOCOLS[g.lower()]['scans_2yr'] * COSTS['ct_scan']
        for g in ['Low', 'Intermediate', 'High']
    )
    sim_savings = []
    for i in range(n_simulations):
        sg = resample(risk_groups, replace=True, random_state=i)
        sc = sum(
            (sg == g).sum() * SURVEILLANCE_PROTOCOLS[g.lower()]['scans_2yr'] * COSTS['ct_scan']
            for g in ['Low', 'Intermediate', 'High']
        )
        sim_savings.append(total_standard - sc)
    obs_savings  = total_standard - stratified_cost
    mean_s       = np.mean(sim_savings)
    ci_lo, ci_hi = np.percentile(sim_savings, [2.5, 97.5])
    pct_saving   = 100 * obs_savings / total_standard
    group_df = pd.DataFrame([{
        'Risk Group': g,
        'N':          int((risk_groups == g).sum()),
        '%':          f"{100*(risk_groups == g).mean():.1f}%",
        'Scans/2yr':  SURVEILLANCE_PROTOCOLS[g.lower()]['scans_2yr'],
        'Cost (£)':   f"£{(risk_groups == g).sum() * SURVEILLANCE_PROTOCOLS[g.lower()]['scans_2yr'] * COSTS['ct_scan']:,.0f}"
    } for g in ['Low', 'Intermediate', 'High']])
    return {
        'observed_savings': obs_savings, 'mean_savings': mean_s,
        'ci_low': ci_lo, 'ci_high': ci_hi, 'pct_saving': pct_saving,
        'total_standard': total_standard, 'total_stratified': stratified_cost,
        'group_breakdown': group_df,
    }

def plot_km_tertiles(time, event, risk_groups, title, low_thresh, high_thresh):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {'Low': 'steelblue', 'Intermediate': 'orange', 'High': 'crimson'}
    for grp, color in colors.items():
        mask = risk_groups == grp
        if mask.sum() == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(time[mask], event[mask], label=f'{grp} (n={mask.sum()})')
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
    try:
        mlr   = multivariate_logrank_test(time, risk_groups, event)
        p_val = mlr.p_value
        ax.set_title(f'{title}\nLog-rank p = {p_val:.4f}', fontsize=12, fontweight='bold')
    except Exception:
        ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (months)', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.05,
            f'Thresholds (training-derived): Low<={low_thresh:.2f}, High>{high_thresh:.2f}',
            transform=ax.transAxes, fontsize=8, color='gray')
    plt.tight_layout()
    plt.show()

def plot_psi2_calibration(psi_scores, time, event, cal_times, title_suffix):
    fig, axes = plt.subplots(1, len(cal_times), figsize=(6 * len(cal_times), 5))
    if len(cal_times) == 1:
        axes = [axes]
    for ax, t in zip(axes, cal_times):
        n_bins    = 10
        quantiles = np.unique(np.percentile(psi_scores, np.linspace(0, 100, n_bins + 1)))
        bin_idx   = np.clip(np.digitize(psi_scores, quantiles[:-1]) - 1,
                            0, len(quantiles) - 2)
        pred_mean, obs_vals = [], []
        for b in range(len(quantiles) - 1):
            mask = bin_idx == b
            if mask.sum() < 3:
                continue
            pred_mean.append(-psi_scores[mask].mean())
            kmf = KaplanMeierFitter()
            kmf.fit(time[mask], event[mask])
            obs_vals.append(kmf.survival_function_at_times([t]).values[0])
        if pred_mean:
            pm   = np.array(pred_mean)
            pm_n = (pm - pm.min()) / (pm.max() - pm.min() + 1e-9)
            ax.scatter(pm_n, obs_vals, s=70, color='steelblue',
                       edgecolors='navy', zorder=5)
            ax.plot([0, 1], [min(obs_vals), max(obs_vals)], 'r--',
                    lw=2, label='Ideal correlation')
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1)
            ax.set_xlabel('Normalised PSI-2 rank (higher = lower risk)')
            ax.set_ylabel('Observed KM survival')
            ax.set_title(f'PSI-2 calibration at {t}m - {title_suffix}',
                         fontweight='bold')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_psi2_roc_derivation(lnr_vals, binary_outcome, lnr_cutoff):
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, thresholds = roc_curve(binary_outcome, lnr_vals)
    auc_val = roc_auc_score(binary_outcome, lnr_vals)
    youden  = tpr - fpr
    opt_idx = int(np.argmax(youden))
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'LNR (AUC={auc_val:.3f})')
    ax.scatter(fpr[opt_idx], tpr[opt_idx], s=120, color='crimson', zorder=5,
               label=f'Youden cutoff = {lnr_cutoff:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('1 - Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_title('ROC Curve: LNR for 24-month Mortality\n(training split)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('psi2_roc_derivation.png', dpi=300, bbox_inches='tight')
    print("Saved: psi2_roc_derivation.png")
    plt.show()

def run_complete_analysis():
    print("PSI-2 - PANCREATIC SURVIVAL INDEX (Redesigned, Data-Driven)")

    df_internal = load_internal_data(INTERNAL_CSV)
    df_external = load_external_data(EXTERNAL_CSV)

    df_internal = engineer_features(df_internal)
    df_external = engineer_features(df_external)

    df_train, df_test = train_test_split(
        df_internal, test_size=0.2, random_state=42,
        stratify=df_internal['event']
    )
    print(f"\nTrain split: N={len(df_train)} | Test split: N={len(df_test)}")

    time_train  = df_train['OS_months'].values
    event_train = df_train['event'].values
    time_test   = df_test['OS_months'].values
    event_test  = df_test['event'].values
    time_ext    = df_external['OS_months'].values
    event_ext   = df_external['event'].values

    print("PSI-2 THRESHOLD DERIVATION (training split only)")

    lnr_cutoff = derive_psi2_lnr_cutoff(df_train, time_train, event_train,
                                          target_months=24)

    psi2_train = calculate_psi2_score(df_train, lnr_cutoff).values

    psi2_low_thresh, psi2_high_thresh = derive_psi2_tertile_thresholds(psi2_train)

    print_psi2_derivation_report(lnr_cutoff, psi2_low_thresh, psi2_high_thresh)

    binary_outcome_train = ((event_train == 1) & (time_train <= 24)).astype(int)
    plot_psi2_roc_derivation(
        lnr_vals       = df_train['LNR'].fillna(0).values,
        binary_outcome = binary_outcome_train,
        lnr_cutoff     = lnr_cutoff
    )

    psi2_test  = calculate_psi2_score(df_test,     lnr_cutoff).values
    psi2_full  = calculate_psi2_score(df_internal, lnr_cutoff).values
    psi2_ext   = calculate_psi2_score(df_external, lnr_cutoff).values

    psi2_test_tertiles = apply_psi2_tertiles(psi2_test, psi2_low_thresh, psi2_high_thresh)
    psi2_full_tertiles = apply_psi2_tertiles(psi2_full, psi2_low_thresh, psi2_high_thresh)
    psi2_ext_tertiles  = apply_psi2_tertiles(psi2_ext,  psi2_low_thresh, psi2_high_thresh)

    print("\nTraining RSF on training split (16 ML features)...")
    imp_rsf = SimpleImputer(strategy='median')

    X_rsf_train_raw = df_train[RSF_FEATURES].copy()
    for col in RSF_FEATURES:
        X_rsf_train_raw[col] = pd.to_numeric(X_rsf_train_raw[col], errors='coerce')
    X_rsf_train = imp_rsf.fit_transform(X_rsf_train_raw)

    y_rsf_train = Surv.from_arrays(
        df_train['event'].astype(bool).values, df_train['OS_months'].values
    )
    rsf_model   = train_rsf_on_split(X_rsf_train, y_rsf_train)
    y_train_ref = y_rsf_train

    ds_net = ds_scaler = ds_imputer = device = None
    if PYCOX_AVAILABLE:
        import os
        if all(os.path.exists(p) for p in [WEIGHTS_PATH, SCALER_PATH, IMPUTER_PATH]):
            try:
                ds_net, ds_scaler, ds_imputer, device = load_pretrained_deepsurv()
                print("DeepSurv loaded successfully.")
            except Exception as e:
                print(f"DeepSurv load failed: {e}")



    rsf_train_scores = rsf_model.predict(X_rsf_train)
    ds_train_scores  = None
    if ds_net is not None:
        X_ds_train_raw = df_train[DEEPSURV_FEATURES].copy()
        X_ds_train_imp = ds_imputer.transform(X_ds_train_raw)
        X_ds_train_sc  = ds_scaler.transform(X_ds_train_imp)
        with torch.no_grad():
            ds_train_scores = ds_net(
                torch.FloatTensor(X_ds_train_sc).to(device)
            ).squeeze().cpu().numpy()

    print("INTERNAL TEST SET EVALUATION")

    psi2_c_test = concordance_index(time_test, -psi2_test, event_test)
    ci_lo, ci_hi = bootstrap_c_index(-psi2_test, time_test, event_test)
    print(f"\nPSI-2 - C-index: {psi2_c_test:.2f} (95% CI: {ci_lo:.2f}-{ci_hi:.2f})")
    psi2_test_metrics = compute_auc_ibs(psi2_test, time_test, event_test,
                                         y_train_ref, label='PSI-2 (Internal)',
                                         train_risk_scores=psi2_train)

    plot_km_tertiles(time_test, event_test, psi2_test_tertiles,
                     "PSI-2 Risk Groups - Internal Test Set",
                     psi2_low_thresh, psi2_high_thresh)

    cal_t = [t for t in [12, 36, 48] if t < time_test.max()]
    if cal_t:
        plot_psi2_calibration(psi2_test, time_test, event_test, cal_t, "Internal Test")

    internal_cost_tertile = calculate_cost_savings(psi2_test_tertiles)
    print(f"\nPSI-2 Cost Savings - Tertile (Internal Test, n={len(df_test)}):")
    print(internal_cost_tertile['group_breakdown'].to_string(index=False))
    print(f"  Observed: £{internal_cost_tertile['observed_savings']:,.0f} "
          f"({internal_cost_tertile['pct_saving']:.1f}%)")
    print(f"  Monte Carlo: £{internal_cost_tertile['mean_savings']:,.0f} "
          f"(95% CI: £{internal_cost_tertile['ci_low']:,.0f}-"
          f"£{internal_cost_tertile['ci_high']:,.0f})")

    X_rsf_test_raw = df_test[RSF_FEATURES].copy()
    for col in RSF_FEATURES:
        X_rsf_test_raw[col] = pd.to_numeric(X_rsf_test_raw[col], errors='coerce')
    X_rsf_test    = imp_rsf.transform(X_rsf_test_raw)
    rsf_test_scores = rsf_model.predict(X_rsf_test)
    rsf_c_test    = concordance_index(time_test, -rsf_test_scores, event_test)
    ci_lo_r, ci_hi_r = bootstrap_c_index(-rsf_test_scores, time_test, event_test)
    print(f"\nRSF - C-index: {rsf_c_test:.2f} (95% CI: {ci_lo_r:.2f}-{ci_hi_r:.2f})")
    rsf_test_metrics = compute_auc_ibs(rsf_test_scores, time_test, event_test,
                                        y_train_ref, label='RSF (Internal)',
                                        train_risk_scores=rsf_train_scores)

    ds_c_test = ci_lo_d = ci_hi_d = None
    ds_test_metrics = {'auc_times': [], 'auc_vals': [], 'mean_auc': np.nan, 'ibs': np.nan}
    if ds_net is not None:
        X_ds_test = df_test[DEEPSURV_FEATURES].copy()
        X_ds_imp  = ds_imputer.transform(X_ds_test)
        X_ds_sc   = ds_scaler.transform(X_ds_imp)
        with torch.no_grad():
            ds_test_scores = ds_net(
                torch.FloatTensor(X_ds_sc).to(device)
            ).squeeze().cpu().numpy()
        ds_c_test = concordance_index(time_test, -ds_test_scores, event_test)
        ci_lo_d, ci_hi_d = bootstrap_c_index(-ds_test_scores, time_test, event_test)
        print(f"DeepSurv - C-index: {ds_c_test:.2f} "
              f"(95% CI: {ci_lo_d:.2f}-{ci_hi_d:.2f})")
        ds_test_metrics = compute_auc_ibs(ds_test_scores, time_test, event_test,
                                           y_train_ref, label='DeepSurv (Internal)',
                                           train_risk_scores=ds_train_scores)

    print(f"ECONOMIC ANALYSIS - FULL INTERNAL COHORT (n={len(df_internal)})")
    print("PSI-2 thresholds remain fixed from training split - no data leakage.")

    full_cost_tertile = calculate_cost_savings(psi2_full_tertiles)
    print(f"\nPSI-2 Cost Savings - Tertile (Full Internal, n={len(df_internal)}):")
    print(full_cost_tertile['group_breakdown'].to_string(index=False))
    print(f"  Observed: £{full_cost_tertile['observed_savings']:,.0f} "
          f"({full_cost_tertile['pct_saving']:.1f}%)")
    print(f"  Monte Carlo: £{full_cost_tertile['mean_savings']:,.0f} "
          f"(95% CI: £{full_cost_tertile['ci_low']:,.0f}-"
          f"£{full_cost_tertile['ci_high']:,.0f})")

    print("EXTERNAL COHORT EVALUATION")

    psi2_c_ext = concordance_index(time_ext, -psi2_ext, event_ext)
    ci_lo_e, ci_hi_e = bootstrap_c_index(-psi2_ext, time_ext, event_ext)
    print(f"\nPSI-2 - C-index: {psi2_c_ext:.2f} (95% CI: {ci_lo_e:.2f}-{ci_hi_e:.2f})")
    psi2_ext_metrics = compute_auc_ibs(psi2_ext, time_ext, event_ext,
                                        y_train_ref, label='PSI-2 (External)',
                                        train_risk_scores=psi2_train)

    plot_km_tertiles(time_ext, event_ext, psi2_ext_tertiles,
                     "PSI-2 Risk Groups - External Cohort",
                     psi2_low_thresh, psi2_high_thresh)

    if cal_t:
        plot_psi2_calibration(psi2_ext, time_ext, event_ext, cal_t, "External Cohort")

    external_cost_tertile = calculate_cost_savings(psi2_ext_tertiles)
    print(f"\nPSI-2 Cost Savings - Tertile (External, n={len(df_external)}):")
    print(external_cost_tertile['group_breakdown'].to_string(index=False))
    print(f"  Observed: £{external_cost_tertile['observed_savings']:,.0f} "
          f"({external_cost_tertile['pct_saving']:.1f}%)")
    print(f"  Monte Carlo: £{external_cost_tertile['mean_savings']:,.0f} "
          f"(95% CI: £{external_cost_tertile['ci_low']:,.0f}-"
          f"£{external_cost_tertile['ci_high']:,.0f})")

    X_rsf_ext_raw = df_external[RSF_FEATURES].copy()
    for col in RSF_FEATURES:
        X_rsf_ext_raw[col] = pd.to_numeric(X_rsf_ext_raw[col], errors='coerce')
    X_rsf_ext     = imp_rsf.transform(X_rsf_ext_raw)
    rsf_ext_scores = rsf_model.predict(X_rsf_ext)
    rsf_c_ext      = concordance_index(time_ext, -rsf_ext_scores, event_ext)
    ci_lo_re, ci_hi_re = bootstrap_c_index(-rsf_ext_scores, time_ext, event_ext)
    print(f"\nRSF - C-index: {rsf_c_ext:.2f} (95% CI: {ci_lo_re:.2f}-{ci_hi_re:.2f})")
    rsf_ext_metrics = compute_auc_ibs(rsf_ext_scores, time_ext, event_ext,
                                       y_train_ref, label='RSF (External)',
                                       train_risk_scores=rsf_train_scores)

    ds_c_ext = ci_lo_de = ci_hi_de = None
    ds_ext_metrics = {'auc_times': [], 'auc_vals': [], 'mean_auc': np.nan, 'ibs': np.nan}
    if ds_net is not None:
        X_ds_ext = df_external[DEEPSURV_FEATURES].copy()
        X_ds_imp_e = ds_imputer.transform(X_ds_ext)
        X_ds_sc_e  = ds_scaler.transform(X_ds_imp_e)
        with torch.no_grad():
            ds_ext_scores = ds_net(
                torch.FloatTensor(X_ds_sc_e).to(device)
            ).squeeze().cpu().numpy()
        ds_c_ext = concordance_index(time_ext, -ds_ext_scores, event_ext)
        ci_lo_de, ci_hi_de = bootstrap_c_index(-ds_ext_scores, time_ext, event_ext)
        print(f"DeepSurv - C-index: {ds_c_ext:.2f} "
              f"(95% CI: {ci_lo_de:.2f}-{ci_hi_de:.2f})")
        ds_ext_metrics = compute_auc_ibs(ds_ext_scores, time_ext, event_ext,
                                          y_train_ref, label='DeepSurv (External)',
                                          train_risk_scores=ds_train_scores)

    print("\nGenerating performance comparison figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    int_results = {
        'PSI-2': {'c': psi2_c_test, 'lo': ci_lo,   'hi': ci_hi},
        'RSF':   {'c': rsf_c_test,  'lo': ci_lo_r,  'hi': ci_hi_r},
    }
    ext_results = {
        'PSI-2': {'c': psi2_c_ext, 'lo': ci_lo_e,  'hi': ci_hi_e},
        'RSF':   {'c': rsf_c_ext,  'lo': ci_lo_re, 'hi': ci_hi_re},
    }
    if ds_net is not None and ds_c_test is not None:
        int_results['DeepSurv'] = {'c': ds_c_test, 'lo': ci_lo_d,  'hi': ci_hi_d}
        ext_results['DeepSurv'] = {'c': ds_c_ext,  'lo': ci_lo_de, 'hi': ci_hi_de}

    colors = {'PSI-2': 'steelblue', 'RSF': 'forestgreen', 'DeepSurv': 'crimson'}
    for ax, (label, results) in zip(axes, [
        ("Internal Test Set", int_results),
        ("External Cohort",   ext_results),
    ]):
        models  = list(results.keys())
        c_vals  = [results[m]['c'] for m in models]
        errs_lo = [max(0, results[m]['c'] - results[m]['lo']) for m in models]
        errs_hi = [max(0, results[m]['hi'] - results[m]['c']) for m in models]
        x = np.arange(len(models))
        ax.bar(x, c_vals, color=[colors[m] for m in models],
               alpha=0.75, edgecolor='black')
        ax.errorbar(x, c_vals, yerr=[errs_lo, errs_hi],
                    fmt='none', ecolor='black', capsize=5, linewidth=2)
        for i, (m, c) in enumerate(zip(models, c_vals)):
            ax.text(i, c + 0.02, f'{c:.2f}', ha='center',
                    fontsize=10, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel('C-index', fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_ylim(0.5, 0.85)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Model Performance: PSI-2 vs ML - Internal & External',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('Fig_PSI2_ML_Comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: Fig_PSI2_ML_Comparison.png")
    plt.show()

    print("CROSS-COHORT LOG-RANK TESTS - Internal Test Set vs External Cohort")
    print("Tests whether survival differs between cohorts within each risk group.")

    for grp in ['Low', 'Intermediate', 'High']:
        int_mask = psi2_test_tertiles == grp
        ext_mask = psi2_ext_tertiles  == grp
        t_int    = time_test[int_mask]
        e_int    = event_test[int_mask]
        t_ext    = time_ext[ext_mask]
        e_ext    = event_ext[ext_mask]
        n_int    = int_mask.sum()
        n_ext    = ext_mask.sum()
        if n_int < 2 or n_ext < 2:
            print(f"  {grp} risk: insufficient patients in one cohort - skipped")
            continue
        try:
            lr    = logrank_test(t_int, t_ext, e_int, e_ext)
            p     = lr.p_value
            p_str = f'{p:.4f}' if p >= 0.0001 else '<0.0001'
            sig   = '***' if p < 0.001 else ('**' if p < 0.01
                     else ('*' if p < 0.05 else 'ns'))
            print(f"  {grp} risk - Internal n={n_int} vs External n={n_ext}: "
                  f"p = {p_str}  {sig}")
        except Exception as ex:
            print(f"  {grp} risk: log-rank failed ({ex})")

    print("\n  A non-significant p-value indicates similar survival within risk group,")
    print("  supporting external generalisability of the PSI-2 stratification.")

    print("FINAL SUMMARY (C-index, 95% CI)")
    print(f"{'Model':<12} {'Internal':<28} {'External'}")
    print(f"{'PSI-2':<12} {psi2_c_test:.2f} [{ci_lo:.2f}-{ci_hi:.2f}]"
          f"          {psi2_c_ext:.2f} [{ci_lo_e:.2f}-{ci_hi_e:.2f}]")
    print(f"{'RSF':<12} {rsf_c_test:.2f} [{ci_lo_r:.2f}-{ci_hi_r:.2f}]"
          f"          {rsf_c_ext:.2f} [{ci_lo_re:.2f}-{ci_hi_re:.2f}]")
    if ds_net and ds_c_test is not None:
        print(f"{'DeepSurv':<12} {ds_c_test:.2f} [{ci_lo_d:.2f}-{ci_hi_d:.2f}]"
              f"          {ds_c_ext:.2f} [{ci_lo_de:.2f}-{ci_hi_de:.2f}]")

    print("TIME-DEPENDENT AUC SUMMARY")
    all_times = sorted(set(
        psi2_test_metrics['auc_times'] + rsf_test_metrics['auc_times']
    ))
    header = f"{'Model':<12} {'Dataset':<12}"
    for t in all_times:
        header += f"  AUC@{t}m"
    header += "  MeanAUC"
    print(header)
    print("-" * (len(header) + 5))

    def fmt_row(label, dataset, metrics):
        row = f"{label:<12} {dataset:<12}"
        for t in all_times:
            if t in metrics['auc_times']:
                idx = metrics['auc_times'].index(t)
                row += f"  {metrics['auc_vals'][idx]:.3f}  "
            else:
                row += "   n/a   "
        row += f"  {metrics['mean_auc']:.3f}" if not np.isnan(metrics['mean_auc']) else "   n/a"
        return row

    print(fmt_row('PSI-2',    'Internal', psi2_test_metrics))
    print(fmt_row('RSF',      'Internal', rsf_test_metrics))
    if ds_net:
        print(fmt_row('DeepSurv', 'Internal', ds_test_metrics))
    print(fmt_row('PSI-2',    'External', psi2_ext_metrics))
    print(fmt_row('RSF',      'External', rsf_ext_metrics))
    if ds_net:
        print(fmt_row('DeepSurv', 'External', ds_ext_metrics))

    print("INTEGRATED BRIER SCORE (IBS) SUMMARY")
    print(f"{'Model':<12} {'Internal IBS':<18} {'External IBS'}")
    print(f"{'PSI-2':<12} {psi2_test_metrics['ibs']:<18.3f} {psi2_ext_metrics['ibs']:.3f}")
    print(f"{'RSF':<12} {rsf_test_metrics['ibs']:<18.3f} {rsf_ext_metrics['ibs']:.3f}")
    if ds_net:
        print(f"{'DeepSurv':<12} {ds_test_metrics['ibs']:<18.3f} {ds_ext_metrics['ibs']:.3f}")

    print("ECONOMIC SUMMARY - TERTILE STRATIFICATION")
    print(f"{'Approach':<30} {'Dataset':<25} {'Observed £':>14} {'Mean MC £':>14} {'95% CI'}")
    for lbl, dataset, cost in [
        ("Tertile (Low/Int/High)",   f"Internal test (n={len(df_test)})",      internal_cost_tertile),
        ("Tertile (Low/Int/High)",   f"Full internal (n={len(df_internal)})",   full_cost_tertile),
        ("Tertile (Low/Int/High)",   f"External (n={len(df_external)})",        external_cost_tertile),
    ]:
        ci = f"£{cost['ci_low']:,.0f} - £{cost['ci_high']:,.0f}"
        print(f"  {lbl:<28} {dataset:<25} £{cost['observed_savings']:>10,.0f}   "
              f"£{cost['mean_savings']:>10,.0f}   {ci}")

    print("\nNotes:")
    print("  PSI-2 thresholds (LNR cutoff, tertiles) derived")
    print("  from training split only. Fixed thresholds applied to all datasets.")
    print("  AUC uses IPCW with training data as censoring reference.")
    print("  IBS computed via KM-based survival matrix (decile binning).")
    print("  All variables drawn from the 16-feature ML model set.")
    print("ANALYSIS COMPLETE")

if __name__ == "__main__":
    run_complete_analysis()
