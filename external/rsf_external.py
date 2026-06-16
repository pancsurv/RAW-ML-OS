import warnings
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import os
from sklearn.utils import resample
from lifelines import KaplanMeierFitter

from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    brier_score,
    integrated_brier_score
)
from sksurv.util import Surv

warnings.filterwarnings('ignore', message='X has feature names')
warnings.filterwarnings('ignore', message='X does not have valid feature names')

MODEL_PATH        = 'rsf_model.joblib'

SCALER_PATH       = 'scaler_rsf.joblib'
IMPUTER_PATH      = 'imputer_rsf.joblib'

Y_TRAIN_CSV_PATH  = 'y_train.csv'
EXTERNAL_CSV_PATH = os.path.join(os.environ.get("RAW_DATA_DIR", "."), "cambook_cleaned.csv")

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

FEATURES = [
    'age', 'bmi', 'asa', 'adjchemo',
    'log_tumoursize', 'histoT', 'histoN', 'differentiation',
    'LNR', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR',
    'asa_age', 'adjchemo_LNR'
]

def evaluate_external_rsf(model_path, scaler_path, imputer_path,
                           y_train_csv, external_csv):
    print("Loading model and preprocessors...")

    for path in [model_path, scaler_path, imputer_path, y_train_csv]:
        if not os.path.exists(path):
            return {"error": f"Required file not found: {path}"}

    rsf_model = joblib.load(model_path)
    scaler    = joblib.load(scaler_path)
    imputer   = joblib.load(imputer_path)

    y_train_df = pd.read_csv(y_train_csv)
    y_train_sksurv = Surv.from_arrays(
        y_train_df['event'].astype(bool),
        y_train_df['OS_months']
    )
    print(f"Training reference loaded: {len(y_train_df)} patients, "
          f"{y_train_df['event'].sum()} events.")

    print(f"Loading external data from {external_csv}...")
    try:
        df = pd.read_csv(external_csv, encoding='latin1',
                         na_values=['#DIV/0!', '#NUM!', 'NA', 'NaN'])
    except Exception as e:
        return {"error": f"Could not load CSV: {e}"}

    df.columns    = df.columns.str.strip()
    df["OS_months"]  = pd.to_numeric(df["OS_months"],  errors="coerce")
    df["alive_dead"] = pd.to_numeric(df["alive_dead"], errors="coerce")



    MAX_FUP = 60

    nan_os = df["OS_months"].isna()
    df.loc[nan_os & (df["alive_dead"] == 1), "OS_months"] = MAX_FUP
    df = df.dropna(subset=["OS_months", "alive_dead"])
    df["event"] = (df["alive_dead"] == 2).astype(int)

    over_fup = df["OS_months"] > MAX_FUP
    df.loc[over_fup, "OS_months"] = MAX_FUP
    df.loc[over_fup, "event"]     = 0

    if 'adjchemo' in df.columns:
        df['adjchemo'] = pd.to_numeric(df['adjchemo'], errors='coerce').map({1.0: 1, 2.0: 0, 3.0: float('nan')})

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

    initial_n = len(df)
    df = df[~((df['event'] == 1) & (df['OS_months'] <= 3))]
    print(f"Excluded {initial_n - len(df)} patients who died within 90 days.")

    df = engineer_features(df)

    for col in FEATURES:
        if col not in df.columns:
            print(f"Warning: Feature '{col}' missing - filling with NaN.")
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"External cohort: {len(df)} patients, {int(df['event'].sum())} events "
          f"({df['event'].mean()*100:.1f}%).")

    X_full  = df[FEATURES]
    y_full  = Surv.from_arrays(df['event'].astype(bool), df['OS_months'])

    X_imp   = pd.DataFrame(imputer.transform(X_full), columns=FEATURES)
    X_sc    = pd.DataFrame(scaler.transform(X_imp.values), columns=FEATURES)

    risk_full = rsf_model.predict(X_sc)

    print("\nRunning 1000 bootstrap iterations for C-index...")
    n_bootstraps = 1000
    c_indices    = []

    for i in range(n_bootstraps):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_bootstraps} complete...")

        df_boot      = resample(df, replace=True, random_state=i)
        df_boot = engineer_features(df_boot)
        X_boot       = df_boot[FEATURES]
        y_boot_event = df_boot['event'].astype(bool)
        y_boot_time  = df_boot['OS_months']

        if len(np.unique(y_boot_event)) < 2:
            continue

        X_imp_b = pd.DataFrame(imputer.transform(X_boot), columns=FEATURES)
        X_sc_b  = pd.DataFrame(scaler.transform(X_imp_b.values), columns=FEATURES)

        try:
            r = rsf_model.predict(X_sc_b)
            c = concordance_index_censored(y_boot_event, y_boot_time, r)[0]
            c_indices.append(c)
        except ValueError:
            continue

    mean_c    = np.mean(c_indices)
    ci_lower  = np.percentile(c_indices, 2.5)
    ci_upper  = np.percentile(c_indices, 97.5)
    print(f"\nC-index: {mean_c:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")

    plt.figure(figsize=(10, 5))
    plt.hist(c_indices, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(mean_c,   color='red',    linestyle='--', lw=2, label=f'Mean: {mean_c:.2f}')
    plt.axvline(ci_lower, color='orange', linestyle=':', lw=2,
                label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
    plt.axvline(ci_upper, color='orange', linestyle=':', lw=2)
    plt.xlabel('C-index'); plt.ylabel('Frequency')
    plt.title('Bootstrap C-index Distribution - External RSF (n=1000)', fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    print("\nComputing time-dependent AUC...")
    td_auc_results = {}

    min_t = df['OS_months'][df['event'] == 1].min()
    max_t = df['OS_months'].max()

    from lifelines import KaplanMeierFitter as _KMF
    _kmf_c = _KMF()
    _kmf_c.fit(df['OS_months'], event_observed=(df['event'] == 0))
    _pos_t    = _kmf_c.survival_function_.index[
        _kmf_c.survival_function_.iloc[:, 0] > 0
    ]
    ipcw_cap  = float(_pos_t.max()) if len(_pos_t) > 0 else max_t * 0.95
    ipcw_safe = ipcw_cap * 0.99

    percentile_times = np.percentile(
        df['OS_months'][df['event'] == 1], np.linspace(10, 90, 15)
    )
    specific_times   = np.array([12, 36, 58])
    valid_specific   = specific_times[
        (specific_times >= min_t) & (specific_times < ipcw_safe)
    ]

    percentile_times = percentile_times[percentile_times < ipcw_safe]
    times = np.unique(np.concatenate([percentile_times, valid_specific]))

    max_cens = float(df['OS_months'][df['event'] == 0].max())               if (df['event'] == 0).sum() > 0 else max_t
    times = times[times < max_cens]

    working_t, working_a = [], []
    for _t in times:
        try:


            _a, _ = cumulative_dynamic_auc(
                y_train_sksurv, y_full, risk_full, np.array([_t])
            )
            working_t.append(_t); working_a.append(float(_a[0]))
        except ValueError:
            pass
    times    = np.array(working_t)
    auc      = np.array(working_a)
    mean_auc = float(np.mean(auc)) if len(auc) > 0 else float('nan')
    try:

        if len(times) == 0: raise ValueError('No valid AUC time points')
        auc, mean_auc = auc, mean_auc

        plt.figure(figsize=(8, 5))
        plt.plot(times, auc, marker='o', markersize=3, label=f'Mean AUC: {mean_auc:.3f}')
        plt.axhline(mean_auc, color='red', linestyle='--')
        plt.title('Time-Dependent AUC - External RSF', fontweight='bold')
        plt.xlabel('Time (months)'); plt.ylabel('AUC')
        plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(0, 1.05)
        plt.tight_layout(); plt.show()

        for t in [12, 36, 58]:
            matches = np.where(np.isclose(times, t, atol=0.1))[0]
            if len(matches) > 0:
                val = auc[matches[0]]
                td_auc_results[f"AUC_{t}mo"] = round(float(val), 4)
                print(f"AUC at {t} months: {val:.2f}")
            else:
                td_auc_results[f"AUC_{t}mo"] = "Not calculable (out of range)"

    except Exception as e:
        print(f"AUC error: {e}")

    print("\nComputing Brier scores...")
    brier_results = {}
    ibs_value     = None

    brier_eval_times = np.array(
        [t for t in [12.0, 36.0, 48.0, 58.0] if min_t <= t < max_t]
    )
    if len(brier_eval_times) > 0:
        try:
            surv_funcs = rsf_model.predict_survival_function(X_sc)
            est_probs  = np.array([[fn(t) for t in brier_eval_times] for fn in surv_funcs])

            times_bs, bs_scores = brier_score(
                y_train_sksurv, y_full, est_probs, brier_eval_times
            )
            brier_results = {str(int(t)): round(float(s), 4)
                             for t, s in zip(times_bs, bs_scores)}
            print(f"Brier scores: {brier_results}")

            ibs_value = integrated_brier_score(
                y_train_sksurv, y_full, est_probs, brier_eval_times
            )
            print(f"IBS: {ibs_value:.4f}")
        except Exception as e:
            print(f"Brier/IBS error: {e}")

    print("\nGenerating calibration plots...")
    cal_times = [t for t in [12, 36, 48] if min_t <= t < max_t]
    if cal_times:
        try:
            surv_funcs = rsf_model.predict_survival_function(X_sc)
            fig, axes  = plt.subplots(1, len(cal_times), figsize=(6 * len(cal_times), 5))
            if len(cal_times) == 1:
                axes = [axes]

            for ax, t in zip(axes, cal_times):
                pred_surv = np.array([fn(t) for fn in surv_funcs])
                n_bins    = 10
                quantiles = np.unique(
                    np.percentile(pred_surv, np.linspace(0, 100, n_bins + 1))
                )
                bin_idx = np.clip(
                    np.digitize(pred_surv, quantiles[:-1]) - 1, 0, len(quantiles) - 2
                )
                pred_mean, obs_vals = [], []
                for b in range(len(quantiles) - 1):
                    mask = bin_idx == b
                    if mask.sum() < 3:
                        continue
                    pred_mean.append(pred_surv[mask].mean())
                    kmf = KaplanMeierFitter()
                    kmf.fit(df['OS_months'].values[mask], df['event'].values[mask])
                    obs_vals.append(kmf.survival_function_at_times([t]).values[0])

                if pred_mean:
                    ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
                    ax.scatter(pred_mean, obs_vals, s=70, color='steelblue',
                               edgecolors='navy', zorder=5, label='Decile bins')
                    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                    ax.set_xlabel('Predicted survival'); ax.set_ylabel('Observed KM survival')
                    ax.set_title(f'Calibration at {t} months', fontweight='bold')
                    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

            plt.suptitle('Calibration Plots - External Cohort (RSF)', fontsize=13, fontweight='bold')
            plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"Calibration plot error: {e}")

    print("\nGenerating SHAP analysis (PermutationExplainer)...")
    try:
        X_shap = np.asarray(X_sc, dtype=float)
        bg_idx = np.random.default_rng(42).choice(len(X_shap),
                                                   min(50, len(X_shap)),
                                                   replace=False)
        bg_data = X_shap[bg_idx]

        def rsf_predict_shap(X):
            return rsf_model.predict(np.asarray(X, dtype=float))

        explainer = shap.explainers.Permutation(
            rsf_predict_shap,
            bg_data,
            feature_names=FEATURES
        )
        n_shap = min(len(X_shap), 99)
        shap_vals = explainer(X_shap[:n_shap],
                              max_evals=2 * len(FEATURES) + 1)
        shap_vals.feature_names = FEATURES

        mean_shap = np.abs(shap_vals.values).mean(axis=0)
        shap_df = pd.DataFrame({
            'Feature':     FEATURES,
            'Mean_|SHAP|': mean_shap
        }).sort_values('Mean_|SHAP|', ascending=False)

        print("\n" + "="*50)
        print("RSF EXTERNAL - Mean |SHAP| values (ranked)")
        print("="*50)
        print(f"{'Rank':<6}{'Feature':<22}{'Mean |SHAP|':>12}")
        print("-"*40)
        for rank, (_, row) in enumerate(shap_df.iterrows(), 1):
            print(f"{rank:<6}{row['Feature']:<22}{row['Mean_|SHAP|']:>12.4f}")
        print("="*50)
        shap_df.to_csv('rsf_external_shap.csv', index=False)

        fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
        shap_sorted = shap_df.sort_values('Mean_|SHAP|')
        ax_bar.barh(shap_sorted['Feature'], shap_sorted['Mean_|SHAP|'],
                    color='#117A65', alpha=0.8)
        ax_bar.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax_bar.set_title("RSF External - Mean |SHAP| Feature Importance",
                          fontsize=12, fontweight='bold')
        ax_bar.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('rsf_external_shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: rsf_external_shap_bar.png")

        try:
            plt.figure(figsize=(10, 7))
            shap.plots.beeswarm(shap_vals, max_display=len(FEATURES), show=False)
            plt.title("RSF External - SHAP Beeswarm", fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig('rsf_external_shap_beeswarm.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved: rsf_external_shap_beeswarm.png")
        except Exception as e:
            print(f"Beeswarm skipped: {e}")

    except Exception as e:
        print(f"SHAP error: {e}")

    return {
        "n_patients":   int(len(df)),
        "n_events":     int(df['event'].sum()),
        "c_index_mean": round(float(mean_c),   2),
        "c_index_ci":   [round(float(ci_lower), 2), round(float(ci_upper), 2)],
        "td_auc":       td_auc_results,
        "brier":        brier_results,
        "IBS":          round(float(ibs_value), 4) if ibs_value is not None else None,
    }

if __name__ == "__main__":
    res = evaluate_external_rsf(
        MODEL_PATH, SCALER_PATH, IMPUTER_PATH,
        Y_TRAIN_CSV_PATH, EXTERNAL_CSV_PATH
    )
    print("\n--- External RSF Validation Results ---")
    print(json.dumps(res, indent=2))
    print("---------------------------------------")
