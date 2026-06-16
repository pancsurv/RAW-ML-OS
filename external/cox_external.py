import os
import logging
import traceback
import joblib
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from sksurv.metrics import (
    brier_score, cumulative_dynamic_auc, integrated_brier_score
)
from sksurv.util import Surv
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)

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

MAX_FUP = 60

def bootstrapped_c_index(durations, events, scores, n_iter=1000, seed=42):
    rng  = np.random.default_rng(seed)
    n    = len(durations)
    vals = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        if events[idx].sum() < 2:
            continue
        try:
            vals.append(concordance_index(durations[idx], scores[idx], events[idx]))
        except Exception:
            continue
    if not vals:
        return np.nan, (np.nan, np.nan)
    return np.mean(vals), tuple(np.percentile(vals, [2.5, 97.5]))

def main():
    try:
        COX_MODEL_PATH = r"C:\Users\athan\cox_model.pkl"
        SCALER_PATH    = "scaler_cox.save"
        IMPUTER_PATH   = "imputer_cox.joblib"
        Y_TRAIN_CSV    = "y_train_cox.csv"
        EXTERNAL_CSV   = r"D:\Users\Downloads\cambook_cleaned.csv"

        for p in [COX_MODEL_PATH, SCALER_PATH, IMPUTER_PATH, EXTERNAL_CSV]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required file not found: {p}")

        cph     = joblib.load(COX_MODEL_PATH)
        scaler  = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)

        if os.path.exists(Y_TRAIN_CSV):
            y_tr = pd.read_csv(Y_TRAIN_CSV)
            y_train_ref = Surv.from_arrays(y_tr['event'].astype(bool), y_tr['OS_months'])
            logging.info(f"Training reference: {len(y_tr)} patients, {y_tr['event'].sum()} events.")
        else:


            raise FileNotFoundError(
                f"Training survival reference {Y_TRAIN_CSV} is required for "
                f"external validation (used as IPCW reference for "
                f"cumulative_dynamic_auc). Run cox_internal.py first to "
                f"generate it. External AUC/Brier cannot be computed without "
                f"this file."
            )

        logging.info(f"Loading external dataset: {EXTERNAL_CSV}")
        df = pd.read_csv(EXTERNAL_CSV, encoding='latin1',
                         na_values=['#DIV/0!', 'NA', 'N/A', 'NaN', '#NUM!'])
        df.columns = df.columns.str.strip()

        df["OS_months"]  = pd.to_numeric(df["OS_months"],  errors="coerce")
        df["alive_dead"] = pd.to_numeric(df["alive_dead"], errors="coerce")




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
        df = df[~((df["event"] == 1) & (df["OS_months"] <= 3))]
        logging.info(f"Excluded {initial_n - len(df)} patients who died within 90 days.")

        df = engineer_features(df)

        for f in FEATURES:
            if f not in df.columns:
                logging.warning(f"Feature '{f}' missing - filling with NaN.")
                df[f] = np.nan
            df[f] = pd.to_numeric(df[f], errors='coerce')

        logging.info(f"External cohort: {len(df)} patients, "
                     f"{int(df['event'].sum())} events ({df['event'].mean()*100:.1f}%).")

        X_imp    = imputer.transform(df[FEATURES])
        X_scaled = scaler.transform(X_imp)

        X_df       = pd.DataFrame(X_scaled, columns=FEATURES)
        durations  = df["OS_months"].values
        events_arr = df["event"].values

        risk_scores = cph.predict_partial_hazard(X_df).values.flatten()

        c_est, (ci_lo, ci_hi) = bootstrapped_c_index(
            durations, events_arr, -risk_scores, n_iter=1000
        )
        logging.info(f"C-index: {c_est:.2f} (95% CI: {ci_lo:.2f} - {ci_hi:.2f})")

        y_ext = Surv.from_arrays(events_arr.astype(bool), durations)

        ref   = y_train_ref

        eval_times  = [12, 36, 58]

        from lifelines import KaplanMeierFitter
        kmf_cens = KaplanMeierFitter()

        kmf_cens.fit(durations, event_observed=(events_arr == 0))
        cens_sf   = kmf_cens.survival_function_

        pos_times = cens_sf.index[cens_sf.iloc[:, 0] > 0]
        ipcw_cap  = float(pos_times.max()) if len(pos_times) > 0 else float(durations.max()) * 0.95

        ipcw_safe = ipcw_cap * 0.99
        valid_times = [t for t in eval_times if t < ipcw_safe]
        logging.info(f"AUC eval times: {valid_times} (censoring KM > 0 up to {ipcw_cap:.1f} months, safe cap: {ipcw_safe:.1f})")

        if valid_times:

            max_cens = float(durations[events_arr == 0].max())                       if (events_arr == 0).sum() > 0 else float(durations.max())
            safe_times = [t for t in valid_times if t < max_cens]
            logging.info(f"IPCW-safe eval times (< {max_cens:.1f}m): {safe_times}")
            td_auc_results = {}
            working_times, working_aucs = [], []
            for t in safe_times:
                try:


                    a_vals, _ = cumulative_dynamic_auc(
                        ref, y_ext, risk_scores, np.array([t], dtype=float)
                    )
                    working_times.append(t)
                    working_aucs.append(float(a_vals[0]))
                    logging.info(f"  AUC at {t} months: {a_vals[0]:.2f}")
                except ValueError as ipcw_err:
                    logging.warning(f"  AUC at {t} months skipped (IPCW): {ipcw_err}")
            if working_times:
                vt_arr    = np.array(working_times, dtype=float)
                auc_vals  = np.array(working_aucs)
                mean_auc  = float(np.mean(auc_vals))
                valid_times = working_times
                logging.info(f"  Mean AUC: {mean_auc:.2f}")
            else:
                vt_arr   = np.array([], dtype=float)
                auc_vals = np.array([])
                mean_auc = float('nan')
                logging.warning("No AUC time points could be computed due to IPCW constraints.")

            surv_df    = cph.predict_survival_function(X_df)
            t_idx      = surv_df.index.values
            probs      = surv_df.values
            brier_eval = np.array([t for t in [12.0, 36.0, 58.0]
                                   if t < durations.max()], dtype=float)

            surv_probs = np.column_stack([
                np.array([np.interp(t, t_idx, probs[:, i]) for i in range(probs.shape[1])])
                for t in brier_eval
            ])

            times_bs, bs_vals = brier_score(ref, y_ext, surv_probs, brier_eval)
            brier_dict = {str(int(t)): round(float(s), 4) for t, s in zip(times_bs, bs_vals)}
            logging.info(f"Brier scores: {brier_dict}")

            all_t     = t_idx[t_idx < durations.max()]
            all_probs = np.column_stack([
                np.array([np.interp(t, t_idx, probs[:, i]) for i in range(probs.shape[1])])
                for t in all_t
            ])
            ibs = integrated_brier_score(ref, y_ext, all_probs, all_t)
            logging.info(f"IBS: {ibs:.2f}")

        cal_times = [t for t in [12, 36, 48] if t < durations.max()]
        if cal_times:
            surv_df = cph.predict_survival_function(X_df)
            t_idx   = surv_df.index.values
            probs   = surv_df.values

            fig, axes = plt.subplots(1, len(cal_times), figsize=(6 * len(cal_times), 5))
            if len(cal_times) == 1:
                axes = [axes]

            for ax, t in zip(axes, cal_times):
                pred_surv = np.array([
                    np.interp(t, t_idx, probs[:, i]) for i in range(probs.shape[1])
                ])
                quantiles = np.unique(
                    np.percentile(pred_surv, np.linspace(0, 100, 11))
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
                    kmf.fit(durations[mask], events_arr[mask])
                    obs_vals.append(kmf.survival_function_at_times([t]).values[0])

                if pred_mean:
                    ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
                    ax.scatter(pred_mean, obs_vals, s=70, color='steelblue',
                               edgecolors='navy', zorder=5)
                    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                    ax.set_xlabel('Predicted survival'); ax.set_ylabel('Observed KM survival')
                    ax.set_title(f'Calibration at {t} months', fontweight='bold')
                    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

            plt.suptitle('Calibration - External Cohort (Cox)', fontsize=13, fontweight='bold')
            plt.tight_layout(); plt.show()

        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations, event_observed=events_arr)
        plt.figure(figsize=(8, 5))
        kmf.plot_survival_function()
        plt.title("KM Survival Curve - External Cohort (Cox)", fontsize=13, fontweight='bold')
        plt.xlabel("Time (months)"); plt.ylabel("Survival Probability")
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

        print("\n--- External Validation Complete ---")
        print(f"N={len(durations)} | Events={int(events_arr.sum())}")
        print(f"C-index: {c_est:.2f} (95% CI: {ci_lo:.2f} - {ci_hi:.2f})")

    except Exception as e:
        logging.critical(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
