import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
import traceback
import shap
import matplotlib.pyplot as plt

from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from sksurv.metrics import brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)

CFG = {
    "nodes": [128, 64],
    "dropout": 0.4485,
    "max_followup": 60,
}

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

class SurvivalPredictor:
    def __init__(self, weights: str, scaler: str, imputer: str,
                 baseline_hazards: str, y_train_csv: str,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.scaler:  StandardScaler = joblib.load(scaler)
        self.imputer: SimpleImputer  = joblib.load(imputer)
        self.features: List[str]     = FEATURES

        imputer_cols = getattr(self.imputer, "feature_names_in_", self.features)
        if set(imputer_cols) != set(self.features):
            raise RuntimeError(
                f"Feature mismatch. Imputer: {list(imputer_cols)}, "
                f"Expected: {self.features}"
            )

        net = tt.practical.MLPVanilla(
            in_features=len(self.features),
            num_nodes=CFG["nodes"],
            out_features=1,
            batch_norm=True,
            dropout=CFG["dropout"],
        )
        self.model = CoxPH(net, tt.optim.Adam)
        self.model.loss = CoxPHLoss()
        state = torch.load(weights, map_location=self.device)
        self.model.net.load_state_dict(state)
        self.model.net.to(self.device).eval()

        logging.info(f"Loading baseline hazards from: {baseline_hazards}")

        try:
            bh_df = pd.read_csv(baseline_hazards, index_col=0, header=0)
            bh_series = bh_df.iloc[:, 0]
            bh_series.index = bh_series.index.astype(float)
        except (ValueError, KeyError):
            bh_df = pd.read_csv(baseline_hazards, index_col=0, header=None)
            bh_series = bh_df.iloc[:, 0]
            bh_series.index = bh_series.index.astype(float)
        bh_series.name = "baseline_hazards_"
        self.model.baseline_hazards_ = bh_series
        logging.info(
            f"Baseline hazards loaded: {len(bh_series)} time points, "
            f"range [{bh_series.index.min():.1f}, {bh_series.index.max():.1f}] months."
        )
        logging.info("Model loaded successfully.")

        if os.path.exists(y_train_csv):
            y_tr = pd.read_csv(y_train_csv)
            self.y_train_ref = Surv.from_arrays(
                y_tr['event'].astype(bool).values,
                y_tr['OS_months'].values
            )
            logging.info(f"Training reference loaded: {len(y_tr)} patients, "
                         f"{int(y_tr['event'].sum())} events.")
        else:


            raise FileNotFoundError(
                f"Training survival reference {y_train_csv} is required for "
                f"external validation (used as IPCW reference for "
                f"cumulative_dynamic_auc and integrated_brier_score). "
                f"Run deepsurv_internal.py first to generate it. External "
                f"AUC/Brier cannot be computed without this file."
            )

    def _prepare(self, df: pd.DataFrame):
        df = df.copy()
        df.columns = df.columns.str.strip()



        required_cols = self.features + ["OS_months", "alive_dead"]

        df["OS_months"]  = pd.to_numeric(df["OS_months"],  errors="coerce")
        df["alive_dead"] = pd.to_numeric(df["alive_dead"], errors="coerce")



        nan_os = df["OS_months"].isna()
        df.loc[nan_os & (df["alive_dead"] == 1), "OS_months"] = CFG["max_followup"]
        df = df.dropna(subset=["OS_months", "alive_dead"])

        df["event"] = (df["alive_dead"] == 2).astype(int)

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

        df = engineer_features(df)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in external data: {missing}")


        initial_n = len(df)
        df = df[~((df["event"] == 1) & (df["OS_months"] <= 3))]
        logging.info(
            f"Excluded {initial_n - len(df)} patients who died within 90 days "
            f"(same eligibility as training cohort)."
        )


        over_fup = df["OS_months"] > CFG["max_followup"]
        df.loc[over_fup, "OS_months"] = CFG["max_followup"]
        df.loc[over_fup, "event"]     = 0

        X_raw      = df[self.features]
        X_imputed  = pd.DataFrame(self.imputer.transform(X_raw), columns=self.features)
        X_scaled   = self.scaler.transform(X_imputed)

        OS  = df["OS_months"].values.astype("float32")
        evt = df["event"].values.astype("int")

        logging.info(f"Prepared {len(OS)} samples, {evt.sum()} events ({evt.mean()*100:.1f}%).")
        return X_scaled, OS, evt, X_imputed

    def evaluate(self, df: pd.DataFrame) -> Dict:
        X_scaled, OS, evt, X_imputed_df = self._prepare(df)

        fup    = CFG["max_followup"]
        OS_cap = np.minimum(OS, fup)
        evt_cap = np.where(OS > fup, 0, evt)

        x_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            risks = self.model.predict(x_t).cpu().numpy().ravel()

        c_est = concordance_index(OS_cap, -risks, evt_cap)
        rng = np.random.default_rng(42)
        boot_vals = []
        for _ in range(1000):
            idx = rng.integers(len(OS_cap), size=len(OS_cap))
            if evt_cap[idx].sum() > 1:
                try:
                    boot_vals.append(
                        concordance_index(OS_cap[idx], -risks[idx], evt_cap[idx])
                    )
                except Exception:
                    continue
        ci_low = ci_high = np.nan
        if boot_vals:
            ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])
        logging.info(f"C-index: {c_est:.2f} (95% CI: {ci_low:.2f} - {ci_high:.2f})")

        brier_dict, ibs = {}, None
        y_struct    = Surv.from_arrays(evt_cap.astype(bool), OS_cap.astype(float))

        y_train_ref = self.y_train_ref

        try:
            _time_field = y_train_ref.dtype.names[1]
            max_train_time = float(y_train_ref[_time_field].max())
        except Exception:
            max_train_time = float(OS_cap.max())

        max_ext_obs   = float(OS_cap.max())
        max_ext_event = float(OS_cap[evt_cap == 1].max()) if evt_cap.sum() > 0 else max_ext_obs

        time_cap = min(max_ext_obs, max_train_time)
        logging.info(f"Brier/IBS time cap: {time_cap:.2f} months "
                     f"(ext max obs: {max_ext_obs:.1f}, train max: {max_train_time:.1f})")

        brier_times = [t for t in [12.0, 36.0, 58.0] if t < max_ext_obs]
        logging.info(f"Brier eval times: {brier_times}")

        if not brier_times:
            logging.warning("No valid Brier time points within follow-up range - skipping.")
        else:
            try:

                bh       = self.model.baseline_hazards_
                bh_times = bh.index.values.astype(float)
                H0       = np.cumsum(bh.values.astype(float))
                n_pat    = len(risks)

                exp_risk = np.exp(risks.astype(float))

                surv_matrix = np.exp(-np.outer(exp_risk, H0))

                def interp_surv(eval_times):
                    return np.column_stack([
                        np.array([np.interp(t, bh_times, surv_matrix[i])
                                  for i in range(n_pat)])
                        for t in eval_times
                    ])

                surv_probs_brier = interp_surv(brier_times)
                logging.info(f"surv_probs_brier shape: {surv_probs_brier.shape}")
                times_bs, bs_scores = brier_score(
                    y_train_ref, y_struct,
                    surv_probs_brier, np.array(brier_times)
                )
                brier_dict = {str(int(t)): round(float(s), 4)
                              for t, s in zip(times_bs, bs_scores)}
                logging.info(f"Brier scores: {brier_dict}")

                ibs_times = bh_times[(bh_times > 0) & (bh_times < time_cap - 0.01)]
                logging.info(f"IBS grid: {len(ibs_times)} points, "
                             f"[{ibs_times.min():.1f}, {ibs_times.max():.1f}] months")

                if len(ibs_times) > 1:
                    surv_probs_ibs = interp_surv(ibs_times)
                    try:
                        ibs = float(integrated_brier_score(
                            y_train_ref, y_struct, surv_probs_ibs, ibs_times
                        ))
                        logging.info(f"IBS: {ibs:.4f}")
                    except Exception as ibs_err:
                        logging.warning(f"integrated_brier_score failed: {ibs_err} - using trapz")
                        _, bs_full = brier_score(
                            y_train_ref, y_struct, surv_probs_ibs, ibs_times
                        )
                        ibs = float(np.trapz(bs_full, ibs_times) / (ibs_times[-1] - ibs_times[0]))
                        logging.info(f"IBS (trapz): {ibs:.4f}")
                else:
                    logging.warning(f"Only {len(ibs_times)} IBS time points after capping.")

            except Exception as e:
                print(f"BRIER EXCEPTION: {type(e).__name__}: {e}")
                logging.error(f"Brier/IBS block failed: {e}")
                import traceback; traceback.print_exc()

        logging.info("Generating calibration plots...")
        cal_times = [t for t in [12, 36, 58] if t < OS_cap.max()]
        if cal_times and brier_times:
            try:

                fig, axes = plt.subplots(1, len(cal_times), figsize=(6 * len(cal_times), 5))
                if len(cal_times) == 1:
                    axes = [axes]

                for ax, t in zip(axes, cal_times):

                    pred_surv = np.array([np.interp(t, bh_times, surv_matrix[i])
                                          for i in range(n_pat)])

                    n_bins    = 10
                    quantiles = np.percentile(pred_surv, np.linspace(0, 100, n_bins + 1))
                    quantiles = np.unique(quantiles)
                    bin_idx   = np.clip(np.digitize(pred_surv, quantiles[:-1]) - 1,
                                        0, len(quantiles) - 2)

                    pred_mean, obs_vals = [], []
                    for b in range(len(quantiles) - 1):
                        mask = bin_idx == b
                        if mask.sum() < 3:
                            continue
                        pred_mean.append(pred_surv[mask].mean())
                        kmf = KaplanMeierFitter()
                        kmf.fit(OS_cap[mask], evt_cap[mask])
                        obs_vals.append(kmf.survival_function_at_times([t]).values[0])

                    if pred_mean:
                        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
                        ax.scatter(pred_mean, obs_vals, s=70, color='steelblue',
                                   edgecolors='navy', zorder=5, label='Decile bins')
                        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                        ax.set_xlabel('Predicted survival'); ax.set_ylabel('Observed KM survival')
                        ax.set_title(f'Calibration at {t} months', fontweight='bold')
                        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

                plt.suptitle('Calibration Plots - External Cohort (Cambridge)',
                             fontsize=13, fontweight='bold')
                plt.tight_layout(); plt.show()
            except Exception as e:
                logging.warning(f"Calibration plot failed: {e}")

        td_auc_results = {}
        if evt_cap.sum() > 0:
            min_time = OS_cap[evt_cap == 1].min()

            max_cens  = float(OS_cap[evt_cap == 0].max())                        if (evt_cap == 0).sum() > 0 else float(OS_cap.max())
            max_time  = min(max_cens, 58.0)
            if max_time > min_time:
                auc_eval_times = np.linspace(min_time, max_time, num=50)

                working_t, working_a = [], []
                for _t in auc_eval_times:
                    try:
                        _a, _ = cumulative_dynamic_auc(
                            y_struct, y_struct, risks, np.array([_t])
                        )
                        working_t.append(_t); working_a.append(float(_a[0]))
                    except ValueError:
                        pass
                if working_t:
                    auc_scores = np.array(working_a)
                    mean_auc   = float(np.mean(auc_scores))
                    plt.figure(figsize=(8, 6))
                    plt.plot(working_t, auc_scores, marker="o", markersize=3,
                             linestyle='-', label="Time-Dependent AUC")
                    plt.axhline(mean_auc, color="red", linestyle="--",
                                label=f"Mean AUC: {mean_auc:.3f}")
                    plt.xlabel("Time (months)"); plt.ylabel("AUC")
                    plt.title("Time-Dependent AUC - External Cohort")
                    plt.grid(True); plt.legend(); plt.ylim(0, 1.05)
                    plt.tight_layout(); plt.show()

                    specific_times = [t for t in [12, 36, 58] if min_time <= t < max_cens]
                    for t in specific_times:
                        try:
                            sp_a, _ = cumulative_dynamic_auc(
                                y_struct, y_struct, risks, np.array([float(t)])
                            )
                            td_auc_results[str(int(t))] = round(float(sp_a[0]), 4)
                        except ValueError as e:
                            logging.warning(f"AUC at {t}m skipped (IPCW): {e}")

        logging.info("Running SHAP analysis (PermutationExplainer)...")

        def predict_fn_shap(x):
            x_t = torch.tensor(np.asarray(x, dtype=np.float32)).to(self.device)
            with torch.no_grad():
                return self.model.predict(x_t).cpu().numpy().ravel()

        X_shap_arr = np.asarray(X_imputed_df.values, dtype=float)
        bg_idx     = np.random.default_rng(42).choice(len(X_shap_arr),
                                                       min(50, len(X_shap_arr)),
                                                       replace=False)
        bg_data    = X_shap_arr[bg_idx]

        explainer   = shap.explainers.Permutation(
            predict_fn_shap,
            bg_data,
            feature_names=FEATURES
        )
        n_shap    = min(len(X_shap_arr), 99)
        shap_vals = explainer(X_shap_arr[:n_shap],
                              max_evals=2 * len(FEATURES) + 1)
        shap_vals.feature_names = FEATURES

        mean_shap = np.abs(shap_vals.values).mean(axis=0)
        shap_df   = pd.DataFrame({
            'Feature':     FEATURES,
            'Mean_|SHAP|': mean_shap
        }).sort_values('Mean_|SHAP|', ascending=False)

        print("\n" + "="*50)
        print("DeepSurv EXTERNAL - Mean |SHAP| values (ranked)")
        print("="*50)
        print(f"{'Rank':<6}{'Feature':<22}{'Mean |SHAP|':>12}")
        print("-"*40)
        for rank, (_, row) in enumerate(shap_df.iterrows(), 1):
            print(f"{rank:<6}{row['Feature']:<22}{row['Mean_|SHAP|']:>12.4f}")
        print("="*50)
        shap_df.to_csv('ds_external_shap.csv', index=False)

        fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
        shap_sorted = shap_df.sort_values('Mean_|SHAP|')
        ax_bar.barh(shap_sorted['Feature'], shap_sorted['Mean_|SHAP|'],
                    color='#884EA0', alpha=0.8)
        ax_bar.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax_bar.set_title("DeepSurv External - Mean |SHAP| Feature Importance",
                          fontsize=12, fontweight='bold')
        ax_bar.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('ds_external_shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved: ds_external_shap_bar.png")

        try:
            plt.figure(figsize=(10, 7))
            shap.plots.beeswarm(shap_vals, max_display=len(FEATURES), show=False)
            plt.title("DeepSurv External - SHAP Beeswarm",
                      fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig('ds_external_shap_beeswarm.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved: ds_external_shap_beeswarm.png")
        except Exception as e:
            logging.warning(f"Beeswarm skipped: {e}")

        return {
            "n_patients": int(len(OS_cap)),
            "n_events": int(evt_cap.sum()),
            "c_index": {
                "estimate": round(float(c_est), 2),
                "ci95": [round(float(ci_low), 2), round(float(ci_high), 2)]
            },
            "td_auc": td_auc_results,
            "brier": brier_dict,
            "IBS": round(float(ibs), 4) if ibs is not None else None,
        }

if __name__ == "__main__":
    WEIGHTS_PATH   = r"C:\Users\athan\model_weights_blh.pickle"

    SCALER_PATH    = r"C:\Users\athan\scaler_ds.joblib"
    IMPUTER_PATH   = r"C:\Users\athan\imputer_ds.joblib"
    BASELINE_PATH  = r"C:\Users\athan\baseline_hazards.csv"

    Y_TRAIN_CSV    = r"C:\Users\athan\y_train_deepsurv.csv"
    EXTERNAL_CSV   = r"D:\Users\Downloads\cambook_cleaned.csv"

    required_files = [WEIGHTS_PATH, SCALER_PATH, IMPUTER_PATH, BASELINE_PATH, EXTERNAL_CSV]

    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Required files not found: {missing}")

    try:
        df_ext = pd.read_csv(
            EXTERNAL_CSV, encoding="latin1",
            na_values=["#DIV/0!", "NA", "N/A", "NaN", "#NUM!"]
        )
        predictor = SurvivalPredictor(
            weights=WEIGHTS_PATH,
            scaler=SCALER_PATH,
            imputer=IMPUTER_PATH,
            baseline_hazards=BASELINE_PATH,
            y_train_csv=Y_TRAIN_CSV,
        )
        res = predictor.evaluate(df_ext)
        print("\n--- External Validation Results ---")
        print(json.dumps(res, indent=2,
                         default=lambda x: x.item() if isinstance(x, np.generic) else x))
        print("-----------------------------------")
    except Exception as e:
        logging.critical(f"Unhandled error: {e}")
        traceback.print_exc()
