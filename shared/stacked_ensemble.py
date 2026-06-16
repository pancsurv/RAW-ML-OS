import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score

warnings.filterwarnings('ignore')

INTERNAL_CSV = r"D:\Users\Downloads\Cleaned_Dataset_for_Analysis.csv"
EXTERNAL_CSV = r"D:\Users\Downloads\cambook_cleaned.csv"

RANDOM_STATE = 42
N_FOLDS      = 5
N_BOOT       = 1000
EVAL_TIMES   = np.array([12, 36, 58], dtype=float)
DS_EPOCHS    = 50
USE_DEEPSURV = True
MAX_FOLLOWUP = 60

SHARED_FEATURES = [
    'age', 'bmi', 'asa', 'adjchemo',
    'log_tumoursize', 'histoT', 'histoN', 'differentiation',
    'LNR', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR',
    'asa_age', 'adjchemo_LNR'
]

PSI_WEIGHTS = {'LNR': 2, 'differentiation': 2, 'histoN': 1, 'histoT': 1, 'adjchemo': -1}

AMSTERDAM_BETA = {
    'age': 0.0247, 'tumour_size_mm': 0.0119, 'ln_ratio': 1.6677,
    'r1': 0.4383, 'poor_diff': 0.4253, 'adj_chemo': -0.5108,
}

DS_PARAMS = {"lr": 0.0001839, "num_nodes": [128, 64],
             "dropout": 0.4485, "weight_decay": 0.0002721, "batch_size": 32}

try:
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH
    from pycox.models.loss import CoxPHLoss
    PYCOX_AVAILABLE = True
except ImportError:
    PYCOX_AVAILABLE = False


def engineer_features(df):
    d = df.copy()
    for col in ['posnodes', 'totnodes', 'histotumoursize', 'asa', 'age', 'adjchemo']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d['LNR'] = np.where(d['totnodes'].fillna(0) > 0, d['posnodes'] / d['totnodes'], 0.0)
    d['log_tumoursize'] = np.log1p(d['histotumoursize'].fillna(0))
    d['asa_age']      = d['asa'] * d['age']
    d['adjchemo_LNR'] = d['adjchemo'] * d['LNR']
    return d


def _recode(df):
    df['adjchemo'] = pd.to_numeric(df['adjchemo'], errors='coerce').map({1: 1, 2: 0, 3: np.nan})
    for c in ['differentiation', 'histoT', 'histoN', 'histoM']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.loc[df.get('differentiation') == 6, 'differentiation'] = np.nan
    df.loc[df.get('histoT') == 5, 'histoT'] = np.nan
    df.loc[df.get('histoN') == 2, 'histoN'] = np.nan
    if 'histoM' in df.columns:
        df.loc[df['histoM'] == 2, 'histoM'] = np.nan
    return df


def load_internal(path):
    df = pd.read_csv(path, na_values=['#NUM!', 'NA', 'NaN', '#DIV/0!'], encoding='latin1')
    df.columns = df.columns.str.strip()
    df['event'] = df['alive_dead'].map({1: 0, 2: 1, 3: 0})
    df = _recode(df)
    df['OS_months'] = pd.to_numeric(df['OS_months'], errors='coerce')
    df.loc[df['OS_months'].isna() & (df['event'] == 0), 'OS_months'] = df['OS_months'].max()
    df = df.dropna(subset=['OS_months', 'event'])
    n0 = len(df)
    df = df[~((df['event'] == 1) & (df['OS_months'] <= 3))]
    df = engineer_features(df).reset_index(drop=True)
    print(f"Internal: {n0} -> {len(df)} after 90-day exclusion, {int(df['event'].sum())} events")
    return df


def load_external(path):
    df = pd.read_csv(path, encoding='latin1', na_values=['#DIV/0!', 'NA', 'N/A', 'NaN', '#NUM!'])
    df.columns = df.columns.str.strip()
    df['OS_months']  = pd.to_numeric(df['OS_months'], errors='coerce')
    df['alive_dead'] = pd.to_numeric(df['alive_dead'], errors='coerce')
    nan_os = df['OS_months'].isna()
    df.loc[nan_os, 'OS_months']  = MAX_FOLLOWUP
    df.loc[nan_os, 'alive_dead'] = 1
    df = df.dropna(subset=['OS_months', 'alive_dead'])
    df['event'] = (df['alive_dead'] == 2).astype(int)
    df = df[~((df['event'] == 1) & (df['OS_months'] <= 3))]
    df['OS_months'] = np.where((df['alive_dead'] == 1) & (df['OS_months'] > MAX_FOLLOWUP),
                               MAX_FOLLOWUP, df['OS_months'])
    df['event'] = np.where((df['alive_dead'] == 1) & (df['OS_months'] == MAX_FOLLOWUP), 0, df['event'])
    df = _recode(df)
    df = engineer_features(df).reset_index(drop=True)
    print(f"External: N={len(df)}, {int(df['event'].sum())} events")
    return df


def derive_lnr_cutoff(df_train, time_train, event_train, target=24):
    from sklearn.metrics import roc_curve
    y = ((event_train == 1) & (time_train <= target)).astype(int)
    if y.sum() < 10:
        return 0.30
    fpr, tpr, thr = roc_curve(y, df_train['LNR'].fillna(0).values)
    return float(thr[int(np.argmax(tpr - fpr))])


def psi_score(df, lnr_cutoff):
    d = df.copy()
    for c in ['LNR', 'differentiation', 'histoN', 'histoT', 'adjchemo']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return (PSI_WEIGHTS['LNR']            * (d['LNR'].fillna(0) > lnr_cutoff).astype(int) +
            PSI_WEIGHTS['differentiation'] * (d['differentiation'].fillna(2) >= 3).astype(int) +
            PSI_WEIGHTS['histoN']          * (d['histoN'].fillna(0) > 0).astype(int) +
            PSI_WEIGHTS['histoT']          * (d['histoT'].fillna(2) >= 3).astype(int) +
            PSI_WEIGHTS['adjchemo']        * (d['adjchemo'].fillna(0) == 1).astype(int)).values


def fit_amsterdam(df_train):
    cols = ['age', 'histotumoursize', 'posnodes', 'totnodes', 'rstatus', 'differentiation', 'adjchemo']
    d = df_train[cols].apply(pd.to_numeric, errors='coerce')
    d['differentiation'] = d['differentiation'].replace(6, np.nan)
    imp = SimpleImputer(strategy='median').fit(d)
    return imp


def amsterdam_score(df, imp):
    cols = ['age', 'histotumoursize', 'posnodes', 'totnodes', 'rstatus', 'differentiation', 'adjchemo']
    d = df[cols].apply(pd.to_numeric, errors='coerce')
    d['differentiation'] = d['differentiation'].replace(6, np.nan)
    d[cols] = imp.transform(d)
    ln_ratio = np.where(d['totnodes'] > 0, np.clip(d['posnodes'] / d['totnodes'], 0, 1), 0.0)
    r1 = (d['rstatus'] > 0).astype(float)
    diff = d['differentiation']
    poor = np.where(diff <= 3, 0.0, np.where(diff == 4, 0.5, np.where(diff == 5, 1.0, 0.0)))
    chemo = d['adjchemo'].clip(0, 1)
    return (AMSTERDAM_BETA['age'] * d['age'] + AMSTERDAM_BETA['tumour_size_mm'] * d['histotumoursize'] +
            AMSTERDAM_BETA['ln_ratio'] * ln_ratio + AMSTERDAM_BETA['r1'] * r1 +
            AMSTERDAM_BETA['poor_diff'] * poor + AMSTERDAM_BETA['adj_chemo'] * chemo).values


def tnm_stage(df):
    d = df.copy()
    for c in ['histoT', 'histoN']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['histoM'] = pd.to_numeric(d['histoM'], errors='coerce').fillna(0) if 'histoM' in d.columns else 0
    d['histoT'] = d['histoT'].fillna(d['histoT'].median())
    d['histoN'] = d['histoN'].fillna(d['histoN'].median())
    T = d['histoT'].round().astype(int).clip(1, 4)
    N = d['histoN'].round().astype(int).clip(0, 2)
    M = (d['histoM'].round().astype(int).clip(0, 1) if hasattr(d['histoM'], 'round')
         else pd.Series(np.zeros(len(d), int)))
    stage = np.ones(len(d), dtype=int)
    for i in range(len(d)):
        t, n, m = T.iloc[i], N.iloc[i], int(M.iloc[i]) if hasattr(M, 'iloc') else 0
        if m == 1:                       stage[i] = 6
        elif t == 4 or n == 2:           stage[i] = 5
        elif n == 1 and t in (1, 2, 3):  stage[i] = 4
        elif t == 3 and n == 0:          stage[i] = 3
        elif t == 2 and n == 0:          stage[i] = 2
        else:                            stage[i] = 1
    return stage.astype(float)


def fit_deepsurv(X_train, dur, evt):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    net = tt.practical.MLPVanilla(in_features=X_train.shape[1], num_nodes=DS_PARAMS["num_nodes"],
                                  out_features=1, batch_norm=True, dropout=DS_PARAMS["dropout"])
    model = CoxPH(net, tt.optim.Adam(lr=DS_PARAMS["lr"], weight_decay=DS_PARAMS["weight_decay"]))
    model.loss = CoxPHLoss()
    xt = torch.tensor(X_train, dtype=torch.float32)
    model.fit(xt, (torch.tensor(dur, dtype=torch.float32), torch.tensor(evt, dtype=torch.float32)),
              batch_size=DS_PARAMS["batch_size"], epochs=DS_EPOCHS, verbose=False)
    return model


def fit_base_models(df_train):
    Xtr_raw = df_train[SHARED_FEATURES].apply(pd.to_numeric, errors='coerce')
    dur = df_train['OS_months'].values.astype(float)
    evt = df_train['event'].values.astype(int)

    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                               random_state=RANDOM_STATE, sample_posterior=False, tol=1e-3)
    Xtr_imp = imputer.fit_transform(Xtr_raw)
    scaler = StandardScaler().fit(Xtr_imp)
    Xtr = scaler.transform(Xtr_imp)
    Xtr_df = pd.DataFrame(Xtr, columns=SHARED_FEATURES)

    cox_df = Xtr_df.copy()
    cox_df['OS_months'] = dur
    cox_df['event'] = evt
    cph = CoxPHFitter(penalizer=0.1).fit(cox_df, 'OS_months', 'event')

    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=5,
                               max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE)
    rsf.fit(Xtr, Surv.from_arrays(evt.astype(bool), dur))

    ds = fit_deepsurv(Xtr, dur, evt) if (USE_DEEPSURV and PYCOX_AVAILABLE) else None

    lnr_cutoff = derive_lnr_cutoff(df_train, dur, evt)
    ams_imp = fit_amsterdam(df_train)

    return {'imputer': imputer, 'scaler': scaler, 'cph': cph, 'rsf': rsf,
            'ds': ds, 'lnr_cutoff': lnr_cutoff, 'ams_imp': ams_imp}


def predict_base(art, df_eval):
    Xev_raw = df_eval[SHARED_FEATURES].apply(pd.to_numeric, errors='coerce')
    Xev = art['scaler'].transform(art['imputer'].transform(Xev_raw))
    Xev_df = pd.DataFrame(Xev, columns=SHARED_FEATURES)

    out = {}
    out['Cox'] = art['cph'].predict_partial_hazard(Xev_df).values.flatten()
    out['RSF'] = art['rsf'].predict(Xev)
    if art['ds'] is not None:
        net = art['ds'].net
        net.eval()
        device = next(net.parameters()).device
        with torch.no_grad():
            xt = torch.tensor(Xev, dtype=torch.float32).to(device)
            out['DeepSurv'] = net(xt).detach().cpu().numpy().ravel()
    out['PSI'] = psi_score(df_eval, art['lnr_cutoff']).astype(float)
    out['Amsterdam'] = amsterdam_score(df_eval, art['ams_imp'])
    out['TNM'] = tnm_stage(df_eval)
    return out


def oof_predictions(df_train):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    keys = ['Cox', 'RSF', 'DeepSurv', 'PSI', 'Amsterdam', 'TNM']
    oof = {k: np.full(len(df_train), np.nan) for k in keys}
    y = df_train['event'].values
    for fold, (tr, va) in enumerate(skf.split(df_train, y), 1):
        print(f"  OOF fold {fold}/{N_FOLDS}")
        art = fit_base_models(df_train.iloc[tr].reset_index(drop=True))
        preds = predict_base(art, df_train.iloc[va].reset_index(drop=True))
        for k, v in preds.items():
            oof[k][va] = v
    oof = {k: v for k, v in oof.items() if not np.all(np.isnan(v))}
    return pd.DataFrame(oof, index=df_train.index)


def bootstrap_c(risk, OS, evt, n_boot=N_BOOT):
    rng = np.random.default_rng(RANDOM_STATE)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(OS), len(OS))
        if evt[idx].sum() < 2:
            continue
        try:
            vals.append(concordance_index(OS[idx], -risk[idx], evt[idx]))
        except Exception:
            continue
    return (np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan,)*3


def td_auc(risk, OS, evt, y_train_ref, times=EVAL_TIMES):
    yt = Surv.from_arrays(evt.astype(bool), OS)
    cap = float(OS.max()) * 0.99
    out = {}
    for t in times[times < cap]:
        val = None
        for ref in (y_train_ref, yt):
            try:
                a, _ = cumulative_dynamic_auc(ref, yt, risk, np.array([t], dtype=float))
                val = round(float(a[0]), 3)
                break
            except Exception:
                continue
        if val is not None:
            out[int(t)] = val
        else:
            print(f"  AUC@{int(t)}m skipped (censoring out of range)")
    return out


def meta_survival_metrics(meta_rsf, X_meta, OS, evt, y_train_ref, times=EVAL_TIMES):
    yt = Surv.from_arrays(evt.astype(bool), OS)
    train_max = float(y_train_ref['time'].max())
    cap = min(float(OS.max()), train_max) * 0.99
    surv_fns = meta_rsf.predict_survival_function(X_meta)
    grid = np.asarray(surv_fns[0].x, dtype=float)
    grid = grid[(grid > float(OS.min())) & (grid < cap)]
    ibs = np.nan
    if len(grid) > 1:
        try:
            S = np.row_stack([fn(grid) for fn in surv_fns])
            ibs = integrated_brier_score(y_train_ref, yt, S, grid)
        except Exception as e:
            print(f"  IBS skipped: {e}")
    brier = {}
    valid = times[(times < cap)]
    if len(valid):
        try:
            S_t = np.row_stack([fn(valid) for fn in surv_fns])
            _, bvals = brier_score(y_train_ref, yt, S_t, valid)
            brier = dict(zip(valid.astype(int), np.round(bvals, 3)))
        except Exception as e:
            print(f"  Brier skipped: {e}")
    return ibs, brier


def calibration_plot(meta_rsf, X_meta, OS, evt, t, fname, title):
    surv_fns = meta_rsf.predict_survival_function(X_meta)
    pred = np.array([fn(t) for fn in surv_fns])
    bins = np.unique(np.percentile(pred, np.linspace(0, 100, 6)))
    idx = np.clip(np.digitize(pred, bins[:-1]) - 1, 0, len(bins) - 2)
    px, oy = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        kmf = KaplanMeierFitter().fit(OS[m], evt[m])
        px.append(pred[m].mean())
        oy.append(float(kmf.survival_function_at_times(t).values[0]))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.scatter(px, oy, s=70, color='steelblue', edgecolors='navy', zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f'Predicted survival at {int(t)} m')
    ax.set_ylabel('Observed (Kaplan-Meier)')
    ax.set_title(title, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def run():
    print("=" * 70)
    print("STACKED ENSEMBLE - random survival forest meta-learner")
    print("=" * 70)

    df_int = load_internal(INTERNAL_CSV)
    df_ext = load_external(EXTERNAL_CSV)

    df_tr, df_te = train_test_split(df_int, test_size=0.2, random_state=RANDOM_STATE,
                                    stratify=df_int['event'])
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)
    print(f"Train n={len(df_tr)} | Internal test n={len(df_te)} | External n={len(df_ext)}")

    print("\nGenerating out-of-fold base predictions on training set...")
    oof = oof_predictions(df_tr)
    cols = list(oof.columns)
    print(f"Base models in ensemble: {cols}")

    dur_tr = df_tr['OS_months'].values.astype(float)
    evt_tr = df_tr['event'].values.astype(int)
    y_train_ref = Surv.from_arrays(evt_tr.astype(bool), dur_tr)

    print("\nTraining meta-learner (RSF) on out-of-fold predictions...")
    meta = RandomSurvivalForest(n_estimators=300, min_samples_leaf=10,
                                max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE)
    meta.fit(oof[cols].values, y_train_ref)

    print("Refitting base models on full training set...")
    art_full = fit_base_models(df_tr)

    results = {}
    for name, df_eval in [('Internal test', df_te), ('External', df_ext)]:
        preds = predict_base(art_full, df_eval)
        Xm = pd.DataFrame({k: preds[k] for k in cols})[cols].values
        OS = df_eval['OS_months'].values.astype(float)
        evt = df_eval['event'].values.astype(int)
        ens_risk = meta.predict(Xm)
        mc, lo, hi = bootstrap_c(ens_risk, OS, evt)
        aucs = td_auc(ens_risk, OS, evt, y_train_ref)
        ibs, brier = meta_survival_metrics(meta, Xm, OS, evt, y_train_ref)
        base_ci = {k: bootstrap_c(preds[k], OS, evt) for k in cols}
        results[name] = {'c': mc, 'lo': lo, 'hi': hi, 'auc': aucs,
                         'ibs': ibs, 'brier': brier,
                         'base_c': {k: v[0] for k, v in base_ci.items()},
                         'base_ci': base_ci}
        calibration_plot(meta, Xm, OS, evt, 12.0,
                         f'ensemble_calibration_{name.split()[0].lower()}.png',
                         f'Ensemble calibration at 12 m - {name}')

    print("\nComputing meta-learner permutation importance (internal test)...")
    preds_te = predict_base(art_full, df_te)
    Xm_te = pd.DataFrame({k: preds_te[k] for k in cols})[cols].values
    y_te = Surv.from_arrays(df_te['event'].astype(bool).values, df_te['OS_months'].values.astype(float))
    imp = permutation_importance(meta, Xm_te, y_te, n_repeats=20, random_state=RANDOM_STATE)
    imp_df = pd.DataFrame({'Base model': cols, 'Importance': imp.importances_mean,
                           'SD': imp.importances_std}).sort_values('Importance', ascending=False)

    landmark = 24.0
    y24 = ((df_te['event'].values == 1) & (df_te['OS_months'].values <= landmark)).astype(int)
    rf_imp_df = None
    if len(np.unique(y24)) == 2:
        rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE).fit(Xm_te, y24)
        rf_imp_df = pd.DataFrame({'Base model': cols, 'Importance': rf.feature_importances_}
                                 ).sort_values('Importance', ascending=False)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  Ensemble C-index: {r['c']:.3f} (95% CI {r['lo']:.3f}-{r['hi']:.3f})")
        print(f"  Base C-indices  : " + ", ".join(f"{k}={v:.3f}" for k, v in r['base_c'].items()))
        print(f"  Time-AUC        : " + ", ".join(f"{t}m={a:.3f}" for t, a in r['auc'].items()))
        print(f"  IBS             : {r['ibs']:.3f}")
        if r['brier']:
            print(f"  Brier           : " + ", ".join(f"{t}m={b:.3f}" for t, b in r['brier'].items()))

    print("\nMeta-learner permutation importance (RSF, internal test):")
    print(imp_df.to_string(index=False))
    if rf_imp_df is not None:
        print(f"\nRandom forest classifier importance (24-month landmark):")
        print(rf_imp_df.to_string(index=False))

    imp_df.to_csv('ensemble_meta_importance.csv', index=False)
    summary = pd.DataFrame([{
        'Cohort': n, 'Ensemble_C': r['c'], 'CI_low': r['lo'], 'CI_high': r['hi'],
        'IBS': r['ibs'], **{f'AUC_{t}m': a for t, a in r['auc'].items()}
    } for n, r in results.items()])
    summary.to_csv('ensemble_performance_summary.csv', index=False)
    print("\nSaved: ensemble_meta_importance.csv, ensemble_performance_summary.csv")

    forest_plot(results, cols)
    print("=" * 70)


def forest_plot(results, cols, fname='ensemble_forest_plot.png'):
    label_map = {'Cox': 'Cox PH', 'RSF': 'Random survival forest', 'DeepSurv': 'DeepSurv',
                 'PSI': 'PSI', 'Amsterdam': 'Amsterdam', 'TNM': 'TNM 8th ed.'}
    order = [m for m in ['TNM', 'Amsterdam', 'PSI', 'Cox', 'RSF', 'DeepSurv'] if m in cols]
    rows = [(label_map.get(m, m), m, False) for m in order] + [('Stacked ensemble', 'ENS', True)]
    y = np.arange(len(rows))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, cohort in zip(axes, ['Internal test', 'External']):
        r = results[cohort]
        for yi, (lbl, key, is_ens) in zip(y, rows):
            if is_ens:
                c, lo, hi = r['c'], r['lo'], r['hi']
            else:
                c, lo, hi = r['base_ci'][key]
            color = '#C0392B' if is_ens else '#1A5276'
            ax.errorbar(c, yi, xerr=[[c - lo], [hi - c]], fmt='o', color=color,
                        ecolor=color, elinewidth=2, capsize=4,
                        markersize=9 if is_ens else 7,
                        markeredgecolor='black', markeredgewidth=0.6, zorder=3)
            ax.text(hi + 0.006, yi, f'{c:.2f} ({lo:.2f}-{hi:.2f})',
                    va='center', fontsize=8.5, color='#333333')
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([lbl for lbl, _, _ in rows], fontsize=10)
        ax.set_xlim(0.45, 0.85)
        ax.set_xlabel('C-index (95% CI)', fontsize=11)
        ax.set_title(cohort, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Discrimination of base models and stacked ensemble',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


if __name__ == "__main__":
    run()
