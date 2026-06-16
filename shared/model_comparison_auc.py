import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

warnings.filterwarnings('ignore')

DATA_PATH          = r"D:\Users\Downloads\Cleaned_Dataset_for_Analysis.csv"
COX_MODEL_PATH     = "cox_model.pkl"
COX_SCALER_PATH    = "scaler_cox.save"
COX_IMPUTER_PATH   = "imputer_cox.joblib"
RSF_MODEL_PATH     = "rsf_model.joblib"


RSF_SCALER_PATH    = "scaler_rsf.joblib"
RSF_IMPUTER_PATH   = "imputer_rsf.joblib"
DS_WEIGHTS_PATH    = "model_weights_blh.pickle"
DS_SCALER_PATH     = "scaler_ds.joblib"
DS_IMPUTER_PATH    = "imputer_ds.joblib"
DS_BASELINE_PATH   = "baseline_hazards.csv"

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

SHARED_FEATURES = [
    'age', 'bmi', 'asa', 'adjchemo',
    'log_tumoursize', 'histoT', 'histoN', 'differentiation',
    'LNR', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR',
    'asa_age', 'adjchemo_LNR'
]

COX_FEATURES = SHARED_FEATURES

N_BOOT        = 1000
RANDOM_STATE  = 42
AUC_TIMES     = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 58], dtype=float)

def _auc_components(y_true, y_score):
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos = y_score[pos_mask]
    neg = y_score[neg_mask]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan

    V10 = np.array([(np.sum(p > neg) + 0.5 * np.sum(p == neg)) / n_neg for p in pos])
    V01 = np.array([(np.sum(n < pos) + 0.5 * np.sum(n == pos)) / n_pos for n in neg])
    auc = V10.mean()
    return auc, V10, V01

def delong_test(y_true, scores_a, scores_b):
    from scipy import stats

    auc_a, V10_a, V01_a = _auc_components(y_true, scores_a)
    auc_b, V10_b, V01_b = _auc_components(y_true, scores_b)

    if np.isnan(auc_a) or np.isnan(auc_b):
        return auc_a, auc_b, np.nan, np.nan

    n_pos = len(V10_a)
    n_neg = len(V01_a)

    S10 = np.cov(np.vstack([V10_a, V10_b]))
    S01 = np.cov(np.vstack([V01_a, V01_b]))

    S = S10 / n_pos + S01 / n_neg

    L = np.array([1, -1])
    var_diff = L @ S @ L
    if var_diff <= 0:
        return auc_a, auc_b, np.nan, np.nan

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))
    return auc_a, auc_b, z, p

def _binary_labels_at_t(OS, events, t):
    labels, mask = [], []
    for i in range(len(OS)):
        if events[i] == 1 and OS[i] <= t:
            labels.append(1); mask.append(i)
        elif OS[i] > t:
            labels.append(0); mask.append(i)
    return np.array(labels), np.array(mask)

def load_and_prepare_data():
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
    data['OS_months'] = pd.to_numeric(data['OS_months'], errors='coerce')
    data.loc[data['OS_months'].isna() & (data['event'] == 0), 'OS_months'] = MAX_FUP
    data = data.dropna(subset=['OS_months', 'event'])
    over_fup = data['OS_months'] > MAX_FUP
    data.loc[over_fup, 'event']     = 0
    data.loc[over_fup, 'OS_months'] = MAX_FUP

    data = data[~((data['event'] == 1) & (data['OS_months'] <= 3))]

    data = engineer_features(data)
    print(f"Dataset: {len(data)} patients, {int(data['event'].sum())} events "
          f"({data['event'].mean()*100:.1f}%)")
    return data

def get_test_split(data, features):

    data = engineer_features(data)
    raw_cols = ['age', 'bmi', 'asa', 'adjchemo',
                'histotumoursize', 'histoT', 'histoN', 'differentiation',
                'totnodes', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR']
    for col in raw_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    X = data[features].copy()
    y = data[['OS_months', 'event']]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y['event']
    )
    return X_test, y_test

def compute_tnm_stage(df):
    d = df.copy()
    for col in ['histoT', 'histoN']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    if 'histoM' in d.columns:
        d['histoM'] = pd.to_numeric(d['histoM'], errors='coerce').fillna(0)
    else:
        d['histoM'] = 0

    d['histoT'] = d['histoT'].fillna(d['histoT'].median())
    d['histoN'] = d['histoN'].fillna(d['histoN'].median())

    T = d['histoT'].round().astype(int).clip(1, 4)
    N = d['histoN'].round().astype(int).clip(0, 2)
    M = d['histoM'].round().astype(int).clip(0, 1)

    stage = np.zeros(len(d), dtype=int)
    for i in range(len(d)):
        t, n, m = T.iloc[i], N.iloc[i], M.iloc[i]
        if m == 1:
            stage[i] = 6
        elif t == 4 or n == 2:
            stage[i] = 5
        elif n == 1 and t in [1, 2, 3]:
            stage[i] = 4
        elif t == 3 and n == 0:
            stage[i] = 3
        elif t == 2 and n == 0:
            stage[i] = 2
        else:
            stage[i] = 1
    return stage.astype(float)

def get_tnm_risks(data, y_test):
    _, data_test = train_test_split(
        data, test_size=0.2, random_state=RANDOM_STATE,
        stratify=data['event']
    )
    data_test = data_test.reset_index(drop=True)
    return compute_tnm_stage(data_test)

def get_cox_risks(X_test, y_test):
    cph     = joblib.load(COX_MODEL_PATH)
    scaler  = joblib.load(COX_SCALER_PATH)
    imputer = joblib.load(COX_IMPUTER_PATH)

    X_imp = imputer.transform(X_test)
    X_sc  = scaler.transform(X_imp)

    df = pd.DataFrame(X_sc, columns=X_test.columns)
    df['OS_months'] = y_test['OS_months'].values
    df['event']     = y_test['event'].values

    risks = cph.predict_partial_hazard(df).values.flatten()
    return risks

def get_rsf_risks(X_test, y_test):
    rsf     = joblib.load(RSF_MODEL_PATH)

    imputer = joblib.load(RSF_IMPUTER_PATH)
    scaler  = joblib.load(RSF_SCALER_PATH)

    X_imp = imputer.transform(X_test)
    X_sc  = scaler.transform(X_imp)
    X_df  = pd.DataFrame(X_sc, columns=X_test.columns)

    return rsf.predict(X_df)

def get_deepsurv_risks(X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imputer = joblib.load(DS_IMPUTER_PATH)
    scaler  = joblib.load(DS_SCALER_PATH)

    X_imp = imputer.transform(X_test)
    X_sc  = scaler.transform(X_imp)

    net = tt.practical.MLPVanilla(
        in_features=X_sc.shape[1],
        num_nodes=[128, 64],
        out_features=1,
        batch_norm=True,
        dropout=0.4485,
    )
    model = CoxPH(net, tt.optim.Adam)
    model.loss = CoxPHLoss()
    state = torch.load(DS_WEIGHTS_PATH, map_location=device)
    model.net.load_state_dict(state)
    model.net.to(device).eval()

    try:
        bh_series = pd.read_csv(DS_BASELINE_PATH, index_col=0, header=0).iloc[:, 0]
        bh_series.index = bh_series.index.astype(float)
    except (ValueError, KeyError):
        bh_series = pd.read_csv(DS_BASELINE_PATH, index_col=0, header=None).iloc[:, 0]
        bh_series.index = bh_series.index.astype(float)
    bh_series.name = "baseline_hazards_"
    model.baseline_hazards_ = bh_series

    x_t = torch.tensor(X_sc, dtype=torch.float32).to(device)
    with torch.no_grad():
        risks = model.predict(x_t).cpu().numpy().ravel()

    return risks

def compute_td_auc_with_ci(risks, OS, events, y_train_sksurv, times, n_boot=N_BOOT):
    y_test_sksurv = Surv.from_arrays(events.astype(bool), OS)

    valid_times = times[times < OS.max()]
    if len(valid_times) == 0:
        return {}, {}, {}

    auc_point, _ = cumulative_dynamic_auc(y_train_sksurv, y_test_sksurv, risks, valid_times)
    point_estimates = {t: a for t, a in zip(valid_times, auc_point)}

    rng = np.random.default_rng(RANDOM_STATE)
    boot_aucs = {t: [] for t in valid_times}
    n = len(OS)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        OS_b    = OS[idx]
        evt_b   = events[idx]
        risks_b = risks[idx]
        y_b     = Surv.from_arrays(evt_b.astype(bool), OS_b)
        try:
            bt = valid_times[valid_times < OS_b.max()]
            if len(bt) == 0:
                continue
            b_auc, _ = cumulative_dynamic_auc(y_train_sksurv, y_b, risks_b, bt)
            for t, a in zip(bt, b_auc):
                boot_aucs[t].append(a)
        except Exception:
            continue

    ci_lo = {t: np.percentile(boot_aucs[t], 2.5)  if boot_aucs[t] else np.nan for t in valid_times}
    ci_hi = {t: np.percentile(boot_aucs[t], 97.5) if boot_aucs[t] else np.nan for t in valid_times}

    return point_estimates, ci_lo, ci_hi

def run_delong_comparison(risks_dict, OS, events, test_times):
    model_names = list(risks_dict.keys())
    rows = []

    for t in test_times:
        labels, mask = _binary_labels_at_t(OS, events, t)
        if len(np.unique(labels)) < 2:
            continue

        pairs = [(model_names[i], model_names[j])
                 for i in range(len(model_names))
                 for j in range(i + 1, len(model_names))]

        for mA, mB in pairs:
            r_a = risks_dict[mA][mask]
            r_b = risks_dict[mB][mask]
            auc_a, auc_b, z, p = delong_test(labels, r_a, r_b)

            n_comparisons = len(pairs) * len(test_times)
            p_adj = min(p * n_comparisons, 1.0) if not np.isnan(p) else np.nan

            sig = ""
            if not np.isnan(p_adj):
                if p_adj < 0.001: sig = "***"
                elif p_adj < 0.01: sig = "**"
                elif p_adj < 0.05: sig = "*"
                else:              sig = "ns"

            rows.append({
                'Time (mo)': int(t),
                'Model A':   mA,
                'Model B':   mB,
                'AUC_A':     round(auc_a, 4) if not np.isnan(auc_a) else np.nan,
                'AUC_B':     round(auc_b, 4) if not np.isnan(auc_b) else np.nan,
                'Z':         round(z, 3)      if not np.isnan(z)     else np.nan,
                'p (raw)':   round(p, 4)      if not np.isnan(p)     else np.nan,
                'p (Bonf.)': round(p_adj, 4)  if not np.isnan(p_adj) else np.nan,
                'Sig.':      sig
            })

    return pd.DataFrame(rows)

def plot_combined_auc(auc_data, delong_df, test_times_for_annotation):
    PALETTE = {
        'DeepSurv': '#C0392B',
        'RSF':      '#1A5276',
        'Cox':      '#1E8449',
        'TNM':      '#7D6608',
    }
    FILL = {
        'DeepSurv': '#F1948A',
        'RSF':      '#7FB3D3',
        'Cox':      '#82E0AA',
        'TNM':      '#F9E79F',
    }

    fig, (ax_main, ax_table) = plt.subplots(
        2, 1, figsize=(11, 10),
        gridspec_kw={'height_ratios': [3, 1.2]},
        facecolor='#FAFAFA'
    )
    fig.subplots_adjust(hspace=0.45)

    ax_main.set_facecolor('#FAFAFA')
    ax_main.spines[['top', 'right']].set_visible(False)
    ax_main.spines[['left', 'bottom']].set_color('#CCCCCC')
    ax_main.tick_params(colors='#444444', labelsize=11)

    for model, d in auc_data.items():
        ts  = np.array(d['times'])
        auc = np.array(d['auc'])
        lo  = np.array(d['lo'])
        hi  = np.array(d['hi'])

        linestyle = '--' if model == 'TNM' else '-'
        ax_main.plot(ts, auc, 'o' + linestyle, color=PALETTE[model],
                     linewidth=2.5, markersize=6, label=model, zorder=3)
        ax_main.fill_between(ts, lo, hi, alpha=0.18,
                              color=FILL[model], zorder=2)

    ax_main.axhline(0.5, color='#AAAAAA', linestyle=':', linewidth=1.2, zorder=1)

    y_annot_base = 0.96
    for t in test_times_for_annotation:
        sub = delong_df[delong_df['Time (mo)'] == t]
        if sub.empty:
            continue
        annot_lines = []
        for _, row in sub.iterrows():
            sig = row['Sig.']
            annot_lines.append(f"{row['Model A']} vs {row['Model B']}: {sig}")
        txt = f"t={t}m\n" + "\n".join(annot_lines)
        ax_main.axvline(t, color='#DDDDDD', linestyle='--', linewidth=0.8, zorder=0)
        ax_main.text(t, y_annot_base, txt, ha='center', va='top', fontsize=7.5,
                     color='#555555', fontfamily='monospace',
                     transform=ax_main.get_xaxis_transform())
        y_annot_base = 0.96 if y_annot_base < 0.96 else 0.96

    ax_main.set_xlim(AUC_TIMES[0] - 2, AUC_TIMES[-1] + 2)
    ax_main.set_ylim(0.45, 1.05)
    ax_main.set_xlabel('Time (months)', fontsize=13, color='#333333')
    ax_main.set_ylabel('Time-Dependent AUC', fontsize=13, color='#333333')
    ax_main.set_title(
        'Time-Dependent AUC: DeepSurv vs RSF vs Cox vs TNM\n'
        '(Shaded regions = 95% bootstrap CI, dashed line = TNM benchmark)',
        fontsize=13, fontweight='bold', color='#222222', pad=12
    )

    legend_handles = [
        Line2D([0], [0], color=PALETTE[m], linewidth=2.5,
               marker='o', markersize=6, label=m)
        for m in auc_data.keys()
    ]
    legend_handles.append(
        mpatches.Patch(color='#CCCCCC', alpha=0.5, label='95% Bootstrap CI')
    )
    ax_main.legend(handles=legend_handles, fontsize=11, framealpha=0.9,
                   loc='lower right', edgecolor='#CCCCCC')

    ax_table.set_facecolor('#FAFAFA')
    ax_table.axis('off')

    table_times = sorted(delong_df['Time (mo)'].unique())

    show_pairs = [('DeepSurv', 'RSF'), ('DeepSurv', 'Cox'), ('RSF', 'Cox')]
    col_labels = ['Comparison'] + [f'{int(t)}m' for t in table_times]
    table_data = []

    for mA, mB in show_pairs:
        row_data = [f'{mA} vs {mB}']
        for t in table_times:
            sub = delong_df[(delong_df['Time (mo)'] == t) &
                            (delong_df['Model A'] == mA) &
                            (delong_df['Model B'] == mB)]
            if sub.empty:
                row_data.append('-')
            else:
                r = sub.iloc[0]
                p_val = r['p (Bonf.)']
                sig   = r['Sig.']
                row_data.append(f"p={p_val:.3f} {sig}" if not np.isnan(p_val) else '-')
        table_data.append(row_data)

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2C3E50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    sig_colours = {'***': '#FDEDEC', '**': '#FEF9E7', '*': '#FDFEFE', 'ns': '#F8F9FA'}
    for i, (mA, mB) in enumerate(show_pairs, start=1):
        for j, t in enumerate(table_times, start=1):
            sub = delong_df[(delong_df['Time (mo)'] == t) &
                            (delong_df['Model A'] == mA) &
                            (delong_df['Model B'] == mB)]
            if not sub.empty:
                s = sub.iloc[0]['Sig.']
                tbl[i, j].set_facecolor(sig_colours.get(s, '#F8F9FA'))
        tbl[i, 0].set_facecolor('#EBF5FB')
        tbl[i, 0].set_text_props(fontweight='bold')

    ax_table.set_title(
        'DeLong Pairwise Tests (Bonferroni-corrected p-values)\n'
        '*** p<0.001   ** p<0.01   * p<0.05   ns = not significant',
        fontsize=10, color='#444444', pad=8
    )

    plt.savefig('combined_auc_delong.png', dpi=300, bbox_inches='tight',
                facecolor='#FAFAFA')
    print("Figure saved: combined_auc_delong.png")
    plt.show()

def main():
    print("="*70)
    print("COMBINED AUC COMPARISON: DeepSurv vs RSF vs Cox vs TNM")
    print("="*70)

    data = load_and_prepare_data()

    X_test_shared, y_test_shared = get_test_split(data, SHARED_FEATURES)
    X_test_cox,    y_test_cox    = X_test_shared, y_test_shared

    OS     = y_test_shared['OS_months'].values.astype(float)
    events = y_test_shared['event'].values.astype(int)

    import os
    y_train_path = "y_train_rsf.csv"
    if os.path.exists(y_train_path):
        y_tr = pd.read_csv(y_train_path)
        y_train_ref = Surv.from_arrays(y_tr['event'].astype(bool), y_tr['OS_months'])
    else:

        _, X_tr_raw, _, y_tr_raw = train_test_split(
            data[SHARED_FEATURES], data[['OS_months', 'event']],
            test_size=0.2, random_state=RANDOM_STATE,
            stratify=data['event']
        )
        y_train_ref = Surv.from_arrays(
            y_tr_raw['event'].astype(bool).values,
            y_tr_raw['OS_months'].values
        )

    print("\nLoading model risk scores...")
    try:
        cox_risks = get_cox_risks(X_test_cox.reset_index(drop=True),
                                  y_test_cox.reset_index(drop=True))
        print(f"  Cox risks:     n={len(cox_risks)}, "
              f"range [{cox_risks.min():.3f}, {cox_risks.max():.3f}]")
    except Exception as e:
        raise RuntimeError(
            f"Cox risk scores could not be computed: {e}. "
            f"Aborting rather than substituting placeholder values."
        ) from e

    try:
        rsf_risks = get_rsf_risks(X_test_shared.reset_index(drop=True),
                                  y_test_shared.reset_index(drop=True))
        print(f"  RSF risks:     n={len(rsf_risks)}, "
              f"range [{rsf_risks.min():.3f}, {rsf_risks.max():.3f}]")
    except Exception as e:
        raise RuntimeError(
            f"RSF risk scores could not be computed: {e}. "
            f"Aborting rather than substituting placeholder values."
        ) from e

    try:
        ds_risks = get_deepsurv_risks(X_test_shared.reset_index(drop=True),
                                      y_test_shared.reset_index(drop=True))
        print(f"  DeepSurv risks: n={len(ds_risks)}, "
              f"range [{ds_risks.min():.3f}, {ds_risks.max():.3f}]")
    except Exception as e:
        raise RuntimeError(
            f"DeepSurv risk scores could not be computed: {e}. "
            f"Aborting rather than substituting placeholder values."
        ) from e

    try:
        tnm_risks = get_tnm_risks(data, y_test_shared)
        print(f"  TNM risks:      n={len(tnm_risks)}, "
              f"range [{tnm_risks.min():.3f}, {tnm_risks.max():.3f}]")
    except Exception as e:
        raise RuntimeError(
            f"TNM risk scores could not be computed: {e}. "
            f"Aborting rather than substituting placeholder values."
        ) from e

    n = min(len(OS), len(cox_risks), len(rsf_risks), len(ds_risks), len(tnm_risks))
    OS        = OS[:n]
    events    = events[:n]
    cox_risks  = cox_risks[:n]
    rsf_risks  = rsf_risks[:n]
    ds_risks   = ds_risks[:n]
    tnm_risks  = tnm_risks[:n]

    risks_dict = {'DeepSurv': ds_risks, 'RSF': rsf_risks,
                  'Cox': cox_risks, 'TNM': tnm_risks}

    print(f"\nComputing time-dependent AUC at {AUC_TIMES} months "
          f"with {N_BOOT} bootstrap iterations...")

    auc_data = {}
    for name, risks in risks_dict.items():
        print(f"  Processing {name}...")
        pt, lo, hi = compute_td_auc_with_ci(risks, OS, events, y_train_ref, AUC_TIMES)
        if pt:
            auc_data[name] = {
                'times': list(pt.keys()),
                'auc':   list(pt.values()),
                'lo':    [lo[t] for t in pt],
                'hi':    [hi[t] for t in pt],
            }

    delong_times = [12, 24, 36, 48]
    print(f"\nRunning pairwise DeLong tests at {delong_times} months...")
    delong_df = run_delong_comparison(risks_dict, OS, events, delong_times)

    print("\nDeLong Test Results (Bonferroni-corrected):")
    print(delong_df.to_string(index=False))

    print("\nGenerating combined AUC figure...")
    plot_combined_auc(auc_data, delong_df, delong_times)

    print("\n" + "="*70)
    print("POINT-ESTIMATE AUC SUMMARY")
    print("="*70)
    summary_rows = []
    for model, d in auc_data.items():
        for t, a, lo_, hi_ in zip(d['times'], d['auc'], d['lo'], d['hi']):
            summary_rows.append({'Model': model, 'Time (mo)': int(t),
                                  'AUC': round(a, 3),
                                  '95% CI Lo': round(lo_, 3),
                                  '95% CI Hi': round(hi_, 3)})
    summary_df = pd.DataFrame(summary_rows)
    pivot = summary_df.pivot_table(index='Time (mo)', columns='Model',
                                    values='AUC', aggfunc='first')
    print(pivot.to_string())

    print("\nDone.")

if __name__ == "__main__":
    main()
