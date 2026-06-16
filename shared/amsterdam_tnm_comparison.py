import os
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torchtuples as tt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score
from sksurv.util import Surv
from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss

warnings.filterwarnings('ignore')

DATA_PATH        = r"D:\Users\Downloads\Cleaned_Dataset_for_Analysis.csv"
COX_MODEL_PATH   = "cox_model.pkl"
COX_SCALER_PATH  = "scaler_cox.save"
COX_IMPUTER_PATH = "imputer_cox.joblib"
RSF_MODEL_PATH   = "rsf_model.joblib"


RSF_SCALER_PATH  = "scaler_rsf.joblib"
RSF_IMPUTER_PATH = "imputer_rsf.joblib"
DS_WEIGHTS_PATH  = "model_weights_blh.pickle"
DS_SCALER_PATH   = "scaler_ds.joblib"
DS_IMPUTER_PATH  = "imputer_ds.joblib"
DS_BASELINE_PATH = "baseline_hazards.csv"
Y_TRAIN_CSV      = "y_train_cox.csv"

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

SHARED_FEATURES = [
    'age', 'bmi', 'asa', 'adjchemo',
    'log_tumoursize', 'histoT', 'histoN', 'differentiation',
    'LNR', 'posnodes', 'rstatus', 'albumin', 'bilirubin', 'NLR',
    'asa_age', 'adjchemo_LNR'
]

RANDOM_STATE = 42
N_BOOT       = 1000
AUC_TIMES    = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 58], dtype=float)

AMSTERDAM_BETA = {
    'age':            0.0247,
    'tumour_size_mm': 0.0119,
    'ln_ratio':       1.6677,
    'r1':             0.4383,
    'poor_diff':      0.4253,
    'adj_chemo':     -0.5108,
}

def load_data():
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
    data = engineer_features(data)
    initial_n = len(data)
    data = data[~((data['event'] == 1) & (data['OS_months'] <= 3))]
    print(f"Dataset: {initial_n} -> {len(data)} after 90-day exclusion, "
          f"{int(data['event'].sum())} events ({data['event'].mean()*100:.1f}%)")
    return data

def get_test_indices(data):
    for col in SHARED_FEATURES:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    _, _, train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        np.arange(len(data)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data['event'].values
    )
    return test_idx

def compute_amsterdam_score(df):
    d = df.copy()
    for col in ['age', 'histotumoursize', 'posnodes', 'totnodes',
                'rstatus', 'differentiation', 'adjchemo']:
        d[col] = pd.to_numeric(d[col], errors='coerce')

    d['differentiation'] = d['differentiation'].replace(6, np.nan)

    imp = SimpleImputer(strategy='median')
    d[['age', 'histotumoursize', 'posnodes', 'totnodes',
       'rstatus', 'differentiation', 'adjchemo']] = imp.fit_transform(
        d[['age', 'histotumoursize', 'posnodes', 'totnodes',
           'rstatus', 'differentiation', 'adjchemo']]
    )

    d['ln_ratio']  = np.where(
        d['totnodes'] > 0,
        np.clip(d['posnodes'] / d['totnodes'], 0, 1),
        0.0
    )
    d['r1']        = (d['rstatus'] > 0).astype(float)

    diff = d['differentiation']
    d['poor_diff'] = np.where(diff <= 3, 0.0,
                    np.where(diff == 4, 0.5,
                    np.where(diff == 5, 1.0, 0.0)))

    d['adj_chemo'] = d['adjchemo'].clip(0, 1)

    score = (
        AMSTERDAM_BETA['age']            * d['age']            +
        AMSTERDAM_BETA['tumour_size_mm'] * d['histotumoursize'] +
        AMSTERDAM_BETA['ln_ratio']       * d['ln_ratio']        +
        AMSTERDAM_BETA['r1']             * d['r1']              +
        AMSTERDAM_BETA['poor_diff']      * d['poor_diff']       +
        AMSTERDAM_BETA['adj_chemo']      * d['adj_chemo']
    )
    return score.values

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

    stage_labels = {1: 'IA', 2: 'IB', 3: 'IIA', 4: 'IIB', 5: 'III', 6: 'IV'}
    return stage, np.array([stage_labels[s] for s in stage])

def get_cox_risks(X_test_df):
    cph     = joblib.load(COX_MODEL_PATH)
    scaler  = joblib.load(COX_SCALER_PATH)
    imputer = joblib.load(COX_IMPUTER_PATH)
    X_imp = imputer.transform(X_test_df[SHARED_FEATURES])
    X_sc  = scaler.transform(X_imp)
    df    = pd.DataFrame(X_sc, columns=SHARED_FEATURES)
    return cph.predict_partial_hazard(df).values.flatten()

def get_rsf_risks(X_test_df):
    rsf     = joblib.load(RSF_MODEL_PATH)

    imputer = joblib.load(RSF_IMPUTER_PATH)
    scaler  = joblib.load(RSF_SCALER_PATH)
    X_imp = imputer.transform(X_test_df[SHARED_FEATURES])
    X_sc  = scaler.transform(X_imp)
    return rsf.predict(pd.DataFrame(X_sc, columns=SHARED_FEATURES))

def get_deepsurv_risks(X_test_df):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imputer = joblib.load(DS_IMPUTER_PATH)
    scaler  = joblib.load(DS_SCALER_PATH)
    X_imp = imputer.transform(X_test_df[SHARED_FEATURES])
    X_sc  = scaler.transform(X_imp)

    net = tt.practical.MLPVanilla(
        in_features=len(SHARED_FEATURES), num_nodes=[128, 64],
        out_features=1, batch_norm=True, dropout=0.4485
    )
    model = CoxPH(net, tt.optim.Adam)
    model.loss = CoxPHLoss()
    model.net.load_state_dict(torch.load(DS_WEIGHTS_PATH, map_location=device))
    model.net.to(device).eval()

    try:
        bh = pd.read_csv(DS_BASELINE_PATH, index_col=0, header=0).iloc[:, 0]
        bh.index = bh.index.astype(float)
    except (ValueError, KeyError):
        bh = pd.read_csv(DS_BASELINE_PATH, index_col=0, header=None).iloc[:, 0]
        bh.index = bh.index.astype(float)
    bh.name  = "baseline_hazards_"
    model.baseline_hazards_ = bh

    with torch.no_grad():
        return model.predict(
            torch.tensor(X_sc, dtype=torch.float32).to(device)
        ).cpu().numpy().ravel()

def _auc_components(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan
    V10 = np.array([(np.sum(p > neg) + 0.5 * np.sum(p == neg)) / n_neg for p in pos])
    V01 = np.array([(np.sum(n < pos) + 0.5 * np.sum(n == pos)) / n_pos for n in neg])
    return V10.mean(), V10, V01

def delong_test(y_true, scores_a, scores_b):
    auc_a, V10_a, V01_a = _auc_components(y_true, scores_a)
    auc_b, V10_b, V01_b = _auc_components(y_true, scores_b)
    if np.isnan(auc_a) or np.isnan(auc_b):
        return auc_a, auc_b, np.nan, np.nan
    S10 = np.cov(np.vstack([V10_a, V10_b]))
    S01 = np.cov(np.vstack([V01_a, V01_b]))
    S   = S10 / len(V10_a) + S01 / len(V01_a)
    L   = np.array([1, -1])
    var = L @ S @ L
    if var <= 0:
        return auc_a, auc_b, np.nan, np.nan
    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return auc_a, auc_b, z, p

def binary_labels_at_t(OS, events, t):
    labels, mask = [], []
    for i in range(len(OS)):
        if events[i] == 1 and OS[i] <= t:
            labels.append(1); mask.append(i)
        elif OS[i] > t:
            labels.append(0); mask.append(i)
    return np.array(labels), np.array(mask)

def run_all_delong(risks_dict, OS, events, test_times):
    model_names = list(risks_dict.keys())
    pairs = [(model_names[i], model_names[j])
             for i in range(len(model_names))
             for j in range(i + 1, len(model_names))]
    n_comparisons = len(pairs) * len(test_times)
    rows = []
    for t in test_times:
        labels, mask = binary_labels_at_t(OS, events, t)
        if len(np.unique(labels)) < 2:
            continue
        for mA, mB in pairs:
            rA = risks_dict[mA][mask]
            rB = risks_dict[mB][mask]
            auc_a, auc_b, z, p = delong_test(labels, rA, rB)
            p_adj = min(p * n_comparisons, 1.0) if not np.isnan(p) else np.nan
            sig = ("***" if p_adj < 0.001 else "**" if p_adj < 0.01
                   else "*" if p_adj < 0.05 else "ns") if not np.isnan(p_adj) else "-"
            rows.append({
                'Time (mo)': int(t), 'Model A': mA, 'Model B': mB,
                'AUC_A': round(auc_a, 3) if not np.isnan(auc_a) else np.nan,
                'AUC_B': round(auc_b, 3) if not np.isnan(auc_b) else np.nan,
                'Z':     round(z, 3)     if not np.isnan(z)     else np.nan,
                'p_raw': round(p, 4)     if not np.isnan(p)     else np.nan,
                'p_Bonf': round(p_adj, 4) if not np.isnan(p_adj) else np.nan,
                'Sig':   sig,
            })
    return pd.DataFrame(rows)

def bootstrap_c_index(risks, OS, events, n_boot=N_BOOT, seed=RANDOM_STATE):
    rng  = np.random.default_rng(seed)
    vals = []
    n    = len(OS)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if events[idx].sum() < 2:
            continue
        try:
            vals.append(concordance_index(OS[idx], -risks[idx], events[idx]))
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan, np.nan
    return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def td_auc_with_ci(risks, OS, events, y_train_ref, times, n_boot=N_BOOT):
    y_test = Surv.from_arrays(events.astype(bool), OS)
    valid  = times[times < OS.max()]
    if not len(valid):
        return {}, {}, {}
    pt_auc, _ = cumulative_dynamic_auc(y_train_ref, y_test, risks, valid)
    point = dict(zip(valid, pt_auc))

    rng = np.random.default_rng(RANDOM_STATE)
    boot = {t: [] for t in valid}
    n = len(OS)
    for _ in range(n_boot):
        idx   = rng.integers(0, n, n)
        y_b   = Surv.from_arrays(events[idx].astype(bool), OS[idx])
        vt    = valid[valid < OS[idx].max()]
        if not len(vt):
            continue
        try:
            b_auc, _ = cumulative_dynamic_auc(y_train_ref, y_b, risks[idx], vt)
            for t, a in zip(vt, b_auc):
                boot[t].append(a)
        except Exception:
            continue
    lo = {t: np.percentile(boot[t], 2.5)  if boot[t] else np.nan for t in valid}
    hi = {t: np.percentile(boot[t], 97.5) if boot[t] else np.nan for t in valid}
    return point, lo, hi

def plot_tnm_km(OS, events, stage_labels):
    stage_order  = ['IA', 'IB', 'IIA', 'IIB', 'III', 'IV']
    stage_colours = {
        'IA': '#1A5276', 'IB': '#2980B9', 'IIA': '#1E8449',
        'IIB': '#F39C12', 'III': '#E67E22', 'IV': '#C0392B'
    }
    present = [s for s in stage_order if s in stage_labels]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    ax.spines[['top', 'right']].set_visible(False)

    kmf = KaplanMeierFitter()
    for s in present:
        mask = stage_labels == s
        n    = mask.sum()
        if n < 3:
            continue
        kmf.fit(OS[mask], events[mask], label=f'{s} (n={n})')
        kmf.plot_survival_function(ax=ax, ci_show=True, color=stage_colours[s],
                                   linewidth=2.0)

    try:
        mlr = multivariate_logrank_test(OS, stage_labels, events)
        p_val = mlr.p_value
        ax.set_title(f'Kaplan-Meier by TNM 8th Edition Stage\nLog-rank p = {p_val:.4f}',
                     fontsize=13, fontweight='bold')
    except Exception:
        ax.set_title('Kaplan-Meier by TNM 8th Edition Stage', fontsize=13, fontweight='bold')

    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig('km_tnm_stage.png', dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    print("Saved: km_tnm_stage.png")
    plt.show()

def plot_combined_auc_5models(auc_data, delong_df, delong_times):
    PALETTE = {
        'DeepSurv':  '#C0392B',
        'RSF':       '#1A5276',
        'Cox':       '#1E8449',
        'Amsterdam': '#8E44AD',
        'TNM':       '#E67E22',
    }
    FILL = {
        'DeepSurv':  '#F1948A',
        'RSF':       '#7FB3D3',
        'Cox':       '#82E0AA',
        'Amsterdam': '#D7BDE2',
        'TNM':       '#FAD7A0',
    }
    LINE = {
        'DeepSurv': '-',  'RSF': '-', 'Cox': '-',
        'Amsterdam': '--', 'TNM': ':'
    }

    fig = plt.figure(figsize=(13, 11), facecolor='#FAFAFA')
    gs  = GridSpec(2, 1, height_ratios=[3, 1.4], hspace=0.5, figure=fig)
    ax_main  = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    ax_main.set_facecolor('#FAFAFA')
    ax_main.spines[['top', 'right']].set_visible(False)
    ax_main.spines[['left', 'bottom']].set_color('#CCCCCC')

    for model, d in auc_data.items():
        ts  = np.array(d['times'])
        auc = np.array(d['auc'])
        lo  = np.array(d['lo'])
        hi  = np.array(d['hi'])
        ax_main.plot(ts, auc, LINE[model], color=PALETTE[model],
                     linewidth=2.5, markersize=5, marker='o',
                     label=model, zorder=3)
        ax_main.fill_between(ts, lo, hi, alpha=0.13, color=FILL[model], zorder=2)

    ax_main.axhline(0.5, color='#BBBBBB', linestyle=':', linewidth=1.2, zorder=1)

    pairs_to_annotate = [
        ('DeepSurv', 'Amsterdam'), ('DeepSurv', 'TNM'),
        ('RSF', 'Amsterdam'),      ('Cox', 'TNM'),
    ]
    for t in delong_times:
        ax_main.axvline(t, color='#E8E8E8', linestyle='--', linewidth=0.8, zorder=0)
        lines = []
        for mA, mB in pairs_to_annotate:
            sub = delong_df[(delong_df['Time (mo)'] == t) &
                            (((delong_df['Model A'] == mA) & (delong_df['Model B'] == mB)) |
                             ((delong_df['Model A'] == mB) & (delong_df['Model B'] == mA)))]
            if not sub.empty:
                sig = sub.iloc[0]['Sig']
                if sig not in ('ns', '-'):
                    lines.append(f'{mA} vs {mB}: {sig}')
        if lines:
            ax_main.text(t, 1.01, f't={t}m\n' + '\n'.join(lines),
                         ha='center', va='bottom', fontsize=6.5, color='#555555',
                         fontfamily='monospace',
                         transform=ax_main.get_xaxis_transform())

    ax_main.set_xlim(AUC_TIMES[0] - 2, AUC_TIMES[-1] + 2)
    ax_main.set_ylim(0.43, 1.08)
    ax_main.set_xlabel('Time (months)', fontsize=13, color='#333333')
    ax_main.set_ylabel('Time-Dependent AUC', fontsize=13, color='#333333')
    ax_main.set_title(
        'Time-Dependent AUC: ML Models vs Amsterdam Score vs TNM 8th Edition\n'
        'Shaded = 95% bootstrap CI  |  Solid = ML models  |  Dashed/dotted = benchmarks',
        fontsize=12, fontweight='bold', color='#222222', pad=10
    )
    handles = [Line2D([0], [0], color=PALETTE[m], linewidth=2.5, linestyle=LINE[m],
                      marker='o', markersize=5, label=m) for m in auc_data]
    handles.append(mpatches.Patch(color='#CCCCCC', alpha=0.5, label='95% Bootstrap CI'))
    ax_main.legend(handles=handles, fontsize=10, framealpha=0.9,
                   loc='lower right', edgecolor='#CCCCCC')

    ax_table.set_facecolor('#FAFAFA')
    ax_table.axis('off')

    display_pairs = [
        ('DeepSurv', 'Amsterdam'), ('RSF', 'Amsterdam'), ('Cox', 'Amsterdam'),
        ('DeepSurv', 'TNM'),       ('RSF', 'TNM'),       ('Cox', 'TNM'),
    ]
    t_cols    = sorted(delong_df['Time (mo)'].unique())
    col_labels = ['Comparison'] + [f'{int(t)}m' for t in t_cols]
    table_data = []
    for mA, mB in display_pairs:
        row = [f'{mA} vs {mB}']
        for t in t_cols:
            sub = delong_df[
                (delong_df['Time (mo)'] == t) &
                (((delong_df['Model A'] == mA) & (delong_df['Model B'] == mB)) |
                 ((delong_df['Model A'] == mB) & (delong_df['Model B'] == mA)))
            ]
            if sub.empty:
                row.append('-')
            else:
                r = sub.iloc[0]
                row.append(f"p={r['p_Bonf']:.3f} {r['Sig']}" if not np.isnan(r['p_Bonf']) else '-')
        table_data.append(row)

    tbl = ax_table.table(cellText=table_data, colLabels=col_labels,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2C3E50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    sig_col = {'***': '#FDEDEC', '**': '#FEF9E7', '*': '#FDFEFE', 'ns': '#F8F9FA', '-': '#F8F9FA'}
    for i, (mA, mB) in enumerate(display_pairs, start=1):
        tbl[i, 0].set_facecolor('#EBF5FB')
        tbl[i, 0].set_text_props(fontweight='bold')
        for j, t in enumerate(t_cols, start=1):
            sub = delong_df[
                (delong_df['Time (mo)'] == t) &
                (((delong_df['Model A'] == mA) & (delong_df['Model B'] == mB)) |
                 ((delong_df['Model A'] == mB) & (delong_df['Model B'] == mA)))
            ]
            if not sub.empty:
                tbl[i, j].set_facecolor(sig_col.get(sub.iloc[0]['Sig'], '#F8F9FA'))

    ax_table.set_title(
        'DeLong Pairwise Tests (Bonferroni-corrected)  |  *** p<0.001  ** p<0.01  * p<0.05  ns = not significant',
        fontsize=9.5, color='#444444', pad=6
    )

    plt.savefig('auc_5models_delong.png', dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    print("Saved: auc_5models_delong.png")
    plt.show()

def print_summary_table(risks_dict, OS, events, y_train_ref, valid_times):
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY - ALL MODELS")
    print("="*80)
    print(f"{'Model':<14} {'C-index':>8} {'95% CI Lo':>10} {'95% CI Hi':>10}", end="")
    for t in valid_times:
        print(f"  AUC@{int(t)}m", end="")
    print()
    print("-"*80)

    y_test = Surv.from_arrays(events.astype(bool), OS)
    vt     = valid_times[valid_times < OS.max()]

    for model, risks in risks_dict.items():
        mean_c, lo, hi = bootstrap_c_index(risks, OS, events)
        print(f"{model:<14} {mean_c:>8.3f} {lo:>10.3f} {hi:>10.3f}", end="")
        if len(vt):
            try:
                auc_vals, _ = cumulative_dynamic_auc(
                    y_train_ref,
                    Surv.from_arrays(events.astype(bool), OS),
                    risks, vt
                )
                for a in auc_vals:
                    print(f"  {a:>7.3f}", end="")
            except Exception:
                print("  [AUC error]", end="")
        print()
    print("="*80)

def main():
    print("="*80)
    print("BENCHMARK COMPARISON: ML Models vs Amsterdam Score vs TNM 8th Edition")
    print("="*80)

    data = load_data()

    test_idx = get_test_indices(data)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    OS     = test_data['OS_months'].values.astype(float)
    events = test_data['event'].values.astype(int)
    print(f"\nTest cohort: n={len(OS)}, events={int(events.sum())} "
          f"({events.mean()*100:.1f}%)")

    if os.path.exists(Y_TRAIN_CSV):
        y_tr       = pd.read_csv(Y_TRAIN_CSV)
        y_train_ref = Surv.from_arrays(y_tr['event'].astype(bool), y_tr['OS_months'])
        print(f"Training reference: {len(y_tr)} patients, {int(y_tr['event'].sum())} events")
    else:


        raise FileNotFoundError(
            f"Training survival reference {Y_TRAIN_CSV} is required for "
            f"the Amsterdam/TNM comparison (used as IPCW reference for "
            f"cumulative_dynamic_auc and integrated_brier_score). Run "
            f"cox_internal.py first to generate it."
        )

    print("\n--- Computing benchmark scores ---")
    amsterdam_scores = compute_amsterdam_score(test_data)
    tnm_stages, tnm_labels = compute_tnm_stage(test_data)

    print(f"Amsterdam score range: [{amsterdam_scores.min():.2f}, {amsterdam_scores.max():.2f}]")
    print("TNM stage distribution:")
    for s in ['IA', 'IB', 'IIA', 'IIB', 'III', 'IV']:
        n = (tnm_labels == s).sum()
        if n > 0:
            print(f"  Stage {s}: n={n} ({100*n/len(tnm_labels):.1f}%)")

    print("\n--- Loading ML model risk scores ---")
    risks_dict = {}

    for name, fn in [('DeepSurv', get_deepsurv_risks),
                     ('RSF',      get_rsf_risks),
                     ('Cox',      get_cox_risks)]:
        try:
            r = fn(test_data)
            risks_dict[name] = r
            print(f"  {name}: loaded, range [{r.min():.3f}, {r.max():.3f}]")
        except Exception as e:
            print(f"  {name}: FAILED ({e}) - skipping")

    risks_dict['Amsterdam'] = amsterdam_scores

    risks_dict['TNM'] = tnm_stages.astype(float)

    print("\n--- C-index (1000-iteration bootstrap) ---")
    for model, risks in risks_dict.items():
        mean_c, lo, hi = bootstrap_c_index(risks, OS, events)
        print(f"  {model:<14}: {mean_c:.3f} (95% CI: {lo:.3f} - {hi:.3f})")

    print("\n--- Generating TNM KM plot ---")
    plot_tnm_km(OS, events, tnm_labels)

    print(f"\n--- Time-dependent AUC ({N_BOOT} bootstrap iterations) ---")
    auc_data = {}
    for model, risks in risks_dict.items():
        print(f"  Processing {model}...")
        pt, lo, hi = td_auc_with_ci(risks, OS, events, y_train_ref, AUC_TIMES)
        if pt:
            auc_data[model] = {
                'times': list(pt.keys()),
                'auc':   list(pt.values()),
                'lo':    [lo[t] for t in pt],
                'hi':    [hi[t] for t in pt],
            }

    delong_times = [12, 24, 36, 48]
    print(f"\n--- DeLong pairwise tests at {delong_times} months ---")
    delong_df = run_all_delong(risks_dict, OS, events, delong_times)
    print("\nDeLong Results (Bonferroni-corrected):")

    benchmark_pairs = delong_df[
        delong_df['Model A'].isin(['Amsterdam', 'TNM']) |
        delong_df['Model B'].isin(['Amsterdam', 'TNM'])
    ]
    print(benchmark_pairs[['Time (mo)', 'Model A', 'Model B',
                            'AUC_A', 'AUC_B', 'Z', 'p_raw', 'p_Bonf', 'Sig']].to_string(index=False))

    delong_output = "delong_results_all_models.csv"
    delong_df.to_csv(delong_output, index=False)
    print(f"\nFull DeLong table saved: {delong_output}")

    print("\n--- Generating combined AUC figure ---")
    plot_combined_auc_5models(auc_data, delong_df, delong_times)

    valid_times = np.array([12, 36, 48], dtype=float)
    print_summary_table(risks_dict, OS, events, y_train_ref, valid_times)

    print("\n--- Brier Score and IBS ---")
    vt = valid_times[valid_times < OS.max()]
    y_test_sksurv = Surv.from_arrays(events.astype(bool), OS)
    if len(vt):
        for model, risks in risks_dict.items():

            if model in ('Amsterdam', 'TNM'):
                continue
            try:
                if model == 'Cox':
                    cph     = joblib.load(COX_MODEL_PATH)
                    scaler  = joblib.load(COX_SCALER_PATH)
                    imputer = joblib.load(COX_IMPUTER_PATH)
                    X_imp   = imputer.transform(test_data[SHARED_FEATURES])
                    X_sc    = scaler.transform(X_imp)
                    df_cox  = pd.DataFrame(X_sc, columns=SHARED_FEATURES)
                    surv_fn = cph.predict_survival_function(df_cox, times=vt)
                    probs   = surv_fn.values.T
                    times_bs, bs_vals = brier_score(y_train_ref, y_test_sksurv, probs, vt)
                    all_t   = cph.predict_survival_function(df_cox).index.values
                    all_t   = all_t[all_t < OS.max()]
                    all_p   = cph.predict_survival_function(df_cox, times=all_t).values.T
                    ibs     = integrated_brier_score(y_train_ref, y_test_sksurv, all_p, all_t)
                elif model == 'RSF':
                    rsf     = joblib.load(RSF_MODEL_PATH)

                    imputer = joblib.load(RSF_IMPUTER_PATH)
                    scaler  = joblib.load(RSF_SCALER_PATH)
                    X_imp   = imputer.transform(test_data[SHARED_FEATURES])
                    X_sc    = scaler.transform(X_imp)
                    X_df    = pd.DataFrame(X_sc, columns=SHARED_FEATURES)
                    surv_fns = rsf.predict_survival_function(X_df)
                    probs    = np.array([[fn(t) for t in vt] for fn in surv_fns])
                    times_bs, bs_vals = brier_score(y_train_ref, y_test_sksurv, probs, vt)
                    all_t_rsf = surv_fns[0].x
                    all_t_rsf = all_t_rsf[all_t_rsf < OS.max()]
                    all_p_rsf = np.array([[fn(t) for t in all_t_rsf] for fn in surv_fns])
                    ibs       = integrated_brier_score(y_train_ref, y_test_sksurv, all_p_rsf, all_t_rsf)
                else:
                    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    imputer = joblib.load(DS_IMPUTER_PATH)
                    scaler  = joblib.load(DS_SCALER_PATH)
                    X_imp   = imputer.transform(test_data[SHARED_FEATURES])
                    X_sc    = scaler.transform(X_imp)
                    net = tt.practical.MLPVanilla(
                        in_features=len(SHARED_FEATURES), num_nodes=[128, 64],
                        out_features=1, batch_norm=True, dropout=0.4485
                    )
                    ds_model = CoxPH(net, tt.optim.Adam)
                    ds_model.loss = CoxPHLoss()
                    ds_model.net.load_state_dict(
                        torch.load(DS_WEIGHTS_PATH, map_location=device)
                    )
                    ds_model.net.to(device).eval()

                    try:
                        bh = pd.read_csv(DS_BASELINE_PATH, index_col=0, header=0).iloc[:, 0]
                        bh.index = bh.index.astype(float)
                    except (ValueError, KeyError):
                        bh = pd.read_csv(DS_BASELINE_PATH, index_col=0, header=None).iloc[:, 0]
                        bh.index = bh.index.astype(float)
                    bh_times = bh.index.values.astype(float)
                    H0       = np.cumsum(bh.values.astype(float))
                    x_t      = torch.tensor(X_sc, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        ds_risks = ds_model.net(x_t).cpu().numpy().ravel()
                    exp_risk    = np.exp(ds_risks)
                    surv_matrix = np.exp(-np.outer(exp_risk, H0))
                    def _interp_ds(eval_times):
                        return np.column_stack([
                            np.array([np.interp(t, bh_times, surv_matrix[i])
                                      for i in range(len(ds_risks))])
                            for t in eval_times
                        ])
                    probs     = _interp_ds(vt)
                    times_bs, bs_vals = brier_score(y_train_ref, y_test_sksurv, probs, vt)
                    all_t_ds  = bh_times[bh_times < OS.max()]
                    all_p_ds  = _interp_ds(all_t_ds)
                    ibs       = integrated_brier_score(y_train_ref, y_test_sksurv, all_p_ds, all_t_ds)

                print(f"  {model}:")
                for t, s in zip(times_bs, bs_vals):
                    print(f"    Brier@{int(t)}m = {s:.3f}")
                print(f"    IBS = {ibs:.3f}")

            except Exception as e:
                print(f"  {model}: Brier/IBS failed - {e}")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)

if __name__ == "__main__":
    main()
