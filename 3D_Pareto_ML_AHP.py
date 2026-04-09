import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
import argparse
import joblib
import glob
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Plot style (English-only)
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ---------------------------------------------------------
# 2. 数据读取与预处理
# ---------------------------------------------------------

def _try_paths(default_name="IR×FER66.xlsx"):
    """在脚本目录/父目录/上传目录中探测数据文件。"""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(here, default_name),
        os.path.join(os.path.dirname(here), default_name),
    ]
    # 兼容本环境的上传目录（用户上传文件通常在 /workspace/.uploads）
    cands.extend(sorted(glob.glob(os.path.join("/workspace/.uploads", f"*{default_name}"))))
    return [p for p in cands if os.path.exists(p)]


def load_data(file_path=None):
    if not file_path:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surrogate_dataset.csv')
    df = pd.read_csv(file_path)
    df['Fert_abs'] = df['FERCUM']
    return df


def _parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--input-xlsx", default=None, help="数据源 Excel 路径（IR×FER66.xlsx）")
    parser.add_argument("--out-dir", default=None, help="输出目录（默认脚本所在目录）")
    parser.add_argument("--weight-mode", default=None, choices=["entropy", "cv"], help="客观权重模式：entropy 或 cv（默认沿用脚本变量 WEIGHT_MODE）")
    parser.add_argument("--reliability-min", type=float, default=None, help="保证率阈值（默认沿用脚本变量 RELIABILITY_MIN）")
    parser.add_argument("--no-reliability-filter", action="store_true", help="禁用保证率过滤（不推荐）")
    return parser.parse_known_args()

_args, _unknown = _parse_args()
OUT_DIR = _args.out_dir or os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

print("正在读取数据...")
df = load_data(_args.input_xlsx)

use_climate_features = True
climate_cols = ['Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN']
RANDOM_STATE = 42
K_CLIMATE = 100
CLIMATE_QS = [0.25, 0.50, 0.75]

ENABLE_RELIABILITY_FILTER = True
RELIABILITY_MIN = 0.80
YIELD_TARGET_FRAC_OF_MAXMEAN = 0.90

WEIGHT_MODE = "entropy"
SUBJECTIVE_WEIGHTS = [0.5, 0.25, 0.25]

# Year-type → preference policy (used ONLY in the ClimateB year-type analysis)
# P25: dry years; P50: normal years; P75: wet years
YEAR_TYPE_POLICY = {
    "P25": "high_yield",
    "P50": "high_yield",
    # Wet years: choose ONE of {"balanced_yield_resource", "resource_saving"}
    "P75": "balanced_yield_resource",
}

# Reliability threshold strategy for year-type selection
# Default = fixed (0.80). "adaptive" can be added later if needed.
RELIABILITY_MODE = "fixed"  # {"fixed","adaptive"}

# Year-type specific reliability constraint for selection only
RELIABILITY_MIN_BY_TAG = {
    "P25": 0.90,
    "P50": 0.80,
    "P75": 0.80,
}

# 命令行覆盖（如提供）
if _args.weight_mode:
    WEIGHT_MODE = str(_args.weight_mode).lower().strip()
if _args.reliability_min is not None:
    RELIABILITY_MIN = float(_args.reliability_min)
if bool(_args.no_reliability_filter):
    ENABLE_RELIABILITY_FILTER = False

drop_cols = ['Fert_abs', 'IRCUM', 'WRR14']
if use_climate_features:
    drop_cols = drop_cols + climate_cols

clim_year_pool = None
if use_climate_features:
    if 'YEAR' in df.columns:
        clim_year_pool = df[['YEAR'] + climate_cols].dropna(subset=['YEAR'] + climate_cols).groupby('YEAR', as_index=False)[climate_cols].mean()
    else:
        clim_year_pool = df[climate_cols].dropna().reset_index(drop=True)

df = df.dropna(subset=drop_cols)
print(f"有效数据样本量: {df.shape[0]}")

feature_cols = ['Fert_abs', 'IRCUM']
if use_climate_features:
    feature_cols = feature_cols + climate_cols

X = df[feature_cols].values
y_yield = df['WRR14'].values

# ---------------------------------------------------------
# 3. 机器学习建模 (随机森林回归)*/
# ---------------------------------------------------------
print("正在训练机器学习模型...")
USE_GPR = False
use_gpr = USE_GPR

rf_yield = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surrogate_model_best_tree.pkl'))
gpr_yield = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surrogate_model_gpr.pkl'))
gpr_scaler = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler_gpr.pkl'))
use_gpr = USE_GPR
# Features order required by surrogate_model_best_tree.pkl: ['FERCUM', 'IRCUM', 'Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN']

# 生成预测网格 (寻找帕累托最优的搜索空间)
fert_grid = np.linspace(max(0, X[:,0].min()), X[:,0].max(), 100)
irr_grid = np.linspace(max(0, X[:,1].min()), X[:,1].max(), 100)
F, I = np.meshgrid(fert_grid, irr_grid)
F_flat = F.ravel()
I_flat = I.ravel()
n_grid = F_flat.size

clim_samples = None
if use_climate_features and (clim_year_pool is not None) and len(clim_year_pool) > 0:
    clim_array = clim_year_pool[climate_cols].to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    if clim_array.shape[0] <= K_CLIMATE:
        clim_samples = clim_array
    else:
        idx = rng.choice(clim_array.shape[0], size=K_CLIMATE, replace=False)
        clim_samples = clim_array[idx]

if use_climate_features and (clim_samples is not None):
    sum_y = np.zeros(n_grid, dtype=float)
    Y_samples = None
    if ENABLE_RELIABILITY_FILTER:
        Y_samples = np.empty((int(clim_samples.shape[0]), n_grid), dtype=float)
    for k in range(clim_samples.shape[0]):
        tmax, tmin, tavg, rain = float(clim_samples[k, 0]), float(clim_samples[k, 1]), float(clim_samples[k, 2]), float(clim_samples[k, 3])
        Xk = np.column_stack([
            F_flat,
            I_flat,
            np.full(n_grid, tmax, dtype=float),
            np.full(n_grid, tmin, dtype=float),
            np.full(n_grid, tavg, dtype=float),
            np.full(n_grid, rain, dtype=float),
        ])
        # Force feature names if necessary, but numpy array is fine if tree model accepts it.
        if use_gpr:
            Xk_gpr = gpr_scaler.transform(pd.DataFrame(Xk, columns=['FERCUM', 'IRCUM', 'Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN']))
            yk = gpr_yield.predict(Xk_gpr)
        else:
            yk = rf_yield.predict(pd.DataFrame(Xk, columns=['FERCUM', 'IRCUM', 'Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN']))
        sum_y += yk
        if Y_samples is not None:
            Y_samples[int(k)] = yk
    pred_yield = sum_y / float(clim_samples.shape[0])
    reliability = None
    target_yield = None
    if Y_samples is not None:
        target_yield = float(YIELD_TARGET_FRAC_OF_MAXMEAN) * float(np.nanmax(pred_yield))
        reliability = np.mean(Y_samples >= target_yield, axis=0)
else:
    if use_climate_features and (clim_year_pool is not None) and len(clim_year_pool) > 0:
        climate_fixed = clim_year_pool[climate_cols].median(numeric_only=True).to_dict()
        X_grid = np.c_[
            F_flat,
            I_flat,
            np.full(n_grid, float(climate_fixed['Acc_TMAX'])),
            np.full(n_grid, float(climate_fixed['Acc_TMIN'])),
            np.full(n_grid, float(climate_fixed['Acc_TAVG'])),
            np.full(n_grid, float(climate_fixed['Acc_RAIN'])),
        ]
    else:
        X_grid = np.c_[F_flat, I_flat]
    if use_gpr:
        X_grid_gpr = gpr_scaler.transform(pd.DataFrame(X_grid, columns=['FERCUM', 'IRCUM', 'Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN']))
        pred_yield = gpr_yield.predict(X_grid_gpr)
    else:
        pred_yield = rf_yield.predict(pd.DataFrame(X_grid, columns=['FERCUM', 'IRCUM', 'Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN']))
    reliability = None
    target_yield = None

# ---------------------------------------------------------
# 4. 提取帕累托前沿 (Pareto Frontier)
# ---------------------------------------------------------
print("正在计算帕累托前沿 (多目标最大化)...")
def identify_pareto(scores):
    # scores: N x 3，我们需要最大化所有目标
    population_size = scores.shape[0]
    pareto_front = np.ones(population_size, dtype=bool)
    for i in range(population_size):
        # 如果存在任何一个点 j，在所有维度上 >= 点 i，且在至少一个维度上 > 点 i
        dominators = np.all(scores >= scores[i], axis=1) & np.any(scores > scores[i], axis=1)
        if np.any(dominators):
            pareto_front[i] = False
    return pareto_front

scores = np.c_[pred_yield, -I_flat, -F_flat]
valid_all = np.all(np.isfinite(scores), axis=1)
if np.any(valid_all):
    pm_valid = identify_pareto(scores[valid_all])
    pareto_mask = np.zeros(n_grid, dtype=bool)
    idx_valid = np.where(valid_all)[0]
    pareto_mask[idx_valid[pm_valid]] = True
else:
    pareto_mask = np.zeros(n_grid, dtype=bool)

pareto_yield = pred_yield[pareto_mask]
pareto_F = F.ravel()[pareto_mask]
pareto_I = I.ravel()[pareto_mask]
pareto_rel = reliability[pareto_mask] if reliability is not None else None

print(f"找到 {np.sum(pareto_mask)} 个帕累托最优点。")

def _minmax_norm(X_mat, high_better):
    X_mat = np.asarray(X_mat, dtype=float)
    X_norm = np.zeros_like(X_mat)
    hb = np.asarray(high_better, dtype=bool)
    if hb.size == 1:
        hb = np.full(X_mat.shape[1], bool(hb.item()), dtype=bool)
    for j in range(X_mat.shape[1]):
        col_min = np.nanmin(X_mat[:, j])
        col_max = np.nanmax(X_mat[:, j])
        denom = col_max - col_min
        if denom == 0:
            X_norm[:, j] = 0.0
        else:
            if hb[j]:
                X_norm[:, j] = (X_mat[:, j] - col_min) / denom
            else:
                X_norm[:, j] = (col_max - X_mat[:, j]) / denom
    return X_norm

def entropy_weights(X_mat, high_better):
    X_norm = _minmax_norm(X_mat, high_better)
    X_norm = np.where(np.isfinite(X_norm), X_norm, np.nan)
    col_sums = np.nansum(X_norm, axis=0)
    P = X_norm / (col_sums + 1e-12)
    P = np.where(P > 0, P, 1e-12)
    n = X_norm.shape[0]
    k = 1.0 / np.log(max(n, 2))
    e = -k * np.nansum(P * np.log(P), axis=0)
    d = 1.0 - e
    w = d / (np.nansum(d) + 1e-12)
    return w

def cv_weights(X_mat, high_better):
    X_norm = _minmax_norm(X_mat, high_better)
    mu = np.nanmean(X_norm, axis=0)
    sd = np.nanstd(X_norm, axis=0, ddof=1) if X_norm.shape[0] >= 2 else np.nanstd(X_norm, axis=0)
    v = sd / (np.abs(mu) + 1e-12)
    v = np.where(np.isfinite(v), v, 0.0)
    s = float(np.sum(v))
    if s <= 0:
        return np.full(X_norm.shape[1], 1.0 / X_norm.shape[1], dtype=float)
    return v / s

def combine_weights_game_theory(w1, w2):
    u1 = np.asarray(w1, dtype=float).reshape(-1)
    u2 = np.asarray(w2, dtype=float).reshape(-1)
    u1 = u1 / (np.sum(u1) + 1e-12)
    u2 = u2 / (np.sum(u2) + 1e-12)
    A = np.array([[float(np.dot(u1, u1)), float(np.dot(u1, u2))],
                  [float(np.dot(u2, u1)), float(np.dot(u2, u2))]], dtype=float)
    b = np.array([float(np.dot(u1, u1)), float(np.dot(u2, u2))], dtype=float)
    try:
        alpha = np.linalg.solve(A, b)
    except Exception:
        alpha = np.array([0.5, 0.5], dtype=float)
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)
    alpha = np.maximum(alpha, 0.0)
    alpha = alpha / (np.sum(alpha) + 1e-12)
    w = alpha[0] * u1 + alpha[1] * u2
    w = np.maximum(w, 0.0)
    w = w / (np.sum(w) + 1e-12)
    return w


def ahp_weights(A):
    """
    AHP 主观权重（特征向量法）+ 一致性检验 CR。
    A: n×n 成对比较矩阵（Saaty 1–9 标度及其倒数）
    返回: (w, CR, CI, lambda_max)
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("AHP 矩阵必须为方阵")
    n = int(A.shape[0])
    if n < 1:
        raise ValueError("AHP 矩阵维度非法")
    if np.any(~np.isfinite(A)) or np.any(A <= 0):
        raise ValueError("AHP 矩阵必须为正数且有限")

    # 主特征向量
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))
    lam = float(eigvals.real[idx])
    v = eigvecs[:, idx].real
    v = np.abs(v)
    w = v / (np.sum(v) + 1e-12)

    # 一致性
    if n <= 2:
        return w, 0.0, 0.0, lam
    CI = (lam - n) / (n - 1 + 1e-12)
    RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = float(RI_TABLE.get(n, 1.49))
    CR = float(CI / (RI + 1e-12))
    return w, CR, float(CI), lam


# Preference scenarios (fixed order: Yield, Irrigation, Fertilizer)
# BASE is used for the "all-years" (Climate Expected) selection and plot (keep unchanged).
PREFERENCE_SCENARIOS_BASE = {
    "high_yield": {
        "label": "High yield",
        "name_cn": "高产",
        "ahp_matrix": [
            [1,   6,   6],
            [1/6, 1, 1/2],
            [1/6, 2,   1],
        ],
        "marker": "^",
        "color": "black",
    },
    "water_saving": {
        "label": "Water-saving",
        "name_cn": "节水",
        "ahp_matrix": [
            [1,   1/6, 2],
            [6,   1,   12],
            [1/2, 1/12, 1],
        ],
        "marker": "o",
        "color": "#1f77b4",
},

    "fert_saving": {
        "label": "Fertilizer-saving",
        "name_cn": "节肥",
        "ahp_matrix": [
            [1,   3/4, 1/4],
            [4/3, 1,   1/3],
            [4,   3,   1],
        ],
        "marker": "s",
        "color": "#2ca02c",
        },
    "resource_saving": {
        "label": "Resource-saving",
        "name_cn": "资源节约",
        "ahp_matrix": [
            [1,   1/5, 1/5],
            [5,   1,   1/5],
            [5,   1,   1],
        ],
        "marker": "D",
        "color": "#d62728",
    },
}

# CLIMATE is used for year-type (P25/P50/P75) analysis: BASE + Balanced(Yield+Resource)
PREFERENCE_SCENARIOS_CLIMATE = {
    **PREFERENCE_SCENARIOS_BASE,
    "balanced_yield_resource": {
        "label": "Balanced (Yield+Resource)",
        "ahp_matrix": [
            [1,   2,   2],
            [1/2, 1,   1],
            [1/2, 1,   1],
        ],
        "marker": "X",
        "color": "#9467bd",
    },
}

# Year-type specific AHP matrices (independent across P25/P50/P75).
# Markers/colors/labels are kept consistent for plotting/legend.
# If you want to tune them later, ONLY change the "ahp_matrix" blocks below.
PREFERENCE_SCENARIOS_BY_TAG = {
    # Dry years: emphasize yield security (yield dominates inputs)
    "P25": {
        **PREFERENCE_SCENARIOS_CLIMATE,
        "high_yield": {
            **PREFERENCE_SCENARIOS_CLIMATE["high_yield"],
            "ahp_matrix": [
                [1,   7,   7],
                [1/7, 1,   1],
                [1/7, 1,   1],
            ],
        },
    },
    # Normal years: keep a moderate high-yield preference (close to the base setup)
    "P50": {
        **PREFERENCE_SCENARIOS_CLIMATE,
        "high_yield": {
            **PREFERENCE_SCENARIOS_CLIMATE["high_yield"],
            "ahp_matrix": [
                [1,   4, 4],
                [1/4, 1, 1],
                [1/4, 1, 1],
            ],
        },
    },
    # Wet years: use an independent balanced matrix (yield + resource)
    "P75": {
        **PREFERENCE_SCENARIOS_CLIMATE,
        "balanced_yield_resource": {
            **PREFERENCE_SCENARIOS_CLIMATE["balanced_yield_resource"],
            "ahp_matrix": [
                [1,   1, 2],
                [1,   1, 4],
                [1/4, 1/4, 1],
            ],
        },
    },
}

# Keep the name used by existing all-years logic
PREFERENCE_SCENARIOS = PREFERENCE_SCENARIOS_BASE


def resolve_objective_weights(X_mat, high_better):
    if WEIGHT_MODE.lower() == "cv":
        return cv_weights(X_mat, high_better)
    return entropy_weights(X_mat, high_better)


def resolve_weights_ahp(X_mat, high_better, ahp_matrix):
    """AHP(主观) + 客观权重 → 博弈论组合，返回 (w_sub, CR, w_obj, w_final)"""
    w_sub, CR, CI, lam = ahp_weights(ahp_matrix)
    w_obj = resolve_objective_weights(X_mat, high_better)
    w_final = combine_weights_game_theory(w_sub, w_obj)
    return w_sub, CR, CI, lam, w_obj, w_final

def resolve_weights(X_mat, high_better):
    if WEIGHT_MODE.lower() == "cv":
        w_obj = cv_weights(X_mat, high_better)
    else:
        w_obj = entropy_weights(X_mat, high_better)
    if SUBJECTIVE_WEIGHTS is None:
        return w_obj
    w_sub = np.asarray(SUBJECTIVE_WEIGHTS, dtype=float).reshape(-1)
    if w_sub.size != X_mat.shape[1]:
        return w_obj
    return combine_weights_game_theory(w_sub, w_obj)

def topsis_scores(X_mat, weights, high_better):
    X_mat = np.asarray(X_mat, dtype=float)
    X_norm = _minmax_norm(X_mat, high_better)
    
    Z = X_norm * weights
    z_plus = np.nanmax(Z, axis=0)
    z_minus = np.nanmin(Z, axis=0)
    d_plus = np.sqrt(np.nansum((Z - z_plus) ** 2, axis=1))
    d_minus = np.sqrt(np.nansum((Z - z_minus) ** 2, axis=1))
    c = d_minus / (d_plus + d_minus + 1e-12)
    return c
df_treat = df.dropna(subset=['Fert_abs', 'IRCUM', 'WRR14']).copy()
df_treat = df_treat.groupby(['Fert_abs', 'IRCUM'], as_index=False).agg(
    Fert_abs=('Fert_abs', 'mean'),
    IRCUM=('IRCUM', 'mean'),
    WRR14=('WRR14', 'mean')
)

X_crit = df_treat[['WRR14', 'IRCUM', 'Fert_abs']].values
hb_mask = [True, False, False]
ew_global = resolve_objective_weights(X_crit, hb_mask)
cand_F = pareto_F
cand_I = pareto_I
cand_y = pareto_yield
cand_rel = pareto_rel

# 先做保证率过滤（>=80%）
n_cand_before = int(np.size(cand_y))
reliability_filter_applied = False
if ENABLE_RELIABILITY_FILTER and (cand_rel is not None):
    keep = np.isfinite(cand_rel) & (cand_rel >= float(RELIABILITY_MIN))
    if np.any(keep):
        cand_F = cand_F[keep]
        cand_I = cand_I[keep]
        cand_y = cand_y[keep]
        cand_rel = cand_rel[keep]
        reliability_filter_applied = True

# 若过滤后为空，回退到不过滤（同时在汇总CSV中标记）
if int(np.size(cand_y)) == 0:
    cand_F = pareto_F
    cand_I = pareto_I
    cand_y = pareto_yield
    cand_rel = pareto_rel
    reliability_filter_applied = False
n_cand_after = int(np.size(cand_y))

pareto_crit_cand = np.c_[cand_y, cand_I, cand_F]

# 4个偏好情景：分别 AHP(主观) + 客观权重 → 博弈论组合 → TOPSIS 选点
scenario_results = []
for key, sc in PREFERENCE_SCENARIOS.items():
    w_sub, CR, CI, lam, w_obj, w_final = resolve_weights_ahp(pareto_crit_cand, hb_mask, sc["ahp_matrix"])
    if (not np.all(np.isfinite(w_final))) or (np.sum(w_final) <= 0):
        w_final = np.full(pareto_crit_cand.shape[1], 1.0 / pareto_crit_cand.shape[1], dtype=float)
    cs = topsis_scores(pareto_crit_cand, w_final, hb_mask)
    bidx = int(np.nanargmax(cs)) if np.any(np.isfinite(cs)) else 0
    scenario_results.append({
        "scenario": key,
        "scenario_label": sc.get("label", key),
        "objective_method": WEIGHT_MODE.lower(),
        "reliability_min": float(RELIABILITY_MIN),
        "reliability_filter_applied": bool(reliability_filter_applied),
        "n_candidates_before": int(n_cand_before),
        "n_candidates_after": int(n_cand_after),
        "CR": float(CR),
        "CI": float(CI),
        "lambda_max": float(lam),
        "w_sub_yield": float(w_sub[0]), "w_sub_irr": float(w_sub[1]), "w_sub_fert": float(w_sub[2]),
        "w_obj_yield": float(w_obj[0]), "w_obj_irr": float(w_obj[1]), "w_obj_fert": float(w_obj[2]),
        "w_final_yield": float(w_final[0]), "w_final_irr": float(w_final[1]), "w_final_fert": float(w_final[2]),
        "best_I_mm": float(cand_I[bidx]),
        "best_F_kgNha": float(cand_F[bidx]),
        "best_Y_kgHa": float(cand_y[bidx]),
        "best_topsis": float(cs[bidx]) if cs.size else float("nan"),
        "best_reliability": float(cand_rel[bidx]) if (cand_rel is not None and np.size(cand_rel)) else float("nan"),
        "_best_idx": int(bidx),
    })

def _yield_uncertainty_for_point(fert, irr, clim_samples_arr):
    """
    计算每个最优点在气候样本下的产量不确定性（±SE）。
    返回: (y_se, y_sig_clim, y_sig_gpr, y_std, K)
    """
    if (not use_climate_features) or (clim_samples_arr is None):
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    clim_samples_arr = np.asarray(clim_samples_arr, dtype=float)
    if clim_samples_arr.ndim != 2 or clim_samples_arr.shape[0] < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), int(clim_samples_arr.shape[0]) if clim_samples_arr.ndim == 2 else 0

    K = int(clim_samples_arr.shape[0])
    y_means = np.empty(K, dtype=float)
    y_vars = np.zeros(K, dtype=float)
    for k in range(K):
        tmax, tmin, tavg, rain = float(clim_samples_arr[k, 0]), float(clim_samples_arr[k, 1]), float(clim_samples_arr[k, 2]), float(clim_samples_arr[k, 3])
        Xk = pd.DataFrame([[float(fert), float(irr), tmax, tmin, tavg, rain]], columns=['FERCUM', 'IRCUM', 'Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN'])
        if use_gpr:
            Xk_gpr = gpr_scaler.transform(Xk)
            y_hat, y_std = gpr_yield.predict(Xk_gpr, return_std=True)
            y_means[k] = float(y_hat[0])
            y_vars[k] = float(y_std[0]) ** 2
        else:
            y_means[k] = float(rf_yield.predict(Xk)[0])

    y_sig_clim = float(np.std(y_means, ddof=1))
    y_sig_gpr = float(np.sqrt(np.mean(y_vars))) if use_gpr else float("nan")
    if np.isfinite(y_sig_gpr):
        y_std = float(np.sqrt(y_sig_clim ** 2 + y_sig_gpr ** 2))
    else:
        y_std = float(y_sig_clim)
    y_se = float(y_std / np.sqrt(K))
    return y_se, y_sig_clim, y_sig_gpr, y_std, K


# 为 4 个情景最优点分别计算 ±SE（不再使用 primary point 概念）
for r in scenario_results:
    y_se, y_sig_clim, y_sig_gpr, y_std, K = _yield_uncertainty_for_point(r["best_F_kgNha"], r["best_I_mm"], clim_samples)
    r["y_se"] = float(y_se)
    r["y_sig_clim"] = float(y_sig_clim)
    r["y_sig_gpr"] = float(y_sig_gpr)
    r["y_std"] = float(y_std)
    r["K_climate"] = int(K)

# 输出情景汇总 CSV（含每个点的 y_se）
out_csv = os.path.join(OUT_DIR, f"BestPoints_AHP_GameTheory_TOPSIS_{WEIGHT_MODE.lower()}_rel{float(RELIABILITY_MIN):.2f}.csv")
pd.DataFrame([{k: v for k, v in r.items() if not str(k).startswith('_')} for r in scenario_results]).to_csv(
    out_csv, index=False, encoding="utf-8-sig"
)

# ---------------------------------------------------------
# 5. 绘制 3D 帕累托最优曲面图 (NCS 风格)
# ---------------------------------------------------------
print("正在生成高美观度 3D 曲面图...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.computed_zorder = False
ax.set_position([0.30, 0.12, 0.60, 0.74])

# 修改视角，使最优点更醒目
ax.view_init(elev=24, azim=32)

# 使用三角剖分绘制连续平滑曲面，并将颜色严格映射到 Yield
try:
    triang = mtri.Triangulation(pareto_I, pareto_F)
    triangles = triang.triangles
    norm = plt.Normalize(vmin=float(pareto_yield.min()), vmax=float(pareto_yield.max()))
    cmap = plt.get_cmap('viridis')
    tri_mean_yield = pareto_yield[triangles].mean(axis=1)
    facecolors = cmap(norm(tri_mean_yield))
    surf = ax.plot_trisurf(pareto_I, pareto_F, pareto_yield,
                           triangles=triangles,
                           edgecolor='white',
                           linewidth=0.1,
                           alpha=0.85,
                           antialiased=True,
                           shade=False)
    surf.set_facecolors(facecolors)
    surf.set_zsort('min')
    surf.set_zorder(1)
except Exception as e:
    print(f"三角剖分失败，使用散点回退方案: {e}")
    surf = ax.scatter(pareto_I, pareto_F, pareto_yield, c=pareto_yield, cmap='viridis', s=60, alpha=0.9, edgecolor='k', zorder=1)

# 标记 4 个情景的最优点（保持“黑三角”为默认主展示点=高产）
for r in scenario_results:
    meta = PREFERENCE_SCENARIOS.get(r["scenario"], {})
    mkr = meta.get("marker", "^")
    col = meta.get("color", "black")
    ax.scatter(
        float(r["best_I_mm"]),
        float(r["best_F_kgNha"]),
        float(r["best_Y_kgHa"]),
        c=col,
        marker=mkr,
        s=240,
        depthshade=False,
        edgecolors='white',
        linewidths=1.2,
        zorder=10,
    )

# 设置轴标签（使用 mathtext 避免“⁻”乱码）
ax.set_xlabel('Irrigation (mm)', fontweight='bold', labelpad=15)
ax.set_ylabel('Fertilizer (kg N·ha$^{-1}$)', fontweight='bold', labelpad=15)
ax.set_zlabel('')
ax.zaxis.set_rotate_label(False)
ax.zaxis.label.set_rotation(90)
fig.text(0.06, 0.52, 'Yield (kg·ha$^{-1}$)', transform=fig.transFigure,
         rotation=90, fontsize=15, fontweight='bold', va='center', ha='left', color='black')

# 设置轴的背景和网格线样式
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')

ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

# 添加色条（以 Yield 映射）
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=12, pad=0.1)
cbar.set_label('Yield (kg·ha$^{-1}$)', fontweight='bold', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# 在图中添加信息框：4 情景最优点（已应用保证率过滤后选取）
hdr = f"Optimal Points (AHP + {WEIGHT_MODE.upper()}-TOPSIS)"
rel_note = f"Reliability filter: {'ON' if reliability_filter_applied else 'OFF'} (min={float(RELIABILITY_MIN):.2f})"
lines = [hdr, rel_note]
for r in scenario_results:
    name_en = str(r.get("scenario_label", r["scenario"]))
    if np.isfinite(r.get("y_se", np.nan)):
        ytxt = f"{float(r['best_Y_kgHa']):,.0f} ± {float(r['y_se']):,.0f}"
    else:
        ytxt = f"{float(r['best_Y_kgHa']):,.0f}"
    reltxt = r.get("best_reliability", np.nan)
    reltxt = f"{float(reltxt):.2f}" if np.isfinite(reltxt) else "NA"
    lines.append(
        f"{name_en}: I {float(r['best_I_mm']):,.1f} | F {float(r['best_F_kgNha']):,.1f} | Y {ytxt} | Rel {reltxt}"
    )
info_text = "\n".join(lines)
ax.text2D(0.89, 0.92, info_text, transform=ax.transAxes,
          fontsize=12, fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#808080', alpha=0.9))

# Legend: show the markers used on the Pareto surface
legend_handles = []
for key, meta in PREFERENCE_SCENARIOS.items():
    label = str(meta.get("label", key))
    legend_handles.append(
        Line2D(
            [0], [0],
            marker=str(meta.get("marker", "o")),
            linestyle="None",
            label=label,
            markerfacecolor=str(meta.get("color", "black")),
            markeredgecolor="white",
            markeredgewidth=1.2,
            markersize=10,
        )
    )
# Move legend slightly downward to avoid covering the title.
# (For 3D axes the title area can overlap with in-axes legend.)
# Move legend left to reduce occlusion of the 3D grid.
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(-0.08, 0.90), frameon=True, fontsize=10)

plt.title('Three-dimensional Pareto Frontier', fontweight='bold', fontsize=18, y=1.02)
# Keep legend (marker->scenario mapping) created above; do not override it.

fig.subplots_adjust(left=0.02, right=0.97, bottom=0.06, top=0.90)
output_png = os.path.join(OUT_DIR, f"3D_Pareto_Surface_AHP_{WEIGHT_MODE.lower()}_rel{float(RELIABILITY_MIN):.2f}.png")
output_pdf = os.path.join(OUT_DIR, f"3D_Pareto_Surface_AHP_{WEIGHT_MODE.lower()}_rel{float(RELIABILITY_MIN):.2f}.pdf")
plt.savefig(output_png, dpi=400, bbox_inches='tight')
try:
    plt.savefig(output_pdf, bbox_inches='tight')
except PermissionError:
    output_pdf = output_pdf.replace('.pdf', '_alt.pdf')
    plt.savefig(output_pdf, bbox_inches='tight')
print(f"图像已成功保存至:\n{output_png}\n{output_pdf}")
plt.close(fig)

if use_climate_features and (clim_year_pool is not None) and len(clim_year_pool) > 0:
    # 将原本按 RAINCUM 划分年份，改为按 Acc_RAIN 划分
    rain_series = clim_year_pool['Acc_RAIN'].astype(float)
    q25 = float(rain_series.quantile(0.25))
    q75 = float(rain_series.quantile(0.75))
    mask_dry = rain_series <= q25
    mask_wet = rain_series >= q75
    mask_mid = (~mask_dry) & (~mask_wet)

    scenarios = [
        {'tag': 'P25', 'label': 'Dry years', 'mask': mask_dry},
        {'tag': 'P50', 'label': 'Normal years', 'mask': mask_mid},
        {'tag': 'P75', 'label': 'Wet years', 'mask': mask_wet},
    ]

    scenario_results = []
    yield_min = np.inf
    yield_max = -np.inf
    _scenario_tmp = []
    for sc in scenarios:
        sub = clim_year_pool.loc[sc['mask'], climate_cols].to_numpy(dtype=float)
        if sub.size == 0:
            sub = clim_year_pool[climate_cols].to_numpy(dtype=float)
        sum_y = np.zeros(n_grid, dtype=float)
        Y_sc = None
        if ENABLE_RELIABILITY_FILTER:
            Y_sc = np.empty((int(sub.shape[0]), n_grid), dtype=float)
        for k in range(sub.shape[0]):
            tmax, tmin, tavg, rain = float(sub[k, 0]), float(sub[k, 1]), float(sub[k, 2]), float(sub[k, 3])
            X_grid_sc = np.column_stack([
                F_flat,
                I_flat,
                np.full(n_grid, tmax, dtype=float),
                np.full(n_grid, tmin, dtype=float),
                np.full(n_grid, tavg, dtype=float),
                np.full(n_grid, rain, dtype=float),
            ])
            if use_gpr:
                X_grid_sc_gpr = gpr_scaler.transform(pd.DataFrame(X_grid_sc, columns=['FERCUM', 'IRCUM', 'Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN']))
                yk = gpr_yield.predict(X_grid_sc_gpr)
            else:
                yk = rf_yield.predict(pd.DataFrame(X_grid_sc, columns=['FERCUM', 'IRCUM', 'Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN']))
            sum_y += yk
            if Y_sc is not None:
                Y_sc[int(k)] = yk
        py = sum_y / float(sub.shape[0])
        rel_sc = None

        # 先基于该年型的 py 构建 Pareto（不依赖 reliability），保证“曲面/点集”代表真实折中集合
        arr_all = np.c_[py, -I_flat, -F_flat]
        valid_all = np.all(np.isfinite(arr_all), axis=1)
        if np.any(valid_all):
            pm_valid = identify_pareto(arr_all[valid_all])
            pm = np.zeros(n_grid, dtype=bool)
            idx_valid = np.where(valid_all)[0]
            pm[idx_valid[pm_valid]] = True
        else:
            pm = np.zeros(n_grid, dtype=bool)

        # 只在“选点/保证率”中使用阈值：按该年型的“最高产量(在Pareto集合内)”的百分比来定义 target_sc
        if Y_sc is not None:
            if np.any(pm) and np.any(np.isfinite(py[pm])):
                max_y_sc = float(np.nanmax(py[pm]))
            else:
                max_y_sc = float(np.nanmax(py))
            target_sc = float(YIELD_TARGET_FRAC_OF_MAXMEAN) * max_y_sc
            rel_sc = np.mean(Y_sc >= target_sc, axis=0)

        F_cand = F_flat[pm].astype(float)
        I_cand = I_flat[pm].astype(float)
        pyy = py[pm]
        crit = np.c_[pyy, I_cand, F_cand]
        valid_p = np.all(np.isfinite(crit), axis=1)
        pyy = pyy[valid_p]
        F_cand, I_cand = F_cand[valid_p], I_cand[valid_p]
        crit = np.c_[pyy, I_cand, F_cand]
        rel_cand = rel_sc[pm][valid_p] if rel_sc is not None else None

        if pyy.size > 0:
            yield_min = float(min(yield_min, float(np.min(pyy))))
            yield_max = float(max(yield_max, float(np.max(pyy))))

        # --- AHP + Game-theory weights selection (same as "all-years" logic) ---
        # Do NOT remove Pareto points. Only apply reliability>=threshold at the index-picking step.
        # Use year-type specific AHP matrices (independent by tag)
        prefs_for_tag = PREFERENCE_SCENARIOS_BY_TAG.get(sc["tag"], PREFERENCE_SCENARIOS_CLIMATE)
        picks = []
        for pref_key, pref in prefs_for_tag.items():
            w_sub, CR, CI, lam, w_obj, w_final = resolve_weights_ahp(crit, hb_mask, pref["ahp_matrix"]) if crit.size else (None, np.nan, np.nan, np.nan, None, None)
            if (w_final is None) or (not np.all(np.isfinite(w_final))) or (np.sum(w_final) <= 0):
                w_final = np.full(crit.shape[1], 1.0 / crit.shape[1], dtype=float) if crit.size else np.array([1/3, 1/3, 1/3], dtype=float)
            cs_sc = topsis_scores(crit, w_final, hb_mask) if crit.size else np.array([], dtype=float)

            bidx = 0
            fallback_used = False
            if cs_sc.size and np.any(np.isfinite(cs_sc)):
                if ENABLE_RELIABILITY_FILTER and (rel_cand is not None):
                    # Apply year-type specific reliability threshold for INDEX SELECTION only
                    rel_min_eff = float(RELIABILITY_MIN_BY_TAG.get(sc["tag"], RELIABILITY_MIN))
                    keep = np.isfinite(rel_cand) & (rel_cand >= rel_min_eff)
                    if np.any(keep):
                        idxs = np.where(keep)[0]
                        bidx = int(idxs[int(np.nanargmax(cs_sc[idxs]))])
                    else:
                        bidx = int(np.nanargmax(cs_sc))
                        fallback_used = True
                else:
                    bidx = int(np.nanargmax(cs_sc))

            if pyy.size:
                best_F_sc = float(F_cand[bidx])
                best_I_sc = float(I_cand[bidx])
                best_y_sc = float(pyy[bidx])
                best_rel_sc = float(rel_cand[bidx]) if (rel_cand is not None and rel_cand.size) else float('nan')
                best_cs_sc = float(cs_sc[bidx]) if cs_sc.size else float("nan")
            else:
                best_F_sc = float('nan')
                best_I_sc = float('nan')
                best_y_sc = float('nan')
                best_rel_sc = float('nan')
                best_cs_sc = float('nan')

            # uncertainty (±SE) for this preference point
            K_sc = int(sub.shape[0])
            if (not np.isfinite(best_F_sc)) or (not np.isfinite(best_I_sc)) or (K_sc <= 1):
                y_se = float('nan')
                y_sig_clim = float('nan')
                y_sig_gpr = float('nan')
            else:
                y_means = np.empty(K_sc, dtype=float)
                y_vars = np.zeros(K_sc, dtype=float)
                for k in range(K_sc):
                    tmax, tmin, tavg, rain = float(sub[k, 0]), float(sub[k, 1]), float(sub[k, 2]), float(sub[k, 3])
                    Xk = pd.DataFrame([[best_F_sc, best_I_sc, tmax, tmin, tavg, rain]], columns=['FERCUM', 'IRCUM', 'Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN'])
                    if use_gpr:
                        Xk_gpr = gpr_scaler.transform(Xk)
                        y_hat, y_std = gpr_yield.predict(Xk_gpr, return_std=True)
                        y_means[k] = float(y_hat[0]); y_vars[k] = float(y_std[0]) ** 2
                    else:
                        y_means[k] = float(rf_yield.predict(Xk)[0])
                y_sig_clim = float(np.std(y_means, ddof=1)) if K_sc >= 2 else float('nan')
                y_sig_gpr = float(np.sqrt(np.mean(y_vars))) if use_gpr else float('nan')
                y_sd_total = float(np.sqrt(y_sig_clim ** 2 + y_sig_gpr ** 2)) if np.isfinite(y_sig_gpr) else y_sig_clim
                y_se = float(y_sd_total / np.sqrt(K_sc))

            picks.append({
                "pref_key": pref_key,
                "pref_label": str(pref.get("label", pref_key)),
                "marker": pref.get("marker", "o"),
                "color": pref.get("color", "black"),
                "CR": float(CR),
                "w_final_yield": float(w_final[0]),
                "w_final_irr": float(w_final[1]),
                "w_final_fert": float(w_final[2]),
                "best_F": best_F_sc,
                "best_I": best_I_sc,
                "best_yield": best_y_sc,
                "best_cs": best_cs_sc,
                "best_rel": best_rel_sc,
                "reliability_min_effective": float(RELIABILITY_MIN_BY_TAG.get(sc["tag"], RELIABILITY_MIN)) if (ENABLE_RELIABILITY_FILTER and (rel_cand is not None)) else float("nan"),
                "fallback_used": bool(fallback_used),
                "y_se": float(y_se),
                "y_sig_clim": float(y_sig_clim),
                "y_sig_gpr": float(y_sig_gpr),
            })

        # Final recommendation for this year type: pick ONLY the preference specified by policy
        policy_pref_key = str(YEAR_TYPE_POLICY.get(sc["tag"], "high_yield"))
        pick_map = {p.get("pref_key"): p for p in picks}
        final_pick = pick_map.get(policy_pref_key)
        if final_pick is None:
            # fallback to the first available pick
            final_pick = picks[0] if picks else None
            policy_pref_key = str(final_pick.get("pref_key")) if final_pick else policy_pref_key

        scenario_results.append({
            'scenario': {**sc, 'n_years': int(sub.shape[0])},
            'pareto_yield': pyy,
            'pareto_fert': F_cand,
            'pareto_irr': I_cand,
            'picks': picks,
            'policy_pref_key': policy_pref_key,
            'final_pick': final_pick,
            '_climate_sub': sub,
            '_full_py': py,
        })

    # Note: previous code applied an extra constraint to force P50 irrigation between P25 and P75.
    # To keep the logic consistent with the "all-years" AHP+TOPSIS selection, that post-adjustment is disabled.

    # Use the same output directory as the rest of the script
    out_dir = OUT_DIR
    # Detailed output: year-type (P25/P50/P75) × preference (scenarios used for climate analysis)
    rows_detail = []
    for r in scenario_results:
        sc = r["scenario"]
        for p in r.get("picks", []):
            rows_detail.append({
                "tag": sc["tag"],
                "label": sc["label"],
                "n_years": sc["n_years"],
                "preference": p["pref_label"],
                "best_I_mm": float(p.get("best_I", np.nan)),
                "best_F_kgNha": float(p.get("best_F", np.nan)),
                "best_Y_kgHa": float(p.get("best_yield", np.nan)),
                "best_cs": float(p.get("best_cs", np.nan)),
                "best_rel": float(p.get("best_rel", np.nan)),
                "y_se": float(p.get("y_se", np.nan)),
                "fallback_used": bool(p.get("fallback_used", False)),
            })
    pd.DataFrame(rows_detail).to_csv(os.path.join(out_dir, 'RF_AHP_GameTOPSIS_BestPoints_ClimateB_detail.csv'), index=False, encoding='utf-8-sig')

    # Policy output: ONLY 1 final recommended point per year type (3 rows)
    rows_policy = []
    for r in scenario_results:
        sc = r["scenario"]
        fp = r.get("final_pick", None) or {}
        rel_min_eff = float(RELIABILITY_MIN_BY_TAG.get(sc["tag"], RELIABILITY_MIN))
        rows_policy.append({
            "tag": sc["tag"],
            "label": sc["label"],
            "n_years": sc["n_years"],
            "policy_pref_key": r.get("policy_pref_key", ""),
            "policy_pref_label": str(fp.get("pref_label", "")),
            "best_I_mm": float(fp.get("best_I", np.nan)),
            "best_F_kgNha": float(fp.get("best_F", np.nan)),
            "best_Y_kgHa": float(fp.get("best_yield", np.nan)),
            "best_cs": float(fp.get("best_cs", np.nan)),
            "best_rel": float(fp.get("best_rel", np.nan)),
            "y_se": float(fp.get("y_se", np.nan)),
            "reliability_mode": str(RELIABILITY_MODE),
            "reliability_min_effective": rel_min_eff if (ENABLE_RELIABILITY_FILTER and (not np.isnan(float(fp.get("best_rel", np.nan))))) else float("nan"),
            "fallback_used": bool(fp.get("fallback_used", False)),
        })
    pd.DataFrame(rows_policy).to_csv(os.path.join(out_dir, 'RF_AHP_GameTOPSIS_BestPoints_ClimateB.csv'), index=False, encoding='utf-8-sig')

    ypool = []
    for r in scenario_results:
        yy = r.get('pareto_yield', np.array([], dtype=float))
        if isinstance(yy, np.ndarray) and yy.size:
            ypool.append(yy[np.isfinite(yy)])
    if ypool:
        ycat = np.concatenate(ypool)
        if ycat.size:
            yield_min = float(np.min(ycat))
            yield_max = float(np.max(ycat))
    if (not np.isfinite(yield_min)) or (not np.isfinite(yield_max)) or (yield_min == yield_max):
        yield_min = 0.0
        yield_max = 1.0

    cmap = plt.get_cmap('viridis')
    normB = plt.Normalize(vmin=yield_min, vmax=yield_max)
    figB = plt.figure(figsize=(16, 6.8))
    lefts = [0.10, 0.385, 0.67]
    bottom = 0.18
    width = 0.26
    height = 0.70
    panel_infos = []
    for i, res in enumerate(scenario_results):
        axB = figB.add_axes([lefts[i], bottom, width, height], projection='3d')
        axB.computed_zorder = False
        axB.view_init(elev=24, azim=32)

        try:
            triangB = mtri.Triangulation(res['pareto_irr'], res['pareto_fert'])
            triB = triangB.triangles
            tri_mean = res['pareto_yield'][triB].mean(axis=1)
            faceB = cmap(normB(tri_mean))
            surfB = axB.plot_trisurf(res['pareto_irr'], res['pareto_fert'], res['pareto_yield'],
                                     triangles=triB,
                                     edgecolor='white',
                                     linewidth=0.1,
                                     alpha=0.85,
                                     antialiased=True,
                                     shade=False)
            surfB.set_facecolors(faceB)
            surfB.set_zsort('min')
            surfB.set_zorder(1)
        except Exception:
            axB.scatter(res['pareto_irr'], res['pareto_fert'], res['pareto_yield'],
                        c=res['pareto_yield'], cmap='viridis', norm=normB, s=20, alpha=0.9, edgecolor='none', zorder=1)

        # Mark ONLY the final recommended point for this year type (policy-based)
        # Special styling for Dry years: use a distinct marker and an English label
        # "High yield + High reliability" (requested) to distinguish from Normal years.
        fp = res.get("final_pick", None) or {}
        fp_marker = str(fp.get("marker", "o"))
        fp_color = str(fp.get("color", "black"))
        fp_label = str(fp.get("pref_label", ""))
        if str(res["scenario"].get("tag", "")) == "P25" and str(res.get("policy_pref_key", "")) == "high_yield":
            fp_marker = "*"
            fp_color = "#d62728"  # distinct from Normal-year marker/color
            fp_label = "High yield + High reliability"

        if fp:
            axB.scatter(float(fp["best_I"]), float(fp["best_F"]), float(fp["best_yield"]),
                        c=fp_color,
                        marker=fp_marker,
                        s=250, depthshade=False,
                        edgecolors='white', linewidths=1.2, zorder=10)

        if i == 1:
            axB.set_xlabel('Irrigation (mm)', fontweight='bold', labelpad=10)
        if i == 0:
            axB.set_ylabel('Fertilizer (kg N·ha$^{-1}$)', fontweight='bold', labelpad=10)
        axB.set_zlabel('')

        axB.xaxis.pane.fill = False
        axB.yaxis.pane.fill = False
        axB.zaxis.pane.fill = False
        axB.xaxis.pane.set_edgecolor('black')
        axB.yaxis.pane.set_edgecolor('black')
        axB.zaxis.pane.set_edgecolor('black')
        axB.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        sc = res['scenario']
        titleB = f"{sc['tag']} ({sc['label']}, n={sc['n_years']})"
        axB.set_title(titleB, fontweight='bold', fontsize=14, pad=8)

        # Info box: show the policy preference + final point only
        linesB = ["Final recommendation (AHP + Game-theory TOPSIS)"]
        if fp:
            se = fp.get("y_se", np.nan)
            if np.isfinite(se):
                ytxt = f"{float(fp['best_yield']):,.0f} ± {float(se):,.0f}"
            else:
                ytxt = f"{float(fp['best_yield']):,.0f}"
            reltxt = fp.get("best_rel", np.nan)
            reltxt = f"{float(reltxt):.2f}" if np.isfinite(reltxt) else "NA"
            linesB.append(f"Policy: {res.get('policy_pref_key','')} → {fp_label}")
            linesB.append(f"I {float(fp['best_I']):,.1f}, F {float(fp['best_F']):,.1f}, Y {ytxt}, Rel {reltxt}")
        panel_infos.append("\n".join(linesB))

    figB.text(0.03, 0.52, 'Yield (kg·ha$^{-1}$)', transform=figB.transFigure,
              rotation=90, fontsize=14, fontweight='bold', va='center', ha='left', color='black')
    for i, infoB in enumerate(panel_infos):
        figB.text(lefts[i] + 0.01, 0.06, infoB, transform=figB.transFigure,
                  fontsize=9.2, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#808080', alpha=0.92),
                  va='bottom', ha='left')

    # Figure-level legend: show the actual marker/label used by each year-type final pick.
    # This avoids confusion when the same preference key (e.g., "high_yield") is used for both P25 and P50
    # but we intentionally style P25 differently ("High yield + High reliability").
    legend_items = []
    for res in scenario_results:
        tag = str(res["scenario"].get("tag", ""))
        policy_k = str(res.get("policy_pref_key", ""))
        meta = PREFERENCE_SCENARIOS_CLIMATE.get(policy_k, {})
        marker = str(meta.get("marker", "o"))
        color = str(meta.get("color", "black"))
        label = str(meta.get("label", policy_k))
        if tag == "P25" and policy_k == "high_yield":
            marker = "*"
            color = "#d62728"
            label = "High yield + High reliability"
        legend_items.append((label, marker, color))

    legend_handles = []
    used_labels = []
    for (label, marker, color) in legend_items:
        if label in used_labels:
            continue
        used_labels.append(label)
        legend_handles.append(
            Line2D([0], [0],
                   marker=marker,
                   linestyle="None",
                   label=label,
                   markerfacecolor=color,
                   markeredgecolor="white",
                   markeredgewidth=1.1,
                   markersize=9)
        )

    # Move legend below the suptitle to avoid covering it.
    figB.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=max(1, len(legend_handles)),
        frameon=True,
        fontsize=10,
    )

    from matplotlib.cm import ScalarMappable
    smB = ScalarMappable(norm=normB, cmap=cmap)
    smB.set_array([])
    cax = figB.add_axes([0.92, 0.24, 0.015, 0.56])
    cbarB = figB.colorbar(smB, cax=cax)
    cbarB.set_label('Yield (kg·ha$^{-1}$)', fontweight='bold', fontsize=14)
    cbarB.ax.tick_params(labelsize=12)

    figB.suptitle('Three-dimensional Pareto Frontier (Climate Scenarios)', fontweight='bold', fontsize=18, y=0.995)

    outB_png = os.path.join(OUT_DIR, f"3D_Pareto_Surface_ClimateB_P25P50P75_{WEIGHT_MODE.lower()}_rel{float(RELIABILITY_MIN):.2f}.png")
    outB_pdf = os.path.join(OUT_DIR, f"3D_Pareto_Surface_ClimateB_P25P50P75_{WEIGHT_MODE.lower()}_rel{float(RELIABILITY_MIN):.2f}.pdf")
    figB.savefig(outB_png, dpi=400, bbox_inches='tight')
    try:
        figB.savefig(outB_pdf, bbox_inches='tight')
    except PermissionError:
        outB_pdf = outB_pdf.replace('.pdf', '_alt.pdf')
        figB.savefig(outB_pdf, bbox_inches='tight')
    print(f"图像已成功保存至:\n{outB_png}\n{outB_pdf}")
    plt.close(figB)
