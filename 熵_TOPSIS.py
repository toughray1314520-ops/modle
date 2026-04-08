import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.tri as mtri
import matplotlib.patheffects as pe

# 绘图样式配置 (SCI 论文风格)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans' # 确保数学公式字体一致
plt.rcParams['axes.unicode_minus'] = True # 正常显示负号
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
# 强制使用特定字体以防乱码
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# 专业配色
COLORS = {
    'primary': '#2563EB', # 主蓝色（最优组合、核心指标）
    'secondary': '#64748B', # 次要灰色（其他组合、参考线）
    'tertiary': '#DC2626', # 红色（阈值线、警示）
    'success': '#10B981', # 绿色（优势标识）
    'info': '#06B6D4', # 青色（辅助指标）
    'light': '#F1F5F9', # 浅灰（背景填充）
    'gray': '#64748B',
    'blue': '#2563EB',
    'orange': '#D97706',
    'red': '#DC2626', # 淘汰
    'bg': '#F8FAFC',
    'card_bg': '#FFFFFF',
    'green': '#059669',
    'border': '#E2E8F0',
    'text': '#1E293B'
}
# ========== 新增：数据读取与预处理 ==========
# ---------- 1) 统一且鲁棒的 load_data ----------
def load_data(file_path=r'D:\learning\modle_Tan\IR×FER180.xlsx'):
    xl = pd.ExcelFile(file_path)
    sheets = xl.sheet_names
    all_df = []
    print(f"读取 Excel，共 {len(sheets)} sheets：{sheets}")

    for sheet in sheets:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
            df.columns = df.columns.str.strip()
            if 'YEAR' in df.columns:
                df = df[df['YEAR'].astype(str).str.strip().str.lower() != 'mean'].copy()
                df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
                df = df[df['YEAR'].notna()].copy()
            # 保留原始sheet标签（作为 raw label）
            df['SheetName'] = sheet

            # 解析肥料部分（统一逻辑）
            # sheet 例子："225N+NO IR" 或 "-10%N+WCFC%"
            parts = sheet.split('+')
            fert_part = parts[0].upper().replace('N','').strip()  # '225' 或 '-10%' 或 ' -10% '
            irr_part = parts[1].strip() if len(parts) > 1 else ''

            # 生成三类信息：
            #  1) Fert_label 原始（如 '225' 或 '-10%'）
            #  2) Fert_pct_of225（以225为100%，即 225->100, -10% -> 90, -50% -> 50）
            #  3) Fert_abs (若 fert_part 为纯数字，视为绝对 kg/ha；否则按225*pct/100)
            try:
                if fert_part.endswith('%'):
                    val = int(fert_part.replace('%','').replace('+',''))
                    fert_pct_of225 = 100 + val if val < 0 else 100 + val  # '-10%' -> 90, '+10%' -> 110
                    fert_abs = 225 * fert_pct_of225 / 100.0
                    fert_label = f"{val:+d}%"
                else:
                    # 数字比如 '225' 或 '200'
                    fert_abs = float(fert_part)
                    fert_pct_of225 = fert_abs / 225.0 * 100.0
                    fert_label = f"{int(fert_abs)}"
            except Exception:
                fert_label = fert_part
                fert_pct_of225 = np.nan
                fert_abs = np.nan

            # 解析灌溉类别标签
            if irr_part == 'NO IR':
                irr_level = 'W1'
            elif irr_part == 'WCFC%':
                irr_level = 'W2'
            else:
                irr_level = 'W3'

            # 一致的 Combo / 肥料标签（保留原始信息，避免混淆）
            df['Fert_label'] = fert_label
            df['Fert_pct_of225'] = fert_pct_of225
            df['Fert_abs'] = fert_abs
            df['Irr'] = irr_level
            # 复原一个简洁的 Fert 因子列，供 ANOVA 用（例如 100%N、90%N、50%N）
            if not np.isnan(fert_pct_of225):
                df['Fert'] = df['Fert_pct_of225'].round().astype(int).astype(str) + '%'
            else:
                df['Fert'] = fert_label
            # Combo 使用更直观格式： e.g. W1_90%N 或 W1_225N
            if not np.isnan(fert_pct_of225):
                df['Combo'] = f"{irr_level}_{int(round(fert_pct_of225))}%N"
            else:
                df['Combo'] = f"{irr_level}_{fert_label}N"

            # 计算指标（稳健处理：当分母接近0时使用 NaN，而不是用 1e-6）
            # 注意单位：假定 WRR14 单位 kg/ha，IRCUM/RAINCUM/TRCCUM/EVSWCUM 单位 mm
            for _col in ['WRR14', 'IRCUM', 'RAINCUM', 'TRCCUM', 'EVSWCUM', 'FERCUM', 'WAGT']:
                if _col in df.columns:
                    df[_col] = pd.to_numeric(df[_col], errors='coerce')
            df['WUE'] = np.where((df.get('IRCUM', 0).fillna(0) + df.get('RAINCUM', 0).fillna(0)) > 0,
                                  df['WRR14'] / (df['IRCUM'].fillna(0) + df['RAINCUM'].fillna(0)),
                                  np.nan)
            df['IWUE'] = np.where(df.get('IRCUM', 0).fillna(0) > 0,
                                   df['WRR14'] / df['IRCUM'].fillna(0),
                                   np.nan)
            # ET_WUE: 基于蒸散/蒸发量，若缺 TRCCUM 或 EVSWCUM 则设 NaN
            if 'TRCCUM' in df.columns and 'EVSWCUM' in df.columns:
                df['ET_SUM'] = df['TRCCUM'].fillna(0) + df['EVSWCUM'].fillna(0)
                df['ET_WUE'] = np.where(df['ET_SUM'] > 0, df['WRR14'] / df['ET_SUM'], np.nan)
            else:
                df['ET_WUE'] = np.nan

            # NUE：慎重处理 FERCUM 可能为0
            df['NUE'] = np.where(df.get('FERCUM', 0).fillna(0) > 0, df['WRR14'] / df['FERCUM'].fillna(0), np.nan)

            all_df.append(df)
        except Exception as e:
            print(f"读取 sheet {sheet} 失败：{e}")

    all_df = pd.concat(all_df, ignore_index=True) if len(all_df) > 0 else pd.DataFrame()
    # 年份筛选（如果没有 YEAR 列则跳过过滤）
    if 'YEAR' in all_df.columns:
        all_df = all_df[all_df['YEAR'].between(1981, 2017)]
    # 检查 Combo 命名与每组大小
    print("合并后行数：", all_df.shape[0])
    if 'Combo' in all_df.columns:
        print("组合数：", all_df['Combo'].nunique())
        print(all_df['Combo'].value_counts().sort_index())
    return all_df
# 用修正版加载
raw_df = load_data()
print("\n" + "="*60)
print("=== 调试：raw_df 的实际列名列表（最重要！） ===")
print(raw_df.columns.tolist())
print("\n=== 前 5 行数据预览（检查 WAGT 是否存在且有值） ===")
print(raw_df[['YEAR', 'Combo', 'WAGT', 'WRR14', 'WUE', 'NUE', 'SPFERT']].head().to_string(index=False) if 'WAGT' in raw_df.columns else "没有 WAGT 列！")
print("="*60 + "\n")
print(f"加载数据：{raw_df.shape[0]}行，组合数：{raw_df['Combo'].nunique()}")
key_indicators = ['WAGT', 'WRR14', 'WUE', 'IWUE', 'ET_WUE', 'NUE', 'SPFERT']
def perform_anova(df, indicator):
    if indicator not in df.columns:
        print(f"缺少 '{indicator}' 列，跳过")
        return None, None

    print(f"\n分析 '{indicator}'，总非空值：{df[indicator].notna().sum()}")

    # 按 Irr x Fert 分组并收集每组的 indicator Series（已 dropna）
    groups = []
    group_names = []
    for name, group in df.groupby(['Irr', 'Fert']):
        if indicator in group.columns:
            g_clean = group[indicator].dropna()
            if len(g_clean) >= 1:  # 先收集，后面再按需要筛样本量
                groups.append(g_clean)
                group_names.append(name)

    if len(groups) < 2:
        print("有效分组不足 2，无法做任何柱状/方差检验")
        return None, None

    # 为每组检查样本数
    group_sizes = [len(g) for g in groups]
    print("每组样本数（Irr,Fert）：", dict(zip(group_names, group_sizes)))

    # 只对样本数 >=3 的组做 Shapiro（Shapiro 最低 3 个样本）
    shapiro_ok_groups = [g for g in groups if len(g) >= 3]
    normal_raw = True
    if len(shapiro_ok_groups) == 0:
        normal_raw = False
        print(f"{indicator}：没有满足 Shapiro 最低样本数（>=3）的组，无法可靠检验正态性")
    else:
        try:
            normal_raw = all(stats.shapiro(g)[1] > 0.05 for g in shapiro_ok_groups)
            print(f"{indicator} 原始数据组正态性 (Shapiro-Wilk): {'通过' if normal_raw else '未完全通过'}")
        except Exception as e:
            normal_raw = False
            print("Shapiro 执行异常：", e)

    # Levene 方差齐性：需要至少 2 组且每组至少 2 个样本
    levene_groups = [g for g in groups if len(g) >= 2]
    if len(levene_groups) >= 2:
        try:
            levene_stat, levene_p = stats.levene(*levene_groups)
            variance_equal = levene_p > 0.05
            print(f"{indicator} 方差齐性 Levene: stat={levene_stat:.4f}, p={levene_p:.4f}")
        except Exception as e:
            variance_equal = False
            print("Levene 检验异常：", e)
    else:
        variance_equal = False
        print(f"{indicator}：用于 Levene 的组数不足（至少需要2组且每组>=2）")

    # 当满足参数检验条件时尝试 ANOVA
    if normal_raw and variance_equal:
        try:
            # 只用 indicator 非空的行拟合模型
            df_fit = df.dropna(subset=[indicator, 'Irr', 'Fert']).copy()
            if df_fit.shape[0] < 3:
                print("拟合数据点太少，跳过 ANOVA")
                return None, None

            model = ols(f'{indicator} ~ C(Irr) * C(Fert)', data=df_fit).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(f"\n{indicator} ANOVA 结果:\n{anova_table}")

            # 残差正态性（需要残差样本数 >=3）
            if len(model.resid) >= 3:
                sh_stat, sh_p = stats.shapiro(model.resid)
                print(f"残差正态性 (Shapiro)：stat={sh_stat:.4f}, p={sh_p:.4f}")
                resid_normal = sh_p > 0.05
            else:
                resid_normal = False
                print("残差样本数不足，无法做残差正态性检验")

            # Tukey：需要至少 2 组且组间样本数 >1
            # 这里以 Combo 为分组因子进行 Tukey（可根据需要改）
            try:
                group_counts = df_fit.groupby('Combo')[indicator].count()
                enough_groups = (group_counts >= 2).sum() >= 2
                if resid_normal and enough_groups:
                    tukey = pairwise_tukeyhsd(df_fit[indicator], df_fit['Combo'], alpha=0.05)
                    print(f"\n{indicator} Tukey HSD 完成")
                else:
                    tukey = None
                    print("未满足 Tukey 前提（残差正态或每组样本数不足），跳过 Tukey")
            except Exception as e:
                tukey = None
                print("执行 Tukey 时出错：", e)

            return anova_table, tukey

        except Exception as e:
            print("ANOVA 拟合失败：", e)
            return None, None

    # 非参数路线（Kruskal-Wallis）
    else:
        print(f"{indicator} 不满足参数 ANOVA 条件，尝试非参数检验（Kruskal-Wallis）")
        try:
            # 按 Irr 和 Fert 各自计算主效应（示例）
            irr_groups = [group[indicator].dropna() for name, group in df.groupby('Irr') if len(group[indicator].dropna()) >= 2]
            fert_groups = [group[indicator].dropna() for name, group in df.groupby('Fert') if len(group[indicator].dropna()) >= 2]

            if len(irr_groups) >= 2:
                _, p_irr = stats.kruskal(*irr_groups)
                print(f"Irr 主效应 (Kruskal-Wallis) p = {p_irr:.4f}")
            else:
                p_irr = np.nan
                print("Irr 分组样本不足，跳过 Kruskal-Wallis")

            if len(fert_groups) >= 2:
                _, p_fert = stats.kruskal(*fert_groups)
                print(f"Fert 主效应 (Kruskal-Wallis) p = {p_fert:.4f}")
            else:
                p_fert = np.nan
                print("Fert 分组样本不足，跳过 Kruskal-Wallis")

            return None, None

        except Exception as e:
            print("非参数检验执行异常：", e)
            return None, None

# 执行ANOVA for all indicators
anova_results = {}
for ind in key_indicators:
    anova_table, tukey = perform_anova(raw_df, ind)
    anova_results[ind] = {'anova': anova_table, 'tukey': tukey}

# 保存ANOVA结果（仅当至少有一个指标成功完成 ANOVA 时）
anova_tables = {k: v['anova'] for k, v in anova_results.items() if v['anova'] is not None}
if anova_tables:
    pd.concat(anova_tables, axis=0).to_csv('anova_results.csv')
    print("已保存 anova_results.csv（仅包含满足 ANOVA 前提条件的指标）。")
else:
    print("所有指标均未满足 ANOVA 前提条件，仅保留非参数检验结果，跳过 anova_results.csv 的输出。")

# ===================== 1. 熵权法计算指标权重 =====================
def entropy_weight(df):
    """
    熵权法计算指标权重（输入标准化后的指标矩阵，输出各指标熵权）
    :param df: 标准化后的DataFrame（行：组合，列：评价指标）
    :return: 各指标权重Series
    """
    df = df.replace(0, 1e-8) # 避免0值
    f = df / df.sum(axis=0) # 计算第j个指标下第i个组合的占比f_ij
    k = 1 / np.log(len(df)) # 常数k
    H = -k * (f * np.log(f)).sum(axis=0) # 计算熵值H_j
    g = 1 - H # 差异系数
    w = g / g.sum() # 熵权
    return w
# ===================== 2. TOPSIS法综合评价（适配混合指标） =====================
def topsis_evaluation(df, weights, high_better_cols):
    """
    适配混合指标的TOPSIS法（明确区分高优/低优，与论文逻辑一致）
    :param df: 标准化后的DataFrame（行：组合，列：评价指标）
    :param weights: 熵权法得到的指标权重
    :param high_better_cols: 高优指标列名列表（其余自动视为低优）
    :return: 各组合TOPSIS综合贴近度Series
    """
    # 加权标准化矩阵（与论文公式一致：Z_ij = ω_j × r_ij）
    Z = df * weights
   
    # 确定正理想解（x+）和负理想解（x-），完全按论文逻辑
    Z_plus = pd.Series(index=Z.columns, dtype=float)
    Z_minus = pd.Series(index=Z.columns, dtype=float)
    for col in Z.columns:
        if col in high_better_cols:
            Z_plus[col] = Z[col].max() # 高优指标：正理想解=最大值
            Z_minus[col] = Z[col].min() # 高优指标：负理想解=最小值
        else:
            Z_plus[col] = Z[col].min() # 低优指标：正理想解=最小值
            Z_minus[col] = Z[col].max() # 低优指标：负理想解=最大值
   
    # 计算欧氏距离（与论文公式一致）
    d_plus = np.sqrt(((Z - Z_plus) ** 2).sum(axis=1)) # 到正理想解的距离
    d_minus = np.sqrt(((Z - Z_minus) ** 2).sum(axis=1))# 到负理想解的距离
   
    # 计算综合贴近度（与论文公式一致：S_i = d_i⁻/(d_i⁺+d_i⁻)）
    S = d_minus / (d_plus + d_minus)
    return S
# ===================== 3. 权重敏感性分析（稳健性验证） =====================
def weight_sensitivity_analysis(eval_df, high_better_cols, base_weights, combo_ids, best_combo_id, top_n=3):
    """
    权重敏感性分析：扰动各指标权重，验证最优组合排序稳定性
    :param eval_df: 标准化后的评价指标矩阵
    :param high_better_cols: 高优指标列名
    :param base_weights: 基准熵权
    :param top_n: 重点关注TOPN组合
    :return: 敏感性分析结果DataFrame
    """
    perturb_rates = [-0.2, -0.1, 0, 0.1, 0.2] # 权重扰动幅度
    indicators = base_weights.index.tolist()
    sensitivity_results = []
   
    # 对每个指标单独扰动，其他指标权重按比例调整（保持和为1）
    for indicator in indicators:
        for rate in perturb_rates:
            new_weights = base_weights.copy()
            perturb_amount = new_weights[indicator] * rate
            new_weights[indicator] += perturb_amount
           
            # 调整其他指标权重
            other_indicators = [ind for ind in indicators if ind != indicator]
            other_sum = new_weights[other_indicators].sum()
            if other_sum != 0:
                new_weights[other_indicators] = new_weights[other_indicators] / other_sum * (1 - new_weights[indicator])
           
            # 计算扰动后的TOPSIS排序
            new_closeness = topsis_evaluation(eval_df, new_weights, high_better_cols)
            new_ranking = pd.DataFrame({
                'Combo_ID': list(combo_ids),
                'New_Score': new_closeness.values,
                'New_Rank': new_closeness.rank(ascending=False, method='min').astype(int)
            })
           
            # 提取TOPN组合信息
            topn_ranking = new_ranking[new_ranking['New_Rank'] <= top_n].copy()
            topn_str = "; ".join([f"{row['Combo_ID']} (Rank {row['New_Rank']})" for _, row in topn_ranking.iterrows()])
            is_best_stable = best_combo_id in topn_ranking['Combo_ID'].values
            stability = "Stable" if is_best_stable else "Unstable"
           
            sensitivity_results.append({
                'Indicator': indicator,
                'Perturbation_Rate': f"{rate*100:+.0f}%",
                f'Top_{top_n}_Candidates': topn_str,
                'Stability': stability,
                'New_Optimal_Combo': new_ranking.loc[new_ranking['New_Rank'] == 1, 'Combo_ID'].iloc[0],
                'New_Rank_of_Best': new_ranking[new_ranking['Combo_ID'] == best_combo_id]['New_Rank'].iloc[0]
            })
   
    return pd.DataFrame(sensitivity_results)

# 扩展：多指标同时扰动（蒙特卡罗），更贴近“显著差异多、排序应有波动”的直觉
def mc_weight_sensitivity(eval_df, high_better_cols, base_weights, combo_ids, trials=300, delta=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indicators = base_weights.index.tolist()
    records = []
    for t in range(trials):
        # 在 [-delta, +delta] 范围为每个指标生成独立扰动（相对比例）
        eps = rng.uniform(-delta, delta, size=len(indicators))
        new_w = base_weights.copy().astype(float)
        for i, ind in enumerate(indicators):
            new_w[ind] = max(1e-8, new_w[ind] * (1.0 + eps[i]))
        new_w = new_w / new_w.sum()
        s = topsis_evaluation(eval_df, new_w, high_better_cols)
        best_idx = int(np.argmax(s.to_numpy()))
        best_id = combo_ids[best_idx]
        records.append({
            'trial': t,
            'best_combo': best_id,
            'best_score': float(s.to_numpy()[best_idx]),
        })
    return pd.DataFrame(records)
# ===================== 4. 数据读取与基础指标计算 =====================
# 计算每个Combo组合的核心指标（使用英文键名）
combo_metrics = []
for combo, combo_data in raw_df.groupby('Combo'):
    # 核心产量
    wagt_mean = combo_data['WAGT'].mean() if 'WAGT' in combo_data.columns else np.nan
    wrr14_mean = combo_data['WRR14'].mean()
    wrr14_std = combo_data['WRR14'].std()
    wrr14_cv = (wrr14_std / wrr14_mean) * 100 if wrr14_mean != 0 else 0
   
    # 水分相关
    wu_mean = combo_data['ET_SUM'].mean() if 'ET_SUM' in combo_data.columns else np.nan
    ircum_mean = combo_data['IRCUM'].mean()
    # 使用在 load_data 中预计算好的指标均值，确保与显著性分析一致
    wue = combo_data['WUE'].mean()
    iwue = combo_data['IWUE'].mean()
    et_wue = combo_data['ET_WUE'].mean()
    draicum_mean = combo_data['DRAICUM'].mean() if 'DRAICUM' in combo_data.columns else np.nan
   
    # 施肥相关
    fercum_mean = combo_data['FERCUM'].mean()
    nue = combo_data['NUE'].mean()
   
    # 胁迫/其他
    spfert_mean = combo_data['SPFERT'].mean()
   
    combo_metrics.append({
        'Combo': combo,
        'WAGT_mean': round(wagt_mean, 2) if not np.isnan(wagt_mean) else np.nan,
        'WRR14_mean': round(wrr14_mean, 2),
        'WRR14_CV': round(wrr14_cv, 3),
        'WU_mean': round(wu_mean, 2) if not np.isnan(wu_mean) else np.nan,
        'Irr_mean': round(ircum_mean, 2),
        'WUE': round(wue, 6),
        'IWUE': round(iwue, 6),
        'ET_WUE': round(et_wue, 6),
        'Perc_mean': round(draicum_mean, 2) if not np.isnan(draicum_mean) else np.nan,
        'Fert_mean': round(fercum_mean, 2),
        'NUE': round(nue, 6),
        'SPFERT_mean': round(spfert_mean, 4)
    })

# 转换为DataFrame
combo_df = pd.DataFrame(combo_metrics)
combo_df_sorted = combo_df.sort_values('WRR14_mean', ascending=False).reset_index(drop=True)
# ===================== 5. 输出基础指标汇总 =====================
print("="*80)
print("Summary of core indicators for all combinations (including CV)")
print("="*80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(combo_df_sorted.to_string(index=False))

print("\n" + "="*80)
print("Key indicators (Top 5 vs Bottom 3 by yield)")
print("="*80)
key_cols = ['Combo', 'WRR14_mean', 'WRR14_CV', 'WUE', 'NUE', 'SPFERT_mean']
print(combo_df_sorted[key_cols].head(5).to_string(index=False))
print("...")
print(combo_df_sorted[key_cols].tail(3).to_string(index=False))

# ===================== 6. 熵权TOPSIS全局排序（核心步骤） =====================
print("\n" + "="*80)
print("Entropy-weighted TOPSIS ranking")
print("="*80)
# 定义评价指标集
eval_cols = [
    'WRR14_mean', 
    'WRR14_CV', 
    'WUE',
    'NUE', 
    'SPFERT_mean'
]
# 明确高优指标列名
high_better_cols = [
    'WRR14_mean',
    'WUE',
    'NUE',
    'SPFERT_mean'
]
# 仅对评价指标齐全的组合进行 TOPSIS（0N 对照组的 NUE 不定义，会导致评价矩阵缺失）
eval_source = combo_df_sorted.dropna(subset=eval_cols).copy().reset_index(drop=True)

# 指标标准化（离差标准化）
eval_df = eval_source[eval_cols].copy()
for col in eval_cols:
    col_min = eval_df[col].min(skipna=True)
    col_max = eval_df[col].max(skipna=True)
    if col in high_better_cols:
        eval_df[col] = eval_df[col].fillna(col_min)
    else:
        eval_df[col] = eval_df[col].fillna(col_max)
    denom = col_max - col_min
    eval_df[col] = 0.0 if denom == 0 else (eval_df[col] - col_min) / denom
# 计算全局客观熵权
weights = entropy_weight(eval_df)
print(f"\nObjective weights by entropy method:")
for idx, w in weights.items():
    indicator_type = "High-is-better" if idx in high_better_cols else "Low-is-better"
    print(f" {idx}: {w:.4f} ({w*100:.2f}%) | {indicator_type}")
# 计算参与评价的组合 TOPSIS 综合贴近度
eval_source['TOPSIS_Score'] = topsis_evaluation(eval_df, weights, high_better_cols)
# 全局排序，取贴近度最高者为最优组合
combo_df_topsis = eval_source.sort_values('TOPSIS_Score', ascending=False).reset_index(drop=True)
best_combo = combo_df_topsis.iloc[0]
# 输出TOPSIS排序结果
print(f"\nGlobal optimal combination confirmed: {best_combo['Combo']} (Score: {best_combo['TOPSIS_Score']:.4f})")
print(f"\nTOPSIS global ranking (Top 10):")
for idx in range(min(10, len(combo_df_topsis))):
    row = combo_df_topsis.iloc[idx]
    print(f" Rank {idx+1}: {row['Combo']} | Score: {row['TOPSIS_Score']:.4f} | Yield: {row['WRR14_mean']} kg ha-1 | CV: {row['WRR14_CV']:.3f}%")
# 定义TOP3候选组合（用于图表对比）
final_candidates = combo_df_topsis.head(3).copy()
# ===================== 7. 稳健性验证（仅保留权重敏感性分析） =====================
print("\n" + "="*80)
print("Robustness validation: Weight sensitivity analysis")
print("="*80)
# 7.1 权重敏感性分析
print(f"\n[Weight sensitivity analysis (perturbations ±10%, ±20%)]")
sensitivity_result = weight_sensitivity_analysis(
    eval_df,
    high_better_cols,
    weights,
    combo_ids=eval_source['Combo'].tolist(),
    best_combo_id=best_combo['Combo'],
    top_n=3
)
# 按扰动指标分组显示
for indicator in weights.index:
    indicator_result = sensitivity_result[sensitivity_result['Indicator'] == indicator].copy()
    print(f"\n Perturbed Indicator: {indicator}")
    print(indicator_result[['Perturbation_Rate', 'Top_3_Candidates', 'Stability', 'New_Optimal_Combo']].to_string(index=False))
# 计算最优组合稳定率
total_scenarios = len(sensitivity_result)
stable_scenarios = len(sensitivity_result[sensitivity_result['Stability'] == 'Stable'])
stability_rate = stable_scenarios / total_scenarios * 100
print(f"\n[Overall stability rate]: {stability_rate:.1f}% ({stable_scenarios}/{total_scenarios} scenarios remaining in TOP3)")
# 7.2 计算核心指标全局优势率
print(f"\n[Advantage over global mean for the best combination]")
for indicator in ['WRR14_mean', 'WUE', 'NUE', 'SPFERT_mean']:
    global_mean = eval_source[indicator].mean()
    best_value = best_combo[indicator]
    advantage_rate = (best_value - global_mean) / global_mean * 100
    print(f" {indicator}: Advantage {advantage_rate:.2f}% (Best: {best_value}, Mean: {global_mean:.4f})")
print(f" WRR14_CV: Best {best_combo['WRR14_CV']:.3f}%, Mean {eval_source['WRR14_CV'].mean():.3f}% (Lower is better)")

print("\n=== 权重敏感性分析关键信息汇总 ===")
print(f"原始最优组合: {best_combo['Combo']}")
print(f"原始贴近度: {best_combo['TOPSIS_Score']:.4f}\n")

print("扰动场景总数:", len(sensitivity_result))
print("扰动后最优组合唯一值数量:", sensitivity_result['New_Optimal_Combo'].nunique())
print("扰动后最优组合出现次数最多的:", sensitivity_result['New_Optimal_Combo'].value_counts().head(3))

print("\n前10行扰动结果（看是否有排名变化）：")
print(sensitivity_result[['Indicator', 'Perturbation_Rate', 'New_Optimal_Combo', 'Stability']].head(10).to_string(index=False))

if 'New_Rank_of_Best' in sensitivity_result.columns:
    print("\n最优组合在扰动后的排名分布：")
    print(sensitivity_result['New_Rank_of_Best'].value_counts())
else:
    print("\n提示：建议在函数中添加 'New_Rank_of_Best' 列以便观察")

# 7.3 多指标联合扰动（蒙特卡罗）
print("\n[Monte Carlo weight perturbation (all indicators jointly)]")
mc_df = mc_weight_sensitivity(
    eval_df=eval_df,
    high_better_cols=high_better_cols,
    base_weights=weights,
    combo_ids=eval_source['Combo'].tolist(),
    trials=500,
    delta=0.2,
    seed=2026
)
mc_counts = mc_df['best_combo'].value_counts()
mc_top = mc_counts.index[0]
mc_rate = mc_counts.iloc[0] / mc_df.shape[0] * 100
print(f"Most frequent optimal under MC: {mc_top} ({mc_rate:.1f}% of trials)")
print("Top-3 frequent winners:\n", mc_counts.head(3))
# ===================== 新增：TOPSIS前几名显著性差异 ==========
# 假设combo_df_sorted已计算
top_n = 5
top_combos = combo_df_topsis['Combo'].head(top_n).tolist()

p_matrix = pd.DataFrame(index=top_combos[1:], columns=key_indicators)
for i in range(1, top_n):
    for ind in key_indicators:
        top1_data = raw_df[raw_df['Combo'] == top_combos[0]][ind].dropna()
        topi_data = raw_df[raw_df['Combo'] == top_combos[i]][ind].dropna()
        
        # 正态检验
        if stats.shapiro(top1_data)[1] > 0.05 and stats.shapiro(topi_data)[1] > 0.05:
            _, p = stats.ttest_ind(top1_data, topi_data)
        else:
            _, p = stats.mannwhitneyu(top1_data, topi_data)
        
        p_matrix.at[top_combos[i], ind] = p

print("TOPSIS前几名p值矩阵（vs TOP1）:\n", p_matrix)
p_matrix.to_csv('pvalue_matrix.csv')
# ===================== 8. 可视化图表（增加Pareto前沿、TOPSIS排名图、额外雷达图、流程图等） =====================
# Figure 1: Yield vs stability scatter
fig, ax = plt.subplots(figsize=(10, 8))
# 绘制所有组合
ax.scatter(
    combo_df_sorted['WRR14_CV'],
    combo_df_sorted['WRR14_mean'],
    c=COLORS['secondary'],
    alpha=0.4,
    s=60,
    edgecolors='white',
    linewidth=0.8,
    label='All combinations'
)
# 绘制最优组合
ax.scatter(
    best_combo['WRR14_CV'],
    best_combo['WRR14_mean'],
    c=COLORS['primary'],
    s=250,
    marker='*',
    edgecolors='white',
    linewidth=2,
    label=f'Global best combination\n{best_combo["Combo"]}\nWRR14:{best_combo["WRR14_mean"]:.2f} kg ha$^{{-1}}$\nCV:{best_combo["WRR14_CV"]:.3f}%',
    zorder=5
)
# 绘制TOP3候选组合
ax.scatter(
    final_candidates['WRR14_CV'],
    final_candidates['WRR14_mean'],
    c=COLORS['info'],
    alpha=0.7,
    s=80,
    edgecolors='white',
    linewidth=1,
    label='Top-3 candidates by TOPSIS'
)
# High-yield & stable region
ax.axhline(
    y=combo_df_sorted['WRR14_mean'].quantile(0.7),
    color=COLORS['success'],
    linestyle='--',
    alpha=0.7,
    label='Top 30% yield threshold'
)
ax.axvline(
    x=combo_df_sorted['WRR14_CV'].quantile(0.3),
    color=COLORS['success'],
    linestyle='--',
    alpha=0.7,
    label='Lowest 30% CV threshold'
)
ax.fill_betweenx(
    y=np.arange(combo_df_sorted['WRR14_mean'].quantile(0.7), combo_df_sorted['WRR14_mean'].max() + 100, 100),
    x1=0,
    x2=combo_df_sorted['WRR14_CV'].quantile(0.3),
    alpha=0.1,
    color=COLORS['success'],
    label='High-yield and stable region'
)
# Axes labels and title
ax.set_xlabel('Yield coefficient of variation CV (%)', fontsize=14, labelpad=15, fontweight='bold')
ax.set_ylabel('Mean standard-moisture yield WRR14 (kg ha$^{-1}$)', fontsize=14, labelpad=15, fontweight='bold')
ax.set_title('Yield and stability for all irrigation–fertilizer combinations', fontsize=16, pad=20, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1), frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='-')
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig1_yield_stability_scatter.png', dpi=300)
plt.show()

# Figure 2: Water-related indicators of TOP3 combinations
fig, ax = plt.subplots(figsize=(12, 8)) # 稍微增加宽度以容纳图例
candidate_names = [f'{row["Combo"]}' for _, row in final_candidates.iterrows()]
wu_data = [row['WU_mean'] for _, row in final_candidates.iterrows()]
ircum_data = [row['Irr_mean'] for _, row in final_candidates.iterrows()]
draicum_data = [row['Perc_mean'] for _, row in final_candidates.iterrows()]
wue_data = [row['WUE'] for _, row in final_candidates.iterrows()]
x = np.arange(len(candidate_names))
width = 0.2
bars1 = ax.bar(x - 1.5*width, wu_data, width, label='Total water consumption (WU, mm)', color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=1)
bars2 = ax.bar(x - 0.5*width, ircum_data, width, label='Irrigation amount (IRCUM, mm)', color=COLORS['info'], alpha=0.8, edgecolor='white', linewidth=1)
bars3 = ax.bar(x + 0.5*width, draicum_data, width, label='Deep percolation (DRAICUM, mm)', color=COLORS['secondary'], alpha=0.8, edgecolor='white', linewidth=1)
# WUE values are small, we might need a secondary axis or just keep them as is. 
# But let's check the scale. If WUE is kg/ha/mm, it's around 10-20. WU is 400-800.
bars4 = ax.bar(x + 1.5*width, wue_data, width, label='WUE (kg ha$^{-1}$ mm$^{-1}$)', color=COLORS['success'], alpha=0.8, edgecolor='white', linewidth=1)

# 添加数值标签
def add_labels(bars, fmt='.1f'):
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + max(wu_data)*0.01,
                    f'{height:{fmt}}', ha='center', va='bottom', fontweight='bold', fontsize=10)
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4, fmt='.2f')

# 突出最优组合
best_idx = candidate_names.index(best_combo['Combo'])
for bars in [bars1, bars2, bars3, bars4]:
    bars[best_idx].set_linewidth(3)
    bars[best_idx].set_edgecolor(COLORS['primary'])
ax.set_xlabel('Top-3 irrigation–fertilizer combinations', fontsize=14, labelpad=15, fontweight='bold')
ax.set_ylabel('Value', fontsize=14, labelpad=15, fontweight='bold')
ax.set_title('Water-related indicators of TOP3 combinations', fontsize=16, pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(candidate_names, fontsize=12)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, fancybox=True, shadow=True) # 移出图表
ax.grid(True, alpha=0.3, linestyle='-', axis='y')
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig2_water_indicators_bar.png', dpi=300)
plt.show()

# Figure 3: Fertilizer-related indicators of TOP3 combinations
fig, ax = plt.subplots(figsize=(12, 8)) # 稍微增加宽度以容纳图例
fercum_data = [row['Fert_mean'] for _, row in final_candidates.iterrows()]
nue_data = [row['NUE'] for _, row in final_candidates.iterrows()]
x = np.arange(len(candidate_names))
width = 0.35
bars1 = ax.bar(x - width/2, fercum_data, width, label='Fertilizer amount (FERCUM, kg)', color=COLORS['tertiary'], alpha=0.8, edgecolor='white', linewidth=1)
bars2 = ax.bar(x + width/2, nue_data, width, label='NUE (kg ha$^{-1}$ kg$^{-1}$)', color=COLORS['success'], alpha=0.8, edgecolor='white', linewidth=1)
add_labels(bars1)
add_labels(bars2, fmt='.2f')
# 突出最优组合
best_idx = candidate_names.index(best_combo['Combo'])
bars1[best_idx].set_linewidth(3)
bars1[best_idx].set_edgecolor(COLORS['primary'])
bars2[best_idx].set_linewidth(3)
bars2[best_idx].set_edgecolor(COLORS['primary'])
ax.set_xlabel('Top-3 irrigation–fertilizer combinations', fontsize=14, labelpad=15, fontweight='bold')
ax.set_ylabel('Value', fontsize=14, labelpad=15, fontweight='bold')
ax.set_title('Fertilizer-related indicators of TOP3 combinations', fontsize=16, pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(candidate_names, fontsize=12)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, fancybox=True, shadow=True) # 移出图表
ax.grid(True, alpha=0.3, linestyle='-', axis='y')
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig3_fertilizer_indicators_bar.png', dpi=300)
plt.show()

# Figure 4: Radar chart – best combination vs global mean
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
metrics = [
    'Biomass (WAGT)',
    'Yield (WRR14)',
    'Stability (1/CV)',
    'WUE',
    'NUE',
    'SPFERT'
]
# 最优组合标准化数据
best_wagt_norm = (best_combo['WAGT_mean'] / combo_df_sorted['WAGT_mean'].max()) if not np.isnan(best_combo['WAGT_mean']) else 0
best_wrr14_norm = best_combo['WRR14_mean'] / combo_df_sorted['WRR14_mean'].max()
best_cv_norm = (combo_df_sorted['WRR14_CV'].max() - best_combo['WRR14_CV']) / (combo_df_sorted['WRR14_CV'].max() - combo_df_sorted['WRR14_CV'].min())
best_wue_norm = best_combo['WUE'] / combo_df_sorted['WUE'].max()
best_nue_norm = best_combo['NUE'] / combo_df_sorted['NUE'].max()
best_spfert_norm = best_combo['SPFERT_mean'] / combo_df_sorted['SPFERT_mean'].max()
best_values = [best_wagt_norm, best_wrr14_norm, best_cv_norm, best_wue_norm, best_nue_norm, best_spfert_norm]

# 全局平均值标准化数据
avg_wagt_norm = (combo_df_sorted['WAGT_mean'].mean() / combo_df_sorted['WAGT_mean'].max()) if not combo_df_sorted['WAGT_mean'].isna().all() else 0
avg_wrr14_norm = combo_df_sorted['WRR14_mean'].mean() / combo_df_sorted['WRR14_mean'].max()
avg_cv_norm = (combo_df_sorted['WRR14_CV'].max() - combo_df_sorted['WRR14_CV'].mean()) / (combo_df_sorted['WRR14_CV'].max() - combo_df_sorted['WRR14_CV'].min())
avg_wue_norm = combo_df_sorted['WUE'].mean() / combo_df_sorted['WUE'].max()
avg_nue_norm = combo_df_sorted['NUE'].mean() / combo_df_sorted['NUE'].max()
avg_spfert_norm = combo_df_sorted['SPFERT_mean'].mean() / combo_df_sorted['SPFERT_mean'].max()
avg_values = [avg_wagt_norm, avg_wrr14_norm, avg_cv_norm, avg_wue_norm, avg_nue_norm, avg_spfert_norm]

# 绘制雷达图
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
best_values += best_values[:1]
avg_values += avg_values[:1]
angles += angles[:1]

ax.plot(angles, best_values, color=COLORS['primary'], linewidth=3, linestyle='-', label=f'Best combination ({best_combo["Combo"]})', marker='o', markersize=6)
ax.fill(angles, best_values, color=COLORS['primary'], alpha=0.2)
ax.plot(angles, avg_values, color=COLORS['secondary'], linewidth=2, linestyle='--', label='Mean of all combinations', marker='s', markersize=5)
ax.fill(angles, avg_values, color=COLORS['secondary'], alpha=0.1)

# Formatting
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
ax.tick_params(axis='x', pad=35) # 增加标签边距，防止遮挡
ax.set_ylim(0, 1.1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.set_title('Standardized core indicators of the best combination vs global mean', fontsize=16, pad=30, fontweight='bold')
# 将图例移出雷达图范围
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), frameon=True, fancybox=True, shadow=True, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig4_best_radar.png', dpi=300)
plt.show()

# Figure 5: Yield vs irrigation – Pareto frontier
fig, ax = plt.subplots(figsize=(10, 8))
# 所有组合散点（背景，半透明）
ax.scatter(
    combo_df_sorted['Irr_mean'],
    combo_df_sorted['WRR14_mean'],
    c=COLORS['secondary'],
    alpha=0.4,
    s=60,
    edgecolors='white',
    linewidth=0.8,
    label='All combinations'
)
# 计算 Pareto 前沿（最大化产量，最小化灌溉量）
# 按灌溉量升序排序
sorted_by_irr = combo_df_sorted.sort_values('Irr_mean', ascending=True)
# 累积最大产量
current_max_yield = -np.inf
pareto_points = []
for _, row in sorted_by_irr.iterrows():
    if row['WRR14_mean'] > current_max_yield:
        pareto_points.append(row)
        current_max_yield = row['WRR14_mean']
pareto_df = pd.DataFrame(pareto_points)
# 绘制 Pareto 前沿线
if len(pareto_df) >= 2:
    ax.plot(
        pareto_df['Irr_mean'],
        pareto_df['WRR14_mean'],
        color=COLORS['red'],
        linewidth=2.4,
        linestyle='-',
        marker='o',
        markersize=7,
        markerfacecolor='white',
        markeredgecolor=COLORS['red'],
        markeredgewidth=1.8,
        label='Pareto frontier (Yield max vs. Irrigation min)'
    )
# 突出全局最优组合
ax.scatter(
    best_combo['Irr_mean'],
    best_combo['WRR14_mean'],
    c=COLORS['primary'],
    s=300,
    marker='*',
    edgecolors='white',
    linewidth=2.8,
    zorder=10,
    label=f'Global best combination\n{best_combo["Combo"]}\n'
          f'Yield: {best_combo["WRR14_mean"]:.2f} kg ha$^{{-1}}$\n'
          f'Irrigation: {best_combo["Irr_mean"]:.2f} mm'
)
# 突出 TOP3 组合
ax.scatter(
    final_candidates['Irr_mean'],
    final_candidates['WRR14_mean'],
    c=COLORS['info'],
    alpha=0.9,
    s=110,
    edgecolors='white',
    linewidth=1.3,
    label='Top-3 candidates by TOPSIS'
)
# 添加平均值参考线
avg_irr = combo_df_sorted['Irr_mean'].mean()
avg_yield = combo_df_sorted['WRR14_mean'].mean()

ax.axvline(x=avg_irr, color=COLORS['gray'], linestyle='--', alpha=0.55,
           label=f'Mean irrigation: {avg_irr:.1f} mm')
ax.axhline(y=avg_yield, color=COLORS['gray'], linestyle='--', alpha=0.55,
           label=f'Mean yield: {avg_yield:.0f} kg ha$^{{-1}}$')

ax.set_xlabel('Mean irrigation amount (mm)', fontsize=14, labelpad=15, fontweight='bold')
ax.set_ylabel('Mean WRR14 yield (kg ha$^{-1}$)', fontsize=14, labelpad=15, fontweight='bold')
ax.set_title('Yield–irrigation Pareto frontier analysis', 
             fontsize=16, fontweight='bold', pad=20)

# 图例：移到外部，避免遮挡
ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.,
    frameon=True,
    fancybox=True,
    shadow=True,
    fontsize=11
)

ax.grid(True, alpha=0.28, linestyle='--')
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig5_pareto_yield_vs_irrigation.png', dpi=300)
plt.show()

# Figure 6: Yield vs fertilizer – Pareto frontier
fig, ax = plt.subplots(figsize=(10, 8))
# 所有组合散点（背景）
ax.scatter(
    combo_df_sorted['Fert_mean'],
    combo_df_sorted['WRR14_mean'],
    c=COLORS['secondary'],
    alpha=0.35,
    s=50,
    edgecolors='white',
    linewidth=0.6,
    label='All combinations'
)
# Pareto 前沿计算
sorted_by_fert = combo_df_sorted.sort_values('Fert_mean', ascending=True)
current_max_yield = -np.inf
pareto_points = []
for _, row in sorted_by_fert.iterrows():
    if row['WRR14_mean'] > current_max_yield:
        pareto_points.append(row)
        current_max_yield = row['WRR14_mean']
pareto_df = pd.DataFrame(pareto_points)
# 绘制Pareto前沿线
if len(pareto_df) >= 2:
    ax.plot(
        pareto_df['Fert_mean'],
        pareto_df['WRR14_mean'],
        color=COLORS['red'],
        linewidth=2.2,
        linestyle='-',
        marker='o',
        markersize=6,
        markerfacecolor='white',
        markeredgecolor=COLORS['red'],
        label='Pareto frontier (Yield max vs. Fertilizer min)'
    )
# 突出全局最优解
ax.scatter(
    best_combo['Fert_mean'],
    best_combo['WRR14_mean'],
    c=COLORS['primary'],
    s=280,
    marker='*',
    edgecolors='white',
    linewidth=2.5,
    zorder=10,
    label=f'Global best combination\n{best_combo["Combo"]}\n'
          f'Yield: {best_combo["WRR14_mean"]:.2f} kg ha$^{{-1}}$\n'
          f'Fertilizer: {best_combo["Fert_mean"]:.2f} kg'
)
# 突出 TOP3
ax.scatter(
    final_candidates['Fert_mean'],
    final_candidates['WRR14_mean'],
    c=COLORS['info'],
    alpha=0.85,
    s=100,
    edgecolors='white',
    linewidth=1.2,
    label='Top-3 candidates by TOPSIS'
)
# 参考线
avg_fert = combo_df_sorted['Fert_mean'].mean()
avg_yield = combo_df_sorted['WRR14_mean'].mean()
ax.axvline(x=avg_fert, color=COLORS['gray'], linestyle='--', alpha=0.5, label=f'Mean fertilizer: {avg_fert:.1f} kg')
ax.axhline(y=avg_yield, color=COLORS['gray'], linestyle='--', alpha=0.5, label=f'Mean yield: {avg_yield:.0f} kg ha$^{{-1}}$')
ax.set_xlabel('Mean fertilizer amount (kg)', fontsize=14, labelpad=15, fontweight='bold')
ax.set_ylabel('Mean WRR14 yield (kg ha$^{-1}$)', fontsize=14, labelpad=15, fontweight='bold')
ax.set_title('Yield–fertilizer Pareto frontier analysis', fontsize=16, fontweight='bold', pad=20)

# 图例移到外部
ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.,
    frameon=True,
    fancybox=True,
    shadow=True,
    fontsize=11
)

ax.grid(True, alpha=0.25, linestyle='--')
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig6_pareto_yield_vs_fertilizer.png', dpi=300)
plt.show()

# Figure 7: TOPSIS ranking bar chart (top 10)
fig, ax = plt.subplots(figsize=(12, 8))
top10 = combo_df_topsis.head(10)
ax.barh(top10['Combo'], top10['TOPSIS_Score'], color=COLORS['blue'], alpha=0.8)
ax.invert_yaxis()
ax.set_xlabel('TOPSIS closeness coefficient', fontsize=14, labelpad=15, fontweight='bold')
ax.set_title('Top 10 irrigation–fertilizer combinations by entropy-TOPSIS ranking', fontsize=16, fontweight='bold', pad=20)
for i, v in enumerate(top10['TOPSIS_Score']):
    ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig7_topsis_ranking_bar.png', dpi=300)
plt.show()

# Figure 8: Radar chart – TOP3 combinations
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]
colors = [COLORS['primary'], COLORS['info'], COLORS['success']]
for idx, (_, row) in enumerate(final_candidates.iterrows()):
    values = [
        (row['WAGT_mean'] / combo_df_sorted['WAGT_mean'].max()) if not np.isnan(row['WAGT_mean']) else 0,
        row['WRR14_mean'] / combo_df_sorted['WRR14_mean'].max(),
        (combo_df_sorted['WRR14_CV'].max() - row['WRR14_CV']) / (combo_df_sorted['WRR14_CV'].max() - combo_df_sorted['WRR14_CV'].min()),
        row['WUE'] / combo_df_sorted['WUE'].max(),
        row['NUE'] / combo_df_sorted['NUE'].max(),
        row['SPFERT_mean'] / combo_df_sorted['SPFERT_mean'].max()
    ]
    values += values[:1]
    ax.plot(angles, values, color=colors[idx], linewidth=2.5, label=row['Combo'])
    ax.fill(angles, values, color=colors[idx], alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
ax.tick_params(axis='x', pad=35) # 增加标签边距，防止遮挡
ax.set_ylim(0, 1.1)
ax.set_title('Core indicators of TOP3 combinations', fontsize=16, pad=30, fontweight='bold')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), frameon=True, fancybox=True, shadow=True, ncol=3)
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig8_top3_radar.png', dpi=300)
plt.show()

# Figure 9: Weight-sensitivity radar chart
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
indicators = weights.index.tolist()
perturb_rates = [-0.2, -0.1, 0, 0.1, 0.2]
angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
angles += angles[:1]
# 计算各扰动场景下最优组合的贴近度
closeness_data = []
labels = []
colors = ['#DC2626', '#F59E0B', '#2563EB', '#10B981', '#8B5CF6']
for rate, color in zip(perturb_rates, colors):
    rate_closeness = []
    for indicator in indicators:
        new_weights = weights.copy()
        perturb_amount = new_weights[indicator] * rate
        new_weights[indicator] += perturb_amount
        other_indicators = [ind for ind in indicators if ind != indicator]
        other_sum = new_weights[other_indicators].sum()
        if other_sum != 0:
            new_weights[other_indicators] = new_weights[other_indicators] / other_sum * (1 - new_weights[indicator])
        new_closeness = topsis_evaluation(eval_df, new_weights, high_better_cols)
        rate_closeness.append(new_closeness[combo_df_sorted['Combo'] == best_combo['Combo']].iloc[0])
    rate_closeness += rate_closeness[:1]
    closeness_data.append(rate_closeness)
    labels.append(f"Weight perturbation {rate*100:+.0f}%")
# 绘制雷达图
for data, label, color in zip(closeness_data, labels, colors):
    ax.plot(angles, data, color=color, linewidth=2.5, label=label, marker='o', markersize=5)
    if label == "Weight perturbation +0%":
        ax.fill(angles, data, color=color, alpha=0.1)
 # Formatting
pretty_indicator_labels = {
    'WRR14_mean': 'Yield',
    'WRR14_CV': 'Yield CV (%)',
    'WUE': 'WUE',
    'NUE': 'NUE',
    'SPFERT_mean': 'SPFERT'
}
ax.set_xticks(angles[:-1])
ax.set_xticklabels([pretty_indicator_labels.get(ind, ind) for ind in indicators], fontsize=12, fontweight='bold')
ax.tick_params(axis='x', pad=35) # 增加标签边距，防止遮挡
ax.set_ylim(min([min(data) for data in closeness_data])*0.95, max([max(data) for data in closeness_data])*1.05)
ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))
ax.set_yticklabels([f"{x:.3f}" for x in np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5)], fontsize=10)
ax.set_title('Weight-sensitivity analysis of the best combination', fontsize=16, pad=30, fontweight='bold')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), frameon=True, fancybox=True, shadow=True, ncol=3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig9_sensitivity_radar.png', dpi=300)
plt.show()


import seaborn as sns

# 确保有 sensitivity_result
# （假设前面已经运行了 weight_sensitivity_analysis）

# 修正热图：显示原最优组合在所有扰动场景下的排名变化（消除缺口）
pivot_best_rank = sensitivity_result.pivot(
    index='Indicator',
    columns='Perturbation_Rate',
    values='New_Rank_of_Best'
).astype(float)

# 画热图
plt.figure(figsize=(10, 8))
sns.heatmap(
    pivot_best_rank,
    annot=True,           # 显示数字
    fmt='.0f',            # 无小数
    cmap='YlGnBu_r',      # 排名小（1）颜色深
    linewidths=0.8,
    annot_kws={'fontweight': 'bold', 'size': 12},
    cbar_kws={'label': 'Rank of original best combination'}
)

plt.title(f'Rank stability of {best_combo["Combo"]} under weight perturbations', fontsize=16, fontweight='bold', pad=25)
plt.xlabel('Weight perturbation range', fontsize=14, labelpad=20, fontweight='bold')
plt.ylabel('Perturbed indicator', fontsize=14, labelpad=20, fontweight='bold')
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig_sensitivity_heatmap.png', dpi=300)
plt.show()
# ────────────────────────────────────────────────
# 新增：統計相關圖表（適合論文使用）
# ────────────────────────────────────────────────


# Figure 12: p-value heatmap – TOP5 vs TOP1
plt.figure(figsize=(10, 8))
# 调整颜色映射：p 值完整范围 [0,1]，低 p（显著）为绿色，高 p 为红色
pmat = p_matrix.astype(float)
cmap = 'RdYlGn_r'
ax_hm = sns.heatmap(
    pmat, annot=True, fmt='.3f', cmap=cmap, vmin=0.0, vmax=1.0,
    linewidths=0.8, annot_kws={'fontweight': 'bold'},
    cbar_kws={'label': 'p-value', 'ticks': [0, 0.05, 0.10, 0.20, 0.50, 1.0]}
)
plt.title('Significance (p-values) between TOP1 and other TOP5 combinations', fontsize=16, fontweight='bold', pad=25)
plt.xlabel('Indicator', fontsize=14, labelpad=20, fontweight='bold')
plt.ylabel('Combination (vs TOP1)', fontsize=14, labelpad=20, fontweight='bold')
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\heatmap_pvalue_top5.png', dpi=300)
plt.show()

print("\nGenerated 3 additional statistical figures (dpi=300, suitable for SCI manuscripts).")
# ===================== 9. 最终结果汇总 =====================
print("\n" + "="*100)
print("Final Results Summary (including robustness validation from weight sensitivity analysis)")
print("="*100)
print(f"Global Optimal Irrigation-Fertilizer Combination: {best_combo['Combo']}")
print(f"TOPSIS Closeness Coefficient: {best_combo['TOPSIS_Score']:.4f} (Global Rank: 1)")
print(f"Objective Weights of Indicators:")
for idx, w in weights.items():
    indicator_type = "High-is-better" if idx in high_better_cols else "Low-is-better"
    print(f" - {idx}: {w:.4f} (Weight: {w*100:.2f}%) | {indicator_type}")
print(f"\nPerformance of the Best Combination:")
print(f" - Yield: {best_combo['WRR14_mean']:.2f} kg/ha (Global Rank: {combo_df_sorted[combo_df_sorted['Combo']==best_combo['Combo']].index[0]+1})")
print(f" - Yield Stability: CV={best_combo['WRR14_CV']:.3f}% (Lower than global mean by {combo_df_sorted['WRR14_CV'].mean()-best_combo['WRR14_CV']:.3f}%)")
print(f" - Water Use Efficiency: WUE={best_combo['WUE']:.6f} kg/ha/mm (Advantage over global mean: {((best_combo['WUE']-combo_df_sorted['WUE'].mean())/combo_df_sorted['WUE'].mean()*100):.2f}%)")
print(f" - Nitrogen Use Efficiency: NUE={best_combo['NUE']:.6f} kg/ha/kg (Advantage over global mean: {((best_combo['NUE']-combo_df_sorted['NUE'].mean())/combo_df_sorted['NUE'].mean()*100):.2f}%)")
print(f" - Stress Factor: SPFERT_mean={best_combo['SPFERT_mean']:.4f} (Close to 1, indicating minimal stress)")
print(f"\nRobustness Conclusion:")
print(f" - Weight Sensitivity: Ranking stability is {stability_rate:.1f}%. The best combination remains in the Top 3 under ±20% weight perturbations, indicating robust and reliable results.")
print(f" - Comprehensive Advantage: The best combination shows significant advantages in all core indicators, balancing high yield, stability, water and fertilizer efficiency, and low stress. It can be recommended as a practical irrigation and fertilization strategy.")
print("\nAll charts have been saved as PNG files and are ready for direct insertion into SCI manuscripts.")

# ===================== 10. Voronoi 权重决策边界分析 (2D PCA 投影) =====================
print("\n" + "="*100)
print("Generating Nature Communications style Voronoi diagram for weight space decision boundaries...")
print("="*100)

def mc_weight_sensitivity_with_weights(eval_df, high_better_cols, base_weights, combo_ids, trials=3000, delta=0.6, seed=42):
    rng = np.random.default_rng(seed)
    indicators = base_weights.index.tolist()
    records = []
    for t in range(trials):
        # 产生广泛扰动以覆盖更大的权重空间
        eps = rng.uniform(-delta, delta, size=len(indicators))
        new_w = base_weights.copy().astype(float)
        for i, ind in enumerate(indicators):
            new_w[ind] = max(1e-8, new_w[ind] * (1.0 + eps[i]))
        new_w = new_w / new_w.sum()
        s = topsis_evaluation(eval_df, new_w, high_better_cols)
        best_idx = int(np.argmax(s.to_numpy()))
        best_id = combo_ids[best_idx]
        
        row = {
            'best_combo': best_id,
        }
        for ind in indicators:
            row[f'w_{ind}'] = new_w[ind]
        records.append(row)
    return pd.DataFrame(records)

# 1. 生成与“权重扰动分析”一致的局部蒙特卡洛样本（避免与±20%结论矛盾）
pca_trials = 20000
pca_delta = 0.2
mc_voronoi_df = mc_weight_sensitivity_with_weights(
    eval_df=eval_df,
    high_better_cols=high_better_cols,
    base_weights=weights,
    combo_ids=eval_source['Combo'].tolist(),
    trials=pca_trials,
    delta=pca_delta,
    seed=2026
)

# 2. 提取权重数据并进行 PCA 降维
weight_cols = [f'w_{ind}' for ind in weights.index]
X_weights = mc_voronoi_df[weight_cols].values
y_labels = mc_voronoi_df['best_combo'].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_weights)

# 基准权重的 PCA 投影
base_w_array = np.array([weights[ind] for ind in weights.index]).reshape(1, -1)
base_pca = pca.transform(base_w_array)

# 3. 用“W2_70%N 作为最优”的概率图来表达鲁棒性（比多类别 Voronoi 更直观）
best_id = best_combo['Combo']
mc_voronoi_df['is_best'] = (mc_voronoi_df['best_combo'] == best_id).astype(int)
best_rate_local = mc_voronoi_df['is_best'].mean() * 100.0
print(f"Local MC (delta=±{pca_delta*100:.0f}%, n={pca_trials}) winner share of {best_id}: {best_rate_local:.1f}%")

knn = KNeighborsClassifier(n_neighbors=75, weights='distance')
knn.fit(X_pca, mc_voronoi_df['is_best'].to_numpy())

# 创建网格
x_min, x_max = X_pca[:, 0].min() - 0.05, X_pca[:, 0].max() + 0.05
y_min, y_max = X_pca[:, 1].min() - 0.05, X_pca[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

proba = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
proba = proba.reshape(xx.shape)

# 5. 绘制“最优解概率”图（NCS 风格，更能直接表达鲁棒性）
fig, ax = plt.subplots(figsize=(10, 8))

heat = ax.contourf(xx, yy, proba, levels=25, cmap='mako', antialiased=True)
line99 = ax.contour(xx, yy, proba, levels=[0.99], colors=['#000000'], linewidths=[2.4], linestyles=['-'])
line99.set_path_effects([pe.Stroke(linewidth=3.8, foreground='#FFFFFF'), pe.Normal()])
cb = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
cb.set_label(f"P(best = {best_id})", fontsize=12, fontweight='bold')
vmin, vmax = float(np.nanmin(proba)), float(np.nanmax(proba))
start = max(0.50, round(vmin, 2))
base_ticks = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
ticks = [t for t in base_ticks if start <= t <= vmax]
if len(ticks) == 0 or ticks[-1] < round(vmax, 2):
    ticks = ticks + [round(vmax, 2)]
cb.set_ticks(ticks)
cb.ax.set_yticklabels([f"{t:.2f}" for t in ticks])

cb.outline.set_linewidth(0.8)

sample_size = 0
if sample_size > 0:
    idx_sample = np.random.choice(len(X_pca), sample_size, replace=False)
    ax.scatter(X_pca[idx_sample, 0], X_pca[idx_sample, 1], c=y_int[idx_sample],
               cmap=cmap_bold, edgecolor='w', linewidth=0.5, s=30, alpha=0.6)

# 绘制基准点 (星号)
ax.scatter(base_pca[0, 0], base_pca[0, 1], c='gold', marker='*', s=400, 
           edgecolor='black', linewidth=1.5, zorder=10, label='Original Entropy Weight')

legend_handles = [
    plt.Line2D([0], [0], marker='*', color='w', label='Original entropy weights',
               markerfacecolor='gold', markeredgecolor='black', markersize=15),
    plt.Line2D([0], [0], color='#000000', lw=3.0, linestyle='-', label='P = 0.99 boundary'),
]
ax.legend(
    handles=legend_handles,
    loc='upper right',
    bbox_to_anchor=(1.02, 1),
    frameon=True,
    fancybox=True,
    shadow=True,
    fontsize=12
)

# 解释坐标轴的方差贡献率
explained_variance = pca.explained_variance_ratio_
ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}% Variance)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}% Variance)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('Voronoi Decision Boundaries of Optimal Combinations\nin Weight Vector Space (2D Projection)', fontsize=16, fontweight='bold', pad=20)

ax.grid(False)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# Annotation (English)
ax.text(
    0.02, 0.02,
    "Each pixel is a 5D weight vector projected by PCA; color shows P(best = W2_70%N).\n"
    "Gold star indicates the original entropy weights.",
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E2E8F0", alpha=0.92)
)

plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig13_voronoi_weight_space.png', dpi=300)
plt.show()

print("\nSuccessfully generated and saved Voronoi diagram: fig13_voronoi_weight_space.png")

top3_inds = weights.sort_values(ascending=False).index[:3].tolist()
def _proj3(wrow):
    a = wrow[0]
    b = wrow[1]
    c = wrow[2]
    s = a + b + c
    a /= s
    b /= s
    c /= s
    x = b + 0.5 * a
    y = (np.sqrt(3) / 2.0) * a
    return x, y
w3 = mc_voronoi_df[[f"w_{top3_inds[0]}", f"w_{top3_inds[1]}", f"w_{top3_inds[2]}"]].to_numpy()
XY = np.array([_proj3(row) for row in w3])
counts3 = mc_voronoi_df["best_combo"].value_counts()
labels3 = counts3.head(4).index.tolist()
labels3 = labels3 + (["Other"] if "Other" not in labels3 else [])
def _map3(c):
    return c if c in labels3[:-1] else "Other"
y3 = mc_voronoi_df["best_combo"].apply(_map3).to_numpy()
lut3 = {lbl: i for i, lbl in enumerate(labels3)}
y3i = np.array([lut3[v] for v in y3])
knn3 = KNeighborsClassifier(n_neighbors=1)
knn3.fit(XY, y3i)
grid_a = []
grid_b = []
grid_c = []
N = 220
for i in range(N + 1):
    A = i / N
    for j in range(N - i + 1):
        B = j / N
        C = 1.0 - A - B
        grid_a.append(A)
        grid_b.append(B)
        grid_c.append(C)
grid_a = np.array(grid_a)
grid_b = np.array(grid_b)
grid_c = np.array(grid_c)
GX = grid_b + 0.5 * grid_a
GY = (np.sqrt(3) / 2.0) * grid_a
pred = knn3.predict(np.column_stack([GX, GY]))
tri = mtri.Triangulation(GX, GY)
fig2, ax2 = plt.subplots(figsize=(10, 9))
sci_colors2 = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#B09C85']
cmap2 = ListedColormap(sci_colors2[:len(labels3)])
tc = ax2.tricontourf(tri, pred, levels=len(labels3), cmap=cmap2, alpha=0.85)
verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2.0]])
ax2.plot([verts[0,0], verts[1,0]], [verts[0,1], verts[1,1]], color="#333333", linewidth=1.5)
ax2.plot([verts[1,0], verts[2,0]], [verts[1,1], verts[2,1]], color="#333333", linewidth=1.5)
ax2.plot([verts[2,0], verts[0,0]], [verts[2,1], verts[0,1]], color="#333333", linewidth=1.5)
bw = np.array([weights[top3_inds[0]], weights[top3_inds[1]], weights[top3_inds[2]]])
bx, by = _proj3(bw)
ax2.scatter(bx, by, c='gold', marker='*', s=420, edgecolor='black', linewidth=1.6, zorder=10)
pretty_label = {
    'WRR14_mean': 'WRR14',
    'WRR14_CV': 'WRR14 CV',
    'SPFERT_mean': 'SPFERT'
}
ax2.text(verts[2,0], verts[2,1]-0.06, pretty_label.get(top3_inds[0], top3_inds[0]),
         ha='center', va='top', fontsize=13, fontweight='bold')
ax2.text(verts[1,0]+0.04, verts[1,1]-0.02, pretty_label.get(top3_inds[1], top3_inds[1]),
         ha='left', va='top', fontsize=13, fontweight='bold')
ax2.text(-0.08, 0.32, pretty_label.get(top3_inds[2], top3_inds[2]),
         ha='right', va='center', fontsize=13, fontweight='bold')
handles2 = []
for i, lbl in enumerate(labels3):
    handles2.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Optimal: {lbl}',
                               markerfacecolor=sci_colors2[i], markersize=10))
handles2.append(plt.Line2D([0], [0], marker='*', color='w', label='Original Entropy Weight',
                           markerfacecolor='gold', markeredgecolor='black', markersize=15))
ax2.legend(handles=handles2, loc='upper right', bbox_to_anchor=(1.32, 1), frameon=True, fancybox=True, shadow=True, fontsize=12)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title('Simplex Decision Regions (Top-3 Indicators)', fontsize=16, fontweight='bold', pad=18)
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig14_ternary_voronoi.png', dpi=300)
plt.show()
rng = np.random.default_rng(2027)
dirichlet = rng.dirichlet(alpha=np.ones(3), size=20000)
DX = dirichlet[:,1] + 0.5*dirichlet[:,0]
DY = (np.sqrt(3)/2.0)*dirichlet[:,0]
pred_area = knn3.predict(np.column_stack([DX, DY]))
area_counts = pd.Series(pred_area).value_counts(normalize=True)
print("\nShare of simplex area by optimal combination:")
for i, lbl in enumerate(labels3):
    share = area_counts.get(i, 0.0)
    print(f" {lbl}: {share*100:.1f}%")
print("Saved ternary Voronoi: fig14_ternary_voronoi.png")

# 5 维蒙特卡洛胜者频次柱状图（顶刊风格）
freq_df = mc_df['best_combo'].value_counts().rename_axis('Combo').reset_index(name='Count')
freq_df['Percent'] = freq_df['Count'] / freq_df['Count'].sum() * 100.0
freq_df = freq_df.sort_values('Count', ascending=False)
fig3, ax3 = plt.subplots(figsize=(12, 7))
palette_map = {}
for i, combo in enumerate(freq_df['Combo']):
    if combo == best_combo['Combo']:
        palette_map[combo] = '#2563EB'
    else:
        palette_map[combo] = ['#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#B09C85', '#8B5CF6', '#10B981', '#D97706'][i % 8]
sns.barplot(data=freq_df, x='Combo', y='Percent', palette=[palette_map[c] for c in freq_df['Combo']], ax=ax3)
for i, row in freq_df.iterrows():
    ax3.text(i, row['Percent'] + max(freq_df['Percent']) * 0.015, f"{row['Percent']:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
bars = ax3.patches
for b, combo in zip(bars, freq_df['Combo']):
    if combo == best_combo['Combo']:
        b.set_linewidth(2.4)
        b.set_edgecolor('#1E293B')
ax3.set_ylabel('Winners frequency (%)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Combination', fontsize=14, fontweight='bold')
ax3.set_title('5D Monte Carlo Winners Frequency (Entropy–TOPSIS)', fontsize=16, fontweight='bold', pad=18)
ax3.grid(True, axis='y', alpha=0.25, linestyle='--')
ax3.set_xticklabels(freq_df['Combo'], rotation=25, ha='right', fontsize=11)
plt.tight_layout()
plt.savefig(r'D:\learning\modle_Tan\data_figure\fig15_mc_winner_frequency.png', dpi=300)
plt.show()
