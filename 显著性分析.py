import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg
import scikit_posthocs as sp
import re
import warnings
from string import ascii_lowercase
warnings.filterwarnings('ignore')
import os
OUTPUT_DIR = r"D:\learning\modle_Tan\数据分析"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)          # ← 这一行最关键，直接把当前目录切换过去
print(f"当前工作目录已切换为：{os.getcwd()}")
# ────────────────────────────────────────────────
# 1. 数据读取（保持原样，略微优化打印，并保存原始数据表）
# ────────────────────────────────────────────────
def load_data(file_path=r'D:\learning\modle_Tan\IR×FER.xlsx'):
    xl = pd.ExcelFile(file_path)
    sheets = xl.sheet_names
    
    all_df = pd.DataFrame()
    
    for sheet in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df.columns = df.columns.str.strip()
        
        # 过滤掉 "Mean" 行（新 IR×FER.xlsx 结构）
        if 'YEAR' in df.columns:
            df = df[df['YEAR'].astype(str).str.lower() != 'mean']
            df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
        
        try:
            parts = str(sheet).split('+')
            fert_part = parts[0].upper().replace('N', '').strip()
            if fert_part.endswith('%'):
                fert_pct = 100 + int(fert_part.replace('%', '').strip())
            else:
                fert_abs = float(fert_part)
                fert_pct = int(round(fert_abs / 225.0 * 100.0))
            
            irr_match = re.search(r'(NO IR|WCFC%|IR)', sheet)
            irr_level = 'W1' if irr_match and 'NO IR' in irr_match.group(1) else \
                        'W2' if irr_match and 'WCFC' in irr_match.group(1) else 'W3'
            
            df['Treatment_Combo'] = f"{irr_level}_{fert_pct}%N"
            df['Irrigation_Level'] = irr_level
            df['Fertilizer_Level'] = f"{fert_pct}%N"
            df['Fertilizer_Pct'] = fert_pct
        except Exception as e:
            print(f"组合解析失败：{sheet} ({e})，跳过")
            continue
        
        # 确保关键列为数值型，避免运算错误
        numeric_cols = ['WRR14', 'IRCUM', 'RAINCUM', 'TRCCUM', 'EVSWCUM', 'FERCUM']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 衍生指标
        df['WUE']    = df['WRR14'] / (df['IRCUM'] + df['RAINCUM'] + 1e-6)
        df['IWUE']   = df['WRR14'] / (df['IRCUM'] + 1e-6)
        df['ET_WUE'] = df['WRR14'] / (df['TRCCUM'] + df['EVSWCUM'] + 1e-6)
        df['NUE']    = df['WRR14'] / (df['FERCUM'] + 1e-6)
        df.loc[df['FERCUM'].fillna(0) <= 0, 'NUE'] = np.nan
        
        all_df = pd.concat([all_df, df], ignore_index=True)
    
    all_df = all_df[all_df['YEAR'].between(1981, 2017)]
    
    print(f"总行数：{all_df.shape[0]}，组合数：{all_df['Treatment_Combo'].nunique()}")
    print("每个组合行数：\n", all_df['Treatment_Combo'].value_counts().sort_index())
    
    # 保存原始数据表，便于Origin导入
    all_df.to_csv('raw_data_all.csv', index=False, encoding='utf-8-sig')
    print("✅ 已保存原始数据表：raw_data_all.csv (包含所有年份和衍生指标)")
    
    return all_df


# ────────────────────────────────────────────────
# 2. 显著性分析（扩展：保存所有测试结果到CSV）
# ────────────────────────────────────────────────
def perform_significance_analysis_by_irr(df, indicator, alpha=0.05):
    details = {}
    if 'Irrigation_Level' not in df.columns or 'Treatment_Combo' not in df.columns:
        return details

    irr_order = {'W1': 0, 'W2': 1, 'W3': 2}
    irr_levels = sorted(df['Irrigation_Level'].dropna().astype(str).unique().tolist(), key=lambda x: irr_order.get(x, 99))

    def _fert_pct(label: str) -> int:
        try:
            return int(str(label).split('_')[1].replace('%N', ''))
        except Exception:
            return -999

    for irr in irr_levels:
        dfi = df[df['Irrigation_Level'].astype(str) == str(irr)].copy()
        if indicator == 'NUE':
            dfi = dfi[dfi['Treatment_Combo'] != 'W1_0%N'].copy()

        groups = []
        group_labels = []
        for name, group in dfi.groupby('Treatment_Combo'):
            values = group[indicator].dropna() if indicator in group.columns else pd.Series(dtype=float)
            if len(values) >= 3:
                groups.append(values)
                group_labels.append(str(name))

        if len(groups) < 2:
            continue

        normality_results = []
        for label, g in zip(group_labels, groups):
            stat, p = stats.shapiro(g)
            normality_results.append({'Group': label, 'Shapiro_Stat': stat, 'p_value': p, 'Normal': p > alpha})
        normality_df = pd.DataFrame(normality_results)
        normality_df.to_csv(f'normality_{indicator}_{irr}_within.csv', index=False, encoding='utf-8-sig')
        normal_pass = all(normality_df['Normal'])

        levene_stat, levene_p = stats.levene(*groups)
        levene_df = pd.DataFrame([{'Levene_Stat': levene_stat, 'p_value': levene_p, 'Homoscedastic': levene_p > alpha}])
        levene_df.to_csv(f'levene_{indicator}_{irr}_within.csv', index=False, encoding='utf-8-sig')

        result = {}
        letters_dict = {}

        if normal_pass and levene_p > alpha:
            df_fit = dfi.dropna(subset=[indicator, 'Fertilizer_Level', 'Treatment_Combo']).copy()
            model = ols(f'{indicator} ~ C(Fertilizer_Level)', data=df_fit).fit()
            anova_table = anova_lm(model, typ=2)
            anova_table.to_csv(f'anova_{indicator}_{irr}_within.csv', encoding='utf-8-sig')

            tukey = pairwise_tukeyhsd(df_fit[indicator], df_fit['Treatment_Combo'], alpha=alpha)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            tukey_df.to_csv(f'tukey_{indicator}_{irr}_within.csv', index=False, encoding='utf-8-sig')

            result['anova'] = anova_table
            result['tukey'] = tukey
            result['method'] = 'ANOVA + Tukey HSD'

            from itertools import combinations
            current_letter = 0
            letters_dict[group_labels[0]] = ascii_lowercase[current_letter]
            for g1, g2 in combinations(group_labels, 2):
                row = tukey_df[(tukey_df['group1'] == g1) & (tukey_df['group2'] == g2)]
                if not row.empty and not row['reject'].values[0]:
                    letters_dict[g2] = letters_dict[g1]
                else:
                    if g2 not in letters_dict:
                        current_letter += 1
                        letters_dict[g2] = ascii_lowercase[current_letter]
        else:
            kw_stat, kw_p = stats.kruskal(*groups)
            kw_df = pd.DataFrame([{'Kruskal_Stat': kw_stat, 'p_value': kw_p, 'Significant': kw_p < alpha}])
            kw_df.to_csv(f'kruskal_{indicator}_{irr}_within.csv', index=False, encoding='utf-8-sig')

            if kw_p < alpha:
                dunn = sp.posthoc_dunn(groups, p_adjust='bonferroni')
                dunn.index = group_labels
                dunn.columns = group_labels
                dunn.to_csv(f'dunn_{indicator}_{irr}_within.csv', encoding='utf-8-sig')

                result['dunn'] = dunn
                result['method'] = 'Kruskal-Wallis + Dunn (Bonferroni)'

                mean_dict = {label: float(np.mean(g)) for label, g in zip(group_labels, groups)}
                sorted_labels = sorted(group_labels, key=lambda x: (-mean_dict.get(x, float('nan')), x))
                letter_groups = []
                letters_dict = {}

                for label in sorted_labels:
                    assigned = False
                    for let_set in letter_groups:
                        if all(dunn.loc[label, g] >= alpha for g in let_set):
                            let_set.add(label)
                            letters_dict[label] = ascii_lowercase[len(letter_groups) - 1]
                            assigned = True
                            break
                    if not assigned:
                        new_set = {label}
                        letter_groups.append(new_set)
                        letters_dict[label] = ascii_lowercase[len(letter_groups) - 1]

        order = sorted(group_labels, key=lambda x: -_fert_pct(x))

        summary_data = []
        for label, g in zip(group_labels, groups):
            summary_data.append({
                'Treatment_Combo': label,
                'Irrigation_Level': str(irr),
                'Mean': float(np.mean(g)),
                'SEM': float(stats.sem(g)),
                'Letter': letters_dict.get(label, '?')
            })
        pd.DataFrame(summary_data).to_csv(f'summary_{indicator}_{irr}_within.csv', index=False, encoding='utf-8-sig')

        details[str(irr)] = {
            'method': result.get('method', 'NA'),
            'letters_dict': letters_dict,
            'order': order,
        }

    return details


def perform_significance_analysis(df, indicator, alpha=0.05):
    if indicator not in df.columns:
        print(f"列 '{indicator}' 不存在，跳过")
        return None, None, None, None

    if indicator == 'NUE':
        df = df[df['Treatment_Combo'] != 'W1_0%N'].copy()
    
    print(f"\n分析指标：{indicator}  (非空值：{df[indicator].notna().sum()})")
    
    groups = []
    group_labels = []
    for name, group in df.groupby('Treatment_Combo'):
        values = group[indicator].dropna()
        if len(values) >= 3:
            groups.append(values)
            group_labels.append(name)
    
    if len(groups) < 2:
        print(f"有效组合不足 {len(groups)}，无法分析")
        return None, None, None, None
    
    # ── 正态性检验 ──
    normality_results = []
    for label, g in zip(group_labels, groups):
        stat, p = stats.shapiro(g)
        normality_results.append({'Group': label, 'Shapiro_Stat': stat, 'p_value': p, 'Normal': p > alpha})
    
    normality_df = pd.DataFrame(normality_results)
    normality_df.to_csv(f'normality_{indicator}.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 保存正态性检验：normality_{indicator}.csv")
    
    normal_pass = all(normality_df['Normal'])
    print(f"正态性检验：{'通过' if normal_pass else '未完全通过'} (最小组 p = {normality_df['p_value'].min():.4f})")
    
    # ── 方差齐性 ──
    levene_stat, levene_p = stats.levene(*groups)
    levene_df = pd.DataFrame([{'Levene_Stat': levene_stat, 'p_value': levene_p, 'Homoscedastic': levene_p > alpha}])
    levene_df.to_csv(f'levene_{indicator}.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 保存方差齐性：levene_{indicator}.csv")
    print(f"Levene 方差齐性：p = {levene_p:.4f} ({'齐' if levene_p > alpha else '不齐'})")
    
    result = {}
    letters_dict = {}  # 初始化字母分组
    
    if normal_pass and levene_p > alpha:
        model = ols(f'{indicator} ~ C(Irrigation_Level) * C(Fertilizer_Level)', data=df).fit()
        anova_table = anova_lm(model, typ=2)
        print(f"\n双因素 ANOVA 结果：\n{anova_table}")
        
        resid_stat, resid_p = stats.shapiro(model.resid)
        resid_df = pd.DataFrame([{'Residual_Shapiro_Stat': resid_stat, 'p_value': resid_p, 'Normal': resid_p > alpha}])
        resid_df.to_csv(f'residual_normality_{indicator}.csv', index=False, encoding='utf-8-sig')
        print(f"✅ 保存残差正态性：residual_normality_{indicator}.csv")
        print(f"残差正态性 p = {resid_p:.4f}")
        
        tukey = pairwise_tukeyhsd(df[indicator], df['Treatment_Combo'], alpha=alpha)
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tukey_df.to_csv(f'tukey_{indicator}.csv', index=False, encoding='utf-8-sig')
        print(f"✅ 保存 Tukey HSD：tukey_{indicator}.csv")
        print(f"Tukey HSD 结果（部分）：\n{tukey.summary()}")
        
        anova_table.to_csv(f'anova_{indicator}.csv', encoding='utf-8-sig')
        print(f"✅ 保存 ANOVA 表：anova_{indicator}.csv")
        
        result['anova'] = anova_table
        result['tukey'] = tukey
        result['method'] = 'ANOVA + Tukey HSD'
        
        # 对于 ANOVA，使用 Tukey 生成字母（简化：使用 Tukey 的 reject 状态合并组）
        # 这里我们模拟字母分配（基于 Tukey 的分组）
        from itertools import combinations
        current_letter = 0
        letters_dict[group_labels[0]] = ascii_lowercase[current_letter]
        for g1, g2 in combinations(group_labels, 2):
            row = tukey_df[(tukey_df['group1'] == g1) & (tukey_df['group2'] == g2)]
            if not row.empty and not row['reject'].values[0]:
                letters_dict[g2] = letters_dict[g1]
            else:
                if g2 not in letters_dict:
                    current_letter += 1
                    letters_dict[g2] = ascii_lowercase[current_letter]
        
    else:
        kw_stat, kw_p = stats.kruskal(*groups)
        kw_df = pd.DataFrame([{'Kruskal_Stat': kw_stat, 'p_value': kw_p, 'Significant': kw_p < alpha}])
        kw_df.to_csv(f'kruskal_{indicator}.csv', index=False, encoding='utf-8-sig')
        print(f"✅ 保存 Kruskal-Wallis：kruskal_{indicator}.csv")
        print(f"Kruskal-Wallis 整体检验：H = {kw_stat:.2f}, p = {kw_p:.4f}")
        
        if kw_p < alpha:
            dunn = sp.posthoc_dunn(groups, p_adjust='bonferroni')
            dunn.index = group_labels
            dunn.columns = group_labels
            dunn.to_csv(f'dunn_{indicator}.csv', encoding='utf-8-sig')
            print(f"✅ 保存 Dunn 矩阵：dunn_{indicator}.csv")
            print("Dunn 事后检验（Bonferroni 校正）：\n", dunn.round(4))
            
            result['dunn'] = dunn
            result['method'] = 'Kruskal-Wallis + Dunn (Bonferroni)'
            # ── 非参数两因素交互补充（Rank Transform + ANOVA on ranks） ──
            print(f"\n=== {indicator} 非参数两因素交互补充检验（Rank Transform） ===")
            df_rank = df.copy()
            df_rank[f'rank_{indicator}'] = df_rank[indicator].rank(method='average')  # 全局排序

            model_rank = ols(f'rank_{indicator} ~ C(Irrigation_Level) * C(Fertilizer_Level)', data=df_rank).fit()
            anova_rank = anova_lm(model_rank, typ=2)

            # 保存为论文表格
            anova_rank.to_csv(f'rank_anova_interaction_{indicator}.csv', encoding='utf-8-sig')
            print(f"✅ 保存Rank ANOVA交互表：rank_anova_interaction_{indicator}.csv")
            print(anova_rank.round(4))

            # 提取交互p值
            inter_p = anova_rank.loc['C(Irrigation_Level):C(Fertilizer_Level)', 'PR(>F)']
            print(f"交互效应 p = {inter_p:.4f} （{'显著' if inter_p < 0.05 else '不显著'}）")
            # ── 生成字母分组 ──
           # 1. 计算每个组的均值（用于排序：最高均值优先给 'a'）
        mean_dict = {label: np.mean(group) for label, group in zip(group_labels, groups)}
        
        # 2. 按均值从高到低排序
        sorted_labels = sorted(group_labels, key=lambda x: -mean_dict[x])
        
        # 3. 每个字母对应一个“组集合”（保证组内所有成员互不显著差异）
        letter_groups = []          # list of sets: 每个 set 是一个字母的成员
        letters_dict = {}           # 最终结果：label -> 'a' / 'b' / ...

        for label in sorted_labels:
            assigned = False
            # 检查能否加入已有的某个字母组（必须与该组**所有成员** p >= alpha）
            for let_set in letter_groups:
                if all(dunn.loc[label, g] >= alpha for g in let_set):
                    let_set.add(label)
                    letters_dict[label] = ascii_lowercase[len(letter_groups) - 1]  # 当前字母
                    assigned = True
                    break
            
            # 不能加入任何已有组 → 开新字母
            if not assigned:
                new_set = {label}
                letter_groups.append(new_set)
                letters_dict[label] = ascii_lowercase[len(letter_groups) - 1]

        # ── 可选：打印检查（调试用，论文不用保留） ──
        print(f"✅ Compact Letter Display 完成（{indicator}）：")
        for lab in sorted(letters_dict, key=lambda x: (-mean_dict[x], x)):
            print(f"  {lab:12} → {letters_dict[lab]}  (均值 {mean_dict[lab]:.2f})")
    
    return result, groups, group_labels, letters_dict


# ────────────────────────────────────────────────
# 3. 绘图函数（保持，但添加总结表保存）
# ────────────────────────────────────────────────
# ────────────────────────────────────────────────
# 科研级柱状图函数（Mean ± SEM + 显著性字母 + 完整图例）
# ────────────────────────────────────────────────
# ────────────────────────────────────────────────
# 科研级柱状图函数（优化版：字母嵌入柱顶、图例右上角、单位标注、英文横轴）
# ────────────────────────────────────────────────
# ────────────────────────────────────────────────
# 最终优化版柱状图（字母纯文本嵌入柱顶、图例外置、单位负号正确、图例文字精确）
# ────────────────────────────────────────────────
# ────────────────────────────────────────────────
# 最终优化版柱状图（字母纯文本嵌入柱顶、图例外置、单位负号正确、图例文字精确）
# ────────────────────────────────────────────────
# ────────────────────────────────────────────────
# 最终优化版柱状图（单位直接上标⁻¹、字母纯文本、图例外置不遮挡）
# ────────────────────────────────────────────────
def plot_boxplot_with_letters(df, indicator, result, groups, group_labels, letters_dict, alpha=0.05):
    import os
    import datetime
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    OUTPUT_DIR = r"D:\learning\modle_Tan\数据分析"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 构建 summary_df
    summary_data = []
    for label, group in zip(group_labels, groups):
        summary_data.append({
            'Treatment_Combo': label,
            'Irrigation_Level': label.split('_')[0],
            'Mean': np.mean(group),
            'SEM': stats.sem(group),
            'Letter': letters_dict.get(label, '?')
        })
    
    summary_df = pd.DataFrame(summary_data)
    if indicator == 'NUE':
        summary_df = summary_df[summary_df['Treatment_Combo'] != 'W1_0%N'].copy()
    
    # 排序 + 英文标签
    def sort_key(x):
        irr, fert = x.split('_')
        pct = int(fert[:-2])
        irr_order = {'W1':0, 'W2':1, 'W3':2}
        return (irr_order.get(irr, 99), -pct)
    
    summary_df = summary_df.sort_values(by='Treatment_Combo', key=lambda x: x.map(sort_key))
    summary_df['English_Label'] = summary_df['Treatment_Combo'].apply(lambda x: x.replace('_', '-'))
    
    # 单位（直接使用 Unicode 上标⁻¹）
    unit_dict = {
        'WAGT': 'g m⁻²',
        'WRR14': 'kg ha⁻¹',
        'WUE': 'kg ha⁻¹ mm⁻¹',
        'IWUE': 'kg ha⁻¹ mm⁻¹',
        'ET_WUE': 'kg ha⁻¹ mm⁻¹',
        'NUE': 'kg kg⁻¹',
        'SPFERT': 'fraction'
    }
    y_unit = unit_dict.get(indicator, '')
    y_label = f"{indicator} ({y_unit})" if y_unit else indicator

    # 设置全局字体为 DejaVu Sans（matplotlib 默认支持所有上标）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 推荐，不要Arial
    plt.rcParams['axes.unicode_minus'] = True  # 负号正常显示

    plt.figure(figsize=(15, 7.5))
    palette = {'W1': '#1f77b4', 'W2': '#ff7f0e', 'W3': '#2ca02c'}
    
    ax = sns.barplot(
        x='English_Label', y='Mean', hue='Irrigation_Level',
        data=summary_df, order=summary_df['English_Label'],
        palette=palette, edgecolor='black', linewidth=1.2, errorbar=None
    )
    
    # SEM 误差条
    for (i, row), patch in zip(enumerate(summary_df.itertuples()), ax.patches):
        x = patch.get_x() + patch.get_width()/2
        y = patch.get_height()
        sem = row.SEM
        ax.errorbar(x, y, yerr=sem, fmt='none',
                    ecolor='black', elinewidth=2, capsize=6, capthick=2)
    
    # 字母：纯文本嵌入柱顶
    for i, row in enumerate(summary_df.itertuples()):
        y_pos = row.Mean + row.SEM + 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(i, y_pos, row.Letter,
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
    
    # 图例
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ['W1 (Rainfed)', 'W2 (WCFC%)', 'W3 (Full Irrigated)'],
              title='Irrigation Regime', title_fontsize=12, fontsize=11,
              loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, ncol=1)
    
    # 美化
    # 保持字体为DejaVu Sans
    plt.title(f'Mean {indicator} under Different Irrigation × Nitrogen Treatments\n'
              f'({result.get("method", "Analysis Method")})',
              fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Irrigation × N Fertilization Treatment', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.25)
    
    # 显著性说明
    ax.text(0.02, 0.03, 'Same letter = no significant difference (P ≥ 0.05)',
            transform=ax.transAxes, fontsize=10, color='dimgray',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_name = f"{indicator}_barplot_mean_sem_letter_{timestamp}"
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}.pdf"), bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图表已保存（单位直接上标⁻¹、字母纯文本、图例外置）：")
    print(f"   {base_name}.png")
    print(f"   {base_name}.pdf")


def plot_violin_with_letters_within(df, indicator, irr_level, within_detail, alpha=0.05):
    import os
    import datetime
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    OUTPUT_DIR = r"D:\learning\modle_Tan\数据分析"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not within_detail or 'order' not in within_detail:
        return

    dfi = df[df['Irrigation_Level'].astype(str) == str(irr_level)].copy()
    if indicator == 'NUE':
        dfi = dfi[dfi['Treatment_Combo'] != 'W1_0%N'].copy()

    order = within_detail.get('order', [])
    letters_dict = within_detail.get('letters_dict', {})

    dfi = dfi.dropna(subset=['Treatment_Combo', indicator]).copy()

    # 单位
    unit_dict = {
        'WAGT': 'g m⁻²',
        'WRR14': 'kg ha⁻¹',
        'WUE': 'kg ha⁻¹ mm⁻¹',
        'IWUE': 'kg ha⁻¹ mm⁻¹',
        'ET_WUE': 'kg ha⁻¹ mm⁻¹',
        'NUE': 'kg kg⁻¹',
        'SPFERT': 'fraction'
    }
    y_unit = unit_dict.get(indicator, '')
    y_label = f"{indicator} ({y_unit})" if y_unit else indicator

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = True

    fig, ax = plt.subplots(figsize=(14.5, 7.2), dpi=300)

    palette = {'W1': '#1f77b4', 'W2': '#ff7f0e', 'W3': '#2ca02c'}
    color = palette.get(str(irr_level), '#4C78A8')

    sns.violinplot(
        data=dfi,
        x='Treatment_Combo',
        y=indicator,
        order=order,
        color=color,
        inner=None,
        cut=0,
        linewidth=0.9,
        ax=ax
    )
    sns.boxplot(
        data=dfi,
        x='Treatment_Combo',
        y=indicator,
        order=order,
        width=0.22,
        showcaps=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#222", "linewidth": 0.9},
        medianprops={"color": "#222", "linewidth": 1.2},
        whiskerprops={"color": "#222", "linewidth": 0.9},
        capprops={"color": "#222", "linewidth": 0.9},
        ax=ax,
        zorder=3,
    )
    sns.stripplot(
        data=dfi,
        x='Treatment_Combo',
        y=indicator,
        order=order,
        color='black',
        size=1.6,
        alpha=0.18,
        jitter=0.18,
        ax=ax,
        zorder=2,
    )

    y = dfi[indicator]
    y_min = float(y.min())
    y_max = float(y.max())
    y_rng = (y_max - y_min) if (y_max > y_min) else 1.0

    cat_max = (
        dfi.groupby('Treatment_Combo', observed=False)[indicator]
        .max()
        .to_dict()
    )

    top_needed = y_max
    for c in order:
        lt = letters_dict.get(str(c), '')
        if not lt:
            continue
        base = float(cat_max.get(str(c), y_max))
        extra = 0.045 * y_rng + 0.012 * y_rng * max(0, len(str(lt)) - 1)
        top_needed = max(top_needed, base + extra)
    ax.set_ylim(y_min - 0.02 * y_rng, top_needed + 0.05 * y_rng)

    for i, c in enumerate(order):
        lt = letters_dict.get(str(c), '')
        if not lt:
            continue
        base = float(cat_max.get(str(c), y_max))
        extra = 0.045 * y_rng + 0.012 * y_rng * max(0, len(str(lt)) - 1)
        ax.text(i, base + extra, str(lt), ha='center', va='bottom', fontsize=11, fontweight='bold', color='#111')

    ax.set_title(
        f"{indicator} within {irr_level} (within-irrigation post-hoc)\n({within_detail.get('method', 'Analysis Method')})",
        pad=14,
        fontweight='bold'
    )
    ax.set_xlabel('N Fertilization Treatment')
    ax.set_ylabel(y_label)
    ax.grid(axis='y', linestyle='--', alpha=0.22)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(c).replace('_', '-') for c in order], rotation=45, ha='right')

    ax.text(
        0.01,
        0.02,
        f"Same letter = no significant difference (P ≥ 0.05)\nWithin {irr_level} post-hoc: {within_detail.get('method', 'NA')}",
        transform=ax.transAxes,
        fontsize=9,
        color='dimgray',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
    )

    fig.tight_layout()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    base_name = f"{indicator}_violin_within_{irr_level}_{timestamp}"
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base_name}.png"), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base_name}.pdf"), bbox_inches='tight')
    plt.close(fig)

    print(f"✅ 组内小提琴图已保存：{base_name}.png / {base_name}.pdf")


# ────────────────────────────────────────────────
# 主程序
# ────────────────────────────────────────────────
if __name__ == "__main__":
    file_path = r'D:\learning\modle_Tan\IR×FER.xlsx'
    raw_df = load_data(file_path)
    
    key_indicators = ['WAGT', 'WRR14', 'WUE', 'IWUE', 'ET_WUE', 'NUE', 'SPFERT']
    
    for indicator in key_indicators:
        result, groups, labels, letters_dict = perform_significance_analysis(raw_df, indicator)
        if result:
            plot_boxplot_with_letters(raw_df, indicator, result, groups, labels, letters_dict)

        within_details = perform_significance_analysis_by_irr(raw_df, indicator)
        for irr_level, detail in within_details.items():
            plot_violin_with_letters_within(raw_df, indicator, irr_level, detail)
    
    print("\n所有分析完成。")
    print("箱线图已保存为 boxplot_指标名.png (带显著性字母)")
    print("所有显著性分析结果已保存为 csv 文件（正态性、方差齐性、ANOVA/Dunn 等），可直接用于论文表格或Origin绘图。")
    print("每个指标的总结表 summary_指标名.csv 最适合Origin：导入后可画柱状图/箱线图带误差条和字母。")
