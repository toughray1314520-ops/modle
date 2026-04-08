import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
EXCEL_PATH = HERE.parent / "IR×FER.xlsx"
if not EXCEL_PATH.exists():
    EXCEL_PATH = HERE.parent / "IR×FER1.xlsx" if (HERE.parent / "IR×FER1.xlsx").exists() else EXCEL_PATH
if not EXCEL_PATH.exists():
    EXCEL_PATH = HERE / "水稻水肥显著性分析汇总_Origin用.xlsx" if (HERE / "水稻水肥显著性分析汇总_Origin用.xlsx").exists() else EXCEL_PATH
CSV_FALLBACK = HERE / "IR×FER.csv"

RAW_SHEET_INDEX_FALLBACK = 7

METRICS = [
    ("WRR14", "WRR14"),
    ("WUE", "WUE"),
    ("NUE", "NUE"),
    ("ET_WUE", "ET_WUE"),
]

BEST_RULE = {
    "WRR14": "max",
    "WUE": "max",
    "NUE": "max",
    "ET_WUE": "min",
}

UNIT_DICT = {
    "WAGT": "g m⁻²",
    "WRR14": "kg ha⁻¹",
    "WUE": "kg ha⁻¹ mm⁻¹",
    "IWUE": "kg ha⁻¹ mm⁻¹",
    "ET_WUE": "kg ha⁻¹ mm⁻¹",
    "NUE": "kg kg⁻¹",
    "SPFERT": "fraction",
}

IRR_PALETTE = {"W1": "#1f77b4", "W2": "#ff7f0e", "W3": "#2ca02c"}


def _parse_combo_sort_key(combo: str) -> tuple[int, int]:
    m = re.match(r"^W(\d+)_([0-9]+)\s*%N$", str(combo).strip())
    if not m:
        return (10**9, -10**9)
    w = int(m.group(1))
    n_pct = int(m.group(2))
    return (w, -n_pct)


def _find_raw_sheet(xls: pd.ExcelFile) -> str | int:
    required = {"Treatment_Combo", "WRR14", "WUE", "NUE"}
    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sh, nrows=5)
        except Exception:
            continue
        cols = set(map(str, df.columns))
        if required.issubset(cols):
            return sh
    return RAW_SHEET_INDEX_FALLBACK


def _load_irfer_workbook(xls: pd.ExcelFile) -> pd.DataFrame:
    out = []
    for sh in xls.sheet_names:
        sh_str = str(sh).strip()
        if sh_str.upper() == "SHEET19" or sh_str == "Sheet19":
            fert_abs = 0.0
            fert_pct = 0
            combo_name = "W1_0N"
            irr_level = "W1"
        else:
            parts = str(sh).split("+")
            fert_raw = parts[0].strip()
            irr_part = parts[1].strip() if len(parts) > 1 else ""
            if irr_part == "NO IR":
                irr_level = "W1"
            elif irr_part == "WCFC%":
                irr_level = "W2"
            else:
                irr_level = "W3"

            if fert_raw == "0N":
                fert_abs = 0.0
                fert_pct = 0
                combo_name = "W1_0N"
            elif "%" in fert_raw:
                match = re.search(r'(-?\d+)%', fert_raw)
                reduction_pct = int(match.group(1)) if match else 0
                fert_abs = 225.0 * (1 + reduction_pct / 100.0)
                fert_pct = int(round(fert_abs / 225.0 * 100.0))
                combo_name = f"{irr_level}_{fert_pct}%N"
            else:
                fert_part = fert_raw.replace("N", "").strip()
                fert_abs = float(fert_part)
                fert_pct = int(round(fert_abs / 225.0 * 100.0))
                combo_name = f"{irr_level}_{fert_pct}%N"

        df = pd.read_excel(xls, sheet_name=sh)
        df.columns = df.columns.map(lambda x: str(x).strip())
        if "YEAR" in df.columns:
            df = df[df["YEAR"].astype(str).str.strip().str.lower() != "mean"].copy()
            df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
            df = df[df["YEAR"].notna()].copy()

        df["Treatment_Combo"] = combo_name
        for col in ["WRR14", "IRCUM", "RAINCUM", "TRCCUM", "EVSWCUM", "FERCUM"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "WRR14" in df.columns and "IRCUM" in df.columns and "RAINCUM" in df.columns:
            denom = df["IRCUM"].fillna(0) + df["RAINCUM"].fillna(0)
            df["WUE"] = np.where(denom > 0, df["WRR14"] / denom, np.nan)
        if "WRR14" in df.columns and "FERCUM" in df.columns:
            df["NUE"] = np.where(df["FERCUM"].fillna(0) > 0, df["WRR14"] / df["FERCUM"], np.nan)
        if "WRR14" in df.columns and "TRCCUM" in df.columns and "EVSWCUM" in df.columns:
            et_denom = df["TRCCUM"].fillna(0) + df["EVSWCUM"].fillna(0)
            df["ET_WUE"] = np.where(et_denom > 0, df["WRR14"] / et_denom, np.nan)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _y_label(metric_col: str) -> str:
    unit = UNIT_DICT.get(metric_col, "")
    return f"{metric_col} ({unit})" if unit else metric_col


def _english_label(combo: str) -> str:
    return str(combo).replace("_", "-")


def _excel_style_letters(n: int) -> list[str]:
    letters = []
    k = 0
    while len(letters) < n:
        s = ""
        x = k
        while True:
            s = chr(ord("a") + (x % 26)) + s
            x = x // 26 - 1
            if x < 0:
                break
        letters.append(s)
        k += 1
    return letters


def _compact_letter_display(*, pvals: pd.DataFrame, means: pd.Series, alpha: float) -> dict[str, str]:
    groups = [str(g) for g in pvals.index.tolist()]
    groups = sorted(groups, key=lambda g: (-float(means.get(g, float("nan"))), g))
    letter_sets: list[set[str]] = []
    group_letters: dict[str, set[int]] = {g: set() for g in groups}
    def compatible(g: str, s: set[str]) -> bool:
        return all(float(pvals.loc[g, h]) >= alpha for h in s)
    for g in groups:
        added_any = False
        for li, s in enumerate(letter_sets):
            if compatible(g, s):
                s.add(g)
                group_letters[g].add(li)
                added_any = True
        if not added_any:
            letter_sets.append({g})
            group_letters[g].add(len(letter_sets) - 1)
    letter_names = _excel_style_letters(len(letter_sets))
    return {g: "".join(letter_names[i] for i in sorted(group_letters[g])) for g in groups}


def _pairwise_pvals(*, df_raw: pd.DataFrame, metric_col: str, alpha: float) -> tuple[pd.DataFrame, str]:
    g = df_raw.dropna(subset=[metric_col, "Treatment_Combo"]).copy()
    g["Treatment_Combo"] = g["Treatment_Combo"].astype(str)
    labels = sorted(g["Treatment_Combo"].unique().tolist())
    series_by = {lab: g.loc[g["Treatment_Combo"] == lab, metric_col].dropna() for lab in labels}
    labels = [lab for lab in labels if len(series_by[lab]) >= 3]
    shapiro_ps = []
    for lab in labels:
        try:
            shapiro_ps.append(float(stats.shapiro(series_by[lab]).pvalue))
        except Exception:
            shapiro_ps.append(0.0)
    normal_pass = all(p > alpha for p in shapiro_ps)
    try:
        levene_p = float(stats.levene(*[series_by[lab] for lab in labels]).pvalue)
    except Exception:
        levene_p = 0.0
    pmat = pd.DataFrame(1.0, index=labels, columns=labels, dtype=float)
    if normal_pass and levene_p >= alpha:
        tukey = pairwise_tukeyhsd(g[metric_col], g["Treatment_Combo"], alpha=alpha)
        tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        for _, row in tukey_df.iterrows():
            a, b, p = str(row["group1"]), str(row["group2"]), float(row["p-adj"])
            if a in pmat.index and b in pmat.columns:
                pmat.loc[a, b] = p
                pmat.loc[b, a] = p
        return pmat, "Tukey HSD"
    try:
        import scikit_posthocs as sp
        groups = [series_by[lab].values for lab in labels]
        d = sp.posthoc_dunn(groups, p_adjust="bonferroni")
        d.index = labels
        d.columns = labels
        for a in labels:
            for b in labels:
                pmat.loc[a, b] = float(d.loc[a, b])
        return pmat, "Dunn (Bonferroni)"
    except Exception:
        pairs: list[tuple[str, str]] = []
        raw_ps: list[float] = []
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                try:
                    p = float(stats.mannwhitneyu(series_by[a], series_by[b], alternative="two-sided").pvalue)
                except Exception:
                    p = 1.0
                pairs.append((a, b))
                raw_ps.append(p)
        if raw_ps:
            _, adj_ps, _, _ = multipletests(raw_ps, alpha=alpha, method="holm")
            for (a, b), p in zip(pairs, adj_ps):
                pmat.loc[a, b] = float(p)
                pmat.loc[b, a] = float(p)
        return pmat, "Mann-Whitney U (Holm)"


def _best_combo(df_raw: pd.DataFrame, metric_col: str) -> str:
    means = df_raw.groupby("Treatment_Combo", dropna=False, observed=False)[metric_col].mean(numeric_only=True)
    rule = BEST_RULE.get(metric_col, "max")
    return str(means.idxmin() if rule == "min" else means.idxmax())


def plot_one_metric(
    *,
    df_raw: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    combo_order: list[str], # 不再使用，只是兼容原main参数
    letters: dict[str, str],
    method: str,
    out_png: Path,
    out_pdf: Path,
):
    # 1. 数据过滤和实际顺序。palette/order永远和df_plot一一对应！
    if metric_col == "NUE":
        df_plot = df_raw[df_raw["Treatment_Combo"] != "W1_0%N"].copy()
    else:
        df_plot = df_raw.copy()
    plot_combo_order = sorted(df_plot["Treatment_Combo"].unique().tolist(), key=_parse_combo_sort_key)
    irr_of = {c: str(c).split("_")[0] if "_" in str(c) else "" for c in combo_order}
    palette = {c: IRR_PALETTE.get(irr_of[c], "#4C78A8") for c in combo_order}
    if "W1_0N" in plot_combo_order:
        palette["W1_0N"] = "#7f7f7f"

    best = _best_combo(df_plot, metric_col)

    fig, ax = plt.subplots(figsize=(15.5, 7.2), dpi=300)
    sns.violinplot(
        data=df_plot,
        x="Treatment_Combo",
        y=metric_col,
        order=plot_combo_order,
        hue="Treatment_Combo",
        palette=palette,
        legend=False,
        inner=None,
        cut=0,
        linewidth=0.9,
        ax=ax,
    )
    sns.boxplot(
        data=df_plot,
        x="Treatment_Combo",
        y=metric_col,
        order=plot_combo_order,
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
        data=df_plot,
        x="Treatment_Combo",
        y=metric_col,
        order=plot_combo_order,
        color="black",
        size=1.6,
        alpha=0.18,
        jitter=0.18,
        ax=ax,
        zorder=2,
    )
    ax.set_title(f"{metric_label} under Different Irrigation × Nitrogen Treatments", pad=14, fontweight='bold', fontsize=14)
    ax.set_xlabel("Irrigation × N Fertilization Treatment", fontweight='bold', fontsize=12)
    ax.set_ylabel(_y_label(metric_col), fontweight='bold', fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    sns.despine(ax=ax, top=True, right=True)
    ax.text(
        0.01,
        0.02,
        f"Same letter = no significant difference (P ≥ 0.05)\nPost-hoc: {method}",
        transform=ax.transAxes,
        fontsize=9,
        color="dimgray",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
    )

    if letters:
        y = df_plot[metric_col]
        y_min = float(y.min())
        y_max = float(y.max())
        y_rng = (y_max - y_min) if (y_max > y_min) else 1.0
        cat_max = (
            df_plot.dropna(subset=[metric_col])
            .groupby("Treatment_Combo", observed=False)[metric_col]
            .max()
            .to_dict()
        )
        top_needed = y_max
        for c in plot_combo_order:
            lt = letters.get(str(c), "")
            if not lt or str(lt).lower() == "nan":
                continue
            base = float(cat_max.get(str(c), y_max))
            extra = 0.045 * y_rng + 0.012 * y_rng * max(0, len(str(lt)) - 1)
            top_needed = max(top_needed, base + extra)
        ax.set_ylim(y_min - 0.02 * y_rng, top_needed + 0.05 * y_rng)
        for i, c in enumerate(plot_combo_order):
            lt = letters.get(str(c), "")
            if not lt or str(lt).lower() == "nan":
                continue
            base = float(cat_max.get(str(c), y_max))
            extra = 0.045 * y_rng + 0.012 * y_rng * max(0, len(str(lt)) - 1)
            ax.text(i, base + extra, str(lt), ha="center", va="bottom", fontsize=11, fontweight="bold", color="#111")
    ax.set_xticks(range(len(plot_combo_order)))
    ax.set_xticklabels([_english_label(c) for c in plot_combo_order], rotation=45, ha="right")
    for tick in ax.get_xticklabels():
        if tick.get_text() == _english_label(best):
            tick.set_color("#D62728")
            tick.set_fontweight("bold")
            break
    from matplotlib.patches import Patch
    legend_labels = {
        "W1": "W1 (Rainfed)",
        "W2": "W2 (WCFC%)",
        "W3": "W3 (Full Irrigated)",
    }
    handles = [
        Patch(
            facecolor=IRR_PALETTE[k],
            edgecolor="black",
            label=legend_labels.get(k, k),
        )
        for k in ["W1", "W2", "W3"]
        if k in IRR_PALETTE
    ]
    ax.legend(
        handles=handles,
        title="Irrigation Regime",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_png, bbox_inches="tight")
        print(f"保存PNG成功: {out_png}")
    except PermissionError:
        print(f"保存PNG时被占用: {out_png}")
    plt.close(fig)


def main():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel not found: {EXCEL_PATH}")

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = True

    if EXCEL_PATH.exists():
        xls = pd.ExcelFile(EXCEL_PATH)
        raw_sheet = _find_raw_sheet(xls)
        df_raw = pd.read_excel(xls, sheet_name=raw_sheet) if isinstance(raw_sheet, str) else pd.read_excel(xls, sheet_name=raw_sheet)
        if "Treatment_Combo" not in df_raw.columns:
            df_raw = _load_irfer_workbook(xls)
            if df_raw.empty:
                raise KeyError("无法从 IR×FER.xlsx 读取数据（未找到可用工作表）")
            raw_sheet = "IR×FER.xlsx"
    else:
        df_raw = pd.read_csv(CSV_FALLBACK)
        if "Treatment_Combo" not in df_raw.columns:
            raise KeyError(f"CSV 缺少 Treatment_Combo 列：{CSV_FALLBACK}")

    # Basic cleanup
    df_raw = df_raw.dropna(subset=["Treatment_Combo"])
    df_raw["Treatment_Combo"] = df_raw["Treatment_Combo"].astype(str)
    combo_order = sorted(df_raw["Treatment_Combo"].unique().tolist(), key=_parse_combo_sort_key)
    df_raw["Treatment_Combo"] = pd.Categorical(df_raw["Treatment_Combo"], categories=combo_order, ordered=True)

    out_dir = HERE
    for metric_col, metric_label in METRICS:
        if metric_col not in df_raw.columns:
            raise KeyError(
                f"Raw sheet '{raw_sheet}' is missing column '{metric_col}'. "
                f"Available: {list(df_raw.columns)}"
            )
        if metric_col == "NUE":
            df_analysis = df_raw[df_raw["Treatment_Combo"] != "W1_0%N"].copy()
            df_plot = df_analysis
            plot_order = [c for c in combo_order if c != "W1_0%N"]
        else:
            df_analysis = df_raw
            df_plot = df_raw
            plot_order = combo_order
        pmat, method = _pairwise_pvals(df_raw=df_analysis, metric_col=metric_col, alpha=0.05)
        means = (
            df_analysis.dropna(subset=[metric_col, "Treatment_Combo"])
            .groupby("Treatment_Combo", observed=False)[metric_col]
            .mean()
        )
        letters = _compact_letter_display(pvals=pmat, means=means, alpha=0.05)
        plot_one_metric(
            df_raw=df_plot,
            metric_col=metric_col,
            metric_label=metric_label,
            combo_order=plot_order,
            letters=letters,
            method=method,
            out_png=out_dir / f"{metric_label}_violin_box_best.png",
            out_pdf=out_dir / f"{metric_label}_violin_box_best.pdf",
        )
    # 导出结果表（按照全量顺序即可）
    summary = (
        df_raw.groupby("Treatment_Combo", observed=False)[["WRR14", "WUE", "NUE", "ET_WUE"]]
        .agg(["mean", "std", "count"])
        .sort_index()
    )
    for col in ["WRR14", "WUE", "NUE", "ET_WUE"]:
        summary[(col, "sem")] = summary[(col, "std")] / (summary[(col, "count")] ** 0.5)
    summary.to_csv(out_dir / "violin_boxplot_summary_mean_sem.csv", encoding="utf-8-sig")

if __name__ == "__main__":
    main()
