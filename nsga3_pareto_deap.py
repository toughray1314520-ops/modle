import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from sklearn.ensemble import RandomForestRegressor

from deap import algorithms, base, creator, tools
import copy


plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
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


RANDOM_STATE = 42
FERT_STEP = 5
IRR_STEP = 5
K_CLIMATE_A = 100
CX_PROB = 0.8
MUT_PROB = 0.15
MUT_INDPB = 0.5
NGEN = 120
REF_P = 12
POP_SIZE = 92


def _try_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(os.path.dirname(here), "IR×FER66.xlsx"),
        os.path.join(here, "IR×FER66.xlsx"),
    ]
    return [p for p in cands if os.path.exists(p)]


def load_data(file_path):
    xl = pd.ExcelFile(file_path)
    sheets = xl.sheet_names
    all_df = []
    for sheet in sheets:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
            df.columns = df.columns.str.strip()
            df['SheetName'] = sheet

            if 'YEAR' in df.columns:
                df = df[df['YEAR'].astype(str).str.strip().str.lower() != 'mean'].copy()
                df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
                df = df[df['YEAR'].notna()].copy()

            for _col in ['WRR14', 'IRCUM', 'RAINCUM', 'FERCUM', 'TAVERC', 'PARCUM']:
                if _col in df.columns:
                    df[_col] = pd.to_numeric(df[_col], errors='coerce')

            fert_abs = np.nan
            if 'FERCUM' in df.columns and df['FERCUM'].notna().any():
                df['Fert_abs'] = df['FERCUM']
            else:
                parts = sheet.split('+')
                fert_part = parts[0].upper().replace('N', '').strip()
                try:
                    if fert_part.endswith('%'):
                        val = int(fert_part.replace('%', '').replace('+', ''))
                        fert_pct = 100 + val
                        fert_abs = 225 * fert_pct / 100.0
                    else:
                        fert_abs = float(fert_part)
                except Exception:
                    fert_abs = np.nan
                df['Fert_abs'] = fert_abs

            ir = df['IRCUM'].fillna(0) if 'IRCUM' in df.columns else pd.Series(0, index=df.index)
            ra = df['RAINCUM'].fillna(0) if 'RAINCUM' in df.columns else pd.Series(0, index=df.index)
            df['WUE'] = np.where((ir + ra) > 0, df['WRR14'] / (ir + ra), np.nan)

            if 'FERCUM' not in df.columns or df['FERCUM'].isnull().all():
                df['FERCUM'] = df['Fert_abs']
            df['NUE'] = np.where(df['FERCUM'].fillna(0) > 0, df['WRR14'] / df['FERCUM'], np.nan)

            all_df.append(df)
        except Exception:
            continue
    return pd.concat(all_df, ignore_index=True) if len(all_df) > 0 else pd.DataFrame()


def round_floor_step(x, step):
    return math.floor(x / step) * step


def round_ceil_step(x, step):
    return math.ceil(x / step) * step


def nondominated(pop):
    return tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]


def nondominated_points_2d(points):
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 2)
    order = np.argsort(-pts[:, 0], kind="mergesort")
    pts = pts[order]
    keep = []
    zmax = -np.inf
    for y, z in pts:
        if z > zmax:
            keep.append((y, z))
            zmax = z
    keep = np.asarray(keep, dtype=float)
    if keep.size == 0:
        return keep.reshape(0, 2)
    return keep[np.argsort(keep[:, 0], kind="mergesort")]


def union_area_rectangles_2d(points, ref=(0.0, 0.0)):
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    ry, rz = float(ref[0]), float(ref[1])
    m = np.isfinite(pts).all(axis=1) & (pts[:, 0] > ry) & (pts[:, 1] > rz)
    pts = pts[m]
    if pts.size == 0:
        return 0.0
    nd = nondominated_points_2d(pts)
    area = 0.0
    prev_y = ry
    for y, z in nd:
        area += (float(y) - prev_y) * float(z - rz)
        prev_y = float(y)
    return float(max(area, 0.0))


def hypervolume_3d(points, ref=(0.0, 0.0, 0.0)):
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    rx, ry, rz = float(ref[0]), float(ref[1]), float(ref[2])
    m = np.isfinite(pts).all(axis=1) & (pts[:, 0] > rx) & (pts[:, 1] > ry) & (pts[:, 2] > rz)
    pts = pts[m]
    if pts.size == 0:
        return 0.0
    xs = np.unique(np.concatenate(([rx], pts[:, 0])))
    xs.sort()
    hv = 0.0
    for k in range(1, xs.size):
        xk = float(xs[k])
        width = xk - float(xs[k - 1])
        if width <= 0:
            continue
        sub = pts[pts[:, 0] >= xk]
        area = union_area_rectangles_2d(sub[:, 1:3], ref=(ry, rz))
        hv += width * area
    return float(max(hv, 0.0))


def save_population_csv(path, gen, pop, fert_values, irr_values):
    fert_values = np.asarray(fert_values, dtype=float)
    irr_values = np.asarray(irr_values, dtype=float)
    rows = []
    for ind in pop:
        iF = int(ind[0]); iI = int(ind[1])
        fert = float(fert_values[iF])
        irr = float(irr_values[iI])
        y = float(ind.fitness.values[0])
        fs = float(ind.fitness.values[1])
        isv = float(ind.fitness.values[2])
        rows.append((gen, iF, iI, fert, irr, y, fs, isv))
    out = pd.DataFrame(rows, columns=["gen", "iF", "iI", "fert", "irr", "yield", "fert_saving", "irr_saving"])
    out.to_csv(path, index=False, encoding="utf-8-sig")


def init_population(pop_size, iF_max, iI_max, seed):
    rng = np.random.default_rng(int(seed))
    combos = [(iF, iI) for iF in range(iF_max + 1) for iI in range(iI_max + 1)]
    rng.shuffle(combos)
    combos = combos[: int(pop_size)]
    return [creator.Individual2([int(iF), int(iI)]) for iF, iI in combos]


def run_nsga3_with_history(eval_func, iF_max, iI_max, seed, ngen, cxpb, mutpb, out_dir, fert_values, irr_values, obj_max):
    random.seed(seed)
    np.random.seed(seed)
    if "FitnessMax3" not in creator.__dict__:
        creator.create("FitnessMax3", base.Fitness, weights=(1.0, 1.0, 1.0))
    if "Individual2" not in creator.__dict__:
        creator.create("Individual2", list, fitness=creator.FitnessMax3)

    ref_points = tools.uniform_reference_points(nobj=3, p=REF_P)
    n_combo = int((iF_max + 1) * (iI_max + 1))
    pop_size = int(POP_SIZE) if int(POP_SIZE) > 0 else len(ref_points)
    pop_size = int(min(pop_size, n_combo))

    toolbox = base.Toolbox()
    toolbox.register("attr_iF", random.randint, 0, iF_max)
    toolbox.register("attr_iI", random.randint, 0, iI_max)
    toolbox.register("individual", tools.initCycle, creator.Individual2, (toolbox.attr_iF, toolbox.attr_iI), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=[0, 0], up=[iF_max, iI_max], indpb=MUT_INDPB)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    toolbox.register("clone", copy.deepcopy)

    pop = init_population(pop_size, iF_max, iI_max, seed)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    os.makedirs(out_dir, exist_ok=True)
    hist_metrics = []
    max_y, max_fs, max_isv = float(obj_max[0]), float(obj_max[1]), float(obj_max[2])
    denom = max(max_y * max_fs * max_isv, 1e-12)

    for gen in range(ngen + 1):
        all_csv = os.path.join(out_dir, f"gen_{gen:04d}_all.csv")
        save_population_csv(all_csv, gen, pop, fert_values, irr_values)

        front = nondominated(pop)
        front_csv = os.path.join(out_dir, f"gen_{gen:04d}_front.csv")
        save_population_csv(front_csv, gen, front, fert_values, irr_values)

        front_pts = np.array([ind.fitness.values for ind in front], dtype=float)
        hv = hypervolume_3d(front_pts, ref=(0.0, 0.0, 0.0)) / denom

        ferts = np.array([fert_values[int(ind[0])] for ind in front], dtype=float)
        irrs = np.array([irr_values[int(ind[1])] for ind in front], dtype=float)
        ys = front_pts[:, 0] if front_pts.size else np.array([], dtype=float)
        cov_f = float(np.nanmax(ferts) - np.nanmin(ferts)) if ferts.size else float("nan")
        cov_i = float(np.nanmax(irrs) - np.nanmin(irrs)) if irrs.size else float("nan")
        cov_y = float(np.nanmax(ys) - np.nanmin(ys)) if ys.size else float("nan")

        hist_metrics.append((gen, float(hv), int(len(front)), cov_i, cov_f, cov_y))

        if gen == ngen:
            break

        offspring = algorithms.varOr(pop, toolbox, lambda_=pop_size, cxpb=cxpb, mutpb=mutpb)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + offspring, k=pop_size)

    metrics_df = pd.DataFrame(hist_metrics, columns=["gen", "hypervolume_norm", "n_nondominated", "cov_irr", "cov_fert", "cov_yield"])
    metrics_df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False, encoding="utf-8-sig")
    return nondominated(pop), metrics_df

def precompute_objectives(model_y, model_n, model_w, climate_arr, fert_values, irr_values):
    climate_arr = np.asarray(climate_arr, dtype=float)
    fert_values = np.asarray(fert_values, dtype=float)
    irr_values = np.asarray(irr_values, dtype=float)
    Fg, Ig = np.meshgrid(fert_values, irr_values, indexing="ij")
    f_flat = Fg.ravel()
    i_flat = Ig.ravel()
    n_combo = f_flat.size
    sum_y = np.zeros(n_combo, dtype=float)
    for k in range(climate_arr.shape[0]):
        rain, tav, par = float(climate_arr[k, 0]), float(climate_arr[k, 1]), float(climate_arr[k, 2])
        X = np.column_stack([
            f_flat,
            i_flat,
            np.full(n_combo, rain, dtype=float),
            np.full(n_combo, tav, dtype=float),
            np.full(n_combo, par, dtype=float),
        ])
        sum_y += model_y.predict(X)
    mean_y = (sum_y / float(climate_arr.shape[0])).reshape(Fg.shape)
    return mean_y


def run_nsga3(eval_func, iF_max, iI_max, seed):
    random.seed(seed)
    np.random.seed(seed)
    if "FitnessMax3" not in creator.__dict__:
        creator.create("FitnessMax3", base.Fitness, weights=(1.0, 1.0, 1.0))
    if "Individual2" not in creator.__dict__:
        creator.create("Individual2", list, fitness=creator.FitnessMax3)

    ref_points = tools.uniform_reference_points(nobj=3, p=REF_P)
    n_combo = int((iF_max + 1) * (iI_max + 1))
    pop_size = int(POP_SIZE) if int(POP_SIZE) > 0 else len(ref_points)
    pop_size = int(min(pop_size, n_combo))

    toolbox = base.Toolbox()
    toolbox.register("attr_iF", random.randint, 0, iF_max)
    toolbox.register("attr_iI", random.randint, 0, iI_max)
    toolbox.register("individual", tools.initCycle, creator.Individual2, (toolbox.attr_iF, toolbox.attr_iI), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=[0, 0], up=[iF_max, iI_max], indpb=MUT_INDPB)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    pop = init_population(pop_size, iF_max, iI_max, seed)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    pop, _ = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=pop_size,
        lambda_=pop_size,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        ngen=NGEN,
        verbose=False,
    )
    front = nondominated(pop)
    return front


def plot_front_3d(front, fert_values, irr_values, out_png, out_pdf, title):
    fert_values = np.asarray(fert_values, dtype=float)
    irr_values = np.asarray(irr_values, dtype=float)
    xs = np.array([irr_values[int(ind[1])] for ind in front], dtype=float)
    ys = np.array([fert_values[int(ind[0])] for ind in front], dtype=float)
    zs = np.array([ind.fitness.values[0] for ind in front], dtype=float)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False
    ax.set_position([0.30, 0.12, 0.60, 0.74])
    ax.view_init(elev=24, azim=32)

    try:
        triang = mtri.Triangulation(xs, ys)
        tri = triang.triangles
        norm = plt.Normalize(vmin=float(np.nanmin(zs)), vmax=float(np.nanmax(zs)))
        cmap = plt.get_cmap('viridis')
        face = cmap(norm(zs[tri].mean(axis=1)))
        surf = ax.plot_trisurf(xs, ys, zs, triangles=tri, edgecolor='white', linewidth=0.1, alpha=0.85, antialiased=True, shade=False)
        surf.set_facecolors(face)
        surf.set_zsort('min')
        surf.set_zorder(1)
    except Exception:
        ax.scatter(xs, ys, zs, c=zs, cmap='viridis',
                   norm=plt.Normalize(vmin=float(np.nanmin(zs)), vmax=float(np.nanmax(zs))),
                   s=18, alpha=0.9, edgecolor='none', zorder=1)

    ax.set_xlabel('Irrigation (mm)', fontweight='bold', labelpad=15)
    ax.set_ylabel('Fertilizer (kg N·ha$^{-1}$)', fontweight='bold', labelpad=15)
    ax.set_zlabel('')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    fig.text(0.06, 0.52, 'Yield (kg·ha$^{-1}$)', transform=fig.transFigure, rotation=90, fontsize=15, fontweight='bold', va='center', ha='left', color='black')

    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(norm=plt.Normalize(vmin=float(np.nanmin(zs)), vmax=float(np.nanmax(zs))), cmap=plt.get_cmap('viridis'))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=12, pad=0.1)
    cbar.set_label('Yield (kg·ha$^{-1}$)', fontweight='bold', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.title(title, fontweight='bold', fontsize=18, y=1.02)
    plt.legend([], [], frameon=False)
    fig.subplots_adjust(left=0.02, right=0.97, bottom=0.06, top=0.90)
    fig.savefig(out_png, dpi=400, bbox_inches='tight')
    try:
        fig.savefig(out_pdf, bbox_inches='tight')
    except PermissionError:
        alt = out_pdf.replace('.pdf', '_alt.pdf')
        fig.savefig(alt, bbox_inches='tight')
    plt.close(fig)


def plot_metrics(metrics_df, out_png, out_pdf, title):
    fig, ax1 = plt.subplots(figsize=(9.8, 5.2))
    ax1.plot(metrics_df["gen"], metrics_df["hypervolume_norm"], color="#1f77b4", linewidth=2.2)
    ax1.set_xlabel("Generation", fontweight="bold")
    ax1.set_ylabel("Hypervolume (normalized)", fontweight="bold", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(metrics_df["gen"], metrics_df["n_nondominated"], color="#2ca02c", linewidth=2.0)
    ax2.set_ylabel("Non-dominated count", fontweight="bold", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    fig.suptitle(title, fontweight="bold", fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig(out_png, dpi=400)
    try:
        fig.savefig(out_pdf)
    except PermissionError:
        fig.savefig(out_pdf.replace(".pdf", "_alt.pdf"))
    plt.close(fig)


def plot_density_if(all_df, front_df, out_png, out_pdf, title):
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    hb = ax.hexbin(all_df["irr"], all_df["fert"], gridsize=45, mincnt=1, cmap="Greys", linewidths=0.0, alpha=0.85)
    sc = ax.scatter(front_df["irr"], front_df["fert"], c=front_df["yield"], cmap="viridis", s=28, edgecolors="black", linewidths=0.3, alpha=0.95, zorder=3)
    ax.set_xlabel("Irrigation (mm)", fontweight="bold")
    ax.set_ylabel("Fertilizer (kg N·ha$^{-1}$)", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=14, pad=8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Yield (kg·ha$^{-1}$)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=400)
    try:
        fig.savefig(out_pdf)
    except PermissionError:
        fig.savefig(out_pdf.replace(".pdf", "_alt.pdf"))
    plt.close(fig)


def plot_tradeoff_2d(all_df, front_df, x_col, y_col, out_png, out_pdf, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.scatter(all_df[x_col], all_df[y_col], c="#BDBDBD", s=8, alpha=0.35, edgecolors="none", zorder=1)
    sc = ax.scatter(front_df[x_col], front_df[y_col], c=front_df["yield"], cmap="viridis", s=26, alpha=0.95, edgecolors="black", linewidths=0.25, zorder=2)
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=14, pad=8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Yield (kg·ha$^{-1}$)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=400)
    try:
        fig.savefig(out_pdf)
    except PermissionError:
        fig.savefig(out_pdf.replace(".pdf", "_alt.pdf"))
    plt.close(fig)


def load_history_all(diag_dir, ngen, sample_max=20000, seed=42):
    paths = [os.path.join(diag_dir, f"gen_{g:04d}_all.csv") for g in range(ngen + 1)]
    frames = []
    for p in paths:
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame(columns=["gen", "iF", "iI", "fert", "irr", "yield", "fert_saving", "irr_saving"])
    df = pd.concat(frames, ignore_index=True)
    if len(df) > int(sample_max):
        df = df.sample(n=int(sample_max), random_state=int(seed))
    return df


def plot_convergence_3d(all_paths, front_paths, out_png, out_pdf, title):
    fig = plt.figure(figsize=(16.0, 5.4))
    for i, (p_all, p_front) in enumerate(zip(all_paths, front_paths)):
        ax = fig.add_subplot(1, len(all_paths), i + 1, projection="3d")
        ax.computed_zorder = False
        ax.view_init(elev=24, azim=32)
        df_all = pd.read_csv(p_all)
        df_front = pd.read_csv(p_front)

        ax.scatter(df_all["irr"], df_all["fert"], df_all["yield"], c="#BDBDBD", s=8, alpha=0.35, edgecolors="none", zorder=1)
        try:
            triang = mtri.Triangulation(df_front["irr"].to_numpy(dtype=float), df_front["fert"].to_numpy(dtype=float))
            tri = triang.triangles
            z = df_front["yield"].to_numpy(dtype=float)
            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(vmin=float(np.nanmin(z)), vmax=float(np.nanmax(z)))
            face = cmap(norm(z[tri].mean(axis=1)))
            surf = ax.plot_trisurf(df_front["irr"], df_front["fert"], df_front["yield"], triangles=tri, edgecolor="white", linewidth=0.08, alpha=0.90, antialiased=True, shade=False)
            surf.set_facecolors(face)
            surf.set_zsort("min")
            surf.set_zorder(2)
        except Exception:
            ax.scatter(df_front["irr"], df_front["fert"], df_front["yield"], c=df_front["yield"], cmap="viridis", s=18, alpha=0.95, edgecolors="black", linewidths=0.25, zorder=2)

        ax.set_xlabel("Irrigation (mm)", fontweight="bold", labelpad=8)
        ax.set_ylabel("Fertilizer (kg N·ha$^{-1}$)", fontweight="bold", labelpad=10)
        ax.set_zlabel("")
        ax.set_title(f"Gen {int(df_all['gen'].iloc[0])}", fontweight="bold", fontsize=12, pad=8)
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("black")
        ax.yaxis.pane.set_edgecolor("black")
        ax.zaxis.pane.set_edgecolor("black")

    fig.suptitle(title, fontweight="bold", fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig(out_png, dpi=400)
    try:
        fig.savefig(out_pdf)
    except PermissionError:
        fig.savefig(out_pdf.replace(".pdf", "_alt.pdf"))
    plt.close(fig)


def _norm01(x):
    x = np.asarray(x, dtype=float)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    d = hi - lo
    if (not np.isfinite(d)) or d <= 0:
        return np.zeros_like(x), lo, hi
    return (x - lo) / d, lo, hi


def add_yield_std(front_df, rf_y, climate_arr):
    df = front_df.copy()
    clim = np.asarray(climate_arr, dtype=float)
    K = int(clim.shape[0])
    stds = np.full(len(df), np.nan, dtype=float)
    if K <= 1:
        df["yield_std"] = stds
        return df

    for i, r in enumerate(df.itertuples(index=False)):
        fert = float(getattr(r, "fert"))
        irr = float(getattr(r, "irr"))
        X = np.column_stack([
            np.full(K, fert, dtype=float),
            np.full(K, irr, dtype=float),
            clim[:, 0],
            clim[:, 1],
            clim[:, 2],
        ])
        yk = rf_y.predict(X).astype(float)
        stds[i] = float(np.std(yk, ddof=1))
    df["yield_std"] = stds
    return df


def pick_by_utility(front_df, w_y, w_i, w_f):
    y01, _, _ = _norm01(front_df["yield"].to_numpy(dtype=float))
    i01, _, _ = _norm01(front_df["irr"].to_numpy(dtype=float))
    f01, _, _ = _norm01(front_df["fert"].to_numpy(dtype=float))
    util = float(w_y) * y01 + float(w_i) * (1.0 - i01) + float(w_f) * (1.0 - f01)
    idx = int(np.nanargmax(util))
    return front_df.iloc[idx], util[idx]


def plot_zone_map(front_df, out_png, out_pdf, title):
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    colors = {"Low input": "#2ca02c", "Balanced": "#1f77b4", "High yield": "#ff7f0e", "High input": "#d62728"}
    for z, sub in front_df.groupby("zone"):
        ax.scatter(sub["irr"], sub["fert"], s=26, c=colors.get(z, "#7f7f7f"), alpha=0.90, edgecolors="black", linewidths=0.25, label=z)
    ax.set_xlabel("Irrigation (mm)", fontweight="bold")
    ax.set_ylabel("Fertilizer (kg N·ha$^{-1}$)", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=14, pad=8)
    ax.legend(loc="best", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=400)
    try:
        fig.savefig(out_pdf)
    except PermissionError:
        fig.savefig(out_pdf.replace(".pdf", "_alt.pdf"))
    plt.close(fig)


def recommend_zones(front_df, out_md, title_prefix, rf_y=None, climate_arr=None, out_dir=None):
    df = front_df.copy()
    qI1, qI2 = df["irr"].quantile(1 / 3), df["irr"].quantile(2 / 3)
    qF1, qF2 = df["fert"].quantile(1 / 3), df["fert"].quantile(2 / 3)
    y1, y2 = df["yield"].quantile(1 / 3), df["yield"].quantile(2 / 3)

    def zone_row(r):
        if (r["irr"] <= qI1) and (r["fert"] <= qF1):
            return "Low input"
        if r["yield"] >= y2:
            return "High yield"
        if (r["irr"] >= qI2) or (r["fert"] >= qF2):
            return "High input"
        return "Balanced"

    df["zone"] = df.apply(zone_row, axis=1)
    picks = {}
    for z in ["Low input", "Balanced", "High yield", "High input"]:
        sub = df[df["zone"] == z]
        if sub.empty:
            continue
        picks[z] = sub.loc[sub["yield"].idxmax()]

    pref_specs = [
        ("Yield-priority", 0.60, 0.20, 0.20),
        ("Balanced", 0.40, 0.30, 0.30),
        ("Input-priority", 0.25, 0.375, 0.375),
    ]
    pref_picks = [(name, *pick_by_utility(df, wy, wi, wf)) for name, wy, wi, wf in pref_specs]

    con_specs = [
        ("Budget (median)", float(df["irr"].quantile(0.50)), float(df["fert"].quantile(0.50))),
        ("Budget (Q75)", float(df["irr"].quantile(0.75)), float(df["fert"].quantile(0.75))),
    ]
    con_picks = []
    for name, i_cap, f_cap in con_specs:
        sub = df[(df["irr"] <= i_cap) & (df["fert"] <= f_cap)]
        if sub.empty:
            continue
        r = sub.loc[sub["yield"].idxmax()]
        con_picks.append((name, i_cap, f_cap, r))

    risk_picks = []
    if (rf_y is not None) and (climate_arr is not None):
        df_r = add_yield_std(df, rf_y, climate_arr)
        for lam in [0.0, 0.5, 1.0]:
            score = df_r["yield"].to_numpy(dtype=float) - float(lam) * df_r["yield_std"].to_numpy(dtype=float)
            idx = int(np.nanargmax(score))
            r = df_r.iloc[idx]
            risk_picks.append((lam, r))
        df = df_r

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# {title_prefix} Decision Zoning\n\n")
        f.write("This file summarizes a simple, explainable zoning of Pareto-optimal strategies.\n\n")
        f.write("## Zone definitions\n")
        f.write("- Low input: irrigation and fertilizer both in the lower tercile of Pareto set.\n")
        f.write("- High yield: yield in the upper tercile of Pareto set.\n")
        f.write("- High input: irrigation or fertilizer in the upper tercile.\n")
        f.write("- Balanced: remaining middle trade-off region.\n\n")
        f.write("## Representative strategies (max yield within each zone)\n")
        for z, r in picks.items():
            f.write(f"- {z}: Irr={r['irr']:.1f} mm, Fert={r['fert']:.1f} kg N/ha, Yield={r['yield']:.0f} kg/ha\n")
        f.write("\n")

        f.write("## Preference-based picks (utility on Pareto set)\n")
        f.write("Utility uses min-max normalization on the Pareto set: Yield higher is better; Irr/Fert lower is better.\n")
        for name, r, u in pref_picks:
            f.write(f"- {name}: Irr={r['irr']:.1f}, Fert={r['fert']:.1f}, Yield={r['yield']:.0f}, Utility={float(u):.3f}\n")
        f.write("\n")

        f.write("## Constraint-based picks (maximize Yield under budgets)\n")
        for name, i_cap, f_cap, r in con_picks:
            f.write(f"- {name} (I≤{i_cap:.1f}, F≤{f_cap:.1f}): Irr={r['irr']:.1f}, Fert={r['fert']:.1f}, Yield={r['yield']:.0f}\n")
        f.write("\n")

        if risk_picks:
            f.write("## Risk-tolerance picks (robust score = mean(Yield) - λ·std(Yield))\n")
            for lam, r in risk_picks:
                f.write(f"- λ={lam:.1f}: Irr={r['irr']:.1f}, Fert={r['fert']:.1f}, Yield={r['yield']:.0f}, Std={r['yield_std']:.0f}\n")
            f.write("\n")

    if out_dir:
        plot_zone_map(df, os.path.join(out_dir, "zone_map.png"), os.path.join(out_dir, "zone_map.pdf"), f"{title_prefix} zoning on Pareto set")


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    paths = _try_paths()
    if len(paths) == 0:
        raise FileNotFoundError("IR×FER66.xlsx not found")
    df_raw = load_data(paths[0])
    climate_cols = ['RAINCUM', 'TAVERC', 'PARCUM']
    if 'YEAR' in df_raw.columns:
        clim_year_pool = df_raw[['YEAR'] + climate_cols].dropna(subset=['YEAR'] + climate_cols).groupby('YEAR', as_index=False)[climate_cols].mean()
    else:
        clim_year_pool = df_raw[climate_cols].dropna().reset_index(drop=True)

    drop_cols = ['Fert_abs', 'IRCUM', 'WRR14'] + climate_cols
    df = df_raw.dropna(subset=drop_cols)

    X = df[['Fert_abs', 'IRCUM'] + climate_cols].values
    y_yield = df['WRR14'].values

    rf_y = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf_y.fit(X, y_yield)

    fert_min = round_floor_step(float(np.nanmin(df['Fert_abs'])), FERT_STEP)
    fert_max = round_ceil_step(float(np.nanmax(df['Fert_abs'])), FERT_STEP)
    irr_min = round_floor_step(float(np.nanmin(df['IRCUM'])), IRR_STEP)
    irr_max = round_ceil_step(float(np.nanmax(df['IRCUM'])), IRR_STEP)
    fert_values = np.arange(fert_min, fert_max + 0.1, FERT_STEP, dtype=float)
    irr_values = np.arange(irr_min, irr_max + 0.1, IRR_STEP, dtype=float)
    iF_max = int(fert_values.size - 1)
    iI_max = int(irr_values.size - 1)

    clim_arr = clim_year_pool[climate_cols].to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    if clim_arr.shape[0] <= K_CLIMATE_A:
        clim_A = clim_arr
    else:
        idx = rng.choice(clim_arr.shape[0], size=K_CLIMATE_A, replace=False)
        clim_A = clim_arr[idx]
    mean_y_A = precompute_objectives(rf_y, None, None, clim_A, fert_values, irr_values)

    def eval_A(ind):
        iF = int(ind[0]); iI = int(ind[1])
        return float(mean_y_A[iF, iI]), float(fert_max - fert_values[iF]), float(irr_max - irr_values[iI])

    diag_dir_A = os.path.join(out_dir, "NSGA3_Diagnostics", "ClimateA")
    obj_max_A = (float(np.nanmax(mean_y_A)), float(fert_max - fert_min), float(irr_max - irr_min))
    front_A, metrics_A = run_nsga3_with_history(
        eval_A,
        iF_max,
        iI_max,
        RANDOM_STATE,
        ngen=NGEN,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        out_dir=diag_dir_A,
        fert_values=fert_values,
        irr_values=irr_values,
        obj_max=obj_max_A,
    )

    out_A_png = os.path.join(out_dir, "3D_Pareto_NSAG3_ClimateA_Expected.png")
    out_A_pdf = os.path.join(out_dir, "3D_Pareto_NSAG3_ClimateA_Expected.pdf")
    plot_front_3d(front_A, fert_values, irr_values, out_A_png, out_A_pdf, "Three-dimensional Pareto Frontier (NSGA-III, Climate Expected)")

    plot_metrics(
        metrics_A,
        os.path.join(diag_dir_A, "metrics.png"),
        os.path.join(diag_dir_A, "metrics.pdf"),
        "NSGA-III Convergence Diagnostics (Climate Expected)",
    )
    all_last_A = os.path.join(diag_dir_A, f"gen_{NGEN:04d}_all.csv")
    front_last_A = os.path.join(diag_dir_A, f"gen_{NGEN:04d}_front.csv")
    df_all_A = load_history_all(diag_dir_A, NGEN, sample_max=20000, seed=RANDOM_STATE)
    df_front_A = pd.read_csv(front_last_A)
    plot_density_if(
        df_all_A,
        df_front_A,
        os.path.join(diag_dir_A, "density_IF.png"),
        os.path.join(diag_dir_A, "density_IF.pdf"),
        "I-F Density (All solutions) + Pareto front (colored by Yield)",
    )
    plot_tradeoff_2d(
        df_all_A,
        df_front_A,
        "fert",
        "yield",
        os.path.join(diag_dir_A, "tradeoff_FY.png"),
        os.path.join(diag_dir_A, "tradeoff_FY.pdf"),
        "Fertilizer vs Yield (Pareto in color)",
        "Fertilizer (kg N·ha$^{-1}$)",
        "Yield (kg·ha$^{-1}$)",
    )
    plot_tradeoff_2d(
        df_all_A,
        df_front_A,
        "irr",
        "yield",
        os.path.join(diag_dir_A, "tradeoff_IY.png"),
        os.path.join(diag_dir_A, "tradeoff_IY.pdf"),
        "Irrigation vs Yield (Pareto in color)",
        "Irrigation (mm)",
        "Yield (kg·ha$^{-1}$)",
    )
    gens_show = [0, int(NGEN / 2), NGEN]
    plot_convergence_3d(
        [os.path.join(diag_dir_A, f"gen_{g:04d}_all.csv") for g in gens_show],
        [os.path.join(diag_dir_A, f"gen_{g:04d}_front.csv") for g in gens_show],
        os.path.join(diag_dir_A, "convergence_3d.png"),
        os.path.join(diag_dir_A, "convergence_3d.pdf"),
        "Dominated (gray) vs Pareto (colored) across generations",
    )
    recommend_zones(
        df_front_A,
        os.path.join(diag_dir_A, "decision_zoning.md"),
        "Climate Expected",
        rf_y=rf_y,
        climate_arr=clim_A,
        out_dir=diag_dir_A,
    )

    rain_series = pd.Series(clim_year_pool['RAINCUM'].to_numpy(dtype=float))
    q25 = float(rain_series.quantile(0.25))
    q75 = float(rain_series.quantile(0.75))
    mask_dry = rain_series <= q25
    mask_wet = rain_series >= q75
    mask_mid = (~mask_dry) & (~mask_wet)
    scenarios = [
        ("P25", "Dry years", clim_year_pool.loc[mask_dry, climate_cols].to_numpy(dtype=float)),
        ("P50", "Normal years", clim_year_pool.loc[mask_mid, climate_cols].to_numpy(dtype=float)),
        ("P75", "Wet years", clim_year_pool.loc[mask_wet, climate_cols].to_numpy(dtype=float)),
    ]

    fronts = []
    for i, (tag, label, clim) in enumerate(scenarios):
        if clim.size == 0:
            clim = clim_arr
        mean_y_B = precompute_objectives(rf_y, None, None, clim, fert_values, irr_values)

        def _make_eval(my):
            def _e(ind):
                iF = int(ind[0]); iI = int(ind[1])
                return float(my[iF, iI]), float(fert_max - fert_values[iF]), float(irr_max - irr_values[iI])
            return _e

        diag_dir_B = os.path.join(out_dir, "NSGA3_Diagnostics", f"ClimateB_{tag}")
        obj_max_B = (float(np.nanmax(mean_y_B)), float(fert_max - fert_min), float(irr_max - irr_min))
        front_B, metrics_B = run_nsga3_with_history(
            _make_eval(mean_y_B),
            iF_max,
            iI_max,
            RANDOM_STATE + 100 + i,
            ngen=NGEN,
            cxpb=CX_PROB,
            mutpb=MUT_PROB,
            out_dir=diag_dir_B,
            fert_values=fert_values,
            irr_values=irr_values,
            obj_max=obj_max_B,
        )
        metrics_B.to_csv(os.path.join(diag_dir_B, "metrics.csv"), index=False, encoding="utf-8-sig")
        all_last_B = os.path.join(diag_dir_B, f"gen_{NGEN:04d}_all.csv")
        front_last_B = os.path.join(diag_dir_B, f"gen_{NGEN:04d}_front.csv")
        df_all_B = load_history_all(diag_dir_B, NGEN, sample_max=20000, seed=RANDOM_STATE + 100 + i)
        df_front_B = pd.read_csv(front_last_B)
        plot_metrics(
            metrics_B,
            os.path.join(diag_dir_B, "metrics.png"),
            os.path.join(diag_dir_B, "metrics.pdf"),
            f"NSGA-III Convergence Diagnostics ({tag})",
        )
        plot_density_if(
            df_all_B,
            df_front_B,
            os.path.join(diag_dir_B, "density_IF.png"),
            os.path.join(diag_dir_B, "density_IF.pdf"),
            f"I-F Density + Pareto front ({tag}, colored by Yield)",
        )
        plot_tradeoff_2d(
            df_all_B,
            df_front_B,
            "fert",
            "yield",
            os.path.join(diag_dir_B, "tradeoff_FY.png"),
            os.path.join(diag_dir_B, "tradeoff_FY.pdf"),
            f"Fertilizer vs Yield ({tag})",
            "Fertilizer (kg N·ha$^{-1}$)",
            "Yield (kg·ha$^{-1}$)",
        )
        plot_tradeoff_2d(
            df_all_B,
            df_front_B,
            "irr",
            "yield",
            os.path.join(diag_dir_B, "tradeoff_IY.png"),
            os.path.join(diag_dir_B, "tradeoff_IY.pdf"),
            f"Irrigation vs Yield ({tag})",
            "Irrigation (mm)",
            "Yield (kg·ha$^{-1}$)",
        )
        plot_convergence_3d(
            [os.path.join(diag_dir_B, f"gen_{g:04d}_all.csv") for g in gens_show],
            [os.path.join(diag_dir_B, f"gen_{g:04d}_front.csv") for g in gens_show],
            os.path.join(diag_dir_B, "convergence_3d.png"),
            os.path.join(diag_dir_B, "convergence_3d.pdf"),
            f"Dominated vs Pareto across generations ({tag})",
        )
        recommend_zones(
            df_front_B,
            os.path.join(diag_dir_B, "decision_zoning.md"),
            f"Climate Scenario {tag}",
            rf_y=rf_y,
            climate_arr=clim,
            out_dir=diag_dir_B,
        )
        fronts.append((tag, label, clim.shape[0], front_B))

    all_y_list = []
    for _, _, _, f in fronts:
        yv = np.array([ind.fitness.values[0] for ind in f], dtype=float)
        if yv.size:
            yv = yv[np.isfinite(yv)]
            if yv.size:
                all_y_list.append(yv)
    all_y = np.concatenate(all_y_list) if all_y_list else np.array([0.0, 1.0], dtype=float)
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    if (not np.isfinite(y_min)) or (not np.isfinite(y_max)) or (y_min == y_max):
        y_min, y_max = 0.0, 1.0
    normB = plt.Normalize(vmin=y_min, vmax=y_max)
    cmap = plt.get_cmap('viridis')

    figB = plt.figure(figsize=(16, 6.8))
    lefts = [0.10, 0.37, 0.64]
    bottom = 0.30
    width = 0.24
    height = 0.58
    for i, (tag, label, n_years, front) in enumerate(fronts):
        ax = figB.add_axes([lefts[i], bottom, width, height], projection='3d')
        ax.computed_zorder = False
        ax.view_init(elev=24, azim=32)

        xs = np.array([irr_values[int(ind[1])] for ind in front], dtype=float)
        ys = np.array([fert_values[int(ind[0])] for ind in front], dtype=float)
        zs = np.array([ind.fitness.values[0] for ind in front], dtype=float)

        ax.set_xlim(irr_min, irr_max)
        ax.set_ylim(fert_min, fert_max)
        ax.set_zlim(y_min, y_max)

        try:
            triang = mtri.Triangulation(xs, ys)
            tri = triang.triangles
            face = cmap(normB(zs[tri].mean(axis=1)))
            surf = ax.plot_trisurf(xs, ys, zs, triangles=tri, edgecolor='white', linewidth=0.1, alpha=0.85, antialiased=True, shade=False)
            surf.set_facecolors(face)
            surf.set_zsort('min')
            surf.set_zorder(1)
        except Exception:
            ax.scatter(xs, ys, zs, c=zs, cmap='viridis', norm=normB, s=16, alpha=0.9, edgecolor='none', zorder=1)

        if i == 1:
            ax.set_xlabel('Irrigation (mm)', fontweight='bold', labelpad=10)
        if i == 0:
            ax.set_ylabel('Fertilizer (kg N·ha$^{-1}$)', fontweight='bold', labelpad=16)
        ax.set_zlabel('')

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        ax.set_title(f"{tag} ({label}, n={n_years})", fontweight='bold', fontsize=14, pad=8)

    figB.text(0.03, 0.52, 'Yield (kg·ha$^{-1}$)', transform=figB.transFigure,
              rotation=90, fontsize=14, fontweight='bold', va='center', ha='left', color='black')

    from matplotlib.cm import ScalarMappable
    smB = ScalarMappable(norm=normB, cmap=cmap)
    smB.set_array([])
    cax = figB.add_axes([0.91, 0.34, 0.02, 0.46])
    cbarB = figB.colorbar(smB, cax=cax)
    cbarB.set_label('Yield (kg·ha$^{-1}$)', fontweight='bold', fontsize=14)
    cbarB.ax.tick_params(labelsize=12)

    figB.suptitle('Three-dimensional Pareto Frontier (NSGA-III, Climate Scenarios)', fontweight='bold', fontsize=18, y=0.97)

    outB_png = os.path.join(out_dir, "3D_Pareto_NSAG3_ClimateB_P25P50P75.png")
    outB_pdf = os.path.join(out_dir, "3D_Pareto_NSAG3_ClimateB_P25P50P75.pdf")
    figB.savefig(outB_png, dpi=400)
    try:
        figB.savefig(outB_pdf)
    except PermissionError:
        figB.savefig(outB_pdf.replace('.pdf', '_alt.pdf'))
    plt.close(figB)


if __name__ == "__main__":
    main()
