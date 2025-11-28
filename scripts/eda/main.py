pip install pandas numpy matplotlib scikit-learn

# ==============================================================
#                              EDA 
# ==============================================================

"""
Script de EDA: carga datos, perfila variables, genera visualizaciones y pruebas estadísticas
para entender la distribución general y la variable objetivo antes del modelado.
Pensado como insumo reproducible dentro del flujo MLOps del proyecto.
"""

import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import (
    f_oneway,
    mannwhitneyu,
    probplot,
    shapiro,
    ttest_ind,
    zscore,
)

# ------------------ ESTILO GLOBAL DE GRÁFICAS ------------------
plt.style.use("ggplot")
colors = [
    "#0077b6", "#00b4d8", "#90e0ef", 
    "#03045e", "#d62828", "#f77f00"]

# --------------------- USER SETTINGS ---------------------------
file_id = '1tYfm5wJXRHZGa5h3fsRA7tnyFUlWESpa'
download_url = f'https://drive.google.com/uc?id={file_id}'
local_path = 'data.csv'
outdir = 'eda_outputs'
os.makedirs(outdir, exist_ok=True)

def savefig(fig, name, dpi=300):
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return path

# ---------------------- READ DATA ------------------------------
df = None
try:
    df = pd.read_csv(download_url)
    source = f'Google Drive (id={file_id})'
except:
    print("⚠ No se pudo leer desde Drive. Intentando local...")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        source = f'Local file: {local_path}'
    else:
        raise FileNotFoundError("No se encontró ningún CSV.")

print("Data loaded from:", source)
print("Shape:", df.shape)
print(df.head())

# ===============================================================
#              PREPROCESAMIENTO Y TIPOS DE DATOS
# ===============================================================

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
datetime_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()

numeric_cols = [c for c in numeric_cols if c not in datetime_cols]
cat_cols = [c for c in cat_cols if c not in datetime_cols]
target_col = "Tlog" if "Tlog" in df.columns else None

# ===============================================================
#               ANALISIS ESTADISTICO GENERAL
# ===============================================================

# ---- Descriptivos extendidos
desc = df[numeric_cols].describe().T
desc["skew"] = df[numeric_cols].skew()
desc["kurtosis"] = df[numeric_cols].kurtosis()
desc.to_csv(os.path.join(outdir, "numeric_summary_extendido.csv"))

# ---- Normalidad Shapiro-Wilk
normtests = []
for col in numeric_cols:
    vals = df[col].dropna()
    if len(vals) > 5000:
        vals = vals.sample(5000, random_state=1)
    W, p = shapiro(vals)
    normtests.append([col, W, p])
norm_df = pd.DataFrame(normtests, columns=["variable", "W", "p_value"])
norm_df.to_csv(os.path.join(outdir, "normality_shapiro.csv"), index=False)

# ---- Z-score outliers
z_outliers = (np.abs(zscore(df[numeric_cols], nan_policy="omit")) > 3).sum(axis=0)
pd.DataFrame({"variable": numeric_cols, "z_outliers": z_outliers}).to_csv(
    os.path.join(outdir, "zscore_outliers.csv"), index=False)

# ---- Spearman + Kendall correlations
corr_spear = df[numeric_cols].corr(method="spearman")
corr_kend = df[numeric_cols].corr(method="kendall")

corr_spear.to_csv(os.path.join(outdir, "corr_spearman.csv"))
corr_kend.to_csv(os.path.join(outdir, "corr_kendall.csv"))

# ---- Covariance matrix
df[numeric_cols].cov().to_csv(os.path.join(outdir, "cov_matrix.csv"))

# ---- Target-specific stats y correlaciones
if target_col:
    tgt = df[target_col]
    tgt_stats = pd.DataFrame(
        {
            "count": [tgt.count()],
            "mean": [tgt.mean()],
            "std": [tgt.std()],
            "median": [tgt.median()],
            "min": [tgt.min()],
            "max": [tgt.max()],
            "skew": [tgt.skew()],
            "kurtosis": [tgt.kurtosis()],
        },
        index=[target_col],)
    
    tgt_stats.to_csv(os.path.join(outdir, "target_summary.csv"))

    corr_target = (
        df[numeric_cols + [target_col]]
        .corr(method="spearman")[target_col]
        .drop(target_col)
        .sort_values(key=np.abs, ascending=False))
    
    corr_target.to_csv(os.path.join(outdir, "target_correlations.csv"))
else:
    corr_target = pd.Series(dtype=float)

# ===============================================================
#                   VISUALIZACIONES MEJORADAS
# ===============================================================

plot_files = []

# ---------------- MISSINGNESS -----------------
fig, ax = plt.subplots(figsize=(12,4))
missing = df.isna().sum()
ax.bar(missing.index, missing.values, color=colors[0])
ax.set_title("Missing values por columna")
ax.set_xticklabels(missing.index, rotation=45, ha="right")
plot_files.append(savefig(fig, "missing.png"))

# ---------------- HISTOGRAMAS KDE -----------------
for col in numeric_cols[:12]:
    fig, ax = plt.subplots(figsize=(7,4))
    vals = df[col].dropna()
    ax.hist(vals, bins=30, color=colors[1], alpha=0.6, edgecolor="black")
    ax.set_title(f"Histograma: {col}")
    plot_files.append(savefig(fig, f"hist_{col}.png"))

# ---------------- BOXPLOTS -----------------
for col in numeric_cols[:12]:
    fig, ax = plt.subplots(figsize=(7,2))
    ax.boxplot(df[col].dropna(), vert=False,
               patch_artist=True, boxprops=dict(facecolor=colors[2], alpha=0.6))
    ax.set_title(f"Boxplot: {col}")
    plot_files.append(savefig(fig, f"box_{col}.png"))

# ---------------- CORRELATION HEATMAPS -----------------
def plot_heatmap(matrix, title, filename):
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plot_files.append(savefig(fig, filename))

plot_heatmap(corr_spear, "Spearman Correlation", "corr_spearman.png")
plot_heatmap(corr_kend, "Kendall Correlation", "corr_kendall.png")

# ---------------- PCA -----------------
if len(numeric_cols) >= 2:
    clean = df[numeric_cols].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(clean)
    pca = PCA(n_components=3)
    Xp = pca.fit_transform(Xs)

    # Scree
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(range(1,4), pca.explained_variance_ratio_, marker="o", color=colors[4])
    ax.set_title("PCA Scree")
    ax.set_xlabel("PC")
    plot_files.append(savefig(fig, "pca_scree.png"))

    # PC1 vs PC2
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(Xp[:,0], Xp[:,1], s=10, alpha=0.5, color=colors[3])
    ax.set_title("PCA - PC1 vs PC2")
    plot_files.append(savefig(fig, "pca_pc1_pc2.png"))

# ---------------- AUTOCORRELACION -----------------
if len(datetime_cols) > 0:
    dt = datetime_cols[0]
    for col in numeric_cols[:5]:
        ts = df[[dt, col]].dropna().set_index(dt)[col]
        fig, ax = plt.subplots(figsize=(10,3))
        ax.acorr(ts - ts.mean(), maxlags=200, color=colors[5])
        ax.set_title(f"Autocorrelación: {col}")
        plot_files.append(savefig(fig, f"acorr_{col}.png"))

# ---------------- PAIRPLOT CASERO -----------------
pairs = numeric_cols[:5]
k = len(pairs)
fig, axes = plt.subplots(k, k, figsize=(3*k, 3*k))
for i, xi in enumerate(pairs):
    for j, yi in enumerate(pairs):
        ax = axes[i,j]
        if i == j:
            ax.hist(df[xi].dropna(), bins=20, color=colors[0], alpha=0.6)
            ax.set_title(xi, fontsize=8)
        else:
            ax.scatter(df[yi], df[xi], s=4, alpha=0.4, color=colors[1])
        ax.set_xticks([]); ax.set_yticks([])
plot_files.append(savefig(fig, "pairplot.png"))

# ---------------- ANALISIS ESPECIFICO DEL TARGET --------------
if target_col:
    vals = df[target_col].dropna()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(vals, bins=40, density=True, color=colors[0], alpha=0.55, edgecolor="black")
    ax.axvline(vals.mean(), color=colors[3], linestyle="--", label="Media")
    ax.axvline(vals.median(), color=colors[4], linestyle=":", label="Mediana")
    ax.set_title(f"Distribución de {target_col}")
    ax.legend()
    plot_files.append(savefig(fig, "target_distribution.png"))

    fig, ax = plt.subplots(figsize=(5,5))
    probplot(vals, dist="norm", plot=ax)
    ax.set_title(f"QQ-Plot {target_col}")
    plot_files.append(savefig(fig, "target_qqplot.png"))

    if "date" in df.columns:
        ts = df.set_index("date")[target_col].sort_index().dropna()
        daily = ts.resample("D").mean()
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(daily.index, daily.values, color=colors[2], linewidth=1.5)
        ax.set_title(f"{target_col} - media diaria")
        ax.set_ylabel(target_col)
        plot_files.append(savefig(fig, "target_daily_trend.png"))

    top_corr_cols = list(corr_target.head(3).index)
    for col in top_corr_cols:
        clean = df[[col, target_col]].dropna()
        if len(clean) == 0:
            continue
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(clean[col], clean[target_col], s=6, alpha=0.4, color=colors[1])
        coef = np.polyfit(clean[col], clean[target_col], 1)
        xs = np.linspace(clean[col].min(), clean[col].max(), 50)
        ax.plot(xs, coef[0] * xs + coef[1], color=colors[4], linewidth=2)
        ax.set_title(f"{target_col} vs {col}")
        plot_files.append(savefig(fig, f"target_vs_{col}.png"))

    if "raining" in df.columns:
        df["rain_flag"] = (df["raining"] > 0).map({True: "Lluvia", False: "Sin lluvia"})
        fig, ax = plt.subplots(figsize=(6,4))
        df.boxplot(column=target_col, by="rain_flag", ax=ax, grid=False, patch_artist=True)
        ax.set_title(f"{target_col} por condición de lluvia")
        ax.set_xlabel("")
        plot_files.append(savefig(fig, "target_by_rain.png"))

    if "date" in df.columns:
        season_map = {12: "Invierno", 1: "Invierno", 2: "Invierno", 3: "Primavera", 4: "Primavera", 5: "Primavera",
                      6: "Verano", 7: "Verano", 8: "Verano", 9: "Otoño", 10: "Otoño", 11: "Otoño"}
        df["season"] = df["date"].dt.month.map(season_map)
        fig, ax = plt.subplots(figsize=(8,4))
        df.boxplot(column=target_col, by="season", ax=ax, grid=False, patch_artist=True)
        ax.set_title(f"{target_col} por estación")
        ax.set_xlabel("")
        plot_files.append(savefig(fig, "target_by_season.png"))


# ---------------- ANALISIS INFERENCIAL -----------------
if target_col:
    inference_rows = []
    if "raining" in df.columns:
        rainy = df.loc[df["raining"] > 0, target_col].dropna()
        dry = df.loc[df["raining"] == 0, target_col].dropna()
        if len(rainy) > 10 and len(dry) > 10:
            t_stat, t_p = ttest_ind(rainy, dry, equal_var=False)
            u_stat, u_p = mannwhitneyu(rainy, dry, alternative="two-sided")
            inference_rows.append({
                "comparacion": "lluvia vs sin lluvia",
                "t_stat": t_stat, "t_pvalue": t_p,
                "u_stat": u_stat, "u_pvalue": u_p,
                "n_lluvia": len(rainy), "n_seco": len(dry),})
            
    if "season" in df.columns and df["season"].notna().sum() > 0:
        groups = [g[target_col].dropna() for _, g in df.groupby("season")]
        if len(groups) >= 2 and all(len(g) > 20 for g in groups):
            f_stat, f_p = f_oneway(*groups)
            inference_rows.append({
                "comparacion": "ANOVA estaciones",
                "f_stat": f_stat,
                "f_pvalue": f_p,
                "k_grupos": len(groups),})
            
    if inference_rows:
        pd.DataFrame(inference_rows).to_csv(os.path.join(outdir, "inferential_tests.csv"), index=False)

# ===============================================================
#                   RESUMEN FINAL
# ===============================================================

print("\n Archivos generados en:", outdir)
for f in sorted(os.listdir(outdir)):
    print(" -", f)

print("\n✔ EDA completado.\n")
