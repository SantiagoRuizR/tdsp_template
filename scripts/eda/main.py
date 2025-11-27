pip install pandas numpy matplotlib scikit-learn

# ==============================================================
#                              EDA 
# ==============================================================

import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro, zscore

# ------------------ ESTILO GLOBAL DE GRÁFICAS ------------------
plt.style.use("ggplot")
colors = [
    "#0077b6", "#00b4d8", "#90e0ef", 
    "#03045e", "#d62828", "#f77f00"
]

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

print("📂 Data loaded from:", source)
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
    os.path.join(outdir, "zscore_outliers.csv"), index=False
)

# ---- Spearman + Kendall correlations
corr_spear = df[numeric_cols].corr(method="spearman")
corr_kend = df[numeric_cols].corr(method="kendall")

corr_spear.to_csv(os.path.join(outdir, "corr_spearman.csv"))
corr_kend.to_csv(os.path.join(outdir, "corr_kendall.csv"))

# ---- Covariance matrix
df[numeric_cols].cov().to_csv(os.path.join(outdir, "cov_matrix.csv"))

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

# ===============================================================
#                   RESUMEN FINAL
# ===============================================================

print("\n📁 Archivos generados en:", outdir)
for f in sorted(os.listdir(outdir)):
    print(" -", f)

print("\n✔ EDA completado.\n")

