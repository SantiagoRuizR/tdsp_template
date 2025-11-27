pip install pandas numpy matplotlib scikit-learn

import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------- USER SETTINGS ----------------
file_id = '1tYfm5wJXRHZGa5h3fsRA7tnyFUlWESpa'
download_url = f'https://drive.google.com/uc?id={file_id}'
local_path = 'data.csv'   # fallback file if URL cannot be reached
outdir = 'eda_outputs'    # output folder for images and CSVs
os.makedirs(outdir, exist_ok=True)
# ------------------------------------------------

def savefig(fig, name, dpi=300):
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return path

# Try reading the URL (works in Colab if internet), else local fallback
df = None
try:
    df = pd.read_csv(download_url)
    source = f'Google Drive (id={file_id})'
except Exception as e:
    print("No se pudo leer desde Drive (quizá sin internet). Intentando archivo local...")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        source = f'Local file: {local_path}'
    else:
        raise FileNotFoundError("No se encontró el CSV. Coloca tu CSV en la ruta 'data.csv' o monta Drive / usa gdown en Colab.")

print("Data loaded from:", source)
print("Shape:", df.shape)

# SAMPLE (first rows)
print("\n=== Primeras filas ===")
print(df.head().to_string())

# STRUCTURE
n_rows, n_cols = df.shape
mem = df.memory_usage(deep=True).sum()
print(f"\nRows: {n_rows}, Columns: {n_cols}, Memory bytes: {mem}")

# Column level summary
col_summary = []
for col in df.columns:
    vals = df[col]
    n_missing = int(vals.isna().sum())
    pct_missing = n_missing / n_rows * 100
    n_unique = int(vals.nunique(dropna=True))
    dtype = str(vals.dtype)
    sample_vals = vals.dropna().unique()[:5].tolist()
    col_summary.append({
        'column': col, 'dtype': dtype,
        'n_missing': n_missing, 'pct_missing': round(pct_missing,2),
        'n_unique': n_unique, 'sample_values': sample_vals
    })
col_summary_df = pd.DataFrame(col_summary).sort_values('pct_missing', ascending=False)
col_summary_df.to_csv(os.path.join(outdir, 'col_summary.csv'), index=False)
print("\nCol summary saved to:", os.path.join(outdir, 'col_summary.csv'))

# Convert 'date' column to datetime explicitly
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Optionally, drop rows where date parsing failed
    # df.dropna(subset=['date'], inplace=True)
    # n_rows = df.shape[0] # Update n_rows if rows were dropped

# Re-split numeric, categorical, and datetime columns after initial type adjustments
# This ensures that `numeric_cols` correctly reflects numerical data types
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
datetime_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()

# Remove any columns from numeric_cols or cat_cols that are now datetimes
numeric_cols = [col for col in numeric_cols if col not in datetime_cols]
cat_cols = [col for col in cat_cols if col not in datetime_cols]

# Numeric descriptive
if len(numeric_cols)>0:
    numeric_desc = df[numeric_cols].describe().T
    numeric_desc['skew'] = df[numeric_cols].skew().values
    numeric_desc['kurtosis'] = df[numeric_cols].kurtosis().values
    # IQR outliers
    iqr = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
    lower = df[numeric_cols].quantile(0.25) - 1.5*iqr
    upper = df[numeric_cols].quantile(0.75) + 1.5*iqr
    outlier_counts = ((df[numeric_cols] < lower) | (df[numeric_cols] > upper)).sum()
    numeric_desc['outlier_count'] = outlier_counts.values
    numeric_desc.reset_index().to_csv(os.path.join(outdir, 'numeric_summary.csv'), index=False)
    print("Numeric summary saved to:", os.path.join(outdir, 'numeric_summary.csv'))
else:
    numeric_desc = pd.DataFrame()

# duplicates
n_duplicates = int(df.duplicated().sum())
print("Duplicate rows:", n_duplicates)

# Outlier summary CSV
outlier_summary = []
for c in numeric_cols:
    vals = df[c].dropna()
    if len(vals)==0:
        continue
    q1,q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_out = int(((vals < lower) | (vals > upper)).sum())
    outlier_summary.append({'column': c, 'n_outliers': n_out, 'pct_outliers': round(n_out/len(vals)*100,2)})
pd.DataFrame(outlier_summary).to_csv(os.path.join(outdir, 'outlier_summary.csv'), index=False)
print("Outlier summary saved to:", os.path.join(outdir, 'outlier_summary.csv'))

# ----- PLOTS (matplotlib only) -----
plot_files = []

# Missingness overview (bar + matrix)
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(121)
missing = df.isna().sum().sort_values(ascending=False)
ax.bar(range(len(missing)), missing.values)
ax.set_xticks(range(len(missing))); ax.set_xticklabels(missing.index, rotation=45, ha='right')
ax.set_title("Missing values per column")
ax.set_ylabel("missing count")
ax2 = fig.add_subplot(122)
mask = df.isna().T.astype(int)
sampled = mask.sample(n=min(1000, mask.shape[1]), axis=1, random_state=1) if mask.shape[1] > 1000 else mask
ax2.imshow(sampled, aspect='auto', interpolation='nearest')
ax2.set_yticks(range(min(50, sampled.shape[0]))); ax2.set_yticklabels(sampled.index[:50], fontsize=8)
ax2.set_title("Missingness matrix (rows sampled if large)")
plot_files.append(savefig(fig, 'missingness_overview.png'))

# Data types distribution
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
types = df.dtypes.astype(str).value_counts()
ax.pie(types.values, labels=types.index, autopct='%1.1f%%', startangle=90)
ax.set_title("Column dtypes distribution")
plot_files.append(savefig(fig, 'dtypes_pie.png'))

# Numeric hist + box (up to 12 vars)
num_to_plot = min(len(numeric_cols), 12)
if num_to_plot>0:
    sel = numeric_cols[:num_to_plot]
    fig, axes = plt.subplots(num_to_plot, 2, figsize=(14, 3*num_to_plot))
    for i,c in enumerate(sel):
        vals = df[c].dropna()
        axes[i,0].hist(vals.values, bins=30)
        axes[i,0].set_title(f'Hist: {c}')
        axes[i,1].boxplot(vals.values, vert=False)
        axes[i,1].set_title(f'Box: {c}')
    plot_files.append(savefig(fig, 'numeric_hist_box.png'))

# Correlation heatmap
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr, aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Correlation matrix")
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cax)
    plot_files.append(savefig(fig, 'correlation_heatmap.png'))

# Pairwise scatter (top 6 by variance)
if len(numeric_cols) >= 2:
    var = df[numeric_cols].var().sort_values(ascending=False)
    pair_cols = var.index[:6].tolist()
    sample_df = df[pair_cols].dropna()
    if sample_df.shape[0] > 2000:
        sample_df = sample_df.sample(2000, random_state=1)
    k = len(pair_cols)
    fig, axes = plt.subplots(k, k, figsize=(3*k, 3*k))
    for i, xi in enumerate(pair_cols):
        for j, yj in enumerate(pair_cols):
            ax = axes[i,j]
            if i==j:
                ax.hist(sample_df[xi].values, bins=20)
                ax.set_title(xi, fontsize=8)
            else:
                ax.scatter(sample_df[yj], sample_df[xi], s=6, alpha=0.6)
            if i < k-1: ax.set_xticks([])
            if j > 0: ax.set_yticks([])
    plot_files.append(savefig(fig, 'pairwise_scatter_matrix.png'))

# Categorical top frequencies
for c in cat_cols:
    vc = df[c].value_counts(dropna=False).head(10)
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111)
    ax.bar(range(len(vc)), vc.values)
    ax.set_xticks(range(len(vc))); ax.set_xticklabels([str(x) for x in vc.index], rotation=45, ha='right', fontsize=8)
    ax.set_title(f"Top categories: {c}")
    plot_files.append(savefig(fig, f'cat_freq_{c[:30].replace(" ", "_")}.png'))

# Time series overview if datetime cols found
if len(datetime_cols)>0 and len(numeric_cols)>0:
    for dtc in datetime_cols:
        tmp = df[[dtc]+numeric_cols].dropna()
        tmp = tmp.set_index(dtc).sort_index()
        if tmp.shape[0] > 0:
            agg = tmp.resample('D').mean().interpolate()
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot(111)
            for col in agg.columns:
                ax.plot(agg.index, agg[col], label=col)
            ax.set_title('Time series: daily mean')
            ax.legend(fontsize=8)
            plot_files.append(savefig(fig, f'timeseries_{dtc}.png'))

# PCA for numeric features
if len(numeric_cols) >= 2:
    clean_numeric = df[numeric_cols].dropna()
    if clean_numeric.shape[0] > 5000:
        clean_numeric = clean_numeric.sample(5000, random_state=1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(clean_numeric)
    pca = PCA(n_components=min(10, Xs.shape[1]))
    Xp = pca.fit_transform(Xs)
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
    ax.set_title('PCA scree')
    ax.set_xlabel('PC index'); ax.set_ylabel('explained variance ratio')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plot_files.append(savefig(fig, 'pca_scree.png'))

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.scatter(Xp[:,0], Xp[:,1], s=6, alpha=0.6)
    ax.set_title('PCA PC1 vs PC2')
    plot_files.append(savefig(fig, 'pca_pc1_pc2.png'))

# Final report print
print("\nFiles generated in folder:", outdir)
for f in sorted(os.listdir(outdir)):
    print(" -", f)
print("\nDone.")
