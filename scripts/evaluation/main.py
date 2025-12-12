"""
Evaluación de modelos entrenados y chequeos básicos de drift.

Uso sugerido:
python scripts/evaluation/main.py --model-path models_fast/random_forest.joblib --data-dir data/selected --output-dir evaluation_outputs
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")


def load_split(data_dir: str, split: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Carga X_split/y_split desde el directorio indicado."""
    X_path = os.path.join(data_dir, f"X_{split}.csv")
    y_path = os.path.join(data_dir, f"y_{split}.csv")
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"No se encontró el split '{split}' en {data_dir}")
    X = pd.read_csv(X_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0).iloc[:, 0]
    return X, y


def load_model(model_path: str):
    """Carga un modelo serializado."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    return joblib.load(model_path)


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], np.ndarray]:
    """Calcula métricas y retorna predicciones."""
    preds = model.predict(X)
    metrics = {
        "r2": r2_score(y, preds),
        "mse": mean_squared_error(y, preds),
        "mae": mean_absolute_error(y, preds),
    }
    return metrics, preds


def save_metrics(metrics: Dict[str, float], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def plot_parity(y_true: pd.Series, y_pred: np.ndarray, output_dir: Path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, edgecolor="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("Predicción vs Real")
    plt.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "parity_plot.png")
    plt.close()


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, output_dir: Path):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, alpha=0.7, color="steelblue")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residual")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de residuales")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "residuals_hist.png")
    plt.close()


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05, max_samples: int = 5000) -> pd.DataFrame:
    """
    Drift simple vía KS-test por columna numérica.
    """
    ref = reference.copy()
    cur = current.copy()
    common_cols = [c for c in ref.columns if c in cur.columns]
    rows = []
    for col in common_cols:
        ref_col = ref[col].dropna()
        cur_col = cur[col].dropna()
        if len(ref_col) == 0 or len(cur_col) == 0:
            continue
        ref_sample = ref_col.sample(n=min(len(ref_col), max_samples), random_state=42) if len(ref_col) > max_samples else ref_col
        cur_sample = cur_col.sample(n=min(len(cur_col), max_samples), random_state=42) if len(cur_col) > max_samples else cur_col
        stat, pvalue = ks_2samp(ref_sample, cur_sample)
        rows.append({"feature": col, "ks_stat": stat, "p_value": pvalue, "drift": pvalue < alpha})
    return pd.DataFrame(rows)


def save_predictions(y_true: pd.Series, y_pred: np.ndarray, output_dir: Path):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=y_true.index)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "predictions.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación de modelos entrenados y chequeo de drift.")
    parser.add_argument("--model-path", required=True, help="Ruta al modelo .joblib a evaluar.")
    parser.add_argument("--data-dir", default="data/selected", help="Directorio con X_train/X_test/y_*.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Split a evaluar.")
    parser.add_argument("--reference-split", default="train", choices=["train", "test"], help="Split de referencia para drift.")
    parser.add_argument("--output-dir", default="evaluation_outputs", help="Directorio donde guardar reportes y gráficas.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Umbral p-value para drift KS.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    model_name = Path(args.model_path).stem
    model_out = output_dir / model_name

    X_eval, y_eval = load_split(args.data_dir, args.split)
    model = load_model(args.model_path)

    metrics, preds = evaluate_model(model, X_eval, y_eval)
    save_metrics(metrics, model_out)
    save_predictions(y_eval, preds, model_out)
    plot_parity(y_eval, preds, model_out)
    plot_residuals(y_eval, preds, model_out)

    drift_report_path: Optional[Path] = None
    try:
        X_ref, _ = load_split(args.data_dir, args.reference_split)
        drift_df = detect_drift(X_ref, X_eval, alpha=args.alpha)
        if not drift_df.empty:
            drift_report_path = model_out / "drift_report.csv"
            drift_df.to_csv(drift_report_path, index=False)
    except Exception as exc:
        print(f"No se pudo calcular drift: {exc}")

    print(f"Reporte guardado en {model_out}")
    print(f"Métricas: {metrics}")
    if drift_report_path:
        print(f"Drift report: {drift_report_path}")


if __name__ == "__main__":
    main()
