"""
Script de selección de características para datos ya preprocesados.

Flujo:
- Carga `X_train/X_test/y_*` desde `data/processed` (o ruta provista).
- Aplica múltiples técnicas de selección (L1/Lasso, importancia de bosques, mutual information).
- Devuelve y guarda conjuntos reducidos listos para modelado (`data/selected` por defecto).

"""
import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


# ----------------------------- CARGA ----------------------------- #
def load_processed(data_dir: str):
    """Lee los artefactos procesados desde disco."""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"), index_col=0)
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"), index_col=0)
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"), index_col=0).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"), index_col=0).iloc[:, 0]
    return X_train, X_test, y_train, y_test


# ----------------------- MÉTODOS DE SELECCIÓN -------------------- #
def select_via_lasso(X: pd.DataFrame, y: pd.Series, random_state: int) -> List[str]:
    """Selecciona features con L1 usando LassoCV."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LassoCV(cv=5, random_state=random_state, n_alphas=50, max_iter=5000)
    model.fit(Xs, y)
    mask = np.abs(model.coef_) > 1e-6
    selected = list(X.columns[mask])
    print(f"[Lasso] Seleccionadas {len(selected)} características (alpha={model.alpha_:.4f}).")
    return selected


def select_via_forest(X: pd.DataFrame, y: pd.Series, random_state: int, top_frac: float = 0.3) -> List[str]:
    """Usa importancia de RandomForestRegressor para quedarnos con el top `top_frac`."""
    forest = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        max_features="sqrt",)
    
    forest.fit(X, y)
    importances = forest.feature_importances_
    n_keep = max(5, int(len(importances) * top_frac))
    top_idx = np.argsort(importances)[::-1][:n_keep]
    selected = list(X.columns[top_idx])
    print(f"[Forest] Seleccionadas {len(selected)} características (top {top_frac*100:.0f}%).")
    return selected


def select_via_mutual_info(X: pd.DataFrame, y: pd.Series, random_state: int, k: int = 20) -> List[str]:
    """Selecciona top-k según mutual information."""
    mi = mutual_info_regression(X, y, random_state=random_state)
    order = np.argsort(mi)[::-1]
    k = min(k, len(order))
    selected = list(X.columns[order[:k]])
    print(f"[MutualInfo] Seleccionadas {len(selected)} características (top-k={k}).")
    return selected


def hybrid_selection(X: pd.DataFrame, y: pd.Series, random_state: int, k_mi: int = 20, top_frac_forest: float = 0.3) -> List[str]:
    """Combina Lasso, mutual information y RandomForest (unión de selecciones)."""
    lasso_feats = select_via_lasso(X, y, random_state)
    mi_feats = select_via_mutual_info(X, y, random_state, k=k_mi)
    forest_feats = select_via_forest(X, y, random_state, top_frac=top_frac_forest)

    combined = list(dict.fromkeys(lasso_feats + mi_feats + forest_feats))
    if not combined:
        combined = list(X.columns)  # fallback
    print(f"[Hybrid] Total combinadas: {len(combined)}.")
    return combined



def save_selected(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_cols: List[str],
    output_dir: str,):
    """Guarda subconjuntos seleccionados y metadatos."""
    os.makedirs(output_dir, exist_ok=True)

    X_train[selected_cols].to_csv(os.path.join(output_dir, "X_train_selected.csv"))
    X_test[selected_cols].to_csv(os.path.join(output_dir, "X_test_selected.csv"))
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), header=True)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), header=True)

    meta = {
        "n_features_selected": len(selected_cols),
        "selected_features": selected_cols,}
    
    pd.Series(meta).to_json(os.path.join(output_dir, "metadata_selection.json"), force_ascii=False, indent=2)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selección de características sobre datos preprocesados.")
    parser.add_argument("--data-dir", default="data/processed", help="Ruta a los artefactos preprocesados.")
    parser.add_argument("--output-dir", default="data/selected", help="Ruta donde se guardará la selección.")
    parser.add_argument(
        "--method",
        choices=["lasso", "forest", "mi", "hybrid"],
        default="hybrid",
        help="Método de selección.",)
    
    parser.add_argument("--random-state", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument("--top-frac-forest", type=float, default=0.3, help="Fracción superior a conservar en RandomForest.")
    parser.add_argument("--k-mi", type=int, default=20, help="Número de características a conservar por mutual information.")
    return parser.parse_args()


def main():
    args = parse_args()

    X_train, X_test, y_train, y_test = load_processed(args.data_dir)
    print(f"Dataset cargado: X_train {X_train.shape}, X_test {X_test.shape}")

    if args.method == "lasso":
        selected = select_via_lasso(X_train, y_train, args.random_state)
    elif args.method == "forest":
        selected = select_via_forest(X_train, y_train, args.random_state, args.top_frac_forest)
    elif args.method == "mi":
        selected = select_via_mutual_info(X_train, y_train, args.random_state, args.k_mi)
    else:
        selected = hybrid_selection(
            X_train, y_train, random_state=args.random_state, k_mi=args.k_mi, top_frac_forest=args.top_frac_forest)

    save_selected(X_train, X_test, y_train, y_test, selected, args.output_dir)
    print(f"Selección completada. Features finales: {len(selected)}. Guardado en {args.output_dir}.")


if __name__ == "__main__":
    main()
