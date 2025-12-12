"""
Entrenamiento y búsqueda de hiperparámetros sobre los datos seleccionados.

Flujo:
- Carga `X_train_selected/X_test_selected/y_*` desde `data/selected` (o `data/processed` si no hay selección).
- Define los modelos en `model_registry.py` con sus espacios de hiperparámetros.
- Ejecuta RandomizedSearchCV (maximizando R2) o ajuste directo según corresponda.
- Reporta R2, MSE y MAE en validación y prueba; guarda resultados en CSV/JSON.

Nota: dependencias opcionales (xgboost, lightgbm, catboost, statsmodels) se omiten si no están instaladas.
"""

import argparse
import os
import sys
import warnings
from typing import Dict, List

# Asegura que los imports `scripts.*` funcionen al ejecutar como script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.training.data_utils import build_fast_subset, load_data
from scripts.training.model_registry import ModelSpec, available_specs
from scripts.training.training_pipeline import run_model_search

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento con búsqueda de hiperparámetros (R2 como métrica).")
    parser.add_argument("--data-dir", default="data/selected", help="Ruta a los datos seleccionados/preprocesados.")
    parser.add_argument("--output-dir", default="models", help="Ruta donde guardar reportes y modelos.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "lasso",
            "ridge",
            "elasticnet",
            "random_forest",
            "gradient_boosting",
            "svr",
            "knn",
            "mlp",
            "linear_predictive",
            "linear_econometric",
        ],
        help="Lista de modelos a ejecutar.",
    )
    parser.add_argument("--n-iter", type=int, default=25, help="Iteraciones para RandomizedSearchCV.")
    parser.add_argument("--cv", type=int, default=5, help="Número de folds de validación.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument("--include-optional", action="store_true", help="Intentar incluir xgboost/lightgbm/catboost si están instalados.")
    parser.add_argument(
        "--fast-sample",
        action="store_true",
        help="Usa un muestreo reducido (≈5k train/2k test) como en notebooks/showcase_modeling.ipynb.",
    )
    return parser.parse_args()


def select_specs(model_names: List[str], catalog: Dict[str, ModelSpec]) -> List[ModelSpec]:
    """Filtra las specs válidas y avisa cuando alguna no existe en el catálogo."""
    selected = []
    for name in model_names:
        if name not in catalog:
            print(f"Modelo '{name}' no disponible; se omite.")
            continue
        selected.append(catalog[name])
    return selected


def maybe_sample_fast(args, X_train, y_train, X_test, y_test):
    """Aplica el muestreo rápido si se solicita, manteniendo la lógica original por defecto."""
    if not args.fast_sample:
        return X_train, y_train, X_test, y_test

    X_train_small, y_train_small, X_test_small, y_test_small = build_fast_subset(
        X_train, y_train, X_test, y_test, max_train=5000, max_test=2000, random_state=args.random_state
    )
    print(
        f"Usando muestreo rápido -> train: {len(X_train_small)} filas, "
        f"test: {len(X_test_small)} filas."
    )
    return X_train_small, y_train_small, X_test_small, y_test_small


def main():
    args = parse_args()
    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    X_train, y_train, X_test, y_test = maybe_sample_fast(args, X_train, y_train, X_test, y_test)

    catalog = available_specs(include_optional=args.include_optional)
    selected_specs = select_specs(args.models, catalog)
    if not selected_specs:
        raise ValueError("No se seleccionaron modelos válidos.")

    results = run_model_search(
        specs=selected_specs,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_iter=args.n_iter,
        cv=args.cv,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )

    print("\nReporte de entrenamiento:")
    for r in results:
        print(
            f"{r['model']}: val_r2={r['val_r2']:.3f}, test_r2={r['test_r2']:.3f}, "
            f"test_mse={r['test_mse']:.4f}, test_mae={r['test_mae']:.4f}"
        )


if __name__ == "__main__":
    main()
