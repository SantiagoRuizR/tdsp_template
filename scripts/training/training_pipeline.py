"""
Funciones de entrenamiento y búsqueda de hiperparámetros.
Separadas de main para que el punto de entrada quede más liviano.
"""

import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from scripts.training.model_registry import ModelSpec

CallbackFn = Callable[[ModelSpec, object, Dict[str, float], Dict], None]


def evaluate(estimator, X_train, y_train, X_test, y_test, val_score: Optional[float]):
    """Calcula métricas en train/test; usa val_score si viene de CV."""
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    metrics = {
        "val_r2": val_score,
        "test_r2": r2_score(y_test, preds),
        "test_mse": mean_squared_error(y_test, preds),
        "test_mae": mean_absolute_error(y_test, preds),
    }

    return metrics, preds


def _run_callbacks(callbacks: List[CallbackFn], spec: ModelSpec, estimator, metrics: Dict[str, float], extras: Dict):
    """Ejecuta callbacks opcionales para instrumentar el pipeline (MLflow, logs, etc.)."""
    for cb in callbacks:
        try:
            cb(spec=spec, estimator=estimator, metrics=metrics, extras=extras)
        except Exception as exc:  # pragma: no cover - callbacks no deben romper entrenamiento
            print(f"[callback] Error en {cb}: {exc}")


def run_model_search(
    specs: List[ModelSpec],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int,
    cv: int,
    random_state: int,
    output_dir: str,
    callbacks: Optional[List[CallbackFn]] = None,
):
    """Ejecuta búsqueda o ajuste directo y guarda resultados."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    callbacks = callbacks or []

    for spec in specs:
        print(f"\n>>> Modelo: {spec.name}")
        estimator = spec.build_estimator()
        start = time.time()
        extras: Dict = {"duration_sec": None}

        if spec.has_search:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=spec.search_space,
                n_iter=n_iter,
                scoring="r2",
                cv=cv,
                random_state=random_state,
                n_jobs=-1,
            )

            search.fit(X_train, y_train)
            best = search.best_estimator_
            val_score = search.best_score_
            extras.update({"best_params": search.best_params_, "cv_results": search.cv_results_})
        else:
            val_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
            val_score = float(np.mean(val_scores))
            best = estimator
            extras.update({"cv_scores": val_scores})

        metrics, _ = evaluate(best, X_train, y_train, X_test, y_test, val_score)
        extras["duration_sec"] = time.time() - start
        results.append(
            {
                "model": spec.name,
                "val_r2": metrics["val_r2"],
                "test_r2": metrics["test_r2"],
                "test_mse": metrics["test_mse"],
                "test_mae": metrics["test_mae"],
            }
        )

        _run_callbacks(callbacks, spec=spec, estimator=best, metrics=metrics, extras=extras)

        # Guardar modelo entrenado cuando sea pequeño/moderno (se omiten wrappers sin dump).
        model_path = os.path.join(output_dir, f"{spec.name}.joblib")
        try:
            import joblib

            joblib.dump(best, model_path)
        except Exception as exc:  # pragma: no cover - depende del modelo
            print(f"No se guardó el modelo {spec.name}: {exc}")

    pd.DataFrame(results).to_csv(os.path.join(output_dir, "training_report.csv"), index=False)
    return results
