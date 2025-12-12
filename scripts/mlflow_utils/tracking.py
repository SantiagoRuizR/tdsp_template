"""
Wrapper de MLflow para el pipeline de entrenamiento existente.

No modifica la lógica de `run_model_search`; simplemente envuelve la llamada
para registrar métricas, parámetros y artefactos en un tracking local.
"""

import os
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import pandas as pd
import sklearn

from scripts.training.model_registry import ModelSpec
from scripts.training.training_pipeline import run_model_search


def default_tracking_uri(base_dir: Optional[Path] = None) -> str:
    """
    Devuelve un tracking URI local en `mlruns/` del repo.
    """
    repo_root = base_dir or Path(__file__).resolve().parents[2]
    return (repo_root / "mlruns").resolve().as_uri()


def setup_mlflow(tracking_uri: Optional[str] = None, experiment_name: str = "model_training") -> str:
    """
    Configura MLflow apuntando a un tracking local por defecto.
    """
    uri = tracking_uri or default_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


def _log_parent_params(
    specs: List[ModelSpec],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_iter: int,
    cv: int,
    random_state: int,
    output_dir: str,
):
    mlflow.log_params(
        {
            "models": ",".join([s.name for s in specs]),
            "n_iter": n_iter,
            "cv": cv,
            "random_state": random_state,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "n_features": X_train.shape[1],
            "output_dir": output_dir,
        }
    )
    mlflow.log_params(
        {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sklearn_version": sklearn.__version__,
        }
    )


def _log_cv_artifacts(model_name: str, extras: Dict, output_dir: str):
    """Guarda resultados de CV/búsqueda para trazabilidad."""
    artifact_paths = []
    if extras.get("cv_results") is not None:
        df = pd.DataFrame(extras["cv_results"])
        path = Path(output_dir) / f"{model_name}_cv_results.csv"
        df.to_csv(path, index=False)
        artifact_paths.append((path, "cv_results"))
        if "params" in extras["cv_results"]:
            df_params = pd.DataFrame(extras["cv_results"]["params"])
            path_params = Path(output_dir) / f"{model_name}_search_params.csv"
            df_params.to_csv(path_params, index=False)
            artifact_paths.append((path_params, "cv_results"))
    elif extras.get("cv_scores") is not None:
        df = pd.DataFrame({"fold": list(range(len(extras["cv_scores"]))), "r2": extras["cv_scores"]})
        path = Path(output_dir) / f"{model_name}_cv_scores.csv"
        df.to_csv(path, index=False)
        artifact_paths.append((path, "cv_results"))

    for path, art_path in artifact_paths:
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=art_path)


def _log_child_run(
    model_name: str,
    metrics: Dict[str, float],
    model_path: str,
    extras: Dict,
    log_models: bool,
    output_dir: str,
):
    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_param("model", model_name)
        if extras.get("best_params"):
            mlflow.log_params({f"best_{k}": v for k, v in extras["best_params"].items()})
        if extras.get("duration_sec") is not None:
            mlflow.log_metric("train_duration_sec", extras["duration_sec"])
        mlflow.log_metrics(metrics)
        if log_models and os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="models")
        _log_cv_artifacts(model_name=model_name, extras=extras, output_dir=output_dir)


def run_model_search_with_mlflow(
    specs: List[ModelSpec],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int,
    cv: int,
    random_state: int,
    output_dir: str,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "model_training",
    run_name: Optional[str] = None,
    log_models: bool = True,
) -> Tuple[List[Dict], str]:
    """
    Ejecuta `run_model_search` y registra métricas/artefactos en MLflow.

    Retorna (results, tracking_uri) para que el notebook pueda mostrar la ruta
    y abrir la UI local (`mlflow ui --backend-store-uri <uri>`).
    """
    uri = setup_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        _log_parent_params(specs, X_train, X_test, n_iter, cv, random_state, output_dir)

        def _cb(spec, estimator, metrics, extras):
            model_path = os.path.join(output_dir, f"{spec.name}.joblib")
            _log_child_run(
                model_name=spec.name,
                metrics=metrics,
                model_path=model_path,
                extras=extras,
                log_models=log_models,
                output_dir=output_dir,
            )

        results = run_model_search(
            specs=specs,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
            output_dir=output_dir,
            callbacks=[_cb],
        )

        report_path = os.path.join(output_dir, "training_report.csv")
        if os.path.exists(report_path):
            mlflow.log_artifact(report_path, artifact_path="reports")

    return results, uri
