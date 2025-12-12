# Tracking de experimentos con MLflow

Este flujo documenta cómo registrar los entrenamientos del proyecto en MLflow sin cambiar la lógica existente del pipeline.

## Puntos clave
- **Wrapper**: `scripts/mlflow_utils/tracking.py` aporta `run_model_search_with_mlflow`, que envuelve `run_model_search` y crea runs anidados por modelo.
- **Artefactos**: se guardan los modelos `.joblib`, `training_report.csv` y trazas de búsqueda (`*_cv_results.csv` o `*_cv_scores.csv` + `*_search_params.csv` cuando aplica).
- **Contexto**: se registran versiones de Python, plataforma y `scikit-learn`.
- **Compatibilidad**: el tracking local usa `mlruns` en la raíz del repo con `Path(...).as_uri()` (funciona en Windows/Linux).

## Ejemplo rápido (notebook)
```python
from pathlib import Path
from scripts.training.model_registry import available_specs
from scripts.mlflow_utils.tracking import run_model_search_with_mlflow

catalog = available_specs(include_optional=True)
model_names = ["linear_predictive", "lasso", "ridge", "random_forest", "gradient_boosting"]
if "xgboost" in catalog:
    model_names.append("xgboost")
specs = [catalog[m] for m in model_names if m in catalog]

tracking_uri = Path(REPO_ROOT, "mlruns").resolve().as_uri()
results, uri = run_model_search_with_mlflow(
    specs=specs,
    X_train=X_train_small,
    y_train=y_train_small,
    X_test=X_test_small,
    y_test=y_test_small,
    n_iter=5,
    cv=3,
    random_state=77,
    output_dir=os.path.join(REPO_ROOT, "models_mlflow"),
    experiment_name="weather_training_fast",
    run_name="fast_sample",
    log_models=True,
    tracking_uri=tracking_uri,
)
print("Tracking URI:", uri)
```

Para ver la UI local:
```
mlflow ui --backend-store-uri "<uri>"
```

## Qué se registra por modelo
- Métricas: `val_r2`, `test_r2`, `test_mse`, `test_mae`, `train_duration_sec`.
- Hiperparámetros óptimos (`best_*`) cuando hay búsqueda aleatoria.
- Resultados de cross-validation o de la búsqueda (`*_cv_results.csv`) para inspeccionar trazas/folds.
- Modelo serializado (`models/<name>.joblib`).

## Extensión
El pipeline `run_model_search` ahora acepta `callbacks`, de modo que se pueden añadir nuevas trazas (por ejemplo, métricas de sistema o logging estructurado) sin modificar la lógica de entrenamiento.
