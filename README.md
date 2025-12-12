# Team Data Science Project (TDSP) - Weather 2020

Plantilla TDSP orientada a MLOps para modelar variables meteorológicas medidas cada 10 minutos en 2020 (Max Planck Institute, Munich). La variable objetivo actual es `Tlog`, con foco en entender sus determinantes y habilitar modelos predictivos para escenarios como gestión energética y planificación operativa.

Proyecto del Diplomado en Machine Learning y Data Science Avanzado (UNAL) - Modulo 3. Incluye pipeline reproducible (adquisicion -> EDA -> preprocesamiento -> seleccion -> entrenamiento) y API de scoring con preprocesamiento/seleccion acoplados.

## Problema de negocio (contexto actual)
- Garantizar previsiones robustas de la variable térmica `Tlog`, apoyando decisiones operativas dependientes de temperatura (energía, ventilación, riego).
- Explorar señales relevantes en presión, humedad, viento y radiación para priorizar sensores y estrategias de mantenimiento de datos.

## Flujo de trabajo TDSP/MLOps hasta ahora
- Adquisición: `scripts/data_acquisition/main.py` descarga el CSV desde Google Drive y permite inspección rápida del origen.
- EDA: `scripts/eda/main.py` perfila el dataset, genera visualizaciones, analiza la distribución de `Tlog` (QQ-plot, estacionalidad, correlaciones) y corre pruebas inferenciales (t-test, Mann-Whitney, ANOVA).
- Preprocesamiento: `scripts/preprocessing/main.py` construye un pipeline reproducible con imputación (mediana/moda), winsorización configurable, One-Hot Encoding, StandardScaler, split train/test y expansión polinómica opcional. Guarda `X_train/X_test/y_*`, metadatos y el `preprocessor.joblib` en `data/processed`.
- Selección de características: `scripts/feature_selection/main.py` aplica Lasso, mutual information y RandomForest (modo híbrido por defecto) y deja `X_train_selected/X_test_selected` en `data/selected`.
- Entrenamiento: `scripts/training/main.py` instancia múltiples modelos y ejecuta RandomizedSearchCV maximizando R2; reporta R2/MSE/MAE y guarda los modelos en `models/`.
- Tracking: `scripts/mlflow_utils/tracking.py` envuelve el entrenamiento para registrar runs y artefactos en MLflow (incluye trazas de busqueda y CV).
- API: `main.py` + `scripts/api/` sirven un modelo preentrenado con el mismo preprocesamiento y selección de features usados en training.

## Estructura del repositorio
```
--- docs/                 # Plantillas TDSP (charter, diccionario de datos, baseline, despliegue, etc.)
--- scripts/              # Orquestadores de cada fase (adquisición, EDA, preprocessing, training, evaluation)
    --- data_acquisition/
    --- eda/
    --- preprocessing/
    --- training/
    --- evaluation/
    --- api/              # App FastAPI (app.py, schemas, rutas, carga de artefactos)
--- notebooks/            # Notebooks de showcase (EDA, modelado rápido con MLflow)
--- experiments/          # Salidas de EDA y gráficos
--- src/                  # Paquete importable (exposición de pipelines y API helpers)
--- requirements.txt      # Dependencias para pip
--- pyproject.toml        # Dependencias/metadata de proyecto (setuptools)
--- LICENSE               # Licencia MIT
```

## Cómo ejecutar los scripts principales
1. (Opcional) Crear entorno: `python -m venv .venv && source .venv/bin/activate` (Linux/macOS) o `.venv\Scripts\activate` (Windows).
2. Instalar dependencias: `pip install -r requirements.txt` (o `pip install -e .` si se usa `pyproject`).
3. Adquisición: `python scripts/data_acquisition/main.py` (descarga el CSV desde Drive).
4. EDA: `python scripts/eda/main.py` (genera archivos en `eda_outputs/`).
5. Preprocesamiento: `python scripts/preprocessing/main.py --input data.csv --target Tlog --poly-degree 2` (artefactos en `data/processed/`).
6. Selección de features: `python scripts/feature_selection/main.py --data-dir data/processed --output-dir data/selected --method hybrid`.
7. Entrenamiento (sin MLflow): `python scripts/training/main.py --data-dir data/selected --output-dir models --models lasso ridge random_forest gradient_boosting linear_predictive`.
8. Entrenamiento con MLflow (tracking local en `mlruns/`):
   ```bash
   python - <<'PY'
   import os
   from pathlib import Path
   from scripts.training.model_registry import available_specs
   from scripts.mlflow_utils.tracking import run_model_search_with_mlflow
   import pandas as pd

   # Carga de datos ya preprocesados/seleccionados
   X_train = pd.read_csv("data/selected/X_train_selected.csv", index_col=0)
   X_test = pd.read_csv("data/selected/X_test_selected.csv", index_col=0)
   y_train = pd.read_csv("data/selected/y_train.csv", index_col=0).iloc[:,0]
   y_test = pd.read_csv("data/selected/y_test.csv", index_col=0).iloc[:,0]

   catalog = available_specs(include_optional=True)
   model_names = ["linear_predictive","lasso","ridge","random_forest","gradient_boosting"]
   if "xgboost" in catalog:
       model_names.append("xgboost")
   specs = [catalog[m] for m in model_names if m in catalog]

   results, uri = run_model_search_with_mlflow(
       specs=specs,
       X_train=X_train, y_train=y_train,
       X_test=X_test, y_test=y_test,
       n_iter=5, cv=3, random_state=77,
       output_dir="models_fast",
       experiment_name="weather_training_fast",
       run_name="fast_sample",
       tracking_uri=Path("mlruns").resolve().as_uri(),
   )
   print("MLflow tracking URI:", uri)
   PY
   ```
   Levantar UI: `mlflow ui --backend-store-uri "<uri impreso>"`.

## API de predicción (FastAPI)
- Artefactos requeridos: `models_fast/<modelo>.joblib`, `data/processed/preprocessor.joblib`, `data/selected/metadata_selection.json` (si hubo selección).
- Variables de entorno (`API_`): `API_MODEL_PATH`, `API_PREPROCESSOR_PATH`, `API_SELECTION_METADATA_PATH`, `API_TITLE`, `API_DESCRIPTION`.
- Arranque local:
  ```
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```
- Endpoints principales:
  - `GET /health`: estado del modelo y features requeridas.
  - `POST /predict`: recibe registros crudos (columnas originales) y devuelve predicciones con preprocesamiento+selección.
  - `POST /evaluate`: métricas R2/MAE/MSE sobre un batch con `y_true`.
  - `POST /drift`: KS-test por feature numérica entre referencia y batch actual.

## Notebooks y experiments
- `notebooks/showcase_modeling.ipynb`: pipeline completo con muestreo rápido (5k/2k), ejecución de modelos y MLflow.
- `notebooks/showcase_eda.ipynb`: exploración visual y estadística.
- `experiments/`: gráficos de EDA generados (`corr`, histogramas, PCA, Prophet, etc.).

## Modelos soportados (training)
- Core: `linear_predictive`, `linear_econometric`, `lasso`, `ridge`, `elasticnet`, `random_forest`, `gradient_boosting`, `svr`, `knn`, `mlp`.
- Opcionales (si están instalados): `xgboost`, `lightgbm`, `catboost`.
- Métrica optimizada: R2 (RandomizedSearchCV o CV simple); se reporta también MSE/MAE en test.

## Resultados de ejemplo (muestra reducida)

| modelo            | val_r2 | test_r2 | test_mse | test_mae |
|-------------------|--------|---------|----------|----------|
| gradient_boosting | ~0.979 | ~0.980  | ~1.17    | ~0.82    |
| random_forest     | ~0.978 | ~0.979  | ~1.25    | ~0.85    |
| linear_predictive | ~0.976 | ~0.976  | ~1.38    | ~0.90    |
| ridge             | ~0.976 | ~0.976  | ~1.38    | ~0.89    |
| lasso             | ~0.975 | ~0.976  | ~1.39    | ~0.91    |

*Nota:* métricas obtenidas con un muestreo rápido (≈5k train / 2k test) para validar el flujo; el entrenamiento completo puede variar ligeramente.

## Interpretación econométrica
- OLS con reducción de VIF y errores robustos HC3 (muestra 10k): R2 ≈ 0.974, Adj. R2 ≈ 0.974, F-stat ≈ 1.24e4 (p < 0.001), Cond. No. ≈ 19.6.
- Efectos destacados: interacción presión-Tpot y presión-densidad positivas; combinaciones de humedad/viento y lluvia-radiación con coeficientes negativos; términos cuadrados de VPdef/VPact positivos.
- Se usó VIF para controlar multicolinealidad y CLT para asumir normalidad de errores dada la muestra grande.
- Más detalles en `docs/modeling/econometric_interpretation.md`.

## Notas sobre datos
- Frecuencia: cada 10 minutos durante 2020.
- Variables: presión (`p`), temperatura (`T`, `Tpot`, `Tdew`, `Tlog`), humedad (`rh`, `VP*`, `sh`), viento (`wv`, `wd`), radiación (`SWDR`, `PAR`), y bandera de lluvia (`raining`). Diccionario en `docs/data/data_dictionary.md`.

## Licencia
Licencia MIT. Ver detalles completos en `LICENSE`.
