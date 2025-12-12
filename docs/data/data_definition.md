# Definición de los datos

## Origen de los datos

- [x] Fuente: serie de tiempo meteorologica 2020 (Instituto Max Planck, Munich) publicada en Kaggle como *Weather long term time series forecasting*.
- [x] Obtencion: descarga directa del CSV desde Google Drive (`file_id=1tYfm5wJXRHZGa5h3fsRA7tnyFUlWESpa`) usando `pandas.read_csv`; si existe `data.csv` en el raiz del repo se usa localmente.

## Especificación de los scripts para la carga de datos

- [x] `scripts/data_acquisition/main.py` expone `load_dataset` que lee un CSV local o descarga desde el id de Google Drive anterior; se puede ejecutar con `python scripts/data_acquisition/main.py` para validar lectura inicial.

## Referencias a rutas o bases de datos origen y destino

- [x] Los datos viven en el sistema de archivos del proyecto (no hay base de datos externa); los artefactos intermedios se guardan bajo `data/` y los modelos en `models/` o `models_fast/`.

### Rutas de origen de datos

- [x] Ubicacion: `data.csv` en el raiz del repositorio, descargado desde `https://drive.google.com/uc?id=1tYfm5wJXRHZGa5h3fsRA7tnyFUlWESpa`.
- [x] Estructura: archivo CSV con ~52,696 filas y 21 columnas (fecha + 20 mediciones numericas); ver detalle de campos en `docs/data/data_dictionary.md`. No presenta valores faltantes.
- [x] Transformacion/limpieza: `scripts/preprocessing/main.py` aplica imputacion (mediana/moda), winsorizacion opcional, One-Hot Encoding, estandarizacion, split train/test y expansion polinomica opcional; guarda `X_train/X_test/y_*`, metadatos y `preprocessor.joblib` en `data/processed/`. `scripts/feature_selection/main.py` carga esos artefactos y genera `X_train_selected/X_test_selected` en `data/selected/` usando Lasso, mutual information y RandomForest en modo hibrido.

### Base de datos de destino

- [x] Destino: almacenamiento local en disco. Modelos y reportes se escriben en `models/` (entrenamiento estandar) o `models_fast/` (muestreo rapido).
- [x] Estructura de salida: `data/processed/` contiene los CSV de train/test y metadatos; `data/selected/` conserva los seleccionados (`X_*_selected.csv`, `metadata_selection.json`); `models*/` almacena `.joblib` por modelo y `training_report.csv`.
- [x] Carga/transformacion a destino: los scripts de entrenamiento (`scripts/training/main.py`) leen `data/selected/` (o `data/processed/` si no existe seleccion) y escriben los artefactos mencionados sin pasos adicionales de transformacion en base de datos.
