# Team Data Science Project (TDSP) – Weather 2020

Plantilla TDSP orientada a MLOps para modelar variables meteorológicas medidas cada 10 minutos en 2020 (Max Planck Institute, Munich). La variable objetivo actual es `Tlog`, con foco en entender sus determinantes y habilitar modelos predictivos para escenarios como gestión energética y planificación operativa.

## Problema de negocio (contexto actual)
- Garantizar previsiones robustas de la variable térmica `Tlog`, apoyando decisiones operativas dependientes de temperatura (energía, ventilación, riego).
- Explorar señales relevantes en presión, humedad, viento y radiación para priorizar sensores y estrategias de mantenimiento de datos.

## Flujo de trabajo TDSP/MLOps hasta ahora
- Adquisición: `scripts/data_acquisition/main.py` descarga el CSV desde Google Drive y permite inspección rápida del origen.
- EDA: `scripts/eda/main.py` perfila el dataset, genera visualizaciones, analiza la distribución de `Tlog` (QQ-plot, estacionalidad, correlaciones) y corre pruebas inferenciales (t-test, Mann–Whitney, ANOVA).
- Preprocesamiento: `scripts/preprocessing/main.py` construye un pipeline reproducible con imputación (mediana/moda), winsorización configurable, One-Hot Encoding, StandardScaler, split train/test y expansión polinómica opcional. Guarda `X_train/X_test/y_*`, metadatos y el `preprocessor.joblib` en `data/processed`.

## Estructura del repositorio
```
├── docs/                 # Plantillas TDSP (charter, diccionario de datos, baseline, despliegue, etc.)
├── scripts/              # Orquestadores de cada fase (adquisición, EDA, preprocessing, training, evaluation)
│   ├── data_acquisition/
│   ├── eda/
│   ├── preprocessing/
│   ├── training/
│   └── evaluation/
├── src/                  # Implementaciones de modelos/servicios
├── requirements.txt      # Dependencias para pip
├── pyproject.toml        # Dependencias/metadata de proyecto (setuptools)
└── LICENSE               # Licencia MIT
```

## Cómo ejecutar los scripts principales
1. (Opcional) Crear entorno: `python -m venv .venv && source .venv/bin/activate` (Linux/macOS) o `.venv\Scripts\activate` (Windows).
2. Instalar dependencias: `pip install -r requirements.txt` (o `pip install -e .` si se usa `pyproject`).
3. Adquisición: `python scripts/data_acquisition/main.py` (descarga el CSV desde Drive).
4. EDA: `python scripts/eda/main.py` (genera archivos en `eda_outputs/`).
5. Preprocesamiento: `python scripts/preprocessing/main.py --input data.csv --target Tlog --poly-degree 2` (artefactos en `data/processed/`).

## Notas sobre datos
- Frecuencia: cada 10 minutos durante 2020.
- Variables: presión (`p`), temperatura (`T`, `Tpot`, `Tdew`, `Tlog`), humedad (`rh`, `VP*`, `sh`), viento (`wv`, `wd`), radiación (`SWDR`, `PAR`), y bandera de lluvia (`raining`). Diccionario en `docs/data/data_dictionary.md`.

## Licencia
Licencia MIT. Ver detalles completos en `LICENSE`.
