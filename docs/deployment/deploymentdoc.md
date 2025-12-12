# Despliegue de la API de predicción (Módulo 3 – Dipl. ML & Data Science UNAL)

## Infraestructura
- **Modelo servido:** `gradient_boosting.joblib` (por defecto) entrenado sobre `Tlog`, con preprocesamiento y selección de features.
- **Artefactos requeridos:**
  - `data/processed/preprocessor.joblib` (pipeline con imputación, winsorización, escalado, polinómicas).
  - `data/selected/metadata_selection.json` (lista de features seleccionadas; opcional si no hubo selección).
  - `models_fast/<modelo>.joblib` (modelo serializado).
- **Plataforma de despliegue:** Contenedor Docker ejecutando FastAPI + Uvicorn. Puede correr en VM, AKS/EKS, ECS/Fargate o App Service.
- **Requisitos técnicos:**
  - Python >= 3.10
  - Dependencias clave: `fastapi`, `uvicorn[standard]`, `scikit-learn`, `pandas`, `numpy`, `mlflow`, `pydantic-settings` (opcional, hay fallback).
  - CPU: 1–2 vCPU; RAM: 2–4 GB (dependiendo del tamaño del preprocesador y el modelo).
- **Requisitos de seguridad:**
  - Exponer vía HTTPS (terminación TLS en el LB/gateway).
  - Autenticación (token/JWT) opcional a nivel de gateway o middleware.
  - Network ACL/NSG/VPC para restringir orígenes.
  - Logs a un backend central (CloudWatch/Log Analytics/ELK).

## Código de despliegue
- **Archivo principal:** `main.py` (expone `app = create_app()`).
- **Módulos API:** `scripts/api/app.py`, `scripts/api/routes.py`, `scripts/api/artifacts.py`, `scripts/api/schemas.py`.
- **Rutas clave:**
  - `GET /health`: estado del modelo y features requeridas.
  - `POST /predict`: recibe registros crudos (columns originales) y devuelve predicciones tras preprocesar + seleccionar features.
  - `POST /evaluate`: mismo payload + `y_true`, calcula R²/MAE/MSE sobre el batch.
  - `POST /drift`: referencia vs. batch actual, drift KS por feature numérica.
- **Variables de entorno (prefijo `API_`):**
  - `API_MODEL_PATH` (por defecto `models_fast/gradient_boosting.joblib`)
  - `API_PREPROCESSOR_PATH` (por defecto `data/processed/preprocessor.joblib`)
  - `API_SELECTION_METADATA_PATH` (por defecto `data/selected/metadata_selection.json`)
  - `API_TITLE`, `API_DESCRIPTION`, `API_VERSION` (metadatos de docs).

## Instrucciones de instalación (local)
1) Crear entorno y activar: `python -m venv .venv && source .venv/bin/activate`  
2) Instalar dependencias: `pip install -r requirements.txt`  
3) Verificar artefactos en disco (`data/processed/preprocessor.joblib`, `models_fast/*.joblib`).  
4) Ejecutar API:  
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Ejemplo de uso
Petición de predicción (JSON abreviado):
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "p": 1010, "T": 12.5, "Tpot": 285, "Tdew": 8.0, "rh": 82,
      "VPmax": 14.0, "VPact": 11.5, "VPdef": 2.5, "sh": 8.0, "H2OC": 12.0,
      "rho": 1200, "wv": 2.0, "max. wv": 4.0, "wd": 180, "rain": 0.0,
      "raining": 0.0, "SWDR": 0.0, "PAR": 0.0, "max. PAR": 0.0
    }]
  }'
```
Respuesta: `{ "model_name": "...", "n_records": 1, "predictions": [...] }`

## Instrucciones de configuración para contenedor
1) Ajustar variables de entorno (`API_MODEL_PATH`, etc.) en el orquestador.  
2) Montar volumen o copiar artefactos de modelo/preprocesamiento en la imagen.  
3) Exponer puerto 8000 interno y publicar vía LB con TLS.  
4) Opcional: añadir autenticación (gateway/JWT) y rate limiting.

## Mantenimiento y MLOps
- **Re-entrenamiento:** usar el pipeline (`scripts/preprocessing`, `scripts/feature_selection`, `scripts/training`) y publicar nuevos artefactos; actualizar `API_MODEL_PATH` al nuevo joblib.  
- **Monitoreo:** usar `/drift` para checks rápidos de distribución; integrar logs/metrics en APM.  
- **Trazabilidad:** MLflow deja runs en `mlruns/`; conservar versión de artefactos desplegados.  
- **Pruebas de humo:** `GET /health` + batch pequeño en `/predict` antes de cada release.
