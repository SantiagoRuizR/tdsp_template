# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto
Pronóstico del clima de Múnich basado en indicadores meteorológicos (TDSP – Diplomado ML & Data Science Avanzado, UNAL).

## Objetivo del Proyecto
Construir modelos predictivos de la variable térmica `Tlog` usando datos meteorológicos de alta resolución (cada 10 minutos en 2020), priorizando:
- Fiabilidad para decisiones operativas (energía, ventilación, riego).
- Identificación de señales clave (presión, humedad, viento, radiación) para orientar mantenimiento de sensores y calidad de datos.
- Ciclo MLOps reproducible: adquisición, preprocesamiento, selección, entrenamiento, tracking (MLflow) y serving (API/Streamlit).

## Alcance del Proyecto
### Incluye
- Datos históricos 2020 (cada 10 minutos) con 20 variables meteorológicas.
- Preprocesamiento reproducible (imputación, winsorización, polinomios opcionales, escalado) y selección de características (Lasso, MI, RandomForest).
- Entrenamiento de un portafolio tabular: OLS/econométrico, Lasso/Ridge/ElasticNet, RandomForest, GradientBoosting, SVR, KNN, MLP; opcionales: XGBoost/LightGBM/CatBoost si están instalados.
- Tracking de experimentos con MLflow y artefactos versionados.
- Serving: API FastAPI (`main_api.py`) y app Streamlit (`main_streamlit.py`) usando el mismo preprocesador/selección.

### Excluye
- Datos fuera del año 2020.
- Modelos deep learning secuenciales (RNN/LSTM/GRU/Transformers) en esta iteración; se priorizó modelos tabulares interpretables y Prophet en notebooks.

## Metodología
- Marco TDSP/MLOps: adquisición → EDA → preprocesamiento → selección → entrenamiento → evaluación → deployment.
- Validación: RandomizedSearchCV con métrica R² (train/val/test), más MSE/MAE de control; muestreo rápido (~5k/2k) para iterar y full-data para versión final.
- Tracking: MLflow (runs anidados por modelo, trazas de búsqueda y artefactos).
- Serving: FastAPI para scoring batch/online y Streamlit para exploración manual/archivo.

## Cronograma (referencial)
| Etapa | Duración | Fechas (referencia) |
|-------|----------|---------------------|
| Entendimiento & datos | 1 semana | 17–23 nov |
| EDA & preprocesamiento | 1 semana | 24–30 nov |
| Selección & modelado | 1 semana | 1–7 dic |
| Tracking & deployment (API/Streamlit) | 1 semana | 8–13 dic |
| Evaluación y entrega | 1 semana | 8–13 dic |

## Equipo del Proyecto
- Santiago Ruiz Rozo — sruiz899@gmail.com
- Pablo Alejandro Reyes Granados — alejogranados229@gmail.com
- Kevin Andrés Martínez Martínez — kevinmartinez.ingbiom@gmail.com

## Stakeholders
- Dirección de Operaciones/Energía: usuarios directos de las predicciones para planificación operativa (ventilación, climatización, riego, eficiencia energética).
- Facilities/Infraestructura: uso de pronósticos para gestión de equipos HVAC y mantenimiento preventivo.
- Ciencia de Datos/Analytics interno: custodia del pipeline y evolución de modelos.
- TI/DevOps: operación de APIs/Streamlit y observabilidad de los servicios.
- Sponsor académico: Jorge E. Camargo, PhD (módulo Metodologías Ágiles para ML, UNAL).

## Aprobaciones
- Sponsor académico: Jorge E. Camargo, PhD.
- Revisión técnica: equipo del diplomado (según iteraciones de entrega).

