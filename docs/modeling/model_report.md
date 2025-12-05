# Reporte del Modelo Final

## Resumen Ejecutivo
- Objetivo: predecir `Tlog` (variable térmica) a partir de indicadores meteorológicos 2020.
- Mejor desempeño (muestra de validación rápida ~5k train / ~2k test tras selección de características):
  - **Gradient Boosting Regressor**: val_R² ≈ 0.979, test_R² ≈ 0.980, test_MSE ≈ 1.17, test_MAE ≈ 0.82.
  - Segundo: Random Forest con métricas muy cercanas.
- El baseline lineal alcanza ~0.976 de R²; los ensambles aportan mejoras marginales en error absoluto.

## Descripción del Problema
Prever la variable `Tlog` (temperatura transformada) para apoyar decisiones operativas dependientes de temperatura (energía, ventilación, riego) usando datos meteorológicos con frecuencia de 10 minutos durante 2020 (Max Planck Institute, Munich). Se requiere reproducibilidad y trazabilidad (TDSP/MLOps) para avanzar hacia despliegue.

## Descripción del Modelo
- **Preprocesamiento**: imputación (mediana/moda), winsorización (p1–p99), One-Hot Encoding, escalado estándar, términos polinómicos grado 2, split 80/20.
- **Selección de características**: estrategia híbrida (Lasso, mutual information, importancia de RandomForest) para reducir dimensionalidad y priorizar señales.
- **Modelos evaluados**: Linear (predictivo/econométrico), Lasso, Ridge, ElasticNet, RandomForest, GradientBoosting, KNN, MLP; opcionales XGBoost/LightGBM/CatBoost (no instalados en la corrida mostrada).
- **Búsqueda**: RandomizedSearchCV optimizando R², con CV=3 (corrida rápida) y n_iter=5 en muestra recortada; guardar modelos en `models/`.

## Evaluación del Modelo
| modelo             | val_r2  | test_r2 | test_mse | test_mae |
|--------------------|---------|---------|----------|----------|
| gradient_boosting  | ~0.979  | ~0.980  | ~1.17    | ~0.82    |
| random_forest      | ~0.978  | ~0.979  | ~1.25    | ~0.85    |
| linear_predictive  | ~0.976  | ~0.976  | ~1.38    | ~0.90    |
| ridge              | ~0.976  | ~0.976  | ~1.38    | ~0.89    |
| lasso              | ~0.975  | ~0.976  | ~1.39    | ~0.91    |

Interpretación: los ensambles capturan ligeras no linealidades y ofrecen menor MAE; la mejora sobre el baseline es marginal pero consistente.

## Conclusiones y Recomendaciones
- **Modelo recomendado**: Gradient Boosting (por rendimiento y simplicidad de despliegue). Random Forest como respaldo estable.
- **Baseline**: mantener la regresión lineal (predictiva/econométrica) para trazabilidad y análisis de coeficientes.
- **Siguientes pasos**: 
  - Ejecutar entrenamiento completo sin muestreo y con CV más robusto (p.ej., CV=5) para confirmar métricas.
  - Probar XGBoost/LightGBM si las dependencias están disponibles; pueden mejorar eficiencia y rendimiento.
  - Monitorear deriva de datos y recalibrar winsorización y selección de características periódicamente.
  - Preparar artefactos de inferencia (pipeline + modelo) para despliegue y documentar el contrato de entrada/salida.

## Referencias
- Documentación scikit-learn (ensemble, linear models, model_selection).  
- Documentación TDSP y scripts internos (`scripts/preprocessing`, `scripts/feature_selection`, `scripts/training`).  
- Datos: Max Planck Institute (Kaggle, 2020 series meteorológicas).
