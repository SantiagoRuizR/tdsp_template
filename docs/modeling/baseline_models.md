# Reporte del Modelo Baseline

## Descripción del modelo
Se usó como línea base una **Regresión Lineal** (versión predictiva con scikit-learn y versión econométrica con statsmodels cuando está disponible). Los datos de entrada se preprocesaron con imputación (mediana/moda), winsorización (p1–p99), One-Hot Encoding, escalado estándar, términos polinómicos grado 2 y split 80/20. La selección híbrida de características (Lasso + MI + RandomForest) redujo el espacio a las variables más informativas antes de entrenar.

## Variables de entrada
- Variables numéricas originales y polinómicas generadas (tras OHE para categorías).  
- Conjunto final seleccionado vía **hybrid selection**: combinación de L1, mutual information y feature importance de RandomForest.

## Variable objetivo
- `Tlog` (temperatura transformada).

## Evaluación del modelo
### Métricas
- R² (métrica objetivo), MSE, MAE.

### Resultados (baseline)
| modelo              | val_r2  | test_r2 | test_mse | test_mae |
|---------------------|---------|---------|----------|----------|
| linear_predictive   | ~0.976  | ~0.976  | ~1.378   | ~0.903   |

*Nota:* valores obtenidos con un muestreo rápido (≈5k train / 2k test) para validar el flujo; el entrenamiento completo puede variar ligeramente.

## Análisis de resultados
- El baseline lineal ya captura ~97% de la varianza de `Tlog`, coherente con relaciones casi lineales entre variables térmicas y la meta.
- La versión econométrica permite inspeccionar betas y significancia (cuando statsmodels está instalado).
- Limitación: no modela no linealidades residuales ni interacciones complejas más allá de los términos polinómicos básicos.

## Conclusiones
- El baseline ofrece un punto de partida sólido; sin embargo, modelos de ensamble (RandomForest, Gradient Boosting) superan ligeramente el desempeño en R² y error absoluto.
- Mantener el baseline como referencia para detectar sobreajuste en modelos más complejos.

## Referencias
- Documentación scikit-learn (LinearRegression).  
- Documentación statsmodels (OLS).  
- Pipeline interno de preprocesamiento y selección de características del proyecto.
