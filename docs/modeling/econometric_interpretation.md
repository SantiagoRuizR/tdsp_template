# Interpretación Econométrica (OLS)

## Resumen

- Modelo: OLS con reducción de multicolinealidad por VIF (umbral=30) y errores robustos HC3.
- Métricas (muestra 10k): R² ≈ 0.974, Adj. R² ≈ 0.974, F-stat ≈ 1.24e4 (p < 0.001), n=10k; Cond. No. ≈ 19.6.
- Conclusión rápida: ajuste fuerte; los signos/p-valores son confiables y la multicolinealidad quedó controlada tras la poda por VIF.

## Variables con mayor efecto (p < 0.05)

- Fuertemente positivas: `num__p Tpot`, `num__p rho`, `num__rho max. PAR`, `num__VPdef^2`, `num__VPact^2`, `num__VPact VPdef`.
- Fuertemente negativas: interacciones de humedad/velocidad-viento (`num__rh max. wv`, `num__rh wv`, `num__VPdef wv`, `num__VPdef wd`), combinación presión-densidad `num__p rho`.
- Efectos mixtos de lluvia/radiación: `num__T rain` (positivo), `num__rain SWDR` y `num__SWDR max. PAR` (negativos).
- Variables retenidas tras VIF (ejemplo 10k): `num__p Tpot`, `num__T rain`, `num__Tdew wv`, `num__Tdew wd`, `num__Tdew raining`, `num__Tdew SWDR`, `num__rh wv`, `num__rh max. wv`, `num__rh SWDR`, `num__VPdef^2`, `num__VPdef wv`, `num__VPdef wd`, `num__VPdef rain`, `num__VPdef raining`, `num__rho wd`, `num__rho max. PAR`, `num__wv^2`, `num__wv raining`, `num__rain raining`, `num__rain SWDR`, `num__raining SWDR`, `num__SWDR max. PAR`, `num__max. PAR^2`, `num__VPact VPdef`, `num__p rho`, `num__VPact^2`, `num__rh^2`, `num__VPdef PAR`.

## Consideraciones

- Se usó HC3 para robustez frente a heterocedasticidad.
- El CLT respalda aproximación a normalidad de errores dada la muestra grande; no se aplicó corrección adicional.
- Las magnitudes están afectadas por escalado y polinomios; interpretar signos y significancia relativa más que valores absolutos.

## Recomendaciones

- Mantener la versión con VIF reducido para interpretación; si se desea mayor parsimonia, eliminar términos con p > 0.1.
- Usar este OLS como lectura de efectos; para producción, los ensambles (Gradient Boosting/Random Forest) siguen siendo la opción recomendada por MAE/R².
