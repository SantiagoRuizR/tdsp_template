# Reporte de Datos

Este documento contiene los resultados del análisis exploratorio de datos.

## Resumen general de los datos

El dataset contiene 21 variables y representa mediciones meteorológicas tomadas cada 10 minutos durante un periodo continuo del año 2020.

**Número de observaciones y variables**

  • Total de filas: 52696

  • Total de columnas: 21

**Tipos de variables**

date: variable temporal (string → datetime)

Variables numéricas continuas: temperatura, presión, humedad, viento, radiación, etc.

Variables binarias: raining

Variables derivadas: Tpot, VPdef, Tlog

**Valores faltantes**

No se observan valores faltantes.

En esta sección se presenta un resumen general de los datos. Se describe el número total de observaciones, variables, el tipo de variables, la presencia de valores faltantes y la distribución de las variables.

## Resumen de calidad de los datos

En esta sección se presenta un resumen de la calidad de los datos. Se describe la cantidad y porcentaje de valores faltantes, valores extremos, errores y duplicados. También se muestran las acciones tomadas para abordar estos problemas.

## Variable objetivo

En esta sección se describe la variable objetivo. Se muestra la distribución de la variable y se presentan gráficos que permiten entender mejor su comportamiento.

La variable objetivo del dataset es:

Tlog

Una transformación logarítmica de la temperatura u otra magnitud térmica.

Distribución

Tiene una distribución aproximadamente lineal y creciente en el tiempo, probablemente asociada al ciclo anual.

No presenta valores extremos.

## Variables individuales

En esta sección se presenta un análisis detallado de cada variable individual. Se muestran estadísticas descriptivas, gráficos de distribución y de relación con la variable objetivo (si aplica). Además, se describen posibles transformaciones que se pueden aplicar a la variable.

A continuación se resume el comportamiento de las principales variables:

Temperatura (T)

Rango típico: ~ -2°C a 20°C en la muestra inicial.

Distribución unimodal.

Alta correlación con Tpot y Tdew.

Humedad relativa (rh)

Valores entre 70% y 95%.

Muy estable, coherente con clima húmedo.

Velocidad del viento (wv)

Comportamiento de baja intensidad, valores entre 0.0 y ~3 m/s.

Algunos picos aislados (máximo ~6 m/s).

Radiación (SWDR, PAR)

Largos tramos en cero (nocturnos).

Subidas bruscas durante el día (no presentes en primeros registros cargados).

Presión atmosférica (p)

Muy estable alrededor de ~1008 hPa.

Tlog (variable objetivo)

Aumenta suavemente con el tiempo según los primeros registros.

Muy correlacionada con temperatura.

## Ranking de variables

En esta sección se presenta un ranking de las variables más importantes para predecir la variable objetivo. Se utilizan técnicas como la correlación, el análisis de componentes principales (PCA) o la importancia de las variables en un modelo de aprendizaje automático.

## Relación entre variables explicativas y variable objetivo

En esta sección se presenta un análisis de la relación entre las variables explicativas y la variable objetivo. Se utilizan gráficos como la matriz de correlación y el diagrama de dispersión para entender mejor la relación entre las variables. Además, se pueden utilizar técnicas como la regresión lineal para modelar la relación entre las variables.
