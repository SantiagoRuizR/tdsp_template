# Reporte de Datos

Este documento presenta los resultados del análisis exploratorio de datos (EDA) realizado sobre el conjunto de mediciones meteorológicas tomadas cada 10 minutos durante el año 2020.

---

## Resumen General del Dataset

###  Dimensiones del conjunto de datos
- **Número total de filas:** 52,696  
- **Número total de columnas:** 21  

### Tipos de variables
- **Variable temporal**  
  - `date`: cadena originalmente, convertida a tipo datetime.
- **Variables numéricas continuas**  
  Temperatura, humedad, presión atmosférica, viento, radiación, entre otras.
- **Variables binarias**  
  - `raining`
- **Variables derivadas**  
  - `Tpot`
  - `VPdef`
  - `Tlog` (variable objetivo)

###  Valores faltantes
- **No se encontraron valores faltantes** en ninguna variable.

---

## Calidad de los Datos

Esta sección evalúa la integridad y consistencia general del dataset.

### Valores faltantes
- No se identificaron valores nulos.

### Valores duplicados
- No se encontraron duplicados relevantes.

### Valores extremos
- Algunas variables presentan valores atípicos esperados:
  - Picos aislados en velocidad del viento.
  - Incrementos abruptos en la radiación durante el día.

### Consistencia general
El dataset muestra:
- Estabilidad en presión y humedad.
- Coherencia temporal en las mediciones.

---

## Variable Objetivo

### Descripción
La variable objetivo es:

- **`Tlog`** — Transformación logarítmica de una magnitud térmica relacionada con la temperatura.

### Distribución
- Tendencia creciente y progresiva a lo largo del tiempo.
- No presenta valores extremos significativos.
- Fuerte correlación con variables térmicas (`T`, `Tdew`, `Tpot`).

---

## Análisis de Variables Individuales

### Temperatura (`T`)
- Rango aproximado: **–2°C a 20°C**.
- Distribución **unimodal**.
- Alta correlación con `Tpot`, `Tdew` y `Tlog`.

### Humedad relativa (`rh`)
- Valores entre **70% y 95%**.
- Comportamiento estable, típico de un clima húmedo.

### Velocidad del viento (`wv`)
- Valores típicos entre **0.0 y ~3 m/s**.
- Picos aislados de hasta **~6 m/s**.

### Radiación solar (`SWDR`, `PAR`)
- Tramos prolongados en cero durante la noche.
- Aumentos marcados durante el día.

### Presión atmosférica (`p`)
- Estabilidad alrededor de **1008 hPa**.
- Sin variaciones bruscas.

### Variable objetivo (`Tlog`)
- Comportamiento suavemente creciente en el tiempo.
- Alta correlación con la temperatura.

---

## Ranking de Importancia de Variables

Basado en correlación y patrones lineales iniciales:

1. **T (Temperatura)**
2. **Tdew (humedad absoluta ligada a temperatura)**
3. **Tpot (Temperatura potencial)**
4. **rh (Humedad relativa)**
5. **VPact (Presión parcial de vapor)**
6. **sh (Razón de mezcla)**
7. **rho (Densidad del aire)**

Variables con menor influencia directa sobre `Tlog`:
- Radiación solar (`SWDR`, `PAR`)
- Velocidad del viento (`wv`)

---

## Relación entre Variables Explicativas y la Variable Objetivo

El análisis revela que:

- La **matriz de correlación** muestra fuerte dependencia entre `Tlog` y las variables térmicas.
- Los **diagramas de dispersión** indican relaciones casi lineales con la temperatura.
- Un modelo como la **regresión lineal** puede representar eficazmente estas relaciones.
- La estructura temporal del dataset sugiere una influencia marcada del **ciclo anual** en las variables térmicas.

---

Las relaciones más fuertes observadas son:

Variable	Correlación con Tlog	Interpretación
T	Alta	Directa, casi lineal
Tdew	Alta	Ligada al contenido de vapor
Tpot	Alta	Transformación termodinámica relacionada
rh	Moderada	Humedad ligada a formación térmica
VPact	Alta	Presión de vapor depende de temperatura
