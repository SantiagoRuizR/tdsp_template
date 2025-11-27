# Reporte de Datos

Este documento presenta los resultados del análisis exploratorio de datos (EDA) realizado sobre el conjunto de mediciones meteorológicas tomadas cada 10 minutos durante el año 2020.

---

## 1. Resumen General del Dataset

### 1.1 Dimensiones del conjunto de datos
- **Número total de filas:** 52,696  
- **Número total de columnas:** 21  

### 1.2 Tipos de variables
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

### 1.3 Valores faltantes
- **No se encontraron valores faltantes** en ninguna variable.

---

## 2. Calidad de los Datos

Esta sección evalúa la integridad y consistencia general del dataset.

### 2.1 Valores faltantes
- No se identificaron valores nulos.

### 2.2 Valores duplicados
- No se encontraron duplicados relevantes.

### 2.3 Valores extremos
- Algunas variables presentan valores atípicos esperados:
  - Picos aislados en velocidad del viento.
  - Incrementos abruptos en la radiación durante el día.

### 2.4 Consistencia general
El dataset muestra:
- Estabilidad en presión y humedad.
- Coherencia temporal en las mediciones.

---

## 3. Variable Objetivo

### 3.1 Descripción
La variable objetivo es:

- **`Tlog`** — Transformación logarítmica de una magnitud térmica relacionada con la temperatura.

### 3.2 Distribución
- Tendencia creciente y progresiva a lo largo del tiempo.
- No presenta valores extremos significativos.
- Fuerte correlación con variables térmicas (`T`, `Tdew`, `Tpot`).

---

## 4. Análisis de Variables Individuales

### 4.1 Temperatura (`T`)
- Rango aproximado: **–2°C a 20°C**.
- Distribución **unimodal**.
- Alta correlación con `Tpot`, `Tdew` y `Tlog`.

### 4.2 Humedad relativa (`rh`)
- Valores entre **70% y 95%**.
- Comportamiento estable, típico de un clima húmedo.

### 4.3 Velocidad del viento (`wv`)
- Valores típicos entre **0.0 y ~3 m/s**.
- Picos aislados de hasta **~6 m/s**.

### 4.4 Radiación solar (`SWDR`, `PAR`)
- Tramos prolongados en cero durante la noche.
- Aumentos marcados durante el día.

### 4.5 Presión atmosférica (`p`)
- Estabilidad alrededor de **1008 hPa**.
- Sin variaciones bruscas.

### 4.6 Variable objetivo (`Tlog`)
- Comportamiento suavemente creciente en el tiempo.
- Alta correlación con la temperatura.

---

## 5. Ranking de Importancia de Variables

Basado en correlación y patrones lineales iniciales:

1. **T (Temperatura)**
2. **Tdew (Temperatura del punto de rocío)**
3. **Tpot (Temperatura potencial)**
4. **rh (Humedad relativa)**
5. **VPact (Presión parcial de vapor)**
6. **sh (Razón de mezcla)**
7. **rho (Densidad del aire)**

Variables con menor influencia directa sobre `Tlog`:
- Radiación solar (`SWDR`, `PAR`)
- Velocidad del viento (`wv`)

---

## 6. Relación entre Variables Explicativas y la Variable Objetivo

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
