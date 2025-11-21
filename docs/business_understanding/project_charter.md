# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto

Pronóstico del clima de Munich básado en indicadores meteorológicos.

## Objetivo del Proyecto

Desarrollar modelos predictivos de temperatura basados en Machine Learning utilizando datos meteorológicos de alta resolución, y así, identificar patrones microclimáticos locales.

## Alcance del Proyecto

### Incluye:

- Se tienen datos históricos de indicadores meteorológicos de 2020 tomados cada 10 minutos. 
- Obtener datos de temperatura coherentes con los valores medidos.
- Se busca pronosticar adecuadamente la temperatura de los días posteriores basado en información de los días o semanas previas.

### Excluye:

- Información anterior o posterior al 2020.

## Metodología

Se empleará un enfoque secuencial de Deep Learning. Para ello se comparará entre las arquitecturas de RNN (Redes Neuronales Recurrentes), LSTM (Long Short-Term Memory), GRU y Time-Series Transformers, para los modelos de los cuales se parte desde cero, y se comparará con los resultados generados por un modelos preentrenado cómo Prophet. Estos serán evaluados bajo críterios como Walk-Fordward Validation para un correcto entendimiento de las series de tiempo.

## Cronograma

| Etapa | Duración Estimada | Fechas |
|------|---------|-------|
| Entendimiento del negocio y carga de datos | 1 semana | del 17 de noviembre al 23 de noviembre |
| Preprocesamiento, análisis exploratorio | 1 semana | del 24 de noviembre al 30 de noviembre |
| Modelamiento y extracción de características | 1 semana | del 1 de diciembre al 7 de diciembre |
| Despliegue | 1 semana | del 8 de diciembre al 13 de diciembre |
| Evaluación y entrega final | 1 semana | del 8 de diciembre al 13 de diciembre |

Hay que tener en cuenta que estas fechas son de ejemplo, estas deben ajustarse de acuerdo al proyecto.

## Equipo del Proyecto

- Santiago Ruiz Rozo <sruiz899@gmail.com>
- Pablo Alejandro Reyes Granados <alejogranados229@gmail.com>
- Kevin Andrés Martínez Martínez <kevinmartinez.ingbiom@gmail.com>

## Presupuesto

N/A

## Stakeholders

- Jorge E. Camargo, PhD
- Profesor del modulo de **Metodologías Ágiles para el Desarrollo de Proyectos con Machine Learning**

## Aprobaciones

- Jorge E. Camargo, PhD

