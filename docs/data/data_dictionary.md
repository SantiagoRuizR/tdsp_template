# Diccionario de datos

## Base de datos

Registro de 19 indicadores meteorológicos medidos en el Instituto Meteorológico Max Planck de Munich, Alemania, durante el año 2020. Los registros fueron tomados cada 10 minutos durante todos los días del año.

| Variable | Descripción | Tipo de dato | Rango/Valores posibles | Fuente de datos |
| --- | --- | --- | --- | --- |
| date | Fecha y hora de la observación. | Datetime | [2020-01-01 , 2020-12-31] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| p | Presión atmosférica en milibares (mbar). | Float | [955.58 , 1020.07] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| T | Temperatura del aire en grados Celsius (°C). | Float | [-6.44 , 34.8] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| Tpot | Temperatura potencial en Kelvin (K); representa la temperatura que tendría una parcela de aire si se moviera a un nivel de presión estándar. | Float | [266.19 , 309.13] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| Tdew | Temperatura del punto de rocío en grados Celsius (°C); indica la temperatura a la cual el aire se satura de humedad. | Float | [-13.81 , 20.5] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| rh | Humedad relativa como porcentaje (%); muestra la cantidad de humedad en el aire en relación con el máximo que puede retener a esa temperatura. | Float | [21.16 , 100.0] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| VPmax | Presión de vapor máxima en milibares (mbar); representa la presión máxima ejercida por el vapor de agua a la temperatura dada. | Float | [3.77 , 55.67] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| VPact | Presión de vapor real en milibares (mbar); indica la presión de vapor de agua actual en el aire. | Float | [2.09 , 24.16] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| VPdef | Déficit de presión de vapor en milibares (mbar); mide la diferencia entre la presión de vapor máxima y la real, utilizado para estimar el potencial de secado. | Float | [0.0 , 42.1] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| sh | Humedad específica en gramos por kilogramo (g/kg); muestra la masa de vapor de agua por kilogramo de aire. | Float | [1.3 , 15.4] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| H2OC | Concentración de vapor de agua en milimoles por mol (mmol/mol) de aire seco. | Float | [2.09 , 24.53] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| rho | Densidad del aire en gramos por metro cúbico (g/m³); refleja la masa de aire por unidad de volumen. | Float | [1107.38 , 1318.52] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| wv | Velocidad del viento en metros por segundo (m/s); mide el movimiento horizontal del aire. | Float | [-9999.0 , 13.77] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| max. wv | Velocidad máxima del viento en metros por segundo (m/s); indica la velocidad de viento más alta registrada durante el periodo. | Float | [0.0 , 22.9] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| wd | Dirección del viento en grados (°); representa la dirección desde la cual sopla el viento. | Float | [0.0 , 360.0] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| rain | Lluvia total en milímetros (mm); muestra la cantidad de precipitación durante el periodo de observación. | Float | [0.0 , 11.2] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| raining | Duración de la lluvia en segundos (s); registra el tiempo durante el cual llovió en el periodo de observación. | Float | [0.0 , 600.0] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| SWDR | Radiación de onda corta descendente en vatios por metro cuadrado (W/m²); mide la radiación solar entrante. | Float | [0.0 , 1115.29] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| PAR | Radiación fotosintéticamente activa en micromoles por metro cuadrado por segundo (µmol/m²/s); indica la cantidad de luz disponible para la fotosíntesis. | Float | [0.0 , 2131.76] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| max. PAR | Máxima radiación fotosintéticamente activa registrada en el periodo de observación en µmol/m²/s. | Float | [-9999.0 , 2498.94] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  
| Tlog | Temperatura registrada en grados Celsius (°C), potencialmente proveniente de un sensor o registrador secundario. | Float | [6.9 , 49.09] | https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting?resource=download |  


- **Variable**: nombre de la variable.
- **Descripción**: breve descripción de la variable.
- **Tipo de dato**: tipo de dato que contiene la variable.
- **Rango/Valores posibles**: rango o valores que puede tomar la variable.
- **Fuente de datos**: fuente de los datos de la variable.
