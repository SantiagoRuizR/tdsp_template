# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:**  
  `prophet_model`

- **Plataforma de despliegue:**  
  Contenedor Docker ejecutado en un servicio de orquestación con una API REST en FastAPI.

- **Requisitos técnicos:**  
  - Python 3.12  
  - Librerías:
    - `prophet` (Meta Prophet)
    - `fastapi`
    - `uvicorn[standard]`
    - `pandas`
    - `numpy`
    - `pydantic` (incluida con FastAPI)  
  - Sistema operativo Linux (Ubuntu 22.04 LTS o similar)  
  - Recursos recomendados:  
    - CPU: 1 vCPU mínimo  
    - RAM: 1–2 GB (dependiendo del tamaño del modelo y horizonte de predicción)  
  - Almacenamiento:  
    - Carpeta persistente para `models_fast/prophet.json` y logs.

- **Requisitos de seguridad:**  
  - Se puede agregar un autenticación mediante token de API o JWT en los endpoints de predicción
  - Comunicación cifrada vía HTTPS (terminación TLS en el load balancer / gateway).  
  - Restricción de acceso por red (IP allowlist o VNet) si se despliega en cloud privado.  

## Código de despliegue

- **Archivo principal:**  
  `main.py`  
  Contiene:
  - Carga del modelo Prophet desde `models_fast/prophet.json`.  
  - Definición de la aplicación FastAPI.  
  - Rutas `/` (salud) y `/forecast` (predicción).

- **Rutas de acceso a los archivos:**  
  - `models_fast/prophet_model.json` – modelo entrenado y serializado con `model_to_json`.  
  - `app/main.py` – código de la API (si usas estructura de paquete).  
  - `app/requirements.txt` – lista de dependencias de Python.  
  - `Dockerfile` – definición de la imagen de contenedor.  

- **Variables de entorno:**  
  - `MODEL_PATH` – ruta al archivo del modelo (por defecto `models_fast/prophet_model.json`).  
  - `API_HOST` – host de escucha de Uvicorn (por defecto `0.0.0.0`).  
  - `API_PORT` – puerto de la API (por defecto `8000`).  
  - `API_TOKEN` – token de autenticación para llamadas a `/forecast` (si se implementa seguridad básica).  
  - `TZ` – zona horaria (`Europe/Berlin`) para asegurar coherencia temporal.

## Documentación del despliegue

- **Instrucciones de instalación:**  
  1. Clonar el repositorio del proyecto.  
  2. Crear y activar un entorno virtual de Python (`python -m venv .venv; source .venv/bin/activate`).  
  3. Instalar dependencias: `pip install -r requirements.txt`.  
  4. Copiar el archivo de modelo entrenado a `models_fast/prophet.json`.  
  5. Opcional: construir la imagen Docker:  
     ```bash
     docker build -t prophet-weather-api .
     ```

- **Instrucciones de configuración:**  
  1. Definir las variables de entorno (`MODEL_PATH`, `API_PORT`, `API_TOKEN`, etc.) en un archivo `.env` o en la configuración del contenedor/servicio.  
  2. Verificar que la zona horaria y el formato de fechas que espera el modelo coinciden con los datos históricos usados para entrenamiento (por ejemplo, `ds` en UTC o `Europe/Berlin`).  
  3. Configurar el balanceador / gateway para exponer el puerto interno 8000 como HTTPS hacia el exterior.  

- **Instrucciones de uso:**  
  1. Levantar la API en local:  
     ```bash
     uvicorn main:app --host 0.0.0.0 --port 8000
     ```
  2. Comprobar salud:  
     ```bash
     curl http://localhost:8000/
     ```
     Respuesta esperada: `{"message": "API Prophet funcionando"}`.  
  3. Solicitar un pronóstico (ejemplo 48 horas adelante):  
     ```bash
     curl -X POST "http://localhost:8000/forecast" \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer <API_TOKEN>" \
          -d '{"periods": 48, "freq": "H"}'
     ```
     La API devolverá un JSON con una lista de objetos `{ds, yhat, yhat_lower, yhat_upper}` para cada punto futuro.  

- **Instrucciones de mantenimiento:**  
  - **Monitoreo:**  
    - Registrar logs de acceso y errores (stdout/stderr del contenedor) y enviarlos a un sistema centralizado (CloudWatch, Log Analytics, etc.).  
    - Monitorizar latencia de respuesta, tasa de errores y consumo de CPU/RAM.  
  - **Re-entrenamiento:**  
    - Establecer un proceso periódico (por ejemplo, mensual) para re-entrenar el modelo Prophet con nuevos datos de la ciudad en Alemania, generar un nuevo `prophet.json` y desplegarlo (blue/green o rolling).  
  - **Actualizaciones:**  
    - Probar nuevas versiones de `prophet` y `fastapi` en un entorno de staging antes de producción.  
    - Documentar cambios de versión de modelo (`prophet_weather_germany_v2`, etc.).  
  - **Backups:**  
    - Mantener copias de seguridad del archivo de modelo y de los datos históricos usados para entrenamiento, para trazabilidad y reproducibilidad.