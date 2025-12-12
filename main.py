"""
Punto de entrada para la API FastAPI.

Carga el preprocesador, selección de features y modelo entrenado para servir
predicciones consistentes con el pipeline de MLOps.
"""

from scripts.api.app import create_app

app = create_app()

# Para correr local:
# uvicorn main:app --reload --port 8000
