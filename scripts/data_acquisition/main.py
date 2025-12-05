"""
Script de adquisición de datos: descarga el CSV desde Google Drive y deja un vistazo inicial.
Pensado para usarse al inicio del pipeline de MLOps para asegurar fuentes reproducibles.
"""

import pandas as pd

# Google drive file id of data.
file_id = '1tYfm5wJXRHZGa5h3fsRA7tnyFUlWESpa'
download_url = f'https://drive.google.com/uc?id={file_id}'

# File is read using pandas.


def load_dataset(input_path=download_url, gdrive_id=file_id):
    """
    Carga el dataset desde una ruta local o, si no está, desde Google Drive.
    Retorna un DataFrame con los datos.
    """
    try:
        return pd.read_csv(input_path)
    except Exception:
        if gdrive_id:
            fallback_url = f"https://drive.google.com/uc?id={gdrive_id}"
            return pd.read_csv(fallback_url)
        raise FileNotFoundError(f"No se pudo cargar el dataset desde {input_path}.")


df = load_dataset()

# Check read data by uncommenting next line.
print(df.head(5))
