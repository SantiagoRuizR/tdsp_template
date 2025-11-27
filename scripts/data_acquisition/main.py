"""
Script de adquisición de datos: descarga el CSV desde Google Drive y deja un vistazo inicial.
Pensado para usarse al inicio del pipeline de MLOps para asegurar fuentes reproducibles.
"""

import pandas as pd

# Google drive file id of data.
file_id = '1tYfm5wJXRHZGa5h3fsRA7tnyFUlWESpa'
download_url = f'https://drive.google.com/uc?id={file_id}'

# File is read using pandas.
df = pd.read_csv(download_url)

# Check read data by uncommenting next line.
print(df.head(5))
