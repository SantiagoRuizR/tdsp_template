"""
Pipeline de preprocesamiento listo para entrenamiento de modelos.

Incluye:
- Carga de datos desde ruta local o Google Drive.
- Detección automática de variables numéricas y categóricas.
- Imputación de nulos, winsorización, escalado y codificación one-hot.
- Split de entrenamiento/prueba y opción de generar características polinómicas.
- Persistencia de datasets procesados y del preprocesador.
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")


# ------------------------- UTILIDADES ------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta valores extremos por cuantiles para reducir el impacto de outliers sin eliminarlos."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower_quantile, axis=0)
        self.upper_ = np.nanquantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        lower = np.broadcast_to(self.lower_, X.shape)
        upper = np.broadcast_to(self.upper_, X.shape)
        return np.clip(X, lower, upper)


@dataclass
class PreprocessArtifacts:
    """Estructura contenedora de los datasets procesados y el preprocesador ajustado."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer


def load_dataset(input_path: str, gdrive_id: Optional[str]):
    """Carga el CSV desde disco y, si falta, intenta descargarlo desde Google Drive con el ID dado."""
    if os.path.exists(input_path):
        return pd.read_csv(input_path)

    if gdrive_id:
        drive_url = f"https://drive.google.com/uc?id={gdrive_id}"
        try:
            return pd.read_csv(drive_url)
        except Exception as exc:  # pragma: no cover - manejo interactivo
            raise FileNotFoundError(
                f"No se pudo leer {input_path} y falló la descarga desde Drive ({exc})."
            )
    raise FileNotFoundError(f"No se encontró el archivo {input_path}.")


def detect_column_types(df: pd.DataFrame, target: str):
    """Identifica columnas numéricas, categóricas y datetime para el set de features excluyendo la variable objetivo."""
    datetime_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    features = [c for c in df.columns if c != target]
    numeric_cols = [c for c in numeric_cols if c in features and c not in datetime_cols]
    cat_cols = [c for c in cat_cols if c in features and c not in datetime_cols]
    datetime_cols = [c for c in datetime_cols if c in features]

    return numeric_cols, cat_cols, datetime_cols


def build_preprocessor(
    numeric_cols: Sequence[str],
    cat_cols: Sequence[str],
    poly_degree: Optional[int],
    winsor_limits: Tuple[float, float],):
    """
    Arma el ColumnTransformer con imputación, winsorización y escalado para numéricos,
    One-Hot-Encoding para categóricos y, opcionalmente, expansión polinómica.
    """
    transformers = []

    if numeric_cols:
        num_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer(lower_quantile=winsor_limits[0], upper_quantile=winsor_limits[1])),]
        
        if poly_degree and poly_degree > 1:
            num_steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))

        num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=num_steps), list(numeric_cols)))

    if cat_cols:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),]
        
        transformers.append(("cat", Pipeline(steps=cat_steps), list(cat_cols)))

    if not transformers:
        raise ValueError("No se detectaron columnas para preprocesar.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def preprocess_dataset(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
    poly_degree: Optional[int],
    winsor_limits: Tuple[float, float],
    drop_datetime: bool,):
    """
    Divide en train/test, ajusta el preprocesador a los datos de entrenamiento y
    devuelve los conjuntos transformados junto con el objeto de preprocesamiento.
    """

    if target not in df.columns:
        raise KeyError(f"La columna objetivo '{target}' no está en el dataset.")

    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    y = df[target]
    X = df.drop(columns=[target])

    numeric_cols, cat_cols, datetime_cols = detect_column_types(df, target)
    if drop_datetime and datetime_cols:
        X = X.drop(columns=datetime_cols)
        numeric_cols = [c for c in numeric_cols if c not in datetime_cols]
        cat_cols = [c for c in cat_cols if c not in datetime_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None)

    preprocessor = build_preprocessor(numeric_cols, cat_cols, poly_degree, winsor_limits)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)

    return PreprocessArtifacts(X_train_df, X_test_df, y_train, y_test, preprocessor)


def save_outputs(artifacts: PreprocessArtifacts, output_dir: str) -> None:
    """Persiste datasets procesados, el preprocesador serializado y metadatos descriptivos en disco."""
    os.makedirs(output_dir, exist_ok=True)

    artifacts.X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=True)
    artifacts.X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=True)
    artifacts.y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=True, header=True)
    artifacts.y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=True, header=True)

    dump(artifacts.preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    meta = {
        "n_train": len(artifacts.X_train),
        "n_test": len(artifacts.X_test),
        "n_features": artifacts.X_train.shape[1],
        "feature_names": artifacts.X_train.columns.tolist(),}
    
    pd.Series(meta).to_json(os.path.join(output_dir, "metadata.json"), force_ascii=False, indent=2)


def parse_args():
    """Define y parsea los argumentos CLI que controlan las opciones de preprocesamiento."""
    parser = argparse.ArgumentParser(description="Preprocesamiento robusto para modelado.")
    parser.add_argument("--input", default="data.csv", help="Ruta al CSV de entrada.")
    parser.add_argument("--gdrive-id", default="1tYfm5wJXRHZGa5h3fsRA7tnyFUlWESpa", help="ID de Google Drive como respaldo.")
    parser.add_argument("--target", default="Tlog", help="Nombre de la variable dependiente.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción para el conjunto de prueba.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument("--poly-degree", type=int, default=None, help="Grado para PolynomialFeatures (>=2 activa la expansión).")
    parser.add_argument("--winsor-lower", type=float, default=0.01, help="Cuantil inferior para winsorización.")
    parser.add_argument("--winsor-upper", type=float, default=0.99, help="Cuantil superior para winsorización.")
    parser.add_argument("--drop-datetime", action="store_true", help="Descartar columnas datetime antes de entrenar.")
    parser.add_argument("--output-dir", default="data/processed", help="Directorio para guardar los artefactos.")
    return parser.parse_args()


def main():
    """Orquesta carga de datos, preprocesamiento parametrizado y guardado de artefactos listos para modelar."""
    args = parse_args()

    df = load_dataset(args.input, args.gdrive_id)
    print(f"Dataset cargado con forma {df.shape}. Columnas: {list(df.columns)}")

    winsor_limits = (args.winsor_lower, args.winsor_upper)
    artifacts = preprocess_dataset(
        df=df,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        poly_degree=args.poly_degree,
        winsor_limits=winsor_limits,
        drop_datetime=args.drop_datetime,)
    
    save_outputs(artifacts, args.output_dir)

    print(
        f"Preprocesamiento finalizado. Train: {artifacts.X_train.shape}, "
        f"Test: {artifacts.X_test.shape}. Artefactos guardados en {args.output_dir}.")


if __name__ == "__main__":
    main()
