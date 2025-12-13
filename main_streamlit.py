"""
App Streamlit para realizar predicciones con el modelo entrenado.

Permite ingresar valores manuales o cargar un CSV/Excel con las columnas crudas
del dataset, aplicando el mismo preprocesamiento y selección de features.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from scripts.app.artifacts import InferenceArtifacts, load_artifacts, predict, validate_raw_columns
from scripts.app.schemas import RAW_FEATURES


@st.cache_resource
def get_artifacts(model_path: str, preprocessor_path: str, selection_metadata_path: str) -> InferenceArtifacts:
    return load_artifacts(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        selection_metadata_path=selection_metadata_path,
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Armoniza nombres de columnas para que coincidan con el esquema crudo."""
    rename_map: Dict[str, str] = {
        "max_wv": "max. wv",
        "max PAR": "max. PAR",
        "max_PAR": "max. PAR",
        "max_wv.": "max. wv",
    }
    df = df.rename(columns=rename_map)
    return df


def run_predictions(df: pd.DataFrame, artifacts: InferenceArtifacts) -> pd.DataFrame:
    df = normalize_columns(df)
    missing = validate_raw_columns(df)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    df = df[RAW_FEATURES]
    preds = predict(df, artifacts)
    df_out = df.copy()
    df_out["prediction"] = preds
    return df_out


def render_prediction_summary(preds: np.ndarray):
    if len(preds) == 0:
        st.info("No hay predicciones para mostrar.")
        return
    avg = float(np.mean(preds))
    if avg < 5:
        msg = "Temperaturas muy bajas, abrigo recomendado."
        emoji = "🧊"
    elif avg < 15:
        msg = "Clima fresco. Una chaqueta ligera bastará."
        emoji = "🌤️"
    elif avg < 25:
        msg = "Clima templado y agradable."
        emoji = "😎"
    else:
        msg = "Día caluroso, mantente hidratado."
        emoji = "🔥"
    st.subheader(f"{emoji} Pronóstico promedio: {avg:.2f}")
    st.write(msg)


def manual_input_form() -> pd.DataFrame:
    st.subheader("Ingreso manual de variables")
    defaults = {
        "p": 1010.0,
        "T": 12.0,
        "Tpot": 285.0,
        "Tdew": 8.0,
        "rh": 80.0,
        "VPmax": 14.0,
        "VPact": 11.0,
        "VPdef": 3.0,
        "sh": 8.0,
        "H2OC": 12.0,
        "rho": 1200.0,
        "wv": 2.0,
        "max. wv": 4.0,
        "wd": 180.0,
        "rain": 0.0,
        "raining": 0.0,
        "SWDR": 0.0,
        "PAR": 0.0,
        "max. PAR": 0.0}
    
    cols = st.columns(3)
    values = {}
    for idx, feature in enumerate(RAW_FEATURES):
        if feature == "date":
            values["date"] = st.date_input("date (opcional)", None)
            continue
        col = cols[idx % 3]
        values[feature] = col.number_input(feature, value=float(defaults.get(feature, 0.0)))
    df = pd.DataFrame([values])
    return df


def main():
    st.title("Weather TDSP - Prediccion (Streamlit)")
    st.write(
        "App para consumir el modelo entrenado con el mismo preprocesamiento y "
        "seleccion de features del pipeline TDSP.")

    models_dir = Path("models_fast")
    model_options = sorted([p.name for p in models_dir.glob("*.joblib")])
    model_choice = st.sidebar.selectbox("Modelo", model_options, index=0 if model_options else None)
    model_path = str(models_dir / model_choice) if model_choice else ""

    preprocessor_path = "data/processed/preprocessor.joblib"
    selection_metadata_path = "data/selected/metadata_selection.json"

    if not model_path:
        st.error("No se encontraron modelos en models_fast/.")
        return

    artifacts = get_artifacts(model_path, preprocessor_path, selection_metadata_path)
    st.sidebar.markdown(f"**Modelo cargado:** {artifacts.model_name}")
    st.sidebar.markdown(f"**Features seleccionadas:** {len(artifacts.selected_features or artifacts.feature_names)}")

    tab_manual, tab_file = st.tabs(["Ingreso manual", "Carga de archivo"])

    with tab_manual:
        df_input = manual_input_form()
        if st.button("Predecir (manual)"):
            try:
                df_pred = run_predictions(df_input, artifacts)
                st.success("Prediccion generada")
                render_prediction_summary(df_pred["prediction"].values)
                st.dataframe(df_pred)
            except Exception as exc:
                st.error(f"Error en la prediccion: {exc}")

    with tab_file:
        st.subheader("Carga un CSV o Excel")
        file = st.file_uploader("Archivo", type=["csv", "xlsx"])
        if file:
            try:
                if file.name.endswith(".csv"):
                    df_file = pd.read_csv(file)
                else:
                    df_file = pd.read_excel(file)
                st.write("Preview:")
                st.dataframe(df_file.head())
                if st.button("Predecir (archivo)"):
                    df_pred = run_predictions(df_file, artifacts)
                    st.success(f"Predicciones generadas: {len(df_pred)} filas")
                    render_prediction_summary(df_pred["prediction"].values)
                    st.dataframe(df_pred.head())
                    csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                    st.download_button("Descargar resultados CSV", data=csv_bytes, file_name="predicciones.csv")
            except Exception as exc:
                st.error(f"Error al procesar el archivo: {exc}")


if __name__ == "__main__":
    main()
