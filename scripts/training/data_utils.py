"""
Funciones auxiliares de carga y muestreo de datos para entrenamiento.
"""

import os
from typing import Tuple

import pandas as pd


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga conjuntos seleccionados; si no existen, usa los preprocesados completos."""
    sel_train = os.path.join(data_dir, "X_train_selected.csv")
    sel_test = os.path.join(data_dir, "X_test_selected.csv")
    base_dir = data_dir
    if not os.path.exists(sel_train):
        sel_train = os.path.join(data_dir, "X_train.csv")
        sel_test = os.path.join(data_dir, "X_test.csv")
    X_train = pd.read_csv(sel_train, index_col=0)
    X_test = pd.read_csv(sel_test, index_col=0)
    y_train = pd.read_csv(os.path.join(base_dir, "y_train.csv"), index_col=0).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(base_dir, "y_test.csv"), index_col=0).iloc[:, 0]
    return X_train, X_test, y_train, y_test


def build_fast_subset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_train: int = 5000,
    max_test: int = 2000,
    random_state: int = 77,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Replica el muestreo rapido usado en `notebooks/showcase_modeling.ipynb`.
    """
    n_train = min(len(X_train), max_train)
    n_test = min(len(X_test), max_test)
    idx_train = X_train.sample(n=n_train, random_state=random_state).index
    idx_test = X_test.sample(n=n_test, random_state=random_state).index

    X_train_small = X_train.loc[idx_train]
    y_train_small = y_train.loc[idx_train]
    X_test_small = X_test.loc[idx_test]
    y_test_small = y_test.loc[idx_test]
    return X_train_small, y_train_small, X_test_small, y_test_small
