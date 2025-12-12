"""
Registro de modelos y especificaciones de hiperparámetros.

Se extrae desde main para hacer el archivo principal más legible y permitir
reutilizar los catálogos desde notebooks o scripts externos.
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass
class ModelSpec:
    """Define el estimador y su espacio de búsqueda."""

    name: str
    search_space: Dict

    def build_estimator(self):
        raise NotImplementedError

    @property
    def has_search(self) -> bool:
        return bool(self.search_space)


class LinearPredictive(ModelSpec):
    def __init__(self):
        super().__init__("linear_predictive", search_space={})

    def build_estimator(self):
        return LinearRegression()


class LinearEconometric(ModelSpec):
    def __init__(self):
        super().__init__("linear_econometric", search_space={})

    def build_estimator(self):
        try:
            import statsmodels.api as sm
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            return LinearRegression()

        class SMWrapper:
            def __init__(self, vif_thresh: float = 30.0, robust_cov: str = "HC3"):
                self.model = None
                self.result = None
                self.vif_thresh = vif_thresh
                self.robust_cov = robust_cov
                self.selected_cols = None
                self.vif_ = None

            def _ensure_df(self, X):
                if isinstance(X, pd.DataFrame):
                    return X.copy()
                X_arr = np.asarray(X)
                cols = [f"feature_{i}" for i in range(X_arr.shape[1])]
                return pd.DataFrame(X_arr, columns=cols)

            def _reduce_vif(self, X_df):
                cols = list(X_df.columns)
                X_mat = X_df.values
                while X_mat.shape[1] > 2:
                    vifs = [variance_inflation_factor(X_mat, i) for i in range(X_mat.shape[1])]
                    max_vif = max(vifs)
                    if max_vif <= self.vif_thresh:
                        break
                    drop_idx = int(np.argmax(vifs))
                    cols.pop(drop_idx)
                    X_mat = X_df[cols].values
                final_vifs = [variance_inflation_factor(X_mat, i) for i in range(X_mat.shape[1])]
                return cols, final_vifs

            def fit(self, X, y):
                X_df = self._ensure_df(X)
                self.selected_cols, self.vif_ = self._reduce_vif(X_df)
                X_sel = X_df[self.selected_cols]

                Xc = sm.add_constant(X_sel, has_constant="add")
                self.model = sm.OLS(y, Xc)
                self.result = self.model.fit(cov_type=self.robust_cov)
                return self

            def predict(self, X):
                if self.result is None:
                    raise NotFittedError("Modelo no ajustado.")
                X_df = self._ensure_df(X)
                if self.selected_cols is not None:
                    missing = set(self.selected_cols) - set(X_df.columns)
                    if missing:
                        raise ValueError(f"Faltan columnas para predicción: {missing}")
                    X_df = X_df[self.selected_cols]
                Xc = sm.add_constant(X_df, has_constant="add")
                return self.result.predict(Xc)

            @property
            def coef_(self):
                if self.result is None:
                    raise NotFittedError("Modelo no ajustado.")
                return self.result.params

            @property
            def summary_(self):
                return self.result.summary().as_text() if self.result else ""

            @property
            def selected_features_(self):
                return self.selected_cols

            @property
            def vif_values_(self):
                return self.vif_

        return SMWrapper()


class LassoModel(ModelSpec):
    def __init__(self):
        space = {"model__alpha": np.logspace(-4, 1, 30)}
        super().__init__("lasso", search_space=space)

    def build_estimator(self):
        return Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=5000))])


class RidgeModel(ModelSpec):
    def __init__(self):
        space = {"model__alpha": np.logspace(-4, 2, 30)}
        super().__init__("ridge", search_space=space)

    def build_estimator(self):
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge())])


class ElasticNetModel(ModelSpec):
    def __init__(self):
        space = {
            "model__alpha": np.logspace(-4, 1, 20),
            "model__l1_ratio": np.linspace(0.1, 0.9, 10),
        }

        super().__init__("elasticnet", search_space=space)

    def build_estimator(self):
        return Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=5000))])


class RandomForestModel(ModelSpec):
    def __init__(self):
        space = {
            "n_estimators": [150, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", 0.7],
        }

        super().__init__("random_forest", search_space=space)

    def build_estimator(self):
        return RandomForestRegressor(n_jobs=-1, random_state=42)


class GradientBoostingModel(ModelSpec):
    def __init__(self):
        space = {
            "n_estimators": [150, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.7, 0.85, 1.0],
        }

        super().__init__("gradient_boosting", search_space=space)

    def build_estimator(self):
        return GradientBoostingRegressor(random_state=42)


class SVRModel(ModelSpec):
    def __init__(self):
        space = {
            "model__C": np.logspace(-2, 3, 10),
            "model__gamma": ["scale", "auto"] + list(np.logspace(-4, -1, 5)),
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],
        }

        super().__init__("svr", search_space=space)

    def build_estimator(self):
        return Pipeline([("scaler", StandardScaler()), ("model", SVR())])


class KNNModel(ModelSpec):
    def __init__(self):
        space = {
            "model__n_neighbors": list(range(3, 31, 2)),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }

        super().__init__("knn", search_space=space)

    def build_estimator(self):
        return Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())])


class MLPModel(ModelSpec):
    def __init__(self):
        space = {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__learning_rate_init": [1e-3, 5e-3, 1e-2],
        }

        super().__init__("mlp", search_space=space)

    def build_estimator(self):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(max_iter=300, random_state=42)),
            ]
        )


def optional_model(name: str, cls):
    try:
        return cls()
    except ImportError as exc:  # pragma: no cover - dependencias opcionales
        print(f"[{name}] Dependencia no instalada: {exc}. Se omite.")
        return None


class XGBoostModel(ModelSpec):
    def __init__(self):
        try:
            from xgboost import XGBRegressor  # noqa
        except ImportError as exc:
            raise ImportError(exc)
        space = {
            "n_estimators": [200, 400, 800],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }

        super().__init__("xgboost", search_space=space)

    def build_estimator(self):
        from xgboost import XGBRegressor

        return XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
        )


class LightGBMModel(ModelSpec):
    def __init__(self):
        try:
            from lightgbm import LGBMRegressor  # noqa
        except ImportError as exc:
            raise ImportError(exc)
        space = {
            "n_estimators": [300, 600, 900],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 63, 127],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }

        super().__init__("lightgbm", search_space=space)

    def build_estimator(self):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=42)


class CatBoostModel(ModelSpec):
    def __init__(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(exc)
        space = {
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "iterations": [300, 600, 900],
            "subsample": [0.7, 0.9, 1.0],
        }

        super().__init__("catboost", search_space=space)

    def build_estimator(self):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(verbose=0, random_state=42, loss_function="RMSE")


def available_specs(include_optional: bool) -> Dict[str, ModelSpec]:
    """Crea el catálogo de modelos disponibles (omitiendo opcionales si no están)."""
    base_specs: List[ModelSpec] = [
        LinearPredictive(),
        LinearEconometric(),
        LassoModel(),
        RidgeModel(),
        ElasticNetModel(),
        RandomForestModel(),
        GradientBoostingModel(),
        SVRModel(),
        KNNModel(),
        MLPModel(),
    ]

    if include_optional:
        for cls, name in [
            (XGBoostModel, "xgboost"),
            (LightGBMModel, "lightgbm"),
            (CatBoostModel, "catboost"),
        ]:
            spec = optional_model(name, cls)
            if spec:
                base_specs.append(spec)

    return {spec.name: spec for spec in base_specs}
