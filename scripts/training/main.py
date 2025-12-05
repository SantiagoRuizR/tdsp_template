"""
Entrenamiento y búsqueda de hiperparámetros sobre los datos seleccionados.

Flujo:
- Carga `X_train_selected/X_test_selected/y_*` desde `data/selected` (o `data/processed` si no hay selección).
- Define modelos como clases con su espacio de hiperparámetros.
- Ejecuta RandomizedSearchCV (maximizando R2) o ajuste directo según corresponda.
- Reporta R2, MSE y MAE en validación y prueba; guarda resultados en CSV/JSON.

Nota: dependencias opcionales (xgboost, lightgbm, catboost, statsmodels) se omiten si no están instaladas.
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


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
            "model__l1_ratio": np.linspace(0.1, 0.9, 10),}
        
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
            "max_features": ["auto", "sqrt", 0.7],}
        
        super().__init__("random_forest", search_space=space)

    def build_estimator(self):
        return RandomForestRegressor(n_jobs=-1, random_state=42)


class GradientBoostingModel(ModelSpec):
    def __init__(self):
        space = {
            "n_estimators": [150, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.7, 0.85, 1.0],}
        
        super().__init__("gradient_boosting", search_space=space)

    def build_estimator(self):
        return GradientBoostingRegressor(random_state=42)


class SVRModel(ModelSpec):
    def __init__(self):
        space = {
            "model__C": np.logspace(-2, 3, 10),
            "model__gamma": ["scale", "auto"] + list(np.logspace(-4, -1, 5)),
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],}
        
        super().__init__("svr", search_space=space)

    def build_estimator(self):
        return Pipeline([("scaler", StandardScaler()), ("model", SVR())])


class KNNModel(ModelSpec):
    def __init__(self):
        space = {
            "model__n_neighbors": list(range(3, 31, 2)),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],}
        
        super().__init__("knn", search_space=space)

    def build_estimator(self):
        return Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())])


class MLPModel(ModelSpec):
    def __init__(self):
        space = {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__learning_rate_init": [1e-3, 5e-3, 1e-2],}
        
        super().__init__("mlp", search_space=space)

    def build_estimator(self):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(max_iter=300, random_state=42)),
            ])


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
            "colsample_bytree": [0.6, 0.8, 1.0],}
        
        super().__init__("xgboost", search_space=space)

    def build_estimator(self):
        from xgboost import XGBRegressor

        return XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,)


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
            "colsample_bytree": [0.6, 0.8, 1.0],}
        
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
            "subsample": [0.7, 0.9, 1.0],}
        
        super().__init__("catboost", search_space=space)

    def build_estimator(self):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(verbose=0, random_state=42, loss_function="RMSE")


def load_data(data_dir: str):
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


def evaluate(estimator, X_train, y_train, X_test, y_test, val_score: Optional[float]):
    """Calcula métricas en train/test; usa val_score si viene de CV."""
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    metrics = {
        "val_r2": val_score,
        "test_r2": r2_score(y_test, preds),
        "test_mse": mean_squared_error(y_test, preds),
        "test_mae": mean_absolute_error(y_test, preds),}
    
    return metrics, preds


def run_model_search(
    specs: List[ModelSpec],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int,
    cv: int,
    random_state: int,
    output_dir: str,):

    """Ejecuta búsqueda o ajuste directo y guarda resultados."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for spec in specs:
        print(f"\n>>> Modelo: {spec.name}")
        estimator = spec.build_estimator()

        if spec.has_search:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=spec.search_space,
                n_iter=n_iter,
                scoring="r2",
                cv=cv,
                random_state=random_state,
                n_jobs=-1,)
            
            search.fit(X_train, y_train)
            best = search.best_estimator_
            val_score = search.best_score_
        else:
            val_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
            val_score = float(np.mean(val_scores))
            best = estimator

        metrics, _ = evaluate(best, X_train, y_train, X_test, y_test, val_score)
        results.append(
            {
                "model": spec.name,
                "val_r2": metrics["val_r2"],
                "test_r2": metrics["test_r2"],
                "test_mse": metrics["test_mse"],
                "test_mae": metrics["test_mae"],})

        # Guardar modelo entrenado cuando sea pequeño/moderno (se omiten wrappers sin dump).
        model_path = os.path.join(output_dir, f"{spec.name}.joblib")
        try:
            import joblib

            joblib.dump(best, model_path)
        except Exception as exc:  # pragma: no cover - depende del modelo
            print(f"No se guardó el modelo {spec.name}: {exc}")

    pd.DataFrame(results).to_csv(os.path.join(output_dir, "training_report.csv"), index=False)
    return results


def available_specs(include_optional: bool) -> Dict[str, ModelSpec]:
    """Crea el catálogo de modelos disponibles (omitiendo opcionales si no están)."""
    base_specs = [
        LinearPredictive(),
        LinearEconometric(),
        LassoModel(),
        RidgeModel(),
        ElasticNetModel(),
        RandomForestModel(),
        GradientBoostingModel(),
        SVRModel(),
        KNNModel(),
        MLPModel(),]
    
    if include_optional:
        for cls, name in [
            (XGBoostModel, "xgboost"),
            (LightGBMModel, "lightgbm"),
            (CatBoostModel, "catboost"),]:

            spec = optional_model(name, cls)
            if spec:
                base_specs.append(spec)
    return {spec.name: spec for spec in base_specs}


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento con búsqueda de hiperparámetros (R2 como métrica).")
    parser.add_argument("--data-dir", default="data/selected", help="Ruta a los datos seleccionados/preprocesados.")
    parser.add_argument("--output-dir", default="models", help="Ruta donde guardar reportes y modelos.")
    parser.add_argument("--models", nargs="+", default=["lasso", "ridge", "elasticnet", "random_forest", "gradient_boosting", "svr", "knn", "mlp", "linear_predictive", "linear_econometric"], help="Lista de modelos a ejecutar.")
    parser.add_argument("--n-iter", type=int, default=25, help="Iteraciones para RandomizedSearchCV.")
    parser.add_argument("--cv", type=int, default=5, help="Número de folds de validación.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument("--include-optional", action="store_true", help="Intentar incluir xgboost/lightgbm/catboost si están instalados.")
    return parser.parse_args()


def main():
    args = parse_args()
    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    catalog = available_specs(include_optional=args.include_optional)

    selected_specs = []
    for name in args.models:
        if name not in catalog:
            print(f"Modelo '{name}' no disponible; se omite.")
            continue
        selected_specs.append(catalog[name])

    if not selected_specs:
        raise ValueError("No se seleccionaron modelos válidos.")

    results = run_model_search(
        specs=selected_specs,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_iter=args.n_iter,
        cv=args.cv,
        random_state=args.random_state,
        output_dir=args.output_dir,)

    print("\nReporte de entrenamiento:")
    for r in results:
        print(
            f"{r['model']}: val_r2={r['val_r2']:.3f}, test_r2={r['test_r2']:.3f}, "
            f"test_mse={r['test_mse']:.4f}, test_mae={r['test_mae']:.4f}")


if __name__ == "__main__":
    main()
