from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    numeric_features: list[str]
    categorical_features: list[str]


def make_hgb_regressor(spec: ModelSpec) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), spec.numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                spec.categorical_features,
            ),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _make_preprocessor(spec: ModelSpec) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), spec.numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                spec.categorical_features,
            ),
        ],
        remainder="drop",
    )


def make_xgb_regressor(spec: ModelSpec) -> Pipeline:
    if importlib.util.find_spec("xgboost") is None:
        raise ImportError("xgboost is not installed")
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline(steps=[("preprocess", _make_preprocessor(spec)), ("model", model)])


def make_catboost_regressor(spec: ModelSpec) -> Pipeline:
    if importlib.util.find_spec("catboost") is None:
        raise ImportError("catboost is not installed")
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=160,
        depth=5,
        learning_rate=0.06,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    return Pipeline(steps=[("preprocess", _make_preprocessor(spec)), ("model", model)])


def make_lightgbm_regressor(spec: ModelSpec) -> Pipeline:
    if importlib.util.find_spec("lightgbm") is None:
        raise ImportError("lightgbm is not installed")
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )
    return Pipeline(steps=[("preprocess", _make_preprocessor(spec)), ("model", model)])


def make_lightgbm_quantile_regressor(spec: ModelSpec, alpha: float) -> Pipeline:
    if importlib.util.find_spec("lightgbm") is None:
        raise ImportError("lightgbm is not installed")
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )
    return Pipeline(steps=[("preprocess", _make_preprocessor(spec)), ("model", model)])


def _make_scaled_linear_preprocessor(spec: ModelSpec) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                spec.numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                spec.categorical_features,
            ),
        ],
        remainder="drop",
    )


def make_ridge_regressor(spec: ModelSpec) -> Pipeline:
    return Pipeline(steps=[("preprocess", _make_scaled_linear_preprocessor(spec)), ("model", Ridge(alpha=1.0))])


def make_elasticnet_regressor(spec: ModelSpec) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _make_scaled_linear_preprocessor(spec)),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.25, random_state=RANDOM_STATE, max_iter=5000)),
        ]
    )


def make_random_forest_regressor(spec: ModelSpec) -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline(steps=[("preprocess", _make_preprocessor(spec)), ("model", model)])


def make_regressor(model_name: str, spec: ModelSpec) -> Pipeline:
    if model_name == "hgb":
        return make_hgb_regressor(spec)
    if model_name == "xgb":
        return make_xgb_regressor(spec)
    if model_name == "catboost":
        return make_catboost_regressor(spec)
    if model_name == "lightgbm":
        return make_lightgbm_regressor(spec)
    if model_name == "ridge":
        return make_ridge_regressor(spec)
    if model_name == "elasticnet":
        return make_elasticnet_regressor(spec)
    if model_name == "random_forest":
        return make_random_forest_regressor(spec)
    raise ValueError(f"Unknown model_name: {model_name}")


def fit_predict_regressor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    target_col: str = "pd",
    model_name: str = "hgb",
) -> tuple[Pipeline, pd.DataFrame]:
    numeric = [col for col in feature_cols if col not in categorical_cols]
    spec = ModelSpec(numeric_features=numeric, categorical_features=categorical_cols)
    train_clean = train.dropna(subset=[target_col]).copy()
    test_clean = test.dropna(subset=[target_col]).copy()
    model = make_regressor(model_name, spec)
    model.fit(train_clean[feature_cols], train_clean[target_col])
    pred = test_clean.copy()
    pred["predicted_pd"] = model.predict(test_clean[feature_cols])
    pred["predicted_pd"] = pred["predicted_pd"].clip(lower=0)
    return model, pred
