from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def build_pipeline(
    features: pd.DataFrame,
    *,
    model_name: str,
    model_params: dict[str, Any],
    random_state: int,
    min_category_frequency: int,
) -> Pipeline:
    """Build a reusable preprocessing-and-model pipeline from training columns."""
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_category_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
    )

    if model_name == "logistic_regression":
        defaults: dict[str, Any] = {"max_iter": 1000, "solver": "liblinear"}
        estimator = LogisticRegression(random_state=random_state, **(defaults | model_params))
    elif model_name == "random_forest":
        defaults = {
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "class_weight": None,
        }
        estimator = RandomForestClassifier(random_state=random_state, **(defaults | model_params))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps=[("preprocessing", preprocessing), ("model", estimator)])

