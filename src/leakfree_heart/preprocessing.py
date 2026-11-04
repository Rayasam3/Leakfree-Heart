"""Leakage-safe preprocessing. Everything here is fitted inside the training fold only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC


def _num_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in NUMERIC if c in X.columns]


def _cat_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in CATEGORICAL if c in X.columns]


def _all_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in FEATURES if c in X.columns]


def make_preprocessor(kind: str = "onehot", missing_indicator: bool = False) -> ColumnTransformer:
    """`onehot`  : median-impute + scale numerics, mode-impute + one-hot categoricals (default).
    `ordinal` : the base paper's treatment - all 13 coded columns imputed and z-scored (used by the NNs)."""
    if kind == "onehot":
        num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=missing_indicator)),
                        ("sc", StandardScaler())])
        cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        return ColumnTransformer([("num", num, _num_cols), ("cat", cat, _cat_cols)], remainder="drop")
    if kind == "ordinal":
        allc = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=missing_indicator)),
                         ("sc", StandardScaler())])
        return ColumnTransformer([("all", allc, _all_cols)], remainder="drop")
    raise ValueError(kind)


class TopKFeatureSelector(BaseEstimator, TransformerMixin):
    """Keeps the k most important raw features, ranked on the *training fold only*
    (random-forest impurity importance on median-imputed data). Using it inside the CV pipeline
    avoids the common leakage of selecting features on the full dataset."""

    def __init__(self, k: int = 8, random_state: int = 0):
        self.k = k
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        imp = SimpleImputer(strategy="median").fit(X)
        Xi = imp.transform(X)
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=self.random_state, n_jobs=1)
        rf.fit(Xi, np.asarray(y))
        order = np.argsort(-rf.feature_importances_)
        self.importances_ = dict(zip(X.columns, rf.feature_importances_))
        self.selected_ = [X.columns[i] for i in order[: self.k]]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_)
