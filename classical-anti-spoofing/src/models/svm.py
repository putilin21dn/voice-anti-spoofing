from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build(C: float, gamma: str | float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    C=C,
                    gamma=gamma,
                    kernel="rbf",
                    class_weight="balanced",
                ),
            ),
        ]
    )
