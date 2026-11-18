from __future__ import annotations

from lightgbm import LGBMClassifier


def build(
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    min_child_samples: int,
    seed: int,
) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
