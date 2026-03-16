"""Model training and evaluation tests."""

import numpy as np
import pandas as pd

from src.models.train import train_baseline


def test_training_smoke_outputs_probabilities() -> None:
    """Smoke test for model training and predict_proba outputs."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.normal(size=(500, 6)),
        columns=["f1", "f2", "f3", "f4", "f5", "f6"],
    )
    df["distress_label"] = rng.integers(0, 2, size=500)

    train = df.iloc[:400]
    val = df.iloc[400:]

    model, _ = train_baseline(train, val, scale_pos_weight=1.0)
    probs = model.predict_proba(val.drop(columns=["distress_label"]))[:, 1]

    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
