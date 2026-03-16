"""Minimal MLflow smoke test for tracking + artifact + model loading."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import mlflow
import pandas as pd


@dataclass(frozen=True)
class SmokeConfig:
    """Runtime configuration for the MLflow smoke test."""

    tracking_uri: str
    experiment_name: str
    constant: float
    log_model: bool


class AddConstantModel(mlflow.pyfunc.PythonModel):  # pragma: no cover - runtime model wrapper
    """Simple model that returns x + constant for each row."""

    def __init__(self, constant: float) -> None:
        """Store model constant used during prediction."""
        self.constant = constant

    def predict(self, context: object, model_input: pd.DataFrame) -> pd.Series:
        """Predict by adding a constant to column `x`."""
        if "x" not in model_input.columns:
            raise ValueError("Input DataFrame must contain an 'x' column.")
        return model_input["x"] + self.constant


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a minimal MLflow smoke test.")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", ""),
        help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI env var.",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "foresight-smoke-test"),
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--constant",
        type=float,
        default=1.5,
        help="Constant value added by the toy model.",
    )
    parser.add_argument(
        "--log-model",
        action="store_true",
        help="Also log a toy pyfunc model (may depend on MLflow server/client compatibility).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SmokeConfig:
    """Build and validate smoke test configuration."""
    tracking_uri = args.tracking_uri.strip()
    if not tracking_uri:
        raise ValueError("Missing tracking URI. Set MLFLOW_TRACKING_URI or pass --tracking-uri.")

    return SmokeConfig(
        tracking_uri=tracking_uri,
        experiment_name=args.experiment_name,
        constant=args.constant,
        log_model=args.log_model,
    )


def run_smoke_test(config: SmokeConfig) -> tuple[str, str]:
    """Run MLflow smoke test and return run_id/model_uri (if logged)."""
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    train_df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    expected = train_df["x"] + config.constant
    signature = infer_signature(train_df, expected)

    with mlflow.start_run(run_name="smoke-test") as run:
        mlflow.log_param("constant", config.constant)
        mlflow.log_metric("example_metric", 1.0)
        mlflow.log_text("mlflow smoke artifact", "smoke.txt")

        if config.log_model:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=AddConstantModel(config.constant),
                input_example=train_df,
                signature=signature,
            )

        run_id = run.info.run_id

    model_uri = ""
    if config.log_model:
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        test_df = pd.DataFrame({"x": [5.0, -1.0]})
        preds = loaded_model.predict(test_df)
        expected_test = test_df["x"] + config.constant

        if not (preds.reset_index(drop=True) == expected_test.reset_index(drop=True)).all():
            raise RuntimeError("Prediction check failed after model reload from MLflow.")

    return run_id, model_uri


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = build_config(args)
    run_id, model_uri = run_smoke_test(config)

    print("MLflow smoke test passed.")
    print(f"Tracking URI: {config.tracking_uri}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Run ID: {run_id}")
    if model_uri:
        print(f"Model URI: {model_uri}")


if __name__ == "__main__":
    main()
