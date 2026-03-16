"""Training stub for future MLflow integration.

Infrastructure for MLflow is provisioned in Terraform. This module intentionally
contains only copy/paste templates and no executable training logic.
"""


def train_and_log_model() -> dict[str, str]:
    """Placeholder training entrypoint.

    Future implementation template:

    # import mlflow
    # from src.config.settings import settings
    #
    # mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # mlflow.set_experiment("foresight-training")
    # with mlflow.start_run(run_name="example-train") as run:
    #     mlflow.log_param("model_type", "<fill-me>")
    #     mlflow.log_metric("f1", 0.0)
    #     # mlflow.<flavor>.log_model(model, artifact_path="model")
    #     return {"run_id": run.info.run_id, "model_uri": f"runs:/{run.info.run_id}/model"}
    """
    raise NotImplementedError("Training is intentionally left as a stub.")


def register_model_from_run(run_id: str) -> dict[str, str]:
    """Placeholder model registration entrypoint.

    Future implementation template:

    # import mlflow
    # from mlflow.tracking import MlflowClient
    # from src.config.settings import settings
    #
    # mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # model_uri = f"runs:/{run_id}/model"
    # registered = mlflow.register_model(model_uri=model_uri, name=settings.mlflow_model_name)
    # MlflowClient().set_registered_model_alias(
    #     name=settings.mlflow_model_name,
    #     alias=settings.mlflow_model_alias,
    #     version=registered.version,
    # )
    """
    raise NotImplementedError("Model registration is intentionally left as a stub.")
