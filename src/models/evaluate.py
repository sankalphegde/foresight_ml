"""Evaluation stub for future MLflow integration."""


def evaluate_model(run_id: str | None = None, model_uri: str | None = None) -> dict[str, str]:
    """Placeholder evaluation entrypoint.

    Future implementation template:

    # import mlflow
    # from src.config.settings import settings
    #
    # mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # resolved_model_uri = model_uri or f"runs:/{run_id}/model"
    # with mlflow.start_run(run_name="example-eval") as run:
    #     mlflow.log_param("evaluated_model_uri", resolved_model_uri)
    #     mlflow.log_metric("eval_f1", 0.0)
    #     return {"evaluation_run_id": run.info.run_id}
    """
    raise NotImplementedError("Evaluation is intentionally left as a stub.")
