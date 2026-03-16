"""Prediction stub for future MLflow integration."""


def predict() -> None:
    """Placeholder prediction entrypoint.

    Future implementation template:

    # import mlflow
    # from src.config.settings import settings
    #
    # mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # model_uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    # model = mlflow.pyfunc.load_model(model_uri)
    # predictions = model.predict(feature_frame)
    """
    raise NotImplementedError("Prediction is intentionally left as a stub.")
