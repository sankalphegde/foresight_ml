"""Executes batch inference using the Production model and attaches SHAP explanations."""

import pandas as pd
import mlflow
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_batch_inference(features_gcs_path: str, version_str: str = "1.0") -> None:
    """
    Loads Production model, scores data, and attaches SHAP explanations.
    """
    model_name = "foresight_xgboost"
    model_uri = f"models:/{model_name}/Production"
    
    logger.info(f"Loading Production model from {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri) #
    
    logger.info(f"Loading latest features from {features_gcs_path}")
    latest_features_df = pd.read_parquet(features_gcs_path)
    
    # Prepare features for prediction
    X_predict = latest_features_df.drop(columns=["firm_id", "fiscal_year", "fiscal_period"])
    
    # Generate distress probability scores (0-1)
    predictions = model.predict(X_predict)
    latest_features_df['distress_probability'] = predictions
    
    # Include basic confidence interval (Simple 5% margin approach)
    latest_features_df['confidence_interval_lower'] = np.clip(predictions - 0.05, 0, 1) 
    latest_features_df['confidence_interval_upper'] = np.clip(predictions + 0.05, 0, 1)

    # Load precomputed SHAP values from Person 4
    shap_path = "gs://financial-distress-data/shap/shap_values.parquet"
    logger.info(f"Loading SHAP values from {shap_path}")
    shap_df = pd.read_parquet(shap_path)
    
    # Extract the necessary columns
    shap_subset = shap_df[["firm_id", "fiscal_year", "fiscal_period", "top_features_json"]]
    
    # Attach precomputed SHAP top_features_json to each scored row
    final_scored_df = pd.merge(
        latest_features_df, 
        shap_subset, 
        on=["firm_id", "fiscal_year", "fiscal_period"], 
        how="left" #
    )
    
    # Write scored output to GCS
    output_path = f"gs://financial-distress-data/inference/scores_v{version_str}/scores.parquet"
    final_scored_df.to_parquet(output_path, index=False)
    logger.info(f"Successfully saved batch inference results to {output_path}")

if __name__ == "__main__":
    run_batch_inference("gs://financial-distress-data/features/latest.parquet", version_str="1.0")