"""Handles the evaluation, registration, and promotion of trained models via MLflow."""

import mlflow
import logging
from mlflow.tracking import MlflowClient
import gcsfs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_and_register_model(run_id: str, test_roc_auc: float, recall_critically_low: bool, model_name: str = "foresight_xgboost", version_str: str = "1.0") -> bool:
    """
    Evaluates model metrics, registers to MLflow, and handles promotion.
    """
    # 1. Acceptance Gate
    if test_roc_auc < 0.80 or recall_critically_low:
        logger.warning(f"Model failed acceptance gate (ROC-AUC: {test_roc_auc}). Registration aborted.")
        return False
        
    client = MlflowClient()
    
    run = client.get_run(run_id)
    source_uri = f"{run.info.artifact_uri}/model"
    
    mv = client.create_model_version(
        name=model_name,
        source=source_uri,
        run_id=run_id
    )

    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging"
    )
    logger.info(f"Model registered and transitioned to Staging (Version {mv.version}).")
    
    # 2. Rollback Check / Promotion Logic
    promote_to_production = True
    prod_models = client.get_latest_versions(model_name, stages=["Production"])
    
    if prod_models:
        prod_model = prod_models[0]
        
        if prod_model.run_id is not None:
            prod_run = client.get_run(prod_model.run_id)
            prod_roc_auc = prod_run.data.metrics.get("test_roc_auc", 0.0)
            
            # Check if new model is better or within 2% tolerance
            if test_roc_auc < (prod_roc_auc - 0.02):
                promote_to_production = False
                logger.warning("New model is significantly worse than Production. Keeping existing Production version.")
        else:
            logger.warning("Current Production model has no run_id. Promoting new model by default.")
            
    if promote_to_production:
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production"
        )
        logger.info(f"Model version {mv.version} promoted to Production!")
        
        # Push the final serialized model artifact to GCS versioned path
        fs = gcsfs.GCSFileSystem()
        source_model = "financial-distress-data/models/xgb_model.pkl"
        source_scaler = "financial-distress-data/models/scaler_pipeline.pkl"
        
        dest_dir = f"financial-distress-data/models/v{version_str}"
        fs.copy(source_model, f"{dest_dir}/xgb_model.pkl")
        fs.copy(source_scaler, f"{dest_dir}/scaler_pipeline.pkl")
        logger.info(f"Artifacts pushed to gs://{dest_dir}/")
        
    return promote_to_production

if __name__ == "__main__":
    # 1. Connect to your specific experiment ID
    experiment_id = "2" 
    
    logger.info(f"Searching for the latest run in experiment ID: {experiment_id}")
    
    # 2. Search for the single most recent run 
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id], 
        order_by=["start_time DESC"], 
        max_results=1
    )
    
    # Use len() to safely check if the search returned any results
    if len(runs) > 0:
        # Use '# type: ignore' to tell VS Code to silence the false alarms
        latest_run_id = runs.iloc[0]["run_id"]  # type: ignore
        test_roc_auc = runs.iloc[0].get("metrics.test_roc_auc", 0.0)  # type: ignore
        recall_critically_low = False 
        
        logger.info(f"Automatically found latest Run ID: {latest_run_id} with ROC-AUC: {test_roc_auc}")
        
        # 4. Trigger your function with the dynamic ID
        evaluate_and_register_model(
            run_id=latest_run_id, 
            test_roc_auc=test_roc_auc, 
            recall_critically_low=recall_critically_low
        )
    else:
        logger.error(f"No runs found in experiment ID {experiment_id}. Cannot register model.")