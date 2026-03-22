import unittest
from unittest.mock import patch, MagicMock
from src.models.registry import evaluate_and_register_model

class TestModelRegistry(unittest.TestCase):

    @patch('src.models.registry.MlflowClient')
    @patch('src.models.registry.mlflow.register_model')
    @patch('src.models.registry.gcsfs.GCSFileSystem')
    def test_worse_model_stays_in_staging(self, mock_gcs, mock_register, mock_client_class):
        """Test that a worse model is NOT promoted to Production."""
        mock_client = mock_client_class.return_value
        
        # Mock the new model registration
        mock_mv = MagicMock()
        mock_mv.version = "2"
        mock_register.return_value = mock_mv
        
        # Mock existing Production model with a high ROC-AUC (0.90)
        mock_prod_model = MagicMock()
        mock_prod_model.run_id = "prod_run_id"
        mock_client.get_latest_versions.return_value = [mock_prod_model]
        
        mock_run = MagicMock()
        mock_run.data.metrics = {"test_roc_auc": 0.90}
        mock_client.get_run.return_value = mock_run
        
        # Try to register a worse model (0.85 is > 2% worse than 0.90)
        result = evaluate_and_register_model(
            run_id="new_run_id", 
            test_roc_auc=0.85, 
            recall_critically_low=False
        )
        
        # Assert it returned False (was not promoted)
        self.assertFalse(result)
        
        # Assert transition to Production was NEVER called
        mock_client.transition_model_version_stage.assert_any_call(
            name="foresight_xgboost", version="2", stage="Staging"
        )
        
        # Ensure we didn't accidentally promote it
        with self.assertRaises(AssertionError):
            mock_client.transition_model_version_stage.assert_any_call(
                name="foresight_xgboost", version="2", stage="Production"
            )

    @patch('src.models.registry.MlflowClient')
    @patch('src.models.registry.mlflow.register_model')
    @patch('src.models.registry.gcsfs.GCSFileSystem')
    def test_better_model_promotes_to_production(self, mock_gcs, mock_register, mock_client_class):
        """Test that a better model IS promoted to Production."""
        mock_client = mock_client_class.return_value
        
        # Mock the new model registration
        mock_mv = MagicMock()
        mock_mv.version = "3"
        mock_register.return_value = mock_mv
        
        # Mock existing Production model with an average ROC-AUC (0.85)
        mock_prod_model = MagicMock()
        mock_prod_model.run_id = "prod_run_id"
        mock_client.get_latest_versions.return_value = [mock_prod_model]
        
        mock_run = MagicMock()
        mock_run.data.metrics = {"test_roc_auc": 0.85}
        mock_client.get_run.return_value = mock_run
        
        # Try to register a better model (0.88)
        result = evaluate_and_register_model(
            run_id="new_run_id", 
            test_roc_auc=0.88, 
            recall_critically_low=False
        )
        
        # Assert it returned True (was promoted)
        self.assertTrue(result)
        
        # Assert transition to Production WAS called
        mock_client.transition_model_version_stage.assert_any_call(
            name="foresight_xgboost", version="3", stage="Production"
        )

if __name__ == '__main__':
    unittest.main()