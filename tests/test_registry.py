import json
import os
import unittest
from unittest.mock import MagicMock, patch

from src.models.registry import evaluate_and_register_model, monitor_model_deletion


class TestModelRegistry(unittest.TestCase):
    @patch("src.models.registry.MlflowClient")
    @patch("src.models.registry.gcsfs.GCSFileSystem")
    def test_worse_model_stays_in_staging(self, mock_gcs, mock_client_class):
        """Test that a worse model is NOT promoted to Production."""
        mock_client = mock_client_class.return_value

        # Mock the new model registration via the lower-level API
        mock_mv = MagicMock()
        mock_mv.version = "2"
        mock_client.create_model_version.return_value = mock_mv

        # Mock existing Production model with a high ROC-AUC (0.90)
        mock_prod_model = MagicMock()
        mock_prod_model.run_id = "prod_run_id"
        mock_client.get_latest_versions.return_value = [mock_prod_model]

        # Create different mock runs for the new model vs the prod model
        new_run = MagicMock()
        new_run.info.artifact_uri = "mock_uri"

        prod_run = MagicMock()
        prod_run.data.metrics = {"test_roc_auc": 0.90}

        def get_run_side_effect(run_id):
            if run_id == "new_run_id":
                return new_run
            if run_id == "prod_run_id":
                return prod_run
            return MagicMock()

        mock_client.get_run.side_effect = get_run_side_effect

        # Try to register a worse model (0.85 is > 2% worse than 0.90)
        result = evaluate_and_register_model(
            run_id="new_run_id", test_roc_auc=0.85, recall_critically_low=False
        )

        # Assert it returned False (was not promoted)
        self.assertFalse(result)

        # Assert transition to Staging WAS called
        mock_client.transition_model_version_stage.assert_any_call(
            name="foresight_xgboost", version="2", stage="Staging"
        )

    @patch("src.models.registry.MlflowClient")
    @patch("src.models.registry.gcsfs.GCSFileSystem")
    def test_better_model_promotes_to_production(self, mock_gcs, mock_client_class):
        """Test that a better model IS promoted to Production."""
        mock_client = mock_client_class.return_value

        # Mock the new model registration
        mock_mv = MagicMock()
        mock_mv.version = "3"
        mock_client.create_model_version.return_value = mock_mv

        # Mock existing Production model with an average ROC-AUC (0.85)
        mock_prod_model = MagicMock()
        mock_prod_model.run_id = "prod_run_id"
        mock_client.get_latest_versions.return_value = [mock_prod_model]

        # Create different mock runs for the new model vs the prod model
        new_run = MagicMock()
        new_run.info.artifact_uri = "mock_uri"

        prod_run = MagicMock()
        prod_run.data.metrics = {"test_roc_auc": 0.85}

        def get_run_side_effect(run_id):
            if run_id == "new_run_id":
                return new_run
            if run_id == "prod_run_id":
                return prod_run
            return MagicMock()

        mock_client.get_run.side_effect = get_run_side_effect

        # Try to register a better model (0.88)
        result = evaluate_and_register_model(
            run_id="new_run_id", test_roc_auc=0.88, recall_critically_low=False
        )

        # Assert it returned True (was promoted)
        self.assertTrue(result)

        # Assert transition to Production WAS called
        mock_client.transition_model_version_stage.assert_any_call(
            name="foresight_xgboost", version="3", stage="Production"
        )

    def test_rollback_alert_generation(self):
        """Verify that monitor_model_deletion correctly writes the rollback alert file."""
        test_file = "tests/test_rollback_alert.json"

        # Ensure cleanup before test
        if os.path.exists(test_file):
            os.remove(test_file)

        try:
            # 1. Trigger the monitor function
            monitor_model_deletion(
                model_name="foresight_xgboost", version="1.0", save_path=test_file
            )

            # 2. Verify file creation
            self.assertTrue(os.path.exists(test_file), "Rollback JSON was not created.")

            # 3. Verify file content
            with open(test_file) as f:
                data = json.load(f)
                self.assertEqual(data["model_name"], "foresight_xgboost")
                self.assertEqual(data["version"], "1.0")
                self.assertEqual(data["status"], "ROLLBACK_REQUIRED")
                self.assertIn("alert_timestamp", data)

        finally:
            # Cleanup after test
            if os.path.exists(test_file):
                os.remove(test_file)


if __name__ == "__main__":
    unittest.main()
