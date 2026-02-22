"""Entry point for distress labeling pipeline."""

from src.config.settings import settings
from src.utils.gcs import read_parquet_from_gcs, write_parquet_to_gcs
from src.utils.logging import get_logger
from src.utils.validation import summarize_class_balance
from src.labeling.distress import DistressLabeler

logger = get_logger(__name__)


def main() -> None:
    
    """Run labeling job and persist labeled dataset."""

    logger.info("Reading panel dataset")
    df = read_parquet_from_gcs(
        [f"gs://{settings.gcs_bucket}/{settings.panel_output_path}"]
    )

    labeler = DistressLabeler(df, settings.prediction_horizon)
    labeled_df = labeler.apply()

    summarize_class_balance(labeled_df, "distress_label")

    logger.info("Saving labeled dataset")
    write_parquet_to_gcs(
        labeled_df,
        settings.gcs_bucket,
        settings.labeled_output_path
    )

    logger.info("Labeling complete")


if __name__ == "__main__":
    main()