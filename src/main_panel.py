from src.config.settings import settings
from src.utils.gcs import list_parquet_files, read_parquet_from_gcs, write_parquet_to_gcs
from src.panel.builder import PanelBuilder
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Listing cleaned parquet files")
    paths = list_parquet_files(settings.gcs_bucket, settings.cleaned_path)

    logger.info("Reading cleaned dataset")
    df = read_parquet_from_gcs(paths)

    logger.info(f"Columns in dataset: {df.columns.tolist()}")

    builder = PanelBuilder(df)
    panel_df = builder.build()

    logger.info("Saving panel dataset")
    write_parquet_to_gcs(panel_df, settings.gcs_bucket, settings.panel_output_path)

    logger.info("Panel building complete")


if __name__ == "__main__":
    main()