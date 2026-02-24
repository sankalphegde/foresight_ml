"""Panel dataset construction and column standardization logic."""

import pandas as pd

from src.utils.logging import get_logger
from src.utils.validation import validate_schema

logger = get_logger(__name__)


class PanelBuilder:
    """Transforms cleaned raw data into standardized panel format."""

    RAW_TO_STANDARD = {
        "cik": "firm_id",
        "end_date": "date",
        "Assets": "total_assets",
        "Liabilities": "total_liabilities",
        "NetIncomeLoss": "net_income",
    }

    REQUIRED_COLUMNS = [
        "firm_id",
        "date",
        "total_assets",
        "total_liabilities",
        "net_income",
    ]

    def __init__(self, df: pd.DataFrame):
        """Initialize builder with cleaned DataFrame."""
        self.df = df.copy()

    def build(self) -> pd.DataFrame:
        """Return standardized panel DataFrame."""
        logger.info("Standardizing column names")
        self.df = self.df.rename(columns=self.RAW_TO_STANDARD)

        logger.info("Validating schema")
        validate_schema(self.df, self.REQUIRED_COLUMNS)

        logger.info("Parsing dates")
        self.df["date"] = pd.to_datetime(self.df["date"])

        logger.info("Removing duplicates")
        self.df = self.df.drop_duplicates(subset=["firm_id", "date"])

        logger.info("Sorting panel")
        self.df = self.df.sort_values(["firm_id", "date"])

        logger.info("Checking missing quarters")
        self._check_missing_quarters()

        logger.info("Creating lag features")
        self._create_lags()

        return self.df

    def _check_missing_quarters(self) -> None:
        quarter_diff = self.df.groupby("firm_id")["date"].diff().dt.days

        gaps = quarter_diff[quarter_diff > 120]
        if not gaps.empty:
            logger.warning("Detected potential missing quarters.")

    def _create_lags(self) -> None:
        lag_columns = [
            "total_assets",
            "total_liabilities",
            "net_income",
        ]

        for col in lag_columns:
            self.df[f"{col}_lag1"] = self.df.groupby("firm_id")[col].shift(1)

            self.df[f"{col}_lag4"] = self.df.groupby("firm_id")[col].shift(4)
