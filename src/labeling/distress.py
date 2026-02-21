import pandas as pd
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DistressLabeler:

    def __init__(self, df: pd.DataFrame, horizon: int):
        self.df = df.copy()
        self.horizon = horizon

    def apply(self) -> pd.DataFrame:
        logger.info("Creating accounting-based distress label")

        self.df["neg_income"] = self.df["net_income"] < 0

        self.df["two_consecutive_losses"] = (
            self.df.groupby("firm_id")["neg_income"]
            .rolling(2)
            .sum()
            .reset_index(level=0, drop=True) == 2
        )

        logger.info("Shifting label forward for prediction horizon")
        self.df["distress_label"] = (
            self.df.groupby("firm_id")["two_consecutive_losses"]
            .shift(-self.horizon)
        )

        self.df["distress_label"] = self.df["distress_label"].fillna(False).astype(int)

        self.df = self.df.drop(columns=["neg_income", "two_consecutive_losses"])

        return self.df