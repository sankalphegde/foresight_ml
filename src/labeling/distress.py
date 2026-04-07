"""Accounting-based financial distress labeling logic.

Composite Definition
--------------------
A firm is labeled distressed at time t+horizon if **2 or more** of the
following 5 signals are simultaneously true at t+horizon:

1. net_income < 0 for 2 consecutive quarters
2. operating_cash_flow < 0 for 2 consecutive quarters
3. debt_to_equity ratio increased by >= 20 % over the past 4 quarters
4. interest_coverage_ratio < 1.5 for 2 consecutive quarters
5. retained_earnings declining for 3 consecutive quarters

All signals are computed per-firm (grouped by firm_id, sorted by date).
The composite boolean is then shifted forward by `horizon` quarters to
produce a forward-looking label with no data leakage.

GCS output path: features/labeled_v1/labeled_panel.parquet
"""

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum positive-rate threshold before emitting a calibration warning.
_MIN_POSITIVE_RATE: float = 0.01


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two Series element-wise, returning NaN where denominator is 0."""
    return numerator.where(denominator != 0, other=float("nan")) / denominator.where(
        denominator != 0, other=float("nan")
    )


class DistressLabeler:
    """Creates forward-looking distress labels based on composite accounting signals."""

    def __init__(self, df: pd.DataFrame, horizon: int) -> None:
        """Initialize the labeler with a panel DataFrame and forecast horizon.

        Parameters
        ----------
        df:
            Panel DataFrame sorted by (firm_id, date).  Must contain columns:
            firm_id, date, net_income, operating_cash_flow, total_debt,
            total_equity, interest_expense, operating_income, retained_earnings.
        horizon:
            Number of quarters to shift the label forward (no-leakage guarantee).
        """
        self.df = df.copy()
        self.horizon = horizon

    # ------------------------------------------------------------------
    # Internal signal builders
    # ------------------------------------------------------------------

    def _signal_neg_income(self, grp: pd.DataFrame) -> pd.Series:
        """Signal 1: net_income < 0 for 2 consecutive quarters."""
        neg = (grp["net_income"] < 0).astype(int)
        return neg.rolling(2).sum() == 2

    def _signal_neg_ocf(self, grp: pd.DataFrame) -> pd.Series:
        """Signal 2: operating_cash_flow < 0 for 2 consecutive quarters."""
        neg = (grp["operating_cash_flow"] < 0).astype(int)
        return neg.rolling(2).sum() == 2

    def _signal_leverage_spike(self, grp: pd.DataFrame) -> pd.Series:
        """Signal 3: debt_to_equity ratio increased by >= 20 % over 4 quarters."""
        de_ratio = _safe_divide(grp["total_debt"], grp["total_equity"])
        de_4q_ago = de_ratio.shift(4)
        # pct change vs 4 quarters ago; NaN when de_4q_ago is 0 or missing
        pct_change = _safe_divide(de_ratio - de_4q_ago, de_4q_ago.abs())
        return pct_change >= 0.20

    def _signal_low_coverage(self, grp: pd.DataFrame) -> pd.Series:
        """Signal 4: interest_coverage_ratio < 1.5 for 2 consecutive quarters."""
        icr = _safe_divide(grp["operating_income"], grp["interest_expense"])
        low = (icr < 1.5).astype(int)
        return low.rolling(2).sum() == 2

    def _signal_declining_retained_earnings(self, grp: pd.DataFrame) -> pd.Series:
        """Signal 5: retained_earnings declining for 3 consecutive quarters.

        All three period-over-period diffs within the rolling 3-quarter window
        must be strictly negative.
        """
        re = grp["retained_earnings"]
        # diff() gives change vs previous period; we need all 3 diffs
        # within a 3-quarter window to be negative.
        # rolling(3).apply checks the 3-point window [t-2, t-1, t]:
        #   diffs are [re[t-1]-re[t-2], re[t]-re[t-1]] — only 2 diffs available.
        # We therefore check that both consecutive diffs are negative across a
        # 3-row window using diff on the original series plus a rolling min.
        d = re.diff()
        # All diffs in the last 3 rows (i.e., 2 diff values + the current one)
        # must be < 0.  Rolling min over 3 rows of d: if max is < 0, all are < 0.
        return d.rolling(3).max() < 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self) -> pd.DataFrame:
        """Generate composite distress labels and return the updated DataFrame.

        Returns:
        -------
        pd.DataFrame
            Original DataFrame with a new integer column ``distress_label``
            (1 = distressed, 0 = non-distressed / boundary).
        """
        logger.info(
            "Computing composite distress label (5 signals, threshold >= 2, horizon=%d)",
            self.horizon,
        )

        df = self.df

        # ---- Compute each signal per firm --------------------------------
        logger.info("Signal 1: consecutive negative net income")
        df["_s1"] = (
            df.groupby("firm_id", group_keys=False).apply(self._signal_neg_income).astype(float)
        )

        logger.info("Signal 2: consecutive negative operating cash flow")
        df["_s2"] = (
            df.groupby("firm_id", group_keys=False).apply(self._signal_neg_ocf).astype(float)
        )

        logger.info("Signal 3: debt-to-equity spike >= 20 %% over 4 quarters")
        df["_s3"] = (
            df.groupby("firm_id", group_keys=False).apply(self._signal_leverage_spike).astype(float)
        )

        logger.info("Signal 4: low interest coverage ratio for 2 consecutive quarters")
        df["_s4"] = (
            df.groupby("firm_id", group_keys=False).apply(self._signal_low_coverage).astype(float)
        )

        logger.info("Signal 5: declining retained earnings for 3 consecutive quarters")
        df["_s5"] = (
            df.groupby("firm_id", group_keys=False)
            .apply(self._signal_declining_retained_earnings)
            .astype(float)
        )

        # ---- Composite signal count (NaN signals treated as 0) -----------
        signal_cols = ["_s1", "_s2", "_s3", "_s4", "_s5"]
        df["_signal_count"] = df[signal_cols].fillna(0).sum(axis=1)

        # ---- Shift forward by horizon quarters (no leakage) --------------
        logger.info("Shifting composite label forward by %d quarter(s)", self.horizon)
        df["distress_label"] = df.groupby("firm_id")["_signal_count"].transform(
            lambda s: (s >= 2).shift(-self.horizon)
        )  # noqa: B023

        # Fill boundary NaNs with 0 (non-distress)
        df["distress_label"] = df["distress_label"].fillna(0).astype(int)

        # ---- Drop intermediate columns -----------------------------------
        df = df.drop(columns=signal_cols + ["_signal_count"])
        self.df = df

        # ---- Post-labeling diagnostics -----------------------------------
        positive_rate: float = float(df["distress_label"].mean())
        logger.info(
            "Distress label positive rate: %.4f (%.2f%%)", positive_rate, positive_rate * 100
        )

        if positive_rate < _MIN_POSITIVE_RATE:
            logger.warning(
                "Positive rate %.4f is below %.2f%%. "
                "Consider relaxing the distress threshold to 1 signal "
                "or revisiting signal definitions.",
                positive_rate,
                _MIN_POSITIVE_RATE * 100,
            )

        return self.df
