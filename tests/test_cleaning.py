import pandas as pd

from src.data.cleaning import clean_and_impute


def test_clean_and_impute_drops_and_fills() -> None:
    df = pd.DataFrame(
        {
            "cik": ["0000000001", "0000000001", "0000000002", None],
            "filing_date": [
                pd.Timestamp("2024-01-01"),
                pd.NaT,
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ],
            "num": [1.0, None, 3.0, 4.0],
            "cat": ["A", None, "B", None],
        }
    )

    cleaned, report = clean_and_impute(df)

    assert report["rows_dropped"] == 2
    assert cleaned["num"].isna().sum() == 0
    assert cleaned["cat"].isna().sum() == 0
    assert report["row_count_after"] == len(cleaned)
