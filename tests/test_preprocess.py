"""Data preprocessing tests."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_preprocess_runs_and_creates_output() -> None:
    """Very simple test.

    Runs preprocessing script and checks that
    the interim parquet file exists.
    """
    # Check if required data files exist, skip if not
    sec_path = Path("data/raw/sec/filings.jsonl")
    fred_path = Path("data/raw/fred/indicators.csv")

    if not sec_path.exists() or not fred_path.exists():
        pytest.skip("Required raw data files not available for preprocessing test")

    # run preprocess script
    result = subprocess.run(
        [sys.executable, "src/data/preprocess.py"],
        capture_output=True,
        text=True,
    )

    # script should exit successfully
    assert result.returncode == 0, result.stderr

    # check output file exists
    out_file = Path("data/interim/panel_base.parquet")
    assert out_file.exists()
