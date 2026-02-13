"""Local pipeline runner for development and testing."""

from dotenv import load_dotenv

from src.data.pipeline.core import (
    fetch_fred_data,
    fetch_sec_data,
    merge_data,
)

load_dotenv()


def main() -> None:
    """Run the full data pipeline locally."""
    sec = fetch_sec_data()
    fred = fetch_fred_data()
    merged = merge_data(sec, fred, output_dir="data/output")
    print("Pipeline completed:", merged)


if __name__ == "__main__":
    main()
