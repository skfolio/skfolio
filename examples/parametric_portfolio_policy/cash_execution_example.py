"""Export the reports from :mod:`plot_cash_execution` to CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_cash_execution import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
