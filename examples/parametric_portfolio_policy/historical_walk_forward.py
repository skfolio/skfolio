"""Export the reports from :mod:`plot_historical_walk_forward` to CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_historical_walk_forward import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    _, report = run(parser.parse_args().output_dir)
    print(json.dumps(report, indent=2))
