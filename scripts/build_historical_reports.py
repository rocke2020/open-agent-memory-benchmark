"""Build offline HTML from existing public evidence without evaluation calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from oamb.reporting.historical import build_historical_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directories", nargs="+", type=Path)
    for root in parser.parse_args().result_directories:
        print(build_historical_report(root))
