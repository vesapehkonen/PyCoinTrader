#!/usr/bin/env python3
"""
Render a performance chart from simulation output.

Usage:
  python plot_performance.py <plot_data.pkl> [out_dir]

Arguments:
  plot_data.pkl   Path to the pickle file with simulation results
  out_dir         Optional path to output directory (default: same as pickle file)

Example:
  python plot_performance.py ../data/outputs/1/data.pkl
  python plot_performance.py ../data/outputs/1/data.pkl ./charts
"""

import sys
import pickle
from pathlib import Path

# allow imports from src/
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from trader.viz.plotting_tools import render_performance_plot


def main(plot_data_file, out_dir=None):
    plot_data_file = Path(plot_data_file)
    with plot_data_file.open("rb") as f:
        plot_data = pickle.load(f)

    df = plot_data["df"]
    trades = plot_data["trades"]
    input_file = plot_data["input_file"]

    out_dir = Path(out_dir) if out_dir else plot_data_file.parent
    render_performance_plot(df, df["Portfolio"], trades, input_file, out_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    plot_data_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    main(plot_data_file, out_dir)
