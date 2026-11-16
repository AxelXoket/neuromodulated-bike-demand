"""Rebuild everything from data/hour.csv in one command: the experiment, the figures, the report and the notebook.

Usage: python tools/rebuild.py
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Each step reads what the steps before it wrote: results/ first, then figures/, then both for the report and notebook.
STEPS = (("-m", "bike_demand.train"), ("tools/figures.py",), ("tools/build_report.py",), ("tools/build_notebook.py",))


def main() -> None:
    for step in STEPS:
        print("rebuild:", " ".join(step), flush=True)
        subprocess.run([sys.executable, *step], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
