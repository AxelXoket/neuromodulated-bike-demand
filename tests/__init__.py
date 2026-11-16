"""Tests import the report and figure tools from tools/ the same way the tools import each other."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
