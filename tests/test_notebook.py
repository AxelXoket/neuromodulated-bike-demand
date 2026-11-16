"""The committed notebook: built from the cell list in tools/build_notebook.py and executed without an error."""
import json
import unittest

import nbformat

from tests import ROOT

import build_notebook
from figures import ORDER

NOTEBOOK = nbformat.read(ROOT / "bike_demand.ipynb", as_version=4)
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))


class CommittedNotebook(unittest.TestCase):
    def test_the_cells_are_the_cell_list(self):
        self.assertEqual([(c.cell_type, c.source) for c in NOTEBOOK.cells], list(build_notebook.CELLS))
        self.assertEqual([c.id for c in NOTEBOOK.cells], [f"cell-{i}" for i in range(1, len(NOTEBOOK.cells) + 1)])

    def test_every_code_cell_ran_in_order_without_an_error(self):
        code = [c for c in NOTEBOOK.cells if c.cell_type == "code"]
        self.assertEqual([c.execution_count for c in code], list(range(1, len(code) + 1)))
        self.assertEqual([o for c in code for o in c.outputs if o.output_type == "error"], [])

    def test_the_outputs_show_the_figures_and_the_measured_scores(self):
        images = [o for c in NOTEBOOK.cells if c.cell_type == "code" for o in c.outputs if "image/png" in o.get("data", {})]
        self.assertEqual(len(images), 9)
        tables = "".join(o["data"].get("text/plain", "") for c in NOTEBOOK.cells if c.cell_type == "code"
                         for o in c.outputs if o.output_type == "execute_result")
        for name in ORDER:
            value = METRICS["test"][name]["rmsle"]
            value = value["mean"] if isinstance(value, dict) else value
            with self.subTest(model=name):
                self.assertIn(f"{value:.3f}", tables)


if __name__ == "__main__":
    unittest.main()
