"""The humidity counterfactual in the report: it refits the saved gradient boosting and changes only the humidity."""
import json
import unittest

import pandas as pd

from tests import ROOT

import build_report
from bike_demand import data

METRICS = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
PREDICTIONS = pd.read_csv(ROOT / "results" / "predictions.csv", parse_dates=["time"])


class HumidityCounterfactual(unittest.TestCase):
    def test_it_starts_from_the_saved_prediction_and_only_the_humidity_moves_it(self):
        day = pd.Timestamp(build_report.HUMID_SATURDAY)
        result = build_report.humidity_counterfactual(METRICS, PREDICTIONS, build_report.HUMID_SATURDAY)
        saved = PREDICTIONS.loc[PREDICTIONS["time"].dt.normalize() == day, "hgb"].sum()
        self.assertAlmostEqual(result["predicted"], saved, delta=1)
        frame = data.load()
        learned = frame[(frame["time"] < pd.Timestamp(data.SPLITS["test"][0])) & (frame["mnth"] == day.month)]
        self.assertAlmostEqual(result["typical"], float(learned["hum"].median()))
        self.assertGreater(result["humidity"], result["typical"])
        self.assertGreater(result["changed"], result["predicted"], "the report says the prediction rises")


if __name__ == "__main__":
    unittest.main()
