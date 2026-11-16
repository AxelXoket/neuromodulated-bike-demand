"""Metrics and the statistics, on small hand-made data. The training loops are in slow_tests/."""
import math
import unittest

import numpy as np
import pandas as pd
import torch

from bike_demand import data, train

SCALING = data.Scaling(np.zeros(1, np.float32), np.ones(1, np.float32), 4.0, 1.5)


class Metrics(unittest.TestCase):
    def test_metrics_match_the_hand_computed_values(self):
        result = train.metrics(np.array([1.0, 3.0]), np.array([2.0, 1.0]))
        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], math.sqrt(2.5))
        self.assertAlmostEqual(result["r2"], -1.5)
        self.assertAlmostEqual(result["rmsle"], math.sqrt(((math.log(3) - math.log(2)) ** 2 + (math.log(2) - math.log(4)) ** 2) / 2))
        self.assertAlmostEqual(result["bias"], -0.5)

    def test_a_perfect_prediction_scores_zero_error_and_r2_of_one(self):
        actual = np.array([5.0, 50.0, 500.0])
        result = train.metrics(actual, actual.copy())
        self.assertEqual((result["mae"], result["rmse"], result["rmsle"], result["bias"]), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(result["r2"], 1.0)

    def test_the_training_score_is_the_reported_rmsle_including_cut_negative_counts(self):
        scaled_prediction = np.array([-4.0, 0.3, 1.2], np.float32)  # the first one maps to a negative count
        scaled_target = np.array([-2.0, 0.1, 1.0], np.float32)
        counts = SCALING.counts(scaled_target.astype(np.float64))
        expected = train.metrics(counts, SCALING.counts(scaled_prediction.astype(np.float64)))["rmsle"]
        got = train.rmsle_scaled(torch.tensor(scaled_prediction), torch.tensor(scaled_target), SCALING)
        self.assertAlmostEqual(got, expected, places=5)

    def test_summary_gives_mean_and_sample_spread(self):
        self.assertEqual(train.summary([{"rmse": 1.0}, {"rmse": 3.0}])["rmse"], {"mean": 2.0, "std": math.sqrt(2.0)})


class Statistics(unittest.TestCase):
    days = np.repeat(np.arange(30), 24)

    def errors(self, scale: float = 1.0, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return scale * rng.uniform(0.01, 0.2, size=(5, len(self.days)))

    def test_the_seed_t_interval_is_the_t_interval_of_the_differences(self):
        result = train.paired_difference([3.0, 4.0, 5.0], [1.0, 1.0, 1.0])
        half = 4.302652729911275 / math.sqrt(3)
        self.assertEqual((result["mean"], result["seeds"]), (3.0, 3))
        self.assertAlmostEqual(result["low"], 3.0 - half)
        self.assertAlmostEqual(result["high"], 3.0 + half)

    def test_two_identical_models_differ_by_exactly_zero(self):
        errors = self.errors()
        result = train.paired_bootstrap(errors, errors.copy(), self.days, reps=500)
        self.assertEqual((result["mean"], result["low"], result["high"]), (0.0, 0.0, 0.0))

    def test_a_model_that_is_worse_on_every_day_gets_an_interval_above_zero(self):
        better = self.errors()
        worse = better * 1.21  # 10% larger log errors on every hour
        result = train.paired_bootstrap(worse, better, self.days, reps=500)
        direct = np.sqrt(worse.mean(1)).mean() - np.sqrt(better.mean(1)).mean()
        self.assertAlmostEqual(result["mean"], direct, places=10)
        self.assertGreater(result["low"], 0)
        self.assertLess(result["low"], result["mean"])
        self.assertGreater(result["high"], result["mean"])

    def test_a_difference_made_on_one_day_is_not_called_certain(self):
        """Every seed shows it, so a seed-only interval would; drawing the days again shows how fragile it is."""
        better = self.errors()
        worse = better.copy()
        worse[:, self.days == 7] += 0.5
        result = train.paired_bootstrap(worse, better, self.days, reps=2000)
        seed_only = train.paired_difference(list(np.sqrt(worse.mean(1))), list(np.sqrt(better.mean(1))))
        self.assertGreater(seed_only["low"], 0)
        self.assertLessEqual(result["low"], 1e-12)

    def test_a_difference_made_by_one_seed_is_not_called_certain(self):
        better = self.errors()
        worse = better.copy()
        worse[0] += 0.3
        result = train.paired_bootstrap(worse, better, self.days, reps=2000)
        self.assertGreater(result["mean"], 0)
        self.assertLessEqual(result["low"], 1e-12)

    def test_the_same_seed_gives_the_same_interval(self):
        a, b = self.errors(1.1), self.errors()
        self.assertEqual(train.paired_bootstrap(a, b, self.days, reps=300), train.paired_bootstrap(a, b, self.days, reps=300))


class SharedBatchOrder(unittest.TestCase):
    """For one seed every kind of network is trained on the same batches, so the comparison is paired."""

    def batches(self, kind: str, seed: int) -> list[int]:
        rows = 40
        frame = pd.DataFrame({"cnt": np.ones(rows)})
        part = data.Split(frame, np.zeros((rows, 4), np.float32), np.zeros((rows, 3), np.float32), np.zeros(rows, np.float32),
                          frame["cnt"].to_numpy())
        _, _, order = train.start(kind, 8, train.Config(0.01, 0.0), seed, part, torch.device("cpu"))
        return torch.randperm(rows, generator=order).tolist()

    def test_every_kind_sees_the_same_batch_order_for_a_seed(self):
        orders = {kind: self.batches(kind, 3) for kind in ("no-state", "concat", "shift", "film")}
        self.assertEqual(len({tuple(order) for order in orders.values()}), 1)
        self.assertNotEqual(self.batches("film", 3), self.batches("film", 4))


if __name__ == "__main__":
    unittest.main()
