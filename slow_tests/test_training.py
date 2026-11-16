"""The training loops on a small hand-made problem: best epoch, patience, seeds, refit, tuning and FiLM."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from bike_demand import data, train
from tests.test_train import SCALING


def toy_parts(rows: int = 96, seed: int = 0) -> dict[str, data.Split]:
    """A tiny problem a network overfits: few noisy training rows, clean validation rows from the same rule."""
    rng = np.random.default_rng(seed)
    def part(n: int, noise: float) -> data.Split:
        x = rng.normal(size=(n, 4)).astype(np.float32)
        s = rng.integers(0, 2, size=(n, 2)).astype(np.float32)
        y = (x[:, 0] * np.where(s[:, 0] > 0, 1.0, -1.0) + noise * rng.normal(size=n)).astype(np.float32)
        frame = pd.DataFrame({"cnt": SCALING.counts(y.astype(np.float64))})
        return data.Split(frame, x, s, y, frame["cnt"].to_numpy())
    return {"train": part(rows, 1.0), "validation": part(256, 0.0)}


class Training(unittest.TestCase):
    on = torch.device("cpu")

    def fit(self, kind: str = "concat", parts=None, **settings):
        parts = parts or toy_parts()
        return train.fit_net(kind, 64, train.Config(0.01, 0.0), 0, parts["train"], parts["validation"], SCALING, self.on,
                             **settings)

    def test_the_returned_network_is_the_one_from_the_best_validation_epoch(self):
        fitted = self.fit(max_epochs=120, patience=120)
        scores = [v for _, v in fitted.history]
        self.assertLess(fitted.best_epoch, len(scores) - 5, "the toy problem must overfit for this test to mean anything")
        validation = toy_parts()["validation"]
        with torch.no_grad():
            prediction = fitted.model(torch.tensor(validation.x), torch.tensor(validation.s))
        self.assertAlmostEqual(train.rmsle_scaled(prediction, torch.tensor(validation.y), SCALING), min(scores), places=5)
        self.assertEqual(fitted.best_epoch, int(np.argmin(scores)))

    def test_training_stops_after_patience_epochs_without_a_better_score(self):
        fitted = self.fit(max_epochs=300, patience=7)
        self.assertEqual(len(fitted.history), fitted.best_epoch + 7 + 1)
        self.assertLess(len(fitted.history), 300)

    def test_the_same_seed_gives_the_same_run(self):
        self.assertEqual(self.fit(max_epochs=5, patience=5).history, self.fit(max_epochs=5, patience=5).history)

    def test_the_refit_trains_exactly_the_chosen_number_of_epochs_and_matches_the_tuning_run(self):
        parts = toy_parts()
        fixed = train.fit_fixed("concat", 64, train.Config(0.01, 0.0), 0, parts["train"], 9, self.on)
        tuned = self.fit(parts=parts, max_epochs=9, patience=9)
        self.assertEqual(len(fixed.history), 9)
        self.assertEqual([loss for loss, _ in fixed.history], [loss for loss, _ in tuned.history])

    def test_tuning_keeps_the_setting_with_the_best_validation_score_and_its_epoch_count(self):
        parts = toy_parts()
        grid = (train.Config(1e-6, 0.0), train.Config(0.01, 0.0))  # the first one barely moves
        with mock.patch.object(train, "GRID", grid), mock.patch.object(train, "TUNING_SEEDS", (0, 1)), \
                mock.patch.object(train, "MAX_EPOCHS", 30), mock.patch.object(train, "PATIENCE", 30):
            tuned = train.tune_net("concat", 32, parts["train"], parts["validation"], SCALING, self.on)
        self.assertEqual(tuned["config"], grid[1])
        self.assertLess(tuned["scores"]["lr=0.01 wd=0.0"], tuned["scores"]["lr=1e-06 wd=0.0"])
        self.assertEqual(len(tuned["histories"]), 2)
        best_epochs = [int(np.argmin([v for _, v in history])) + 1 for history in tuned["histories"]]
        self.assertEqual(tuned["epochs"], int(round(np.mean(best_epochs))))
        self.assertAlmostEqual(tuned["scores"]["lr=0.01 wd=0.0"],
                               float(np.mean([min(v for _, v in history) for history in tuned["histories"]])))

    def test_film_learns_a_rule_that_flips_with_the_state(self):
        parts = toy_parts(rows=512)
        best = {kind: min(v for _, v in self.fit(kind, parts, max_epochs=60, patience=60).history) for kind in ("film", "no-state")}
        self.assertLess(best["film"], 0.5 * best["no-state"])


if __name__ == "__main__":
    unittest.main()
