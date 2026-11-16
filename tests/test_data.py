"""The data step: what reaches a model, and that nothing from later months leaks into it."""
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from bike_demand import data

RAW = pd.read_csv(data.DATA, parse_dates=["dteday"])


def prepared_from(raw: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "hour.csv"
        raw.to_csv(path, index=False)
        return data.prepare(path)


class Cleaning(unittest.TestCase):
    def test_the_rider_counts_that_add_up_to_the_target_never_reach_a_model(self):
        noisy = RAW.copy()
        noisy["casual"], noisy["registered"] = noisy["cnt"] * 7, -noisy["cnt"]
        original, _ = data.prepare()
        changed, _ = prepared_from(noisy)
        for name in original:
            with self.subTest(split=name):
                np.testing.assert_array_equal(original[name].x, changed[name].x)
                np.testing.assert_array_equal(original[name].s, changed[name].s)
                pd.testing.assert_frame_equal(data.table(original[name].frame), data.table(changed[name].frame))
        self.assertTrue(set(data.LEAKS).isdisjoint(data.load().columns))

    def test_heavy_weather_joins_the_bad_weather_group(self):
        raw = RAW.head(3).copy()
        raw["weathersit"] = [1, 3, 4]
        first = data.STATE_NAMES.index("weather 1")
        weather = data.state(data.clean(raw))[:, first:first + len(data.WEATHER)]
        np.testing.assert_array_equal(weather, [[1, 0, 0], [0, 0, 1], [0, 0, 1]])

    def test_a_humidity_gap_is_filled_in_time_from_the_hours_around_it(self):
        raw = RAW.head(4).drop(index=2).copy()  # hours 0, 1 and 3: hour 1 sits a third of the way from 0 to 3
        raw["hum"] = [0.5, 0.0, 0.8]
        np.testing.assert_allclose(data.clean(raw)["hum"], [0.5, 0.6, 0.8])

    def test_rows_come_out_in_time_order(self):
        times = data.clean(RAW.sample(frac=1, random_state=0))["time"]
        self.assertTrue(times.is_monotonic_increasing)
        self.assertEqual(len(times), len(RAW))


class Splitting(unittest.TestCase):
    parts = data.split(data.load())

    def test_train_validation_and_test_follow_each_other_in_time_and_cover_every_row(self):
        train, validation, test = (self.parts[name] for name in ("train", "validation", "test"))
        self.assertLess(train["time"].max(), validation["time"].min())
        self.assertLess(validation["time"].max(), test["time"].min())
        self.assertEqual(len(train) + len(validation) + len(test), len(RAW))
        self.assertEqual((str(test["dteday"].min().date()), str(test["dteday"].max().date())), ("2012-10-01", "2012-12-31"))

    def test_the_refit_rows_are_training_and_validation_months_and_no_test_month(self):
        prepared, _ = data.prepare()
        fit = prepared["fit"].frame
        self.assertEqual(len(fit), len(prepared["train"].y) + len(prepared["validation"].y))
        self.assertLess(fit["time"].max(), prepared["test"].frame["time"].min())
        self.assertTrue(fit["time"].is_monotonic_increasing)

    def test_scaling_is_measured_on_the_training_months_only(self):
        later = RAW["dteday"] >= "2012-07-01"
        distorted = RAW.copy()
        distorted.loc[later, "temp"] *= 10
        distorted.loc[later, "cnt"] *= 100
        _, original = data.prepare()
        _, changed = prepared_from(distorted)
        np.testing.assert_array_equal(original.input_mean, changed.input_mean)
        np.testing.assert_array_equal(original.input_std, changed.input_std)
        self.assertEqual((original.target_mean, original.target_std), (changed.target_mean, changed.target_std))


class Features(unittest.TestCase):
    parts, scaling = data.prepare()

    def test_every_one_hot_block_marks_exactly_one_level(self):
        train = self.parts["train"]
        raw = data.raw_inputs(train.frame)
        weather = data.STATE_NAMES.index("weather 1")
        blocks = {"hour": (raw, 0, 24), "weekday": (train.s, 0, 7), "weather": (train.s, weather, weather + 3)}
        for name, (matrix, first, last) in blocks.items():
            with self.subTest(block=name):
                np.testing.assert_array_equal(matrix[:, first:last].sum(1), 1)
        self.assertEqual(raw.shape[1], len(data.INPUT_NAMES))
        self.assertEqual(train.s.shape[1], len(data.STATE_NAMES))

    def test_no_state_column_follows_from_the_hour_month_and_year(self):
        """Within one (hour, month, year) every state column still takes both values, so x cannot stand in for s."""
        frame = self.parts["train"].frame
        groups = frame.groupby(["hr", "mnth", "yr"]).ngroup().to_numpy()
        s = self.parts["train"].s
        for column, name in enumerate(data.STATE_NAMES):
            values_per_group = pd.Series(s[:, column]).groupby(groups).nunique()
            with self.subTest(state=name):
                self.assertGreater(int((values_per_group > 1).sum()), 0)
        self.assertNotIn("workingday", data.TABLE_COLUMNS)
        self.assertNotIn("season", data.TABLE_COLUMNS)

    def test_scaled_weather_columns_have_mean_zero_and_spread_one_on_train(self):
        x = self.parts["train"].x
        columns = [data.INPUT_NAMES.index(name) for name in data.SCALED]
        np.testing.assert_allclose(x[:, columns].mean(0), 0, atol=1e-5)
        np.testing.assert_allclose(x[:, columns].std(0), 1, atol=1e-4)
        self.assertEqual(set(np.unique(x[:, data.INPUT_NAMES.index("hour 8")])), {0.0, 1.0})

    def test_the_target_goes_back_to_the_counts_and_never_below_zero(self):
        test = self.parts["test"]
        np.testing.assert_allclose(self.scaling.counts(test.y), test.counts, rtol=1e-4)
        self.assertEqual(self.scaling.counts(np.array([-50.0]))[0], 0.0)

    def test_months_wrap_around_the_year(self):
        frame = pd.DataFrame({"mnth": [12, 1, 7], "hr": 0, "yr": 0, "temp": 0, "hum": 0, "windspeed": 0})
        month = [data.INPUT_NAMES.index("month sin"), data.INPUT_NAMES.index("month cos")]
        december, january, july = data.raw_inputs(frame)[:, month]
        self.assertLess(np.linalg.norm(december - january), np.linalg.norm(january - july))


class Inspection(unittest.TestCase):
    def test_missing_hours_are_the_hours_of_the_days_with_no_row(self):
        two_days = pd.DataFrame({"dteday": pd.to_datetime(["2012-01-01"] * 24 + ["2012-01-02"] * 23)})
        self.assertEqual(data.missing_hours(two_days), 1)
        self.assertEqual(data.missing_hours(data.load()), 165)

    def test_an_hour_far_from_its_group_is_marked_unusual_and_the_others_are_not(self):
        counts = list(range(90, 111)) + [2]
        frame = pd.DataFrame({"cnt": counts, "hr": 8, "workingday": 1})
        marked = data.unusual_hours(frame)
        self.assertEqual(marked.tolist(), [False] * 21 + [True])


if __name__ == "__main__":
    unittest.main()
