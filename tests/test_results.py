"""The saved experiment: results/ agrees with itself, with the protocol and with the figures drawn from it."""
import json
import pathlib
import re
import unittest

import numpy as np
import pandas as pd

from tests import ROOT

from bike_demand import data, models, train

METRICS = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
PREDICTIONS = pd.read_csv(ROOT / "results" / "predictions.csv", parse_dates=["time"])
NAMES = ("ridge", "hgb", *models.KINDS)


class Protocol(unittest.TestCase):
    def test_every_model_was_scored_on_the_test_months_once(self):
        self.assertEqual(set(METRICS["test"]), set(NAMES))
        test = data.split(data.load())["test"]
        self.assertEqual(len(PREDICTIONS), len(test))
        np.testing.assert_array_equal(PREDICTIONS["time"].to_numpy(), test["time"].to_numpy())
        np.testing.assert_array_equal(PREDICTIONS["cnt"].to_numpy(), test["cnt"].to_numpy())

    def test_the_networks_were_the_same_size_and_tuned_on_the_same_grid(self):
        sizes = METRICS["parameters"]
        for kind in models.KINDS:
            with self.subTest(kind=kind):
                self.assertLessEqual(abs(sizes[kind] - sizes["concat"]) / sizes["concat"], 0.01)
                self.assertEqual(len(METRICS["tuning"][kind]), len(train.GRID))
                chosen = METRICS["chosen"][kind]
                self.assertIn(train.Config(chosen["lr"], chosen["weight_decay"]), train.GRID)
        self.assertTrue(set(METRICS["final_seeds"]).isdisjoint(METRICS["tuning_seeds"]))
        self.assertEqual(METRICS["primary"], "rmsle")

    def test_the_settings_were_chosen_by_validation_score(self):
        for name in NAMES:
            if name == "ridge":
                continue
            scores = METRICS["tuning"][name]
            chosen = METRICS["chosen"][name]
            key = (f"lr={chosen['learning_rate']} leaves={chosen['max_leaf_nodes']}" if name == "hgb"
                   else f"lr={chosen['lr']} wd={chosen['weight_decay']}")
            with self.subTest(model=name):
                self.assertEqual(min(scores, key=scores.get), key)
        alphas = METRICS["tuning"]["ridge"]
        self.assertEqual(float(min(alphas, key=alphas.get)), METRICS["chosen"]["ridge"]["alpha"])


class SavedNumbersAgree(unittest.TestCase):
    def test_each_seed_s_metrics_can_be_recomputed_from_its_saved_predictions(self):
        actual = PREDICTIONS["cnt"].to_numpy()
        for kind in models.KINDS:
            for seed, run in zip(METRICS["final_seeds"], METRICS["test_runs"][kind]):
                again = train.metrics(actual, PREDICTIONS[f"{kind} s{seed}"].to_numpy())
                with self.subTest(kind=kind, seed=seed):
                    self.assertAlmostEqual(again["rmsle"], run["rmsle"], places=4)
                    self.assertAlmostEqual(again["rmse"], run["rmse"], places=2)
        for name in ("ridge", "hgb"):
            self.assertAlmostEqual(train.metrics(actual, PREDICTIONS[name].to_numpy())["rmsle"], METRICS["test"][name]["rmsle"], places=4)

    def test_the_bootstrap_point_is_the_difference_of_the_seed_averaged_rmsle(self):
        self.assertIn("hgb - concat", METRICS["comparisons"])
        for pair, result in METRICS["comparisons"].items():
            a, b = pair.split(" - ")
            # a network's score is its seed mean; gradient boosting is one fixed model
            mean = {k: np.mean([r["rmsle"] for r in METRICS["test_runs"][k]]) if k in METRICS["test_runs"]
                    else METRICS["test"][k]["rmsle"] for k in (a, b)}
            with self.subTest(pair=pair):
                self.assertAlmostEqual(result["bootstrap"]["mean"], mean[a] - mean[b], places=4)
                self.assertLessEqual(result["bootstrap"]["low"], result["bootstrap"]["mean"])
                self.assertGreaterEqual(result["bootstrap"]["high"], result["bootstrap"]["mean"])
                self.assertEqual(result["bootstrap"]["reps"], train.BOOTSTRAP_REPS)

    def test_the_seed_average_column_is_the_mean_of_the_seed_columns(self):
        for kind in models.KINDS:
            columns = [f"{kind} s{seed}" for seed in METRICS["final_seeds"]]
            with self.subTest(kind=kind):
                np.testing.assert_allclose(PREDICTIONS[columns].mean(axis=1), PREDICTIONS[kind], atol=2e-3)


class Readme(unittest.TestCase):
    def test_the_readme_table_and_intervals_are_the_saved_results(self):
        import build_report
        from figures import LABELS, number
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        rows = {line.split("|")[1].strip(): [cell.strip() for cell in line.split("|")[3:6]]
                for line in readme.splitlines() if line.startswith("| ") and line.count("|") == 6}
        for name in NAMES:
            test = METRICS["test"][name]
            value = {key: test[key]["mean"] if isinstance(test[key], dict) else test[key] for key in ("rmsle", "rmse", "r2")}
            with self.subTest(model=name):
                self.assertEqual(rows[LABELS[name]], [number(value["rmsle"], 3), number(value["rmse"], 1), number(value["r2"], 3)])
        for pair in ("film - concat", "film - shift", "concat - no-state", "hgb - concat"):
            interval = build_report.interval(METRICS["comparisons"][pair]["bootstrap"])
            with self.subTest(pair=pair):
                self.assertIn(interval.split(" [")[1].rstrip("]"), readme)


class FigureChoices(unittest.TestCase):
    def test_the_model_called_best_has_the_lowest_validation_score(self):
        import figures
        best = figures.best_on_validation(METRICS)
        scores = {name: min(METRICS["tuning"][name].values()) for name in NAMES}
        self.assertEqual(scores[best], min(scores.values()))

    def test_hours_with_no_record_stay_empty_in_the_plots(self):
        import figures
        hourly = figures.hourly(PREDICTIONS)
        test = data.split(data.load())["test"]
        self.assertEqual(int(hourly["cnt"].isna().sum()), data.missing_hours(test))
        self.assertEqual(len(hourly), test["dteday"].nunique() * len(data.HOURS))


class Figures(unittest.TestCase):
    EXPECTED = {"eda_saatlik_profil", "eda_dagilim", "eda_gunluk_seri", "eda_hava", "sonuc_karsilastirma", "sonuc_ogrenme",
                "sonuc_tahmin", "sonuc_hata", "sonuc_durum_profili"}

    def test_every_figure_of_the_report_is_there_and_drawn(self):
        from PIL import Image
        found = {path.stem for path in (ROOT / "figures").glob("*.png")}
        self.assertEqual(found, self.EXPECTED)
        for name in self.EXPECTED:
            with Image.open(ROOT / "figures" / f"{name}.png") as image, self.subTest(figure=name):
                self.assertGreater(image.width, 600)
                self.assertGreater(len(np.unique(np.asarray(image.convert("L")))), 20)


class TickLabels(unittest.TestCase):
    """Axis numbers print like the text: a comma for decimals, a dot for thousands; dates stay as drawn."""

    def saved_labels(self, draw) -> tuple:
        import tempfile
        from unittest import mock

        import matplotlib.pyplot as plt

        import figures
        figure, ax = plt.subplots()
        draw(ax)
        limits = (ax.get_xlim(), ax.get_ylim())
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(figures, "FIGURES", pathlib.Path(tmp)), \
                mock.patch.object(figures.plt, "close"):
            figures.save(figure, "probe")
        labels = ([t.get_text() for t in ax.get_xticklabels()], [t.get_text() for t in ax.get_yticklabels()])
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), limits)
        figures.plt.close(figure)
        return labels

    def test_decimals_and_thousands_are_printed_the_turkish_way(self):
        x, y = self.saved_labels(lambda ax: ax.plot([0, 1500], [0.05, 0.3]))
        self.assertIn("1.000", x)
        self.assertIn("0,10", y)
        self.assertFalse(any("." in label for label in y))

    def test_a_log_axis_with_plain_labels_is_converted_too(self):
        import matplotlib.ticker as ticker

        def draw(ax):
            ax.plot([1, 2], [0.05, 0.5])
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        _, y = self.saved_labels(draw)
        self.assertIn("0,1", y)
        self.assertIn("0,05", y)
        self.assertFalse(any("." in label for label in y))

    def test_date_labels_keep_their_dots(self):
        import matplotlib.dates as mdates

        def draw(ax):
            ax.plot(pd.date_range("2012-10-22", periods=5, freq="D"), [1, 2, 3, 4, 5])
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        x, _ = self.saved_labels(draw)
        shown = [label for label in x if label]
        self.assertGreaterEqual(len(shown), 2)
        self.assertTrue(all(re.fullmatch(r"\d\d\.10", label) for label in shown), shown)


class FontChoice(unittest.TestCase):
    """The figures ask only for installed fonts, so a machine without the report font (Colab) draws with no warning."""

    @staticmethod
    def warnings_when_drawn(family: list) -> list:
        import logging

        import matplotlib.pyplot as plt
        records = []
        handler = logging.Handler(level=logging.WARNING)
        handler.emit = records.append
        logger = logging.getLogger("matplotlib.font_manager")
        logger.addHandler(handler)
        try:
            with plt.rc_context({"font.family": family}):
                figure, ax = plt.subplots()
                ax.set_title("Saatlik kiralama")
                figure.canvas.draw()
                plt.close(figure)
        finally:
            logger.removeHandler(handler)
        return [record.getMessage() for record in records]

    def test_a_missing_report_font_is_left_out_and_drawing_logs_no_warning(self):
        import uuid
        from unittest import mock

        import matplotlib.font_manager as font_manager

        import figures
        installed = {font.name for font in font_manager.fontManager.ttflist}
        missing = f"Missing {uuid.uuid4().hex}"
        with mock.patch.object(figures, "PREFERRED_FONT", missing):
            family = figures.font_family(installed)
        self.assertEqual(family, [figures.FALLBACK_FONT])
        self.assertEqual(self.warnings_when_drawn(family), [])
        self.assertNotEqual(self.warnings_when_drawn([missing, figures.FALLBACK_FONT]), [])  # the warning avoided

    def test_an_installed_report_font_comes_first_and_the_figures_use_the_choice(self):
        import matplotlib.font_manager as font_manager
        import matplotlib.pyplot as plt

        import figures
        both = {figures.PREFERRED_FONT, figures.FALLBACK_FONT}
        self.assertEqual(figures.font_family(both), [figures.PREFERRED_FONT, figures.FALLBACK_FONT])
        installed = {font.name for font in font_manager.fontManager.ttflist}
        self.assertEqual(plt.rcParams["font.family"], figures.font_family(installed))


    def test_on_a_machine_without_the_report_font_the_module_picks_only_the_bundled_one(self):
        import importlib
        from unittest import mock

        import matplotlib.font_manager as font_manager
        import matplotlib.pyplot as plt

        import figures
        without = [font for font in font_manager.fontManager.ttflist if font.name != figures.PREFERRED_FONT]
        try:
            with plt.rc_context(), mock.patch.object(font_manager.fontManager, "ttflist", without):
                reloaded = importlib.reload(figures)
                self.assertEqual(plt.rcParams["font.family"], [reloaded.FALLBACK_FONT])
        finally:
            importlib.reload(figures)


class AxisLabels(unittest.TestCase):
    def test_both_panels_of_the_distribution_name_what_they_count(self):
        import tempfile
        from unittest import mock

        import figures
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(figures, "FIGURES", pathlib.Path(tmp)), \
                mock.patch.object(figures.plt, "close"):
            figures.distribution(data.load())
            figure = figures.plt.gcf()
        try:
            self.assertEqual([ax.get_ylabel() for ax in figure.axes], ["Saat sayısı", "Saat sayısı"])
            self.assertTrue(all(ax.get_xlabel() for ax in figure.axes))
        finally:
            figures.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
