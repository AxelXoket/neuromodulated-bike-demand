"""docs/rapor.pdf and the helpers that write it: page count, links, numbers taken from results/, clean lines."""
import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import unittest
from unittest import mock

import fitz
import numpy as np
import pandas as pd

from tests import ROOT

import build_report
from bike_demand import data
from figures import LABELS, ORDER, number

PDF = ROOT / "docs" / "rapor.pdf"
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
LONG_DASHES = (chr(0x2013), chr(0x2014))


def pdf_text() -> str:
    with fitz.open(PDF) as doc:
        return "\n".join(page.get_text() for page in doc)


def flat(text: str) -> str:
    """The text with line breaks and no-break spaces turned into plain spaces, for phrase searches."""
    return re.sub(r"\s+", " ", text.replace(" ", " ").replace("‑", "-"))


class TheReport(unittest.TestCase):
    text = pdf_text()

    def test_the_report_has_the_suggested_three_to_five_pages(self):
        with fitz.open(PDF) as doc:
            self.assertIn(doc.page_count, (3, 4, 5))

    def test_the_links_to_the_code_the_notebook_and_the_data_are_clickable(self):
        with fitz.open(PDF) as doc:
            uris = {link["uri"] for page in doc for link in page.get_links() if "uri" in link}
        self.assertLessEqual({build_report.REPO_URL, build_report.COLAB_URL, data.SOURCE_URL}, uris)
        self.assertIn("doi:10.1007/s13748-013-0040-3", flat(self.text))

    def test_every_model_s_test_rmsle_is_printed_as_measured(self):
        printed = flat(self.text)
        for name in ORDER:
            value = METRICS["test"][name]["rmsle"]
            value = value["mean"] if isinstance(value, dict) else value
            with self.subTest(model=name):
                self.assertIn(number(value, 3), printed)
                self.assertIn(LABELS[name], printed)

    def test_the_verdict_on_the_main_question_follows_the_interval(self):
        primary = METRICS["comparisons"]["film - concat"]["bootstrap"]
        printed = flat(self.text)
        self.assertIn(build_report.verdict("film", "concat", primary), printed)
        self.assertIn(build_report.interval(primary), printed)

    def test_no_long_dash_no_word_split_at_a_hyphen_and_no_number_cut_from_its_word(self):
        self.assertFalse(any(dash in self.text for dash in LONG_DASHES))
        lines = [line.strip() for line in self.text.splitlines() if line.strip()]
        self.assertGreater(len(lines), 200)
        split_words = [line for line in lines if re.search(r"[^\W\d]-$", line)]
        self.assertEqual(split_words, [])
        cut = [f"{a} | {b}" for a, b in zip(lines, lines[1:]) if re.search(r"(?<![\w.,])\d+(?:[.,]\d+)*$", a)
               and re.match(r"[a-zçğıöşü]", b) and not re.match(r"Şekil|\d", a)]
        self.assertEqual(cut, [])

    def test_no_paragraph_ends_on_a_line_with_a_single_word(self):
        """A paragraph here is a block whose lines sit one under another from the same left edge; cells side by side in
        a table row are not one."""
        lone, paragraphs = [], 0
        with fitz.open(PDF) as doc:
            for page in doc:
                for block in page.get_text("dict")["blocks"]:
                    lines = [("".join(span["text"] for span in line["spans"]).strip(), line["bbox"]) for line in block.get("lines", [])]
                    lines = [(text, box) for text, box in lines if text]
                    stacked = all(b[1] > a[1] + 2 and abs(b[0] - a[0]) < 3 for (_, a), (_, b) in zip(lines, lines[1:]))
                    if len(lines) >= 2 and stacked:
                        paragraphs += 1
                        if len(lines[-1][0].split()) == 1 and not lines[-2][0].endswith("/"):  # an address breaks only at a slash
                            lone.append(f"p{page.number + 1}: {lines[-2][0][-30:]} | {lines[-1][0]}")
        self.assertGreater(paragraphs, 20)
        self.assertEqual(lone, [])

    def test_the_printed_addresses_are_plain_text_that_still_opens_when_copied(self):
        self.assertNotIn(chr(0x2011), self.text)
        joined = "".join(self.text.split())
        for url in (build_report.REPO_URL, build_report.COLAB_URL, data.SOURCE_URL):
            with self.subTest(url=url):
                self.assertIn(url, joined)


class Helpers(unittest.TestCase):
    def test_the_verdict_has_three_outcomes(self):
        self.assertIn("ayırt edilemiyor", build_report.verdict("film", "concat", {"low": -0.01, "high": 0.02}))
        self.assertIn("MLP FiLM, MLP durum girdide modelinden daha iyi",
                      build_report.verdict("film", "concat", {"low": -0.03, "high": -0.01}))
        self.assertIn("MLP durum girdide, MLP FiLM modelinden daha iyi",
                      build_report.verdict("film", "concat", {"low": 0.01, "high": 0.03}))

    def test_numbers_use_a_dot_for_thousands_and_a_comma_for_decimals(self):
        self.assertEqual(number(17379), "17.379")
        self.assertEqual(number(1234.56, 1), "1.234,6")
        self.assertEqual(number(0.4330, 3), "0,433")

    def test_signed_numbers_and_intervals_use_turkish_decimals(self):
        self.assertEqual(build_report.signed(0.0123, 3), "+0,012")
        self.assertEqual(build_report.signed(-0.0089, 3), "-0,009")
        self.assertEqual(build_report.signed(0.0001, 3), "0,000")
        self.assertEqual(build_report.interval({"mean": 0.002, "low": -0.009, "high": 0.012}), "+0,002 [-0,009; +0,012]")

    def test_a_number_stays_with_its_word_and_a_hyphenated_word_stays_whole(self):
        self.assertEqual(build_report.e("bütün deney 202 sn"), "bütün deney&nbsp;202&nbsp;sn")
        self.assertEqual(build_report.e("%23 payı"), "%23&nbsp;payı")
        self.assertEqual(build_report.e("Ekim-Aralık"), '<span class="nw">Ekim-Aralık</span>')
        self.assertEqual(build_report.e("a < b"), "a &lt;&nbsp;b")

    def test_a_paragraph_never_ends_on_a_lone_word(self):
        self.assertEqual(build_report.e("bir iki üç dört"), "bir iki üç&nbsp;dört")
        self.assertEqual(build_report.e("tek"), "tek")

    def test_a_link_breaks_only_after_a_slash_and_its_text_stays_plain(self):
        shown = build_report.link("https://x.io/a-b/c")
        self.assertEqual(shown, '<a href="https://x.io/a-b/c">https:/<wbr>/<wbr>x.io/<wbr><span class="nw">a-b</span>/<wbr>c</a>')

    def test_a_stuck_feels_like_day_is_one_where_atemp_holds_still_while_temp_moves(self):
        days = pd.to_datetime(["2012-01-01"] * 3 + ["2012-01-02"] * 3 + ["2012-01-03"] * 3)
        raw = pd.DataFrame({"dteday": days, "atemp": [0.2, 0.3, 0.4, 0.24, 0.24, 0.24, 0.5, 0.5, 0.5],
                            "temp": [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.5, 0.5, 0.5]})
        self.assertEqual(build_report.stuck_atemp_days(raw), [pd.Timestamp("2012-01-02")])

    def test_days_of_one_year_name_the_year_once(self):
        same = [pd.Timestamp("2012-08-17"), pd.Timestamp("2012-12-29")]
        self.assertEqual(build_report.days_in_one_year(same), "17 Ağustos ve 29 Aralık 2012 günlerinde")
        mixed = [pd.Timestamp("2011-03-10"), pd.Timestamp("2012-12-29")]
        self.assertEqual(build_report.days_in_one_year(mixed), "10 Mart 2011 ve 29 Aralık 2012 günlerinde")

    def test_the_unusual_hours_are_split_by_side_year_month_and_weather(self):
        rows = 20
        frame = pd.DataFrame({"hr": 8, "workingday": 1, "cnt": [100] * rows, "yr": 1, "mnth": 6, "weathersit": 1,
                              "dteday": pd.date_range("2012-06-01", periods=rows, freq="D")})
        columns = ["cnt", "yr", "mnth", "weathersit", "dteday"]
        frame.loc[0, columns] = [2, 0, 1, 3, pd.Timestamp("2011-01-05")]  # low, first year, January, a wet day
        frame.loc[1, "cnt"] = 3  # low, second year, dry
        frame.loc[2, columns] = [9000, 0, 3, 1, pd.Timestamp("2011-03-05")]  # high, first year, dry
        frame.loc[3, columns] = [4, 1, 1, 1, pd.Timestamp("2012-01-10")]  # low, January of the second year
        self.assertEqual(build_report.unusual_breakdown(frame), {"low": 3, "first_year": 2, "january": 1, "wet_hours": 1})

    def test_the_worst_hours_of_a_day_come_in_time_order_with_their_share(self):
        time = pd.date_range("2012-11-07", periods=24, freq="h")
        actual = np.full(24, 100.0)
        predicted = actual.copy()
        predicted[[2, 0, 1]] = [5.0, 10.0, 20.0]
        predicted[10] = 110.0
        predictions = pd.DataFrame({"time": time, "cnt": actual, "m": predicted})
        worst = build_report.worst_hours(predictions, "m", "2012-11-07")
        self.assertEqual(worst["hours"], [0, 1, 2])
        self.assertEqual(worst["predicted"], [10, 20, 5])
        self.assertEqual(worst["actual"], [100, 100, 100])
        self.assertGreater(worst["share"], 0.99)
        self.assertLess(worst["share"], 1.0)

    def test_the_holiday_range_covers_only_the_holidays_before_the_test_months(self):
        time = pd.to_datetime(["2012-07-04 08:00", "2012-07-04 09:00", "2012-09-03 08:00", "2012-12-25 08:00"])
        frame = pd.DataFrame({"time": time, "dteday": time.normalize(), "holiday": 1, "cnt": [300, 200, 50, 5]})
        self.assertEqual(build_report.holiday_range(frame), (50, 500))

    def test_worst_days_rank_the_days_by_their_share_of_the_log_error(self):
        time = pd.date_range("2012-12-01", periods=72, freq="h")
        actual = np.full(72, 100.0)
        predicted = actual.copy()
        predicted[24:48] = 300.0  # the second day is far off
        predicted[48:50] = 150.0  # the third day a little
        predictions = pd.DataFrame({"time": time, "cnt": actual, "m": predicted})
        worst = build_report.worst_days(predictions, "m", count=2)
        self.assertEqual([day.day for day in worst.index], [2, 3])
        self.assertAlmostEqual(float(worst["share"].sum()), 1.0)
        self.assertAlmostEqual(build_report.share_of_days(predictions, "m", ["2012-12-02"]), float(worst["share"].iloc[0]))
        self.assertAlmostEqual(build_report.day_ratio(predictions, "m", "2012-12-02"), 3.0)
        ranks = [build_report.day_rank(predictions, "m", day) for day in ("2012-12-01", "2012-12-02", "2012-12-03")]
        self.assertEqual(ranks, [3, 1, 2])

    def test_the_conclusion_follows_the_interval_and_names_the_best_model(self):
        unclear = {"mean": 0.002, "low": -0.009, "high": 0.012}
        self.assertIn("desteklenmedi", build_report.conclusion(unclear, "hgb"))
        self.assertIn("destek buldu", build_report.conclusion({"mean": -0.02, "low": -0.03, "high": -0.01}, "hgb"))
        self.assertIn("çürüdü", build_report.conclusion({"mean": 0.02, "low": 0.01, "high": 0.03}, "hgb"))
        self.assertIn(LABELS["hgb"], build_report.conclusion(unclear, "hgb"))
        self.assertIn("+0,002 [-0,009; +0,012]", build_report.conclusion(unclear, "hgb"))
        versus = {"mean": -0.004, "low": -0.024, "high": 0.016}
        with_versus = build_report.conclusion(unclear, "hgb", versus)
        self.assertIn("-0,004 [-0,024; +0,016]", with_versus)
        self.assertIn("ayırt edilemiyor", with_versus.split("test farkı")[1])


class Rebuild(unittest.TestCase):
    """tools/rebuild.py, the one command that remakes results, figures, report and notebook."""

    def test_a_failed_step_stops_the_rebuild(self):
        import rebuild
        real_run, commands = subprocess.run, []

        def second_step_fails(command, **kwargs):
            commands.append(command)
            return real_run([sys.executable, "-c", f"raise SystemExit({int(len(commands) == 2)})"], **kwargs)

        with mock.patch.object(rebuild.subprocess, "run", second_step_fails), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                rebuild.main()
        self.assertEqual(len(commands), 2)

    def test_every_step_runs_a_real_script_with_this_python_from_the_project_root(self):
        import rebuild
        with mock.patch.object(rebuild.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            rebuild.main()
        self.assertEqual(run.call_count, 4)
        for call in run.call_args_list:
            command = call.args[0]
            with self.subTest(command=command):
                self.assertEqual(command[0], sys.executable)
                self.assertEqual(call.kwargs["cwd"], ROOT)
                found = (importlib.util.find_spec(command[2]) is not None if command[1] == "-m"
                         else (ROOT / command[1]).is_file())
                self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
