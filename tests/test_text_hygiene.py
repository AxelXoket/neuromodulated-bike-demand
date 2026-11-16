"""Every text file in the repository: no long dash, no banned style word, no trace of who asked or wrote, LF endings."""
import re
import subprocess
import unittest

from tests import ROOT

TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".csv", ".ipynb", ".gitignore", ".gitattributes"}
# Both lists are built from parts so this file does not match itself.
STYLE_BANNED = tuple(a + b for a, b in (("del", "ve"), ("cru", "cial"), ("rob", "ust"), ("comprehen", "sive"), ("nuan", "ced"),
                                         ("multi", "faceted"), ("further", "more"), ("more", "over"), ("piv", "otal"),
                                         ("land", "scape"), ("tap", "estry"), ("under", "score"), ("fos", "ter"),
                                         ("show", "case"), ("intri", "cate"), ("vib", "rant"), ("funda", "mental"),
                                         ("signifi", "cant"), ("inter", "play")))
TRACES = ("own" + "er", "maint" + "ainer", "ag" + "ent", "aj" + "an", "kullanıcı " + "istedi", "talep " + "üzerine",
          "istendiği " + "gibi", "clau" + "de", "anthro" + "pic", "co-" + "authored", "generated " + "with")


# One pass per list over each file instead of one pass per word.
BANNED_PATTERN = re.compile(r"\b(" + "|".join(STYLE_BANNED) + ")")
TRACE_PATTERN = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, TRACES)) + r")(?!\w)")


def kept_byte_for_byte(names: list[str]) -> set[str]:
    """Files whose text attribute is unset in .gitattributes: git stores them as they are, like the UCI data file."""
    report = subprocess.run(["git", "check-attr", "text", "--", *names], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.splitlines()
    return {line.split(": ")[0] for line in report if line.endswith(": text: unset")}


def repo_text_files() -> list:
    """Files git would commit (tracked or new, ignored ones left out), limited to text that git may normalise."""
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    text = [name for name in listed if (ROOT / name).suffix in TEXT_SUFFIXES or name.startswith(".git")]
    kept = kept_byte_for_byte(text)
    return [ROOT / name for name in text if name not in kept]


def problems(text: str) -> list[str]:
    found = [f"long dash U+{ord(c):04X}" for c in (chr(0x2013), chr(0x2014)) if c in text]
    low = text.lower().replace("İ", "i").replace("I", "ı")
    found += [f"banned word {word}" for word in sorted(set(BANNED_PATTERN.findall(low)))]
    found += [f"trace {word}" for word in sorted(set(TRACE_PATTERN.findall(low)))]
    if "\r" in text:
        found.append("CRLF line ending")
    return found


class RepositoryText(unittest.TestCase):
    def test_every_text_file_is_clean(self):
        files = repo_text_files()
        self.assertGreaterEqual(len(files), 15)
        for path in files:
            with self.subTest(file=path.relative_to(ROOT).as_posix()):
                self.assertEqual(problems(path.read_bytes().decode("utf-8")), [])

    def test_the_scan_sees_the_code_the_notebook_and_the_readme(self):
        names = {path.relative_to(ROOT).as_posix() for path in repo_text_files()}
        for expected in ("bike_demand/train.py", "tools/build_report.py", "bike_demand.ipynb", "README.md", "results/metrics.json"):
            self.assertIn(expected, names)

    def test_only_the_upstream_data_file_is_kept_byte_for_byte(self):
        self.assertEqual(kept_byte_for_byte(["data/hour.csv", "results/predictions.csv", "README.md"]), {"data/hour.csv"})


class Detector(unittest.TestCase):
    def test_each_rule_fires_on_a_planted_violation(self):
        self.assertIn(f"long dash U+{0x2014:04X}", problems(f"a {chr(0x2014)} b"))
        self.assertIn(f"banned word {STYLE_BANNED[0]}", problems(f"we {STYLE_BANNED[0]} here"))
        self.assertIn(f"trace {TRACES[0]}", problems(f"the {TRACES[0]} decided"))
        self.assertIn("CRLF line ending", problems("a\r\nb"))

    def test_every_listed_word_fires_on_its_own(self):
        for word in STYLE_BANNED:
            self.assertIn(f"banned word {word}", problems(f"a {word} b"))
        for word in TRACES:
            self.assertIn(f"trace {word}", problems(f"a {word} b"))

    def test_clean_text_passes(self):
        self.assertEqual(problems("Sade bir satır.\nİkinci satır - kısa çizgiyle; FiLM ve gradient boosting.\n"), [])


if __name__ == "__main__":
    unittest.main()
