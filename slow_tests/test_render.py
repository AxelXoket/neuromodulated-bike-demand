"""The page checks in a real headless Edge: a page that does not fit is refused before any PDF is written."""
import pathlib
import tempfile
import unittest

from tests import ROOT  # noqa: F401

import render

CSS = "body { margin: 0; font: 16px 'Segoe UI'; } .page { width: 400px; height: 200px; overflow: hidden; }"


def report(body: str) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        page = pathlib.Path(tmp) / "page.html"
        page.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}"
                        f"{render.FIT_SCRIPT}</body></html>", encoding="utf-8")
        return render.fit_report(page, 900, 1200)


class FitReport(unittest.TestCase):
    def test_a_fitting_page_has_no_problems(self):
        self.assertEqual(report('<section class="page"><p>Kısa bir satır.</p></section>'), [])

    def test_a_fixed_page_that_overflows_is_reported(self):
        problems = report('<section class="page" data-name="sayfa 1">' + "<p>Satır</p>" * 30 + "</section>")
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("sayfa 1: content "))

    def test_text_wrapping_past_its_limit_is_reported(self):
        problems = report('<p data-max-lines="1" data-name="başlık" style="width:120px">bir iki üç dört beş altı yedi</p>')
        self.assertEqual(len(problems), 1)
        self.assertRegex(problems[0], r"^başlık: [2-9] lines, at most 1$")

    def test_an_image_that_did_not_load_is_reported(self):
        problems = report('<img src="file:///no/such/figure.png" alt="">')
        self.assertEqual(problems, ["image did not load: file:///no/such/figure.png"])


class Printing(unittest.TestCase):
    def test_a_page_is_printed_to_a_whole_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            page, pdf = pathlib.Path(tmp) / "page.html", pathlib.Path(tmp) / "page.pdf"
            page.write_text("<!doctype html><html><body><p>Merhaba</p></body></html>", encoding="utf-8")
            render.html_to_pdf(page, pdf)
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            self.assertTrue(pdf.read_bytes().rstrip().endswith(b"%%EOF"))


if __name__ == "__main__":
    unittest.main()
