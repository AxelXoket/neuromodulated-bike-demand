"""Headless Edge helpers: HTML to PDF and a text fit report read back from the rendered page.

Edge runs with --headless=new, no GPU, no first-run screen and a throwaway profile, so no window ever opens.
"""
import json
import os
import pathlib
import re
import subprocess
import tempfile
import time

EDGE = pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
# Edge writes the hidden flag as an attribute (hidden=""), so the tag may carry more than the id.
FIT_REPORT = re.compile(r'<pre id="fit-report"[^>]*>(.*?)</pre>', re.DOTALL)

# Runs inside every page. Each element with data-max-lines may not wrap past that many lines and nothing may stick out
# of its box; a page, canvas, slide, card or table cell may not hold more than its size. The result lands in
# <pre id="fit-report">.
FIT_SCRIPT = """<script>
addEventListener("load", () => {
  const problems = [];
  const name = (el) => (el.dataset.name || el.textContent.trim().slice(0, 40));
  // Lines are counted from the boxes the text was laid out in, so every line-height counts. With a tight line-height
  // the boxes of two lines overlap, so a box opens a new line when its middle lies below the current line's bottom.
  const lineCount = (el) => {
    const range = document.createRange();
    range.selectNodeContents(el);
    const middle = (r) => (r.top + r.bottom) / 2;
    const boxes = [...range.getClientRects()].filter((r) => r.width > 0 && r.height > 0).sort((a, b) => middle(a) - middle(b));
    let lines = 0, bottom = -Infinity;
    for (const box of boxes) {
      if (middle(box) > bottom) { lines += 1; bottom = box.bottom; }
    }
    return lines;
  };
  for (const el of document.querySelectorAll("[data-max-lines]")) {
    const lines = lineCount(el);
    if (lines > Number(el.dataset.maxLines)) problems.push(`${name(el)}: ${lines} lines, at most ${el.dataset.maxLines}`);
    if (el.scrollWidth > el.clientWidth + 1) problems.push(`${name(el)}: wider than its box`);
  }
  for (const el of document.querySelectorAll(".page, .canvas, .slide, .box, .card, .swatch, td, th")) {
    const label = el.dataset.name || el.className || el.tagName.toLowerCase();
    if (el.scrollHeight > el.clientHeight + 1) problems.push(`${label}: content ${el.scrollHeight}px in a ${el.clientHeight}px box`);
    if (el.scrollWidth > el.clientWidth + 1) problems.push(`${label}: content wider than the box`);
    // scrollHeight only sees overflow at the bottom and right; text pushed up or left (flex-end, negative margins)
    // shows up as a child that starts before the box does.
    const edge = el.getBoundingClientRect();
    const outside = [...el.querySelectorAll("*")].some((child) => {
      const r = child.getBoundingClientRect();
      return (r.width > 0 || r.height > 0) && (r.top < edge.top - 1 || r.left < edge.left - 1);
    });
    if (outside) problems.push(`${label}: content above or left of the box`);
  }
  for (const image of document.images) {
    if (!image.naturalWidth) problems.push(`image did not load: ${image.getAttribute("src").slice(-60)}`);
  }
  const report = document.createElement("pre");
  report.id = "fit-report";
  report.hidden = true;
  report.textContent = JSON.stringify(problems);
  document.body.append(report);
});
</script>"""


def edge_args(profile: str) -> list[str]:
    return [str(EDGE), "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
            "--hide-scrollbars", "--lang=tr-TR", "--accept-lang=tr-TR", "--virtual-time-budget=3000"]


def run_edge(profile: str, args: list[str]) -> subprocess.CompletedProcess:
    # With __COMPAT_LAYER set (Windows sets it for some parent apps) Edge restarts itself without it and the first
    # process exits at once, so --dump-dom returns nothing. Dropping the variable keeps Edge in this one process.
    env = {key: value for key, value in os.environ.items() if key.upper() != "__COMPAT_LAYER"}
    return subprocess.run(edge_args(profile) + args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=env, timeout=120)


def wait_for_file(path: pathlib.Path, ready, timeout: float = 60.0, poll: float = 0.25) -> None:
    """Edge's launcher can exit before the file is complete, so wait until `ready(bytes)` says it is."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if ready(path.read_bytes()):
                return
        except OSError:
            pass
        time.sleep(poll)
    raise SystemExit(f"Edge did not finish {path} within {timeout:.0f} seconds")


def fit_report(html_path: pathlib.Path, width: int, height: int) -> list[str]:
    """Load the page at the given window size and return the fit problems its script found."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        run = run_edge(profile, [f"--window-size={width},{height}", "--dump-dom", html_path.as_uri()])
    match = FIT_REPORT.search(run.stdout)
    if not match:
        raise SystemExit(f"no fit report from {html_path.name}; Edge exit code {run.returncode}, "
                         f"{len(run.stdout)} characters of output, error output: {run.stderr[-300:]!r}")
    return json.loads(match.group(1).replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))


def html_to_pdf(html_path: pathlib.Path, pdf_path: pathlib.Path) -> None:
    pdf_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        run_edge(profile, ["--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.as_uri()])
        wait_for_file(pdf_path, lambda data: data.rstrip().endswith(b"%%EOF"))


def windowed_edges() -> int:
    """How many Edge processes have a visible window; a headless run must leave this number unchanged."""
    script = "@(Get-Process msedge -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 }).Count"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True).stdout.strip()
    return int(out or 0)
