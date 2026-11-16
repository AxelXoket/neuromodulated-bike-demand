"""Mutation check: break one rule at a time and confirm that the tests turn red.

Each mutant in tools/mutants.json replaces one exact piece of text in one file. Everything runs in a temporary copy of
the project, so the working tree is never touched. A mutant is caught when its test lane fails.

Usage: python tools/mutation_check.py              all lanes
       python tools/mutation_check.py --lane fast  only the mutants checked without Edge
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
MUTANTS = ROOT / "tools" / "mutants.json"
LANES = {
    "fast": ["tests.test_data", "tests.test_models", "tests.test_train", "tests.test_results", "tests.test_report",
             "tests.test_notebook", "slow_tests.test_training", "slow_tests.test_counterfactual"],
    "edge": ["slow_tests.test_render"],
}
COPIED = ("bike_demand", "tools", "tests", "slow_tests", "data", "results", "figures", "docs", "bike_demand.ipynb",
          "README.md")


def run_lane(copy: pathlib.Path, lane: str) -> bool:
    """True when every test of the lane passes."""
    run = subprocess.run([sys.executable, "-B", "-m", "unittest", *LANES[lane]], cwd=copy, capture_output=True, text=True,
                         timeout=600)
    return run.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=sorted(LANES))
    args = parser.parse_args()
    mutants = [m for m in json.loads(MUTANTS.read_text(encoding="utf-8")) if args.lane in (None, m["lane"])]
    with tempfile.TemporaryDirectory() as tmp:
        copy = pathlib.Path(tmp)
        for part in COPIED:
            if (ROOT / part).is_dir():
                shutil.copytree(ROOT / part, copy / part, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(ROOT / part, copy / part)
        for lane in {m["lane"] for m in mutants}:
            if not run_lane(copy, lane):
                raise SystemExit(f"the {lane} lane is red before any mutation")
        missed = []
        for number, mutant in enumerate(mutants, 1):
            path = copy / mutant["file"]
            original = path.read_bytes()
            before = hashlib.md5(original).hexdigest()
            text = original.decode("utf-8")
            if text.count(mutant["old"]) != 1:
                raise SystemExit(f"mutant {number} does not match exactly once in {mutant['file']}: {mutant['old']!r}")
            try:
                path.write_bytes(text.replace(mutant["old"], mutant["new"]).encode("utf-8"))
                caught = not run_lane(copy, mutant["lane"])
            finally:
                path.write_bytes(original)
                if hashlib.md5(path.read_bytes()).hexdigest() != before:
                    raise SystemExit(f"mutant {number}: {mutant['file']} was not restored")
            print(f"{number:2d} {'caught' if caught else 'MISSED'}  {mutant['file']}: {mutant['why']}")
            if not caught:
                missed.append(number)
    print(f"\n{len(mutants) - len(missed)} of {len(mutants)} mutants caught")
    if missed:
        raise SystemExit(f"missed: {missed}")


if __name__ == "__main__":
    main()
