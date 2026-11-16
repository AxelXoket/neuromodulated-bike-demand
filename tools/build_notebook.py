"""Build bike_demand.ipynb from the cell list below and execute it in place, so GitHub and Colab show the outputs.

Usage: python tools/build_notebook.py
The cell list is the source of truth; tests/test_notebook.py checks the committed notebook against it.
"""
import pathlib
import sys

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "bike_demand.ipynb"
REPO = "https://github.com/AxelXoket/neuromodulated-bike-demand"

CELLS = [
    ("markdown", f"""# Saatlik bisiklet kiralama tahmini: günün durumu ağa nasıl verilmeli?

[![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelXoket/neuromodulated-bike-demand/blob/main/bike_demand.ipynb)

UCI Bike Sharing verisinde (Washington, 2011-2012, saatlik) kiralama sayısını tahmin ediyorum. Soru: günün durumu (haftanın günü, tatil, hava durumu) ağa sıradan bir girdi olarak mı verilmeli, yoksa gizli birimleri ölçekleyip kaydıran bir ayar sinyali (FiLM) olarak mı? Eşit parametreli dört sinir ağı ve iki klasik model aynı zaman bölmesinde karşılaştırılıyor.

Bu not defteri kayıtlı sonuçları (`results/`) okur ve grafikleri çizer; en sondaki hücre isterseniz bütün deneyi yeniden koşar. Rapor: `docs/rapor.pdf`. Kod: {REPO}"""),
    ("code", """# Setup: in Colab the project is fetched from the public repository; locally it is already here.
import os
import subprocess
import sys

if not os.path.isdir("bike_demand"):
    if not os.path.isdir("neuromodulated-bike-demand"):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/AxelXoket/neuromodulated-bike-demand.git"], check=True)
    os.chdir("neuromodulated-bike-demand")
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]

import json

import pandas as pd
from IPython.display import Image, display

import figures
from bike_demand import data, models, train

pd.set_option("display.width", 120)"""),
    ("markdown", """## 1. Veri

`casual` ve `registered` toplamda `cnt`'yi verdiği için, `atemp` de `temp` ile neredeyse aynı (korelasyon 0,988) ve iki gün boyunca tek değerde takılı olduğu için çıkarılıyor. Bölme zamana göre: eğitim Ocak 2011-Haziran 2012, doğrulama Temmuz-Eylül 2012, test Ekim-Aralık 2012."""),
    ("code", """frame = data.load()
parts = data.split(frame)
print(f"{len(frame)} saat, {frame['dteday'].nunique()} gün, {data.missing_hours(frame)} saat hiç kayıtlı değil")
print({name: len(part) for name, part in parts.items()})
print("x (saatin kendisi):", len(data.INPUT_NAMES), "sütun; s (günün durumu):", len(data.STATE_NAMES), "sütun")
frame.head()"""),
    ("markdown", """## 2. Veri keşfi

İş günlerinde iki tepe, hafta sonu tek bir öğleden sonra eğrisi var: günün türü eğrinin şeklini değiştiriyor. Sayılar sağa çarpık, bu yüzden modeller log(1 + kiralama) öğreniyor."""),
    ("code", """for path in (figures.hourly_profile(frame), figures.distribution(frame), figures.weather_effect(frame), figures.daily_series(frame)):
    display(Image(filename=str(path)))"""),
    ("markdown", """## 3. Sonuçlar

Kayıtlı deney (RTX 5080, 10 seed) `results/metrics.json` içinde. Birincil metrik test RMSLE; ağlarda 10 seed ortalaması."""),
    ("code", """metrics = json.loads(open("results/metrics.json", encoding="utf-8").read())
mean = lambda value: value["mean"] if isinstance(value, dict) else value
table = pd.DataFrame({figures.LABELS[name]: {key: round(mean(value), 3) for key, value in metrics["test"][name].items()}
                      for name in figures.ORDER}).T
table[["rmsle", "rmse", "mae", "r2", "bias"]]"""),
    ("code", """pd.DataFrame({pair: {"fark": round(result["bootstrap"]["mean"], 4), "alt": round(result["bootstrap"]["low"], 4),
                     "üst": round(result["bootstrap"]["high"], 4)} for pair, result in metrics["comparisons"].items()}).T"""),
    ("code", """for path in figures.results_figures():
    display(Image(filename=str(path)))"""),
    ("markdown", """## 4. Deneyi yeniden koşmak

`RUN_TRAINING = True` yapın. GPU varsa kullanılır; RTX 5080'de bütün deney yaklaşık 3,5 dakika sürdü, CPU'da daha uzun sürer. Farklı donanımda sayılar son basamaklarda kayabilir."""),
    ("code", """RUN_TRAINING = False
if RUN_TRAINING:
    train.deterministic()
    outcome = train.run(train.device())
    rerun = pd.DataFrame({name: {key: round(mean(value), 3) for key, value in outcome["metrics"]["test"][name].items()}
                          for name in figures.ORDER}).T
    display(rerun)"""),
]


def build() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                              "language_info": {"name": "python"}})
    notebook.cells = [nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
                      for kind, source in CELLS]
    return notebook


def main() -> None:
    notebook = build()
    NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    for number, cell in enumerate(notebook.cells, 1):
        cell.id = f"cell-{number}"  # fixed ids, so a rebuild changes only what really changed
        cell.get("metadata", {}).pop("execution", None)
    NOTEBOOK.write_bytes((nbformat.writes(notebook).rstrip("\n") + "\n").encode("utf-8"))  # LF on every system
    print(NOTEBOOK.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
