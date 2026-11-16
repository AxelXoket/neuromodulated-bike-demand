"""Every figure of the report, drawn from data/hour.csv and results/. Run: python tools/figures.py"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bike_demand import data, models  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
DPI = 200
INK, MUTED, GRID = "#1E293B", "#64748B", "#E2E8F0"
COLOURS = {"ridge": "#94A3B8", "hgb": "#0F766E", "no-state": "#F59E0B", "concat": "#2563EB", "shift": "#7C3AED",
           "film": "#DB2777", "actual": "#1E293B"}
LABELS = {"ridge": "Ridge", "hgb": "Gradient boosting", "no-state": "MLP durumsuz", "concat": "MLP durum girdide",
          "shift": "MLP yalnız kaydırma", "film": "MLP FiLM"}
ORDER = ("ridge", "hgb", *models.KINDS)
WEATHER_NAMES = {1: "Açık", 2: "Sisli, bulutlu", 3: "Yağmur, kar"}
# Each label ends left of its dip, in empty space below the line: (day of the dip, label, where the label ends)
EVENTS = (("2012-10-29", "Sandy kasırgası", ("2012-10-16", 1500)), ("2012-12-25", "Noel", ("2012-12-16", 450)))
NUMERIC_LABEL = re.compile(r"-?\d+(?:\.\d+)?")
MINUS_SIGN = chr(0x2212)  # matplotlib prints negative ticks with the minus sign, not a hyphen

PREFERRED_FONT, FALLBACK_FONT = "Segoe UI", "DejaVu Sans"  # matplotlib ships DejaVu Sans; Colab has no Segoe UI


def font_family(installed: set[str]) -> list[str]:
    """The report font where it is installed, else only the bundled one: a missing name would log a warning per figure."""
    return [PREFERRED_FONT, FALLBACK_FONT] if PREFERRED_FONT in installed else [FALLBACK_FONT]


plt.rcParams.update({"font.family": font_family({font.name for font in font_manager.fontManager.ttflist}), "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def number(value: float, decimals: int = 0) -> str:
    """Turkish number format: 1.234,5"""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def turkish_ticks(figure: plt.Figure) -> None:
    """Numeric tick labels in the format of the text (0,05 and 1.500); date axes and category labels stay as drawn."""
    figure.canvas.draw()  # the tick labels exist only once the layout is drawn
    for ax in figure.axes:
        for axis, set_limits in ((ax.xaxis, ax.set_xlim), (ax.yaxis, ax.set_ylim)):
            if not isinstance(axis.get_major_formatter(), (ticker.ScalarFormatter, ticker.FuncFormatter, ticker.LogFormatter)):
                continue
            labels = [label.get_text().replace(MINUS_SIGN, "-") for label in axis.get_majorticklabels()]
            if not any(labels) or not all(NUMERIC_LABEL.fullmatch(text) for text in labels if text):
                continue
            limits = tuple(axis.get_view_interval())  # a copy: the interval is a live view that set_ticks changes
            axis.set_ticks(axis.get_majorticklocs(), labels=[number(float(text), len(text.partition(".")[2])) if text else ""
                                                             for text in labels])  # each label keeps its own decimals
            set_limits(*limits)  # fixed ticks would otherwise widen the axis to the outermost tick


def save(figure: plt.Figure, name: str) -> pathlib.Path:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.png"
    turkish_ticks(figure)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", metadata={"Software": None})
    plt.close(figure)
    return path


# ---------- data exploration ----------

def hourly_profile(frame: pd.DataFrame) -> pathlib.Path:
    figure, ax = plt.subplots(figsize=(4.6, 2.35))
    for working, label, colour in ((1, "İş günü", COLOURS["concat"]), (0, "Hafta sonu ve tatil", COLOURS["film"])):
        mean = frame[frame["workingday"] == working].groupby("hr")["cnt"].mean()
        ax.plot(mean.index, mean.to_numpy(), color=colour, lw=2, label=label)
    ax.set(xlabel="Saat", ylabel="Ortalama kiralama", xticks=range(0, 24, 3))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    return save(figure, "eda_saatlik_profil")


def distribution(frame: pd.DataFrame) -> pathlib.Path:
    figure, (left, right) = plt.subplots(1, 2, figsize=(4.6, 2.4))
    left.hist(frame["cnt"], bins=40, color=COLOURS["ridge"])
    left.set(xlabel="Saatlik kiralama", ylabel="Saat sayısı")
    right.hist(np.log1p(frame["cnt"]), bins=40, color=COLOURS["hgb"])
    right.set(xlabel="log(1 + kiralama)", ylabel="Saat sayısı")
    figure.tight_layout()
    return save(figure, "eda_dagilim")


def daily_series(frame: pd.DataFrame) -> pathlib.Path:
    daily = frame.groupby("dteday")["cnt"].sum()
    figure, ax = plt.subplots(figsize=(9.4, 2.3))
    shades = {"train": ("Eğitim", "#F1F5F9"), "validation": ("Doğrulama", "#E0F2FE"), "test": ("Test", "#FCE7F3")}
    for name, (first, last) in data.SPLITS.items():
        label, colour = shades[name]
        ax.axvspan(pd.Timestamp(first), pd.Timestamp(last) + pd.Timedelta(days=1), color=colour, lw=0)
        ax.text(pd.Timestamp(first) + pd.Timedelta(days=6), daily.max() * 1.02, label, color=MUTED, fontsize=8, va="bottom")
    ax.plot(daily.index, daily.to_numpy(), color=INK, lw=0.8)
    for day, label, (label_day, label_y) in EVENTS:
        when = pd.Timestamp(day)
        ax.annotate(label, (when, daily[when]), xytext=(pd.Timestamp(label_day), label_y), fontsize=8, color=INK,
                    va="center", ha="right", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6, "shrinkA": 2, "shrinkB": 1})
    ax.set(ylabel="Günlük kiralama", ylim=(0, daily.max() * 1.12))
    ax.margins(x=0.005)
    return save(figure, "eda_gunluk_seri")


def weather_effect(frame: pd.DataFrame) -> pathlib.Path:
    rush = frame[(frame["workingday"] == 1) & frame["hr"].isin([8, 17, 18])]
    groups = [rush[rush["weathersit"] == w]["cnt"].to_numpy() for w in data.WEATHER]
    figure, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.boxplot(groups, tick_labels=[f"{WEATHER_NAMES[w]}\n({number(len(g))} saat)" for w, g in zip(data.WEATHER, groups)],
               widths=0.5, medianprops={"color": COLOURS["film"], "lw": 1.6}, flierprops={"markersize": 2, "alpha": 0.4})
    ax.set(ylabel="Kiralama (iş günü 8, 17, 18)")
    return save(figure, "eda_hava")


# ---------- results ----------

def load_results() -> tuple[dict, pd.DataFrame]:
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["time"])
    return metrics, predictions


def best_on_validation(metrics: dict) -> str:
    """The model with the lowest validation RMSLE of its chosen setting: picked without looking at the test months."""
    return min(ORDER, key=lambda name: min(metrics["tuning"][name].values()))


def mean_of(result) -> float:
    return result["mean"] if isinstance(result, dict) else result


def comparison(metrics: dict) -> pathlib.Path:
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 2.1), sharey=True)
    for ax, key, title, decimals, pad in ((axes[0], "rmsle", "Test RMSLE (birincil)", 3, 0.015),
                                          (axes[1], "rmse", "Test RMSE (kiralama/saat)", 1, 3.0)):
        for i, name in enumerate(ORDER):
            value = mean_of(metrics["test"][name][key])
            ax.barh(i, value, color=COLOURS[name], height=0.6)
            if name in metrics["test_runs"]:
                runs = [r[key] for r in metrics["test_runs"][name]]
                ax.scatter(runs, [i] * len(runs), s=5, color=INK, zorder=3)
                value = max(value, max(runs))
            ax.text(value + pad, i, number(mean_of(metrics["test"][name][key]), decimals), va="center", fontsize=8, color=INK)
        ax.set(xlabel=title)
        ax.grid(axis="y", visible=False)
    axes[0].set(yticks=range(len(ORDER)), yticklabels=[LABELS[n] for n in ORDER])
    axes[0].invert_yaxis()
    figure.tight_layout()
    return save(figure, "sonuc_karsilastirma")


def tuning_curves(metrics: dict, kind: str) -> np.ndarray:
    """(seeds, epochs, 2) training loss and validation RMSLE of the chosen setting's tuning seeds, cut to the shortest
    run so that every epoch shown has every seed."""
    runs = metrics["histories"][kind]
    length = min(len(run) for run in runs)
    return np.asarray([run[:length] for run in runs], dtype=float)


def learning_curves(metrics: dict) -> pathlib.Path:
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.4, 2.1))
    for kind in models.KINDS:
        curves = tuning_curves(metrics, kind)
        epochs = np.arange(1, curves.shape[1] + 1)
        for ax, column in ((left, 0), (right, 1)):
            values = curves[:, :, column]
            ax.plot(epochs, values.mean(0), color=COLOURS[kind], lw=1.2, label=LABELS[kind],
                    ls="--" if kind == "shift" else "-")  # dashed, or it hides under the concat line
            ax.fill_between(epochs, values.min(0), values.max(0), color=COLOURS[kind], alpha=0.15, lw=0)
        chosen = metrics["chosen"][kind]["epochs"]
        right.scatter([chosen], [curves[:, chosen - 1, 1].mean()], color=COLOURS[kind], s=16, zorder=3)
    left.set(xlabel="Epoch", ylabel="Eğitim kaybı (MSE)", yscale="log")
    left.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    left.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    left.yaxis.set_minor_formatter(ticker.NullFormatter())
    right.set(xlabel="Epoch", ylabel="Doğrulama RMSLE")
    figure.tight_layout()
    figure.legend(*left.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(models.KINDS))
    return save(figure, "sonuc_ogrenme")


def hourly(predictions: pd.DataFrame) -> pd.DataFrame:
    """Every hour of the test months, with the hours the system recorded nothing left empty instead of bridged."""
    every = pd.date_range(predictions["time"].min().normalize(), predictions["time"].max().normalize() + pd.Timedelta(hours=23),
                          freq="h")
    return predictions.set_index("time").reindex(every)


def actual_vs_predicted(predictions: pd.DataFrame, best: str) -> pathlib.Path:
    window = hourly(predictions).loc["2012-10-22":"2012-11-04 23:00"]
    figure, ax = plt.subplots(figsize=(9.4, 2.4))
    ax.plot(window.index, window["cnt"], color=COLOURS["actual"], lw=1.1, label="Gerçek")
    shown = dict.fromkeys((best, "film"))
    for name in shown:
        ax.plot(window.index, window[name], color=COLOURS[name], lw=1.0, ls="-" if name == best else "--",
                label=LABELS[name] + ("" if name in ("ridge", "hgb") else " (10 seed ortalaması)"))
    empty = window.index[window["cnt"].isna()]
    ax.axvspan(empty.min(), empty.max() + pd.Timedelta(hours=1), color="#FCE7F3", lw=0)
    ax.text(empty.min() + pd.Timedelta(hours=2), window["cnt"].max() * 0.9, "Sandy:\nkayıt yok", fontsize=8, color=INK)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.set(ylabel="Kiralama/saat")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    ax.margins(x=0.005)
    return save(figure, "sonuc_tahmin")


def error_breakdown(predictions: pd.DataFrame, test_weather: pd.Series, best: str) -> pathlib.Path:
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.4, 2.4))
    shown = list(dict.fromkeys((best, "film", "concat")))
    hours = predictions["time"].dt.hour
    for name in shown:
        error = (predictions[name] - predictions["cnt"]).abs()
        by_hour = error.groupby(hours).mean()
        left.plot(by_hour.index, by_hour.to_numpy(), color=COLOURS[name], lw=1.5, label=LABELS[name])
    left.set(xlabel="Saat", ylabel="Ortalama mutlak hata", xticks=range(0, 24, 3))
    figure.legend(*left.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(shown))
    width = 0.8 / len(shown)
    for i, name in enumerate(shown):
        log_error = (np.log1p(predictions[name]) - np.log1p(predictions["cnt"])) ** 2
        by_weather = np.sqrt(log_error.groupby(test_weather.to_numpy()).mean())
        right.bar(np.arange(len(data.WEATHER)) + (i - (len(shown) - 1) / 2) * width, by_weather.reindex(data.WEATHER).to_numpy(),
                  width=width, color=COLOURS[name])
    counts = test_weather.value_counts()
    right.set(xticks=range(len(data.WEATHER)), ylabel="RMSLE",
              xticklabels=[f"{WEATHER_NAMES[w]}\n({number(counts.get(w, 0))} saat)" for w in data.WEATHER])
    right.grid(axis="x", visible=False)
    figure.tight_layout()
    return save(figure, "sonuc_hata")


def state_profiles(metrics: dict) -> pathlib.Path:
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 2.5), sharey=True)
    styles = {"Pazartesi, açık": (COLOURS["concat"], "-"), "Pazartesi, yağmurlu": (COLOURS["concat"], "--"),
              "Cumartesi, açık": (COLOURS["film"], "-"), "Cumartesi, yağmurlu": (COLOURS["film"], "--")}
    for ax, kind in zip(axes, ("concat", "film")):
        for label, values in metrics["profiles"][kind].items():
            colour, line = styles[label]
            ax.plot(data.HOURS, values, color=colour, ls=line, lw=1.5, label=label)
        ax.set(title=LABELS[kind], xlabel="Saat", xticks=range(0, 24, 3))
        ax.title.set_fontsize(9)
    axes[0].set(ylabel="Tahmin (kiralama/saat)")
    figure.legend(*axes[1].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=4)
    figure.tight_layout()
    return save(figure, "sonuc_durum_profili")


def results_figures() -> list[pathlib.Path]:
    metrics, predictions = load_results()
    best = best_on_validation(metrics)
    test = data.split(data.load())["test"]
    return [comparison(metrics), learning_curves(metrics), actual_vs_predicted(predictions, best),
            error_breakdown(predictions, test["weathersit"], best), state_profiles(metrics)]


if __name__ == "__main__":
    frame = data.load()
    made = [hourly_profile(frame), distribution(frame), daily_series(frame), weather_effect(frame)]
    if (RESULTS / "metrics.json").exists():
        made += results_figures()
    for path in made:
        print(path.relative_to(ROOT))
