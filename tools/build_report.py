"""The project report (docs/rapor.pdf), written from data/hour.csv and results/: every number on the page is computed.

Run after python -m bike_demand.train and python tools/figures.py:  python tools/build_report.py
"""
from __future__ import annotations

import html
import json
import pathlib
import re
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import render  # noqa: E402
from bike_demand import data, models, train  # noqa: E402
from figures import LABELS, ORDER, best_on_validation, number, tuning_curves  # noqa: E402

DOCS = ROOT / "docs"
REPO_URL = "https://github.com/AxelXoket/neuromodulated-bike-demand"
NOTEBOOK = "bike_demand.ipynb"
COLAB_URL = f"https://colab.research.google.com/github/AxelXoket/neuromodulated-bike-demand/blob/main/{NOTEBOOK}"
CITATION = ("Fanaee-T, H. ve Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. "
            "Progress in Artificial Intelligence, 2, 113-127. doi:10.1007/s13748-013-0040-3")
FILM_CITATION = "Perez, E. ve diğerleri (2018). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018."
DAYS_TR = ("Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar")
MONTHS_TR = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")
WEATHER_TR = {1: "açık", 2: "sisli, bulutlu", 3: "yağmurlu"}
SANDY = ("2012-10-29", "2012-10-30")
CHRISTMAS = ("2012-12-24", "2012-12-25", "2012-12-26")
AFTER_THANKSGIVING = "2012-11-23"
HUMID_SATURDAY = "2012-12-08"  # a dry Saturday in very humid air
HUMID = 0.9  # humidity from which an hour counts as very humid
# A number and the word after it stay on one line ("202 sn", "%23 payı"), and so do the parts of a hyphenated word.
NUMBER_AND_WORD = re.compile(r"(?<![\w.,])(%?\d+(?:[.,]\d+)*) (?=[^\W\d])")
HYPHENATED = re.compile(r"[\w%]+(?:-[\w%]+)+")
LAST_SPACE = re.compile(r" (?=\S+$)")
ELECTION_NIGHT = "2012-11-07"  # the hours after the 6 November 2012 presidential election


# ---------- text helpers ----------

def e(text: str) -> str:
    escaped = NUMBER_AND_WORD.sub("\\1&nbsp;", html.escape(text, quote=False))
    escaped = LAST_SPACE.sub("&nbsp;", escaped)  # the last two words share a line: no paragraph ends on a lone word
    return HYPHENATED.sub(lambda m: f'<span class="nw">{m.group(0)}</span>', escaped)


def link(url: str) -> str:
    """A visible address may break only after a slash, never at a hyphen inside a name. The text stays plain ASCII,
    so an address copied from the page still opens."""
    segments = [f'<span class="nw">{part}</span>' if "-" in part else part for part in html.escape(url).split("/")]
    return f'<a href="{html.escape(url)}">{"/<wbr>".join(segments)}</a>'


def pct(share: float, decimals: int = 0) -> str:
    return "%" + number(100 * share, decimals)


def tr_date(day) -> str:
    day = pd.Timestamp(day)
    return f"{day.day} {MONTHS_TR[day.month - 1]}"


def signed(value: float, decimals: int) -> str:
    text = number(abs(value), decimals)
    if text == number(0, decimals):
        return text
    return ("+" if value > 0 else "-") + text


def interval(result: dict, decimals: int = 3) -> str:
    return f"{signed(result['mean'], decimals)} [{signed(result['low'], decimals)}; {signed(result['high'], decimals)}]"


def verdict(a: str, b: str, result: dict) -> str:
    """What the 95% interval of RMSLE(a) - RMSLE(b) allows to say."""
    if result["high"] < 0:
        return f"{LABELS[a]}, {LABELS[b]} modelinden daha iyi; aralık tamamen sıfırın altında."
    if result["low"] > 0:
        return f"{LABELS[b]}, {LABELS[a]} modelinden daha iyi; aralık tamamen sıfırın üstünde."
    return f"{LABELS[a]} ile {LABELS[b]} ayırt edilemiyor; aralık sıfırı içeriyor."


def mean_of(value) -> float:
    return value["mean"] if isinstance(value, dict) else value


def days_in_one_year(days: list[pd.Timestamp]) -> str:
    """'17 Ağustos ve 29 Aralık 2012 günlerinde': the year once when the days share it, else after each day."""
    if len({day.year for day in days}) == 1:
        return f"{' ve '.join(tr_date(day) for day in days)} {days[0].year} günlerinde"
    return " ve ".join(f"{tr_date(day)} {day.year}" for day in days) + " günlerinde"


def span(first: str, last: str) -> str:
    a, b = pd.Timestamp(first), pd.Timestamp(last)
    return f"{MONTHS_TR[a.month - 1]} {a.year}-{MONTHS_TR[b.month - 1]} {b.year}"


# ---------- facts computed from the data ----------

def data_facts() -> dict:
    raw = pd.read_csv(data.DATA, parse_dates=["dteday"])
    frame = data.load()
    parts = data.split(frame)
    stuck = stuck_atemp_days(raw)
    zero_humidity = raw[raw["hum"] == 0]
    working = frame[frame["workingday"] == 1].groupby("hr")["cnt"].mean()
    hours_per_day = frame.groupby("dteday").size()
    return {
        "rows": len(raw), "days": raw["dteday"].nunique(), "first": raw["dteday"].min(), "last": raw["dteday"].max(),
        "missing": data.missing_hours(frame), "missing_test": data.missing_hours(parts["test"]),
        "sandy_hours": [int(hours_per_day.get(pd.Timestamp(day), 0)) for day in SANDY],
        "test_days": parts["test"]["dteday"].nunique(),
        "skew": float(raw["cnt"].skew()), "median": float(raw["cnt"].median()), "max": int(raw["cnt"].max()),
        "temp_atemp": float(raw["temp"].corr(raw["atemp"])),
        "stuck_days": stuck, "stuck_value": float(raw.loc[raw["dteday"] == stuck[0], "atemp"].iloc[0]),
        "hum_zero": len(zero_humidity), "hum_zero_day": zero_humidity["dteday"].mode()[0],
        "wind_zero": int((raw["windspeed"] == 0).sum()), "heavy": int((raw["weathersit"] == 4).sum()),
        "unusual": int(data.unusual_hours(frame).sum()), "unusual_parts": unusual_breakdown(frame),
        "level": {name: float(parts[name]["cnt"].mean()) for name in ("train", "validation")},
        "year_means": raw.groupby("yr")["cnt"].mean().tolist(),
        "split_rows": {name: len(part) for name, part in parts.items()},
        "rain_drop": rain_drop(frame),
        "morning_peak": int(working.loc[:11].idxmax()), "evening_peak": int(working.loc[12:].idxmax()),
        "holiday_range": holiday_range(frame),
    }


def holiday_range(frame: pd.DataFrame) -> tuple[int, int]:
    """Smallest and largest daily rentals on the holidays the models learned from (training and validation months)."""
    learned = frame[(frame["time"] < pd.Timestamp(data.SPLITS["test"][0])) & (frame["holiday"] == 1)]
    totals = learned.groupby("dteday")["cnt"].sum()
    return int(totals.min()), int(totals.max())


def stuck_atemp_days(raw: pd.DataFrame) -> list[pd.Timestamp]:
    """Days on which atemp holds one value through every hour while temp moves: a broken feels-like series."""
    levels = raw.groupby("dteday").agg(atemp=("atemp", "nunique"), temp=("temp", "nunique"))
    return list(levels[(levels["atemp"] == 1) & (levels["temp"] > 1)].index)


def unusual_breakdown(frame: pd.DataFrame) -> dict[str, int]:
    """Where the unusual hours fall: below their group, in the first year, in January 2011, in hours of rain or snow."""
    flags = data.unusual_hours(frame)
    log_count = np.log1p(frame["cnt"])
    below = log_count < log_count.groupby([frame["hr"], frame["workingday"]]).transform("median")
    wet_hour = frame["weathersit"] == max(data.WEATHER)
    first_year = frame["yr"] == 0
    return {"low": int((flags & below).sum()), "first_year": int((flags & first_year).sum()),
            "january": int((flags & first_year & (frame["mnth"] == 1)).sum()), "wet_hours": int((flags & wet_hour).sum())}


def rain_drop(frame: pd.DataFrame) -> float:
    """How much smaller the median working-day rush-hour count is in rain than in clear weather."""
    rush = frame[(frame["workingday"] == 1) & frame["hr"].isin([8, 17, 18])]
    median = rush.groupby("weathersit")["cnt"].median()
    return float(1 - median[3] / median[1])


# ---------- facts computed from the results ----------

def squared_log_error(predictions: pd.DataFrame, name: str) -> pd.Series:
    return (np.log1p(predictions[name]) - np.log1p(predictions["cnt"])) ** 2


def worst_days(predictions: pd.DataFrame, name: str, count: int = 5) -> pd.DataFrame:
    """The test days that hold the largest share of a model's squared log error."""
    squared = squared_log_error(predictions, name)
    days = predictions["time"].dt.normalize()
    table = pd.DataFrame({"share": squared.groupby(days).sum() / squared.sum(),
                          "actual": predictions["cnt"].groupby(days).sum(),
                          "predicted": predictions[name].groupby(days).sum()})
    return table.sort_values("share", ascending=False).head(count)


def day_rank(predictions: pd.DataFrame, name: str, day: str) -> int:
    """Where a day stands when the test days are ordered by their share of a model's squared log error (1 = worst)."""
    order = list(worst_days(predictions, name, count=len(predictions)).index)
    return order.index(pd.Timestamp(day)) + 1


def day_total(predictions: pd.DataFrame, day: str) -> int:
    return int(predictions.loc[predictions["time"].dt.normalize() == pd.Timestamp(day), "cnt"].sum())


def share_of_days(predictions: pd.DataFrame, name: str, days) -> float:
    squared = squared_log_error(predictions, name)
    return float(squared[predictions["time"].dt.normalize().isin(pd.to_datetime(list(days)))].sum() / squared.sum())


def day_ratio(predictions: pd.DataFrame, name: str, day: str) -> float:
    """Predicted rentals over actual rentals for one day."""
    chosen = predictions["time"].dt.normalize() == pd.Timestamp(day)
    return float(predictions.loc[chosen, name].sum() / predictions.loc[chosen, "cnt"].sum())


def worst_hours(predictions: pd.DataFrame, name: str, day: str, count: int = 3) -> dict:
    """The hours of one day that hold the largest share of that day's squared log error, in time order."""
    chosen = predictions[predictions["time"].dt.normalize() == pd.Timestamp(day)]
    squared = squared_log_error(chosen, name)
    top = squared.nlargest(count).index.sort_values()
    return {"hours": chosen.loc[top, "time"].dt.hour.tolist(), "actual": chosen.loc[top, "cnt"].round().astype(int).tolist(),
            "predicted": chosen.loc[top, name].round().astype(int).tolist(), "share": float(squared[top].sum() / squared.sum())}


def humidity_counterfactual(metrics: dict, predictions: pd.DataFrame, day: str) -> dict:
    """The chosen gradient boosting refitted, and its prediction for one day again with the humidity set to the
    median of the same month in the fitting months."""
    parts, scaling = data.prepare()
    choice = metrics["chosen"]["hgb"]
    model = train.hgb(choice["learning_rate"], choice["max_leaf_nodes"], choice["rounds"]).fit(
        data.table(parts["fit"].frame), parts["fit"].y)
    test, fit = parts["test"].frame, parts["fit"].frame
    rows = data.table(test[test["dteday"] == pd.Timestamp(day)])
    typical = float(fit.loc[fit["mnth"] == pd.Timestamp(day).month, "hum"].median())
    predicted = float(scaling.counts(model.predict(rows)).sum())
    saved = float(predictions.loc[predictions["time"].dt.normalize() == pd.Timestamp(day), "hgb"].sum())
    if abs(predicted - saved) > 1:
        raise SystemExit(f"the refitted gradient boosting predicts {predicted:.0f} for {day}, the saved one {saved:.0f}")
    humid = fit["hum"] >= HUMID
    return {"humidity": float(rows["hum"].mean()), "typical": typical, "predicted": predicted,
            "changed": float(scaling.counts(model.predict(rows.assign(hum=typical))).sum()),
            "humid_wet": float((fit.loc[humid, "weathersit"] == max(data.WEATHER)).mean())}


def weather_rmsle(predictions: pd.DataFrame, weather: pd.Series, name: str) -> dict[int, float]:
    squared = squared_log_error(predictions, name)
    return {int(w): float(np.sqrt(squared[weather.to_numpy() == w].mean())) for w in data.WEATHER}


def humid_rank(predictions: pd.DataFrame, nets: list[str]) -> int:
    """The humid Saturday's best place among the worst days of the given nets; the text says how many days come first."""
    rank = min(day_rank(predictions, kind, HUMID_SATURDAY) for kind in nets)
    if rank < 2:
        raise SystemExit(f"{HUMID_SATURDAY} is a net's worst day; the sentence about it no longer holds")
    return rank


def result_facts(metrics: dict, predictions: pd.DataFrame) -> dict:
    best = best_on_validation(metrics)
    test_frame = data.split(data.load())["test"]
    stateful = [k for k in models.KINDS if k != "no-state"]
    no_state_curve = tuning_curves(metrics, "no-state")[:, :, 1].mean(0)
    by_weather = weather_rmsle(predictions, test_frame["weathersit"], best)
    biases = {name: mean_of(metrics["test"][name]["bias"]) for name in ORDER}
    monthly = metrics["monthly_bias"][best]
    working_days = test_frame.groupby("dteday")["workingday"].first()
    holidays = test_frame.groupby("dteday")["holiday"].first()
    return {
        "best": best,
        "best_on_test": min(ORDER, key=lambda n: mean_of(metrics["test"][n]["rmsle"])),
        "no_state_curve": (float(no_state_curve.min()), float(no_state_curve[-1])),
        "rain_hours": int((test_frame["weathersit"] == max(data.WEATHER)).sum()),
        "christmas_day_holiday": bool(holidays.get(pd.Timestamp(CHRISTMAS[1])) == 1),
        "christmas_day_ratio": day_ratio(predictions, best, CHRISTMAS[1]),
        "election_night": worst_hours(predictions, best, ELECTION_NIGHT),
        "humidity": humidity_counterfactual(metrics, predictions, HUMID_SATURDAY) if best == "hgb" else None,
        "rain_over_clear": by_weather[3] / by_weather[1],
        "christmas_share": share_of_days(predictions, best, CHRISTMAS),
        "christmas_working": [day for day in CHRISTMAS if working_days.get(pd.Timestamp(day)) == 1],
        "christmas_ratio": [day_ratio(predictions, best, day) for day in CHRISTMAS if working_days.get(pd.Timestamp(day)) == 1],
        "thanksgiving_working": bool(working_days.get(pd.Timestamp(AFTER_THANKSGIVING)) == 1),
        "thanksgiving_share": {name: share_of_days(predictions, name, [AFTER_THANKSGIVING]) for name in (best, "film")},
        "humid_saturday_ratio": day_ratio(predictions, best, HUMID_SATURDAY),
        "humid_saturday_net_rank": humid_rank(predictions, stateful),
        "leave_share": share_of_days(predictions, best, [CHRISTMAS[0], AFTER_THANKSGIVING]),
        "after_sandy_ratio": day_ratio(predictions, best, SANDY[1]),
        "all_low": all(value < 0 for value in biases.values()),
        "lowest_month": int(min(monthly, key=monthly.get)),
    }


# ---------- page parts ----------

CSS = """
@page { size: A4; margin: 0; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Segoe UI", sans-serif; color: #1E293B; }
.page { width: 210mm; height: 297mm; padding: 13mm 15mm 12mm; overflow: hidden; position: relative; page-break-after: always;
  font-size: 9.4pt; line-height: 1.4; }
.page:last-child { page-break-after: auto; }
h1 { font-size: 17pt; line-height: 1.2; }
.sub { color: #475569; font-size: 10pt; margin-top: 3px; }
.meta { font-size: 8.6pt; color: #475569; margin-top: 6px; padding-bottom: 7px; border-bottom: 2px solid #DB2777; }
h2 { font-size: 11.5pt; color: #BE185D; margin: 8px 0 3px; }
h3 { font-size: 9.8pt; margin: 6px 0 2px; }
p + p { margin-top: 4px; }
ul { padding-left: 15px; } li + li { margin-top: 1px; }
a { color: #1D4ED8; text-decoration: none; }
.nw { white-space: nowrap; }
table { width: 100%; border-collapse: collapse; font-size: 8.5pt; line-height: 1.3; margin-top: 4px; }
th { background: #FCE7F3; text-align: left; padding: 3px 5px; font-weight: 600; }
td { border-bottom: 1px solid #E2E8F0; padding: 3px 5px; vertical-align: top; }
td.n, th.n { text-align: right; white-space: nowrap; }
tr.best td { font-weight: 600; }
figure { margin-top: 5px; }
figure img { width: 100%; display: block; }
figcaption { font-size: 8.2pt; color: #475569; margin-top: 2px; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }
.pair { align-items: end; }
.pair + p { margin-top: 5px; }
.formula { font-family: Cambria, serif; font-size: 10.5pt; text-align: center; margin: 4px 0; }
.note { font-size: 8.2pt; color: #475569; }
.foot { position: absolute; left: 15mm; right: 15mm; bottom: 6mm; display: flex; justify-content: space-between;
  font-size: 7.6pt; color: #64748B; }
"""


def figure(name: str, caption: str) -> str:
    return f'<figure><img src="{(ROOT / "figures" / f"{name}.png").as_uri()}" alt=""><figcaption>{e(caption)}</figcaption></figure>'


def bullets(items) -> str:
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def page(body: str, number_: int, total: int) -> str:
    return (f'<section class="page" data-name="sayfa {number_}">{body}<div class="foot"><span>{html.escape(REPO_URL)}</span>'
            f"<span>{number_} / {total}</span></div></section>")


def results_table(metrics: dict, best: str) -> str:
    head = ('<tr><th>Model</th><th class="n">RMSLE</th><th class="n">RMSE</th><th class="n">MAE</th><th class="n">R²</th>'
            '<th class="n">Sapma</th><th class="n">Doğrulama RMSLE</th></tr>')
    rows = []
    for name in ORDER:
        test = metrics["test"][name]

        def cell(key: str, decimals: int) -> str:
            value = test[key]
            text = number(mean_of(value), decimals)
            if isinstance(value, dict):
                text += f" ± {number(value['std'], decimals)}"
            return f'<td class="n">{text}</td>'
        rows.append(f'<tr class="{"best" if name == best else ""}"><td>{e(LABELS[name])}</td>{cell("rmsle", 3)}{cell("rmse", 1)}'
                    f'{cell("mae", 1)}{cell("r2", 3)}{cell("bias", 1)}'
                    f'<td class="n">{number(min(metrics["tuning"][name].values()), 3)}</td></tr>')
    return f"<table>{head}{''.join(rows)}</table>"


def models_table(metrics: dict) -> str:
    how = {"no-state": "verilmiyor", "concat": "girdiye eklenir", "shift": "her katmana eklenir: + B·s",
           "film": "her katmanı ölçekler ve kaydırır"}
    rows = "".join(f"<tr><td>{e(LABELS[k])}</td><td>{e(how[k])}</td><td class='n'>{metrics['widths'][k]}</td>"
                   f"<td class='n'>{number(metrics['parameters'][k])}</td><td class='n'>{number(metrics['chosen'][k]['lr'], 3)}</td>"
                   f"<td class='n'>{number(metrics['chosen'][k]['weight_decay'], 2)}</td><td class='n'>{metrics['chosen'][k]['epochs']}</td></tr>"
                   for k in models.KINDS)
    return ('<table><tr><th>Ağ</th><th>Durum nasıl giriyor</th><th class="n">Genişlik</th><th class="n">Parametre</th>'
            f'<th class="n">Öğrenme hızı</th><th class="n">Weight decay</th><th class="n">Epoch</th></tr>{rows}</table>')


def worst_table(predictions: pd.DataFrame, name: str) -> str:
    frame = data.split(data.load())["test"]
    days = frame.groupby("dteday").agg(working=("workingday", "first"), weather=("weathersit", "max"))
    rows = "".join(f"<tr><td>{e(tr_date(day))}, {DAYS_TR[day.dayofweek]}</td>"
                   f"<td>{'iş günü' if days.loc[day, 'working'] else 'tatil / hafta sonu'}</td><td>{WEATHER_TR[days.loc[day, 'weather']]}</td>"
                   f"<td class='n'>{number(row.actual)}</td><td class='n'>{number(row.predicted)}</td><td class='n'>{pct(row.share, 1)}</td></tr>"
                   for day, row in worst_days(predictions, name).iterrows())
    return ('<table><tr><th>Gün</th><th>Veride</th><th>En kötü hava</th><th class="n">Gerçek</th><th class="n">Tahmin</th>'
            f'<th class="n">Hatadaki pay</th></tr>{rows}</table>')


# ---------- the report ----------

def conclusion(primary: dict, best: str, versus: dict | None = None) -> str:
    difference = f"RMSLE farkı {interval(primary)}"
    if primary["high"] < 0:
        claim = f"FiLM, durumu girdiye ekleyen ağdan iyi çıktı ({difference}); aralık tamamen sıfırın altında. Hipotezim bu veride destek buldu."
    elif primary["low"] > 0:
        claim = f"FiLM, durumu girdiye ekleyen ağdan kötü çıktı ({difference}); aralık tamamen sıfırın üstünde. Hipotezim bu veride çürüdü."
    else:
        claim = (f"FiLM ile durumu girdiye ekleyen ağ ayırt edilemiyor ({difference}; aralık sıfırı içeriyor). Hipotezim bu veri ve "
                 "bu boyutta desteklenmedi; bunu da bir sonuç sayıyorum.")
    if versus is None:
        return f"{claim} Doğrulamada en iyi model {LABELS[best]}."
    return (f"{claim} Doğrulamada en iyi model {LABELS[best]}; durumu girdiye ekleyen ağla test farkı {interval(versus)}. "
            f"{verdict(best, 'concat', versus)}")


def build(metrics: dict, predictions: pd.DataFrame) -> list[str]:
    f, r = data_facts(), result_facts(metrics, predictions)
    best, test, comparisons = r["best"], metrics["test"], metrics["comparisons"]
    primary = comparisons["film - concat"]["bootstrap"]
    split_rows = f["split_rows"]
    stateful = [k for k in models.KINDS if k != "no-state"]
    stateful_rmsle = [mean_of(test[k]["rmsle"]) for k in stateful]
    state_columns = f"{len(data.WEEKDAYS)} + 1 + {len(data.WEATHER)}"
    grid = train.GRID

    title = f"""
<h1>{e("Saatlik bisiklet kiralama tahmini: günün durumu ağa nasıl verilmeli?")}</h1>
<p class="sub">{e("UCI Bike Sharing verisinde durumla ölçeklenen (FiLM) bir sinir ağı, eşit boyutlu üç ağ ve iki klasik modelle karşılaştırma")}</p>
<p class="meta">AxelXoket · Kasım 2026 · Kod: {link(REPO_URL)}<br>Colab: {link(COLAB_URL)}</p>"""

    p1 = title + f"""
<h2>1. Problem tanımı</h2>
<p>{e("Washington'daki Capital Bikeshare sisteminde her saat kaç bisiklet kiralanacağını tahmin ediyorum. Girdi: saat, ay, yıl, o saatin gerçekleşen havası (sıcaklık, nem, rüzgâr, hava durumu), haftanın günü ve resmi tatil bilgisi. Çıktı: o saatteki toplam kiralama sayısı. Sistem genelindeki saatlik talep, filo ve personel planının temel girdisi.")}</p>
<p>{e("Araştırma sorusu: günün durumu (haftanın günü, tatil ve o saatin hava durumu) ağa sıradan bir girdi olarak mı verilmeli, yoksa ağın gizli birimlerini ölçekleyip kaydıran bir ayar sinyali (FiLM) olarak mı? Bu, durumla ayarlanan ağlar üzerine yaptığım Neuro_Morph araştırmasının küçük bir sürümü: durum bağlantıların çalışma biçimini değiştirirse ağ aynı girdiye daha iyi cevap verir mi?")}</p>
<h2>2. Veri seti</h2>
<p>{e(f"UCI Machine Learning Repository, Bike Sharing Dataset (hour.csv): {number(f['rows'])} saatlik kayıt, {number(f['days'])} gün, {tr_date(f['first'])} {f['first'].year} ile {tr_date(f['last'])} {f['last'].year} arası. Lisans CC BY 4.0.")}
Kaynak: {link(data.SOURCE_URL)}. {e("Atıf: " + CITATION + ".")}</p>
<table><tr><th>Değişken</th><th>Anlamı</th><th>Kullanım</th></tr>
<tr><td>cnt</td><td>{e("Saatlik toplam kiralama (hedef)")}</td><td>{e("log(1 + cnt) olarak öğrenilir")}</td></tr>
<tr><td>hr, mnth, yr</td><td>{e("Saat, ay, yıl (2011 = 0, 2012 = 1)")}</td><td>{e(f"x: saat {len(data.HOURS)} sütun, ay sinüs ve kosinüs, yıl")}</td></tr>
<tr><td>temp, hum, windspeed</td><td>{e("Ölçeklenmiş sıcaklık, nem, rüzgâr")}</td><td>{e("x: eğitim aylarına göre standartlaştırılır")}</td></tr>
<tr><td>{e("weekday, holiday, weathersit")}</td><td>{e("Haftanın günü, resmi tatil, hava durumu (1 açık, 2 sisli, 3 yağmur ve kar, 4 şiddetli)")}</td><td>{e(f"s: günün durumu, {state_columns} sütun")}</td></tr>
<tr><td>workingday, season</td><td>{e("İş günü; mevsim")}</td><td>{e("Çıkarıldı: iş günü gün ve tatilden türüyor; mevsim takvimi izliyor, ay zaten girdi")}</td></tr>
<tr><td>casual, registered</td><td>{e("Üyeliksiz ve üyeli kiralama")}</td><td>{e("Çıkarıldı: toplamları tam olarak cnt")}</td></tr>
<tr><td>atemp, instant</td><td>{e("Hissedilen sıcaklık; satır numarası")}</td><td>{e(f"Çıkarıldı: atemp ile temp korelasyonu {number(f['temp_atemp'], 3)}")}</td></tr></table>
<p style="margin-top:5px">{e(f"Hedef sağa çarpık (çarpıklık {number(f['skew'], 2)}; medyan {number(f['median'])}, en çok {number(f['max'])}); sınıf yok, bu bir regresyon problemi. {f['missing']} saat hiç kayıtlı değil; bunların {f['missing_test']} tanesi test aylarında, {2 * len(data.HOURS) - sum(f['sandy_hours'])} tanesi Sandy kasırgasının iki gününde. 2012'nin saat ortalaması ({number(f['year_means'][1], 1)}) 2011'inkinin ({number(f['year_means'][0], 1)}) {number(f['year_means'][1] / f['year_means'][0], 2)} katı.")}</p>
<h2>3. Veri analizi ve ön işleme</h2>
<div class="row pair">{figure("eda_saatlik_profil", "Şekil 1. Saat başına ortalama kiralama.")}{figure("eda_dagilim", "Şekil 2. Hedefin dağılımı ve log(1 + x) dönüşümü.")}</div>
<p>{e(f"Şekil 1: iş günlerinde saat {f['morning_peak']} ve {f['evening_peak']} tepeleri var; hafta sonu tek ve geniş bir öğleden sonra eğrisi var. Günün türü eğrinin yalnız seviyesini değil şeklini değiştiriyor; FiLM'i tam bu yüzden denedim. Şekil 2: sayılar sağa çarpık; log(1 + x) dağılımı toparlıyor, bu yüzden modeller log ölçeğinde öğreniyor ve birincil metrik RMSLE (log(1 + sayı) farklarının karelerinin ortalamasının karekökü).")}</p>"""

    train_span, validation_span, test_span = (span(*data.SPLITS[name]) for name in ("train", "validation", "test"))
    preprocessing = [
        e("casual ve registered çıkarıldı; toplamları her satırda cnt'ye eşit, modele cevabı verirlerdi."),
        e(f"atemp çıkarıldı: temp ile korelasyonu {number(f['temp_atemp'], 3)}; {days_in_one_year(f['stuck_days'])} 24 saat tek değerde ({number(f['stuck_value'], 4)}) takılı."),
        e(f"weathersit 4 ({f['heavy']} satır) 3 ile birleştirildi; tek başına öğrenilemeyecek kadar az."),
        e(f"{tr_date(f['hum_zero_day'])} {f['hum_zero_day'].year} günü nem {f['hum_zero']} saat boyunca 0; sensör boşluğu sayıp çevresindeki saatlerden zamana göre doldurdum."),
        e(f"Rüzgâr {number(f['wind_zero'])} saatte 0; sakin hava ile ölçülmemiş saat ayırt edilemiyor, olduğu gibi bıraktım."),
        e(f"Aykırı değer: saat ve gün türü grubunda log(1 + cnt) için {number(data.IQR_FACTOR, 1)} IQR dışı {f['unusual']} saat ({pct(f['unusual'] / f['rows'], 1)}); {f['unusual_parts']['low']} tanesi düşük tarafta, {f['unusual_parts']['first_year']} tanesi talebin düşük olduğu 2011'de ({f['unusual_parts']['january']} tanesi Ocak 2011'de), {f['unusual_parts']['wet_hours']} tanesi yağmurlu ya da karlı saatlerde. Gerçek düşük talep; silmedim."),
        e(f"Eksik {f['missing']} saat doldurulmadı: hedefi bilinmeyen bir saat uydurulamaz."),
        e("Kodlama: saat, gün ve hava durumu one-hot; ay sinüs ve kosinüs (Aralık ile Ocak yan yana). Sıcaklık, nem ve rüzgâr eğitim aylarının ortalaması ve standart sapmasıyla ölçeklendi; hedef log(1 + cnt), yine eğitim istatistikleriyle."),
        e(f"Bölme zamana göre: eğitim {train_span} ({number(split_rows['train'])} saat), doğrulama {validation_span} ({number(split_rows['validation'])}), test {test_span} ({number(split_rows['test'])}). Rastgele bölme komşu saatleri iki tarafa dağıtır ve skoru şişirir."),
        e("Yeniden boyutlandırma ve veri artırma tablo verisinde gerekmiyor."),
    ]
    p2 = f"""
<div class="row">{figure("eda_hava", "Şekil 3. İş günü 8, 17 ve 18 saatlerinde hava durumuna göre kiralama.")}<div>
<p style="margin-top:6px">{e(f"Şekil 3: yağmur ve karda iş günü yoğun saatlerinin medyanı açık havaya göre {pct(f['rain_drop'])} düşük. Hava durumu güçlü bir durum bilgisi; ama yağmurlu saatler azınlıkta, bu yüzden hatanın en büyük olacağı yer de burası.")}</p>
<p>{e(f"Şekil 4: talep 2011'den 2012'ye büyüyor ve mevsimle inip çıkıyor. Bu yüzden ayarları doğrulama aylarında seçip son modeli eğitim ve doğrulama aylarının tamamıyla yeniden eğittim; yalnız eğitim aylarını gören bir model sonbahar 2012 için düşük bir seviye öğrenir. Test ayları Sandy kasırgasını ({tr_date(SANDY[0])}) ve Noel'i içeriyor.")}</p></div></div>
{figure("eda_gunluk_seri", "Şekil 4. Günlük kiralama; eğitim, doğrulama ve test dönemleri.")}
<h3>Ön işleme adımları</h3>{bullets(preprocessing)}
<h2>4. Model ve yöntem</h2>
<p>{e("İki klasik model karşılaştırma tabanı: Ridge (doğrusal, one-hot girdiler) ve Gradient boosting (HistGradientBoostingRegressor; aynı bilgiyi ham tablo olarak okur). Dört sinir ağı yalnız günün durumunun ağa nasıl girdiğinde ayrılıyor. Hepsi iki gizli katmanlı (ReLU) ve tek çıkışlı; parametre sayıları %1 içinde eşit, böylece fark büyüklükten değil yapıdan gelir.")}</p>
<p>{e(f"Boyutu ayarlamadım, baştan seçtim: durumu girdiye ekleyen ağın genişliği {models.REFERENCE_WIDTH}. Gecikmeli değerleri ve dizi modellerini bilerek kullanmadım: önceki saatlerin sayısı günün durumunu dolaylı olarak taşır ve aranan farkı örter.")}</p>
<p class="formula">h = ReLU((W·x + b) ⊙ (1 + A·s) + B·s)</p>
<p>{e("FiLM'de durum s girdiye eklenmiyor; her gizli birimi 1 + A·s ile ölçekliyor ve B·s ile kaydırıyor. A ve B sıfırdan başlıyor, yani FiLM ağı durumsuz ağ olarak doğuyor. Yalnız kaydırma ağı ölçeklemeyi atıyor (yalnız + B·s): FiLM onu geçerse kazanç çarpımsal kısımdan gelmiş demektir.")}</p>
{models_table(metrics)}"""

    lrs = sorted({c.lr for c in grid})
    decays = sorted({c.weight_decay for c in grid})
    hgb_choice = metrics["chosen"]["hgb"]
    p3 = f"""
<h2>5. Eğitim süreci</h2>
{bullets([
        e(f"Kayıp: log ölçeğinde MSE; optimizer AdamW; batch {train.BATCH}. Izgara: öğrenme hızı {' ve '.join(number(v, 3) for v in lrs)}, weight decay {' ve '.join(number(v, 2) for v in decays)}; her ayar {len(metrics['tuning_seeds'])} seed (rastgeleliği sabitleyen sayı) ile denendi, ağ başına aynı bütçe."),
        e(f"Erken durdurma: doğrulama RMSLE'si {train.PATIENCE} epoch boyunca (epoch: eğitim verisinin tamamından bir geçiş) iyileşmezse durur, en iyi epoch'un ağırlıkları geri yüklenir. Seçilen epoch sayısı, ayar seed'lerinin ortalama en iyi epoch'u."),
        e(f"Son modeller eğitim ve doğrulama aylarıyla ({number(metrics['rows']['fit'])} saat) seçilen epoch kadar yeniden eğitildi; {len(metrics['final_seeds'])} farklı seed, ayar seed'lerinden ayrı. Her seed'de batch sırası bütün ağlar için aynı."),
        e(f"Gradient boosting: öğrenme hızı {' ve '.join(number(v, 2) for v in sorted({lr for lr, _ in train.HGB_GRID}))}, yaprak {' ve '.join(str(v) for v in sorted({n for _, n in train.HGB_GRID}))}; tur sayısı da doğrulamada seçildi ({hgb_choice['rounds']} tur, {number(hgb_choice['learning_rate'], 2)}, {hgb_choice['max_leaf_nodes']} yaprak). Ridge'de alfa {'; '.join(number(a, 1) for a in train.RIDGE_ALPHAS)} arasından {number(metrics['chosen']['ridge']['alpha'], 1)}."),
        e(f"Donanım: {metrics['device']}, PyTorch {metrics['torch'].split('+')[0]}; bütün deney {number(metrics['seconds'])} sn. Seed'ler sabit, sonuçlar aynı komutla yeniden üretiliyor."),
    ])}
{figure("sonuc_ogrenme", f"Şekil 5. Seçilen ayarla {len(metrics['tuning_seeds'])} ayar seed'inin ortalama eğitim kaybı ve doğrulama RMSLE'si; bant seed'lerin aralığı, nokta seçilen epoch sayısı.")}
<p>{e(f"Şekil 5: doğrulama hatası ilk epoch'larda dibe iniyor, sonra eğitim kaybı inmeye devam ederken yerinde sayıp dalgalanıyor. Olası bir neden: doğrulama ayları eğitim aylarından çok daha yoğun (saat ortalaması {number(f['level']['validation'])}, eğitimde {number(f['level']['train'])}). En iyi epoch erken geldiği için erken durdurma belirleyici. Durumsuz ağın ortalama doğrulama eğrisi {number(r['no_state_curve'][0], 2)} değerine inip sonra {number(r['no_state_curve'][1], 2)} değerine çıkıyor; ağ günün türünü bilmediği için iş günü ile hafta sonu eğrilerini ayıramıyor olabilir.")}</p>
<h2>6. Değerlendirme sonuçları</h2>
{results_table(metrics, best)}
<p class="note">{e(f"Test: {test_span}. Ağlarda {len(metrics['final_seeds'])} seed ortalaması ± standart sapma. Sapma: ortalama (tahmin - gerçek); eksi, modelin az tahmin ettiğini gösterir. Kalın satır doğrulamada en iyi model.")}</p>
{figure("sonuc_karsilastirma", "Şekil 6. Test RMSLE ve RMSE; noktalar tek tek seed'ler.")}
<p>{e(f"Doğrulamada en iyi model {LABELS[best]}" + (", testte de en düşük RMSLE onun" if r['best_on_test'] == best else f"; testte en düşük RMSLE ise {LABELS[r['best_on_test']]}") + f": RMSLE {number(mean_of(test[best]['rmsle']), 3)}, RMSE {number(mean_of(test[best]['rmse']), 1)}, R² {number(mean_of(test[best]['r2']), 3)}. Durumu kullanan üç ağın RMSLE'si {number(min(stateful_rmsle), 3)} ile {number(max(stateful_rmsle), 3)} arasında.")}</p>
<p>{e(f"Farkları seed ve test günleri üzerinden iki seviyeli eşleştirilmiş bootstrap ile ölçtüm (köşeli parantez %95 aralık; {number(train.BOOTSTRAP_REPS)} tekrar; her turda seed'ler ve {f['test_days']} test günü yeniden çekiliyor, iki model için aynı). FiLM eksi durum girdide: {interval(primary)}. {verdict('film', 'concat', primary)} FiLM eksi yalnız kaydırma: {interval(comparisons['film - shift']['bootstrap'])}. Durumun kendisi ise büyük fark yaratıyor: durum girdide eksi durumsuz {interval(comparisons['concat - no-state']['bootstrap'])}. {LABELS['hgb']} eksi durum girdide {interval(comparisons['hgb - concat']['bootstrap'])}.")}</p>"""

    best_label = LABELS[best]
    sandy_first, sandy_second = f["sandy_hours"]
    ratios = r["christmas_ratio"]
    hgb_only, hgb_refit = metrics["train_only_test"]["hgb"], test["hgb"]
    christmas_days = " ve ".join(str(pd.Timestamp(day).day) for day in r["christmas_working"])
    christmas_span = f"{pd.Timestamp(CHRISTMAS[0]).day}-{tr_date(CHRISTMAS[-1])}"
    thanksgiving = (f"Şükran Günü'nün ertesi günü ({tr_date(AFTER_THANKSGIVING)}) de veride iş günü ama ABD'de pek çok kişi o gün izinli; "
                    f"bu gün {LABELS['film']} hatasının {pct(r['thanksgiving_share']['film'], 1)} payını, {best_label} hatasının "
                    f"{pct(r['thanksgiving_share'][best], 1)} payını taşıyor.") if r["thanksgiving_working"] else ""
    humidity = r["humidity"]
    humid = (f"{tr_date(HUMID_SATURDAY)} Cumartesi durumu kullanan hiçbir ağın hata payı en büyük {r['humid_saturday_net_rank'] - 1} günü arasında değil; "
             f"{best_label} ise gerçeğin {pct(r['humid_saturday_ratio'])} kadarını tahmin ediyor. O gün nem ortalaması {pct(humidity['humidity'])} ve yağmur yok. Modelin öğrendiği aylarda nemi {pct(HUMID)} ve üstü olan "
            f"saatlerin {pct(humidity['humid_wet'])} kadarı yağmurlu; ağaçlar yüksek nemi düşük talep diye öğrenmiş. Nemi öğrendiği "
            f"{MONTHS_TR[pd.Timestamp(HUMID_SATURDAY).month - 1]} saatlerinin medyanına ({pct(humidity['typical'])}) çekince tahmin {number(humidity['predicted'])} kiralamadan "
            f"{number(humidity['changed'])} kiralamaya çıkıyor; gerçek {number(day_total(predictions, HUMID_SATURDAY))}.") if humidity else ""
    night = r["election_night"]
    election = (f"{tr_date(ELECTION_NIGHT)} gününün hatasının {pct(night['share'])} kadarı gece "
                f"{', '.join(f'{h:02d}.00' for h in night['hours'][:-1])} ve {night['hours'][-1]:02d}.00 saatlerinde: "
                f"{', '.join(map(str, night['actual'][:-1]))} ve {night['actual'][-1]} kiralama var, model "
                f"{', '.join(map(str, night['predicted'][:-1]))} ve {night['predicted'][-1]} bekliyor. Bu saatler 6 Kasım 2012 "
                f"başkanlık seçimi gecesinin hemen ardından; veride böyle bir olayı gösteren sütun yok.")
    christmas_day = (f"25 Aralık ise veride resmi tatil, ama tatil bilgisi tek bir sütun: eğitim ve doğrulama aylarındaki "
                     f"tatillerde günlük kiralama {number(f['holiday_range'][0])} ile {number(f['holiday_range'][1])} arasında. "
                     f"Model Noel gününü öteki tatillerden ayıramıyor ve gerçeğin {number(r['christmas_day_ratio'], 1)} katını "
                     f"tahmin ediyor.") if r["christmas_day_holiday"] else ""
    level = (f"Seviye: bütün modeller test aylarını az tahmin ediyor (sapma eksi); {best_label} en çok {MONTHS_TR[r['lowest_month'] - 1]} ayında. "
             if r["all_low"] else "")
    p4 = f"""
<h2>7. Hata analizi</h2>
{figure("sonuc_tahmin", "Şekil 7. Test aylarından iki hafta: gerçek ve tahmin; pembe bant kayıt olmayan saatler.")}
<p>{e(f"Şekil 7: günlük iki tepe ve hafta sonu eğrisi iyi yakalanıyor; hata tepe saatlerin yüksekliğinde. Sandy'nin {tr_date(SANDY[0])} gününde yalnız {sandy_first} saat, {tr_date(SANDY[1])} gününde {sandy_second} saat kayıt var; öteki saatler veride yok. {tr_date(SANDY[1])} gününün kayıtlı saatlerinde {best_label} gerçeğin {number(r['after_sandy_ratio'], 1)} katını tahmin ediyor, çünkü hiçbir girdi kasırgayı bilmiyor.")}</p>
{worst_table(predictions, best)}
<p class="note">{e(f"{best_label} modelinin karesel log hatasında en büyük payı olan test günleri. Hatadaki pay: o günün hatasının toplam hataya oranı.")}</p>
<p>{e(f"Noel'in üç günü ({christmas_span}) {best_label} modelinin toplam karesel log hatasının {pct(r['christmas_share'])} payını tek başına taşıyor. Veride {christmas_days} Aralık iş günü görünüyor; eğitim aylarında iş gününe denk gelen bir Noel arifesi yok. Model bu günleri sıradan bir iş günü sanıp gerçeğin {number(min(ratios), 1)} ile {number(max(ratios), 1)} katını tahmin ediyor. {christmas_day}")}</p>
<p>{e(f"{election} {humid} {thanksgiving}")}</p>
{figure("sonuc_hata", "Şekil 8. Saate göre ortalama mutlak hata ve hava durumuna göre RMSLE.")}
<p>{e(f"Şekil 8: mutlak hata iş günü tepe saatlerinde en büyük, çünkü sayılar orada büyük. Göreli hata ise yağmurda büyük: yağmurlu test saatlerinde {best_label} modelinin RMSLE'si açık havanın {number(r['rain_over_clear'], 1)} katı. Yağmurlu saatler testte az ({r['rain_hours']} saat). {level}Yeniden eğitim sapmayı azaltıyor: yalnız eğitim aylarıyla kurulan {LABELS['hgb']} modelinin sapması {number(hgb_only['bias'], 1)}, RMSLE'si {number(hgb_only['rmsle'], 3)}; eğitim ve doğrulama aylarıyla {number(hgb_refit['bias'], 1)} ve {number(hgb_refit['rmsle'], 3)}.")}</p>"""

    p5 = f"""
{figure("sonuc_durum_profili", f"Şekil 9. {MONTHS_TR[9]} 2012'de tipik havalı bir gün için dört durumda tahmin edilen saatlik eğri ({len(metrics['final_seeds'])} seed ortalaması).")}
<p>{e("Şekil 9: iki ağ da aynı saat girdisinden durumla iki farklı eğri kuruyor: Pazartesi iki tepe, Cumartesi tek öğleden sonra eğrisi; yağmur hepsini aşağı çekiyor. FiLM'in öğrendiği eğri, durumu girdiye ekleyen ağın eğrisinden belirgin biçimde ayrılmıyor.")}</p>
<h2>8. Sonuç ve geliştirme önerileri</h2>
<p>{e(f"Günün durumu bu problemde belirleyici: durumu bilmeyen ağın RMSLE'si {number(mean_of(test['no-state']['rmsle']), 3)}, bilen ağlarınki {number(min(stateful_rmsle), 3)} ile {number(max(stateful_rmsle), 3)} arasında. " + conclusion(primary, best, comparisons.get(f'{best} - concat', {}).get('bootstrap')))}</p>
<h3>Güçlü yönler</h3>{bullets([
        e("Zamana göre bölme, yalnız eğitim aylarından ölçeklenen girdiler ve sızıntı yapan sütunların çıkarılması testlerle korunuyor."),
        e("Ağlar eşit parametreyle, aynı ayar bütçesiyle ve aynı batch sırasıyla karşılaştırıldı; fark iddiası seed ve gün gürültüsünü birlikte sayan bir aralığa bağlı."),
        e("Yalnız kaydırma kolu, FiLM'in toplamsal ve çarpımsal parçasını ayırıyor."),
    ])}
<h3>Zayıf yönler</h3>{bullets([
        e("Hava, o saatin gerçekleşen havası; gerçek bir ileri tahminde hava tahmininin kendi hatası da eklenir."),
        e("Tatil bilgisi resmi tatil listesi kadar; Noel arifesi ve Şükran Günü ertesi gibi fiilî izin günlerini bilmiyor."),
        e(f"Test yalnız {test_span}; iki yıllık veride başka bir yılın aynı aylarıyla tekrar edilemiyor."),
        e("Log ölçeğinde öğrenilen tahmin sayıya çevrilince medyana yakın kalır, ortalamayı biraz düşük tahmin eder; bütün modellerde aynı, karşılaştırmayı bozmuyor."),
    ])}
<h3>Geliştirme önerileri</h3>{bullets([
        e(f"Takvime fiilî izin günlerini (Noel arifesi, Şükran Günü ertesi) eklemek; bu iki gün {best_label} hatasının {pct(r['leave_share'])} payını taşıyor."),
        e("Kısa vadeli tahmin için önceki saatlerin kiralama sayılarını (gecikmeli değerler) girdi yapmak."),
        e(f"Durumun daha zengin olduğu bir problemde (etkinlikler, istasyon bazında veri) FiLM'i yeniden denemek; durum burada {len(data.STATE_NAMES)} sütun."),
        e("Tek sayı yerine aralık tahmini (quantile kaybı) ile filo planına güven payı vermek."),
    ])}
<h2>Kaynaklar ve bağlantılar</h2>{bullets([
        f"{e('Kod, sonuçlar ve testler:')} {link(REPO_URL)}",
        f"{e('Colab not defteri:')} {link(COLAB_URL)}",
        f"{e('Veri:')} {link(data.SOURCE_URL)} {e('(CC BY 4.0)')}",
        e(CITATION),
        e(FILM_CITATION),
    ])}"""
    bodies = [p1, p2, p3, p4, p5]
    return [page(body, i, len(bodies)) for i, body in enumerate(bodies, 1)]


def write(pages: list[str], target: pathlib.Path) -> pathlib.Path:
    document = (f'<!doctype html><html lang="tr"><head><meta charset="utf-8"><title>Saatlik bisiklet kiralama tahmini</title>'
                f'<style>{CSS}</style></head><body>{"".join(pages)}{render.FIT_SCRIPT}</body></html>')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page_path = pathlib.Path(tmp) / "rapor.html"
        page_path.write_text(document, encoding="utf-8")
        problems = render.fit_report(page_path, 900, 1200)
        if problems:
            raise SystemExit(f"the report does not fit: {problems}")
        render.html_to_pdf(page_path, target)
    return target


def main() -> None:
    metrics = json.loads((train.RESULTS / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(train.RESULTS / "predictions.csv", parse_dates=["time"])
    before = render.windowed_edges()
    print(write(build(metrics, predictions), DOCS / "rapor.pdf").relative_to(ROOT))
    if render.windowed_edges() != before:
        raise SystemExit("an Edge window opened during the build")


if __name__ == "__main__":
    main()
