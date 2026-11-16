"""UCI Bike Sharing (hourly): load, clean, split by time and turn into model inputs.

The networks get two separate groups of inputs:
- x, what the hour looks like: hour of day, month, year, temperature, humidity, wind.
- s, the state of the day: weekday, holiday, weather situation.
No column of s can be computed from x, so "no state" really means no day type and no weather situation. workingday
(weekday plus holiday) is left out because it is built from columns already given; season follows the calendar
date, and the month is already an input.
Every statistic a model sees (feature scaling, target scaling) is fitted on the training months only.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hour.csv"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"

# casual + registered equals cnt in every row, so either one would hand the answer to the model.
LEAKS = ("casual", "registered")
# atemp follows temp (correlation 0.988) and holds one value (0.2424) all day on 2012-08-17 and 2012-12-29.
DROPPED = ("instant", "atemp", *LEAKS)
SPLITS = {
    "train": ("2011-01-01", "2012-06-30"),
    "validation": ("2012-07-01", "2012-09-30"),
    "test": ("2012-10-01", "2012-12-31"),
}
WEATHER = (1, 2, 3)  # 4 (heavy rain, ice, snow) has 3 rows in two years and joins 3
HOURS = tuple(range(24))
WEEKDAYS = tuple(range(7))  # 0 is Sunday
SCALED = ("temp", "hum", "windspeed")
INPUT_NAMES = (*(f"hour {h}" for h in HOURS), "month sin", "month cos", "yr", *SCALED)
STATE_NAMES = (*(f"weekday {d}" for d in WEEKDAYS), "holiday", *(f"weather {w}" for w in WEATHER))
# The classical models read the same information as a plain table; weather is a category, the rest are numbers.
TABLE_COLUMNS = ("hr", "mnth", "yr", *SCALED, "weekday", "holiday", "weathersit")
CATEGORICAL = ("weathersit",)
IQR_FACTOR = 1.5  # the usual Tukey fence


def load(path: pathlib.Path = DATA) -> pd.DataFrame:
    return clean(pd.read_csv(path, parse_dates=["dteday"]))


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.drop(columns=list(DROPPED))
    frame["time"] = frame["dteday"] + pd.to_timedelta(frame["hr"], unit="h")
    frame = frame.sort_values("time", ignore_index=True)
    frame["weathersit"] = frame["weathersit"].clip(upper=max(WEATHER))
    # Humidity reads 0 for the whole of 2011-03-10: a sensor gap, filled in time from the hours around it.
    humidity = frame.set_index("time")["hum"].replace(0, np.nan)
    frame["hum"] = humidity.interpolate(method="time").to_numpy()
    return frame


def split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    days = frame["dteday"]
    return {name: frame[(days >= start) & (days <= end)].reset_index(drop=True) for name, (start, end) in SPLITS.items()}


def missing_hours(frame: pd.DataFrame) -> int:
    """Hours with no row at all: the system recorded nothing (165 in two years, most of them during Hurricane Sandy)."""
    return frame["dteday"].nunique() * len(HOURS) - len(frame)


def unusual_hours(frame: pd.DataFrame) -> pd.Series:
    """Hours far from what is normal for that hour and day type: outside IQR_FACTOR IQR of log1p(cnt) in the group."""
    log_count = np.log1p(frame["cnt"])
    groups = log_count.groupby([frame["hr"], frame["workingday"]])
    low, high = groups.transform(lambda v: v.quantile(0.25)), groups.transform(lambda v: v.quantile(0.75))
    spread = high - low
    return (log_count < low - IQR_FACTOR * spread) | (log_count > high + IQR_FACTOR * spread)


def one_hot(values: pd.Series, levels: tuple) -> np.ndarray:
    return (values.to_numpy()[:, None] == np.array(levels)[None, :]).astype(np.float32)


def raw_inputs(frame: pd.DataFrame) -> np.ndarray:
    angle = 2 * np.pi * (frame["mnth"].to_numpy() - 1) / 12
    columns = [one_hot(frame["hr"], HOURS), np.sin(angle)[:, None], np.cos(angle)[:, None], frame[["yr", *SCALED]].to_numpy()]
    return np.hstack(columns).astype(np.float32)


def state(frame: pd.DataFrame) -> np.ndarray:
    columns = [one_hot(frame["weekday"], WEEKDAYS), frame[["holiday"]].to_numpy(), one_hot(frame["weathersit"], WEATHER)]
    return np.hstack(columns).astype(np.float32)


def target(frame: pd.DataFrame) -> np.ndarray:
    """log1p of the hourly count: the counts are right-skewed (skew 1.28) and every hour has at least one rental."""
    return np.log1p(frame["cnt"].to_numpy()).astype(np.float32)


@dataclass(frozen=True)
class Scaling:
    """Means and spreads measured on the training months; applied unchanged to every other row."""
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, train: pd.DataFrame) -> Scaling:
        x, y = raw_inputs(train), target(train)
        mean, std = np.zeros(x.shape[1], np.float32), np.ones(x.shape[1], np.float32)
        scaled = [INPUT_NAMES.index(name) for name in SCALED]
        mean[scaled], std[scaled] = x[:, scaled].mean(0), x[:, scaled].std(0)
        return cls(mean, std, float(y.mean()), float(y.std()))

    def inputs(self, frame: pd.DataFrame) -> np.ndarray:
        return (raw_inputs(frame) - self.input_mean) / self.input_std

    def scaled_target(self, frame: pd.DataFrame) -> np.ndarray:
        return (target(frame) - self.target_mean) / self.target_std

    def counts(self, scaled_prediction: np.ndarray) -> np.ndarray:
        """Back to rentals per hour; a negative count is not possible, so it is cut at 0."""
        return np.clip(np.expm1(scaled_prediction * self.target_std + self.target_mean), 0, None)


@dataclass(frozen=True)
class Split:
    frame: pd.DataFrame
    x: np.ndarray
    s: np.ndarray
    y: np.ndarray  # scaled log1p target
    counts: np.ndarray


def make_split(frame: pd.DataFrame, scaling: Scaling) -> Split:
    return Split(frame, scaling.inputs(frame), state(frame), scaling.scaled_target(frame), frame["cnt"].to_numpy().astype(np.float64))


def prepare(path: pathlib.Path = DATA) -> tuple[dict[str, Split], Scaling]:
    """train, validation and test, plus "fit": train and validation together, used once the settings are chosen."""
    parts = split(load(path))
    scaling = Scaling.fit(parts["train"])
    prepared = {name: make_split(part, scaling) for name, part in parts.items()}
    prepared["fit"] = make_split(pd.concat([parts["train"], parts["validation"]], ignore_index=True), scaling)
    return prepared, scaling


def table(frame: pd.DataFrame) -> pd.DataFrame:
    """The columns the classical models read, with the weather situation typed as a category."""
    out = frame[list(TABLE_COLUMNS)].copy()
    for column in CATEGORICAL:
        out[column] = pd.Categorical(out[column], categories=WEATHER)
    return out
