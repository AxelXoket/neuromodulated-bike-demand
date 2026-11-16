"""The experiment: settings are chosen on the validation months, then every model is refitted on training plus
validation months and scored once on the test months (October to December 2012).

Refitting matters here: rentals kept growing in 2012, and a model that never saw July to September learns a level that
is too low for the autumn.

Run: python -m bike_demand.train
Writes results/metrics.json and results/predictions.csv (one column per model and seed).
"""
from __future__ import annotations

import copy
import json
import math
import os
import pathlib
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from bike_demand import data, models

RESULTS = data.ROOT / "results"
BATCH = 256
MAX_EPOCHS = 400
PATIENCE = 40  # epochs without a better validation score before tuning stops
PRIMARY = "rmsle"  # the loss is on log counts, so the comparison that decides is on the same scale
PRIMARY_PAIR = ("film", "concat")
BOOTSTRAP_REPS = 10_000


@dataclass(frozen=True)
class Config:
    lr: float
    weight_decay: float


GRID = tuple(Config(lr, wd) for lr in (1e-3, 3e-3) for wd in (0.0, 1e-2))
TUNING_SEEDS = (0, 1, 2)
FINAL_SEEDS = tuple(range(10, 20))  # apart from the tuning seeds, so no final run was seen while choosing
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
HGB_GRID = tuple((lr, leaves) for lr in (0.05, 0.1) for leaves in (31, 63))
HGB_MAX_ITER = 2000
# The FiLM figure: one October 2012 hour with the test months' typical weather, under four states of the day.
PROFILE_STATES = {"Pazartesi, açık": (1, 0, 1), "Pazartesi, yağmurlu": (1, 0, 3), "Cumartesi, açık": (6, 0, 1),
                  "Cumartesi, yağmurlu": (6, 0, 3)}


def deterministic() -> None:
    """Same seed, same numbers: cuBLAS needs a fixed workspace for that, set before the first CUDA call."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- metrics ----------

def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1 - np.sum(error ** 2) / np.sum((actual - actual.mean()) ** 2)),
        "rmsle": float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2))),
        "bias": float(np.mean(error)),  # above 0: the model expects more rentals than there were
    }


def rmsle_scaled(prediction: torch.Tensor, target: torch.Tensor, scaling: data.Scaling) -> float:
    """RMSLE straight from scaled log predictions; clamping at 0 is the same as cutting negative counts at 0."""
    log_prediction = torch.clamp(prediction * scaling.target_std + scaling.target_mean, min=0)
    log_actual = target * scaling.target_std + scaling.target_mean
    return float(torch.sqrt(torch.mean((log_prediction - log_actual) ** 2)))


def summary(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {key: {"mean": float(np.mean([r[key] for r in runs])), "std": float(np.std([r[key] for r in runs], ddof=1))}
            for key in runs[0]}


# ---------- networks ----------

@dataclass
class Fitted:
    model: models.Net
    history: list[tuple[float, float]]  # (training loss, validation RMSLE) per epoch; validation is nan when there is none
    best_epoch: int


def tensors(part: data.Split, on: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.as_tensor(a, device=on) for a in (part.x, part.s, part.y))


@torch.no_grad()
def predict(model: models.Net, part: data.Split, on: torch.device) -> np.ndarray:
    model.eval()
    x, s, _ = tensors(part, on)
    return model(x, s).cpu().numpy()


def start(kind: str, width: int, config: Config, seed: int, part: data.Split, on: torch.device):
    torch.manual_seed(seed)
    model = models.Net(kind, part.x.shape[1], part.s.shape[1], width).to(on)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # The batch order depends on the seed only, so every kind sees the same batches for the same seed.
    return model, optimiser, torch.Generator().manual_seed(seed)


def train_epoch(model: models.Net, optimiser, order: torch.Generator, x, s, y) -> float:
    model.train()
    permutation = torch.randperm(len(y), generator=order).to(y.device)
    total = torch.zeros((), device=y.device)
    for first in range(0, len(y), BATCH):
        batch = permutation[first:first + BATCH]
        loss = torch.nn.functional.mse_loss(model(x[batch], s[batch]), y[batch])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        total += loss.detach() * len(batch)
    return float(total) / len(y)


def fit_net(kind: str, width: int, config: Config, seed: int, train: data.Split, validation: data.Split,
            scaling: data.Scaling, on: torch.device, max_epochs: int | None = None, patience: int | None = None) -> Fitted:
    """Train until the validation score stops improving, then return the network from its best epoch."""
    max_epochs = MAX_EPOCHS if max_epochs is None else max_epochs
    patience = PATIENCE if patience is None else patience
    model, optimiser, order = start(kind, width, config, seed, train, on)
    x, s, y = tensors(train, on)
    vx, vs, vy = tensors(validation, on)
    best, best_state, best_epoch, history = math.inf, None, 0, []
    for epoch in range(max_epochs):
        loss = train_epoch(model, optimiser, order, x, s, y)
        model.eval()
        with torch.no_grad():
            score = rmsle_scaled(model(vx, vs), vy, scaling)
        history.append((loss, score))
        if score < best:
            best, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return Fitted(model, history, best_epoch)


def fit_fixed(kind: str, width: int, config: Config, seed: int, part: data.Split, epochs: int, on: torch.device) -> Fitted:
    """Train for a set number of epochs with no validation to watch: the refit on training plus validation months."""
    model, optimiser, order = start(kind, width, config, seed, part, on)
    x, s, y = tensors(part, on)
    history = [(train_epoch(model, optimiser, order, x, s, y), math.nan) for _ in range(epochs)]
    return Fitted(model, history, epochs - 1)


def tune_net(kind: str, width: int, train: data.Split, validation: data.Split, scaling: data.Scaling,
             on: torch.device) -> dict:
    """Mean best validation RMSLE per setting over the tuning seeds; the chosen setting, its epoch count and the
    histories of its tuning seeds."""
    table = {}
    for config in GRID:
        runs = [fit_net(kind, width, config, seed, train, validation, scaling, on) for seed in TUNING_SEEDS]
        table[config] = (float(np.mean([min(v for _, v in r.history) for r in runs])),
                         int(round(np.mean([r.best_epoch + 1 for r in runs]))), [r.history for r in runs])
    config = min(table, key=lambda c: table[c][0])
    return {"scores": {f"lr={c.lr} wd={c.weight_decay}": v[0] for c, v in table.items()}, "config": config,
            "epochs": table[config][1], "histories": table[config][2]}


# ---------- classical baselines ----------

def ridge_inputs(part: data.Split) -> np.ndarray:
    return np.hstack([part.x, part.s])


def tune_ridge(train: data.Split, validation: data.Split, scaling: data.Scaling) -> dict:
    scores = {a: metrics(validation.counts, scaling.counts(Ridge(alpha=a).fit(ridge_inputs(train), train.y).predict(
        ridge_inputs(validation))))["rmsle"] for a in RIDGE_ALPHAS}
    return {"scores": {str(a): s for a, s in scores.items()}, "alpha": min(scores, key=scores.get)}


def hgb(lr: float, leaves: int, rounds: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, max_iter=rounds, early_stopping=False,
                                         categorical_features="from_dtype", random_state=0)


def tune_hgb(train: data.Split, validation: data.Split, scaling: data.Scaling) -> dict:
    """Rounds are chosen on validation like the networks' epochs: the round count with the best validation RMSLE."""
    scores = {}
    for lr, leaves in HGB_GRID:
        full = hgb(lr, leaves, HGB_MAX_ITER).fit(data.table(train.frame), train.y)
        staged = [metrics(validation.counts, scaling.counts(p))["rmsle"] for p in full.staged_predict(data.table(validation.frame))]
        scores[(lr, leaves)] = (min(staged), int(np.argmin(staged)) + 1)
    lr, leaves = min(scores, key=lambda k: scores[k][0])
    return {"scores": {f"lr={k[0]} leaves={k[1]}": v[0] for k, v in scores.items()}, "learning_rate": lr,
            "max_leaf_nodes": leaves, "rounds": scores[(lr, leaves)][1]}


# ---------- statistics ----------

def paired_difference(a: list[float], b: list[float]) -> dict[str, float]:
    """Mean of a - b over seeds that shared their batch order, with a 95% t interval (seed noise only)."""
    difference = np.asarray(a) - np.asarray(b)
    mean, spread = float(difference.mean()), float(difference.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(difference) - 1) * spread / math.sqrt(len(difference)))
    return {"mean": mean, "low": mean - half, "high": mean + half, "seeds": len(difference)}


def squared_log_errors(actual: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """(seeds, rows) squared differences of log1p counts."""
    return (np.log1p(predictions) - np.log1p(actual)[None, :]) ** 2


def paired_bootstrap(errors_a: np.ndarray, errors_b: np.ndarray, days: np.ndarray, reps: int | None = None,
                     seed: int = 0) -> dict[str, float]:
    """RMSLE of a minus RMSLE of b, both averaged over seeds, with a 95% interval that counts both kinds of luck:
    which seeds were run (seeds drawn again, the same for both models) and which days the test months held
    (days drawn again, the same for both models)."""
    reps = BOOTSTRAP_REPS if reps is None else reps
    seeds, day_count = errors_a.shape[0], int(days.max()) + 1
    per_day = [np.stack([np.bincount(days, weights=row, minlength=day_count) for row in errors]) for errors in (errors_a, errors_b)]
    hours = np.bincount(days, minlength=day_count).astype(float)
    rng = np.random.default_rng(seed)
    day_weights = rng.multinomial(day_count, np.full(day_count, 1 / day_count), size=reps).T  # (days, reps)
    seed_weights = rng.multinomial(seeds, np.full(seeds, 1 / seeds), size=reps)  # (reps, seeds)
    def rmsle(errors: np.ndarray) -> np.ndarray:
        per_seed = np.sqrt(errors @ day_weights / (hours @ day_weights))  # (seeds, reps)
        return np.einsum("rs,sr->r", seed_weights, per_seed) / seeds
    draws = rmsle(per_day[0]) - rmsle(per_day[1])
    point = float(np.sqrt(per_day[0].sum(1) / hours.sum()).mean() - np.sqrt(per_day[1].sum(1) / hours.sum()).mean())
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean": point, "low": float(low), "high": float(high), "reps": reps}


# ---------- the whole experiment ----------

def state_profiles(model: models.Net, test: data.Split, scaling: data.Scaling, on: torch.device) -> dict[str, list[float]]:
    """Predicted rentals for each hour of a typical October 2012 day, under each state in PROFILE_STATES."""
    typical = test.frame[test.frame["mnth"] == 10][list(data.SCALED)].median()
    hours = pd.DataFrame({"hr": data.HOURS, "mnth": 10, "yr": 1, **{c: typical[c] for c in data.SCALED}})
    x = torch.as_tensor(scaling.inputs(hours), device=on)
    profiles = {}
    for label, (weekday, holiday, weather) in PROFILE_STATES.items():
        frame = pd.DataFrame({"weekday": [weekday] * len(hours), "holiday": holiday, "weathersit": weather})
        s = torch.as_tensor(data.state(frame), device=on)
        model.eval()
        with torch.no_grad():
            profiles[label] = scaling.counts(model(x, s).cpu().numpy()).tolist()
    return profiles


def run(on: torch.device) -> dict:
    started = time.time()
    parts, scaling = data.prepare()
    train, validation, test, fit = (parts[name] for name in ("train", "validation", "test", "fit"))
    input_dim, state_dim = test.x.shape[1], test.s.shape[1]
    widths = models.widths(input_dim, state_dim)
    results: dict = {
        "device": torch.cuda.get_device_name(0) if on.type == "cuda" else "cpu", "torch": torch.__version__,
        "rows": {name: len(part.y) for name, part in parts.items()},
        "missing_hours": {name: data.missing_hours(part.frame) for name, part in parts.items() if name != "fit"},
        "unusual_hours": int(data.unusual_hours(data.load()).sum()),
        "inputs": list(data.INPUT_NAMES), "states": list(data.STATE_NAMES), "widths": widths,
        "parameters": {k: models.parameter_count(models.Net(k, input_dim, state_dim, w)) for k, w in widths.items()},
        "grid": [asdict(c) for c in GRID], "tuning_seeds": list(TUNING_SEEDS), "final_seeds": list(FINAL_SEEDS),
        "primary": PRIMARY, "tuning": {}, "chosen": {}, "test": {}, "test_runs": {}, "train_only_test": {},
        "monthly_bias": {}, "histories": {}, "profiles": {}}
    predictions = pd.DataFrame({"time": test.frame["time"], "cnt": test.counts})

    ridge_choice = tune_ridge(train, validation, scaling)
    hgb_choice = tune_hgb(train, validation, scaling)
    results["tuning"].update(ridge=ridge_choice["scores"], hgb=hgb_choice["scores"])
    results["chosen"].update(ridge={"alpha": ridge_choice["alpha"]},
                             hgb={k: hgb_choice[k] for k in ("learning_rate", "max_leaf_nodes", "rounds")})
    classical = {
        "ridge": lambda part: Ridge(alpha=ridge_choice["alpha"]).fit(ridge_inputs(part), part.y).predict(ridge_inputs(test)),
        "hgb": lambda part: hgb(hgb_choice["learning_rate"], hgb_choice["max_leaf_nodes"], hgb_choice["rounds"]).fit(
            data.table(part.frame), part.y).predict(data.table(test.frame)),
    }
    for name, refit in classical.items():
        predictions[name] = scaling.counts(refit(fit))
        results["test"][name] = metrics(test.counts, predictions[name].to_numpy())
        results["train_only_test"][name] = metrics(test.counts, scaling.counts(refit(train)))

    errors = {}
    for kind in models.KINDS:
        tuned = tune_net(kind, widths[kind], train, validation, scaling, on)
        results["tuning"][kind] = tuned["scores"]
        results["chosen"][kind] = {**asdict(tuned["config"]), "epochs": tuned["epochs"]}
        results["histories"][kind] = tuned["histories"]
        runs, seed_predictions, profiles = [], [], []
        for seed in FINAL_SEEDS:
            fitted = fit_fixed(kind, widths[kind], tuned["config"], seed, fit, tuned["epochs"], on)
            counts = scaling.counts(predict(fitted.model, test, on))
            predictions[f"{kind} s{seed}"] = counts
            seed_predictions.append(counts)
            runs.append(metrics(test.counts, counts))
            profiles.append(state_profiles(fitted.model, test, scaling, on))
        results["test_runs"][kind] = runs
        results["test"][kind] = summary(runs)
        results["profiles"][kind] = {label: np.mean([p[label] for p in profiles], axis=0).tolist() for label in PROFILE_STATES}
        errors[kind] = squared_log_errors(test.counts, np.stack(seed_predictions))
        predictions[kind] = np.mean(seed_predictions, axis=0)

    months = test.frame["mnth"].to_numpy()
    for name in ("ridge", "hgb", *models.KINDS):
        error = predictions[name].to_numpy() - test.counts
        results["monthly_bias"][name] = {str(m): float(error[months == m].mean()) for m in sorted(set(months))}

    # Gradient boosting is one fixed model: its errors repeated once per seed let the same bootstrap draw the network's
    # seeds and both models' days, and its score repeated per seed gives the seed-only interval.
    errors["hgb"] = np.repeat(squared_log_errors(test.counts, predictions["hgb"].to_numpy()[None, :]), len(FINAL_SEEDS), axis=0)
    per_seed = {name: [r[PRIMARY] for r in runs] for name, runs in results["test_runs"].items()}
    per_seed["hgb"] = [results["test"]["hgb"][PRIMARY]] * len(FINAL_SEEDS)
    days = pd.factorize(test.frame["dteday"])[0]
    pairs = (PRIMARY_PAIR, ("film", "shift"), ("shift", "concat"), ("concat", "no-state"), ("hgb", "concat"))
    results["comparisons"] = {f"{a} - {b}": {
        "bootstrap": paired_bootstrap(errors[a], errors[b], days),
        "seed_t": paired_difference(per_seed[a], per_seed[b]),
    } for a, b in pairs}
    results["seconds"] = round(time.time() - started, 1)
    return {"metrics": results, "predictions": predictions}


def save(outcome: dict, folder: pathlib.Path = RESULTS) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(outcome["metrics"], indent=1, sort_keys=True) + "\n", encoding="utf-8",
                                          newline="\n")
    outcome["predictions"].to_csv(folder / "predictions.csv", index=False, float_format="%.3f", lineterminator="\n")


def main() -> None:
    deterministic()
    on = device()
    print(f"training on {on}")
    outcome = run(on)
    save(outcome)
    for name, value in outcome["metrics"]["test"].items():
        print(name, json.dumps(value))
    print(json.dumps(outcome["metrics"]["comparisons"], indent=1))
    print("seconds", outcome["metrics"]["seconds"])


if __name__ == "__main__":
    main()
