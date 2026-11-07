"""Эмпирическая проверка состоятельности оценок.

Оценка theta_n состоятельна, если theta_n -> theta по вероятности:
для любого eps > 0  P(|theta_n - theta| > eps) -> 0 при n -> inf.
Генеральная совокупность здесь конечна (весь датасет), поэтому выборки
берём с возвращением: это n независимых наблюдений из эмпирического
распределения совокупности, и истинное theta = theta(совокупности).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

ESTIMATORS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean,
    "var": lambda s: np.var(s, ddof=1),
    "median": np.median,
}

# Значение параметра для совокупности (для дисперсии делим на N, это дисперсия распределения совокупности)
POPULATION_VALUE: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean,
    "var": lambda s: np.var(s, ddof=0),
    "median": np.median,
}


def consistency_curve(population, estimator: str, ns, reps: int = 1000, eps: float | None = None,
                      seed: int = 0) -> pd.DataFrame:
    """Для каждого n: смещение, СКО оценки и доля |theta_n - theta| > eps по reps выборкам."""
    pop = np.asarray(population, dtype=float)
    est = ESTIMATORS[estimator]
    theta = POPULATION_VALUE[estimator](pop)
    if eps is None:
        # для среднего и медианы eps = 0.1·σ совокупности, для дисперсии eps = 0.1·σ²
        eps = 0.1 * pop.std() if estimator != "var" else 0.1 * pop.var()
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        samples = rng.choice(pop, size=(reps, n), replace=True)
        values = np.apply_along_axis(est, 1, samples)
        err = values - theta
        rows.append({"n": n, "theta": theta, "mean_est": values.mean(), "bias": err.mean(),
                     "sd_est": values.std(ddof=1), "p_far": np.mean(np.abs(err) > eps), "eps": eps})
    return pd.DataFrame(rows)


def running_estimate(population, estimator: str, n_max: int | None = None, seed: int = 0):
    """Одна траектория: оценка по первым n элементам одной случайной перестановки, n = 2..n_max."""
    pop = np.asarray(population, dtype=float)
    rng = np.random.default_rng(seed)
    x = rng.choice(pop, size=n_max or pop.size, replace=True)
    est = ESTIMATORS[estimator]
    ns = np.unique(np.geomspace(2, x.size, 200).astype(int))
    return ns, np.array([est(x[:n]) for n in ns]), POPULATION_VALUE[estimator](pop)
