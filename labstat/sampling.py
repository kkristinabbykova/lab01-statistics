"""Выборка из генеральной совокупности: с возвращением и без.

Стандартная ошибка выборочного среднего:
  с возвращением:  SE = sigma / sqrt(n)
  без возвращения: SE = sigma / sqrt(n) * sqrt((N - n) / (N - 1))   (поправка на конечность совокупности)
Здесь sigma это СКО совокупности (делим на N).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def draw(population, n: int, replace: bool, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray(population), size=n, replace=replace)


def se_mean_theory(population, n: int, replace: bool) -> float:
    pop = np.asarray(population, dtype=float)
    N = pop.size
    se = pop.std(ddof=0) / np.sqrt(n)
    if not replace:
        se *= np.sqrt((N - n) / (N - 1))
    return float(se)


def se_mean_experiment(population, ns, reps: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Сравнить СКО выборочного среднего по reps повторным выборкам с теорией для обоих способов."""
    pop = np.asarray(population, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        with_r = rng.choice(pop, size=(reps, n), replace=True).mean(axis=1)
        without = np.array([rng.choice(pop, size=n, replace=False).mean() for _ in range(reps)])
        rows.append({
            "n": n, "n/N": n / pop.size,
            "se_with_emp": with_r.std(ddof=1), "se_with_theory": se_mean_theory(pop, n, True),
            "se_without_emp": without.std(ddof=1), "se_without_theory": se_mean_theory(pop, n, False),
        })
    return pd.DataFrame(rows)


def ci_mean(sample, level: float = 0.95, N: int | None = None) -> tuple[float, float]:
    """Доверительный интервал Стьюдента для среднего; если задан N, с поправкой на конечность."""
    from scipy import stats

    s = np.asarray(sample, dtype=float)
    n = s.size
    se = s.std(ddof=1) / np.sqrt(n)
    if N is not None:
        se *= np.sqrt((N - n) / (N - 1))
    t = stats.t.ppf(0.5 + level / 2, df=n - 1)
    return float(s.mean() - t * se), float(s.mean() + t * se)
