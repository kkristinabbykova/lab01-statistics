"""Описательные статистики одной величины (то, что было в ДЗ с пингвинами)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def describe(x) -> pd.Series:
    """Сводка по выборке x. Дисперсия несмещённая (ddof=1), эксцесс избыточный (у нормального 0)."""
    x = np.asarray(x, dtype=float)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    vals, counts = np.unique(x, return_counts=True)
    return pd.Series(
        {
            "n": x.size,
            "mean": x.mean(),
            "var": x.var(ddof=1),
            "std": x.std(ddof=1),
            "cv": x.std(ddof=1) / x.mean(),
            "min": x.min(),
            "q25": q1,
            "median": med,
            "q75": q3,
            "max": x.max(),
            "iqr": q3 - q1,
            "mode": vals[counts.argmax()],
            "skew": stats.skew(x),
            "kurt": stats.kurtosis(x),
            "n_unique": vals.size,
            "step": np.min(np.diff(vals)) if vals.size > 1 else np.nan,
        }
    )


def iqr_outliers(x, k: float = 1.5) -> np.ndarray:
    """Маска выбросов по правилу Тьюки: вне [Q1 - k*IQR, Q3 + k*IQR]."""
    x = np.asarray(x, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def compare_table(columns: dict) -> pd.DataFrame:
    """Таблица describe() для нескольких наборов: {'название': массив}."""
    return pd.DataFrame({name: describe(v) for name, v in columns.items()})
