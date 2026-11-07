"""Нормализация и преобразования шкалы."""
from __future__ import annotations

import numpy as np
from scipy import stats


def zscore(x, mu: float | None = None, sigma: float | None = None) -> np.ndarray:
    """z = (x - mu) / sigma. Если X ~ N(mu, sigma²), то Z ~ N(0, 1)."""
    x = np.asarray(x, dtype=float)
    mu = x.mean() if mu is None else mu
    sigma = x.std(ddof=1) if sigma is None else sigma
    return (x - mu) / sigma


def minmax(x) -> np.ndarray:
    """Перевод в [0, 1]. Форму распределения не меняет, только сдвиг и масштаб."""
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min())


def log_z(x) -> np.ndarray:
    """Для логнормальной величины: ln X ~ N(mu, sigma²), дальше z-преобразование."""
    return zscore(np.log(np.asarray(x, dtype=float)))


def boxcox(x):
    """Преобразование Бокса–Кокса: (x^lambda - 1)/lambda, lambda подбирается по максимуму правдоподобия."""
    y, lam = stats.boxcox(np.asarray(x, dtype=float))
    return y, lam
