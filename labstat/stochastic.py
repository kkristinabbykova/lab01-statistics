"""Проверки стохастической природы величины.

Идея: если бы величина была детерминированной функцией остальных
измерений, то (1) модель по остальным признакам объясняла бы её почти
полностью, с остатками порядка точности прибора; (2) у вин с одинаковыми
остальными измерениями значение совпадало бы.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cv_r2_linear(X, y, k: int = 5, seed: int = 0) -> dict:
    """R² и СКО остатков линейной регрессии y по X на k-кратной кросс-валидации (numpy, без sklearn)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    resid = np.empty_like(y)
    for f in folds:
        train = np.setdiff1d(idx, f)
        A = np.column_stack([np.ones(train.size), X[train]])
        beta, *_ = np.linalg.lstsq(A, y[train], rcond=None)
        pred = np.column_stack([np.ones(f.size), X[f]]) @ beta
        resid[f] = y[f] - pred
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)
    return {"r2_cv": r2, "resid_std": resid.std(ddof=1), "resid": resid}


def cv_r2_knn(X, y, k_neighbors: int = 15, k: int = 5, seed: int = 0) -> dict:
    """То же, что cv_r2_linear, но прогноз нелинейный: среднее y у k ближайших соседей.

    Признаки стандартизируются по обучающей части. Если бы y была любой (не только линейной)
    гладкой функцией остальных измерений, ближайшие соседи давали бы почти точный прогноз.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    resid = np.empty_like(y)
    for f in np.array_split(idx, k):
        train = np.setdiff1d(idx, f)
        mu, sd = X[train].mean(axis=0), X[train].std(axis=0)
        sd[sd == 0] = 1
        A, Bm = (X[train] - mu) / sd, (X[f] - mu) / sd
        d2 = (Bm**2).sum(1)[:, None] + (A**2).sum(1)[None, :] - 2 * Bm @ A.T
        nn = np.argpartition(d2, k_neighbors, axis=1)[:, :k_neighbors]
        resid[f] = y[f] - y[train][nn].mean(axis=1)
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)
    return {"r2_cv": r2, "resid_std": resid.std(ddof=1), "resid": resid}


def same_inputs_different_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Группы вин, у которых ВСЕ остальные измерения совпадают, а target разный."""
    others = [c for c in df.columns if c != target]
    g = df.groupby(others)[target].agg(["nunique", "min", "max", "count"])
    return g[g["nunique"] > 1].reset_index(drop=True)


def runs_test(x) -> dict:
    """Критерий серий Вальда–Вольфовица (выше/ниже медианы) для порядка значений в таблице.

    H0: порядок строк случаен. Мало серий (z < 0) значит, что похожие значения идут подряд.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    s = x[x != med] > med
    n1, n2 = s.sum(), (~s).sum()
    runs = 1 + np.sum(s[1:] != s[:-1])
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / np.sqrt(var)
    return {"runs": int(runs), "expected": mu, "z": z, "p": 2 * stats.norm.sf(abs(z))}


def lag_autocorr(x, lag: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
