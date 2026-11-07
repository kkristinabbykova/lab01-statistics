"""Зависимость и независимость двух величин."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def correlations(x, y) -> pd.DataFrame:
    """Пирсон (линейная связь), Спирмен и Кендалл (монотонная связь) с p-значениями."""
    rows = []
    for name, fn in [("pearson", stats.pearsonr), ("spearman", stats.spearmanr), ("kendall", stats.kendalltau)]:
        r = fn(x, y)
        rows.append({"method": name, "coef": float(r.statistic), "p": float(r.pvalue)})
    return pd.DataFrame(rows).set_index("method")


def chi2_independence(x, y, bins: int = 4) -> dict:
    """Хи-квадрат на таблице сопряжённости по квантильным интервалам.

    H0: P(X in A, Y in B) = P(X in A) * P(Y in B) для всех клеток.
    Работает и для нелинейной связи, в отличие от корреляции.
    """
    bx = pd.qcut(x, bins, labels=False, duplicates="drop")
    by = pd.qcut(y, bins, labels=False, duplicates="drop")
    table = pd.crosstab(bx, by)
    res = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    cramer_v = np.sqrt(res.statistic / (n * k))
    return {"chi2": res.statistic, "dof": res.dof, "p": res.pvalue, "cramer_v": cramer_v, "table": table,
            "expected": pd.DataFrame(res.expected_freq, index=table.index, columns=table.columns)}


def permutation_spearman(x, y, n_perm: int = 2000, seed: int = 0) -> dict:
    """Перестановочный тест: перемешиваем y, разрушая связь, и смотрим, как часто |rho| не меньше наблюдаемого."""
    rng = np.random.default_rng(seed)
    x = stats.rankdata(x)
    y = stats.rankdata(y)
    obs = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"rho": obs, "p": p, "null": null}
