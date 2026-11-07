"""Обёртки над критериями: единый вывод {критерий, H0, статистика, p, решение}."""
from __future__ import annotations

import numpy as np
from scipy import stats

ALPHA = 0.05


def _result(test: str, h0: str, stat: float, p: float, alpha: float = ALPHA, **extra) -> dict:
    return {"test": test, "H0": h0, "stat": float(stat), "p": float(p),
            "decision": "отвергаем H0" if p < alpha else "нет оснований отвергнуть H0", **extra}


def normality_dagostino(x, alpha: float = ALPHA) -> dict:
    """K² Д'Агостино–Пирсона: объединяет отклонения асимметрии и эксцесса от нуля."""
    r = stats.normaltest(x)
    return _result("D'Agostino K²", "X ~ нормальное", r.statistic, r.pvalue, alpha)


def shapiro(x, alpha: float = ALPHA) -> dict:
    r = stats.shapiro(x)
    return _result("Shapiro–Wilk", "X ~ нормальное", r.statistic, r.pvalue, alpha)


def welch_t(a, b, alpha: float = ALPHA) -> dict:
    """t-критерий Уэлча: средние двух совокупностей, дисперсии не считаются равными."""
    r = stats.ttest_ind(a, b, equal_var=False)
    return _result("Welch t", "E[A] = E[B]", r.statistic, r.pvalue, alpha,
                   diff=float(np.mean(a) - np.mean(b)))


def mann_whitney(a, b, alternative: str = "two-sided", alpha: float = ALPHA) -> dict:
    """U Манна–Уитни: P(A > B) = 1/2 (сдвиг распределений), не требует нормальности."""
    r = stats.mannwhitneyu(a, b, alternative=alternative)
    auc = r.statistic / (len(a) * len(b))  # оценка P(A > B)
    h0 = {"two-sided": "P(A > B) = 1/2", "greater": "P(A > B) ≤ 1/2", "less": "P(A > B) ≥ 1/2"}[alternative]
    return _result(f"Mann–Whitney ({alternative})", h0, r.statistic, r.pvalue, alpha,
                   p_a_gt_b=float(auc))


def holm(pvalues, alpha: float = ALPHA) -> np.ndarray:
    """Поправка Холма на множественные проверки: вернуть маску отвергнутых H0."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    reject = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject
