"""Подбор распределения и проверка согласия."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Кандидаты: нормальное, логнормальное (3 параметра: форма s, сдвиг loc, масштаб),
# гамма (3 параметра), Вейбулл и скошенное нормальное для сравнения.
CANDIDATES = ["norm", "lognorm", "gamma", "weibull_min", "skewnorm"]


def fit(name: str, x, **kw):
    return getattr(stats, name).fit(np.asarray(x, dtype=float), **kw)


def fit_candidates(x, names=CANDIDATES) -> pd.DataFrame:
    """MLE-подбор каждого кандидата; AIC = 2k - 2 ln L (меньше значит лучше), статистика КС D."""
    x = np.asarray(x, dtype=float)
    rows = []
    for name in names:
        d = getattr(stats, name)
        params = d.fit(x)
        ll = np.sum(d.logpdf(x, *params))
        k = len(params)
        D = stats.kstest(x, d.cdf, args=params).statistic
        rows.append({"dist": name, "k": k, "loglik": ll, "AIC": 2 * k - 2 * ll,
                     "BIC": k * np.log(x.size) - 2 * ll, "KS_D": D, "params": tuple(np.round(params, 4))})
    return pd.DataFrame(rows).sort_values("AIC").set_index("dist")


def detect_step(x) -> float:
    """Шаг округления данных: самая частая разность между соседними различными значениями."""
    vals = np.unique(np.round(np.asarray(x, dtype=float), 8))
    d = np.round(np.diff(vals), 6)
    d = d[d > 0]
    if d.size == 0:
        return 0.0
    u, c = np.unique(d, return_counts=True)
    return float(u[c.argmax()])


def round_to_grid(x, step: float, origin: float = 0.0):
    """Округлить к сетке origin + k·step (так записаны данные прибора)."""
    if not step:
        return np.asarray(x, dtype=float)
    return origin + np.round((np.asarray(x, dtype=float) - origin) / step) * step


def _fit_grouped(name: str, edges, obs, start):
    """Оценка параметров по сгруппированным данным: максимум мультиномиального правдоподобия
    sum(o_i * ln p_i(θ)). Именно для таких оценок χ² имеет df = k − 1 − (число параметров)."""
    from scipy import optimize

    d = getattr(stats, name)

    def nll(theta):
        pr = np.diff(d.cdf(edges, *theta))
        if not np.all(np.isfinite(pr)) or np.any(pr <= 0):
            return 1e300
        return -np.sum(obs * np.log(pr))

    res = optimize.minimize(nll, np.asarray(start, dtype=float), method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
    return tuple(res.x)


def chi2_gof(x, name: str, params=None, n_bins: int = 20, step: float | None = None,
             min_expected: float = 5.0) -> dict:
    """Хи-квадрат Пирсона для проверки согласия с распределением name(params).

    1. Данные записаны с шагом step (pH 0.01, спирт 0.1). Значение 3.18 на самом деле
       означает интервал [3.175; 3.185). Поэтому границы интервалов ставим на ПОЛУШАГАХ
       сетки: каждое записанное значение целиком попадает в один интервал, и ожидаемая
       частота считается для того же интервала, что и наблюдаемая.
    2. Границы выбираем около квантилей данных (примерно равные частоты), затем соседние
       интервалы объединяем, пока ожидаемая частота < min_expected.
    3. Степени свободы: (число интервалов) − 1 − (число оценённых параметров).
       Это верно, если параметры оценены по тем же интервалам. Поэтому при params=None
       параметры оцениваются по сгруппированным данным (_fit_grouped). Если передать
       params, оценённые по исходным данным, критерий становится слегка либеральным
       (теорема Чернова–Лемана). Калибровка в tests/ проверяет долю ложных отвержений ≈ 5 %.
    """
    x = np.asarray(x, dtype=float)
    d = getattr(stats, name)
    step = detect_step(x) if step is None else step
    q = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    if step:
        origin = x.min()
        q = origin + (np.floor((q - origin) / step) + 0.5) * step
    edges = np.concatenate([[-np.inf], np.unique(q), [np.inf]])
    obs = np.histogram(x, edges)[0].astype(float)
    grouped = params is None
    if grouped:
        params = _fit_grouped(name, edges, obs, d.fit(x))
    exp = np.diff(d.cdf(edges, *params)) * x.size
    o, e = [], []
    acc_o = acc_e = 0.0
    for oi, ei in zip(obs, exp):
        acc_o += oi
        acc_e += ei
        if acc_e >= min_expected:
            o.append(acc_o); e.append(acc_e); acc_o = acc_e = 0.0
    if acc_e > 0:
        o[-1] += acc_o; e[-1] += acc_e
    o, e = np.array(o), np.array(e)
    chi2 = np.sum((o - e) ** 2 / e)
    dof = len(o) - 1 - len(params)
    p = stats.chi2.sf(chi2, dof)
    return {"chi2": chi2, "dof": dof, "p": p, "bins": len(o), "step": step, "params": tuple(params),
            "fit": "по интервалам" if grouped else "заданы",
            "decision": "отвергаем H0" if p < 0.05 else "нет оснований отвергнуть H0"}


def ks_bootstrap(x, name: str, B: int = 300, step: float | None = None, seed: int = 0) -> dict:
    """КС с оценёнными параметрами через параметрический бутстреп (идея Лиллиефорса).

    Табличное p-значение КС неверно, если параметры оценены по той же выборке, и ещё
    сильнее неверно для округлённых данных (много совпадений). Поэтому:
    моделируем B выборок из подобранного распределения, ОКРУГЛЯЕМ их к той же сетке,
    что и данные, на каждой заново оцениваем параметры и считаем D.
    p = (число D_boot >= D_obs + 1) / (B + 1).
    """
    x = np.asarray(x, dtype=float)
    d = getattr(stats, name)
    step = detect_step(x) if step is None else step
    origin = x.min()
    rng = np.random.default_rng(seed)
    params = d.fit(x)
    D_obs = stats.kstest(x, d.cdf, args=params).statistic
    D_boot = np.empty(B)
    for b in range(B):
        xb = round_to_grid(d.rvs(*params, size=x.size, random_state=rng), step, origin)
        pb = d.fit(xb)
        D_boot[b] = stats.kstest(xb, d.cdf, args=pb).statistic
    p = (np.sum(D_boot >= D_obs) + 1) / (B + 1)
    return {"D": D_obs, "p": p, "params": params, "step": step, "B": B,
            "decision": "отвергаем H0" if p < 0.05 else "нет оснований отвергнуть H0"}


def calibration(name: str, params, n: int, step: float, sims: int = 200, B: int = 0,
                alpha: float = 0.05, seed: int = 0) -> dict:
    """Доля отвержений H0 на данных, которые ДЕЙСТВИТЕЛЬНО из name(params) и округлены к шагу step.

    Для правильного критерия она должна быть около alpha. Если B > 0, проверяется и КС-бутстреп
    (он дорогой, поэтому для него берётся min(sims, 50) повторов).
    """
    d = getattr(stats, name)
    rng = np.random.default_rng(seed)
    rej_chi2, rej_ks = [], []
    for i in range(sims):
        x = round_to_grid(d.rvs(*params, size=n, random_state=rng), step)
        rej_chi2.append(chi2_gof(x, name, step=step)["p"] < alpha)
        if B and i < min(sims, 50):
            rej_ks.append(ks_bootstrap(x, name, B=B, step=step, seed=int(rng.integers(1 << 31)))["p"] < alpha)
    out = {"chi2_reject_rate": float(np.mean(rej_chi2)), "chi2_sims": sims}
    if rej_ks:
        out |= {"ks_reject_rate": float(np.mean(rej_ks)), "ks_sims": len(rej_ks)}
    return out
