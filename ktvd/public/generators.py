"""Датчики случайных чисел для интерактивного ноутбука (КТвД, marimo).

Модуль самодостаточный: только numpy и scipy, без labstat. Поэтому его копия
работает и в браузере (marimo → WebAssembly, Pyodide).

Схема: базовый равномерный датчик U(0, 1) → метод → выборка из нужного распределения.

Базовые датчики
    pcg64      numpy по умолчанию (O'Neill, 2014)
    mt19937    вихрь Мерсенна (Matsumoto, Nishimura, 1998)
    lcg        линейный конгруэнтный, константы MMIX Кнута: x ← a·x + c mod 2⁶⁴
    randu      «плохой» ЛКГ IBM: x ← 65539·x mod 2³¹; тройки лежат на 15 плоскостях

Методы
    quantile   квантильное преобразование: X = F̂⁻¹(U), F̂ — эмпирическая ФР
               с линейной интерполяцией между порядковыми статистиками
    inverse    метод обратной функции по подобранной модели: X = F⁻¹(U; θ̂)
    box_muller Бокс–Мюллер: Z = √(−2 ln U₁)·cos(2πU₂), затем Z → модель
    polar      полярный метод Марсальи (без синуса и косинуса), затем Z → модель
    ratio      метод отношений Киндермана–Монахана для N(0, 1), затем Z → модель
    neumann    метод Неймана (отбор): точка (x, y) равномерно в прямоугольнике
               под max f, принимается при y ≤ f(x); f — гистограммная плотность данных

Переход от N(0, 1) к модели:
    norm        X = μ + σZ
    lognorm     X = θ + e^{μ}·exp(sZ)
    skewnorm    X = loc + scale·(δ|Z₀| + √(1−δ²)·Z₁),  δ = a/√(1+a²)
                (представление Аззалини: нужны две независимые N(0, 1))
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

BASES = ("pcg64", "mt19937", "lcg", "randu")
METHODS = ("quantile", "inverse", "box_muller", "polar", "ratio", "neumann")
FAMILIES = ("norm", "lognorm", "skewnorm")

METHOD_RU = {
    "quantile": "Квантильное преобразование (эмпирическая F⁻¹)",
    "inverse": "Метод обратной функции по модели",
    "box_muller": "Бокс–Мюллер → модель",
    "polar": "Полярный метод Марсальи → модель",
    "ratio": "Метод отношений (Киндерман–Монахан) → модель",
    "neumann": "Метод Неймана (отбор) по гистограмме",
}
BASE_RU = {
    "pcg64": "PCG64 (numpy по умолчанию)",
    "mt19937": "Вихрь Мерсенна MT19937",
    "lcg": "ЛКГ, константы MMIX (Кнут)",
    "randu": "RANDU (плохой ЛКГ IBM, 1960-е)",
}


# ----------------------------------------------------------------------- базовые датчики U(0, 1)
class Uniform:
    """Поток равномерных чисел в (0, 1). Ноль исключён: нужен для ln U."""

    def __init__(self, base: str = "pcg64", seed: int = 2026):
        if base not in BASES:
            raise ValueError(f"base должен быть одним из {BASES}")
        self.base = base
        if base == "pcg64":
            self._g = np.random.Generator(np.random.PCG64(seed))
        elif base == "mt19937":
            self._g = np.random.Generator(np.random.MT19937(seed))
        elif base == "lcg":
            self._x = np.uint64(seed) | np.uint64(1)
        else:   # randu: начальное значение должно быть нечётным
            self._x = (int(seed) | 1) % 2**31

    def __call__(self, n: int) -> np.ndarray:
        n = int(n)
        if self.base in ("pcg64", "mt19937"):
            u = self._g.random(n)
            u[u == 0] = 0.5 / 2**53
            return u
        if self.base == "lcg":
            a, c = np.uint64(6364136223846793005), np.uint64(1442695040888963407)
            out = np.empty(n, dtype=np.uint64)
            x = self._x
            with np.errstate(over="ignore"):
                for i in range(n):
                    x = a * x + c
                    out[i] = x
            self._x = x
            return ((out >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0**53   # старшие 53 бита
        out = np.empty(n)
        x = self._x
        for i in range(n):
            x = (65539 * x) % 2**31
            out[i] = x
        self._x = x
        return out / 2**31


# ----------------------------------------------------------------------- N(0, 1)
def box_muller(U: Uniform, n: int) -> np.ndarray:
    m = (n + 1) // 2
    u1, u2 = U(m), U(m)
    r = np.sqrt(-2.0 * np.log(u1))
    return np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])[:n]


def polar(U: Uniform, n: int) -> tuple[np.ndarray, float]:
    """Возвращает (выборка, доля принятых пар). Теоретическая доля π/4 ≈ 0.785."""
    out, tried, got = [], 0, 0
    while got < n:
        m = max(16, int((n - got) / 0.78 / 2) + 16)
        v1, v2 = 2 * U(m) - 1, 2 * U(m) - 1
        s = v1 * v1 + v2 * v2
        ok = (s > 0) & (s < 1)
        f = np.sqrt(-2 * np.log(s[ok]) / s[ok])
        out += [v1[ok] * f, v2[ok] * f]
        tried += m
        got += 2 * int(ok.sum())
    return np.concatenate(out)[:n], got / 2 / tried


def ratio_of_uniforms(U: Uniform, n: int) -> tuple[np.ndarray, float]:
    """Киндерман–Монахан: (u, v) равномерно в [0,1]×[−√(2/e), √(2/e)], X = v/u,
    принимается при X² ≤ −4 ln u. Теоретическая доля принятых √(π e)/4 ≈ 0.731."""
    b = math.sqrt(2 / math.e)
    out, tried, got = [], 0, 0
    while got < n:
        m = max(16, int((n - got) / 0.73) + 16)
        u, v = U(m), (2 * U(m) - 1) * b
        x = v / u
        ok = x * x <= -4 * np.log(u)
        out.append(x[ok])
        tried += m
        got += int(ok.sum())
    return np.concatenate(out)[:n], got / tried


def std_normal(U: Uniform, n: int, method: str) -> tuple[np.ndarray, float]:
    if method == "box_muller":
        return box_muller(U, n), 1.0
    if method == "polar":
        return polar(U, n)
    if method == "ratio":
        return ratio_of_uniforms(U, n)
    raise ValueError(method)


# ----------------------------------------------------------------------- модели
def fit(family: str, x) -> tuple:
    """Параметры scipy (ММП). Для lognorm и skewnorm со сдвигом."""
    return getattr(stats, family).fit(np.asarray(x, float))


def from_normal(family: str, params: tuple, z0: np.ndarray, z1: np.ndarray | None = None) -> np.ndarray:
    if family == "norm":
        loc, scale = params
        return loc + scale * z0
    if family == "lognorm":
        s, loc, scale = params
        return loc + scale * np.exp(s * z0)
    if family == "skewnorm":
        a, loc, scale = params
        d = a / math.sqrt(1 + a * a)
        return loc + scale * (d * np.abs(z0) + math.sqrt(1 - d * d) * z1)
    raise ValueError(family)


def empirical_quantile(data, u: np.ndarray) -> np.ndarray:
    """F̂⁻¹(u): кусочно-линейная обратная к эмпирической ФР (узлы (i−0.5)/n)."""
    xs = np.sort(np.asarray(data, float))
    p = (np.arange(1, xs.size + 1) - 0.5) / xs.size
    return np.interp(u, p, xs)


def neumann(U: Uniform, data, n: int, bins: int = 40) -> tuple[np.ndarray, float]:
    """Отбор по гистограммной плотности. Внутри интервала точка равномерна."""
    dens, edges = np.histogram(np.asarray(data, float), bins=bins, density=True)
    lo, hi, top = edges[0], edges[-1], dens.max()
    out, tried, got = [], 0, 0
    while got < n:
        m = max(64, int((n - got) * top * (hi - lo)) + 64)
        x = lo + (hi - lo) * U(m)
        y = top * U(m)
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
        ok = y <= dens[idx]
        out.append(x[ok])
        tried += m
        got += int(ok.sum())
    return np.concatenate(out)[:n], got / tried   # теория: 1 / (top·(hi−lo))


# ----------------------------------------------------------------------- единая точка входа
def generate(data, n: int, *, method: str = "quantile", family: str = "skewnorm",
             base: str = "pcg64", seed: int = 2026, step: float = 0.0, params: tuple | None = None):
    """Сгенерировать n чисел «как data». Возвращает (выборка, служебная информация).

    step > 0: округлить к шагу записи прибора (у pH 0.01, у кислотности и спирта 0.1).
    """
    if method not in METHODS:
        raise ValueError(f"method должен быть одним из {METHODS}")
    U = Uniform(base, seed)
    info = {"method": method, "base": base, "family": family, "accept": 1.0}
    if method == "quantile":
        x = empirical_quantile(data, U(n))
    elif method == "neumann":
        x, info["accept"] = neumann(U, data, n)
    else:
        params = params if params is not None else fit(family, data)
        info["params"] = tuple(float(p) for p in params)
        if method == "inverse":
            x = getattr(stats, family).ppf(U(n), *params)
        else:
            z0, acc = std_normal(U, n, method)
            z1 = std_normal(U, n, method)[0] if family == "skewnorm" else None
            x, info["accept"] = from_normal(family, params, z0, z1), acc
    if step:
        # второе округление убирает хвосты вида 6.800000000000001: иначе значения
        # не совпадают с записанными в данных 6.8, и КС видит ложные ступеньки ФР
        x = np.round(np.round(x / step) * step, 10)
    return x, info


def compare(real, gen) -> dict:
    """Метрики реальных и сгенерированных данных + двухвыборочные критерии."""
    real, gen = np.asarray(real, float), np.asarray(gen, float)

    def m(x):
        q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
        return {"среднее": x.mean(), "СКО": x.std(ddof=1), "асимметрия": stats.skew(x),
                "эксцесс": stats.kurtosis(x), "q05": q[0], "q25": q[1], "медиана": q[2], "q75": q[3], "q95": q[4]}

    ks = stats.ks_2samp(real, gen)
    edges = np.unique(np.quantile(real, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    o_r = np.histogram(real, edges)[0]
    o_g = np.histogram(gen, edges)[0]
    chi2, p_chi, dof, _ = stats.chi2_contingency(np.vstack([o_r, o_g]))
    return {"real": m(real), "gen": m(gen), "ks_D": float(ks.statistic), "ks_p": float(ks.pvalue),
            "chi2": float(chi2), "chi2_p": float(p_chi), "chi2_dof": int(dof)}
