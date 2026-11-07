"""Датчик случайных значений pH белого вина (спринт 2).

Автор: Быкова К. Д. (ИСУ 501322).
pH ~ логнормальное со сдвигом: ln(X - theta) ~ N(mu, sigma^2).
Параметры подобраны в спринте 1 (scipy lognorm: s, loc, scale = 0.1363, 2.0877, 1.0904).
Основной метод: полярный метод Марсальи (N(0,1)) -> X = theta + exp(mu + sigma*Z).
Для сравнения: метод обратных функций X = theta + exp(mu + sigma * Phi^{-1}(U)).
"""
import numpy as np
from scipy import stats

sigma = 0.1363
theta = 2.0877
mu = np.log(1.0904)


def polar_normal(n, rng):
    """Полярный метод Марсальи: n чисел из N(0, 1) и доля принятых точек."""
    result, tried, accepted = [], 0, 0
    while len(result) < n:
        u1, u2 = rng.random(), rng.random()
        v1 = 2 * u1 - 1
        v2 = 2 * u2 - 1
        s = v1**2 + v2**2
        tried += 1
        if s > 0 and s < 1:
            accepted += 1
            factor = np.sqrt(-2 * np.log(s) / s)
            result.append(v1 * factor)
            result.append(v2 * factor)
    return np.array(result[:n]), accepted / tried


def ph_polar(n, rng):
    """pH через полярный метод."""
    z, acc = polar_normal(n, rng)
    x = theta + np.exp(mu + sigma * z)
    return x, acc


def ph_inverse(n, rng):
    """pH через метод обратных функций (для сравнения)."""
    u = rng.random(n)
    z = stats.norm.ppf(u)
    x = theta + np.exp(mu + sigma * z)
    return x
