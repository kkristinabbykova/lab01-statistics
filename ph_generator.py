"""Датчик случайных значений pH белого вина (спринт 2).

Автор: Быкова К. Д. (ИСУ 501322).
pH ~ логнормальное со сдвигом: ln(X - theta) ~ N(mu, sigma^2).
Параметры подобраны в спринте 1 (scipy lognorm: s, loc, scale = 0.1363, 2.0877, 1.0904).
Шаг 1: полярный метод Марсальи для N(0, 1).
"""
import numpy as np

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

