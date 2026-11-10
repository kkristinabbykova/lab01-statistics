"""Автотесты датчика pH (спринт 2). Автор: Быкова К. Д."""
import numpy as np
from scipy import stats

from labstat import data
from ph_generator import polar_normal, ph_polar, ph_inverse, theta, mu

ph = data.load_wine("white")["ph"].to_numpy()


def test_polar_mean():
    z, acc = polar_normal(20000, np.random.default_rng(1))
    assert np.abs(np.mean(z)) < 0.03


def test_acceptance():
    z, acc = polar_normal(20000, np.random.default_rng(2))
    assert np.abs(acc - np.pi / 4) < 0.02


def test_seed():
    a, _ = ph_polar(100, np.random.default_rng(5))
    b, _ = ph_polar(100, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_like_data():
    x, _ = ph_polar(len(ph), np.random.default_rng(2026))
    p = stats.ks_2samp(np.round(x, 2), ph).pvalue
    assert p > 0.05


def test_polar_is_standard_normal():
    z, _ = polar_normal(20000, np.random.default_rng(3))
    assert np.abs(np.std(z) - 1) < 0.03
    assert stats.kstest(z, "norm").pvalue > 0.01


def test_inverse_like_data():
    x = ph_inverse(len(ph), np.random.default_rng(2026))
    p = stats.ks_2samp(np.round(x, 2), ph).pvalue
    assert p > 0.05


def test_median_at_zero():
    # при Z = 0 датчик выдаёт theta + e^mu, это должно быть около медианы данных
    assert np.abs(theta + np.exp(mu) - np.median(ph)) < 0.01
