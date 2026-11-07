"""Автотесты датчика pH (спринт 2). Автор: Быкова К. Д."""
import numpy as np
from scipy import stats

from labstat import data
from ph_generator import polar_normal, ph_polar

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
