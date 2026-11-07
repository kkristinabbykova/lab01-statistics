"""Автотесты датчика pH (спринт 2). Автор: Быкова К. Д."""
import numpy as np

from ph_generator import polar_normal


def test_polar_mean():
    z, acc = polar_normal(20000, np.random.default_rng(1))
    assert np.abs(np.mean(z)) < 0.03


def test_acceptance():
    z, acc = polar_normal(20000, np.random.default_rng(2))
    assert np.abs(acc - np.pi / 4) < 0.02

