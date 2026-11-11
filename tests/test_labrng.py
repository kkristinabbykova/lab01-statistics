"""Автотесты датчиков. Пороги p > 0.001: при правильном датчике ложное падение маловероятно,
а seed фиксирован, так что тесты детерминированы."""
import math

import numpy as np
import pytest
from scipy import stats

from labrng import generators as G

N = 20_000


@pytest.mark.parametrize("base", ["pcg64", "mt19937", "lcg"])
def test_uniform_is_uniform(base):
    u = G.Uniform(base, seed=7)(N)
    assert ((u > 0) & (u < 1)).all()
    assert stats.kstest(u, "uniform").pvalue > 1e-3
    # соседние значения не коррелированы
    assert abs(np.corrcoef(u[:-1], u[1:])[0, 1]) < 0.03


@pytest.mark.parametrize("base", G.BASES)
def test_seed_reproducible(base):
    assert np.array_equal(G.Uniform(base, 5)(100), G.Uniform(base, 5)(100))
    assert not np.array_equal(G.Uniform(base, 5)(100), G.Uniform(base, 6)(100))


def test_randu_lattice():
    """Известный дефект RANDU: x_{k+2} = 6·x_{k+1} − 9·x_k (mod 2³¹), тройки лежат на 15 плоскостях."""
    x = np.round(G.Uniform("randu", 1)(1000) * 2**31).astype(np.int64)
    assert np.all((x[2:] - 6 * x[1:-1] + 9 * x[:-2]) % 2**31 == 0)


@pytest.mark.parametrize("method", ["box_muller", "polar", "ratio"])
def test_standard_normal(method):
    z, acc = G.std_normal(G.Uniform("pcg64", 11), N, method)
    assert z.size == N
    assert stats.kstest(z, "norm").pvalue > 1e-3
    expected = {"box_muller": 1.0, "polar": math.pi / 4, "ratio": math.sqrt(math.pi * math.e) / 4}[method]
    assert acc == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize("family,params", [("norm", (3.2, 0.15)), ("lognorm", (0.13, 0.42, 6.38)),
                                           ("skewnorm", (1.8, 3.04, 0.21))])
def test_from_normal_matches_scipy(family, params):
    U = G.Uniform("pcg64", 3)
    z0, z1 = G.box_muller(U, N), G.box_muller(U, N)
    x = G.from_normal(family, params, z0, z1)
    assert stats.kstest(x, getattr(stats, family)(*params).cdf).pvalue > 1e-3


@pytest.fixture(scope="module")
def sample():
    return stats.skewnorm.rvs(1.8, 3.04, 0.21, size=4898, random_state=1)


@pytest.mark.parametrize("method", G.METHODS)
def test_generate_like_data(method, sample):
    x, info = G.generate(sample, 5000, method=method, family="skewnorm", seed=2026)
    assert x.size == 5000
    assert stats.ks_2samp(x, sample).pvalue > 1e-3
    assert abs(x.mean() - sample.mean()) < 0.01
    assert 0 < info["accept"] <= 1


def test_neumann_acceptance(sample):
    dens, edges = np.histogram(sample, bins=40, density=True)
    _, acc = G.neumann(G.Uniform("pcg64", 1), sample, N)
    assert acc == pytest.approx(1 / (dens.max() * (edges[-1] - edges[0])), rel=0.05)


def test_rounding_to_instrument_step(sample):
    x, _ = G.generate(sample, 1000, step=0.01)
    assert np.allclose(np.round(x * 100), x * 100)


def test_quantile_is_inverse_of_ecdf():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    assert G.empirical_quantile(data, np.array([0.125, 0.375, 0.5])) == pytest.approx([1.0, 2.0, 2.5])


def test_compare_same_data(sample):
    r = G.compare(sample, sample)
    assert r["ks_D"] == 0 and r["chi2_p"] > 0.99
    assert r["real"]["среднее"] == r["gen"]["среднее"]


def test_rounding_matches_data_grid():
    """Округлённые значения должны совпадать с записью в данных (регрессия: 6.800000000000001 != 6.8)."""
    data = np.round(stats.norm.rvs(6.85, 0.84, size=4898, random_state=2) * 10) / 10
    x, _ = G.generate(data, 3000, method="quantile", step=0.1)
    assert set(np.unique(x)) <= set(np.unique(data)) | set(np.round(np.arange(0, 20, 0.1), 10))
    assert stats.ks_2samp(x, data).pvalue > 1e-3
