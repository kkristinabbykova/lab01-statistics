"""Автотесты библиотеки labstat. Запуск из корня репозитория: python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from labstat import data, dependence, describe, distributions, estimation, hypotheses, normalize, sampling, stochastic

RNG = np.random.default_rng(42)


# ---------- данные
def test_white_wine_shape_and_columns():
    df = data.load_wine("white")
    assert df.shape == (4898, 12)
    assert {"ph", "fixed_acidity", "alcohol", "quality"} <= set(df.columns)
    assert df.isna().sum().sum() == 0


def test_red_wine_shape():
    assert data.load_wine("red").shape == (1599, 12)


def test_wrong_kind_raises():
    with pytest.raises(ValueError):
        data.load_wine("rose")


# ---------- описательные статистики
def test_describe_matches_numpy():
    x = RNG.normal(5, 2, 1000)
    d = describe.describe(x)
    assert d["mean"] == pytest.approx(x.mean())
    assert d["var"] == pytest.approx(x.var(ddof=1))
    assert d["median"] == pytest.approx(np.median(x))
    assert d["skew"] == pytest.approx(stats.skew(x))


def test_iqr_outliers_finds_planted_values():
    x = np.concatenate([RNG.normal(0, 1, 500), [15.0, -15.0]])
    mask = describe.iqr_outliers(x)
    assert mask[-1] and mask[-2]
    assert mask.mean() < 0.05


# ---------- стохастичность
def test_cv_r2_close_to_one_for_deterministic_target():
    X = RNG.normal(size=(500, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5
    assert stochastic.cv_r2_linear(X, y)["r2_cv"] > 0.999


def test_cv_r2_near_zero_for_pure_noise():
    X = RNG.normal(size=(500, 3))
    y = RNG.normal(size=500)
    assert stochastic.cv_r2_linear(X, y)["r2_cv"] < 0.05


def test_runs_test_random_order_not_rejected_and_sorted_rejected():
    x = RNG.normal(size=2000)
    assert stochastic.runs_test(x)["p"] > 0.01
    assert stochastic.runs_test(np.sort(x))["p"] < 1e-10


def test_same_inputs_different_target():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4], "t": [10, 11, 12]})
    g = stochastic.same_inputs_different_target(df, "t")
    assert len(g) == 1 and g.loc[0, "min"] == 10 and g.loc[0, "max"] == 11


# ---------- зависимость
def test_independent_variables_not_rejected():
    x, y = RNG.normal(size=3000), RNG.normal(size=3000)
    assert dependence.chi2_independence(x, y)["p"] > 0.01
    assert dependence.permutation_spearman(x, y, n_perm=500)["p"] > 0.01


def test_dependent_variables_rejected():
    x = RNG.normal(size=3000)
    y = -x + RNG.normal(size=3000)
    corr = dependence.correlations(x, y)
    assert corr.loc["spearman", "coef"] < -0.5
    assert dependence.chi2_independence(x, y)["p"] < 1e-10


def test_chi2_finds_nonmonotonic_dependence():
    # корреляция около нуля, но связь есть: y = x² + шум
    x = RNG.normal(size=3000)
    y = x**2 + 0.1 * RNG.normal(size=3000)
    assert abs(dependence.correlations(x, y).loc["spearman", "coef"]) < 0.1
    assert dependence.chi2_independence(x, y)["p"] < 1e-10


# ---------- состоятельность
@pytest.mark.parametrize("est", ["mean", "var", "median"])
def test_consistency_probability_goes_to_zero(est):
    pop = RNG.lognormal(0, 0.3, 5000)
    df = estimation.consistency_curve(pop, est, [10, 100, 3000], reps=300)
    assert df["p_far"].iloc[0] > df["p_far"].iloc[-1]
    assert df["p_far"].iloc[-1] < 0.05


def test_mean_estimator_sd_scales_as_root_n():
    pop = RNG.normal(0, 1, 5000)
    df = estimation.consistency_curve(pop, "mean", [25, 2500], reps=500)
    ratio = df["sd_est"].iloc[0] / df["sd_est"].iloc[1]
    assert ratio == pytest.approx(10, rel=0.15)


# ---------- выборка
def test_without_replacement_has_no_repeats():
    idx = sampling.draw(np.arange(1000), 900, replace=False)
    assert len(np.unique(idx)) == 900


def test_se_formula_full_population_is_zero():
    pop = RNG.normal(size=500)
    assert sampling.se_mean_theory(pop, 500, replace=False) == pytest.approx(0)
    assert sampling.se_mean_theory(pop, 500, replace=True) > 0


def test_se_experiment_matches_theory():
    pop = RNG.normal(10, 2, 2000)
    df = sampling.se_mean_experiment(pop, [200, 1600], reps=800)
    assert np.allclose(df["se_with_emp"], df["se_with_theory"], rtol=0.1)
    assert np.allclose(df["se_without_emp"], df["se_without_theory"], rtol=0.1)


def test_ci_covers_mean_about_95_percent():
    hits = 0
    for s in range(400):
        x = np.random.default_rng(s).normal(3, 1, 50)
        lo, hi = sampling.ci_mean(x)
        hits += lo <= 3 <= hi
    assert 0.92 <= hits / 400 <= 0.98


# ---------- нормализация
def test_zscore_mean_zero_sd_one():
    z = normalize.zscore(RNG.normal(7, 3, 1000))
    assert z.mean() == pytest.approx(0, abs=1e-12)
    assert z.std(ddof=1) == pytest.approx(1)


def test_minmax_range():
    m = normalize.minmax(RNG.normal(size=100))
    assert m.min() == 0 and m.max() == 1


def test_log_z_makes_lognormal_symmetric():
    x = RNG.lognormal(1, 0.8, 5000)
    assert abs(stats.skew(normalize.log_z(x))) < 0.15 < stats.skew(x)


# ---------- распределения
def test_fit_recovers_normal_parameters():
    mu, sigma = distributions.fit("norm", RNG.normal(3.2, 0.15, 20000))
    assert mu == pytest.approx(3.2, abs=0.01)
    assert sigma == pytest.approx(0.15, rel=0.03)


def test_fit_candidates_prefers_true_family():
    x = RNG.gamma(4, 0.6, 5000) + 8
    table = distributions.fit_candidates(x, ["norm", "gamma"])
    assert table.index[0] == "gamma"


def test_chi2_gof_accepts_true_and_rejects_wrong_model():
    x = RNG.normal(0, 1, 3000)
    assert distributions.chi2_gof(x, "norm", (0, 1))["p"] > 0.01
    assert distributions.chi2_gof(RNG.exponential(1, 3000), "norm", (1, 1))["p"] < 1e-6


def test_ks_bootstrap_p_value_reasonable_for_true_model():
    x = RNG.normal(5, 1, 300)
    assert distributions.ks_bootstrap(x, "norm", B=200)["p"] > 0.01


# ---------- гипотезы
def test_normality_tests_detect_skewed_data():
    assert hypotheses.normality_dagostino(RNG.exponential(size=1000))["decision"] == "отвергаем H0"
    assert hypotheses.shapiro(RNG.normal(size=300))["p"] > 0.01


def test_welch_and_mann_whitney_detect_shift():
    a, b = RNG.normal(1, 1, 500), RNG.normal(0, 1, 500)
    assert hypotheses.welch_t(a, b)["p"] < 1e-6
    r = hypotheses.mann_whitney(a, b, alternative="greater")
    assert r["p"] < 1e-6 and r["p_a_gt_b"] > 0.7


def test_holm_correction():
    assert list(hypotheses.holm([0.001, 0.04, 0.03, 0.5])) == [True, False, False, False]
    assert list(hypotheses.holm([0.001, 0.012, 0.03])) == [True, True, True]


# ---------- калибровка критериев согласия (доля ложных отвержений около α = 0.05)
def test_chi2_calibration_on_rounded_normal_data():
    # данные как у pH: N(3.19, 0.15²), округление до 0.01, N = 4898
    rate = distributions.calibration("norm", (3.188, 0.151), n=4898, step=0.01, sims=300, seed=7)["chi2_reject_rate"]
    assert 0.02 <= rate <= 0.09


def test_chi2_calibration_on_rounded_gamma_data():
    # как у спирта: гамма со сдвигом, шаг 0.1
    rate = distributions.calibration("gamma", (4.12, 7.95, 0.623), n=4898, step=0.1, sims=100, seed=8)["chi2_reject_rate"]
    assert rate <= 0.12


def test_ks_bootstrap_calibration_on_rounded_normal_data():
    res = distributions.calibration("norm", (3.188, 0.151), n=1000, step=0.01, sims=40, B=99, seed=9)
    assert res["ks_reject_rate"] <= 0.125


def test_chi2_and_ks_reject_wrong_family():
    x = distributions.round_to_grid(RNG.gamma(2, 1, 3000), 0.01)
    assert distributions.chi2_gof(x, "norm")["p"] < 1e-6
    assert distributions.ks_bootstrap(x, "norm", B=99)["p"] < 0.05


def test_detect_step():
    assert distributions.detect_step(np.round(RNG.normal(3, 0.2, 2000), 2)) == pytest.approx(0.01)


def test_knn_r2_high_for_nonlinear_deterministic_target():
    X = RNG.uniform(-2, 2, size=(2000, 2))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2          # нелинейная формула
    assert stochastic.cv_r2_linear(X, y)["r2_cv"] < 0.5
    assert stochastic.cv_r2_knn(X, y, k_neighbors=5)["r2_cv"] > 0.95


def test_mann_whitney_one_sided_h0_label():
    r = hypotheses.mann_whitney(RNG.normal(size=50), RNG.normal(size=50), alternative="greater")
    assert r["H0"] == "P(A > B) ≤ 1/2"
