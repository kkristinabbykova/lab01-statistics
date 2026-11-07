"""Графики в одном стиле. Каждый график сохраняется в figures/ и возвращает fig."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # синий, оранжевый, бирюзовый
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "lines.linewidth": 2,
    })


def grid_bins(x, target: int = 45):
    """Интервалы гистограммы, кратные шагу округления данных.

    Значения в датасете записаны с шагом (pH 0.01, спирт 0.1). Если ширина интервала
    не кратна шагу, в одни интервалы попадает больше «узлов сетки», чем в другие,
    и на гистограмме появляются ложные пики.
    """
    x = np.asarray(x, dtype=float)
    vals = np.unique(np.round(x, 8))
    d = np.round(np.diff(vals), 6)
    d = d[d > 0]
    if d.size == 0:
        return 10
    u, c = np.unique(d, return_counts=True)
    step = u[c.argmax()]
    width = step * max(1, int(round((x.max() - x.min()) / step / target)))
    return np.arange(x.min() - step / 2, x.max() + width, width)


def save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    return fig


def eda_panel(x, label: str, unit: str, fits: dict, fname: str):
    """Гистограмма плотности + подобранные плотности, ящик с усами, Q-Q к нормальному."""
    style()
    x = np.asarray(x, dtype=float)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [2.2, 1, 1.4]})
    ax[0].hist(x, bins=grid_bins(x), density=True, color=C1, alpha=0.35, edgecolor="white", linewidth=0.5, label="данные")
    grid = np.linspace(x.min(), x.max(), 400)
    for (name, params), col in zip(fits.items(), [C1, C2, C3]):
        ax[0].plot(grid, getattr(stats, name).pdf(grid, *params), color=col, label=name)
    ax[0].set(xlabel=f"{label}, {unit}", ylabel="плотность", title=f"{label}: гистограмма и модели")
    ax[0].legend()
    ax[1].boxplot(x, vert=True, widths=0.5, patch_artist=True,
                  boxprops={"facecolor": "#d6e6f8", "edgecolor": C1}, medianprops={"color": C2},
                  flierprops={"marker": "o", "markersize": 3, "markerfacecolor": C1, "markeredgecolor": "none", "alpha": .5})
    ax[1].set(ylabel=f"{label}, {unit}", xticks=[], title="ящик с усами")
    (osm, osr), (slope, inter, _) = stats.probplot(x, dist="norm")
    ax[2].plot(osm, osr, "o", ms=2.5, color=C1, alpha=0.5)
    ax[2].plot(osm, slope * osm + inter, color=C2, linewidth=1.5)
    ax[2].set(xlabel="квантили N(0, 1)", ylabel=f"квантили данных, {unit}", title="Q-Q к нормальному")
    fig.tight_layout()
    return save(fig, fname)


def consistency_panel(curves: dict, running: dict, label: str, fname: str):
    """Слева: одна траектория оценок при росте n. Справа: P(|оценка - θ| > ε) от n."""
    style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for (est, (ns, vals, theta)), col in zip(running.items(), [C1, C2, C3]):
        ax[0].plot(ns, vals / theta, color=col, label=est, linewidth=1.5)
    ax[0].axhline(1, color=INK2, linewidth=1, linestyle="--")
    ax[0].set(xscale="log", xlabel="объём выборки n", ylabel="оценка / значение в совокупности",
              title=f"{label}: оценка при росте n (одна траектория)")
    ax[0].set_ylim(0.5, 1.5)
    ax[0].legend()
    for (est, df), col in zip(curves.items(), [C1, C2, C3]):
        ax[1].plot(df["n"], df["p_far"], "o-", color=col, ms=5, label=f"{est}, ε={df['eps'].iloc[0]:.3g}")
    ax[1].set(xscale="log", xlabel="объём выборки n", ylabel="P(|оценка − θ| > ε)",
              title="доля «далёких» оценок, 1000 повторов", ylim=(-0.02, 1.02))
    ax[1].legend()
    fig.tight_layout()
    return save(fig, fname)


def sampling_panel(pop, s_with, s_without, label: str, unit: str, fname: str):
    """Эмпирические функции распределения совокупности и двух выборок."""
    style()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for data, col, name in [(pop, INK2, f"совокупность N={len(pop)}"),
                            (s_with, C1, f"с возвращением n={len(s_with)}"),
                            (s_without, C2, f"без возвращения n={len(s_without)}")]:
        xs = np.sort(data)
        ax.step(xs, np.arange(1, xs.size + 1) / xs.size, where="post", color=col, label=name,
                linewidth=2 if col == INK2 else 1.5)
    ax.set(xlabel=f"{label}, {unit}", ylabel="F(x)", title=f"{label}: ЭФР совокупности и выборок")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, fname)


def se_panel(tables: dict, fname: str):
    """Отношение эмпирической СКО среднего к σ/√n для двух способов отбора."""
    style()
    fig, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 3.4), sharey=True)
    for ax, (label, df) in zip(np.atleast_1d(axes), tables.items()):
        base = df["se_with_theory"]
        ax.plot(df["n/N"], df["se_with_emp"] / base, "o", color=C1, ms=6, label="с возвр., опыт")
        ax.plot(df["n/N"], df["se_without_emp"] / base, "s", color=C2, ms=6, label="без возвр., опыт")
        grid = np.linspace(0, df["n/N"].max(), 100)
        N = df["n"].iloc[0] / df["n/N"].iloc[0]
        ax.plot(grid, np.sqrt((N - grid * N) / (N - 1)), color=C2, linewidth=1.2, label="√((N−n)/(N−1))")
        ax.axhline(1, color=C1, linewidth=1.2)
        ax.set(xlabel="доля отбора n/N", title=label)
    np.atleast_1d(axes)[0].set_ylabel("SE / (σ/√n)")
    np.atleast_1d(axes)[0].legend(loc="lower left")
    fig.tight_layout()
    return save(fig, fname)


def normalization_panel(pairs: dict, fname: str):
    """Гистограммы до и после преобразования, поверх них плотность N(0, 1)."""
    style()
    fig, ax = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.4))
    grid = np.linspace(-4, 4, 300)
    for a, (title, z) in zip(np.atleast_1d(ax), pairs.items()):
        a.hist(z, bins=grid_bins(z), density=True, color=C1, alpha=0.35, edgecolor="white", linewidth=0.5)
        a.plot(grid, stats.norm.pdf(grid), color=C2, label="N(0, 1)")
        a.set(title=title, xlabel="z, безразмерная", xlim=(-4.5, 4.5))
        a.text(0.02, 0.95, f"асимм. {stats.skew(z):.2f}\nэксцесс {stats.kurtosis(z):.2f}",
               transform=a.transAxes, va="top", fontsize=9, color=INK2)
    np.atleast_1d(ax)[0].set_ylabel("плотность")
    np.atleast_1d(ax)[0].legend(loc="upper right")
    fig.tight_layout()
    return save(fig, fname)


def scatter_dependence(x, y, xl: str, yl: str, title: str, fname: str):
    style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    hb = ax.hexbin(x, y, gridsize=35, cmap="Blues", mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=ax, label="число вин в ячейке")
    # медиана y в децилях x показывает форму связи
    edges = np.quantile(x, np.linspace(0, 1, 11))
    mid = 0.5 * (edges[1:] + edges[:-1])
    med = [np.median(y[(x >= lo) & (x <= hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
    ax.plot(mid, med, "o-", color=C2, ms=5, label="медиана по децилям")
    ax.set(xlabel=xl, ylabel=yl, title=title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save(fig, fname)


def corr_heatmap(corr, fname: str):
    style()
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, label="ρ Спирмена")
    ax.set_title("Связи между всеми измерениями (белые вина)")
    fig.tight_layout()
    return save(fig, fname)


def two_groups(a, b, la: str, lb: str, xlabel: str, title: str, fname: str):
    style()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.histogram_bin_edges(np.concatenate([a, b]), bins=45)
    ax.hist(a, bins=bins, density=True, color=C1, alpha=0.45, label=la, edgecolor="white", linewidth=0.4)
    ax.hist(b, bins=bins, density=True, color=C2, alpha=0.45, label=lb, edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(a), color=C1, linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(b), color=C2, linestyle="--", linewidth=1.5)
    ax.set(xlabel=xlabel, ylabel="плотность", title=title)
    ax.legend()
    fig.tight_layout()
    return save(fig, fname)


def stochastic_panel(y, resid, label: str, unit: str, step: float, fname: str):
    """Остатки лучшей линейной модели по остальным 10 измерениям против шага округления прибора."""
    style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    pred = y - resid
    ax[0].scatter(pred, y, s=4, color=C1, alpha=0.3)
    # оси по 0.1–99.9 % квантилям, чтобы единичные выбросы не сжимали картинку
    lo, hi = np.quantile(np.concatenate([y, pred]), [0.001, 0.999])
    ax[0].plot([lo, hi], [lo, hi], color=C2, linewidth=1.2, label="y = прогноз")
    ax[0].set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"прогноз по 10 другим измерениям, {unit}",
              ylabel=f"измерено, {unit}", title=f"{label}: прогноз против факта")
    ax[0].legend()
    r_lo, r_hi = np.quantile(resid, [0.001, 0.999])
    ax[1].hist(resid[(resid >= r_lo) & (resid <= r_hi)], bins=60, color=C1, alpha=0.5, edgecolor="white", linewidth=0.4)
    ax[1].axvspan(-step / 2, step / 2, color=C2, alpha=0.6, label=f"± половина шага прибора ({step:g})")
    ax[1].set(xlabel=f"остаток, {unit}", ylabel="число вин", title="остатки модели (без 0.2 % крайних)")
    ax[1].legend()
    fig.tight_layout()
    return save(fig, fname)
