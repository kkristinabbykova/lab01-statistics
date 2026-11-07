"""Графики для датасета шрифтов в том же стиле, что labstat.plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from labstat.plots import C1, C2, C3, INK2, save, style

CAT_COLORS = {"sans": C1, "serif": C2, "display": C3, "handwriting": "#9b59b6", "mono": INK2, "": "#999999"}
CAT_RU = {"sans": "гротески (sans)", "serif": "антиквы (serif)", "display": "акцидентные (display)",
          "handwriting": "рукописные", "mono": "моноширинные", "": "без категории"}
COL_RU = {"x_height": "x-height", "cap_height": "высота прописных", "stem": "толщина штриха «l»",
          "stem_n": "толщина штриха «n»", "contrast": "контраст «o»", "width_o": "ширина «o» / x-height",
          "adv_mean": "средняя ширина a–z", "x_to_cap": "x-height / cap height", "weight_class": "вес OS/2"}


def scatter_by_category(df, x: str, y: str, xl: str, yl: str, title: str, fname: str):
    style()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for cat, g in df.groupby("category"):
        ax.scatter(g[x], g[y], s=22, alpha=0.75, color=CAT_COLORS.get(cat, "#999"),
                   label=f"{CAT_RU.get(cat, cat)} ({len(g)})", edgecolor="white", linewidth=0.4)
    # медиана y по квинтилям x показывает форму связи
    edges = np.quantile(df[x], np.linspace(0, 1, 6))
    mid = 0.5 * (edges[1:] + edges[:-1])
    med = [df[y][(df[x] >= lo) & (df[x] <= hi)].median() for lo, hi in zip(edges[:-1], edges[1:])]
    ax.plot(mid, med, "o-", color="#0b0b0b", ms=4, linewidth=1.4, label="медиана по квинтилям")
    ax.set(xlabel=xl, ylabel=yl, title=title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return save(fig, fname)


def box_by_category(df, col: str, label: str, unit: str, fname: str):
    style()
    cats = [c for c in CAT_RU if c in set(df.category)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    data = [df.loc[df.category == c, col].to_numpy() for c in cats]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": "#0b0b0b"},
                    flierprops={"marker": "o", "markersize": 3, "alpha": .5})
    for patch, c in zip(bp["boxes"], cats):
        patch.set_facecolor(CAT_COLORS[c])
        patch.set_alpha(0.45)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, 1):
        ax.plot(i + rng.uniform(-0.15, 0.15, d.size), d, ".", ms=3, color=INK2, alpha=0.5)
    ax.set_xticks(range(1, len(cats) + 1), [f"{CAT_RU[c]}\nn = {len(d)}" for c, d in zip(cats, data)], fontsize=8)
    ax.set(ylabel=f"{label}, {unit}", title=f"{label} по категориям шрифтов")
    fig.tight_layout()
    return save(fig, fname)
