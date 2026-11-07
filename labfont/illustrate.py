"""Схема измерений: буквы H x l n o по настоящим контурам шрифта и линии, по которым мерили."""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from fontTools.ttLib import TTFont

from .measure import PolygonPen, ink_runs, measure_font, regular_instance

C1, C2, C3, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def _outline(f, ch):
    gs = f.getGlyphSet()
    name = f.getBestCmap()[ord(ch)]
    pen = PolygonPen(gs)
    gs[name].draw(pen)
    return pen.polys, f["hmtx"][name][0]


def measurement_figure(font_bytes: bytes, title: str | None = None, ax=None):
    """Нарисовать «Hxlno» и подписать x-height, cap height, толщину штриха и контраст."""
    font = TTFont(io.BytesIO(font_bytes))
    m = measure_font(font)
    f = regular_instance(font)
    upm = f["head"].unitsPerEm
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(10, 4.2))
    x0, pos = 0.0, {}
    for ch in "Hxlno":
        polys, adv = _outline(f, ch)
        pos[ch] = (x0, polys)
        # правило чёт-нечет: рисуем контуры как один путь
        from matplotlib.path import Path as MPath
        from matplotlib.patches import PathPatch
        verts, codes = [], []
        for p in polys:
            pts = [((x + x0) / upm, y / upm) for x, y in p]
            verts += pts + [pts[0]]
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1) + [MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor="#d6e6f8", edgecolor=C1, linewidth=1))
        x0 += adv
    right = x0 / upm
    for y, lab, col in ((0, "базовая линия", INK2), (m.x_height, f"x-height = {m.x_height:.3f}", C2),
                        (m.cap_height, f"cap height = {m.cap_height:.3f}", C3)):
        ax.hlines(y, -0.03, right + 0.02, color=col, linewidth=1, linestyle="--" if y else "-")
        ax.text(right + 0.05, y, lab, va="center", fontsize=9, color=col, bbox={"fc": "white", "ec": "none", "pad": 1})
    # толщина штриха «l» на середине x-height
    lx0, lpolys = pos["l"]
    y = m.x_height * upm * 0.5
    run = max(ink_runs(lpolys, y), key=lambda r: r[1] - r[0])
    ax.annotate("", xy=((run[0] + lx0) / upm, y / upm), xytext=((run[1] + lx0) / upm, y / upm),
                arrowprops={"arrowstyle": "<->", "color": C2, "lw": 1.6})
    ax.text((run[1] + lx0) / upm + 0.02, y / upm, f"штрих {m.stem:.3f}", fontsize=9, color=C2, va="center", bbox={"fc": "white", "ec": "none", "pad": 1})
    # контраст «o»: толстый бок и тонкий верх
    ox0, opolys = pos["o"]
    xs = [x for p in opolys for x, _ in p]
    ys = [yy for p in opolys for _, yy in p]
    ymid, xmid = (min(ys) + max(ys)) / 2, (min(xs) + max(xs)) / 2
    side = ink_runs(opolys, ymid)[0]
    top = ink_runs(opolys, xmid, "v")[-1]
    ax.annotate("", xy=((side[0] + ox0) / upm, ymid / upm), xytext=((side[1] + ox0) / upm, ymid / upm),
                arrowprops={"arrowstyle": "<->", "color": C3, "lw": 1.6})
    ax.annotate("", xy=((xmid + ox0) / upm, top[0] / upm), xytext=((xmid + ox0) / upm, top[1] / upm),
                arrowprops={"arrowstyle": "<->", "color": C3, "lw": 1.6})
    ax.text((max(xs) + ox0) / upm + 0.02, ymid / upm, f"контраст\n{m.contrast:.2f}", fontsize=9, color=C3, va="center")
    ax.set_xlim(-0.05, right + 0.45)
    ax.set_ylim(-0.12, max(m.cap_height, 0.8) + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"{m.family}: что измеряется (доли кегля, кегль = 1)", fontsize=11)
    return (fig if own else ax), m
