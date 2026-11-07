"""Измерение пропорций шрифта прямо по контурам букв (fontTools).

Все длины переводятся в доли кегля: делятся на unitsPerEm (UPM). Кегль, он же em,
в цифровом шрифте равен UPM единиц, поэтому x-height = 0.52 означает: при кегле
20 pt высота строчных 0.52 · 20 = 10.4 pt.

Что измеряется:
    x_height   высота строчной «x» (верх контура), доли кегля
    cap_height высота прописной «H», доли кегля
    stem       толщина основного штриха: ширина вертикали «l» на середине x-height, доли кегля
    stem_n     то же по левой вертикали «n» (контроль: у «l» бывают хвосты и засечки)
    contrast   отношение толстого штриха «o» (бока) к тонкому (верх и низ)
    width_o    ширина «o» по контуру / x-height (узкий или широкий шрифт)
    adv_mean   средняя ширина строчных a–z по advance width, доли кегля

Толщина считается «сканирующей линией»: контур превращается в многоугольник,
ищутся точки пересечения с горизонталью y = const, между парами пересечений
лежат участки «чернил» (правило ненулевой обмотки, как у растеризатора).
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

from fontTools.pens.basePen import BasePen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont, TTLibError

import logging
logging.getLogger("fontTools").setLevel(logging.ERROR)   # предупреждения о датах в старых файлах

FONT_EXT = (".ttf", ".otf")
CYR_TEST = "жЖ"


# ---------------------------------------------------------------- контур -> многоугольники
class PolygonPen(BasePen):
    """Переводит контур в список замкнутых ломаных. Кривые Безье дробятся на `steps` отрезков."""

    def __init__(self, glyphset, steps: int = 24):
        super().__init__(glyphset)
        self.steps = steps
        self.polys: list[list[tuple[float, float]]] = []
        self._cur: list[tuple[float, float]] = []

    def _moveTo(self, p):
        if self._cur:
            self.polys.append(self._cur)
        self._cur = [p]

    def _lineTo(self, p):
        self._cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self._cur[-1]
        for i in range(1, self.steps + 1):
            t = i / self.steps
            mt = 1 - t
            x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
            y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
            self._cur.append((x, y))

    def _qCurveToOne(self, p1, p2):
        p0 = self._cur[-1]
        for i in range(1, self.steps + 1):
            t = i / self.steps
            mt = 1 - t
            x = mt**2 * p0[0] + 2 * mt * t * p1[0] + t**2 * p2[0]
            y = mt**2 * p0[1] + 2 * mt * t * p1[1] + t**2 * p2[1]
            self._cur.append((x, y))

    def _closePath(self):
        if self._cur:
            self.polys.append(self._cur)
        self._cur = []

    _endPath = _closePath


def ink_runs(polys, y: float, axis: str = "h") -> list[tuple[float, float]]:
    """Участки «чернил» на линии y = const (axis='h') или x = const (axis='v').

    Правило ненулевой обмотки (как растеризатор шрифтов): у каждого пересечения есть
    направление ребра (+1 вверх, −1 вниз); точка закрашена, если сумма направлений слева
    от неё не равна нулю. Так перекрывающиеся контуры (обычны в вариативных шрифтах)
    сливаются, а не «выедают» друг друга, как было бы по правилу чёт-нечет.
    Возвращает отрезки (начало, конец) по другой координате, по возрастанию.
    """
    a, b = (0, 1) if axis == "h" else (1, 0)   # a: координата вдоль линии, b: фиксированная
    hits = []
    for poly in polys:
        n = len(poly)
        for i in range(n):
            p, q = poly[i], poly[(i + 1) % n]
            if (p[b] <= y < q[b]) or (q[b] <= y < p[b]):
                t = (y - p[b]) / (q[b] - p[b])
                hits.append((p[a] + t * (q[a] - p[a]), 1 if q[b] > p[b] else -1))
    hits.sort()
    runs, wind, start = [], 0, None
    for x, d in hits:
        before = wind
        wind += d
        if before == 0 and wind != 0:
            start = x
        elif before != 0 and wind == 0:
            runs.append((start, x))
    return runs


# ---------------------------------------------------------------- работа с файлом шрифта
def _name(font: TTFont, *ids: int) -> str:
    for i in ids:
        rec = font["name"].getDebugName(i)
        if rec:
            return rec
    return ""


def is_italic(font: TTFont) -> bool:
    os2 = font["OS/2"] if "OS/2" in font else None
    if os2 is not None and os2.fsSelection & 1:
        return True
    if "post" in font and abs(font["post"].italicAngle) > 0.5:
        return True
    return bool(re.search(r"italic|oblique", _name(font, 17, 2), re.I))


def regular_instance(font: TTFont) -> TTFont:
    """Для вариативного шрифта: экземпляр с весом 400, шириной 100, без наклона."""
    if "fvar" not in font:
        return font
    from fontTools.varLib.instancer import instantiateVariableFont
    loc = {}
    for ax in font["fvar"].axes:
        target = {"wght": 400, "wdth": 100, "ital": 0, "slnt": 0, "opsz": None}.get(ax.axisTag, None)
        if target is None:
            target = ax.defaultValue
        loc[ax.axisTag] = min(max(target, ax.minValue), ax.maxValue)
    return instantiateVariableFont(font, loc, inplace=False)


@dataclass
class FontMeasure:
    family: str
    style: str
    file: str
    category: str
    upm: int
    weight_class: int
    is_variable: bool
    is_mono: bool
    has_cyrillic: bool
    italic_angle: float
    x_height: float
    cap_height: float
    stem: float
    stem_n: float
    contrast: float
    width_o: float
    adv_mean: float
    x_to_cap: float
    os2_x_height: float
    no_lowercase: bool   # строчные заменены капителью/прописными: x-height не настоящий


def _glyph(font: TTFont, ch: str):
    gname = font.getBestCmap().get(ord(ch))
    if gname is None:
        return None, None
    gs = font.getGlyphSet()
    pen = PolygonPen(gs)
    gs[gname].draw(pen)
    bp = BoundsPen(gs)
    gs[gname].draw(bp)
    return pen.polys, bp.bounds


def _stem_at(polys, y):
    runs = ink_runs(polys, y, "h")
    return max((r[1] - r[0] for r in runs), default=math.nan)


def measure_font(font: TTFont, *, file: str = "", category: str = "") -> FontMeasure:
    """Измерить один шрифт (статический или вариативный, тогда берётся вес 400)."""
    variable = "fvar" in font
    f = regular_instance(font)
    upm = f["head"].unitsPerEm
    cmap = f.getBestCmap()
    for ch in "xHlno":
        if ord(ch) not in cmap:
            raise ValueError(f"нет буквы «{ch}»")

    px, bx = _glyph(f, "x")
    ph, bh = _glyph(f, "H")
    pl, bl = _glyph(f, "l")
    pn, bn = _glyph(f, "n")
    po, bo = _glyph(f, "o")
    if None in (bx, bh, bl, bn, bo):
        raise ValueError("пустой контур одной из букв (возможно, шрифт-пиктограммы)")

    xh = bx[3]
    stem = _stem_at(pl, xh * 0.5)
    runs_n = ink_runs(pn, xh * 0.35, "h")
    stem_n = (runs_n[0][1] - runs_n[0][0]) if runs_n else math.nan

    # «o»: толстые бока на середине высоты, тонкие верх/низ на вертикали через центр
    ymid = (bo[1] + bo[3]) / 2
    xmid = (bo[0] + bo[2]) / 2
    side = ink_runs(po, ymid, "h")
    topb = ink_runs(po, xmid, "v")
    if len(side) >= 2 and len(topb) >= 2:
        thick = (side[0][1] - side[0][0] + side[-1][1] - side[-1][0]) / 2
        thin = (topb[0][1] - topb[0][0] + topb[-1][1] - topb[-1][0]) / 2
        contrast = thick / thin if thin > 0 and thick > 0 else math.nan
    else:
        contrast = math.nan

    hmtx = f["hmtx"]
    advs = [hmtx[cmap[ord(c)]][0] for c in "abcdefghijklmnopqrstuvwxyz" if ord(c) in cmap]
    os2 = f["OS/2"] if "OS/2" in f else None
    os2_xh = getattr(os2, "sxHeight", 0) if os2 is not None else 0

    return FontMeasure(
        family=_name(font, 16, 1),
        style="wght 400 (вариативный)" if variable else _name(font, 17, 2),
        file=file,
        category=category,
        upm=upm,
        weight_class=int(os2.usWeightClass) if os2 is not None else 0,
        is_variable=variable,
        is_mono=bool(f["post"].isFixedPitch) if "post" in f else False,
        has_cyrillic=all(ord(c) in cmap for c in CYR_TEST),
        italic_angle=float(f["post"].italicAngle) if "post" in f else 0.0,
        x_height=xh / upm,
        cap_height=bh[3] / upm,
        stem=stem / upm,
        stem_n=stem_n / upm,
        contrast=contrast,
        width_o=(bo[2] - bo[0]) / xh,
        adv_mean=sum(advs) / len(advs) / upm,
        x_to_cap=xh / bh[3],
        os2_x_height=(os2_xh or 0) / upm,
        no_lowercase=bool(xh / bh[3] > 0.9 or re.search(r"\bSC$|Small ?Caps|Unicase", _name(font, 16, 1) or _name(font, 1))),
    )


# ---------------------------------------------------------------- обход папки / архивов
def iter_font_files(root: Path):
    """Все .ttf/.otf в папке и во вложенных zip. Отдаёт (имя, байты, категория, папка семейства).

    Категория = имя папки первого уровня внутри root (sans, serif, display, ...),
    если шрифты разложены по папкам; иначе пустая строка.
    """
    root = Path(root)
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        cat = rel.parts[0].lower() if len(rel.parts) > 1 else ""
        if p.suffix.lower() in FONT_EXT:
            # category/Family_Folder/[static/]file.ttf -> папка семейства из Google Fonts
            fam_dir = rel.parts[1] if len(rel.parts) > 2 else ""
            yield str(rel), p.read_bytes(), cat, fam_dir
        elif p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for info in z.infolist():
                    if info.filename.lower().endswith(FONT_EXT) and "__MACOSX" not in info.filename:
                        parts = info.filename.split("/")
                        fam_dir = parts[0] if len(parts) > 1 and parts[0] != "static" else p.stem
                        yield f"{rel}/{info.filename}", z.read(info), cat, fam_dir


def _rank(font: TTFont) -> tuple:
    """Насколько файл похож на «Regular»: сначала прямой, потом вариативный, потом вес ближе к 400."""
    w = font["OS/2"].usWeightClass if "OS/2" in font else 400
    return (is_italic(font), 0 if "fvar" in font else 1, abs(w - 400))


def measure_collection(root: Path, *, one_per_family: bool = True, verbose: bool = False):
    """Измерить коллекцию. Возвращает (DataFrame измерений, список пропущенных файлов с причиной).

    one_per_family=True: по одному начертанию на семейство (ближайшее к Regular).
    Начертания одного семейства рисовал один дизайнер, они зависимы; для статистики
    наблюдением считается семейство.
    """
    import pandas as pd

    best: dict[str, tuple] = {}
    skipped = []
    for name, raw, cat, fam_dir in iter_font_files(root):
        try:
            font = TTFont(io.BytesIO(raw), lazy=False)
        except (TTLibError, Exception) as e:   # noqa: BLE001  повреждённый файл
            skipped.append((name, f"не читается: {e}"))
            continue
        # Папка семейства надёжнее имени в файле: у статических файлов оптических размеров
        # (Inter 18pt, Fraunces 72pt) и ширин (Roboto SemiCondensed) своё имя семейства
        fam = fam_dir.replace("_", " ") if fam_dir else (_name(font, 16, 1) or Path(name).stem)
        key = fam if one_per_family else f"{fam}|{_name(font, 17, 2)}|{name}"
        r = _rank(font)
        if key not in best or r < best[key][0]:
            best[key] = (r, name, raw, cat)

    rows = []
    for key, (_, name, raw, cat) in sorted(best.items()):
        try:
            m = measure_font(TTFont(io.BytesIO(raw)), file=name, category=cat)
            if one_per_family:
                m.family = key
            rows.append(asdict(m))
        except Exception as e:   # noqa: BLE001
            skipped.append((name, str(e)))
            if verbose:
                print("пропуск", name, e)
    return pd.DataFrame(rows), skipped
