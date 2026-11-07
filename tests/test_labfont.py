"""Тесты измерителя шрифтов на синтетическом шрифте с заранее известной геометрией.

Шрифт строится fontTools.FontBuilder: UPM = 1000, буквы из прямоугольников.
«l» = вертикаль шириной 100, «x» = прямоугольник высотой 500, «H» высотой 700,
«o» = прямоугольное кольцо: бока толщиной 120, верх и низ толщиной 60 (контраст 2).
"""
import io
import zipfile

import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from labfont.measure import ink_runs, measure_collection, measure_font

UPM = 1000


def rect(pen, x0, y0, x1, y1, clockwise=True):
    pts = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)] if clockwise else [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    pen.moveTo(pts[0])
    for p in pts[1:]:
        pen.lineTo(p)
    pen.closePath()


def make_font(stem=100, xh=500, cap=700, o_side=120, o_top=60, family="Test Sans", weight=400, italic=False):
    letters = "abcdefghijklmnopqrstuvwxyz"
    glyph_order = [".notdef", "H"] + list(letters)
    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap({ord(c): c for c in glyph_order[1:]})
    glyphs = {}
    for g in glyph_order:
        pen = TTGlyphPen(None)
        if g == "l":
            rect(pen, 50, 0, 50 + stem, 750)
        elif g == "H":
            rect(pen, 50, 0, 150, cap)
            rect(pen, 450, 0, 550, cap)
            rect(pen, 150, cap / 2 - 40, 450, cap / 2 + 40)
        elif g == "o":
            rect(pen, 50, 0, 450, xh)                                   # внешний
            rect(pen, 50 + o_side, o_top, 450 - o_side, xh - o_top, clockwise=False)  # внутренний
        elif g == "n":
            rect(pen, 50, 0, 50 + stem, xh)
            rect(pen, 350, 0, 350 + stem, xh)
        elif g == ".notdef":
            rect(pen, 0, 0, 10, 10)
        else:
            rect(pen, 50, 0, 450, xh)
        glyphs[g] = pen.glyph()
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics({g: (500, 50) for g in glyph_order})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": family, "styleName": "Italic" if italic else "Regular"})
    fb.setupOS2(sxHeight=xh, sCapHeight=cap, usWeightClass=weight, fsSelection=1 if italic else 0x40)
    fb.setupPost(italicAngle=-10 if italic else 0)
    buf = io.BytesIO()
    fb.save(buf)
    return buf.getvalue()


def test_ink_runs_square_ring():
    outer = [(0, 0), (0, 10), (10, 10), (10, 0)]
    inner = [(3, 2), (7, 2), (7, 8), (3, 8)]
    assert ink_runs([outer, inner], 5, "h") == [(0, 3), (7, 10)]
    assert ink_runs([outer, inner], 5, "v") == [(0, 2), (8, 10)]


def test_measure_known_geometry():
    m = measure_font(TTFont(io.BytesIO(make_font())))
    assert m.x_height == pytest.approx(0.5)
    assert m.cap_height == pytest.approx(0.7)
    assert m.stem == pytest.approx(0.1)
    assert m.stem_n == pytest.approx(0.1)
    assert m.contrast == pytest.approx(2.0)
    assert m.x_to_cap == pytest.approx(5 / 7)
    assert m.width_o == pytest.approx(400 / 500)
    assert m.adv_mean == pytest.approx(0.5)


@pytest.mark.parametrize("stem", [40, 100, 180])
def test_stem_scales(stem):
    m = measure_font(TTFont(io.BytesIO(make_font(stem=stem))))
    assert m.stem == pytest.approx(stem / UPM)


def test_one_per_family_prefers_regular(tmp_path):
    """В архиве семейства лежат Bold, Italic и Regular: берётся прямой вес 400."""
    with zipfile.ZipFile(tmp_path / "Test_Sans.zip", "w") as z:
        z.writestr("static/TestSans-Bold.ttf", make_font(stem=180, weight=700))
        z.writestr("static/TestSans-Italic.ttf", make_font(stem=95, italic=True))
        z.writestr("static/TestSans-Regular.ttf", make_font(stem=100))
    df, skipped = measure_collection(tmp_path)
    assert len(df) == 1 and not skipped
    assert df.stem.iloc[0] == pytest.approx(0.1)


def test_category_from_folder(tmp_path):
    (tmp_path / "serif").mkdir()
    (tmp_path / "sans").mkdir()
    (tmp_path / "serif" / "A.ttf").write_bytes(make_font(family="Alpha"))
    (tmp_path / "sans" / "B.ttf").write_bytes(make_font(family="Beta"))
    df, _ = measure_collection(tmp_path)
    assert dict(zip(df.family, df.category)) == {"Alpha": "serif", "Beta": "sans"}


def test_font_without_latin_is_skipped(tmp_path):
    (tmp_path / "bad.ttf").write_bytes(b"not a font")
    (tmp_path / "ok.ttf").write_bytes(make_font())
    df, skipped = measure_collection(tmp_path)
    assert len(df) == 1 and len(skipped) == 1
