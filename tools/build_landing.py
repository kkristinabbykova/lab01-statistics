"""Собрать лендинг docs/index.html из данных работы (GitHub Pages публикует папку docs/).

    python tools/build_landing.py            # или через tools/build_site.py вместе с экспортом marimo

Страница самодостаточная: данные (квантили pH, измерения шрифтов) встраиваются как JSON,
живой датчик pH работает на чистом JavaScript. Если рядом есть docs/datchik/ (экспорт
marimo), страница ссылается на него.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPO = os.environ.get("LAB_REPO_URL", "").rstrip("/")   # например https://github.com/<логин>/lab01


def wine_block():
    w = pd.read_csv(ROOT / "data" / "winequality-white.csv", sep=";")
    x = np.sort(w["pH"].to_numpy())
    p = (np.arange(1, x.size + 1) - 0.5) / x.size
    q = np.interp(np.linspace(0, 1, 201), p, x)
    edges = np.arange(2.705, 3.845, 0.02)
    h, _ = np.histogram(x, bins=edges, density=True)
    rows = []
    for col, nm, unit in (("pH", "pH", "ед. pH"), ("fixed acidity", "фиксированная кислотность", "г/дм³"),
                          ("alcohol", "спирт", "% об.")):
        v = w[col]
        rows.append({"name": nm, "unit": unit, "mean": round(v.mean(), 3), "sd": round(v.std(), 3),
                     "skew": round(float(stats.skew(v)), 2)})
    return {"q": [round(float(v), 4) for v in q], "edges": [round(float(e), 3) for e in edges],
            "hist": [round(float(v), 4) for v in h], "n": int(x.size), "rows": rows}


def fonts_block():
    f = pd.read_csv(ROOT / "data" / "fonts.csv")
    f["category"] = f["category"].fillna("")
    f = f[~f.no_lowercase]
    pts = [{"f": r.family, "c": r.category, "x": round(r.x_height, 4), "s": round(r.stem, 4)} for r in f.itertuples()]
    med = f.groupby("category").x_height.median().round(3).to_dict()
    rho = stats.spearmanr(f.x_height, f.stem)[0]
    stems = sorted(round(float(v), 4) for v in f.stem)
    return {"pts": pts, "med": med, "rho": round(float(rho), 2), "n": len(f), "stems": stems}


def link(path, text):
    return f'<a href="{REPO}/blob/main/{path}">{text}</a>' if REPO else f"<code>{path}</code>"


def main():
    DOCS.mkdir(exist_ok=True)
    (DOCS / "img").mkdir(exist_ok=True)
    for name in ("rv01_rv02_dependence.png", "fonts_measure.png", "rv01_eda.png"):
        if (ROOT / "figures" / name).exists():
            shutil.copy(ROOT / "figures" / name, DOCS / "img" / name)
    data = {"wine": wine_block(), "fonts": fonts_block()}
    has_marimo = (DOCS / "datchik" / "index.html").exists()
    marimo = ('<a class="button" href="datchik/">Открыть полный датчик</a>' if has_marimo else
              '<span class="note">Полная версия на marimo появится здесь после команды '
              '<code>python tools/build_site.py</code>.</span>')
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)) \
        .replace("__MARIMO__", marimo) \
        .replace("__L_NB__", link("lab01.ipynb", "lab01.ipynb")) \
        .replace("__L_FONTS__", link("ktvd/fonts.ipynb", "ktvd/fonts.ipynb")) \
        .replace("__L_DATCHIK__", link("ktvd/datchik.py", "ktvd/datchik.py")) \
        .replace("__N_FONTS__", str(data["fonts"]["n"])) \
        .replace("__RHO__", f"{data['fonts']['rho']:.2f}".replace(".", ",")) \
        .replace("__XH_SANS__", f"{data['fonts']['med'].get('sans', float('nan')):.3f}".replace(".", ",")) \
        .replace("__XH_SERIF__", f"{data['fonts']['med'].get('serif', float('nan')):.3f}".replace(".", ","))
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")
    print("docs/index.html собран", "(есть marimo-датчик)" if has_marimo else "(без marimo-экспорта)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lab01 · случайность в бокале и в букве</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Golos+Text:wght@400..900&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#f6f7f5; --ink:#1a1d24; --muted:#5c6170; --rule:#dcdfd8;
  --blue:#2a78d6; --orange:#eb6834; --straw:#d8c56a; --green:#1baf7a;
  --measure:68ch;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);
  font:400 1.0625rem/1.6 "Golos Text",system-ui,-apple-system,"Segoe UI",sans-serif;}
a{color:var(--blue);text-underline-offset:.18em}
a:focus-visible,button:focus-visible,select:focus-visible,input:focus-visible,circle:focus-visible{outline:3px solid var(--orange);outline-offset:3px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em;background:#e9ebe6;padding:.08em .35em;border-radius:3px}
.wrap{max-width:1120px;margin:0 auto;padding:0 clamp(1.1rem,4vw,2.5rem)}
.text{max-width:var(--measure)}

/* --- hero: толщина каждой буквы — случайная величина --- */
header{padding:clamp(3rem,9vw,7rem) 0 2.5rem}
h1{font-size:clamp(2.9rem,9.2vw,7.4rem);line-height:.98;letter-spacing:-.02em;margin:0 0 1.6rem;font-weight:400}
h1 .w{display:block}
h1 span.l{transition:font-weight .5s cubic-bezier(.2,.7,.2,1)}
.lede{font-size:clamp(1.1rem,1.6vw,1.3rem);color:var(--muted);margin:0 0 1.4rem}
.strip{height:10px;border-radius:5px;margin:2.6rem 0 0;
  background:linear-gradient(90deg,#c0392b 0%,var(--orange) 22%,var(--straw) 45%,var(--green) 66%,var(--blue) 100%)}
.strip-caption{display:flex;justify-content:space-between;color:var(--muted);font-size:.85rem;margin-top:.4rem}
button,.button{font:inherit;font-weight:600;font-size:.98rem;background:var(--ink);color:var(--paper);
  border:0;border-radius:999px;padding:.62rem 1.15rem;cursor:pointer;text-decoration:none;display:inline-block}
button:hover,.button:hover{background:#353a46}
.note{color:var(--muted);font-size:.95rem}

/* --- секции --- */
section{padding:3.5rem 0;border-top:1px solid var(--rule)}
h2{font-size:clamp(1.7rem,3.4vw,2.6rem);line-height:1.1;font-weight:750;letter-spacing:-.01em;margin:0 0 1rem}
h3{font-size:1.15rem;font-weight:650;margin:2rem 0 .5rem}
p{margin:0 0 1rem}
figure{margin:1.8rem 0}
figure img{width:100%;max-width:760px;height:auto;background:#fff;border-radius:6px}
figure.wide img{max-width:100%}
figcaption{color:var(--muted);font-size:.92rem;margin-top:.5rem;max-width:var(--measure)}

.scroll{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;max-width:820px;font-variant-numeric:tabular-nums;margin:1.2rem 0}
th,td{text-align:left;padding:.55rem .8rem .55rem 0;border-bottom:1px solid var(--rule)}
th{font-weight:600;color:var(--muted);font-size:.92rem}
td.num{text-align:right}

/* --- живой датчик --- */
.lab{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(0,1fr);gap:1.6rem;align-items:start;margin-top:1.4rem}
.controls{display:flex;flex-wrap:wrap;gap:1rem 1.4rem;align-items:end;margin:1.2rem 0 .4rem}
label{display:flex;flex-direction:column;gap:.3rem;font-size:.9rem;color:var(--muted)}
select,input[type=range]{font:inherit;color:var(--ink)}
select{padding:.4rem .5rem;border:1px solid #c9cdc5;border-radius:6px;background:#fff}
.chart{background:#fff;border-radius:8px;padding:.6rem}
.chart svg,.chart canvas{display:block;width:100%;height:auto}
.readout{font-variant-numeric:tabular-nums;margin-top:.7rem;font-size:.95rem}
.readout b{font-weight:650}
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1rem;font-size:.88rem;color:var(--muted);margin:.5rem 0 0}
.sw{display:inline-block;width:.8em;height:.8em;border-radius:2px;margin-right:.35em;vertical-align:-.05em}

/* --- шрифты --- */
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:1.4rem 2rem;margin:1.6rem 0;max-width:900px}
.fact strong{display:block;font-size:2.4rem;line-height:1.05;font-weight:800;font-variant-numeric:tabular-nums}
.fact span{color:var(--muted);font-size:.95rem}
#tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--paper);font-size:.85rem;
  padding:.3rem .55rem;border-radius:4px;opacity:0;transition:opacity .12s;white-space:nowrap}
footer{padding:3rem 0 4rem;border-top:1px solid var(--rule);color:var(--muted);font-size:.95rem}
footer p{max-width:var(--measure)}

@media (max-width:760px){.lab{grid-template-columns:1fr} th,td{padding-right:.6rem;font-size:.95rem}}
@media (prefers-reduced-motion:reduce){h1 span.l{transition:none}}
</style>
</head>
<body>
<header class="wrap">
  <h1 id="title" aria-label="Случайность в бокале и в букве"><span class="w">Случайность</span><span class="w">в бокале</span><span class="w">и в букве</span></h1>
  <p class="lede text">Лабораторная работа по прикладной статистике: три величины из химии белых вин,
  две величины из пропорций шрифтов и датчик случайных чисел, который их воспроизводит.
  Толщина каждой буквы в заголовке — случайная величина: она взята из измеренной толщины штриха шрифтов из нашего датасета.</p>
  <button id="reroll" type="button">Перебросить толщину букв</button>
  <div class="strip" role="img" aria-label="Шкала pH от кислого к щелочному"></div>
  <div class="strip-caption"><span>кислее</span><span>pH белого вина: от 2,72 до 3,82</span><span>щелочнее</span></div>
</header>

<main>
<section class="wrap" id="wine">
  <h2>Вино: три измерения, ни одно не вычисляется формулой</h2>
  <div class="text">
  <p>4898 образцов белого вина Vinho Verde (Cortez и др., 2009). У каждого лаборатория измерила 11 показателей.
  Мы взяли три: pH, фиксированную кислотность и содержание спирта. Даже по десяти остальным измерениям их нельзя
  предсказать точно: остаток модели во много раз больше шага прибора. Значит, это случайные величины.</p>
  </div>
  <div class="scroll" role="region" aria-label="Таблица величин" tabindex="0"><table>
    <thead><tr><th>Величина</th><th class="num">Среднее</th><th class="num">СКО</th><th class="num">Асимметрия</th><th>Единицы</th></tr></thead>
    <tbody id="wine-rows"></tbody>
  </table></div>
  <figure>
    <img src="img/rv01_rv02_dependence.png" alt="Тепловая карта: чем выше фиксированная кислотность, тем ниже pH" loading="lazy">
    <figcaption>pH и фиксированная кислотность одной природы, и они зависимы: ρ Спирмена −0,42, χ² и перестановочный тест отвергают независимость.</figcaption>
  </figure>
  <p class="note">Полное исследование: __L_NB__ (автор раздела — Нефедьева Антонина, трек «Инфохимия»).</p>
</section>

<section class="wrap" id="rng">
  <h2>Датчик: случайное число, похожее на pH</h2>
  <div class="text">
  <p>Датчик берёт равномерное число <i>U</i> от 0 до 1 и превращает его в pH через обратную функцию распределения:
  <i>X</i> = <i>F</i><sup>−1</sup>(<i>U</i>). Здесь <i>F</i> — эмпирическая функция распределения 4898 вин.
  Если базовый датчик хороший, гистограмма сгенерированных чисел ложится на гистограмму данных.</p>
  </div>
  <div class="controls">
    <label>Базовый датчик
      <select id="base">
        <option value="math">Math.random (xorshift128+)</option>
        <option value="mmix">ЛКГ, константы MMIX (Кнут)</option>
        <option value="randu">RANDU, IBM 1960-х</option>
      </select>
    </label>
    <label>Сколько чисел: <output id="nout">4900</output>
      <input id="n" type="range" min="100" max="20000" step="100" value="4900">
    </label>
    <button id="go" type="button">Сгенерировать заново</button>
  </div>
  <div class="lab">
    <div class="chart">
      <svg id="hist" viewBox="0 0 640 300" role="img" aria-label="Гистограмма pH: данные и датчик"></svg>
      <div class="legend"><span><i class="sw" style="background:#bcd6f3"></i>данные, 4898 вин</span><span><i class="sw" style="background:var(--orange)"></i>датчик</span></div>
    </div>
    <div>
      <div class="chart"><canvas id="trip" width="520" height="520" aria-label="Тройки чисел базового датчика"></canvas></div>
      <p class="note" style="margin-top:.5rem">Тройки чисел базового датчика в проекции (6<i>u</i><sub>k+1</sub> − 9<i>u</i><sub>k</sub>) mod 1 по горизонтали и <i>u</i><sub>k+2</sub> по вертикали. У RANDU все точки ложатся на прямую: это его известный дефект.</p>
    </div>
  </div>
  <p class="readout" id="readout"></p>
  <p>__MARIMO__</p>
  <p class="note">В полной версии шесть методов (обратная функция, Бокс–Мюллер, полярный, метод отношений, метод Неймана, квантильный) для всех трёх величин и таблица сравнения: __L_DATCHIK__.</p>
</section>

<section class="wrap" id="fonts">
  <h2>Шрифты: высота строчных и толщина штриха</h2>
  <div class="text">
  <p>Второй датасет мы собрали сами: шрифтовые семейства из Google Fonts (сейчас их __N_FONTS__), у каждого измерено начертание Regular
  прямо по контурам букв. Все размеры даны в долях кегля, то есть не зависят от размера, которым набран текст.</p>
  </div>
  <figure class="wide">
    <img src="img/fonts_measure.png" alt="Буквы H x l n o шрифта Playfair с линиями измерений" loading="lazy">
    <figcaption>Что измеряется: x-height (верх строчной «x»), высота прописных, толщина вертикали «l» и контраст «o».</figcaption>
  </figure>
  <div class="facts">
    <div class="fact"><strong>__XH_SANS__</strong><span>медианная x-height гротесков</span></div>
    <div class="fact"><strong>__XH_SERIF__</strong><span>медианная x-height антикв: строчные ниже, критерий Манна–Уитни это подтверждает</span></div>
    <div class="fact"><strong>ρ = __RHO__</strong><span>связь x-height и толщины штриха: чем выше строчные, тем толще штрих</span></div>
  </div>
  <div class="chart" style="position:relative;max-width:820px">
    <svg id="scatter" viewBox="0 0 640 380" role="img" aria-label="Точечный график: x-height и толщина штриха по шрифтам"></svg>
    <div id="tip" role="status"></div>
    <div class="legend" id="flegend"></div>
  </div>
  <p class="note">Наведите на точку или выберите её клавишей Tab, чтобы увидеть название шрифта. Исследование: __L_FONTS__.</p>
</section>
</main>

<footer class="wrap">
  <p>Lab01, прикладная статистика, осень 2026. Нефедьева Антонина Сергеевна (Инфохимия): исследование вин.
  Быкова Кристина Денисовна (КТвД): оформление, датасет шрифтов, интерактивный датчик, этот сайт.</p>
  <p>Данные: Wine Quality, UCI ML Repository, DOI 10.24432/C56S3T; шрифты: Google Fonts (SIL OFL).</p>
</footer>

<script>
const D = __DATA__;
const CAT = {sans:["гротески","#2a78d6"], serif:["антиквы","#eb6834"], display:["акцидентные","#1baf7a"],
             handwriting:["рукописные","#9b59b6"], mono:["моноширинные","#5c6170"]};
const nf = (v, d=3) => v.toFixed(d).replace(".", ",");

/* --- заголовок: вес буквы = квантиль толщины штриха --- */
const title = document.getElementById("title");
const stems = D.fonts.stems, sMin = stems[0], sMax = stems[stems.length - 1];
title.querySelectorAll(".w").forEach(w => {
  w.innerHTML = [...w.textContent].map(ch => ch === " " ? " " : `<span class="l" aria-hidden="true">${ch}</span>`).join("");
});
function reroll() {
  title.querySelectorAll("span.l").forEach(s => {
    const st = stems[Math.floor(Math.random() * stems.length)];
    // ранг толщины среди шрифтов -> вес 400..900
    const r = stems.indexOf(st) / (stems.length - 1);
    s.style.fontWeight = Math.round(400 + 500 * r);
    s.title = `толщина штриха ${nf(st)}`;
  });
}
document.getElementById("reroll").addEventListener("click", reroll);
reroll();

/* --- таблица вин --- */
document.getElementById("wine-rows").innerHTML = D.wine.rows.map(r =>
  `<tr><td>${r.name}</td><td class="num">${nf(r.mean)}</td><td class="num">${nf(r.sd)}</td><td class="num">${nf(r.skew, 2)}</td><td>${r.unit}</td></tr>`).join("");

/* --- базовые датчики --- */
function makeBase(kind, seed = 2026) {
  if (kind === "randu") { let x = (seed | 1) >>> 0; return () => (x = Number((65539n * BigInt(x)) % 2147483648n)) / 2147483648; }
  if (kind === "mmix") {
    let x = BigInt(seed); const a = 6364136223846793005n, c = 1442695040888963407n, m = (1n << 64n) - 1n;
    return () => { x = (a * x + c) & m; return (Number(x >> 11n) + 0.5) / 9007199254740992; };
  }
  return Math.random;
}
const Q = D.wine.q;
function quantile(u) {           // эмпирическая F^-1 с линейной интерполяцией
  const t = u * (Q.length - 1), i = Math.min(Math.floor(t), Q.length - 2);
  return Q[i] + (t - i) * (Q[i + 1] - Q[i]);
}

/* --- гистограмма --- */
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
const E = D.wine.edges, H = D.wine.hist;
const W = 640, Hh = 300, pad = {l: 44, r: 12, t: 12, b: 40};
const xs = v => pad.l + (v - E[0]) / (E[E.length - 1] - E[0]) * (W - pad.l - pad.r);
const yMax = Math.max(...H) * 1.25;
const ys = v => Hh - pad.b - v / yMax * (Hh - pad.t - pad.b);
const svg = document.getElementById("hist");
function axes(s, xticks, yticks, xlab, ylab, X, Y, w, h, xd = 2, yd = 0) {
  xticks.forEach(t => { s.append(el("line", {x1: X(t), x2: X(t), y1: pad.t, y2: h - pad.b, stroke: "#e6e5e0"}));
    const tx = el("text", {x: X(t), y: h - pad.b + 18, "text-anchor": "middle", "font-size": 12, fill: "#5c6170"}); tx.textContent = nf(t, xd); s.append(tx); });
  yticks.forEach(t => { s.append(el("line", {x1: pad.l, x2: w - pad.r, y1: Y(t), y2: Y(t), stroke: "#e6e5e0"}));
    const tx = el("text", {x: pad.l - 6, y: Y(t) + 4, "text-anchor": "end", "font-size": 12, fill: "#5c6170"}); tx.textContent = nf(t, yd); s.append(tx); });
  const xl = el("text", {x: (pad.l + w - pad.r) / 2, y: h - 6, "text-anchor": "middle", "font-size": 13, fill: "#1a1d24"}); xl.textContent = xlab; s.append(xl);
  const yl = el("text", {x: 12, y: (pad.t + h - pad.b) / 2, "text-anchor": "middle", "font-size": 13, fill: "#1a1d24", transform: `rotate(-90 12 ${(pad.t + h - pad.b) / 2})`}); yl.textContent = ylab; s.append(yl);
}
function drawHist(gen) {
  svg.replaceChildren();
  axes(svg, [2.8, 3.0, 3.2, 3.4, 3.6, 3.8], [0, 1, 2, 3], "pH, ед. pH", "плотность", xs, ys, W, Hh);
  H.forEach((v, i) => svg.append(el("rect", {x: xs(E[i]) + .5, y: ys(v), width: Math.max(0, xs(E[i + 1]) - xs(E[i]) - 1), height: ys(0) - ys(v), fill: "#bcd6f3"})));
  const cnt = new Array(H.length).fill(0), bw = E[1] - E[0];
  gen.forEach(v => { const i = Math.floor((v - E[0]) / bw); if (i >= 0 && i < cnt.length) cnt[i]++; });
  let d = "";
  cnt.forEach((c, i) => { const y = ys(c / gen.length / bw); d += `${i ? "L" : "M"}${xs(E[i])},${y}L${xs(E[i + 1])},${y}`; });
  svg.append(el("path", {d, fill: "none", stroke: "#eb6834", "stroke-width": 2}));
}

/* --- тройки --- */
const cv = document.getElementById("trip"), ctx = cv.getContext("2d");
function drawTriples(kind) {
  const u = makeBase(kind, 7), n = 3000, a = [];
  for (let i = 0; i < n + 2; i++) a.push(u());
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.strokeStyle = "#e6e5e0"; ctx.strokeRect(20.5, 20.5, cv.width - 40, cv.height - 40);
  ctx.fillStyle = "rgba(26,29,36,.55)";
  for (let k = 0; k < n; k++) {
    const x = ((6 * a[k + 1] - 9 * a[k]) % 1 + 1) % 1, y = a[k + 2];
    ctx.fillRect(20 + x * (cv.width - 40), cv.height - 20 - y * (cv.height - 40), 2.2, 2.2);
  }
}

/* --- запуск --- */
const nIn = document.getElementById("n"), nOut = document.getElementById("nout"), baseIn = document.getElementById("base");
function run() {
  const n = +nIn.value, u = makeBase(baseIn.value, 2026 + Math.floor(Math.random() * 1e6));
  const gen = Array.from({length: n}, () => Math.round(quantile(u()) * 100) / 100);  // шаг прибора 0,01
  drawHist(gen);
  drawTriples(baseIn.value);
  const m = gen.reduce((s, v) => s + v, 0) / n, sd = Math.sqrt(gen.reduce((s, v) => s + (v - m) ** 2, 0) / (n - 1));
  const sorted = gen.slice().sort((p, q) => p - q);
  let D_ = 0;   // приближённый D Колмогорова по 201 квантилю данных
  Q.forEach((q, i) => { let lo = 0, hi = n; while (lo < hi) { const mid = (lo + hi) >> 1; sorted[mid] <= q ? lo = mid + 1 : hi = mid; }
    D_ = Math.max(D_, Math.abs(lo / n - i / (Q.length - 1))); });
  document.getElementById("readout").innerHTML =
    `Датчик: среднее <b>${nf(m)}</b>, СКО <b>${nf(sd)}</b>; данные: среднее <b>3,188</b>, СКО <b>0,151</b>. ` +
    `Расстояние Колмогорова между функциями распределения <b>${nf(D_)}</b>.`;
}
nIn.addEventListener("input", () => { nOut.textContent = nIn.value; });
nIn.addEventListener("change", run);
baseIn.addEventListener("change", run);
document.getElementById("go").addEventListener("click", run);
run();

/* --- точечный график шрифтов --- */
const sc = document.getElementById("scatter"), tip = document.getElementById("tip");
const P = D.fonts.pts, SW = 640, SH = 380;
const xr = [Math.min(...P.map(p => p.x)) - .01, Math.max(...P.map(p => p.x)) + .01];
const sr = [0, Math.max(...P.map(p => p.s)) * 1.08];
const X = v => pad.l + (v - xr[0]) / (xr[1] - xr[0]) * (SW - pad.l - pad.r);
const Y = v => SH - pad.b - (v - sr[0]) / (sr[1] - sr[0]) * (SH - pad.t - pad.b);
axes(sc, [0.40, 0.45, 0.50, 0.55].filter(t => t > xr[0] && t < xr[1]), [0, 0.1, 0.2, 0.3].filter(t => t < sr[1]),
     "x-height, доли кегля", "толщина штриха, доли кегля", X, Y, SW, SH, 2, 2);
P.forEach(p => {
  const c = el("circle", {cx: X(p.x), cy: Y(p.s), r: 6, fill: (CAT[p.c] || ["", "#999"])[1], "fill-opacity": .85,
    stroke: "#fff", "stroke-width": 1.2, tabindex: 0, "aria-label": `${p.f}: x-height ${nf(p.x)}, штрих ${nf(p.s)}`});
  const show = () => { tip.textContent = `${p.f}: x-height ${nf(p.x)}, штрих ${nf(p.s)}`;
    const b = sc.getBoundingClientRect(), k = b.width / SW;
    tip.style.left = `${X(p.x) * k + 12}px`; tip.style.top = `${Y(p.s) * k - 10}px`; tip.style.opacity = 1; };
  const hide = () => { tip.style.opacity = 0; };
  c.addEventListener("mouseenter", show); c.addEventListener("focus", show);
  c.addEventListener("mouseleave", hide); c.addEventListener("blur", hide);
  sc.append(c);
});
const present = [...new Set(P.map(p => p.c))];
document.getElementById("flegend").innerHTML = Object.entries(CAT).filter(([k]) => present.includes(k))
  .map(([k, [name, col]]) => `<span><i class="sw" style="background:${col}"></i>${name} (${P.filter(p => p.c === k).length})</span>`).join("");
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
