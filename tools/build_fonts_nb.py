"""Собрать ktvd/fonts.ipynb: доп. задание КТвД по датасету шрифтов.

    python -m labfont.collect fonts/ data/fonts.csv     # 1. измерить коллекцию
    python tools/build_fonts_nb.py                       # 2. собрать ноутбук (числа в тексте из расчёта)
    python tools/run_nb.py ktvd/fonts.ipynb              # 3. выполнить (или jupyter nbconvert --execute)

Числа в блоках «Результаты» подставляются из того же расчёта, что и в ноутбуке, с тем же seed,
поэтому текст и выводы ячеек не расходятся. Выводы «своими словами» оставлены автору (TODO КТвД).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labstat import dependence, describe as dsc, distributions as dist, hypotheses as hyp, stochastic  # noqa: E402

SEED = 2026
DATA = ROOT / "data" / "fonts.csv"
OTHERS = ["cap_height", "contrast", "width_o", "adv_mean", "weight_class"]


def f(x, d=3):
    return f"{x:.{d}f}"


def pexp(p):
    """p-значение для текста в LaTeX."""
    if p < 1e-4:
        e = int(np.floor(np.log10(p)))
        return f"p \\approx 10^{{{e}}}"
    return f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.4f}"


def analyse(df):
    r = {"N": len(df), "cats": df.category.value_counts().to_dict()}
    full = df.dropna(subset=OTHERS + ["x_height", "stem"])
    for v in ("x_height", "stem"):
        y = full[v].to_numpy()
        X = full[[c for c in OTHERS + ["x_height", "stem"] if c != v]].to_numpy()
        lin = stochastic.cv_r2_linear(X, y, k=5, seed=SEED)
        knn = stochastic.cv_r2_knn(X, y, k_neighbors=5, k=5, seed=SEED)
        y = df[v].to_numpy()
        d = dsc.describe(y)
        fits = dist.fit_candidates(y)
        r[v] = {"lin": lin, "knn": knn, "d": d, "sd_model": full[v].std(ddof=1), "n_model": len(full), "fits": fits, "out": int(dsc.iqr_outliers(y).sum()),
                "within": df.groupby("category")[v].std().mean() / y.std(ddof=1),
                "k2": hyp.normality_dagostino(y)}
    x, s = df.x_height, df.stem
    r["corr"] = dependence.correlations(x, s)
    r["chi"] = dependence.chi2_independence(x, s, bins=4 if len(df) >= 80 else 2)
    r["perm"] = dependence.permutation_spearman(x, s, n_perm=2000, seed=SEED)
    r["within_rho"] = {c: stats.spearmanr(g.x_height, g.stem)[0] for c, g in df.groupby("category") if len(g) >= 8}
    sans, serif = df.x_height[df.category == "sans"], df.x_height[df.category == "serif"]
    r["h_sans"] = hyp.mann_whitney(sans, serif, alternative="greater") if len(sans) > 3 and len(serif) > 3 else None
    r["sans_serif"] = (sans.median() if len(sans) else np.nan, serif.median() if len(serif) else np.nan)
    return r


def cap(s):
    return s if s[:1].isascii() else s[:1].upper() + s[1:]


RU = {"cap_height": "высота прописных", "contrast": "контраст", "width_o": "ширина «o»",
      "adv_mean": "средняя ширина букв", "weight_class": "вес OS/2"}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}, "tags": ["hide-input"]},
            "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


UP = "\n\n<sub>[↑ к содержанию](#toc)</sub>"


def build(df, r, sample_font: str | None, excluded=(), n_raw=None):
    cells = []
    cats = ", ".join(f"{k or 'без категории'} {v}" for k, v in sorted(r["cats"].items(), key=lambda t: -t[1]))
    X, S = r["x_height"], r["stem"]
    cells.append(md(f"""
# Lab01 · КТвД · доп. задание: пропорции шрифтов

**Две случайные величины одной природы: высота строчных (x-height) и толщина основного штриха.**

| | |
|---|---|
| Датасет | собственный: шрифты Google Fonts, измерены по контурам букв скриптом `labfont` (fontTools) |
| Генеральная совокупность | собранная коллекция, $N = {r['N']}$ семейств; одно начертание на семейство, ближайшее к Regular |
| Категории | {cats} |
| Случайные величины | RV01 **x-height**, RV02 **толщина штриха «l»**; обе в долях кегля |
| Автор | Быкова Кристина Денисовна (ИСУ 501322), трек КТвД |
| Код | `labfont/` (измерения, схема), `labstat/` (статистика, общая с основной частью), тесты `tests/test_labfont.py` |

<a id="toc"></a>
**Содержание**

- [1. Dataset: коллекция шрифтов](#ds)
    - [1.1 Как измерено](#ds-how)
    - [1.2 Обзор данных](#ds-eda)
- [2. Random Variable 01 (x-height)](#rv1)
    - [2.1 Description (природа события)](#rv1-desc)
    - [2.2 EDA & Distribution](#rv1-eda)
- [3. Random Variable 02 (толщина штриха)](#rv2)
    - [3.1 Description (природа события)](#rv2-desc)
    - [3.2 EDA & Distribution](#rv2-eda)
- [4. Independence: x-height и толщина штриха](#dep)
    - [4.1 Гипотеза: у гротесков x-height больше, чем у антикв](#hyp)
- [5. Resume](#resume)
    - [5.1 Итоговые метрики](#res-metrics)
    - [5.2 Выводы](#res-concl)
    - [5.3 Литература](#res-lit)
    - [5.4 cat_mark_table](#res-marks)

*Код в ячейках скрыт; раскрыть — щелчок по полоске слева от ячейки. Основная часть работы: [`../lab01.ipynb`](../lab01.ipynb).*
"""))
    cells.append(code(f"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import sys; sys.path.insert(0, "..")   # пакеты labstat и labfont лежат в корне репозитория

from labstat import dependence, describe as dsc, distributions as dist, hypotheses as hyp, stochastic, plots
from labfont import fontplots as fp
from labfont.illustrate import measurement_figure

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.precision", 4)
plots.style()
SEED = {SEED}
raw = pd.read_csv("../data/fonts.csv")
raw["category"] = raw["category"].fillna("")
# Шрифты без настоящих строчных (капитель SC, униказ): их «x» — это капитель или прописная
print("Исключены (нет строчных букв):", raw.family[raw.no_lowercase].tolist())
fonts = raw[~raw.no_lowercase].reset_index(drop=True)
OTHERS = {OTHERS!r}
fonts.shape
"""))
    cells.append(md(f"""
<a id="ds"></a>
## 1. Dataset: коллекция шрифтов

<a id="ds-how"></a>
### 1.1 Как измерено

Каждое семейство скачано с fonts.google.com («Get font → Download all») и разложено по папкам категорий.
Скрипт `python -m labfont.collect fonts/ data/fonts.csv` берёт из архива одно начертание, ближайшее к Regular
(прямое, вес 400; у вариативного шрифта создаётся экземпляр с $\\mathrm{{wght}} = 400$), и меряет контуры букв.

Все длины делятся на размер кегля в единицах шрифта (UPM), поэтому величины **безразмерные, в долях кегля**:
x-height $= 0.5$ значит, что при кегле 16 px строчные буквы высотой 8 px.

- **x-height**: верхняя точка контура «x»;
- **толщина штриха**: ширина вертикали «l» на линии $y = \\tfrac12\\,$x-height. Контур дробится в многоугольник,
  точки пересечения с горизонталью сортируются, между парами лежит «чернила» (правило чёт-нечет);
- **контраст**: толщина боков «o» к толщине верха и низа; **ширина**: ширина «o» к x-height.

**Очистка.** Измерено {n_raw} семейств. Исключены {len(excluded)} шрифта без настоящих строчных букв
({", ".join(excluded)}): в капительных (SC) и униказовых шрифтах на месте «x» стоит капитель или прописная,
и «x-height» у них — это не высота строчных. Spectral SC вдобавок повторяет Spectral. Начертания оптических размеров
(Inter 18pt, Fraunces 72pt) и ширин (Roboto SemiCondensed) объединены с основным семейством по папке Google Fonts.
Категории взяты из каталога fonts.google.com (`data/font_categories.csv`).

Почему одно начертание на семейство: Bold, Italic и Regular одного семейства рисовал один дизайнер,
это зависимые наблюдения. Наблюдение здесь — семейство.
""" + UP))
    if sample_font:
        cells.append(code(f"""
fig, m = measurement_figure(Path("../data/fonts_sample/{sample_font}").read_bytes())
plots.save(fig, "fonts_measure.png");
"""))
        cells.append(md('<a id="fig1"></a>\n*Рис. 1. Схема измерений на контурах настоящего шрифта: x-height, высота прописных, толщина штриха «l» и контраст «o».*'))
    cells.append(md(f"""
<a id="ds-eda"></a>
### 1.2 Обзор данных
""" + UP))
    cells.append(code("""
print("Семейств:", len(fonts), "| с кириллицей:", int(fonts.has_cyrillic.sum()), "| вариативных:", int(fonts.is_variable.sum()))
print("Вес OS/2 у выбранных начертаний:", fonts.weight_class.value_counts().sort_index().to_dict())
fonts.groupby("category")[["x_height", "stem", "contrast", "width_o"]].median().round(3)
"""))
    cells.append(code("""
cols = ["x_height", "cap_height", "x_to_cap", "stem", "contrast", "width_o", "adv_mean"]
plots.corr_heatmap(fonts[cols].rename(columns=fp.COL_RU).corr(method="spearman"), "fonts_corr.png");
"""))
    cells.append(md('<a id="fig2"></a>\n*Рис. 2. Ранговые корреляции Спирмена $\\rho$ между измерениями шрифтов.*'))

    for key, rv, num, name, aid, unit_note in (
        ("x_height", "RV01", 2, "x-height", "rv1", "высота строчных"),
        ("stem", "RV02", 3, "толщина штриха", "rv2", "толщина вертикали «l»"),
    ):
        R = r[key]
        best = R["fits"].index[0]
        cells.append(md(f'<a id="{aid}"></a>\n## {num}. Random Variable 0{num - 1} ({name})' + UP))
        cells.append(md(f"""
<a id="{aid}-desc"></a>
### {num}.1 Description (природа события)

**Величина.** {unit_note.capitalize()} в долях кегля; точность измерения $1/\\mathrm{{UPM}}$ (у большинства шрифтов
$1/1000$ или $1/2048$ кегля), то есть $\\le 0.001$.

**Событие.** Из коллекции случайно выбирается семейство шрифтов, у его Regular измеряется {unit_note}.

**Что посчитано для проверки стохастической природы.** Прогноз величины по остальным измерениям
({", ".join(RU[c] for c in OTHERS)} и вторая величина) на 5-кратной кросс-валидации: линейная регрессия и $k = 5$ ближайших
соседей. Если бы величина вычислялась формулой из остальных, $R^2 \\approx 1$, а остаток был бы порядка
точности $0.001$. Плюс разброс внутри одной категории: если бы категория (гротеск, антиква…) задавала величину,
внутри категории разброса бы не было.

**TODO КТвД: природа события своими словами.** _Вопросы: кто и почему выбирает {name} шрифта? можно ли его
вычислить по другим пропорциям или по категории? что из расчёта ниже говорит «за» и «против»?_
""" + UP))
        cells.append(code(f"""
full = fonts.dropna(subset=OTHERS + ["x_height", "stem"])   # без шрифтов с неизмеренным контрастом
y = full["{key}"].to_numpy()
Xo = full[[c for c in OTHERS + ["x_height", "stem"] if c != "{key}"]].to_numpy()
lin = stochastic.cv_r2_linear(Xo, y, k=5, seed=SEED)
knn = stochastic.cv_r2_knn(Xo, y, k_neighbors=5, k=5, seed=SEED)
print(f"Линейная модель: R² = {{lin['r2_cv']:.3f}}, СКО остатка {{lin['resid_std']:.4f}} кегля")
print(f"5 ближайших соседей: R² = {{knn['r2_cv']:.3f}}, СКО остатка {{knn['resid_std']:.4f}} кегля")
print(f"Наблюдений в модели: {{len(full)}} (без шрифтов с неизмеренным контрастом)")
print(f"СКО самой величины {{y.std(ddof=1):.4f}}; точность измерения ≤ 0.001")
fp.box_by_category(fonts, "{key}", "{name}", "доли кегля", "fonts_{key}_by_cat.png");
"""))
        cells.append(md(f'<a id="fig_{key}_cat"></a>\n*Рис. {num + 1 if key == "x_height" else 5}. {cap(name)} по категориям шрифтов: ящики с усами и сами наблюдения.*'))
        cells.append(md(f"""
**Результаты:**

- Линейная модель по остальным измерениям: $R^2 = {f(R['lin']['r2_cv'])}$, СКО остатка {f(R['lin']['resid_std'], 4)} кегля{f" (отрицательный $R^2$ на кросс-валидации: прогноз на новых шрифтах хуже, чем просто среднее; 6 признаков при {R['n_model']} наблюдениях, модель переобучается)" if R['lin']['r2_cv'] < 0 else ""}.
- 5 ближайших соседей: $R^2 = {f(R['knn']['r2_cv'])}$, СКО остатка {f(R['knn']['resid_std'], 4)} кегля.
- СКО самой величины {f(R['sd_model'], 4)} (шрифтов в модели: {R['n_model']}), точность измерения $\\le 0.001$: даже лучший остаток ({f(min(R['lin']['resid_std'], R['knn']['resid_std']), 4)}) больше точности, отношение $\\approx {min(R['lin']['resid_std'], R['knn']['resid_std']) / 0.001:.0f}$.
- Средний по категориям разброс внутри категории составляет {R['within'] * 100:.0f} % общего СКО.
"""))
        cells.append(md(f'<a id="{aid}-eda"></a>\n### {num}.2 EDA & Distribution' + UP))
        cells.append(code(f"""
x_{key} = fonts["{key}"].to_numpy()
print("Выбросов по правилу 1.5·IQR:", dsc.iqr_outliers(x_{key}).sum())
display(dsc.compare_table({{"все": x_{key}}} | {{c: g["{key}"] for c, g in fonts.groupby("category")}}).T)
fit_{key} = dist.fit_candidates(x_{key})
fit_{key}
"""))
        cells.append(code(f"""
best_{key} = {{nm: dist.fit(nm, x_{key}) for nm in ["norm"] + [d for d in fit_{key}.index if d != "norm"][:2]}}
plots.eda_panel(x_{key}, "{name}", "доли кегля", best_{key}, "fonts_{key}_eda.png", bins="fd");
"""))
        figno = 4 if key == "x_height" else 6
        cells.append(md(f'<a id="fig_{key}_eda"></a>\n*Рис. {figno}. {cap(name)}: гистограмма с подобранными плотностями, ящик с усами, Q-Q к нормальному.*'))
        d = R["d"]
        cells.append(md(f"""
**Результаты:**

- $N = {r['N']}$, среднее {f(d['mean'])}, медиана {f(d['median'])}, СКО {f(d['std'], 4)}, коэффициент вариации {d['cv'] * 100:.1f} %.
- Асимметрия ${d['skew']:+.2f}$, эксцесс ${d['kurt']:+.2f}$; диапазон {f(d['min'])}–{f(d['max'])}; выбросов по $1.5\\cdot\\mathrm{{IQR}}$: {R['out']}.
- Лучшее семейство по AIC: {best}; D'Agostino $K^2$ для нормального: ${pexp(R['k2']['p'])}$.

**TODO КТвД: что видно на гистограмме и Q-Q, своими словами.**
"""))

    corr = r["corr"]
    chi, perm = r["chi"], r["perm"]
    within = ", ".join(f"{k or 'без категории'} {v:+.2f}" for k, v in r["within_rho"].items())
    cells.append(md(f"""
<a id="dep"></a>
## 4. Independence: x-height и толщина штриха

Обе величины одной природы: пропорции одной и той же буквы, их задаёт один дизайнер. По условию для таких величин
проверяется независимость. $H_0$: x-height и толщина штриха независимы. Проверки: корреляции Пирсона, Спирмена,
Кендалла; $\\chi^2$ на таблице сопряжённости по квантилям; перестановочный тест для $\\rho$ Спирмена
(2000 перестановок). Отдельно $\\rho$ внутри каждой категории: общая связь может возникать только из-за смеси
категорий (парадокс Симпсона).
""" + UP))
    cells.append(code("""
display(dependence.correlations(fonts.x_height, fonts.stem))
chi = dependence.chi2_independence(fonts.x_height, fonts.stem, bins=4 if len(fonts) >= 80 else 2)
perm = dependence.permutation_spearman(fonts.x_height, fonts.stem, n_perm=2000, seed=SEED)
print(f"χ² = {chi['chi2']:.1f}, df = {chi['dof']}, p = {chi['p']:.3g}, V Крамера = {chi['cramer_v']:.3f}")
print(f"перестановочный тест: ρ = {perm['rho']:.3f}, p = {perm['p']:.4f}")
print("ρ внутри категорий:", {c: round(stats.spearmanr(g.x_height, g.stem)[0], 3) for c, g in fonts.groupby("category") if len(g) >= 8})
fp.scatter_by_category(fonts, "x_height", "stem", "x-height, доли кегля", "толщина штриха, доли кегля",
                       "x-height и толщина штриха", "fonts_dependence.png");
"""))
    cells.append(md('<a id="fig7"></a>\n*Рис. 7. x-height и толщина штриха; цвет — категория, чёрная линия — медиана толщины по квинтилям x-height.*'))
    dec = "отвергается" if perm["p"] < 0.05 else "не отвергается"
    cells.append(md(f"""
**Результаты:**

- Пирсон {corr.loc['pearson', 'coef']:+.3f}, Спирмен {corr.loc['spearman', 'coef']:+.3f}, Кендалл {corr.loc['kendall', 'coef']:+.3f}
  (для Спирмена ${pexp(corr.loc['spearman', 'p'])}$).
- $\\chi^2 = {chi['chi2']:.1f}$ при $\\mathrm{{df}} = {chi['dof']}$, ${pexp(chi['p'])}$, $V$ Крамера {chi['cramer_v']:.3f}.
- Перестановочный тест: $\\rho = {perm['rho']:+.3f}$, ${pexp(perm['p'])}$. $H_0$ о независимости при $\\alpha = 0.05$ **{dec}**.
- $\\rho$ Спирмена внутри категорий: {within}.

**TODO КТвД: вывод о зависимости своими словами.** _Вопросы: есть ли связь, насколько она сильна? держится ли она
внутри категорий? как это объяснить с точки зрения дизайна шрифта?_
"""))
    if r["h_sans"]:
        h = r["h_sans"]
        hdec = "отвергается" if h["p"] < 0.05 else "не отвергается"
        cells.append(md(f"""
<a id="hyp"></a>
### 4.1 Гипотеза: у гротесков x-height больше, чем у антикв

В типографике принято считать, что у гротесков строчные выше, чем у антикв. Проверяем односторонним критерием
Манна–Уитни: $H_0$: $P(X_{{\\text{{sans}}}} > X_{{\\text{{serif}}}}) \\le 1/2$, $\\alpha = 0.05$.
""" + UP))
        cells.append(code("""
sans, serif = fonts.x_height[fonts.category == "sans"], fonts.x_height[fonts.category == "serif"]
h_sans = hyp.mann_whitney(sans, serif, alternative="greater")
print(f"медиана x-height: гротески {sans.median():.3f} (n = {len(sans)}), антиквы {serif.median():.3f} (n = {len(serif)})")
pd.Series(h_sans)
"""))
        cells.append(md(f"""
**Результаты:** медиана x-height у гротесков {f(r['sans_serif'][0])}, у антикв {f(r['sans_serif'][1])};
критерий Манна–Уитни ${pexp(h['p'])}$, $H_0$ **{hdec}**.
"""))

    cells.append(md('<a id="resume"></a>\n## 5. Resume\n\n<a id="res-metrics"></a>\n### 5.1 Итоговые метрики' + UP))
    cells.append(code("""
rows = []
for col, nm in [("x_height", "x-height"), ("stem", "толщина штриха")]:
    v = fonts[col]
    rows.append({"величина": nm, "N": len(v), "среднее": v.mean(), "СКО": v.std(), "медиана": v.median(),
                 "асимметрия": stats.skew(v), "эксцесс": stats.kurtosis(v),
                 "R² по остальным": stochastic.cv_r2_linear(full[[c for c in OTHERS + ["x_height", "stem"] if c != col]].to_numpy(), full[col].to_numpy(), k=5, seed=SEED)["r2_cv"],
                 "лучшая по AIC": dist.fit_candidates(v.to_numpy()).index[0]})
pd.DataFrame(rows).set_index("величина").round(4)
"""))
    cells.append(md("""
<a id="res-concl"></a>
### 5.2 Выводы

**TODO КТвД: выводы своими словами** (преподаватель просит «как на лабах по физике»).
_Вопросы: почему x-height и толщина штриха — случайные величины? какие у них распределения? зависимы ли они и
почему? что подтвердила гипотеза про гротески? какие ограничения у коллекции (не случайный отбор, мало шрифтов
в категории, Google Fonts — не все шрифты мира)?_

<a id="res-lit"></a>
### 5.3 Литература

1. Брингхерст Р. Основы стиля в типографике. М.: Д. Аронов, 2006 (Bringhurst R. The Elements of Typographic Style).
2. Королькова А. Живая типографика. М.: IndexMarket, 2007.
3. Microsoft. OpenType specification: таблицы `head` (unitsPerEm), `OS/2` (sxHeight, usWeightClass), `hmtx`, `fvar`.
4. fontTools: библиотека для работы со шрифтами на Python. github.com/fonttools/fonttools.
5. Google Fonts: каталог шрифтов с открытой лицензией (SIL OFL). fonts.google.com.

**TODO КТвД:** добавить источники из библиотек (РГБ, РНБ, ИТМО, eLibrary): за подбор литературы отдельный балл.

<a id="res-marks"></a>
### 5.4 cat_mark_table

| cat_id | notebook_link | coeff | comment |
|--------|---------------|-------|---------|
| КТвД: доп. датасет | [1](#ds) | | собственный датасет, собран и измерен своим скриптом |
| КТвД: случайная величина | [2.1](#rv1-desc), [3.1](#rv2-desc) | | $R^2$ по остальным измерениям и остаток против точности |
| КТвД: (не)зависимость | [4](#dep) | | корреляции, $\\chi^2$, перестановочный тест, проверка внутри категорий |
| КТвД: EDA | [1.2](#ds-eda), [2.2](#rv1-eda), [3.2](#rv2-eda) | | описательные статистики, категории, подбор семейств |
| КТвД: гипотеза | [4.1](#hyp) | | гротески против антикв, Манн–Уитни |
| КТвД: оформление | [содержание](#toc) | | нумерация, якоря, скрытый код, подписи рисунков, LaTeX |
""" + UP))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def load(data=DATA):
    raw = pd.read_csv(data)
    raw["category"] = raw["category"].fillna("")
    return raw, raw[~raw.no_lowercase].reset_index(drop=True)


def main(data=DATA, out=ROOT / "ktvd" / "fonts.ipynb"):
    raw, df = load(data)
    excluded = raw.family[raw.no_lowercase].tolist()
    r = analyse(df)
    sample_dir = ROOT / "data" / "fonts_sample"
    sample = next((p.name for p in sorted(sample_dir.glob("*")) if p.suffix.lower() in (".ttf", ".otf")), None) \
        if sample_dir.exists() else None
    nb = build(df, r, sample, excluded, len(raw))
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{out}: {len(nb['cells'])} ячеек, N = {r['N']}")


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:]))
