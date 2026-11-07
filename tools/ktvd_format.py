"""Оформление lab01.ipynb по критериям КТвД (запускается один раз, история в git).

Что делает:
  1. Нумерует разделы (1 → 1.1 → 1.1.1) и ставит якорь перед каждым заголовком.
  2. Строит полное вложенное оглавление и ссылки «↑ к содержанию».
  3. Подписывает рисунки («Рис. N. …») отдельной ячейкой под графиком.
  4. Переводит математику в тексте в LaTeX ($…$): σ/√n, χ², 10⁻⁴², H0, p = …, ρ, ε и т. п.
     Абзацы с «TODO» (вопросы для автора выводов) не трогаются.
  5. Скрывает код: metadata jupyter.source_hidden + тег hide-input (для экспорта в HTML).

    python tools/ktvd_format.py lab01.ipynb
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ----------------------------------------------------------------------- 1. нумерация
TOP = {"dataset": "1", "rv01": "2", "rv02": "3", "rv03": "4", "resume": "5"}
SUB = {  # заголовок ### -> номер подраздела
    "Description (природа события)": 1, "EDA & Distribution": 2, "Independence & Estimations": 3,
    "Normalization": 4, "Sample vs Population": 5, "Hypotheses": 6, "Resume": 7,
}
RESUME_SUB = {"Итоговые метрики": ("1", "resume-metrics"),
              "Все проверенные гипотезы и поправка Холма": ("2", "resume-hyp"),
              "Выводы по датасету": ("3", "resume-conclusions"),
              "Литература и материалы": ("4", "resume-lit"),
              "cat_mark_table": ("5", "resume-marks")}
RV_NAME = {"rv01": "pH", "rv02": "Фиксированная кислотность", "rv03": "Спирт"}

# ----------------------------------------------------------------------- 3. подписи рисунков
CAPTIONS = {
    "dataset_corr": "Ранговые корреляции Спирмена $\\rho$ между всеми измерениями белых вин",
    "stochastic": "{n}: прогноз по 10 другим измерениям против измеренного значения (слева) и остатки модели в сравнении с шагом записи прибора (справа)",
    "eda": "{n}: гистограмма с подобранными плотностями, ящик с усами и Q-Q-график к нормальному распределению",
    "rv01_rv02_dependence": "Совместное распределение pH и фиксированной кислотности (цвет: число вин в ячейке) и медиана pH по децилям кислотности",
    "consistency": "{n}: траектория оценок при росте $n$ (слева) и доля оценок, отклонившихся от значения в совокупности больше чем на $\\varepsilon$ (справа)",
    "normalization": "{n}: распределение после преобразования в сравнении с плотностью $N(0, 1)$",
    "sampling": "{n}: эмпирические функции распределения совокупности и двух выборок $n = 300$",
    "se": "{n}: СКО выборочного среднего при отборе с возвращением и без, в долях $\\sigma/\\sqrt{{n}}$",
    "rv01_white_vs_red": "Распределение pH белых и красных вин, пунктир: средние",
    "rv03_quality": "Содержание спирта в винах с оценкой качества $\\ge 7$ и в остальных, пунктир: средние",
}

# ----------------------------------------------------------------------- 4. LaTeX
SUP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")


def _m(s: str) -> str:
    """Внутри формулы: типографский минус -> обычный."""
    return s.replace("−", "-")


def _pow(m):
    return _m(m.group(0)).translate(SUP)


RULES: list[tuple[str, object]] = [
    (r"σ/√n·√\(\(N−n\)/\(N−1\)\)", r"$\\frac{\\sigma}{\\sqrt{n}}\\sqrt{\\frac{N-n}{N-1}}$"),
    (r"σ/√3000 = 0\.0028", r"$\\sigma/\\sqrt{3000} = 0.0028$"),
    (r"σ/√n", r"$\\sigma/\\sqrt{n}$"),
    (r"ε = 0\.1·σ²", r"$\\varepsilon = 0.1\\,\\sigma^2$"),
    (r"ε = 0\.1·σ", r"$\\varepsilon = 0.1\\,\\sigma$"),
    (r"ε = (\d[\d.]*)", r"$\\varepsilon = \1$"),
    (r"P\(\|оценка − θ\| > ε\)", r"$P(|\\hat\\theta_n - \\theta| > \\varepsilon)$"),
    (r"n²/2N ≈ 9", r"$n^2/2N \\approx 9$"),
    (r"pH = −lg a\(H⁺\)", r"$\\mathrm{pH} = -\\lg a(\\mathrm{H}^+)$"),
    (r"\|z\| ≤ 1, 2, 3", r"$|z| \\le 1, 2, 3$"),
    (r"z от −11\.8 до −16\.3", r"$z$ от $-11.8$ до $-16.3$"),
    (r"t Уэлча = −27\.8", r"$t$ Уэлча $= -27.8$"),
    (r"n = 10 → 3000", r"$n = 10 \\to 3000$"),
    # степени десяти: 3·10⁻⁶, p ≈ 10⁻⁴²
    (r"\bp ([<≈=]) (\d+·)?10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
     lambda m: "$p " + {"<": "<", "≈": "\\approx", "=": "="}[m.group(1)] + " "
     + (m.group(2)[:-1] + "\\cdot " if m.group(2) else "") + "10^{" + m.group(3).translate(SUP) + "}$"),
    (r"(\d+·)?10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
     lambda m: "$" + (m.group(1)[:-1] + "\\cdot " if m.group(1) else "") + "10^{" + m.group(2).translate(SUP) + "}$"),
    (r"\bp ([<≈=]) (\d[\d./]*\d|\d)", lambda m: "$p " + {"<": "<", "≈": "\\approx", "=": "="}[m.group(1)] + " " + m.group(2) + "$"),
    (r"χ² = (\d[\d.]*)", r"$\\chi^2 = \1$"),
    (r"χ²", r"$\\chi^2$"),
    (r"K²", r"$K^2$"),
    (r"R²", r"$R^2$"),
    (r"ρ = ([+−-]?\d[\d.]*)", lambda m: "$\\rho = " + _m(m.group(1)) + "$"),
    (r"\|ρ\|", r"$|\\rho|$"),
    (r"ρ", r"$\\rho$"),
    (r"\bV = (\d[\d.]*)", r"$V = \1$"),
    (r"\bH0\b", r"$H_0$"),
    (r"\bdf = (\d+)", r"$\\mathrm{df} = \1$"),
    (r"\bdf (\d+)", r"$\\mathrm{df} = \1$"),
    (r"\b([nNk]) = (\d+(?: / \d+)+)", r"$\1 = \2$"),
    (r"\b([nNk]) = (\d+)", r"$\1 = \2$"),
    (r"1\.5·IQR", r"$1.5\\cdot\\mathrm{IQR}$"),
    (r"\bD Колмогорова", r"$D$ Колмогорова"),
    (r"P\(A > B\) = (\d[\d.]*)", r"$P(A > B) = \1$"),
    (r"P\(A > B\)", r"$P(A > B)$"),
    (r"α = 0\.05", r"$\\alpha = 0.05$"),
    (r"μ, σ, N", r"$\\mu$, $\\sigma$, $N$"),
    (r"θ = 0", r"$\\theta = 0$"),
    (r"`s` = σ, `loc` = θ", r"`s` = $\\sigma$, `loc` = $\\theta$"),
    (r"λ Бокса–Кокса = −1\.48", r"$\\lambda$ Бокса–Кокса $= -1.48$"),
    (r"\bln X\b", r"$\\ln X$"),
    (r"N\(0, 1\)", r"$N(0, 1)$"),
    (r"\bk ближайш", r"$k$ ближайш"),
    (r"\bV Крамера", r"$V$ Крамера"),
    (r"\bD больше", r"$D$ больше"),
    (r"(?<=[\s(])n(?=[\s.,;)]|$)", r"$n$"),   # одиночная латинская n в русском тексте — объём выборки
    (r"ε", r"$\\varepsilon$"),
    (r"σ", r"$\\sigma$"),
    (r"θ", r"$\\theta$"),
]
PROTECT = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+\$|`[^`]*`|\]\([^)]*\)|<[^>]+>)", re.S)


def latexify(text: str) -> str:
    for pat, repl in RULES:
        parts = PROTECT.split(text)
        for i in range(0, len(parts), 2):          # чётные куски: обычный текст
            parts[i] = re.sub(pat, repl, parts[i])
        text = "".join(parts)
    return text


def latexify_cell(src: str) -> str:
    """По абзацам. После абзаца с TODO всё до следующего заголовка не трогаем."""
    out, skip = [], False
    for para in src.split("\n\n"):
        if para.lstrip().startswith(("#", "<a id")):
            skip = False
        if "TODO" in para and not para.lstrip().startswith("|"):
            skip = True
        if skip or para.lstrip().startswith("|"):
            # строки таблицы cat_mark_table: правим только те, где нет TODO
            if para.lstrip().startswith("|") and not skip:
                para = "\n".join(latexify(l) if "TODO" not in l else l for l in para.split("\n"))
            out.append(para)
        else:
            out.append(latexify(para))
    return "\n\n".join(out)


# ----------------------------------------------------------------------- cat_mark_table
def fix_mark_table(src: str) -> str:
    """Ссылки таблицы ведут на точные подразделы; добавлены строки критериев КТвД."""
    rows = {
        "lab.ic.01.02": ("[RV01 зависимость](#rv01-indep), [RV02](#rv02-indep), [RV03](#rv03-indep)",
                         "[2.3.1](#rv01-dep), [3.3.1](#rv02-dep), [4.3.1](#rv03-dep)"),
        "lab.ic.01.03": ("[RV01](#rv01-indep), [RV02](#rv02-indep), [RV03](#rv03-indep)",
                         "[2.3.2](#rv01-cons), [3.3.2](#rv02-cons), [4.3.2](#rv03-cons)"),
        "lab.ic.01.12": ("[сводка](#resume)", "[сводка 5.2](#resume-hyp)"),
        "lab.ic.01.19": ("[литература](#resume)", "[5.4](#resume-lit)"),
        "lab.ic.01.21": ("[литература](#resume)", "[5.4](#resume-lit)"),
        "lab.ic.01.24": ("[итоговые метрики](#resume)", "[5.1](#resume-metrics), [5.2](#resume-hyp)"),
    }
    out = []
    for line in src.split("\n"):
        for cid, (old, new) in rows.items():
            if line.startswith(f"| {cid} |"):
                line = line.replace(old, new)
        out.append(line)
    src = "\n".join(out)
    ktvd = (
        "\n| КТвД: структура | [содержание](#toc) | | индексы разделов 1 → 1.1 → 1.1.1, якорь у каждого заголовка, «↑ к содержанию» |"
        "\n| КТвД: скрытие кода | весь ноутбук | | код-ячейки свёрнуты (`source_hidden`, тег `hide-input` для HTML-экспорта) |"
        "\n| КТвД: графики | [Рис. 1](#fig1)–[Рис. 22](#fig22) | | единый стиль `labstat.plots`, русские подписи, единицы на всех осях, нумерация и подписи рисунков |"
        "\n| КТвД: формулы | весь ноутбук | | математика в тексте в LaTeX, единые обозначения ($H_0$, $p$, $\\chi^2$, $\\sigma/\\sqrt{n}$) |"
        "\n| КТвД: доп. задание | репозиторий `lab01-ktvd` | | датасет шрифтов, 2 случайные величины, EDA, датчик на marimo, сайт |"
    )
    marker = "| lab.ic.01.24 |"
    i = src.find(marker)
    j = src.find("\n", i)
    j = len(src) if j == -1 else j
    return src[:j] + ktvd + src[j:]


# ----------------------------------------------------------------------- сборка
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def main(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    toc: list[tuple[int, str, str, str]] = []   # (уровень, номер, текст, якорь)
    top, top_id, fig_no = None, None, 0
    cells = []
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code":
            cell.setdefault("metadata", {})
            cell["metadata"]["jupyter"] = {"source_hidden": True}
            cell["metadata"]["tags"] = sorted(set(cell["metadata"].get("tags", [])) | {"hide-input"})
            cells.append(cell)
            m = re.search(r'"(\w+)\.png"', src)
            if m and any("image/png" in o.get("data", {}) for o in cell.get("outputs", [])):
                fig_no += 1
                fname = m.group(1)
                key = fname if fname in CAPTIONS else fname.split("_", 1)[1]
                cap = CAPTIONS[key].format(n=RV_NAME.get(fname.split("_")[0], ""))
                cells.append(md(f"<a id=\"fig{fig_no}\"></a>\n*Рис. {fig_no}. {cap}.*"))
            continue
        if cell["cell_type"] != "markdown" or src.startswith("# Lab01 - sprint 1"):
            cells.append(cell)
            continue
        if src.startswith("# Lab01 - sprint 2"):
            cells.append(cell)
            continue

        lines = src.split("\n")
        new, pending_id, sub = [], None, None
        for ln in lines:
            m = re.match(r'<a id="([\w-]+)"></a>', ln)
            if m:
                pending_id = m.group(1)
                continue
            h = re.match(r"(#{2,4}) (.+)", ln)
            if h:
                lvl, title = len(h.group(1)), h.group(2)
                if lvl == 2:
                    top_id = pending_id
                    top = TOP[top_id]
                    num, aid = top + ".", top_id
                elif lvl == 3 and top_id == "resume":
                    k, aid = RESUME_SUB[title]
                    num = f"{top}.{k}"
                elif lvl == 3:
                    sub = SUB[title]
                    num, aid = f"{top}.{sub}", pending_id
                    CUR["sub"] = sub
                else:   # уровень 4: мат. описание внутри EDA
                    num, aid = f"{top}.{CUR['sub']}.1", pending_id
                new += [f'<a id="{aid}"></a>', f"{h.group(1)} {num} {title}"]
                toc.append((lvl, num, title, aid))
                pending_id = None
                continue
            new.append(ln)
        src = "\n".join(new)
        # «**Зависимость.**» и «**Состоятельность оценок.**» становятся подзаголовками 4-го уровня
        for bold, slug, k in (("**Зависимость.** ", "dep", 1), ("**Состоятельность оценок.** ", "cons", 2)):
            pos = src.find(bold)
            if pos == 0 or (pos > 0 and src[pos - 2:pos] == "\n\n"):
                aid = f"{top_id}-{slug}"
                num = f"{top}.{CUR['sub']}.{k}"
                title = bold.strip("* .")
                src = src[:pos] + f'<a id="{aid}"></a>\n#### {num} {title}\n\n' + src[pos + len(bold):]
                toc.append((4, num, title, aid))
        src = re.sub(r"\n{3,}", "\n\n", src)      # двойные пустые строки после заголовков
        src = latexify_cell(src)
        if "| cat_id |" in src and "lab.ic.01.02" in src:
            src = fix_mark_table(src)
        if re.search(r"^#{2,3} ", src, re.M):
            src = src.rstrip() + "\n\n<sub>[↑ к содержанию](#toc)</sub>"
        cell["source"] = src.splitlines(keepends=True)
        cells.append(cell)
    nb["cells"] = cells

    # ------------------------------------------------------------- титульная ячейка + оглавление
    first = nb["cells"][0]
    src = "".join(first["source"])
    toc_md = ['<a id="toc"></a>', "**Содержание**", ""]
    for lvl, num, title, aid in toc:
        indent = "    " * (lvl - 2)
        toc_md.append(f"{indent}- [{num} {title}](#{aid})")
    toc_md += ["", "**Часть КТвД.** Оформление этого ноутбука сделано здесь. Доп. задание КТвД (датасет шрифтов, интерактивный датчик на marimo, сайт) лежит в отдельном репозитории `lab01-ktvd` (Быкова К. Д.).", "",
               "*Код в ячейках скрыт, чтобы документ читался как отчёт: раскрыть — щелчок по полоске слева от ячейки "
               "(JupyterLab) или по «…». Все рисунки пронумерованы, подпись под каждым.*"]
    src = re.sub(r"\*\*Содержание\*\*.*", "\n".join(toc_md), src, flags=re.S)
    src = src.replace("| Нейросети |", "| КТвД | Быкова Кристина Денисовна (ИСУ 501322, группа 27): оформление, доп. задание, датчик на marimo, сайт |\n| Нейросети |")
    first["source"] = src.splitlines(keepends=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"разделов в оглавлении: {len(toc)}, рисунков: {fig_no}")


CUR = {"sub": 0}

if __name__ == "__main__":
    main(Path(sys.argv[1]))
