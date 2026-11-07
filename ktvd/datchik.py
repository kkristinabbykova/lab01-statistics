import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium", app_title="Lab01 · интерактивный датчик")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Lab01 · интерактивный датчик случайных чисел

        Датчик воспроизводит величины из исследования белых вин (Wine Quality, $N = 4898$):
        **pH**, **фиксированную кислотность** и **спирт**. Схема любого датчика:

        $$U \sim U(0, 1)\ \xrightarrow{\ \text{метод}\ }\ X \sim F_{\text{данных}}$$

        Выберите величину, базовый равномерный датчик и метод. Все графики и таблицы
        пересчитываются сразу. Внизу сравнение всех методов на одной величине.
        """
    )
    return


@app.cell
def _(mo):
    import io
    import time
    import types
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    def read_text(p):
        """Файл рядом с ноутбуком: локально это путь, в браузере (WASM) это URL."""
        try:
            return Path(str(p)).read_text(encoding="utf-8")
        except Exception:
            from pyodide.http import open_url
            return open_url(str(p)).read()

    _pub = mo.notebook_location() / "public"
    # Код датчиков: копия labrng/generators.py (tools/build_site.py держит её в актуальном виде)
    G = types.ModuleType("generators")
    exec(read_text(_pub / "generators.py"), G.__dict__)
    wine = pd.read_csv(io.StringIO(read_text(_pub / "winequality-white.csv")), sep=";")

    VARS = {
        "pH": ("pH", "ед. pH", 0.01),
        "фиксированная кислотность": ("fixed acidity", "г/дм³", 0.1),
        "спирт": ("alcohol", "% об.", 0.1),
    }
    C1, C2, INK2 = "#2a78d6", "#eb6834", "#52514e"
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "axes.edgecolor": INK2,
        "legend.frameon": False, "axes.titlesize": 11, "axes.labelsize": 10,
    })
    return C1, C2, G, INK2, VARS, np, pd, plt, time, wine


@app.cell
def _(G, VARS, mo):
    var = mo.ui.dropdown(options=list(VARS), value="pH", label="Величина")
    base = mo.ui.dropdown(options={v: k for k, v in G.BASE_RU.items()},
                          value=G.BASE_RU["pcg64"], label="Базовый датчик U(0, 1)")
    method = mo.ui.dropdown(options={v: k for k, v in G.METHOD_RU.items()},
                            value=G.METHOD_RU["quantile"], label="Метод")
    family = mo.ui.dropdown(options={"скошенное нормальное": "skewnorm", "логнормальное со сдвигом": "lognorm",
                                     "нормальное": "norm"},
                            value="скошенное нормальное", label="Модель (для методов через модель)")
    n = mo.ui.slider(start=100, stop=20000, step=100, value=4900, label="Объём выборки n", show_value=True)
    seed = mo.ui.number(start=0, stop=10**9, value=2026, label="seed")
    rounding = mo.ui.checkbox(value=True, label="округлять к шагу прибора")
    mo.vstack([mo.hstack([var, base, method]), mo.hstack([family, n, seed, rounding])])
    return base, family, method, n, rounding, seed, var


@app.cell
def _(G, VARS, base, family, method, n, rounding, seed, var, wine):
    col, unit, step = VARS[var.value]
    real = wine[col].to_numpy(float)
    gen, info = G.generate(real, n.value, method=method.value, family=family.value,
                           base=base.value, seed=int(seed.value), step=step if rounding.value else 0.0)
    cmp_ = G.compare(real, gen)
    return cmp_, gen, info, real, step, unit


@app.cell
def _(C1, C2, G, INK2, base, gen, np, plt, real, seed, step, unit, var):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    _lo, _hi = np.quantile(np.concatenate([real, gen]), [0.001, 0.999])
    _w = step * max(1, round((_hi - _lo) / step / 45))
    _bins = np.arange(_lo - step / 2, _hi + _w, _w)
    ax[0, 0].hist(real, bins=_bins, density=True, color=C1, alpha=0.35, label="данные")
    ax[0, 0].hist(gen, bins=_bins, density=True, histtype="step", color=C2, linewidth=1.8, label="датчик")
    ax[0, 0].set(xlabel=f"{var.value}, {unit}", ylabel="плотность", title="Гистограммы")
    ax[0, 0].legend()
    for _x, _c, _l in ((real, C1, "данные"), (gen, C2, "датчик")):
        _s = np.sort(_x)
        ax[0, 1].step(_s, np.arange(1, _s.size + 1) / _s.size, where="post", color=_c, label=_l)
    ax[0, 1].set(xlabel=f"{var.value}, {unit}", ylabel="F(x)", title="Эмпирические функции распределения",
                 xlim=(_lo, _hi))
    ax[0, 1].legend(loc="lower right")
    _p = np.linspace(0.005, 0.995, 199)
    _qr, _qg = np.quantile(real, _p), np.quantile(gen, _p)
    ax[1, 0].plot(_qr, _qg, "o", ms=3, color=C1)
    ax[1, 0].plot([_qr[0], _qr[-1]], [_qr[0], _qr[-1]], color=C2, linewidth=1.2, label="y = x")
    ax[1, 0].set(xlabel=f"квантили данных, {unit}", ylabel=f"квантили датчика, {unit}", title="Q-Q: датчик против данных")
    ax[1, 0].legend()
    # Проекция троек (u_k, u_{k+1}, u_{k+2}), в которой видны плоскости RANDU
    _u = G.Uniform(base.value, int(seed.value))(3002)
    _y = (6 * _u[1:-1] - 9 * _u[:-2]) % 1.0
    ax[1, 1].plot(_y, _u[2:], ".", ms=2, color=INK2, alpha=0.6)
    ax[1, 1].set(xlabel=r"$(6u_{k+1} - 9u_k)\ \mathrm{mod}\ 1$", ylabel=r"$u_{k+2}$", xlim=(0, 1), ylim=(0, 1),
                 title="Базовый датчик: тройки чисел")
    fig.tight_layout()
    fig
    return


@app.cell
def _(G, cmp_, info, method, mo, pd):
    _rows = pd.DataFrame({"данные": cmp_["real"], "датчик": cmp_["gen"]})
    _rows["разница"] = _rows["датчик"] - _rows["данные"]
    _ok = cmp_["ks_p"] > 0.05 and cmp_["chi2_p"] > 0.05
    _acc_theory = {"polar": "π/4 ≈ 0.785", "ratio": "√(πe)/4 ≈ 0.731"}.get(method.value, "")
    _acc = f"Доля принятых точек: **{info['accept']:.3f}**" + (f" (теория {_acc_theory})." if _acc_theory else ".")
    _params = f"Параметры модели (ММП): `{tuple(round(p, 4) for p in info['params'])}`." if "params" in info else ""
    mo.vstack([
        mo.md(f"### Метрики: данные против датчика ({G.METHOD_RU[method.value]})"),
        mo.ui.table(_rows.round(4).reset_index(names="статистика"), selection=None, pagination=False),
        mo.callout(mo.md(
            f"Критерий Колмогорова–Смирнова (две выборки): $D = {cmp_['ks_D']:.4f}$, $p = {cmp_['ks_p']:.3f}$.  \n"
            f"$\\chi^2$ однородности по децилям данных: $\\chi^2 = {cmp_['chi2']:.1f}$, "
            f"$\\mathrm{{df}} = {cmp_['chi2_dof']}$, $p = {cmp_['chi2_p']:.3f}$.  \n"
            + ("**$H_0$ «датчик и данные из одного распределения» не отвергается** при $\\alpha = 0.05$."
               if _ok else "**$H_0$ отвергается**: датчик заметно отличается от данных.")
            + f"  \n{_acc} {_params}"), kind="success" if _ok else "warn"),
    ])
    return


@app.cell
def _(method, mo):
    _txt = {
        "quantile": r"""
**Квантильное преобразование.** Если $U \sim U(0,1)$ и $F$ — функция распределения, то
$X = F^{-1}(U)$ имеет распределение $F$, потому что $P(X \le x) = P(U \le F(x)) = F(x)$.
Вместо неизвестной $F$ берём эмпирическую: порядковые статистики $x_{(i)}$ ставим в узлы
$p_i = (i - 0.5)/n$ и соединяем линейно. Модель не нужна, но датчик не выйдет за $[x_{(1)}, x_{(n)}]$.""",
        "inverse": r"""
**Метод обратной функции по модели.** То же $X = F^{-1}(U)$, но $F$ — подобранное семейство
с параметрами ММП. Качество датчика ограничено качеством модели: если гипотеза о семействе
отвергнута (см. раздел Hypotheses в ноутбуке), датчик тоже будет отличаться от данных.""",
        "box_muller": r"""
**Бокс–Мюллер.** Из двух независимых $U_1, U_2$:
$Z_1 = \sqrt{-2\ln U_1}\cos 2\pi U_2,\ Z_2 = \sqrt{-2\ln U_1}\sin 2\pi U_2$ — две независимые $N(0,1)$.
Затем $Z \to X$: для скошенного нормального $X = \mathrm{loc} + \mathrm{scale}\,(\delta|Z_0| + \sqrt{1-\delta^2}\,Z_1)$,
$\delta = a/\sqrt{1+a^2}$; для логнормального $X = \theta + e^{\mu} e^{sZ}$.""",
        "polar": r"""
**Полярный метод Марсальи.** $V_1, V_2 \sim U(-1,1)$, $S = V_1^2 + V_2^2$; пара принимается при $0 < S < 1$
(доля $\pi/4$). Тогда $Z_{1,2} = V_{1,2}\sqrt{-2\ln S / S}$. Синус и косинус не нужны.""",
        "ratio": r"""
**Метод отношений (Киндерман–Монахан).** $(u, v)$ равномерно в $[0,1]\times[-\sqrt{2/e}, \sqrt{2/e}]$,
$X = v/u$ принимается при $X^2 \le -4\ln u$. Принятые $X \sim N(0,1)$, доля принятых $\sqrt{\pi e}/4$.""",
        "neumann": r"""
**Метод Неймана (отбор).** Плотность $f$ — гистограмма данных (40 интервалов). Точка $(x, y)$ бросается
равномерно в прямоугольник $[a, b]\times[0, \max f]$ и принимается при $y \le f(x)$. Принятые $x$ имеют
плотность $f$; доля принятых $1 / (\max f\,(b - a))$.""",
    }[method.value]
    mo.md(_txt)
    return


@app.cell
def _(mo):
    mo.md(r"""### Сравнение всех методов на выбранной величине""")
    return


@app.cell
def _(G, VARS, base, family, n, pd, real, rounding, seed, step, time, var):
    _rows = []
    for _m in G.METHODS:
        _t = time.perf_counter()
        _x, _i = G.generate(real, n.value, method=_m, family=family.value, base=base.value,
                            seed=int(seed.value), step=step if rounding.value else 0.0)
        _dt = (time.perf_counter() - _t) * 1000
        _c = G.compare(real, _x)
        _rows.append({"метод": G.METHOD_RU[_m], "Δ среднего": _c["gen"]["среднее"] - _c["real"]["среднее"],
                      "Δ СКО": _c["gen"]["СКО"] - _c["real"]["СКО"],
                      "Δ асимметрии": _c["gen"]["асимметрия"] - _c["real"]["асимметрия"],
                      "КС p": _c["ks_p"], "χ² p": _c["chi2_p"], "доля принятых": _i["accept"], "время, мс": _dt})
    summary = pd.DataFrame(_rows).round(4)
    summary
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        **Как читать.** Методы «через модель» (обратная функция, Бокс–Мюллер, полярный, отношений)
        повторяют данные настолько, насколько хороша модель. Квантильное преобразование и метод Неймана
        моделью не пользуются и повторяют данные почти точно, зато не обобщают: не дают значений
        за пределами наблюдённых. Базовый датчик RANDU на графике троек ложится на прямые: это его
        известный дефект (Knuth, TAOCP, т. 2, § 3.3.4).

        Код датчиков: `labrng/generators.py`, автотесты: `tests/test_labrng.py`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
