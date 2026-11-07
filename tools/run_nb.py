"""Мини-исполнитель .ipynb без Jupyter: выполняет код-ячейки по порядку и записывает выводы.

    python tools/run_nb.py lab01.ipynb

Нужен, когда Jupyter не установлен (например, в CI). Поддерживает то, что используется
в ноутбуке: print, display(DataFrame), значение последнего выражения, графики matplotlib.
Обычный способ всё так же `jupyter nbconvert --to notebook --execute --inplace lab01.ipynb`.
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _rich(obj):
    """Представление объекта как в Jupyter: text/plain и, если есть, text/html."""
    data = {"text/plain": repr(obj)}
    if hasattr(obj, "_repr_html_"):
        try:
            html = obj._repr_html_()
            if html:
                data["text/html"] = html
        except Exception:  # noqa: BLE001
            pass
    return data


def _lines(s: str):
    return s.splitlines(keepends=True)


def run(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    os.chdir(path.parent)
    sys.path.insert(0, str(path.parent))
    outputs: list = []
    state = {"buf": io.StringIO()}
    ns: dict = {"__name__": "__main__"}
    import pandas as pd
    pd.set_option("display.max_columns", None)   # как в Jupyter: таблица переносится, а не обрезается
    pd.set_option("display.width", 80)

    def flush():
        text = state["buf"].getvalue()
        if text:
            outputs.append({"output_type": "stream", "name": "stdout", "text": _lines(text)})
            state["buf"].seek(0)
            state["buf"].truncate()

    def display(*objs):
        flush()
        for o in objs:
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {k: _lines(v) for k, v in _rich(o).items()}})

    ns["display"] = display
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        outputs.clear()
        src = "".join(cell["source"])
        buf = state["buf"] = io.StringIO()
        plt.close("all")
        try:
            tree = ast.parse(src)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr) and not src.rstrip().endswith(";"):
                last = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, f"<cell {count}>", "exec"), ns)
                val = eval(compile(last, f"<cell {count}>", "eval"), ns) if last else None
        except Exception:  # noqa: BLE001
            print(f"Ошибка в ячейке {count}:\n{src[:300]}", file=sys.stderr)
            traceback.print_exc()
            raise SystemExit(1)
        flush()   # stdout и display() идут в том порядке, в каком вызывались
        result = list(outputs)
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", bbox_inches="tight")
            result.append({"output_type": "display_data", "metadata": {},
                           "data": {"image/png": base64.b64encode(b.getvalue()).decode(),
                                    "text/plain": [f"<Figure size {int(fig.get_figwidth()*fig.dpi)}x{int(fig.get_figheight()*fig.dpi)}>"]}})
        plt.close("all")
        if val is not None and not isinstance(val, matplotlib.figure.Figure):
            result.append({"output_type": "execute_result", "execution_count": count, "metadata": {},
                           "data": {k: _lines(v) for k, v in _rich(val).items()}})
        cell["outputs"] = result
        cell["execution_count"] = count
        print(f"ячейка {count} ок", file=sys.stderr)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
