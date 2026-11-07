"""Собрать датасет из папки шрифтов.

    python -m labfont.collect fonts/ data/fonts.csv

В папке fonts/ лежат zip-архивы из Google Fonts («Get font → Download all»)
или распакованные .ttf/.otf. Если разложить их по подпапкам sans/, serif/,
display/, handwriting/, mono/, имя подпапки попадёт в столбец category.
"""
import sys
from pathlib import Path

from .measure import measure_collection


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, out = Path(argv[0]), Path(argv[1])
    df, skipped = measure_collection(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Измерено семейств: {len(df)}, пропущено файлов: {len(skipped)} -> {out}")
    for name, why in skipped:
        print(f"  пропуск {name}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
