"""Собрать файлы для marimo-ноутбука и сайта.

    python tools/build_site.py

1. Копирует код датчиков и данные в ktvd/public/: в браузере (WASM) ноутбук берёт их оттуда.
2. Если установлен marimo, экспортирует ktvd/datchik.py в интерактивный сайт docs/datchik/
   (marimo export html-wasm). Лендинг docs/index.html ведёт на него.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "ktvd" / "public"


def main():
    PUB.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "labrng" / "generators.py", PUB / "generators.py")
    shutil.copy(ROOT / "data" / "winequality-white.csv", PUB / "winequality-white.csv")
    print("public/ обновлена")
    if shutil.which("marimo") is None:
        print("marimo не установлен: pip install marimo, затем запустить скрипт ещё раз для экспорта сайта")
        return 0
    out = ROOT / "docs" / "datchik"
    cmd = ["marimo", "export", "html-wasm", str(ROOT / "ktvd" / "datchik.py"), "-o", str(out), "--mode", "run", "-f"]
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
