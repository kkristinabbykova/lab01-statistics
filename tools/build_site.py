"""Собрать файлы для marimo-ноутбука и сайта.

    python tools/build_site.py

1. Копирует код датчиков и данные в ktvd/public/: в браузере (WASM) ноутбук берёт их оттуда.
2. Если установлен marimo, экспортирует ktvd/datchik.py в интерактивный сайт docs/datchik/
   (marimo export html-wasm).
3. Собирает лендинг docs/index.html (tools/build_landing.py), он ведёт на датчик.
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
    code = 0
    if shutil.which("marimo") is None:
        print("marimo не установлен: pip install marimo, затем запустить скрипт ещё раз для экспорта датчика")
    else:
        out = ROOT / "docs" / "datchik"
        cmd = ["marimo", "export", "html-wasm", str(ROOT / "ktvd" / "datchik.py"), "-o", str(out), "--mode", "run", "-f"]
        print(" ".join(cmd))
        code = subprocess.call(cmd)
    subprocess.call([sys.executable, str(ROOT / "tools" / "build_landing.py")])   # лендинг docs/index.html
    return code


if __name__ == "__main__":
    sys.exit(main())
