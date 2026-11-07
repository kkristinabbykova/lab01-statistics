"""Разложить скачанные шрифты по папкам категорий для labfont.collect.

    python tools/unpack_fonts.py ПАПКА_С_ZIP_И_TTF fonts/

Google Fonts при каждом «Download all» отдаёт всю накопленную подборку, поэтому
архивы повторяют друг друга. Скрипт берёт каждое семейство один раз (по папке
семейства в архиве) и кладёт в fonts/<категория>/<семейство>/. Категории берутся
из data/font_categories.csv; семейства без категории перечисляются в конце,
их нужно дописать в CSV (категория по каталогу fonts.google.com).
"""
import csv
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(src, dst):
    src, dst = Path(src), Path(dst)
    cats = {r["family_folder"]: r["category"] for r in csv.DictReader(open(ROOT / "data" / "font_categories.csv", encoding="utf-8"))}
    seen, unknown = set(), set()
    for z in sorted(src.glob("*.zip")):
        try:
            zf = zipfile.ZipFile(z)
        except zipfile.BadZipFile:
            print("пропуск, не zip:", z.name)
            continue
        for n in zf.namelist():
            parts = n.split("/")
            if not n.lower().endswith((".ttf", ".otf")) or len(parts) < 2 or parts[0] in ("static", "__MACOSX"):
                continue
            fam = parts[0]
            if fam not in cats:
                unknown.add(fam)
                continue
            out = dst / cats[fam] / fam / "/".join(parts[1:])
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(n))
            seen.add(fam)
    for f in sorted(src.glob("*.[ot]tf")):   # отдельные файлы: семейство по имени в файле
        fam = next((k for k in cats if k.lower().replace("_", "-") in f.name.lower()), None)
        if fam is None:
            unknown.add(f.name)
            continue
        out = dst / cats[fam] / fam / f.name.split("-", 1)[-1]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(f.read_bytes())
        seen.add(fam)
    print(f"семейств разложено: {len(seen)}")
    if unknown:
        print("нет категории в data/font_categories.csv:", ", ".join(sorted(unknown)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
