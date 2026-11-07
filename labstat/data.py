"""Загрузка датасета Wine Quality (Cortez et al., 2009).

Файлы лежат в data/. Если ноутбук открыт в Colab без папки data/,
данные скачиваются с зеркала на GitHub. Контрольная сумма SHA-256
проверяет, что файл тот же самый, что и при выполнении работы.
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MIRROR = "https://raw.githubusercontent.com/zygmuntz/wine-quality/master/winequality/winequality-{kind}.csv"

SHA256 = {
    "white": "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836",
    "red": "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e",
}

# Короткие имена колонок, чтобы удобно писать df.ph, df.fixed_acidity
RENAME = {
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "chlorides": "chlorides",
    "free sulfur dioxide": "free_so2",
    "total sulfur dioxide": "total_so2",
    "density": "density",
    "pH": "ph",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
    "quality": "quality",
}

UNITS = {
    "fixed_acidity": "г/дм³ (винная к-та)",
    "volatile_acidity": "г/дм³ (уксусная к-та)",
    "citric_acid": "г/дм³",
    "residual_sugar": "г/дм³",
    "chlorides": "г/дм³ (NaCl)",
    "free_so2": "мг/дм³",
    "total_so2": "мг/дм³",
    "density": "г/см³",
    "ph": "ед. pH",
    "sulphates": "г/дм³ (K₂SO₄)",
    "alcohol": "% об.",
    "quality": "балл 0–10",
}


def _read_bytes(kind: str) -> bytes:
    local = DATA_DIR / f"winequality-{kind}.csv"
    if local.exists():
        return local.read_bytes()
    with urllib.request.urlopen(MIRROR.format(kind=kind), timeout=60) as r:
        return r.read()


def load_wine(kind: str = "white", check_hash: bool = True) -> pd.DataFrame:
    """Вернуть таблицу вин: kind = 'white' (N=4898) или 'red' (N=1599)."""
    if kind not in SHA256:
        raise ValueError("kind должен быть 'white' или 'red'")
    raw = _read_bytes(kind)
    if check_hash and hashlib.sha256(raw).hexdigest() != SHA256[kind]:
        raise ValueError(f"Файл winequality-{kind}.csv отличается от исходного")
    df = pd.read_csv(io.BytesIO(raw), sep=";")
    return df.rename(columns=RENAME)
