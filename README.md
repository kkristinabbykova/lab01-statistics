# Lab01: исследование датасета и датчик случайных чисел

Прикладная статистика, осень 2026, трек «Инфохимия» (lab.ic.01). Сейчас готов **спринт 1**: всё до датчика случайных чисел.

Основной файл работы: **`lab01.ipynb`**. Структура разделов по условию лабораторной:

```
# Lab01 - sprint 1
## Dataset: Wine Quality (белые вина)
## Random Variable 01 (pH)
## Random Variable 02 (фиксированная кислотность)
## Random Variable 03 (спирт)
   ### Description · EDA & Distribution · Independence & Estimations
   ### Normalization · Sample vs Population · Hypotheses · Resume
## Lab resume
   ### Итоговые метрики · гипотезы · выводы · литература · cat_mark_table
# Lab01 - sprint 2   (будет)
```

## Что где лежит

| Путь | Что это |
|---|---|
| `lab01.ipynb` | исследование, выполненный ноутбук с графиками |
| `labstat/` | личный пакет: громоздкий код вынесен сюда, чтобы ноутбук читался |
| `tests/` | автотесты пакета (pytest) |
| `figures/` | все графики ноутбука в PNG |
| `data/` | исходные CSV Wine Quality (белые и красные) |
| `ai.md` | использование нейросети: модель, промпты, что сделано моделью |

## Как запустить

```bash
pip install -r requirements.txt
python -m pytest -q          # 37 тестов, около 10 с
jupyter notebook lab01.ipynb # выполнить сверху вниз
```

Ноутбук запускается из корня репозитория. В Colab сначала клонировать репозиторий и перейти в его папку, чтобы импортировался `labstat`. Если папки `data/` нет, данные скачаются с зеркала на GitHub, контрольная сумма SHA-256 проверяется.

## Данные

Wine Quality: Cortez P., Cerdeira A., Almeida F., Matos T., Reis J. Modeling wine preferences by data mining from physicochemical properties // Decision Support Systems. 2009. Vol. 47, № 4. P. 547–553. UCI Machine Learning Repository, DOI 10.24432/C56S3T. Файлы взяты с зеркала github.com/zygmuntz/wine-quality, размер совпадает с оригиналом UCI (264 426 и 84 199 байт).

## Авторы

Нефедьева Антонина Сергеевна, ИСУ 502335; напарник по журналу: ИСУ 501322.
