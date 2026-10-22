# Таблицы из PDF. Источник: Atomskills2026_preparing/13.03.2026/C10_M10.ipynb
# pip install pymupdf camelot-py pandas
# В исходнике SAMPLE2 не задан, SAMPLE3/4 — абсолютные пути. Здесь один SAMPLE.

import fitz
import camelot

SAMPLE = "text.pdf"

# 1. PyMuPDF: таблицы первой страницы, списки строк / DataFrame / Markdown
with fitz.open(SAMPLE) as pdf:
    page = pdf[0]
    tables = page.find_tables().tables
    print(tables[0].extract())
    df = tables[0].to_pandas()
    print(df)
    print(tables[0].to_markdown())

# 2. Camelot: найденные таблицы документа
tables = camelot.read_pdf(SAMPLE, split_text=True)
print(tables[0].data)
print(tables[0].df)
# Как в исходнике, tables[0] предполагает, что найдена хотя бы одна таблица.
