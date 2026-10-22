# Полнота календаря, интерполяция, CCF.
# Источник: ViTya/lesson 1 — Визуализация временных рядов и не только/*.ipynb
# Путь data/... — относительно папки исходного урока.

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stattools
import statsmodels.graphics.tsaplots as tsaplots

df = pd.read_parquet("data/consumption.parquet")
df["date"] = pd.to_datetime(df["date"])

# 1. Проверка ключа и календаря: число строк не гарантирует полноту ряда
# pivot/reindex ниже требуют уникального ключа (территория, дата, категория).
print("Дубликаты:", df.duplicated(["territory_id", "date", "category"]).sum())
print("Территорий:", df.territory_id.nunique())
print(df.isna().sum())  # Не показывает отсутствующие строки!
full_dates = pd.date_range(df.date.min(), df.date.max(), freq="MS")
categories = df.category.unique()

# Проверяем каждую пару «месяц — категория», включая NaN в value
expected_pairs = pd.MultiIndex.from_product([full_dates, categories])
coverage = df.pivot(index="territory_id", columns=["date", "category"], values="value")
coverage = coverage.reindex(columns=expected_pairs)
complete = coverage.notna().all(axis=1)
print(pd.DataFrame({"rows": df.groupby("territory_id").size(), "complete": complete}))
unfull_data = df.loc[df.territory_id.isin(complete.index[~complete])]

# 2. Полный календарь только для неполных территорий
# Как в исходнике: этот блок рассчитан на наличие неполных рядов.
additional_records = []
for t_id in unfull_data.territory_id.unique():
    terr_data = unfull_data[unfull_data.territory_id == t_id]
    for col in categories:
        col_data = terr_data[terr_data.category == col].set_index("date").reindex(full_dates)
        col_data.index.name = "date"
        col_data["territory_id"], col_data["category"] = t_id, col
        additional_records.append(col_data)
additional_df = pd.concat(additional_records).reset_index()
print("Строк до/после:", len(unfull_data), len(additional_df))

# 3. Интерполяция внутри каждой территории и категории
# limit_area="inside": края ряда и полностью пустые категории остаются NaN.
# Для прогноза интерполяция с будущими значениями создаёт утечку.
interpolated_df = additional_df.copy().set_index("date")
interpolated_df["value"] = interpolated_df.groupby(["territory_id", "category"])["value"].transform(
    lambda values: values.interpolate(method="time", limit_area="inside")
)
interpolated_df = interpolated_df.reset_index()
print(pd.DataFrame({
    "before": [additional_df.value.isna().sum()],
    "after": [interpolated_df.value.isna().sum()],
}, index=["value"]))

# Исходные значения и оценки сохраняем раздельно
comparison = additional_df.copy()
comparison["interpolated_value"] = interpolated_df["value"]
print(comparison.loc[comparison.value.isna() & comparison.interpolated_value.notna()].head(10))

# 4. CCF — связь двух рядов при разных сдвигах
# Сначала проверить календарь: лаг — одно наблюдение, не обязательно месяц.
# Общий тренд/сезонность могут создавать связь; CCF не доказывает пользу для прогноза.
pivot_data = df[df.territory_id == 1].pivot(index="date", columns="category", values="value")
print(stattools.ccf(pivot_data["Маркетплейсы"], pivot_data["Продовольствие"]))
tsaplots.plot_ccf(pivot_data["Маркетплейсы"], pivot_data["Продовольствие"])
plt.show()
