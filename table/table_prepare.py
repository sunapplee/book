# Подготовка таблиц. Источник: atomskills_2026/A/C1_MA.ipynb
# Чтение Extended_climate_data.zip — из atomskills_2026/B/C2_MB.ipynb.
# Пути — относительно исходной папки A; ZIP создаётся в исходном проекте.

import os
import pandas as pd
import geopandas as gpd

# 1. Нестандартный Excel: убираем строку шапки, задаём имена, объединяем годы
cultures = [
    "Озимая пшеница", "Ячмень озимый", "Пшеница яровая", "Ячмень яровой",
    "Кукуруза", "Рис", "Сахарная свекла", "Подсолнечник", "Рапс озимый",
]
new_columns = ["municipality"] + [
    f"{metric} / {culture}" for culture in cultures
    for metric in ["Посевная площадь (га)", "Валовый сбор (ц)"]
]


def refactor_table(df):
    df = df.iloc[1:].copy()
    df.columns = new_columns
    return df.replace("-", 0).fillna(0)  # В исходной задаче: культура отсутствует


agri_dfs = []
for year in [2021, 2022, 2023, 2024]:
    df = refactor_table(pd.read_excel(f"../data/{year}_Agricultural_production.xlsx"))
    df["year"] = year
    agri_dfs.append(df)
Agricultural_production = pd.concat(agri_dfs)

# 2. Согласование названий районов по подстроке
# Исходная эвристика mun_name[2:] зависит от формата названий.
# При нескольких совпадениях выбирается первое; map оставит NaN для ненайденных.
municipalities = gpd.read_file("../data/Krasnodar_Municipalities_GeoJson.txt")
convert_mun = {}
for mun_name in Agricultural_production["municipality"]:
    mask = municipalities["r_reg_name"].str.contains(mun_name[2:])
    converted_name = municipalities.loc[mask, "r_reg_name"].values
    if len(converted_name):  # В исходнике if converted_name: неоднозначно для массива
        convert_mun[mun_name] = converted_name[0]
Agricultural_production["municipality"] = Agricultural_production["municipality"].map(convert_mun)

# 3. Объединение источников по общим ключам (inner merge)
Day_light_data = pd.read_csv("../data/Krasnodar_day_length_1995_2025.csv")
Day_light_data["year"] = pd.to_datetime(Day_light_data["date"]).dt.year
economic_df = pd.read_csv("../data/Krasnodar_economic_structure_1995_2025.csv").rename(columns={"municipality": "admin_center"})
gdp_per_capita_df = pd.read_csv("../data/Krasnodar_gdp_per_capita_1995_2025.csv").rename(columns={"municipality": "admin_center"})
infrastructure_df = pd.read_csv("../data/Krasnodar_infrastructure_1995_2025.csv").rename(columns={"municipality": "admin_center"})
Municipalities_data = (
    Day_light_data.merge(economic_df, on=["admin_center", "year"])
    .merge(gdp_per_capita_df, on=["admin_center", "year"])
    .merge(infrastructure_df, on=["admin_center", "year"])
)

# 4. Накопленная сумма и окно — по каждому району
Extended_climate_data = pd.read_csv("Extended_climate_data.zip")
Extended_climate_data = Extended_climate_data.sort_values(["municipality", "date"])
Extended_climate_data["cumulative_temp"] = Extended_climate_data.groupby("municipality")["temp"].cumsum()
# Ошибка трактовки в исходнике: rolling(14) — 14 строк, а не 14 дней!
# На одну дату приходится несколько точек; имя исходного признака сохранено.
Extended_climate_data["cumulative_2_week_temp"] = (
    Extended_climate_data.groupby("municipality")["temp"].rolling(14, min_periods=1)
    .sum().reset_index(drop=True, level=0)
)

# 5. Климатические агрегаты по району, году и месяцу
Extended_climate_data["date"] = pd.to_datetime(Extended_climate_data["date"])
Extended_climate_data["month"] = Extended_climate_data["date"].dt.month
Extended_climate_data["year"] = Extended_climate_data["date"].dt.year
climate_columns = ["temp", "tmin", "tmax", "precipitation", "solar_radiation", "wind_speed"]
Monthly_climate_data = Extended_climate_data.groupby(["municipality", "year", "month"]).agg(**{
    f"{col}_{stat}": (col, stat) for col in climate_columns for stat in ["min", "max", "mean"]
}).reset_index()
Monthly_climate_data.to_csv("Monthly_climate_data_1995_2025.csv", index=False)

# 6. Много CSV → общая таблица → суточное и месячное потребление
folder = "../data/Krasnodar_hourly_energy_consumption"
energy_df = pd.concat([pd.read_csv(f"{folder}/{name}") for name in os.listdir(folder)])
# Формат исходных CSV: первые две колонки — район и дата, остальные — часы.
energy_df["consumption_day"] = energy_df.iloc[:, 2:].sum(axis=1)
Daily_energy_consumption = energy_df[["municipality", "date", "consumption_day"]]
Daily_energy_consumption.to_csv("Daily_energy_consumption_1995_2025.csv", index=False)
energy_df["month"] = pd.to_datetime(energy_df["date"]).dt.to_period("M")
Monthly_energy_consumption = energy_df.groupby(["municipality", "month"]).agg(
    consumption_month=("consumption_day", "sum")
).reset_index()
Monthly_energy_consumption.to_csv("Monthly_energy_consumption _1995_2025.csv", index=False)
