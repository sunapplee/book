# Геоданные: sjoin, точки внутри полигона, пространственная интерполяция.
# Источник: atomskills_2026/A/C1_MA.ipynb; пути — относительно исходной папки A.

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import contains_xy
from scipy.interpolate import griddata

Climate_data = pd.read_csv("../data/Climate_1995_2026.csv")
municipalities = gpd.read_file("../data/Krasnodar_Municipalities_GeoJson.txt").set_crs("EPSG:4326")

# 1. Координаты → геометрия; район ищем только для уникальных точек
Climate_data["geometry"] = gpd.points_from_xy(Climate_data["lon"], Climate_data["lat"])
Climate_data = gpd.GeoDataFrame(Climate_data, geometry="geometry", crs=4326)
unique_points = Climate_data.drop_duplicates(subset="geometry").geometry
points_gdf = gpd.GeoDataFrame(geometry=unique_points, crs=4326)
result = gpd.sjoin(points_gdf, municipalities, predicate="within", how="left").fillna(-1)
Climate_data = Climate_data.merge(result[["geometry", "r_reg_name"]], on="geometry").rename(
    columns={"r_reg_name": "municipality"}
)
# within не включает границу полигона; -1 — точка не попала ни в один район.
Climate_data = Climate_data[Climate_data["municipality"] != -1]

# 2. Подготовка климатических признаков перед пространственной интерполяцией
cols2interpolate = ["precipitation", "temp", "tmax", "tmin", "solar_radiation", "wind_speed", "wind_direction"]
for col in cols2interpolate:
    if col in Climate_data.columns:
        Climate_data[col] = Climate_data.groupby(["lat", "lon"])[col].transform(
            lambda x: x.interpolate(method="linear")
        )
        Climate_data[col] = Climate_data.groupby(["lat", "lon"])[col].transform(lambda x: x.ffill().bfill())

# 3. Для каждой пары «район — дата» создаём дополнительные точки
# Пороги исходной задачи: город — 10 записей, район — 30; добавление с запасом.
grouped = Climate_data.groupby(["municipality", "date"])
boundaries = municipalities.set_index("r_reg_name")["geometry"]
rows = []
for (municipality_name, date), sample in grouped:
    polygon = boundaries[municipality_name]
    if "город" in municipality_name and len(sample) >= 10:
        continue
    elif len(sample) >= 30:
        continue
    n_sample = 30 - len(sample) + 5
    minx, miny, maxx, maxy = polygon.bounds
    xs, ys = np.random.uniform(minx, maxx, 100), np.random.uniform(miny, maxy, 100)
    mask = contains_xy(polygon, xs, ys)
    inside_points = np.column_stack([xs[mask], ys[mask]])[:n_sample]

    # nearest — значения ближайшей исходной точки для всех климатических колонок
    interpolated = griddata(
        sample[["lon", "lat"]].values, sample[cols2interpolate].values,
        inside_points, method="nearest",
    )
    rows.append((np.column_stack([inside_points, interpolated]), date, municipality_name))

# 4. Новые точки → GeoDataFrame → объединение с исходными наблюдениями
# Как в исходнике: должны быть группы для расширения, иначе dfs пуст для concat.
dfs = []
for data, date, municipality in rows:
    df = pd.DataFrame(data, columns=["lon", "lat"] + cols2interpolate)
    df["date"], df["municipality"] = date, municipality
    dfs.append(df)
extra_df = pd.concat(dfs, ignore_index=True)
extra_df["geometry"] = gpd.points_from_xy(extra_df["lon"], extra_df["lat"])
extra_df = gpd.GeoDataFrame(extra_df, geometry="geometry", crs=4326)
Extended_climate_data = pd.concat([Climate_data, extra_df])
print(Extended_climate_data.shape)

# Ограничение исходника: 100 кандидатов не гарантируют нужного числа точек внутри
# полигона; полностью отсутствующие пары «район — дата» groupby не создаёт.
