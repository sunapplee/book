# Высота рельефа для точек из локального DEM (GeoTIFF или BIL).
# По мотивам as26/top/Elevation.ipynb.
# pip install rasterio pandas numpy

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


# 1. Точки в WGS84. Порядок координат: долгота (lon), широта (lat).
points = pd.read_csv("data/points.csv")  # Колонки: lon, lat

# 2. Переводим координаты в CRS растра и читаем высоты.
# DEM целиком в память не загружается. Для BIL нужны сопутствующие файлы.
with rasterio.open("data/elevation.tif") as dem:
    if dem.crs is None:
        raise ValueError("У DEM не указана система координат.")
    print("CRS:", dem.crs, "Разрешение:", dem.res, "NoData:", dem.nodata)

    x, y = transform("EPSG:4326", dem.crs, points.lon.tolist(), points.lat.tolist())
    heights = np.ma.vstack(list(dem.sample(zip(x, y), indexes=1, masked=True)))
    # masked=True: NoData и точки за пределами покрытия станут NaN.
    heights = heights.astype("float64").filled(np.nan).ravel()
    points["elevation"] = heights * dem.scales[0] + dem.offsets[0]

# 3. Результат — признак высоты для каждой исходной точки.
# Единицы и вертикальную систему высот проверяем в описании DEM.
print(points.head())
print("Без высоты:", points.elevation.isna().sum())
# points.to_csv("points_with_elevation.csv", index=False)
