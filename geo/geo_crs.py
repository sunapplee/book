# CRS: назначение, преобразование, выбор проекции для площади.
# Во всех трёх примерах geo_crs / geo_intersection / geo_folium — один файл границ.
# pip install geopandas matplotlib

import geopandas as gpd
from shapely.geometry import Point


# 1. CRS исходного слоя. total_bounds: min_x, min_y, max_x, max_y.
muns = gpd.read_file("data/municipalities.geojson")
print("CRS:", muns.crs)
print("Границы:", muns.total_bounds)
print("Единицы:", muns.crs.axis_info[0].unit_name)


# 2. set_crs задаёт смысл уже имеющихся координат, числа не меняются.
# Исходные координаты известны: x = долгота, y = широта, в градусах WGS84.
point = gpd.GeoSeries([Point(39, 45)]).set_crs("EPSG:4326")
print("Исходная точка:", point.iloc[0])

# to_crs пересчитывает координаты. Это не переименование CRS.
metric_crs = muns.estimate_utm_crs()
metric_point = point.to_crs(metric_crs)
print("UTM:", metric_crs, metric_point.iloc[0])
print("Обратно в WGS84:", metric_point.to_crs(4326).iloc[0])


# 3. Одинаковые районы, две проекции, площади в км².
# EPSG:3857 подходит для веб-карты, но здесь сильно искажает площадь.
areas = muns[["r_reg_name"]].copy()
areas["web_mercator_km2"] = muns.to_crs(3857).area / 1_000_000
areas["utm_km2"] = muns.to_crs(metric_crs).area / 1_000_000
print(areas.head())

# estimate_utm_crs выбирает UTM-зону по расположению слоя.
# Для этих районов — EPSG:32637. Проверяем область применения выбранной CRS.
# Для большой территории через несколько зон или полярных областей выбор другой.
# Площадь в EPSG:4326 измеряется в градусах², а не в м².
# Не исправляем неизвестную исходную CRS простым назначением удобного EPSG.
