# Сохранение GeoTIFF после обрезки: значения + геопривязка + NoData.
# По примеру задания 9 в geo_visualization_remaining.ipynb.
# Пути — относительно папки geo_practice.
# pip install rasterio shapely

import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon


# 1. В этом DEM CRS — EPSG:4326, поэтому полигон задан в lon/lat.
# Для другого растра сначала приведите полигон к его CRS.
polygon = Polygon([(38.9, 45.4), (39.5, 45.4), (39.5, 46), (38.9, 46)])
with rasterio.open("data/dem/DEM90_N45-50_E030-045.tif") as src:
    crop, crop_transform = mask(src, [polygon], crop=True, nodata=src.nodata)
    profile = src.profile.copy()
    scales, offsets = src.scales, src.offsets


# 2. crop имеет форму (каналы, высота, ширина).
# После обрезки меняются размеры и transform; CRS и NoData берём из profile.
profile.update(
    driver="GTiff",
    count=crop.shape[0],
    height=crop.shape[1],
    width=crop.shape[2],
    dtype=crop.dtype.name,
    transform=crop_transform,
    compress="deflate",
)
with rasterio.open("cropped_dem.tif", "w", **profile) as dst:
    dst.write(crop)
    dst.scales, dst.offsets = scales, offsets


# 3. Повторно открываем файл: проверяем данные и метаданные.
with rasterio.open("cropped_dem.tif") as saved:
    print("Форма:", saved.read().shape)
    print("CRS:", saved.crs, "Границы:", saved.bounds)
    print("Transform:", saved.transform, "NoData:", saved.nodata)
    values = saved.read(1, masked=True)  # NoData исключается из статистики.
    print("Минимум / максимум / среднее:", values.min(), values.max(), values.mean())

# Для массива без обрезки transform можно оставить исходным.
# Для одного индекса, например NDVI: count=1, dtype="float32" и подходящий NoData.
# Одноканальный массив (H, W) записывается как dst.write(array, 1).
# Нельзя просто скопировать count/dtype/NoData многоканального снимка в индекс.
