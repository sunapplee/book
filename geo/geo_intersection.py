# Пересечение районов с областью: геометрии, площади, доли, карта.
# pip install geopandas matplotlib numpy

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon


# 1. Область задана в lon/lat. Для всех площадей используем одну UTM-проекцию.
muns = gpd.read_file("data/municipalities.geojson")
crs = muns.estimate_utm_crs()
muns = muns.to_crs(crs)
area = gpd.GeoSeries([
    Polygon([(40, 45), (40, 46), (41, 46), (41, 45)])
], crs=4326).to_crs(crs)


# 2. intersection создаёт геометрии; .area получает из них числа.
# sjoin только связывает объекты и не обрезает полигон района.
parts = muns.copy()
parts["geometry"] = muns.intersection(area.iloc[0])
parts["area_km2"] = parts.area / 1_000_000
parts["district_km2"] = muns.area / 1_000_000
parts = parts[parts["area_km2"] > 0].copy()
parts["share_pct"] = 100 * parts["area_km2"] / parts["district_km2"]
print(parts[["r_reg_name", "area_km2", "share_pct"]])

# Сумма частей и площадь покрытой области: перекрытия районов дадут двойной счёт.
sum_km2 = parts["area_km2"].sum()
covered_km2 = area.intersection(muns.geometry.union_all()).area.sum() / 1_000_000
print("Сумма частей / покрытая площадь:", sum_km2, covered_km2)
print("Площади совпадают:", np.isclose(sum_km2, covered_km2, rtol=1e-6, atol=1e-6))


# 3. Геометрии сохранены: можно показать именно обрезанные части районов.
fig, ax = plt.subplots(figsize=(10, 7))
if parts.empty:
    print("Область не пересекает районы.")
else:
    parts.plot(column="share_pct", cmap="YlOrRd", legend=True, ax=ax)
muns.boundary.plot(ax=ax, color="gray", linewidth=0.5)
area.boundary.plot(ax=ax, color="blue", linewidth=2)
ax.set_title("Доля площади района внутри области, %")
ax.set_axis_off()
plt.show()

# parts.to_file("intersections.gpkg", driver="GPKG")
