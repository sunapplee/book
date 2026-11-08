# Folium через GeoPandas: именованные слои, переключатель, HTML-заголовок.
# pip install geopandas folium mapclassify matplotlib

import folium
import geopandas as gpd


muns = gpd.read_file("data/municipalities.geojson")
muns["area_km2"] = muns.to_crs(muns.estimate_utm_crs()).area / 1_000_000
muns = muns.to_crs(4326)  # Веб-карта принимает геоданные в WGS84.

# 1. Первая explore создаёт карту, остальные добавляют слои через m=m.
# name — название слоя в переключателе, а не заголовок всей карты.
m = muns.explore(
    column="area_km2", cmap="YlOrRd", legend=True,
    name="Площадь районов, км²", tooltip=["r_reg_name", "area_km2"],
)
muns.explore(
    m=m, name="Границы районов", color="black",
    style_kwds={"fill": False, "weight": 1},
)
muns.iloc[[0]].explore(
    m=m, name="Выделенный район", color="red", show=True,
    style_kwds={"fill": False, "weight": 4},
)

# 2. Заголовок — отдельный HTML-элемент.
title_html = "<h3>Районы Краснодарского края</h3>"
m.get_root().html.add_child(folium.Element(title_html))

# 3. Добавляем управление после всех слоёв. show=False скрывает слой при открытии.
folium.LayerControl(collapsed=False).add_to(m)
m.save("geo_layers.html")
m  # Отображение в Jupyter.
