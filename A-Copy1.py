#!/usr/bin/env python
# coding: utf-8

# In[33]:


original_set = pd.read_csv("data/Original_Set.csv")
original_set['date'] = pd.to_datetime(original_set['date'])


# In[34]:


original_set['geometry'] = gpd.GeoSeries.from_wkt(original_set['geometry'])
original_set_gdf = gpd.GeoDataFrame(original_set, geometry='geometry')

original_set_gdf.head()


# In[40]:


districs['geometry'] = gpd.GeoSeries.from_wkt(districs['geometry'])


# In[41]:


districs = gpd.GeoDataFrame(districs, geometry='geometry')


# In[42]:


districs['geometry'].dtype


# In[43]:


unique_geometry_gdf = original_set_gdf.drop_duplicates(subset=['geometry'])
vals = {}
for _, v in tqdm(unique_geometry_gdf.iterrows()):
    current_point = v["geometry"]
    for k, multipolygon_dist in districs.iterrows():
        if multipolygon_dist['geometry'].covers(current_point):
            vals[current_point] = multipolygon_dist['district_name']
            break
    else:
        vals[current_point]= -1


# In[44]:


original_set_gdf['district'] = original_set_gdf['geometry'].apply(lambda x: vals.get(x, -1))


# Всё работает
# 
# Некоторые точки могут не иметь района, потому что не принадлежат ни к одному мультипогону. Это, вероятно, выбросы. Точки, которые вне РО, находятся.

# In[45]:


out = original_set_gdf[original_set_gdf["district"].values == -1]["geometry"]
out.crs = "EPSG:4326"


# Точки где название района - это -1 - выброс

# Произведём интеграцую данных ЧС с привязкой к временным меткам и координатам

# Для этого загрузим файл с данной информацией

# In[48]:


emer["district_name"] = emer["district_name"].progress_apply(lambda x: orjson.loads(x).get("name"))


# Напишем ф-ю для проверки и обработки пропусков
# 
# Т.к. наши данные - это временные ряды, то заполнять пропуски будем интерполяцией в рамках отдельного района

# Посмотрим, что нам надо сперва заполнять (какие признаки)

# In[56]:


gdf.isna().sum()


# Что отсюда видем, что много пропусков про ЧС в тех точках, которые выбросы. Они там в данной задаче не интересны. Три фичи, которые имеют пропуски:
# - solar_radiation
# - wind_speed
# - wind_direction
# 
# Значит, эти 3 фичи и будем заполнять

# Теперь напишем ф-ю

# Т.к. пропуски находятся в начале ряда, то заполнять будем методом bfill. Это тут случай, когда начало ряда адекватно никак не заполнять
# 
# Для всего остального применим плайновую интерполяцию третьего порядка, которая будет лучше модель целевую зависимость относительно линейной и полиномиальной интерполяции

# In[57]:


def impute_nans(df, cols=("solar_radiation", "wind_speed", "wind_direction")):
    for col in cols:
        df.loc[:, col] = df[col].interpolate("linear").bfill()
    return df


# Отлично пропусков нет

# Напишем функцию обработки выбросов для каждого района
# 
# Выбросы надо искать геопространственные, т.е. точки, которые не принадлежат к границе РО. Ранее мы делали маппинг исходных точек к границам районов РО. Значит, заиспользуем эту информацию, чтоюы отсекать выбросы.

# Посмотрим, так ли это

# In[60]:


out = gdf[gdf["district"].values == -1]["geometry"]
out.crs = "EPSG:4326"


# In[61]:


out.shape


# In[62]:


out.explore(location=[47, 38.5], zoom_start=8)


# Как видно, это так. Значит, районы, где district == -1 - это выброс

# In[63]:


def drop_outliers(df):
    df_without_outliers = df[df["district"] != -1]
    return df_without_outliers


# Применим ф-ю

# In[64]:


gdf = drop_outliers(gdf)


# Проверим

# In[65]:


gdf[gdf["district"] == -1]


# Выбросов нет
# 
# Проверим, не удалил какие-то регионы

# In[66]:


gdf['district'].unique().shape


# Как видно, стало 44 региона, а изначально было 45
# 
# Значит, всё сделано верно

# Значит:
# - данные на наличе геопространственных выбросов проверил путём отрисовки своей гипотезы на карте
# - выполнил удаление out of bounds точек
# - Проверил (визуализация не требуется, т.к. будет пустая карта без выбросов. Района пока отрисовывать не требуется)

# Данные с высотами готовы

# Теперь каждой точке из Original_Set назначим высоту

# Для начала выделим уникальные точки, чтобы убрать избыточность будущих вычислений

# Созданим пространственную сетку, а затем сделаем интерполяцию по нужным параметрам
# На остальные параметры сделаем просто - affinity propagation

# In[245]:


get_ipython().run_cell_magic('time', '', 'lat_step = 3e-2\nlon_step = 3e-2\nlat_range = np.arange(gdf["lat"].min(), gdf["lat"].max() + lat_step, lat_step)\nlon_range = np.arange(gdf["lon"].min(), gdf["lon"].max() + lon_step, lon_step)\nlon_grid, lat_grid = np.meshgrid(lon_range, lat_range)\n\npoints = gdf[["lat", "lon"]].values\n\nvariables = ["temp", "tmax", "tmin"]\n\ndata = {"lat": lat_grid.ravel(), "lon": lon_grid.ravel()}\nfor var in tqdm(variables):\n    values = gdf[var].values\n    interp_values = griddata(points, values, (lat_grid, lon_grid), method=\'linear\')\n    data[var] = interp_values.ravel()\n\ninterpolated_data = pd.DataFrame(data)\n')


# In[225]:


interpolated_district = []
for _, v in tqdm(interpolated_data.iterrows()):
    current_point = Point(v['lon'], v['lat'])
    for k, region in districs.iterrows():
        if region.geometry.covers(current_point):
            interpolated_district.append(region['district_name'])
            break
    else:
        interpolated_district.append(-1)


# In[71]:


interpolated_data['district'] = interpolated_district


# In[76]:


interpolated_data = interpolated_data[interpolated_data['district'] != -1]
interpolated_data['geometry'] = interpolated_data.apply(lambda x: Point(x['lon'], x['lat']), axis=1)


# In[77]:


gdf_unique = gdf.drop_duplicates(subset=['district']).drop(columns=variables + ['lat', 'lon', 'geometry'])

interpolated_data_full = interpolated_data.merge(gdf_unique, on='district')
interpolated_data_full.shape


# In[78]:


Extended_Set = pd.concat([gdf, interpolated_data_full])


# In[79]:


Extended_Set.shape


# In[80]:


unique_regions_points = Extended_Set['geometry'].unique()

len(unique_regions_points)


# # Первичная визуализация 

# Выполним визуализацию

# Отберём уникальные точки для отрисоки

# In[81]:


unique_regions_points = gdf['geometry'].unique()

len(unique_regions_points)


# In[82]:


geo = gpd.GeoSeries(unique_regions_points)
geo.crs = 'EPSG:4326'
geo.crs


# In[83]:


rostov = gpd.read_file('data/Ростовская область_Rostov region.geojson')
rostov.sample(2)


# In[84]:


gpd.GeoSeries(rostov.union_all()).crs = 'EPSG:4326'


# In[85]:


unique_regions = gdf['district'].unique()
for uq in unique_regions:
    tmp = gdf[gdf["district"] == uq]
    name = tmp['district'].unique()[0]
    values = tmp['geometry']
    print(name, values.nunique())


# In[119]:


gdf[gdf['district'] == 'Zverevo urban district']


# In[131]:


rostov_boundary = gpd.GeoSeries(rostov.unary_union)
rostov_boundary.crs = 'EPSG:4326'
m = rostov_boundary.explore(style_kwds={'fill': None}, color='black', m=m)
geo.explore(m=m, color='red')

m


# Границы региона есть. Границы районов тоже есть

# На карте также явно видно, что три района не имеют точек

# Сделаем визулизацию для Extended_set

# In[86]:


unique_regions_points = Extended_Set['geometry'].unique()
unique_regions_points = gpd.GeoSeries(unique_regions_points)
len(unique_regions_points)


# In[218]:


rostov_boundary = gpd.GeoSeries(rostov.union_all())
rostov_boundary.crs = 'EPSG:4326'
m = rostov_boundary.explore(style_kwds={'fill': None}, color='black')
# unique_regions_points.explore(m=m, color='red')
m


# In[88]:


unique_regions = Extended_Set['district'].unique()
for uq in unique_regions:
    tmp = Extended_Set[Extended_Set["district"] == uq]
    name = tmp['district'].unique()[0]
    values = tmp['geometry']
    print(name, values.nunique())


# In[ ]:


Extended_Set


# In[ ]:


get_ipython().run_cell_magic('time', '', "import zipfile\n\n# Создание архива\nwith zipfile.ZipFile('example.zip', 'w') as myzip:\n    myzip.write(Extended_Set.to_csv())\n")


# In[ ]:


import shutil
import os

os.makedirs("temp_folder", exist_ok=True)
shutil.copy("data.csv", "temp_folder/data.csv")

shutil.make_archive("data_archive", "zip", root_dir="temp_folder")

