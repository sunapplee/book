# Кластеризация временных рядов: статистики, K-Shape, DTW.
# Источник: HT2026/notebooks/ts_clustering.ipynb
# pip install pandas numpy matplotlib scikit-learn tslearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import KShape, TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


# 1. Данные: объект, дата, значение. Один объект = один временной ряд.
df = pd.read_csv("data/time_series.csv")  # Колонки: date, object_id, value
df["date"] = pd.to_datetime(df["date"])

series = df.pivot_table(index="date", columns="object_id", values="value", aggfunc="mean")
series = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="D"))
series = series.interpolate(method="time", limit_area="inside")
# Оставляем полные непостоянные ряды на общем календаре.
series = series.dropna(axis=1)
series = series.loc[:, series.std() > 0].T
if len(series) < 3:
    raise ValueError("Нужно хотя бы 3 полных непостоянных ряда на общем календаре.")


# 2. Статистики + KMeans: группируем по уровню и разбросу значений.
stats = pd.DataFrame({
    "mean": series.mean(axis=1), "std": series.std(axis=1),
    "min": series.min(axis=1), "max": series.max(axis=1),
})
X_stats = StandardScaler().fit_transform(stats)

# Метод локтя и силуэт. Число кластеров выбираем по силуэту.
scores = []
for k in range(2, min(10, len(series) - 1) + 1):
    model = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_stats)
    score = silhouette_score(X_stats, labels) if 1 < len(set(labels)) < len(labels) else np.nan
    scores.append({"k": k, "inertia": model.inertia_, "silhouette": score})

scores = pd.DataFrame(scores).set_index("k")
print(scores)
scores.plot(subplots=True, marker="o", figsize=(8, 5), grid=True)
plt.tight_layout()
plt.show()
if scores.silhouette.notna().sum() == 0:
    raise ValueError("Не удалось получить разбиение для оценки силуэта.")
n_clusters = int(scores.silhouette.idxmax())

kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_stats = kmeans.fit_predict(X_stats)


# 3. K-Shape: группируем по форме, допуская временной сдвиг.
# Нормализация каждого ряда убирает различия уровня и амплитуды.
# Форма массива: (число рядов, длина ряда, число признаков).
X = TimeSeriesScalerMeanVariance().fit_transform(series.to_numpy()[:, :, None])
kshape = KShape(n_clusters=n_clusters, random_state=42, n_init=3)
labels_kshape = kshape.fit_predict(X)


# 4. DTW + KMeans: допускаем локальное растяжение и сжатие времени.
# Для примера оставляем то же k; на своих данных подбираем его для каждого метода.
dtw = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42, n_init=2)
labels_dtw = dtw.fit_predict(X)

result = pd.DataFrame({
    "KMeans": labels_stats, "K-Shape": labels_kshape, "DTW": labels_dtw,
}, index=series.index)
print(result)
for name in result:
    print(name, result[name].value_counts().sort_index().to_dict())


# 5. Метрика DTW: силуэт считаем по матрице DTW-расстояний.
# Полная матрица дорогая: здесь пример для небольшого числа рядов.
distances = cdist_dtw(X)
np.fill_diagonal(distances, 0)
if 1 < len(set(labels_dtw)) < len(labels_dtw):
    print("DTW silhouette:", round(silhouette_score(distances, labels_dtw, metric="precomputed"), 3))
# Силуэт статистик и DTW рассчитан в разных пространствах — напрямую их не сравниваем.


# 6. Графики формы рядов и центров кластеров.
for name, labels, centers in [
    ("KMeans", labels_stats, None),
    ("K-Shape", labels_kshape, kshape.cluster_centers_),
    ("DTW", labels_dtw, dtw.cluster_centers_),
]:
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 3), squeeze=False)
    for cluster, ax in enumerate(axes[0]):
        members = X[labels == cluster, :, 0]
        ax.plot(members.T, color="gray", alpha=0.3)
        if len(members):
            center = members.mean(axis=0) if centers is None else centers[cluster, :, 0]
            ax.plot(center, color="red", linewidth=2)
        ax.set_title(f"{name}: кластер {cluster}, n={len(members)}")
        ax.set_xlabel("Шаг времени")
    plt.tight_layout()
    plt.show()

# Средние статистики в исходных единицах помогают описать кластеры.
print(stats.groupby(labels_stats).mean())
# result.to_csv("time_series_clusters.csv")
