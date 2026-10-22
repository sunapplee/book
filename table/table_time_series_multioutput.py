# Прогноз сразу на 7 дней (или 24 часа) через CatBoost MultiRMSE.
# По мотивам atomskills2026_solution/C/С3_MC.ipynb.
# pip install pandas numpy scikit-learn catboost

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Регулярные ряды: date, object_id, value. Для одного ряда object_id = "series".
# Прогноз строим после получения value в момент t: цели — t+1 ... t+horizon.
freq, horizon = "D", 7  # Для почасовых данных: "h", 24.
step = pd.Timedelta(1, unit=freq)
df = pd.read_csv("data/time_series.csv", parse_dates=["date"])
df = df.sort_values(["object_id", "date"]).reset_index(drop=True)
df["object_id"] = df.object_id.astype(str)
if df.duplicated(["object_id", "date"]).any():
    raise ValueError("Есть дубликаты объекта и даты.")
g = df.groupby("object_id", sort=False)
gaps = g.date.diff().dropna()
if not gaps.eq(step).all():
    raise ValueError("Сначала восстановите регулярный календарь каждого ряда.")


# 2. Признаки используют только значения, известные к моменту t.
features = ["object_id", "value"]
for lag in (1, 2, 7):
    col = f"lag_{lag}"
    df[col] = g.value.shift(lag)
    features.append(col)
df["mean_7"] = g.value.transform(lambda s: s.rolling(7).mean())
df["month"], df["dow"], df["hour"] = df.date.dt.month, df.date.dt.dayofweek, df.date.dt.hour
features += ["mean_7", "month", "dow", "hour"]

targets = [f"step_{i}" for i in range(1, horizon + 1)]
for i, col in enumerate(targets, start=1):
    df[col] = g.value.shift(-i)
data = df.dropna(subset=features + targets).copy()


# 3. Разделение по общим датам, а не по перемешанным строкам.
dates = data.date.sort_values().unique()
if len(dates) < 5 * horizon:
    raise ValueError("Слишком короткая история для трёх выборок и заданного горизонта.")
val_start, test_start = pd.Timestamp(dates[int(len(dates) * .7)]), pd.Timestamp(dates[int(len(dates) * .85)])
target_end = data.date + horizon * step
# Убираем окна, чьи целевые даты пересекают границу следующей выборки.
train = data[target_end < val_start]
val = data[(data.date >= val_start) & (target_end < test_start)]
test = data[data.date >= test_start]
if min(len(train), len(val), len(test)) == 0:
    raise ValueError("После разделения есть пустая выборка; увеличьте историю.")

model = CatBoostRegressor(
    loss_function="MultiRMSE", iterations=300, depth=5,
    learning_rate=.05, random_seed=42, verbose=False, allow_writing_files=False,
)
model.fit(
    train[features], train[targets], cat_features=["object_id"],
    eval_set=(val[features], val[targets]), early_stopping_rounds=30,
)


# 4. Отдельные метрики для каждого шага. Базовый прогноз — последнее значение.
pred = model.predict(test[features])
metrics = []
for i, col in enumerate(targets):
    y = test[col]
    metrics.append({
        "step": i + 1,
        "MAE": mean_absolute_error(y, pred[:, i]),
        "RMSE": np.sqrt(mean_squared_error(y, pred[:, i])),
        "MAE_last_value": mean_absolute_error(y, test.value),
    })
print(pd.DataFrame(metrics).set_index("step").round(3))
# Это скользящая оценка: для каждого t уже известно фактическое value(t).


# 5. Прогноз от последнего наблюдения каждого объекта.
latest = df.groupby("object_id", sort=False).tail(1).dropna(subset=features)
forecast = pd.DataFrame(model.predict(latest[features]), columns=targets)
forecast.insert(0, "object_id", latest.object_id.to_numpy())
forecast.insert(1, "origin_date", latest.date.to_numpy())
print(forecast)
# forecast.to_csv("forecast.csv", index=False)
# model.save_model("multioutput.cbm")
