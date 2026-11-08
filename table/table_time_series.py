#!/usr/bin/env python
# coding: utf-8

# # Временные ряды
# 
# Это частая задача на чемпионатах.
# 
# Для упрощения тут проведем все обработку и построение модели.

# ## Импорт библиотек

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ## Импорт данных

# In[5]:


df = pd.read_csv('data/Air_Quality.csv')
df.head()


# Из набора данных будем анализировать изменение параметра AQI в городе Бразилиа.

# In[6]:


ts = df[df['City'] == 'Brasilia'][['Date', 'AQI']]

# Приводим колонку даты к типу datetime и делаем дату признаком
ts['Date'] = pd.to_datetime(ts['Date'])
ts = ts.set_index('Date')
ts.head(3)


# Если в данных есть пропуски, нужно провести интерполяцию.

# In[7]:


ts = ts.interpolate()

ts.head(3)


# In[8]:


fig, ax = plt.subplots(1, 1, figsize=(10, 3))

ts[:1000].plot(ax=ax)


# ## Декомпозиция временного ряда
# Разложим временной ряд на:
# - тренд
# - сезонную компоненту
# - остатки

# In[9]:


decomposition = STL(ts['AQI'], period=24)
res = decomposition.fit()
res


# In[10]:


trend = res.trend
trend.head(3)


# In[11]:


seasonal = res.seasonal
seasonal.head(3)


# In[12]:


resid = res.resid
resid.head(3)


# In[13]:


res.plot()
plt.show()


# ## Модель для прогноза по временному ряду
# 
# Классическая постановка задачи - прогноз целевого признака на определенный срок. Будем строить прогноз AQI на 2 года.
# 
# Можно использовать множество моделей для предсказания временных рядов. Например: SARIMAX, ARIMA и др.
# 
# 
# Хорошим методом будет подход на основе календарных признаков для обучения модели ML. Если неизвестно, какие еще признаки могут дать - это выход из трудного положения.

# Выделяем из даты год, месяц, день, час в признаки для обучения

# In[14]:


ts = ts.reset_index()
# ts['year'] = ts['Date'].dt.year
ts['month'] = ts['Date'].dt.month
ts['day'] = ts['Date'].dt.day
ts['hour'] = ts['Date'].dt.hour

X = ts[['hour', 'day', 'month']]
y = ts['AQI']

X.head(3)


# После того, как мы получили табличные данные, может использовать любую модель. Например, линейную регрессию.

# In[15]:


model = LinearRegression()
model.fit(X, y)


# ## Генерируем будущий временной ряд

# In[16]:


# Последняя дата в исходных данных
last_date = ts['Date'].max()

# Создаём будущие даты на 2 года вперёд (почасово)
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(hours=1),
    periods=2 * 365 * 24,
    freq='h'
)

future_df = pd.DataFrame({'Date': future_dates})


# In[17]:


# future_df['year'] = future_df['Date'].dt.year
future_df['month'] = future_df['Date'].dt.month
future_df['day'] = future_df['Date'].dt.day
future_df['hour'] = future_df['Date'].dt.hour

X_test = future_df[['hour', 'day', 'month']]

X_test.head(3)


# ## Инференс

# In[18]:


y_pred = model.predict(X_test)


# ## Визуализация прогноза
# 
# Чтобы график выглядел красивым, отсечем часть исторических данных

# In[19]:


plt.figure(figsize=(20, 3))

sns.lineplot(data=ts.iloc[7000:], x='Date', y='AQI', label='Исторические данные')
sns.lineplot(x=future_df['Date'].iloc[:200], y=y_pred[:200], color='red', label='Прогноз')
plt.legend()
plt.title('Прогноз целевой переменной на определенный срок');


# Такой прогноз выходит не очень красивым, но сильно экономит время. Для получения баллов за простой прогноз - отлично.

# Но если нужно построить качественный прогноз, используем модель SARIMAX.

# ## SARIMAX

# In[20]:


# Приводим данные к знакомому формату
ts = ts.set_index('Date')
ts = ts['AQI']
ts.head()


# In[21]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    y,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 24),
)

results = model.fit()
print(results.summary())


# In[22]:


# Прогноз на 2 года вперёд
n_steps = 2 * 365 * 24  # 2 года, почасово, тк данные по часу

forecast = results.get_forecast(steps=n_steps)

forecast_mean = forecast.predicted_mean

future_index = pd.date_range(
    start=ts.index[-1] + pd.Timedelta(hours=1),
    periods=n_steps,
    freq='h'
)

forecast_mean.index = future_index
forecast_mean.head()


# In[23]:


plt.figure(figsize=(20, 3))

ts = ts.reset_index()
forecast_mean.reset_index()

sns.lineplot(data=ts.iloc[7000:], x='Date', y='AQI', label='Исторические данные')
sns.lineplot(x=forecast_mean.index[:500], y=forecast_mean[:500], color='red', label='Прогноз')
plt.legend()
plt.title('Прогноз целевой переменной на определенный срок');


# Модель уловила сезонность и тренд, и сделала прогноз лучше линейной регрессии.

# Лаги

df_test = pd.DataFrame(np.arange(100), columns=["value"])

for i in range(10):
    df_test[f'lag_{i}'] = df_test["value"].shift(i)

df_test
