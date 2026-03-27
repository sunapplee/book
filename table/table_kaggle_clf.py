#!/usr/bin/env python
# coding: utf-8

# # Краткое описание задачи
# 
# В рамках соревнования Kaggle Tabular Playground Series 2026 необходимо разработать модель машинного обучения для прогнозирования вероятности сердечного заболевания на основе предоставленного синтетически сгенерированного табличного датасета. Данные получены с помощью глубокой генеративной модели и содержат различные клинические и демографические признаки, позволяющие смоделировать задачу классификации в медицинском контексте.
# 
# Цель проекта — построить алгоритм, способный по входным признакам пациента оценить вероятность наличия Heart Disease. Качество решения оценивается по метрике ROC AUC, что позволяет корректно измерять способность модели различать положительный и отрицательный классы при всех возможных порогах.

# <a id=0></a>
# ## Содержание

# * [Содержание](#0)
# * [Импорт библиотек](#1)
# * [1. Импорт данных](#1-1)
# * [2. Разведочный анализ данных](#2)
# * [3. Baseline-модель](#3)
#   * [3-1. Обучение Baseline-модели](#3-1)
#   * [3-2. Инференс Baseline-модели](#3-2)
# * [4. Предобработка данных](#4)
#   * [4-1. Генерация признаков](#4-1)
#   * [4-2. Отбор признаков](#4-2)
#   * [4.3 Промежуточное обучение модели](#4-3)
# * [5. Выбор оптимального алгоритма](#5)
#     * [5.1 CatBoost](#5-1)
#     * [5.2 LightGBM](#5-2)
#     * [5.3 XGBoost](#5-3)
#     * [5.4 Random Forest](#5-4)
#     * [5.5 Логистическая регрессия](#5-5)
#     * [5.6 Лучшие алгоритмы](#5-6)
# * [6. Подбор гиперпараметров](#6)
#     * [6.1 CatBoost](#6-1)
#     * [6.2 LightGBM](#6-2)
# * [7. Смешивание моделей](#7)

# <a id=1></a>
# ## Импорт библиотек

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

import optuna

import joblib

from sklearn.preprocessing import PolynomialFeatures

from sklearn.inspection import permutation_importance
import shap


# ###

# <a id=1-1></a>
# # 1. Импорт данных

# In[3]:


train = pd.read_csv('data/train.csv')

train.head()


# Heart Disease (Presence / Absence) кодируется так:
# 
# 0 → Absence (заболевания нет)
# 
# 1 → Presence (заболевание есть)

# In[4]:


train['Heart Disease'] = train['Heart Disease'].map(lambda x: 1 if x == 'Presence' else 0)

train.head(3)


# ###

# <a id=2></a>
# # 2. Разведочный анализ данных
# 
# Перед обучением первой модели проанализируем данные.

# In[5]:


print(f'Размер датасета {train.shape}')


# ### 2.1. Описательные статистики.

# In[6]:


train.describe()


# ### 2.2. Пропуски

# In[7]:


train.isna().sum()


# Пропусков не обнаружено.

# ### 2.3. Дубликаты

# In[8]:


train.duplicated().sum()


# Дупликатов не обнаружено.

# ### 2.4. Распределение

# In[9]:


fig, axs = plt.subplots(3, 5, figsize=(22, 10))

for col, ax in itertools.zip_longest(train.columns, axs.ravel()):

    # Удаляем лишние графики
    if col == None:
        fig.delaxes(ax)
        continue

    train[col].plot(ax=ax, kind='hist', bins=50)

    ax.set_title(col)


# Из графиков заметны детали:
# 1. ```FBS over 120, Chest pain type, Slope of ST, Number of vessels fluro``` имеют сильный дисбаланс классов
# 2. ```Cholesterol, ST depression``` имеет ассиметрию, длинный хвост

# ### 2.5. Выбросы

# In[10]:


fig, axs = plt.subplots(3, 5, figsize=(22, 10))

for col, ax in itertools.zip_longest(train.columns, axs.ravel()):

    # Удаляем лишние графики
    if col == None:
        fig.delaxes(ax)
        continue

    train[col].plot(ax=ax, kind='box')

    ax.set_title(col)


# Самые сильные выбросы имеют: ```Cholesterol, ST depression, Max HR, BP```.
# Удалять выбросы пока не будем, так как именно эти колонки могут быть факторами риска обнаружения забоелвания.

# ### 2.6. Корреляция

# In[11]:


plt.figure(figsize=(8, 6))
sns.heatmap(train.corr())


# Мультиколлениарности не обнаружено.
# 
# Cамые коррелирующие с целевой переменной признаки: ```Thallium, Chest pain type```

# ### 2.7. Дисбаланс классов 

# In[12]:


cat_columns = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro']

for col in cat_columns:
    print(f'\nАнализ классов для: ')
    print((train[col].value_counts(normalize=True) * 100).round(2))


# Из анализа видно, что некоторые признаки имеют сильный дисбаланс.

# <a id=3></a>
# # 3. Baseline-модель

# Для начального решения удалим колонку ```id```, разделим данные на тренирочную и тстовую выборку в пропорции ```80/20```.

# In[13]:


X = train.drop(columns=['id', 'Heart Disease'])
y = train['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.head()


# В качестве baseline-модели используем ```логистическую регрессию```.

# <a id=3-1></a>
# ## 3.1 Обучение Baseline-модели

# In[14]:


baseline_model = LogisticRegression(max_iter=500)

baseline_model.fit(X_train, y_train)

y_pred_proba = baseline_model.predict_proba(X_test)[:, 1]

y_pred_proba


# <a id=3-2></a>
# ## 3.2 Инференс baseline-модели

# Основная метрика соревнования - ```ROC-AUC```. Для подсчета на нашей тестовой выборке используем ```roc_auc_score``` из ```sklearn```.

# In[15]:


print(f'ROC AUC score для LogisticRegression: {roc_auc_score(y_test, y_pred_proba):.2f}')


# Отличная метрика для тестовой выборки, перейдем к сохранению инференса для валидационной основной выборке.

# In[16]:


test = pd.read_csv('data/test.csv')
test = test.drop(columns=['id'])

sample = pd.read_csv('data/sample_submission.csv')


# In[17]:


test_y_pred = baseline_model.predict_proba(test)[:, 1]

sample['Heart Disease'] = test_y_pred

sample.to_csv('outputs/baseline.csv', index=False)


# Для ```baseline-модели``` получили отличных результат, предобработаем данные, чтобы получить результат еще лучше! 

# <a id=4></a>
# # 4. Предобработка данных

# <a id=4-1></a>
# ## 4.1 Генерация признаков

# Для автогенерации признаков нам могла бы подойти библиотека ```feature-tools```. Однако мы имеет только 1 таблицу, и построить междутабличных отношений, на основании которых могли бы генерировать признаки, мы **не можем**. 
# 
# Поэтому возьмем метод ```PolynomialFeatures``` из библиотеки ```sklearn``` для создания комбинации фич. 

# In[33]:


# Инициализируем метод
poly = PolynomialFeatures(2)
# Генерируем фичи
X = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names_out())


# In[34]:


print(X['1'].unique())
# Признак "1" - является константной, поэтому удаляем ее

X = X.drop(columns=['1'])


# In[35]:


print(f'Новый размер датасета: {X.shape}')
X.head()


# Мы получили более сотни новых признаков. Для обучения классический модели ML такое количество признаков **слишком большое**. Произведем отбор признаков.

# <a id=4-2></a>
# ## 4.2 Отбор признаков

# Для оптимизации датасета и улучшения его качества, нам нужно оставить самые информативные признаки.
# 
# В работе лучше использовать несколько подходов, так результат будет объективнее. Поэтому оценим важность признаков можем с помощью **статистических методов** и **методов машинного обучения**.

# ### Статистический метод

# Воспользуемся **коэффициентом корреляции Пирсона**. Будем оценивать корреляцию каждого признака с целевой переменной.
# 
# Значение ```1``` означает идеальную положительную линейную связь с целевой переменной,
# 
# Значение ```-1``` — идеальную отрицательную,
# 
# Значение ```0``` — отсутствие линейной зависимости.
# 
# Такой подход позволит оценить важность с точки зрения статистики.

# Но сначала рассмотрим признаки на предмет мультиколлениарности.

# #### Мультиколлениарность

# In[21]:


# Добавляем в датасет для анализа целевую переменную
data_for_analysis = X.copy()

# Для ускорения процесса, отберем 20000 случайных строк
# Так мы сможем гарантировать, что в график вошли разные части данных 
corr_matrix = data_for_analysis.sample(20000).corr().abs()

print('Названия колонок - признаки, названия строк - признаки, значения - их коэффициент Пирсона.')

# Отобразим часть корреляций, для улучшения читаемость
corr_matrix.iloc[:10, :10]


# In[22]:


plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Корреляционная матрица")
plt.show()


# Агрегация значений и нахождение мультиколлениарных признаков: 

# In[23]:


# Получаем верхний треугольник (чтобы не учитывать дубликаты)
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Находим признаки, у которых ЕСТЬ хотя бы одна корреляция > 0.95 с другими признаками
high_corr_features = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > 0.95)]

print('Кол-во признаков, коррелирующиx между собой:', len(high_corr_features))


# #### Корреляция с целевой переменной

# Оценим корреляцию (взаимосвязь) каждого признака с целевой переменной.

# In[24]:


data_for_analysis['target'] = y

# Считаем коррелияцию вместе с целевой переменной
corr_with_target = data_for_analysis.corr().abs()


# In[25]:


# Сохраняем признаки, у которых корреляция < 0.3
low_corr_target = corr_with_target.loc[:, corr_with_target['target'] < 0.3].columns


# ### Методы машинного обучения

# Для дальнейшнего анализа мы будем оценивать взаимосвязь признаков по инференсу модели, поэтому обучим простую модель RandomForest для классификации.

# In[73]:


# Для ускорения анализа берем лишь часть данных
sample_indx = X.sample(150000).index


# In[74]:


# Делим выборку на обучающую и тестовую выбору в пропорции 95/5
# На обучающей будет обучать модель, а на тестовой проверять важность признаков

X_train, X_test, y_train, y_test = train_test_split(X.loc[sample_indx], y[sample_indx],
                                                   test_size=0.05)


# In[75]:


get_ipython().run_cell_magic('time', '', "# Инициализируем модель, тут нам важен инференс, а не подбор гиперпараметров\nmodel_for_analysis = CatBoostClassifier(n_estimators=400,\n                                          task_type='GPU',\n                                       verbose=0)\nmodel_for_analysis.fit(X_train, y_train)\n")


# #### Permutation feature importance

# Вычисление важности признаков (*Permutation feature importance*) - это метод проверки модели, который измеряет вклад каждого признака в характеристики подобранной модели в заданном наборе табличных данных.
# 
# Этот метод особенно полезен для нелинейных или непрозрачных моделей и включает в себя *случайное перемешивание значений одного признака и наблюдение за результирующим ухудшением оценки модели*. Разрывая связь между признаками и целью, мы определяем, насколько модель полагается на этот конкретный признак.
# 
# Permutation feature importance будем считать по каждой модели (по каждому признаку).

# In[76]:


get_ipython().run_cell_magic('time', '', '# С помощью метода из библиотеки sklearn выполняем алгоритм permutation importance\nperm_importance = permutation_importance(\n    model_for_analysis, X_test, y_test, n_repeats=5\n)\n')


# In[77]:


# Агрегируем значения для удобного вывода
importances = pd.Series(
    perm_importance.importances_mean,
    index=X_test.columns
).sort_values(ascending=False)

print('Для каждого признака есть свой коэффицент важности')
importances.head(6)


# In[78]:


# Отображаем график для наглядности
plt.figure(figsize=(10, 4))
importances.plot(kind="bar")
plt.title("Permutation Importance")
plt.show()


# In[79]:


# Сохраняем неинформативные признаки для целевого признака
perm_bad_features = importances[importances < 0].index.tolist()


# #### SHAP
# 
# Для анализа важности признаков с помощью **SHAP** (*SHapley Additive exPlanations*) используется другой подход, чем permutation importance. 
# 
# SHAP основывается на теории кооперативных игр и предоставляет согласованные оценки вклада каждого признака в предсказание модели для каждого отдельного наблюдения. Это даёт более глубокое понимание поведения модели.

# In[80]:


get_ipython().run_cell_magic('time', '', '\n# small_X_test = X_test.sample(500)\n\nexplainer = shap.TreeExplainer(model_for_analysis)\n\n# # Вычисляем SHAP значения\nshap_values = explainer.shap_values(X_test, check_additivity=False)\nshap_df = pd.DataFrame(shap_values, columns=X_test.columns)\n\n# Среднее абсолютное значение SHAP для каждого признака\nshap_importances = shap_df.abs().mean(axis=0).sort_values(ascending=False)\n\nprint("Важность признаков по SHAP:")\nprint(shap_importances.head(6))\n')


# In[81]:


plt.figure(figsize=(10, 10))
# Общий график важности признаков
shap.summary_plot(shap_values, X_test, show=False, max_display=8)
plt.title("SHAP Feature Importance")
plt.show()


# In[82]:


# Сохраняем неинформативные признаки для целевого признака
shap_bad_features = shap_importances[shap_importances > 0.006].index.tolist()


# Мы собрали набор признаков. Обнаружили мультиколлинеарность и слабую корреляцию некоторых признаков с целевым показателем.
# 
# 
# Для объективной оценки важности использовали **4** метода:
# * Анализ мультиколлинеарности
# * Корреляция с целевой переменной
# * Permutation Importance
# * SHAP
# 
# На основе результатов каждого из методов были получены списки **наименее значимых признаков**. 
# Признаки, которые показали низкую важность по одному из четырёх методам, признанаем **наименее информативными** и подлежат **удалению** из датасета. Таким образом, мы *объективно отберём только полезные признаки*.

# In[108]:


bad_features = set(shap_bad_features) & \
               set(perm_bad_features) | \
               set(low_corr_target) & \
               set(high_corr_features)


# In[109]:


print(f'Кол-во "плохих" признаков: {len(bad_features)}')


# <a id=4-3></a>
# ## 4.3 Промежуточное обучение модели
# 
# На основе оставшихся признаков базовую обучим модель ```CatBoostClassifier```.

# In[37]:


# Чтобы не запускать заново
# Загружаем модель
catboost_model = CatBoostClassifier().load_model('models/catboost_best.cbm')

# Получаем имена колонок, на которых обучалась модель
col2infer = catboost_model.feature_names_

X_filtered = X[col2infer]


# In[38]:


# Подготовка данных
# X_filtered = X.drop(columns=bad_features)

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2)


# In[111]:


# Настройкой гиперпараметров займемся в следующей главе, а сейчас используем стандартные
model_filtered_feature = CatBoostClassifier(n_estimators=1000,
                                               task_type='GPU',
                                           verbose=0)

# Обучаем модель
model_filtered_feature.fit(X_train, y_train)


# ### Инференс промежуточной модели

# In[112]:


y_pred_proba = model_filtered_feature.predict_proba(X_test)[:, 1]

y_pred_proba


# Основная метрика соревнования - ```ROC-AUC```. Для подсчета на нашей тестовой выборке используем ```roc_auc_score``` из ```sklearn```.

# In[113]:


print(f'ROC AUC score для Catboost после отбора признаков: {roc_auc_score(y_test, y_pred_proba):.2f}')


# In[114]:


test = pd.read_csv('data/test.csv')
test = test.drop(columns=['id'])

sample = pd.read_csv('data/sample_submission.csv')


# In[115]:


test_y_pred = baseline_model.predict_proba(test)[:, 1]

sample['Heart Disease'] = test_y_pred

sample.to_csv('outputs/filtered_3.csv', index=False)


# После отбора признаков получили результат чуть хуже бэйслайна. Исправим это, выбрав оптимальную модель для обучения! 

# ###

# <a id=5></a>
# # 5. Выбор оптимального алгоритма
# 
# Для решения задачи рассмотрим несколько моделей, каждая из которых обладает своими преимуществами:
# 
# 1. **CatBoost** — поддерживает обучение на **GPU**, что значительно ускоряет процесс.
# 2. **LightGBM** — отличается высокой **скоростью обучения** и эффективной работой на больших наборах данных.
# 3. **XGBoost** — поддерживает обучение на **GPU**, мощный алгоритм градиентного бустинга, хорошо зарекомендовавший себя в соревнованиях и прикладных задачах; устойчив к переобучению и предоставляет гибкую настройку гиперпараметров.
# 4. **Random Forest** — это ансамбль независимых деревьев решений, обучаемых на различных подвыборках данных. Модель хорошо работает на табличных данных и служит надёжным ориентиром при сравнении с бустингом.
# 5. **Логистическая регрессия** — простая **линейная модель**, которая служит хорошей базовой линиейи позволяет оценить, насколько сложные алгоритмы действительно улучшают качество.
# 
# Определим **2 лидеров** по метрике, которым дальше будем подбирать гиперпараметры.

# In[116]:


# Подготовка данных
X_filtered = X.drop(columns=bad_features)

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=11)


# <a id=5-1></a>
# ## 5.1 CatBoost

# In[127]:


get_ipython().run_cell_magic('time', '', '# Используем стандартные гиперпаарметры\nmodel_filtered_feature = CatBoostClassifier(n_estimators=1000,\n                                               task_type=\'GPU\',\n                                               verbose=0)\n\n# Обучаем модель\nmodel_filtered_feature.fit(X_train, y_train)\n\n# Считаем метрику\ny_pred_proba = model_filtered_feature.predict_proba(X_test)[:, 1]\ny_pred_proba\nprint(f\'ROC AUC score для RandomForest после отбора признаков: {roc_auc_score(y_test, y_pred_proba):.4f}\')\n\n# Сохраняем модель\nmodel_filtered_feature.save_model("models/catboost_default.cbm")\n')


# <a id=5-2></a>
# ## 5.2 LightGBM

# In[128]:


get_ipython().run_cell_magic('time', '', '# Используем стандартные гиперпараметры\nmodel_lgb = lgb.LGBMClassifier(\n    n_estimators=1000,\n    verbose=-1\n)\n\n# Обучаем модель\nmodel_lgb.fit(X_train, y_train)\n\n# Предсказания\ny_pred_proba = model_lgb.predict_proba(X_test)[:, 1]\n\n# ROC AUC\nauc = roc_auc_score(y_test, y_pred_proba)\nprint(f\'ROC AUC score для LightGBM после отбора признаков: {auc:.4f}\')\n\n# Сохраняем модель\nmodel_lgb.booster_.save_model("models/lightgbm_default.txt")\n')


# <a id=5-3></a>
# ## 5.3 XGBoost

# In[130]:


get_ipython().run_cell_magic('time', '', '# Используем стандартные гиперпараметры\nmodel_xgb = xgb.XGBClassifier(\n    n_estimators=1000,\n    eval_metric=\'logloss\',\n    device=\'cuda\',\n    use_label_encoder=False\n)\n\n# Обучаем модель\nmodel_xgb.fit(X_train, y_train)\n\n# Предсказания\ny_pred_proba = model_xgb.predict_proba(X_test)[:, 1]\n\n# ROC AUC\nauc = roc_auc_score(y_test, y_pred_proba)\nprint(f\'ROC AUC score для XGBoost после отбора признаков: {auc:.4f}\')\n\n# Сохраняем модель\nmodel_xgb.save_model("models/xgboost_default.json")\n')


# <a id=5-4></a>
# ## 5.4 Random Forest

# In[136]:


get_ipython().run_cell_magic('time', '', '# Используем стандартные гиперпараметры\nmodel_rf = RandomForestClassifier(\n    n_estimators=1000,\n    random_state=42,\n    n_jobs=-1\n)\n\n# Обучаем модель\nmodel_rf.fit(X_train, y_train)\n\n# Предсказания\ny_pred_proba = model_rf.predict_proba(X_test)[:, 1]\n\n# Метрика\nauc = roc_auc_score(y_test, y_pred_proba)\nprint(f\'ROC AUC score для RandomForest после отбора признаков: {auc:.4f}\')\n\n# Сохраняем модель\njoblib.dump(model_rf, "models/random_forest_default.pkl")\n')


# <a id=5-5></a>
# ## 5.5 Логистическая регрессия

# In[138]:


get_ipython().run_cell_magic('time', '', '# Используем стандартные гиперпараметры\nmodel_lr = LogisticRegression(\n    max_iter=1000,     # чтобы модель точно сошлась\n    solver=\'liblinear\' # хорошо работает для бинарной классификации\n)\n\n# Обучаем модель\nmodel_lr.fit(X_train, y_train)\n\n# Предсказания\ny_pred_proba = model_lr.predict_proba(X_test)[:, 1]\n\n# ROC AUC\nauc = roc_auc_score(y_test, y_pred_proba)\nprint(f\'ROC AUC score для Logistic Regression: {auc:.4f}\')\n\n# Сохраняем модель\njoblib.dump(model_lr, "models/logreg_default.pkl")\n')


# <a id=5-6></a>
# ## 5.6 Лучшие алгоритмы
# 
# Самую лучшую метрику показалаи алгоритмы ```LightGBM``` и ```CatBoost```. Проведем подбор гиперпараметров и обучим самые **мощные модели**.

# ###

# <a id=6></a>
# # 6. Подбор гиперпараметров

# <a id=6-1></a>
# ## 6.1 CatBoost
# 
# Объявим функцию обучения `Catboost` с возращением предсказаний на `Kfold` валидации

# In[179]:


def fit_catboost(trial, train, val):
    X_train, y_train = train
    X_val, y_val = val

    param = {
        'iterations' : 400,
        'task_type': 'GPU',
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 2, 50),
        "colsample_bylevel": 1.0, # На GPU только так

        "auto_class_weights": trial.suggest_categorical("auto_class_weights", ["SqrtBalanced", "Balanced", "None"]),
        "depth": trial.suggest_int("depth", 3, 9),

        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "used_ram_limit": "14gb",
        "eval_metric": "AUC",
    }


    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 20)

    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)


    clf = CatBoostClassifier(
        **param,
        thread_count=-1,
        random_seed=42,
        allow_writing_files=False,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=0,
        plot=False,
        early_stopping_rounds=10,
    )

    y_pred = clf.predict(X_val)
    return clf, y_pred


# Напишем функцию **objective** в которую поместим `Kfold` валидацию, чтобы подбирать лучшие гиперпараметры на всем датасете.

# In[182]:


def objective(trial, return_models=False):
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores, models = [], []

    for train_idx, valid_idx in kf.split(X_filtered):
        train_data = X_filtered.iloc[train_idx, :], y.iloc[train_idx]
        valid_data = X_filtered.iloc[valid_idx, :], y.iloc[valid_idx]

        # Подаем trials для перебора
        model, y_pred = fit_catboost(trial, train_data, valid_data) # Определили выше
        scores.append(roc_auc_score(y_pred, valid_data[1]))
        models.append(model)
        break


    result = np.mean(scores)

    if return_models:
        return result, models
    else:
        return result


# 🚀 Запускаем Optuna!

# In[183]:


optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction="maximize")
study.optimize(objective,
               n_trials=100,
               n_jobs=1, # Запуск без параллельности на GPU
               show_progress_bar=True)


# Посмотрим на **лучшие** параметры.

# In[184]:


print("Best trial: score {}, params {}".format(study.best_trial.value, study.best_trial.params))


# Обучим итоговые модели уже на них.

# In[185]:


valid_scores, models = objective(
    optuna.trial.FixedTrial(study.best_params),
    return_models=True,
)

valid_scores, len(models)


# #### Инфренс лучшей модели

# In[77]:


test = pd.read_csv('data/test.csv')
sample = pd.read_csv('data/sample_submission.csv')


# In[78]:


# Так как модель обучалась на сгенерированных признаках, сделаем тоже самое с тестовой выборкой
# Инициализируем метод
poly = PolynomialFeatures(2)
# Генерируем фичи
test = pd.DataFrame(poly.fit_transform(test), columns=poly.get_feature_names_out())
test = test[col2infer]

test.shape


# In[83]:


test_y_pred = models[0].predict(test)

sample['Heart Disease'] = test_y_pred

sample.to_csv('outputs/best_lgbm.csv', index=False)


# <a id=6-2></a>
# ## 6.2 LightGBM

# Объявим функцию обучения `LightGBM` с возращением предсказаний на `Kfold` валидации

# In[63]:


def fit_lgbm(trial, train, val):
    X_train, y_train = train
    X_val, y_val = val

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt",]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.02),
        "num_leaves": trial.suggest_int("num_leaves", 15, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 5),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 5),

        # GPU
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "verbosity": -1,

        # стабильность
        "seed": 42,
    }

    # LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    # Обучение
    clf = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    # Предсказания
    y_pred = clf.predict(X_val)
    return clf, y_pred


# Напишем функцию **objective** в которую поместим `Kfold` валидацию, чтобы подбирать лучшие гиперпараметры на всем датасете.

# In[64]:


def objective(trial, return_models=False):
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores, models = [], []

    for train_idx, valid_idx in kf.split(X_filtered):
        train_data = X_filtered.iloc[train_idx, :], y.iloc[train_idx]
        valid_data = X_filtered.iloc[valid_idx, :], y.iloc[valid_idx]

        # Подаем trials для перебора
        model, y_pred = fit_lgbm(trial, train_data, valid_data) # Определили выше
        scores.append(roc_auc_score(valid_data[1], y_pred))
        models.append(model)

    result = np.mean(scores)

    if return_models:
        return result, models
    else:
        return result


# 🚀 Запускаем Optuna!

# In[65]:


optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction="maximize")
study.optimize(objective,
               n_trials=100,
               n_jobs=1, # Запуск без параллельности на GPU
               show_progress_bar=True)


# Посмотрим на **лучшие** параметры.

# In[66]:


print("Best trial: score {}, params {}".format(study.best_trial.value, study.best_trial.params))


# Обучим итоговые модели уже на них.

# In[67]:


valid_scores, models = objective(
    optuna.trial.FixedTrial(study.best_params),
    return_models=True,
)

valid_scores, len(models)


# Сохраняем лучшую модель.

# In[86]:


# Сохраняем модель
models[0].save_model("models/lightgbm_best.txt")


# ###

# <a id=7></a>
# # 7. Смешивание моделей

# Сейчас проведём блендинг — объединим предсказания трёх моделей (`LightGBM`, `CatBoost`, `Logistic Regression`) через взвешенное усреднение. Каждой модели назначим вес в соответствии с её качеством: наибольший вес получает `LightGBM` как лучшая модель, затем `CatBoost`, наименьший — `Logistic Regression`. Проверим **10** различных комбинаций весов и выберем ту, которая даёт наилучший результат на валидационной выборке.

# In[19]:


import pandas as pd


# In[24]:


# Загружаем решения

pred_lr = pd.read_csv('outputs/baseline.csv')
pred_cat = pd.read_csv('outputs/best_catboost.csv')
pred_lgbm = pd.read_csv('outputs/best_lgbm.csv')


# In[25]:


# 10 комбинаций весов (lgbm, catboost, logreg)
# Сумма весов = 1.0, lgbm получает наибольший вес
combinations = [
    (0.6,  0.3,  0.1),   # 1. Ставка на lgbm
    (0.5,  0.3,  0.2),   # 2. Чуть больше lr
    (0.5,  0.4,  0.1),   # 3. Больше catboost
    (0.4,  0.4,  0.2),   # 4. lgbm и cat поровну
    (0.7,  0.2,  0.1),   # 5. Сильный акцент на lgbm
    (0.6,  0.2,  0.2),   # 6. cat и lr поровну
    (0.45, 0.35, 0.2),   # 7. Плавный градиент
    (0.55, 0.35, 0.1),   # 8. Почти без lr
    (0.4,  0.3,  0.3),   # 9. lr наравне с cat
    (0.34, 0.33, 0.33),  # 10. Почти равные веса
]


# In[26]:


pred_lgbm


# In[27]:


# Проходимся по всем комбинациям

for i, (w_lgbm, w_cat, w_lr) in enumerate(combinations, 1):
    blend = (
        w_lgbm * pred_lgbm['Heart Disease'] +
        w_cat * pred_cat['Heart Disease'] +
        w_lr * pred_lr['Heart Disease']
    )

    submission = pd.DataFrame({
        'id': pred_lgbm['id'],
        'Heart Disease': blend
    })

    filename = f'outputs/blend_{i}_lgbm{w_lgbm}_cat{w_cat}_lr{w_lr}.csv'
    submission.to_csv(filename, index=False)

    print(f"{i:<4} {w_lgbm:<8} {w_cat:<8} {w_lr:<8} {filename}")

print("\nГотово! Все 10 файлов сохранены в папку outputs/")

