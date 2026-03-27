#!/usr/bin/env python
# coding: utf-8

# # Модуль посвящен обучению модели и ее инференсу.
# 
# 
# Обучать будет классификацию и регрессию. Обычно в 3 модулю как раз несколько задач.

# # Импорт библиотек

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import RandomizedSearchCV

from imblearn.over_sampling import SMOTE

import joblib


# # Импорт данных
# 
# Мы предобрабатывали данные в предыдущих модулях, поэтому с данными работаем минимально.

# In[5]:


df = pd.read_parquet('data/data.parquet')

df.head()


# In[6]:


X = df.drop(['AQI'], axis=1)
y = df['AQI']


# In[7]:


# Разделяем данные ДО масштабирования
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[8]:


# Масштабируем ТОЛЬКО по тренировочным данным
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)


# In[9]:


joblib.dump(scaler, 'scaler.pkl')


# # Модель регресии

# ## Обучение моделей
# 
# 
# Для обучения регрессии берем 3 модели: Линейную регрессию, Градиентный бустинг (или Случайный лес) и нейронную сеть (в критериях может быть отдельно использование нейронки, поэтому берем всегда).
# 
# 
# Такой набор моделей объясняется их разноплановостью, что лучше скажется на итоговом результате.

# ### Линейная регрессия

# In[10]:


model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)


# ### Градиентный бустинг

# In[11]:


model_cb = CatBoostRegressor(
    verbose=0,
    # task_type='GPU'
)

model_cb.fit(X_train, y_train)
y_pred_cb = model_cb.predict(X_test)


# ### Нейронная сеть

# In[12]:


model_nn = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=1000,
    random_state=42
)

model_nn.fit(X_train, y_train)
y_pred_nn = model_nn.predict(X_test)


# ## Метрики качества регрессии
# 
# 1. MAE — Mean Absolute Error
# 
# 
# 2. RMSE — Root Mean Squared Error
# 
# 
# 3. R² — коэффициент детерминации

# In[13]:


print('MAE - чем меньше, тем лучше')
print(f'MAE для LinearRegression: {mean_absolute_error(y_test, y_pred_lr):.3f}')
print(f'MAE для CatBoostRegressor: {mean_absolute_error(y_test, y_pred_cb):.3f}')
print(f'MAE для MLPRegressor: {mean_absolute_error(y_test, y_pred_nn):.3f}')


# In[14]:


print('RMSE - чем меньше, тем лучше')
print(f'RMSE для LinearRegression: {root_mean_squared_error(y_test, y_pred_lr):.3f}')
print(f'RMSE для CatBoostRegressor: {root_mean_squared_error(y_test, y_pred_cb):.3f}')
print(f'RMSE для MLPRegressor: {root_mean_squared_error(y_test, y_pred_nn):.3f}')


# In[15]:


print('R2 - чем выше, тем лучше')
print(f'R2 для LinearRegression: {r2_score(y_test, y_pred_lr):.3f}')
print(f'R2 для CatBoostRegressor: {r2_score(y_test, y_pred_cb):.3f}')
print(f'R2 для MLPRegressor: {r2_score(y_test, y_pred_nn):.3f}')


# Лучшеми моделями стала CatBoostRegressor и Нейронная сеть, подберем гиперпараметры для них с помощью RandomizedSearchCV.

# ## Catboost с подобранными гиперпараметрами.
# 
# Для подбора гиперпараметров Catboost нужно откатиться до scikit-learn==1.5.2
# 
# Если возникают проблемы, можно просто заменить модель Catboost на RandomForest, с ним таких проблем не будет.
# 
# Однако, для начала лучше попробовать градиентный бустинг, так как он выдает обычно качество лучше.

# In[14]:


get_ipython().system('pip install scikit-learn==1.5.2 -q')


# In[15]:


param_dist_cb = {
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "iterations": [300, 500, 800],
    "l2_leaf_reg": [1, 3, 5, 7, 9]
}

cb = CatBoostRegressor(verbose=0,
                      # task_type='GPU'
                      )

random_cb = RandomizedSearchCV(
    cb,
    param_distributions=param_dist_cb,
    cv=3,
    # n_jobs=-1,
    verbose=1,
    n_iter=3
)

random_cb.fit(X_train, y_train)

best_cb = random_cb.best_estimator_
y_pred_cb = best_cb.predict(X_test)

print(f'MAE для CatBoostRegressor: {mean_absolute_error(y_test, y_pred_cb):.3f}')
print(f'RMSE для CatBoostRegressor: {root_mean_squared_error(y_test, y_pred_cb):.3f}')
print(f'R2 для CatBoostRegressor: {r2_score(y_test, y_pred_cb):.3f}')


# ## Нейронная сеть с подобранными гиперпараметрами.

# In[16]:


param_grid_nn = {
    "hidden_layer_sizes": [(32, ), (64, ), (64, 32), (128, 64), (128, 128)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.005, 0.01]
}

model_nn = MLPRegressor()


random_nn = RandomizedSearchCV(
    model_nn,
    param_distributions=param_grid_nn,
    cv=3,
    # n_jobs=-1,
    verbose=1,
    n_iter=3
)

random_nn.fit(X_train, y_train)

best_nn = random_nn.best_estimator_
y_pred_nn = best_nn.predict(X_test)

print(f'MAE для MLPRegressor: {mean_absolute_error(y_test, y_pred_nn):.3f}')
print(f'RMSE для MLPRegressor: {root_mean_squared_error(y_test, y_pred_nn):.3f}')
print(f'R2 для MLPRegressor: {r2_score(y_test, y_pred_nn):.3f}')


# Лучшей моделью стала CatboostRegressor с подобранными гиперпараметрами. Визуализируем прогноз модели с фактическими данными.

# ## Визуализация предсказаний
# 

# In[17]:


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred_cb, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()


# В целом по графику видна отличная корреляция между предсказанием и реальнами данными.

# ## Сохранение моделей классификации
# 
# Для сохранения моделей sklearn используем библиотеку joblib. Хотя для catboost-моделей этот метод тоже подходит, лучше сохранить модель собственными методами.

# In[18]:


best_cb.save_model('catboost_model_regression.cbm', format='cbm')

joblib.dump(best_nn, 'nn_model_regression.pkl');


# In[20]:


# ===========================================
# 🟩 Сохранение данных для будущего fine-tuning
# ===========================================

joblib.dump(X_train, 'old_X_train_reg.pkl')
joblib.dump(y_train, 'old_y_train_reg.pkl')

joblib.dump(X_test, 'X_test_reg.pkl')
joblib.dump(y_test, 'y_test_reg.pkl')


# # Модель классификации

# ## Предобработка данных

# In[22]:


# Превращаем регрессию в классификацию
# y — регрессионный таргер
# labels = ["low", "medium", "high"]
bins = np.quantile(y, [0, 0.22, 0.89, 1.0])
labels = [0, 1, 2]

y_class = pd.cut(y, bins=bins, labels=labels, include_lowest=True)


# Видим, что классы несбалансированны, класс 1 в 6 раз больше чем класс 2.

# In[23]:


y_class.value_counts(normalize=True) * 100


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=y_class)
plt.title("Class distribution")
plt.show()


# ## Устранение дисбаланса
# 
# Для устранения дисбаланса используем метод SMOTE, он создаст синтетические новые примеры для редкого класса в нашем датасете.
# 
# 
# Делает это так:
# 
# 1. выбирает случайный объект редкого класса
# 
# 2. выбирает его соседей
# 
# 3. интерполирует (генерирует точки между ними)

# In[26]:


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# После генерации проверим данные на сбалансированность классов.

# In[27]:


y_train_resampled.value_counts(normalize=True) * 100


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=y_train_resampled)
plt.title("Class distribution")
plt.show()


# Видим, что после использования метода SMOTE баланс идеален, что поможет лучше обучить модель и вычислить метрики.

# ## Обучение моделей
# 
# Аналогично регрессии, берем разностороние методы классификации.

# ### Логистическая регрессия

# In[29]:


log_reg = LogisticRegression()

log_reg.fit(X_train_resampled, y_train_resampled)
y_pred_lr = log_reg.predict(X_test)


# ### Градиентный бустинг

# In[30]:


cb = CatBoostClassifier(verbose=0)

cb.fit(X_train_resampled, y_train_resampled)
y_pred_cb = cb.predict(X_test)


# ### Нейронная сеть

# In[31]:


nn = MLPClassifier()

nn.fit(X_train_resampled, y_train_resampled)
y_pred_nn = nn.predict(X_test)


# ## Метрики качества классификации
# 
# 
# Для multi-class классификации, как в нашем случае, подойдут:
# 
# - Accuracy
# 
# - Precision / Recall / F1 (в среднем)
# 
# - ROC/AUC (макро или micro)
# 
# - Confusion Matrix

# In[32]:


print("Accuracy — чем выше, тем лучше")
print(f"Accuracy для LogisticRegression: {accuracy_score(y_test, y_pred_lr):.3f}")
print(f"Accuracy для CatBoostClassifier: {accuracy_score(y_test, y_pred_cb):.3f}")
print(f"Accuracy для MLPClassifier: {accuracy_score(y_test, y_pred_nn):.3f}")


# In[33]:


print("F1 (macro) — чем выше, тем лучше")
print(f"F1-macro для LogisticRegression: {f1_score(y_test, y_pred_lr, average='macro'):.3f}")
print(f"F1-macro для CatBoostClassifier: {f1_score(y_test, y_pred_cb, average='macro'):.3f}")
print(f"F1-macro для MLPClassifier: {f1_score(y_test, y_pred_nn, average='macro'):.3f}")


# In[34]:


print("Precision (macro) — чем выше, тем лучше")
print(f"Precision-macro для LogisticRegression: {precision_score(y_test, y_pred_lr, average='macro'):.3f}")
print(f"Precision-macro для CatBoostClassifier: {precision_score(y_test, y_pred_cb, average='macro'):.3f}")
print(f"Precision-macro для MLPClassifier: {precision_score(y_test, y_pred_nn, average='macro'):.3f}")


# In[35]:


print("Recall (macro) — чем выше, тем лучше")
print(f"Recall-macro для LogisticRegression: {recall_score(y_test, y_pred_lr, average='macro'):.3f}")
print(f"Recall-macro для CatBoostClassifier: {recall_score(y_test, y_pred_cb, average='macro'):.3f}")
print(f"Recall-macro для MLPClassifier: {recall_score(y_test, y_pred_nn, average='macro'):.3f}")


# In[36]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — Logistic Regression")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()


# In[37]:


cm_cb = confusion_matrix(y_test, y_pred_cb)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_cb, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — CatBoostClassifier")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()


# In[38]:


cm_nn = confusion_matrix(y_test, y_pred_nn)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — Neural Network")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()


# Исходя из метрик, лучшими моделями стали CatboostClassifier и нейронная сеть. Займемся подбором гиперпараметров для этих моделей и сравним их снова.

# ## CatBoostClassifier с подобранными гиперпараметрами
# 
# Для подбора гиперпараметров Catboost нужно откатиться до scikit-learn==1.5.2
# 
# Если возникают проблемы, можно просто заменить модель Catboost на RandomForest, с ним таких проблем не будет.
# 
# Однако, для начала лучше попробовать градиентный бустинг, так как он выдает обычно качество лучше.

# In[39]:


get_ipython().system('pip install scikit-learn==1.5.2 -q')


# In[40]:


param_dist_cb_cls = {
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "iterations": [300, 500, 800],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
}

cb_cls = CatBoostClassifier(
    verbose=0,
    # task_type='GPU'
)

random_cb_cls = RandomizedSearchCV(
    cb_cls,
    param_distributions=param_dist_cb_cls,
    cv=3,
    # n_jobs=-1,
    verbose=1,
    n_iter=4,
)

random_cb_cls.fit(X_train, y_train)

best_cb_cls = random_cb_cls.best_estimator_
y_pred_cb = best_cb_cls.predict(X_test)
y_proba_cb = best_cb_cls.predict_proba(X_test)

print(f"Accuracy для CatBoostClassifier: {accuracy_score(y_test, y_pred_cb):.3f}")
print(f"F1-macro для CatBoostClassifier: {f1_score(y_test, y_pred_cb, average='macro'):.3f}")
print(f"Precision-macro для CatBoostClassifier: {precision_score(y_test, y_pred_cb, average='macro'):.3f}")
print(f"Recall-macro для CatBoostClassifier: {recall_score(y_test, y_pred_cb, average='macro'):.3f}")


# ## MLPClassifier с подобранными гиперпараметрами

# In[41]:


param_grid_nn_cls = {
    "hidden_layer_sizes": [(64, 32), (128, 64)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01],
}

model_nn_cls = MLPClassifier(
)

random_nn_cls = RandomizedSearchCV(
    model_nn_cls,
    param_distributions=param_grid_nn_cls,
    cv=3,
    # n_jobs=-1,
    verbose=1,
    n_iter=3
)

random_nn_cls.fit(X_train, y_train)

best_nn_cls = random_nn_cls.best_estimator_
y_pred_nn = best_nn_cls.predict(X_test)
y_proba_nn = best_nn_cls.predict_proba(X_test)

print(f"Accuracy для MLPClassifier: {accuracy_score(y_test, y_pred_nn):.3f}")
print(f"F1-macro для MLPClassifier: {f1_score(y_test, y_pred_nn, average='macro'):.3f}")
print(f"Precision-macro для MLPClassifier: {precision_score(y_test, y_pred_nn, average='macro'):.3f}")
print(f"Recall-macro для MLPClassifier: {recall_score(y_test, y_pred_nn, average='macro'):.3f}")


# Лучшей моделью с подобранными гиперпараметрами стала CatBoostClassifier, для нее построим ROC-AUC.

# ## Построение ROC-AUC

# In[46]:


# Массив классов
classes = np.unique(y_test)

# Бинаризуем y_test
y_test_bin = label_binarize(y_test, classes=classes)

# Предсказанные вероятности CatBoost
y_proba = best_cb_cls.predict_proba(X_test)

plt.figure(figsize=(8, 6))

# Построение ROC-кривой для каждого класса по отдельности
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f"Class {cls} (AUC = {roc_auc:.3f})")

# Линия случайного классификатора
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curves — CatBoostClassifier (OvR)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# Для каждого класса метрика высокая. Модель справляется очень хорошо.

# ## Распределения вероятностей

# In[47]:


import pandas as pd

proba_df = pd.DataFrame(y_proba, columns=[f"proba_{cls}" for cls in classes])

plt.figure(figsize=(10, 5))
sns.histplot(proba_df, kde=True, bins=25)
plt.title("Distribution of Predicted Probabilities — CatBoostClassifier")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.show()


# ## Сохранение моделей классификации
# 
# Для сохранения моделей sklearn используем библиотеку joblib. Хотя для catboost-моделей этот метод тоже подходит, лучше сохранить модель собственными методами.

# In[48]:


best_cb_cls.save_model('catboost_model_classification.cbm', format='cbm')

joblib.dump(best_nn_cls, 'nn_model_classification.pkl')


# ## Функция для дообучения
# 
# Часто на чемпионатах просят организовать непрерывное обучение моделей с помощью AirFlow. 
# 
# Создадим функцию для дообучения моделей, которую можно будет перенести в Airflow или интерфейс в будущем.
# 
# Аналогичную функцию можно реализовать и для классификации. 

# In[56]:


def fine_tuning_regression(new_data: pd.DataFrame) -> dict:
    """
    Корректный fine-tuning регрессионных моделей (MLPRegressor + CatBoostRegressor).
    Модели переобучаются на объединении старых + новых данных,
    используя тот же scaler и тот же фиксированный test из joblib.
    """

    # -------------------------------------------------
    # 1. ЗАГРУЗКА СТАРЫХ АРТЕФАКТОВ
    # -------------------------------------------------
    scaler = joblib.load('scaler.pkl')
    old_X = joblib.load('old_X_train_reg.pkl')
    old_y = joblib.load('old_y_train_reg.pkl')

    cb_model = CatBoostRegressor()
    cb_model.load_model('catboost_model_regression.cbm')

    nn_model = joblib.load('nn_model_regression.pkl')

    # -------------------------------------------------
    # 2. ПОДГОТОВКА НОВЫХ ДАННЫХ
    # -------------------------------------------------
    if 'AQI' not in new_data.columns:
        raise ValueError("В new_data должен быть столбец 'AQI'.")

    X_new = new_data.drop('AQI', axis=1)
    y_new = new_data['AQI'].astype(float)

    # применяем тот же scaler, что и при обучении
    X_new_scaled = scaler.transform(X_new)

    # -------------------------------------------------
    # 3. ОБЪЕДИНЕНИЕ СТАРЫХ + НОВЫХ ДАННЫХ
    # -------------------------------------------------
    X_full = np.vstack([old_X, X_new_scaled])
    y_full = np.concatenate([old_y, y_new])

    # -------------------------------------------------
    # 4. ПОВТОРНОЕ ОБУЧЕНИЕ МОДЕЛЕЙ
    # -------------------------------------------------
    nn_model.fit(X_full, y_full)
    cb_model.fit(X_full, y_full)

    # -------------------------------------------------
    # 5. ТЕСТИРОВАНИЕ НА ЗАГРУЖЕННОМ (joblib) TEST НАБОРЕ
    # -------------------------------------------------
    X_test_scaled = joblib.load('X_test_reg.pkl')
    y_test = joblib.load('y_test_reg.pkl')

    y_pred_cb = cb_model.predict(X_test_scaled)
    y_pred_nn = nn_model.predict(X_test_scaled)

    # -------------------------------------------------
    # 6. РАСЧЁТ МЕТРИК
    # -------------------------------------------------
    metrics = {
        "CatBoostRegressor": {
            "MAE": round(mean_absolute_error(y_test, y_pred_cb), 3),
            "RMSE": round(root_mean_squared_error(y_test, y_pred_cb), 3),
            "R2": round(r2_score(y_test, y_pred_cb), 3),
        },
        "MLPRegressor": {
            "MAE": round(mean_absolute_error(y_test, y_pred_nn), 3),
            "RMSE": round(root_mean_squared_error(y_test, y_pred_nn), 3),
            "R2": round(r2_score(y_test, y_pred_nn), 3),
        }
    }

    # -------------------------------------------------
    # 7. СОХРАНЕНИЕ ОБНОВЛЁННЫХ МОДЕЛЕЙ И TRAIN ДАННЫХ
    # -------------------------------------------------
    cb_model.save_model('catboost_model_regression.cbm')
    joblib.dump(nn_model, 'nn_model_regression.pkl')

    joblib.dump(X_full, 'old_X_train_reg.pkl')
    joblib.dump(y_full, 'old_y_train_reg.pkl')

    return metrics


# In[57]:


# Проверяем функцию

new_data = X_train.copy()
new_data['AQI'] = y_train
fine_tuning_regression(new_data)

