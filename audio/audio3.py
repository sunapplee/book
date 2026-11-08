#!/usr/bin/env python
# coding: utf-8

# # 🔹 Обучение моделей и подбор гиперпараметров

# 1. импорт библиотек

# In[2]:


# Работа с файлами и массивами
import os
import numpy as np

import joblib

# Обработка аудио (загрузка wav + MFCC)
import librosa

# Разметка и обучение
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Модели классификации
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Метрики качества
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# 2. пути

# In[3]:


# Корневая папка датасета
DATA_DIR = "cats_dogs"

# Папки с train и test
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")


# 3. извлекаем признаки с помощью MFCC

# In[4]:


def extract_features_mfcc(
    file_path,
    n_mfcc=30,
    n_fft=2048,
    hop_length=512,
    add_delta=True
):
    """
    Читает аудио-файл и возвращает вектор признаков на основе MFCC.

    Параметры:
    - n_mfcc: число MFCC коэффициентов (обычно 13–40).
    - n_fft: размер окна БПФ (больше -> точнее по частоте, но тяжелее).
    - hop_length: шаг окна (меньше -> лучше по времени, но больше кадров).
    - add_delta: добавлять ли дельта-признаки (изменение MFCC по времени).
    """
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)

    features = list(mfcc_mean) + list(mfcc_std)

    if add_delta:
        delta = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta, axis=1)
        delta_std  = np.std(delta, axis=1)

        delta2 = librosa.feature.delta(mfcc, order=2)
        delta2_mean = np.mean(delta2, axis=1)
        delta2_std  = np.std(delta2, axis=1)

        features += list(delta_mean) + list(delta_std)
        features += list(delta2_mean) + list(delta2_std)

    return np.array(features)


# 4. сборка X_train y_train X_test y_test

# In[5]:


def load_dataset(folder):
    """
    Обходит все подпапки в `folder`, считает признаки для каждого .wav
    и формирует матрицу X и вектор y.
    """
    X = []
    y = []

    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            file_path = os.path.join(label_path, fname)

            # извлекаем MFCC-признаки
            features = extract_features_mfcc(file_path)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

X_train, y_train = load_dataset(TRAIN_DIR)
X_test,  y_test  = load_dataset(TEST_DIR)


# 5. кодируем метки классов в числа (моделям удобнее работать с числами (0,1) а не со строками)

# In[6]:


encoder = LabelEncoder()

# Обучаем энкодер на обучающей разметке
y_train_enc = encoder.fit_transform(y_train)

# Применяем обученный энкодер к тестовой разметке
y_test_enc = encoder.transform(y_test)

joblib.dump(encoder, "encoder.pkl");


# 6. Масштабируем признаки

# In[7]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Применяем scaler к test
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

X_train[2].shape


# 7. Обучение моделей + настройка параметров
# 
# 1) LogisticRegression
# 2) RandomForest

# 1) LogisticRegression

# In[8]:


log_reg = LogisticRegression(
    C=1.0, # [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    penalty='l2', # l1 или l2
    solver='lbfgs', # 'liblinear', 'lbfgs', 'saga'
    max_iter=1000  # [500, 1000, 2000]
)


log_reg.fit(X_train, y_train_enc)
y_pred_lr = log_reg.predict(X_test)


# In[9]:


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# подбор лучших параметров c помощью GridSearchCV (рассмотрим множество вариаций тк на чемпе даже тысячные важны важны)

# In[10]:


from sklearn.model_selection import GridSearchCV
param_grid_lr = [
    # 1. lbfgs (только l2, самый быстрый)
    {
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs'],
        'max_iter': [1000, 2000],
        'fit_intercept': [True],
        'class_weight': ['balanced', None]
    },

    # 2. liblinear (l1/l2, но dual ТОЛЬКО для l2)
    {
        'penalty': ['l1'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'],
        'max_iter': [1000, 2000],
        'fit_intercept': [True],
        'class_weight': ['balanced', None],
        'dual': [False]  # l1 НЕ поддерживает dual=True
    },
    {
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'],
        'max_iter': [1000, 2000],
        'fit_intercept': [True],
        'class_weight': ['balanced', None],
        'dual': [False, True]  # l2 поддерживает dual
    },

    # 3. saga (все типы penalty)
    {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['saga'],
        'max_iter': [1000, 2000],
        'fit_intercept': [True],
        'class_weight': ['balanced', None]
    },
    {
        'penalty': ['elasticnet'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['saga'],
        'max_iter': [1000, 2000],
        'fit_intercept': [True],
        'class_weight': ['balanced', None],
        'l1_ratio': [0.1, 0.5, 0.9]
    }
]

grid_lr = GridSearchCV(
    LogisticRegression(),
    param_grid_lr,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,  
    verbose=1, 
    error_score=0.0  
)


grid_lr.fit(X_train, y_train_enc)


print(grid_lr.best_params_)
print(" Лучший F1:", grid_lr.best_score_)

best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test)


# 2) RandomForest 

# In[11]:


rf = RandomForestClassifier(
    n_estimators=200,   # число деревьев
    max_depth=None,  # глубина не ограничена
    random_state=42,
    n_jobs=-1 # использовать все ядра
)

rf.fit(X_train, y_train_enc)
y_pred_rf = rf.predict(X_test)


# подбор параметров

# In[12]:


param_grid_rf = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1
)

grid_rf.fit(X_train, y_train_enc)

print(grid_rf.best_params_)
print("Лучший F1:", grid_rf.best_score_)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)


# 7. Метрики 

# In[18]:


y_pred_rf = best_rf.predict(X_test)
y_pred_lr = best_lr.predict(X_test)

print("LR Accuracy:", round(accuracy_score(y_test_enc, y_pred_lr), 3))
print("LR F1:", round(f1_score(y_test_enc, y_pred_lr, average='macro'), 3))
print()
print("RF Accuracy:", round(accuracy_score(y_test_enc, y_pred_rf), 3))
print("RF F1:", round(f1_score(y_test_enc, y_pred_rf, average='macro'), 3))


# также можно обучить модели такие как:

# In[19]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

# 1. SVM
svm = SVC(random_state=42)
svm.fit(X_train, y_train_enc)
y_pred_svm = svm.predict(X_test)

# 2. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train_enc)
y_pred_dt = dt.predict(X_test)

# 3. Naive Bayes  
nb = GaussianNB()
nb.fit(X_train, y_train_enc)
y_pred_nb = nb.predict(X_test)

# 4. KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train_enc)
y_pred_knn = knn.predict(X_test)

# 5. LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train_enc)
y_pred_lda = lda.predict(X_test)

# 6. Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train_enc)
y_pred_gb = gb.predict(X_test)


# на нашем примере без подбора параметров видим такой результат:

# In[20]:


models = {
    'SVM': y_pred_svm,
    'DT': y_pred_dt,
    'NB': y_pred_nb,
    'KNN': y_pred_knn,
    'LDA': y_pred_lda,
    'GB': y_pred_gb
}

for name, y_pred in models.items():
    acc = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average='macro')
    print(f"{name}: Acc={acc:.3f} F1={f1:.3f}")


# # 🔹 По частотам

# In[21]:


import numpy as np
import librosa
import os

# Функция частоты
def get_freq(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)
    return np.mean(mfcc)

cat_freqs = []
dog_freqs = []

for file in os.listdir('cats_dogs/train/cat'):
    freq = get_freq(f'cats_dogs/train/cat/{file}')
    cat_freqs.append(freq)

for file in os.listdir('cats_dogs/train/dog'):
    freq = get_freq(f'cats_dogs/train/dog/{file}')
    dog_freqs.append(freq)

cat_avg = np.mean(cat_freqs)
dog_avg = np.mean(dog_freqs)
threshold = (cat_avg + dog_avg) / 2

print(f"Кошки: {cat_avg:.1f}")
print(f"Собаки: {dog_avg:.1f}")
print(f"Порог: {threshold:.1f}")


# считаем на тесте

# In[22]:


correct = 0
total = 0

for label in ['cat', 'dog']:
    for file in os.listdir(f'cats_dogs/test/{label}'):
        if file.endswith('.wav'):
            freq = get_freq(f'cats_dogs/test/{label}/{file}')
            pred = 'dog' if freq > threshold else 'cat'

            if pred == label:
                correct += 1
            total += 1

print(f"\nAccuracy: {correct}/{total} = {correct/total:.3f}")


# ##

# # Сохранение лучшей модели
# 
# Аналогично модулю 3 для табличных данных, мы сохраним модель через библиотеку ```joblib```

# In[23]:


joblib.dump(best_rf, 'audio_model.pkl');


# In[40]:


joblib.dump(X_train, 'old_X_audio.pkl');
joblib.dump(y_train_enc, 'old_y_audio.pkl');
joblib.dump(X_test, "X_test_audio.pkl");
joblib.dump(y_test_enc, "y_test_audio.pkl");


# # Функция дообучения

# In[44]:


def fine_tuning_audio(new_audio_paths, new_labels) -> dict:
    """
    Корректный fine-tuning аудио-классификатора.
    Модель дообучается на объединении старых + новых данных
    с использованием того же scaler, encoder и фиксированного test-дека.
    """

    # -------------------------------------------------
    # 1. ЗАГРУЗКА СТАРЫХ АРТЕФАКТОВ
    # -------------------------------------------------
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")

    old_X = joblib.load("old_X_audio.pkl")      # numpy array
    old_y = joblib.load("old_y_audio.pkl")      # numpy array

    model = joblib.load("audio_model.pkl")      # sklearn модель

    # -------------------------------------------------
    # 2. ПОДГОТОВКА НОВЫХ ДАННЫХ
    # -------------------------------------------------
    X_new = []
    for path in new_audio_paths:
        feats = extract_features_mfcc(path)  # Твоя функция MFCC → vector
        X_new.append(feats)

    X_new = np.array(X_new)

    # Преобразуем метки через старый encoder
    y_new = encoder.transform(new_labels)

    # применяем тот же scaler, что и при обучении
    X_new_scaled = scaler.transform(X_new)

    # -------------------------------------------------
    # 3. ОБЪЕДИНЕНИЕ СТАРЫХ + НОВЫХ ДАННЫХ
    # -------------------------------------------------
    X_full = np.vstack([old_X, X_new_scaled])
    y_full = np.concatenate([old_y, y_new])

    # -------------------------------------------------
    # 4. ПОВТОРНОЕ ОБУЧЕНИЕ МОДЕЛИ
    # -------------------------------------------------
    model.fit(X_full, y_full)

    # -------------------------------------------------
    # 5. ТЕСТИРОВАНИЕ НА ЗАГРУЖЕННОМ (joblib) TEST НАБОРЕ
    # -------------------------------------------------
    X_test = joblib.load("X_test_audio.pkl")
    y_test = joblib.load("y_test_audio.pkl")

    y_pred = model.predict(X_test)

    # -------------------------------------------------
    # 6. РАСЧЁТ МЕТРИК
    # -------------------------------------------------
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro"), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro"), 4),
    }

    # -------------------------------------------------
    # 7. СОХРАНЕНИЕ ОБНОВЛЁННОЙ МОДЕЛИ И TRAIN ДАННЫХ
    # -------------------------------------------------
    joblib.dump(model, "audio_model.pkl")

    joblib.dump(X_full, "old_X_audio.pkl")
    joblib.dump(y_full, "old_y_audio.pkl")

    return metrics


# In[45]:


# Проверяем функцию

X_new = []
y_new = []

for label in os.listdir(TRAIN_DIR):
    label_path = os.path.join(TRAIN_DIR, label)
    for fname in os.listdir(label_path):
        file_path = os.path.join(label_path, fname)

        X_new.append(file_path)
        y_new.append(label)


fine_tuning_audio(X_new, y_new)

