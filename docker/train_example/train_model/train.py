# Источник: sibur_preparing/train_model/train.py
import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

X, y = np.load("X_train.npy"), np.load("y_train.npy")
model = LinearRegression().fit(X, y)
prediction = model.predict(np.load("X_test.npy"))
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/model.pkl")
with open("outputs/predictions.txt", "w", encoding="utf-8") as f:
    f.write(f"Прогноз:\n{prediction}")
