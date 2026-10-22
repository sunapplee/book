# Источник: REA/Preparing_Rea2026123/Module 4/API/tables.py
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from sklearn import metrics

def inference_table(test: pd.DataFrame):
    cb_model = CatBoostRegressor()
    cb_model.load_model('table_models/catboost_model_regression.cbm')
    nn_model = joblib.load('table_models/nn_model_regression.pkl')
    scaler = joblib.load('table_models/scaler.pkl')
    test = scaler.transform(test)
    y_pred_cb = cb_model.predict(test)
    y_pred_nn = nn_model.predict(test)
    return {
        'catboost': y_pred_cb.round(3)[0],
        'neuralnet': y_pred_nn.round(3)[0]
    }

def fine_tuning_regression(new_data: pd.DataFrame) -> dict:
    scaler = joblib.load('table_models/scaler.pkl')
    old_X = joblib.load('table_models/old_X_train_reg.pkl')
    old_y = joblib.load('table_models/old_y_train_reg.pkl')
    cb_model = CatBoostRegressor()
    cb_model.load_model('table_models/catboost_model_regression.cbm')
    nn_model = joblib.load('table_models/nn_model_regression.pkl')
    if 'AQI' not in new_data.columns:
        raise ValueError("В new_data должен быть столбец 'AQI'.")
    X_new = new_data.drop('AQI', axis=1)
    y_new = new_data['AQI'].astype(float)
    X_new_scaled = scaler.transform(X_new)
    X_full = np.vstack([old_X, X_new_scaled])
    y_full = np.concatenate([old_y, y_new])
    nn_model.fit(X_full, y_full)
    cb_model.fit(X_full, y_full)
    X_test_scaled = joblib.load('table_models/X_test_reg.pkl')
    y_test = joblib.load('table_models/y_test_reg.pkl')
    y_pred_cb = cb_model.predict(X_test_scaled)
    y_pred_nn = nn_model.predict(X_test_scaled)
    model_metrics = {
        "CatBoostRegressor": {
            "MAE": round(metrics.mean_absolute_error(y_test, y_pred_cb), 3),
            "RMSE": round(metrics.root_mean_squared_error(y_test, y_pred_cb), 3),
            "R2": round(metrics.r2_score(y_test, y_pred_cb), 3),
        },
        "MLPRegressor": {
            "MAE": round(metrics.mean_absolute_error(y_test, y_pred_nn), 3),
            "RMSE": round(metrics.root_mean_squared_error(y_test, y_pred_nn), 3),
            "R2": round(metrics.r2_score(y_test, y_pred_nn), 3),
        }
    }
    cb_model.save_model('table_models/catboost_model_regression.cbm')
    joblib.dump(nn_model, 'table_models/nn_model_regression.pkl')
    joblib.dump(X_full, 'table_models/old_X_train_reg.pkl')
    joblib.dump(y_full, 'table_models/old_y_train_reg.pkl')
    return model_metrics
