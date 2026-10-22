# Источник: REA/Профессионалы2024/airflow/dags/main.py
# Настроить пути; предобработка до split сохранена из исходника (утечка данных).
from datetime import datetime
import sqlite3
import joblib
import pandas as pd
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def train_model():
    con = sqlite3.connect('../data/db.db')
    sql = 'select * from clustered_data'
    data = pd.read_sql(sql, con=con)
    num_features = data.describe().columns.tolist()
    cat_features = [col for col in data.columns if col not in num_features]
    X = data.drop(['cluster'], axis=1)
    y = data['cluster']
    encoder = LabelEncoder()
    X[cat_features] = X[cat_features].apply(encoder.fit_transform)
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    X = pd.DataFrame(preprocessor.fit_transform(X), columns=preprocessor.get_feature_names_out())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    with open('metric.txt', 'a') as file:
        file.write(f'Обучение модели {datetime.now()}\n')
        file.write(f'Acccuracy: {accuracy_score(y_test, lr_pred):.2f}\n')
        file.write(f'Recall: {recall_score(y_test, lr_pred, average="micro"):.2f}\n')
        file.write(f'F1: {f1_score(y_test, lr_pred, average="micro"):.2f}\n')
    joblib.dump(lr, 'lr_stream.pkl')

with DAG("train_dag", schedule="*/2 * * * *", start_date=datetime(2025, 1, 1), catchup=False) as dag:
    train_operator = PythonOperator(python_callable=train_model, task_id="train_model")
