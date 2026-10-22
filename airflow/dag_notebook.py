# Источник: REA/Preparing_Rea2026123/AirFlow/airflow.ipynb
# Заменить <USER>; файл положить в $AIRFLOW_HOME/dags/.
from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

USER = "<USER>"
NOTEBOOK_IN = f"/home/{USER}/ml_project/notebooks/train.ipynb"
OUT_ROOT = f"/home/{USER}/Desktop/c_m"

with DAG(
    dag_id="one_night_train_to_c_m", start_date=datetime(2025, 1, 1),
    schedule=None, catchup=False, tags=["ml", "one-off"],
) as dag:
    prepare_dir = BashOperator(
        task_id="prepare_dir",
        bash_command=f'mkdir -p "{OUT_ROOT}/{{{{ ds_nodash }}}}"',
    )
    run_notebook = BashOperator(
        task_id="run_notebook",
        bash_command=f'papermill "{NOTEBOOK_IN}" "{OUT_ROOT}/executed_train_{{{{ ds_nodash }}}}.ipynb" '
                     f'-p output_dir "{OUT_ROOT}/{{{{ ds_nodash }}}}"',
    )
    check_outputs = BashOperator(
        task_id="check_outputs",
        bash_command=f'test -f "{OUT_ROOT}/{{{{ ds_nodash }}}}/metrics.json" && '
                     f'test -f "{OUT_ROOT}/{{{{ ds_nodash }}}}/model.pt"',
    )
    prepare_dir >> run_notebook >> check_outputs
