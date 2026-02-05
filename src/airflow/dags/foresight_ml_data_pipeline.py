from datetime import datetime

from airflow import DAG
from airflow.providers.google.cloud.operators.run import (
    CloudRunExecuteJobOperator,
)

PROJECT_ID = "financial-distress-ew"
REGION = "us-central1"

with DAG(
    dag_id="foresight_ingestion",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["foresight-ml", "ingestion"],
) as dag:

    run_sec_ingestion = CloudRunExecuteJobOperator(
        task_id="run_sec_ingestion",
        project_id=PROJECT_ID,
        region=REGION,
        job_name="sec-ingestion",
        overrides={
            "containerOverrides": [
                {
                    "env": [
                        {"name": "EXECUTION_DATE", "value": "{{ ds }}"},
                    ]
                }
            ]
        },
    )

    run_fred_ingestion = CloudRunExecuteJobOperator(
        task_id="run_fred_ingestion",
        project_id=PROJECT_ID,
        region=REGION,
        job_name="fred-ingestion",
        overrides={
            "containerOverrides": [
                {
                    "env": [
                        {"name": "EXECUTION_DATE", "value": "{{ ds }}"},
                    ]
                }
            ]
        },
    )

    run_sec_ingestion >> run_fred_ingestion
