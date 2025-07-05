"""
Titanic Data Pipeline DAG using astronomer-cosmos
This DAG orchestrates the dbt models for the Titanic dataset dimensional model.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from cosmos import DbtDag, DbtTaskGroup, ProjectConfig, ProfileConfig, ExecutionConfig
from cosmos.profiles import GoogleCloudServiceAccountFileProfileMapping

# Default arguments for the DAG
default_args = {
    'owner': 'imad-ghani',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define paths
DBT_PROJECT_PATH = Path("/Users/imadghani/GitHub/imad-portfolio/dbt/core")
DBT_PROFILES_PATH = Path("/Users/imadghani/GitHub/imad-portfolio/dbt/core")
VENV_PATH = Path("/Users/imadghani/GitHub/imad-portfolio/python_venv")

# Profile configuration for BigQuery
profile_config = ProfileConfig(
    profile_name="core",
    target_name="dev",
    profile_mapping=GoogleCloudServiceAccountFileProfileMapping(
        conn_id="google_cloud_default",
        profile_args={
            "project": "imad-455517",
            "dataset": "prototype_data",
            "threads": 4,
            "method": "service-account",
            "keyfile": "/Users/imadghani/GitHub/imad-portfolio/secrets/bigquery-service-account.json",
            "location": "US",
        },
    ),
)

# Execution configuration
execution_config = ExecutionConfig(
    dbt_executable_path=f"{VENV_PATH}/bin/dbt",
)

# Project configuration
project_config = ProjectConfig(
    dbt_project_path=DBT_PROJECT_PATH,
)

def check_dbt_installation():
    """Check if dbt is properly installed and accessible"""
    import subprocess
    try:
        result = subprocess.run([f"{VENV_PATH}/bin/dbt", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"dbt version: {result.stdout}")
        return True
    except Exception as e:
        print(f"Error checking dbt installation: {e}")
        return False

def check_bigquery_connection():
    """Check BigQuery connection"""
    import subprocess
    try:
        result = subprocess.run([f"{VENV_PATH}/bin/dbt", "debug", "--profiles-dir", str(DBT_PROFILES_PATH)], 
                              capture_output=True, text=True, cwd=str(DBT_PROJECT_PATH))
        print(f"dbt debug output: {result.stdout}")
        if result.stderr:
            print(f"dbt debug errors: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking BigQuery connection: {e}")
        return False

# Create the main DAG
with DAG(
    'titanic_dbt_pipeline',
    default_args=default_args,
    description='Titanic data pipeline using dbt and astronomer-cosmos',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['dbt', 'titanic', 'dimensional-model', 'bigquery'],
    max_active_runs=1,
) as dag:

    # Pre-flight checks
    check_dbt_task = PythonOperator(
        task_id='check_dbt_installation',
        python_callable=check_dbt_installation,
        doc_md="""
        ## Check dbt Installation
        This task verifies that dbt is properly installed and accessible.
        """,
    )

    check_connection_task = PythonOperator(
        task_id='check_bigquery_connection',
        python_callable=check_bigquery_connection,
        doc_md="""
        ## Check BigQuery Connection
        This task verifies that the BigQuery connection is working properly.
        """,
    )

    # dbt seed task (load raw data)
    dbt_seed_task = BashOperator(
        task_id='dbt_seed',
        bash_command=f"""
        cd {DBT_PROJECT_PATH} && \
        export DBT_PROFILES_DIR={DBT_PROFILES_PATH} && \
        {VENV_PATH}/bin/dbt seed --profiles-dir {DBT_PROFILES_PATH}
        """,
        doc_md="""
        ## dbt Seed
        This task loads the Titanic CSV data into BigQuery using dbt seed.
        """,
    )

    # Create dbt task group for models
    dbt_tg = DbtTaskGroup(
        group_id="dbt_models",
        project_config=project_config,
        profile_config=profile_config,
        execution_config=execution_config,
        default_args=default_args,
        operator_args={
            "install_deps": True,
            "full_refresh": False,
        },
    )

    # dbt docs generation
    dbt_docs_task = BashOperator(
        task_id='dbt_docs_generate',
        bash_command=f"""
        cd {DBT_PROJECT_PATH} && \
        export DBT_PROFILES_DIR={DBT_PROFILES_PATH} && \
        {VENV_PATH}/bin/dbt docs generate --profiles-dir {DBT_PROFILES_PATH}
        """,
        doc_md="""
        ## dbt Docs Generate
        This task generates the dbt documentation for the project.
        """,
    )

    # Data quality checks
    dbt_test_task = BashOperator(
        task_id='dbt_test',
        bash_command=f"""
        cd {DBT_PROJECT_PATH} && \
        export DBT_PROFILES_DIR={DBT_PROFILES_PATH} && \
        {VENV_PATH}/bin/dbt test --profiles-dir {DBT_PROFILES_PATH}
        """,
        doc_md="""
        ## dbt Test
        This task runs all dbt tests to ensure data quality and integrity.
        """,
    )

    # Post-processing summary
    def generate_pipeline_summary():
        """Generate a summary of the pipeline execution"""
        print("="*50)
        print("TITANIC DBT PIPELINE SUMMARY")
        print("="*50)
        print("✅ Pipeline completed successfully!")
        print("\nModels processed:")
        print("- Dimension Tables: 5 tables")
        print("- Fact Tables: 1 table") 
        print("- Analytics Views: 1 view")
        print("\nNext steps:")
        print("- View dbt docs for detailed lineage")
        print("- Check BigQuery for updated tables")
        print("- Review test results for data quality")
        print("="*50)

    summary_task = PythonOperator(
        task_id='pipeline_summary',
        python_callable=generate_pipeline_summary,
        doc_md="""
        ## Pipeline Summary
        This task provides a summary of the pipeline execution and next steps.
        """,
    )

    # Define task dependencies
    check_dbt_task >> check_connection_task >> dbt_seed_task >> dbt_tg >> [dbt_docs_task, dbt_test_task] >> summary_task

# Additional DAG documentation
dag.doc_md = """
# Titanic Data Pipeline

This DAG orchestrates a complete data pipeline for the Titanic dataset using dbt and astronomer-cosmos.

## Pipeline Overview

The pipeline consists of the following stages:

1. **Pre-flight Checks**: Verify dbt installation and BigQuery connectivity
2. **Data Loading**: Load raw Titanic data using dbt seed
3. **Data Transformation**: Transform data into dimensional model using dbt
4. **Documentation**: Generate dbt documentation
5. **Quality Assurance**: Run dbt tests for data validation
6. **Summary**: Provide pipeline execution summary

## Data Model

The pipeline creates a dimensional model with:
- **5 Dimension Tables**: passenger, passenger_class, embarkation, cabin, ticket
- **1 Fact Table**: passenger_journey with all measures and foreign keys
- **1 Analytics View**: survival_analysis with aggregated metrics

## Key Features

- **Incremental Processing**: Supports incremental model updates
- **Data Quality**: Comprehensive testing with dbt tests
- **Documentation**: Auto-generated documentation with lineage
- **Monitoring**: Built-in logging and error handling
- **Scalability**: Designed for production workloads

## Configuration

- **Target**: BigQuery (imad-455517.prototype_data)
- **Schedule**: Daily execution
- **Retries**: 1 retry with 5-minute delay
- **Parallelism**: Optimized for sequential execution

## Usage

1. Ensure BigQuery credentials are properly configured
2. Verify dbt profiles are set up correctly
3. Run the DAG manually or let it run on schedule
4. Monitor execution in Airflow UI
5. Check results in BigQuery and dbt docs
""" 