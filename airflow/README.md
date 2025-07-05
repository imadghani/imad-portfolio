# Airflow Local Development Environment

This directory contains a local Airflow setup for orchestrating the Titanic data pipeline using dbt and astronomer-cosmos.

## 🚀 Quick Start

### 1. Start Airflow Services

```bash
# Start both webserver and scheduler
./airflow/start_airflow.sh start

# Or start services individually
./airflow/start_airflow.sh webserver  # Start webserver only
./airflow/start_airflow.sh scheduler  # Start scheduler only
```

### 2. Access Airflow UI

- **URL**: http://localhost:8080
- **Username**: admin
- **Password**: admin

### 3. Run the Pipeline

1. Navigate to the Airflow UI
2. Find the `titanic_dbt_pipeline` DAG
3. Enable the DAG by clicking the toggle switch
4. Trigger the DAG manually or wait for the scheduled run

## 📁 Project Structure

```
airflow/
├── dags/                    # DAG definitions
│   └── titanic_dbt_dag.py  # Main Titanic pipeline DAG
├── logs/                    # Airflow logs
├── plugins/                 # Custom plugins
├── config/                  # Configuration files
├── airflow.cfg             # Airflow configuration
├── airflow.db              # SQLite database
├── start_airflow.sh        # Startup script
└── README.md               # This file
```

## 🔧 Configuration

### Airflow Configuration

The `airflow.cfg` file contains the main configuration:

- **Executor**: SequentialExecutor (for local development)
- **Database**: SQLite (local file-based database)
- **Webserver**: Port 8080
- **Scheduler**: Enabled with optimized settings

### dbt Integration

The DAG uses astronomer-cosmos to integrate with dbt:

- **dbt Project**: `/Users/imadghani/GitHub/imad-portfolio/dbt/core`
- **Target**: BigQuery (imad-455517.prototype_data)
- **Profile**: Uses service account authentication

## 📊 Pipeline Overview

The `titanic_dbt_pipeline` DAG orchestrates the following tasks:

### 1. Pre-flight Checks
- **check_dbt_installation**: Verify dbt is properly installed
- **check_bigquery_connection**: Test BigQuery connectivity

### 2. Data Loading
- **dbt_seed**: Load Titanic CSV data into BigQuery

### 3. Data Transformation (dbt_models Task Group)
- **Dimension Tables**: 
  - dim_passenger
  - dim_passenger_class
  - dim_embarkation
  - dim_cabin
  - dim_ticket
- **Fact Tables**:
  - fact_passenger_journey
- **Analytics Views**:
  - survival_analysis

### 4. Documentation & Testing
- **dbt_docs_generate**: Generate dbt documentation
- **dbt_test**: Run data quality tests

### 5. Summary
- **pipeline_summary**: Provide execution summary

## 🛠️ Manual Commands

### Basic Airflow Commands

```bash
# Set environment
export AIRFLOW_HOME=/Users/imadghani/GitHub/imad-portfolio/airflow

# List DAGs
airflow dags list

# Test DAG
airflow dags test titanic_dbt_pipeline

# Run specific task
airflow tasks test titanic_dbt_pipeline check_dbt_installation 2024-01-01

# Check DAG status
airflow dags state titanic_dbt_pipeline
```

### dbt Commands (for debugging)

```bash
# Navigate to dbt project
cd /Users/imadghani/GitHub/imad-portfolio/dbt/core

# Test dbt connection
dbt debug

# Run dbt models
dbt run

# Run dbt tests
dbt test

# Generate docs
dbt docs generate
dbt docs serve
```

## 🔍 Monitoring & Debugging

### Airflow UI Features

1. **DAG View**: Visual representation of the pipeline
2. **Task Logs**: Detailed execution logs for each task
3. **Gantt Chart**: Task execution timeline
4. **Graph View**: Task dependencies and status

### Log Locations

- **Airflow Logs**: `airflow/logs/`
- **dbt Logs**: `dbt/core/logs/`
- **Task Logs**: Available in Airflow UI

### Common Issues

1. **DAG Import Errors**: Check Python path and imports
2. **dbt Connection Issues**: Verify BigQuery credentials
3. **Task Failures**: Check task logs in Airflow UI

## 🔒 Security Notes

- Default setup uses basic authentication (admin/admin)
- BigQuery service account key is used for authentication
- For production, implement proper authentication and secrets management

## 🚀 Production Considerations

When moving to production:

1. **Executor**: Switch to CeleryExecutor or KubernetesExecutor
2. **Database**: Use PostgreSQL or MySQL instead of SQLite
3. **Authentication**: Implement LDAP, OAuth, or other enterprise auth
4. **Monitoring**: Add monitoring and alerting
5. **Secrets**: Use Airflow Connections and Variables for secrets
6. **Resources**: Configure appropriate resource limits

## 📚 Additional Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [astronomer-cosmos Documentation](https://astronomer.github.io/astronomer-cosmos/)
- [dbt Documentation](https://docs.getdbt.com/)
- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)

## 🆘 Troubleshooting

### Common Commands

```bash
# Reset Airflow database
airflow db reset

# Check Airflow version
airflow version

# List connections
airflow connections list

# Test connection
airflow connections test google_cloud_default
```

### Support

For issues or questions:
1. Check the Airflow UI logs
2. Review the dbt logs
3. Verify BigQuery connectivity
4. Check the GitHub repository for updates 