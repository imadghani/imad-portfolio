# Imad's Data Engineering Portfolio

A comprehensive data engineering project demonstrating modern data stack implementation with **dbt**, **Apache Airflow**, **BigQuery**, and **astronomer-cosmos** for orchestrating dimensional modeling workflows.

## 🏗️ Project Overview

This project showcases a complete data engineering pipeline using the **Titanic dataset** to demonstrate:

- **Dimensional Modeling** with dbt and BigQuery
- **Workflow Orchestration** with Apache Airflow and astronomer-cosmos
- **Data Quality Testing** and documentation
- **Modern Data Stack** best practices

## 🛠️ Tech Stack

- **Data Warehouse**: Google BigQuery
- **Transformation**: dbt (data build tool)
- **Orchestration**: Apache Airflow with astronomer-cosmos
- **Language**: Python, SQL
- **Infrastructure**: Local development with cloud data warehouse

## 📁 Project Structure

```
imad-portfolio/
├── dbt/core/                    # dbt project for data transformations
│   ├── models/
│   │   ├── dimensions/          # Dimension tables (5 tables)
│   │   ├── facts/              # Fact tables (1 table)
│   │   └── analytics/          # Analytics views (1 view)
│   ├── seeds/                  # Raw data (Titanic CSV)
│   ├── macros/                 # Custom dbt macros
│   └── profiles.yml            # dbt BigQuery connection
├── airflow/                    # Airflow orchestration
│   ├── dags/                   # DAG definitions
│   ├── config/                 # Airflow configuration
│   └── start_airflow.sh        # Startup script
├── scripts/                    # Python utilities
└── secrets/                    # Service account credentials
```

## 📊 Data Model

### Dimensional Model Architecture

The project implements a **star schema** with the Titanic dataset:

#### Dimension Tables
- **`dim_passenger`** - Passenger demographics and details
- **`dim_ticket`** - Ticket information and pricing
- **`dim_passenger_class`** - Passenger class details
- **`dim_embarkation`** - Embarkation port information  
- **`dim_cabin`** - Cabin location and deck information

#### Fact Table
- **`fact_passenger_journey`** - Central fact table linking all dimensions

#### Analytics Layer
- **`survival_analysis`** - Pre-aggregated survival statistics by various dimensions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Cloud Platform account with BigQuery enabled
- Service account with BigQuery permissions

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd imad-portfolio

# Create virtual environment
python -m venv python_venv
source python_venv/bin/activate  # On Windows: python_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure BigQuery

1. Create a service account in Google Cloud Console
2. Download the JSON key file
3. Place it in `secrets/bigquery-service-account.json`
4. Update `dbt/core/profiles.yml` with your project details

### 3. Start Airflow

```bash
# Start both webserver and scheduler
./airflow/start_airflow.sh

# Or start individually
./airflow/start_airflow.sh webserver
./airflow/start_airflow.sh scheduler
```

### 4. Access Airflow UI

- **URL**: http://localhost:8080
- **Username**: admin
- **Password**: admin

### 5. Run the Pipeline

1. Navigate to the Airflow UI
2. Find the `titanic_dbt_pipeline` DAG
3. Toggle it ON and trigger a run

## 🔧 Key Features

### dbt Implementation
- **Incremental models** for efficient data processing
- **Custom macros** for reusable transformations
- **Data quality tests** ensuring data integrity
- **Documentation** with model descriptions and column definitions

### Airflow Orchestration
- **Pre-flight checks** for environment validation
- **dbt task groups** using astronomer-cosmos
- **Error handling** and retry mechanisms
- **Dependency management** between tasks

### Data Quality
- **Source data validation** before processing
- **Model testing** with dbt tests
- **Data lineage** tracking through dbt docs

## 📈 Pipeline Workflow

The Airflow DAG (`titanic_dbt_pipeline`) orchestrates:

1. **Pre-flight Checks** - Validate environment and connections
2. **Data Seeding** - Load raw Titanic data to BigQuery
3. **Dimensional Modeling** - Build star schema with dbt
4. **Data Quality Testing** - Run dbt tests
5. **Documentation** - Generate dbt docs
6. **Pipeline Summary** - Log completion status

## 🧪 Available Scripts

### Data Exploration
- **`scripts/explore_dimensional_model.py`** - Analyze the dimensional model
- **`scripts/query_titanic.py`** - Query examples and data exploration

### Utilities
- **`scripts/check_datasets.py`** - Verify BigQuery datasets
- **`scripts/cleanup_titanic.py`** - Clean up test data
- **`scripts/test_bigquery.py`** - Test BigQuery connectivity

## 🔍 Data Insights

The dimensional model enables analysis of:
- **Survival rates** by passenger class, gender, age groups
- **Ticket pricing** patterns and correlations
- **Embarkation port** demographics
- **Cabin location** impact on survival
- **Family relationships** and survival patterns

## 📚 Documentation

### dbt Documentation
Generate and view dbt documentation:
```bash
cd dbt/core
dbt docs generate
dbt docs serve
```

### Model Lineage
The dbt docs provide interactive lineage graphs showing data flow from raw data through dimensions to analytics.

## 🛡️ Data Quality & Testing

- **Source freshness** checks
- **Referential integrity** tests
- **Data completeness** validation
- **Business logic** verification

## 🔧 Configuration

### dbt Configuration
- **`dbt_project.yml`** - Project settings and model configs
- **`profiles.yml`** - BigQuery connection details
- **Model configs** - Materialization strategies and schemas

### Airflow Configuration
- **`airflow.cfg`** - Airflow settings optimized for local development
- **DAG configuration** - Retry policies and scheduling
- **Connection management** - BigQuery and dbt integrations

## 🚨 Troubleshooting

### Common Issues

1. **BigQuery Authentication**
   - Verify service account permissions
   - Check file path in profiles.yml

2. **Airflow Startup**
   - Ensure port 8080 is available
   - Check database initialization

3. **dbt Connection**
   - Run `dbt debug` to verify setup
   - Validate BigQuery project/dataset names

## 📊 Performance Considerations

- **Incremental models** for large datasets
- **Partitioning** strategies in BigQuery
- **Clustering** for query optimization
- **Resource management** in Airflow

## 🔮 Future Enhancements

- [ ] Add more data sources
- [ ] Implement data quality monitoring
- [ ] Add CI/CD pipeline
- [ ] Containerize with Docker
- [ ] Add real-time streaming components
- [ ] Implement data cataloging

## 🤝 Contributing

This is a portfolio project, but suggestions and improvements are welcome!

## 📄 License

This project is for educational and portfolio purposes.

---

**Built with ❤️ by Imad** - Demonstrating modern data engineering practices with real-world tools and techniques.
