# Imad's Data Engineering Portfolio

A comprehensive data engineering project demonstrating modern data stack implementation with **dbt**, **Apache Airflow**, **BigQuery**, and **astronomer-cosmos**, plus interactive applications for data visualization and ML model serving.

## 🏗️ Project Overview

This project showcases a complete data engineering ecosystem using the **Titanic dataset** to demonstrate:

- **Dimensional Modeling** with dbt and BigQuery
- **Workflow Orchestration** with Apache Airflow and astronomer-cosmos
- **Interactive Data Dashboard** with Streamlit and Plotly
- **ML Model Training & Serving** with scikit-learn and Flask
- **Pipeline Monitoring** with real-time Airflow dashboard
- **Modern Data Stack** best practices

## 🛠️ Tech Stack

- **Data Warehouse**: Google BigQuery
- **Transformation**: dbt (data build tool)
- **Orchestration**: Apache Airflow with astronomer-cosmos
- **Visualization**: Streamlit, Plotly, Seaborn
- **ML**: scikit-learn, Flask API
- **Language**: Python, SQL
- **Infrastructure**: Local development with cloud data warehouse

## 📁 Project Structure

```
imad-portfolio/
├── apps/                        # Interactive applications
│   ├── dashboard/              # Streamlit data dashboard
│   ├── ml_model/               # ML model training & API
│   └── pipeline_monitor/       # Airflow monitoring dashboard
├── dbt/core/                   # dbt project for data transformations
│   ├── models/
│   │   ├── dimensions/         # Dimension tables (5 tables)
│   │   ├── facts/             # Fact tables (1 table)
│   │   └── analytics/         # Analytics views (1 view)
│   ├── seeds/                 # Raw data (Titanic CSV)
│   └── profiles.yml           # dbt BigQuery connection
├── airflow/                   # Airflow orchestration
│   ├── dags/                  # DAG definitions
│   ├── start_airflow.sh       # Startup script
│   └── setup_env.sh           # Environment setup
├── scripts/                   # Python utilities
└── secrets/                   # Service account credentials
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

- Python 3.12+ (compatibility fixes included)
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

### 3. Start Applications

#### Data Pipeline (Airflow)
```bash
cd airflow
./start_airflow.sh standalone
# Access: http://localhost:8080 (admin/admin)
```

#### Data Dashboard
```bash
cd apps/dashboard
./run_dashboard.sh
# Access: http://localhost:8501
```

#### ML Model API
```bash
cd apps/ml_model
./run_ml_model.sh api
# Access: http://localhost:5001
```

#### Pipeline Monitor
```bash
cd apps/pipeline_monitor
./run_pipeline_monitor.sh
# Access: http://localhost:8502
```

## 🔧 Key Features

### Data Pipeline (Airflow + dbt)
- **Automated orchestration** with astronomer-cosmos
- **Data quality testing** and validation
- **Dimensional modeling** with star schema
- **Error handling** and retry mechanisms
- **Python 3.12 compatibility** (Flask, WTForms fixes)

### Interactive Dashboard
- **Real-time data visualization** with Plotly
- **Survival analysis** with interactive filters
- **Demographic insights** and geographic analysis
- **Data export** capabilities

### ML Model Serving
- **Multiple algorithms** (Random Forest, Gradient Boosting, SVM)
- **Model comparison** and performance metrics
- **REST API** for predictions
- **Batch prediction** support

### Pipeline Monitoring
- **Real-time DAG monitoring** from Airflow database
- **Task success rates** and performance metrics
- **Log analysis** and troubleshooting
- **Auto-refresh** capabilities

## 📈 Pipeline Workflow

The Airflow DAG (`titanic_dbt_pipeline`) orchestrates:

1. **Pre-flight Checks** - Validate environment and connections
2. **Data Seeding** - Load raw Titanic data to BigQuery
3. **Dimensional Modeling** - Build star schema with dbt
4. **Data Quality Testing** - Run dbt tests
5. **Pipeline Summary** - Log completion status

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

## 🛡️ Data Quality & Testing

- **Source freshness** checks
- **Referential integrity** tests
- **Data completeness** validation
- **Business logic** verification
- **ML model validation** with cross-validation

## 🔧 Configuration

### Environment Setup
Use the provided setup script for consistent environment configuration:
```bash
cd airflow
source setup_env.sh
```

### dbt Configuration
- **`dbt_project.yml`** - Project settings and model configs
- **`profiles.yml`** - BigQuery connection details
- **Model configs** - Materialization strategies and schemas

### Airflow Configuration
- **`airflow.cfg`** - Optimized for local development
- **Example DAGs disabled** - Only shows your custom DAG
- **Proper AIRFLOW_HOME** configuration

## 🚨 Troubleshooting

### Common Issues

#### Python 3.12 Compatibility
- **Flask compatibility** - Fixed with Flask 2.2.5
- **WTForms compatibility** - Fixed with WTForms 3.2.1
- **cgi.escape deprecation** - Resolved with proper package versions

#### Environment Issues
- **AIRFLOW_HOME not set** - Use `source setup_env.sh`
- **DAG not found** - Ensure correct environment variables
- **BigQuery permissions** - Check service account credentials

#### Port Conflicts
- **Dashboard**: http://localhost:8501
- **ML API**: http://localhost:5001
- **Pipeline Monitor**: http://localhost:8502
- **Airflow**: http://localhost:8080

## 📚 Documentation

### dbt Documentation
```bash
cd dbt/core
dbt docs generate
dbt docs serve
```

### API Documentation
- **ML Model API**: http://localhost:5001/health
- **Model Info**: http://localhost:5001/model/info
- **Prediction**: POST to http://localhost:5001/predict

## 🎯 Live Applications

1. **Data Dashboard**: Interactive Titanic analysis with filters and exports
2. **ML Model API**: Survival prediction service with 80.45% accuracy
3. **Pipeline Monitor**: Real-time Airflow monitoring and analytics
4. **Airflow UI**: Complete workflow orchestration and monitoring

This portfolio demonstrates production-ready data engineering practices with modern tools and frameworks, showcasing the complete data lifecycle from ingestion to insights.

## 🤝 Contributing

This is a portfolio project, but suggestions and improvements are welcome!

## 📄 License

This project is for educational and portfolio purposes.

---

**Built with ❤️ by Imad** - Demonstrating modern data engineering practices with real-world tools and techniques.
