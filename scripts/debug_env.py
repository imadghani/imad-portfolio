import sys
import os
from dotenv import load_dotenv
from google.cloud import bigquery

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("PYTHONPATH:", sys.path[:3])  # Show first 3 paths

# Check if we're in virtual environment
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("✅ Running in virtual environment")
else:
    print("❌ NOT running in virtual environment")

# Check for required packages
try:
    import google.cloud.bigquery
    print("✅ BigQuery package available")
except ImportError as e:
    print(f"❌ BigQuery package missing: {e}")

try:
    import dotenv
    print("✅ python-dotenv package available")
except ImportError as e:
    print(f"❌ python-dotenv package missing: {e}")

# Check environment variables
print(f"GCP_PROJECT: {os.getenv('GCP_PROJECT', 'NOT SET')}")

# Load environment variables
load_dotenv()

# Set up BQ client
client = bigquery.Client()
project = os.getenv("GCP_PROJECT")

datasets = list(client.list_datasets())

if not datasets:
    print("No datasets found.")
else:
    for dataset in datasets:
        print(f"\n📁 Dataset: {dataset.dataset_id}")
        # List tables in each dataset
        tables = list(client.list_tables(f"{project}.{dataset.dataset_id}"))
        if not tables:
            print(f"   No tables found in {dataset.dataset_id}")
        else:
            for table in tables:
                print(f"   📊 Table: {table.table_id}") 