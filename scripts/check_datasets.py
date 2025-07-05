import os
from dotenv import load_dotenv
from google.cloud import bigquery

# Load environment variables
load_dotenv()

# Set up BQ client
client = bigquery.Client()
project = os.getenv("GCP_PROJECT")

print(f"Project: {project}")
print("\nDatasets in project:")

# List all datasets
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