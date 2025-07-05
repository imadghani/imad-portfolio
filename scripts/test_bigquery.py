import os
from dotenv import load_dotenv
from google.cloud import bigquery

# Load environment variables
load_dotenv()

# Check path to credentials
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
print(f"Using credentials from: {cred_path}")

# Set up BQ client
client = bigquery.Client()
project = os.getenv("GCP_PROJECT")
dataset = os.getenv("BQ_DATASET")

# List tables in your dataset
tables = list(client.list_tables(f"{project}.{dataset}"))
if not tables:
    print(f"No tables found in dataset `{dataset}`.")
else:
    print(f"Tables in `{dataset}`:")
    for table in tables:
        print(f" - {table.table_id}")
