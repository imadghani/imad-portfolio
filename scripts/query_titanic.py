import os
from dotenv import load_dotenv
from google.cloud import bigquery

# Load environment variables
load_dotenv()

# Set up BQ client
client = bigquery.Client()
project = os.getenv("GCP_PROJECT")

# Query the Titanic data
query = f"""
SELECT 
    COUNT(*) as total_passengers,
    AVG(Age) as avg_age,
    SUM(Survived) as survivors,
    ROUND(SUM(Survived) * 100.0 / COUNT(*), 2) as survival_rate_pct,
    COUNT(DISTINCT Pclass) as passenger_classes,
    COUNT(DISTINCT Sex) as gender_count
FROM `{project}.prototype_data.titanic`
"""

print("🚢 Titanic Dataset Summary:")
print("=" * 50)

# Execute query
query_job = client.query(query)
results = query_job.result()

for row in results:
    print(f"Total Passengers: {row.total_passengers}")
    print(f"Average Age: {row.avg_age:.1f} years")
    print(f"Survivors: {row.survivors}")
    print(f"Survival Rate: {row.survival_rate_pct}%")
    print(f"Passenger Classes: {row.passenger_classes}")
    print(f"Gender Count: {row.gender_count}")

print("\n" + "=" * 50)
print("📊 Sample Data (First 5 rows):")
print("=" * 50)

# Query sample data
sample_query = f"""
SELECT 
    PassengerId,
    Survived,
    Pclass,
    Name,
    Sex,
    Age,
    Fare
FROM `{project}.prototype_data.titanic`
ORDER BY PassengerId
LIMIT 5
"""

sample_job = client.query(sample_query)
sample_results = sample_job.result()

for row in sample_results:
    print(f"ID: {row.PassengerId} | Survived: {row.Survived} | Class: {row.Pclass} | {row.Name[:30]}... | {row.Sex} | Age: {row.Age} | Fare: ${row.Fare}") 