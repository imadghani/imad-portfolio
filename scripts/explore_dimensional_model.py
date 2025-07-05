import os
from dotenv import load_dotenv
from google.cloud import bigquery
import pandas as pd

# Load environment variables
load_dotenv()

# Set up BQ client
client = bigquery.Client()
project = os.getenv("GCP_PROJECT")

print("🚢 TITANIC DIMENSIONAL MODEL EXPLORATION")
print("=" * 60)

# 1. Dimension Tables Overview
print("\n📊 DIMENSION TABLES OVERVIEW:")
print("-" * 40)

dimensions = [
    ("dim_passenger", "Passenger demographics"),
    ("dim_passenger_class", "Ticket class information"),
    ("dim_embarkation", "Port of embarkation"),
    ("dim_cabin", "Cabin location details"),
    ("dim_ticket", "Ticket information")
]

for table_name, description in dimensions:
    query = f"SELECT COUNT(*) as count FROM `{project}.prototype_data.{table_name}`"
    result = client.query(query).result()
    count = list(result)[0].count
    print(f"• {table_name}: {count} records - {description}")

# 2. Fact Table Overview
print("\n📈 FACT TABLE OVERVIEW:")
print("-" * 40)
fact_query = f"""
SELECT 
    COUNT(*) as total_journeys,
    SUM(survived_flag) as total_survivors,
    ROUND(AVG(survival_rate) * 100, 2) as overall_survival_rate,
    ROUND(AVG(fare_amount), 2) as avg_fare,
    ROUND(AVG(family_size), 1) as avg_family_size
FROM `{project}.prototype_data.fact_passenger_journey`
"""

result = client.query(fact_query).result()
for row in result:
    print(f"• Total Journeys: {row.total_journeys}")
    print(f"• Total Survivors: {row.total_survivors}")
    print(f"• Overall Survival Rate: {row.overall_survival_rate}%")
    print(f"• Average Fare: ${row.avg_fare}")
    print(f"• Average Family Size: {row.avg_family_size}")

# 3. Survival Analysis by Class and Gender
print("\n🎯 SURVIVAL ANALYSIS BY CLASS AND GENDER:")
print("-" * 50)
survival_query = f"""
SELECT 
    dpc.class_name,
    dp.gender,
    COUNT(*) as passengers,
    SUM(f.survived_flag) as survivors,
    ROUND(AVG(f.survival_rate) * 100, 2) as survival_rate_pct
FROM `{project}.prototype_data.fact_passenger_journey` f
JOIN `{project}.prototype_data.dim_passenger` dp ON f.passenger_key = dp.passenger_key
JOIN `{project}.prototype_data.dim_passenger_class` dpc ON f.passenger_class_key = dpc.passenger_class_key
GROUP BY dpc.class_name, dp.gender
ORDER BY dpc.class_name, dp.gender
"""

result = client.query(survival_query).result()
for row in result:
    print(f"• {row.class_name} - {row.gender}: {row.survivors}/{row.passengers} ({row.survival_rate_pct}%)")

# 4. Embarkation Port Analysis
print("\n🌍 EMBARKATION PORT ANALYSIS:")
print("-" * 40)
port_query = f"""
SELECT 
    de.port_name,
    de.country,
    COUNT(*) as passengers,
    SUM(f.survived_flag) as survivors,
    ROUND(AVG(f.survival_rate) * 100, 2) as survival_rate_pct,
    ROUND(AVG(f.fare_amount), 2) as avg_fare
FROM `{project}.prototype_data.fact_passenger_journey` f
JOIN `{project}.prototype_data.dim_embarkation` de ON f.embarkation_key = de.embarkation_key
GROUP BY de.port_name, de.country
ORDER BY passengers DESC
"""

result = client.query(port_query).result()
for row in result:
    print(f"• {row.port_name}, {row.country}: {row.passengers} passengers, {row.survival_rate_pct}% survival, ${row.avg_fare} avg fare")

# 5. Age Group Analysis
print("\n👥 AGE GROUP SURVIVAL ANALYSIS:")
print("-" * 40)
age_query = f"""
SELECT 
    dp.age_group,
    COUNT(*) as passengers,
    SUM(f.survived_flag) as survivors,
    ROUND(AVG(f.survival_rate) * 100, 2) as survival_rate_pct,
    ROUND(AVG(f.fare_amount), 2) as avg_fare
FROM `{project}.prototype_data.fact_passenger_journey` f
JOIN `{project}.prototype_data.dim_passenger` dp ON f.passenger_key = dp.passenger_key
GROUP BY dp.age_group
ORDER BY 
    CASE dp.age_group 
        WHEN 'Child' THEN 1
        WHEN 'Adult' THEN 2
        WHEN 'Senior' THEN 3
        ELSE 4
    END
"""

result = client.query(age_query).result()
for row in result:
    print(f"• {row.age_group}: {row.passengers} passengers, {row.survival_rate_pct}% survival, ${row.avg_fare} avg fare")

# 6. Family Size Impact
print("\n👨‍👩‍👧‍👦 FAMILY SIZE IMPACT ON SURVIVAL:")
print("-" * 40)
family_query = f"""
SELECT 
    CASE 
        WHEN f.family_size = 1 THEN 'Alone'
        WHEN f.family_size BETWEEN 2 AND 4 THEN 'Small Family (2-4)'
        WHEN f.family_size BETWEEN 5 AND 7 THEN 'Large Family (5-7)'
        ELSE 'Very Large Family (8+)'
    END as family_category,
    COUNT(*) as passengers,
    SUM(f.survived_flag) as survivors,
    ROUND(AVG(f.survival_rate) * 100, 2) as survival_rate_pct
FROM `{project}.prototype_data.fact_passenger_journey` f
GROUP BY family_category
ORDER BY AVG(f.family_size)
"""

result = client.query(family_query).result()
for row in result:
    print(f"• {row.family_category}: {row.passengers} passengers, {row.survival_rate_pct}% survival")

print("\n" + "=" * 60)
print("🎉 Dimensional Model Successfully Created!")
print("💡 Use these tables for advanced analytics, ML models, and dashboards!")
print("=" * 60) 