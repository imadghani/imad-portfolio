{{ config(materialized='table') }}

WITH

passenger_data AS (
    SELECT 
        PassengerId,
        Name,
        Sex,
        Age,
        SibSp,
        Parch,
        -- Extract title from name
        REGEXP_EXTRACT(Name, r', ([^.]+)\.') AS title,
        -- Calculate family size
        SibSp + Parch + 1 AS family_size,
        -- Determine if passenger is alone
        CASE 
            WHEN SibSp + Parch = 0 THEN TRUE 
            ELSE FALSE 
        END AS is_alone,
        -- Age groups
        CASE 
            WHEN Age IS NULL THEN 'Unknown'
            WHEN Age < 18 THEN 'Child'
            WHEN Age < 65 THEN 'Adult'
            ELSE 'Senior'
        END AS age_group
    FROM 
        {{ ref('titanic') }}
)

SELECT 
    -- Surrogate key
    ROW_NUMBER() OVER (ORDER BY PassengerId) AS passenger_key,
    -- Natural key
    PassengerId AS passenger_id,
    -- Passenger details
    Name AS full_name,
    title,
    Sex AS gender,
    Age AS age,
    age_group,
    SibSp AS siblings_spouses,
    Parch AS parents_children,
    family_size,
    is_alone,
    -- Metadata
    CURRENT_TIMESTAMP() AS created_at
FROM 
    passenger_data 