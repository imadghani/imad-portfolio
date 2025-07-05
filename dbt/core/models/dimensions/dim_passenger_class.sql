{{ config(materialized='table') }}

WITH

distinct_classes AS (
    SELECT DISTINCT 
        Pclass 
    FROM 
        {{ ref('titanic') }}
    WHERE 
        Pclass IS NOT NULL
)

SELECT 
    -- Surrogate key
    ROW_NUMBER() OVER (ORDER BY Pclass) AS passenger_class_key,
    -- Natural key
    Pclass AS class_number,
    -- Class details
    CASE 
        WHEN Pclass = 1 THEN 'First Class'
        WHEN Pclass = 2 THEN 'Second Class'
        WHEN Pclass = 3 THEN 'Third Class'
    END AS class_name,
    CASE 
        WHEN Pclass = 1 THEN 'Luxury accommodations, finest dining, premium service'
        WHEN Pclass = 2 THEN 'Comfortable accommodations, good dining, standard service'
        WHEN Pclass = 3 THEN 'Basic accommodations, simple dining, economy service'
    END AS class_description,
    -- Metadata
    CURRENT_TIMESTAMP() AS created_at
FROM 
    distinct_classes
ORDER BY 
    Pclass 