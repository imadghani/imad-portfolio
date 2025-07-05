{{ config(materialized='table') }}

WITH

distinct_ports AS (
    SELECT DISTINCT 
        CASE 
            WHEN Embarked IS NULL OR Embarked = '' THEN 'Unknown'
            ELSE Embarked 
        END AS Embarked
    FROM 
        {{ ref('titanic') }}
)

SELECT 
    -- Surrogate key
    ROW_NUMBER() OVER (ORDER BY Embarked) AS embarkation_key,
    -- Natural key
    Embarked AS port_code,
    -- Port details
    CASE 
        WHEN Embarked = 'S' THEN 'Southampton'
        WHEN Embarked = 'C' THEN 'Cherbourg'
        WHEN Embarked = 'Q' THEN 'Queenstown'
        ELSE 'Unknown'
    END AS port_name,
    CASE 
        WHEN Embarked = 'S' THEN 'England'
        WHEN Embarked = 'C' THEN 'France'
        WHEN Embarked = 'Q' THEN 'Ireland'
        ELSE 'Unknown'
    END AS country,
    -- Metadata
    CURRENT_TIMESTAMP() AS created_at
FROM 
    distinct_ports
ORDER BY 
    Embarked 