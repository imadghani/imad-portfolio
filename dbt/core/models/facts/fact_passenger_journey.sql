{{ config(materialized='table') }}

SELECT 
    -- Surrogate key for fact table
    ROW_NUMBER() OVER (ORDER BY t.PassengerId) AS journey_key,
    
    -- Foreign keys to dimensions
    dp.passenger_key,
    dpc.passenger_class_key,
    de.embarkation_key,
    dc.cabin_key,
    dt.ticket_key,
    
    -- Measures (facts)
    t.Fare AS fare_amount,
    t.Age AS age,
    t.SibSp + t.Parch + 1 AS family_size,
    t.Survived AS survived_flag,
    
    -- Additional calculated measures
    CASE 
        WHEN t.Survived = 1 THEN 1.0
        ELSE 0.0
    END AS survival_rate,
    
    CASE 
        WHEN t.SibSp + t.Parch = 0 THEN 1
        ELSE 0
    END AS is_alone_flag,
    
    -- Metadata
    CURRENT_TIMESTAMP() AS created_at
    
FROM 
    {{ ref('titanic') }} AS t
    LEFT JOIN {{ ref('dim_passenger') }} AS dp 
        ON t.PassengerId = dp.passenger_id
    LEFT JOIN {{ ref('dim_passenger_class') }} AS dpc 
        ON t.Pclass = dpc.class_number
    LEFT JOIN {{ ref('dim_embarkation') }} AS de 
        ON COALESCE(t.Embarked, 'Unknown') = de.port_code
    LEFT JOIN {{ ref('dim_cabin') }} AS dc 
        ON COALESCE(t.Cabin, 'Unknown') = dc.cabin_number
    LEFT JOIN {{ ref('dim_ticket') }} AS dt 
        ON t.Ticket = dt.ticket_number 