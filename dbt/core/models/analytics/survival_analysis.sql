{{ config(materialized='view') }}

SELECT 
    -- Passenger class analysis
    dpc.class_name,
    dp.gender,
    dp.age_group,
    de.port_name,
    dc.deck_name,
    
    -- Survival metrics
    COUNT(*) AS total_passengers,
    SUM(f.survived_flag) AS survivors,
    ROUND(AVG(f.survival_rate) * 100, 2) AS survival_rate_pct,
    
    -- Fare analysis
    ROUND(AVG(f.fare_amount), 2) AS avg_fare,
    ROUND(MIN(f.fare_amount), 2) AS min_fare,
    ROUND(MAX(f.fare_amount), 2) AS max_fare,
    
    -- Family analysis
    ROUND(AVG(f.family_size), 1) AS avg_family_size,
    SUM(f.is_alone_flag) AS alone_passengers,
    ROUND(SUM(f.is_alone_flag) * 100.0 / COUNT(*), 2) AS alone_percentage

FROM 
    {{ ref('fact_passenger_journey') }} AS f
    LEFT JOIN {{ ref('dim_passenger') }} AS dp 
        ON f.passenger_key = dp.passenger_key
    LEFT JOIN {{ ref('dim_passenger_class') }} AS dpc 
        ON f.passenger_class_key = dpc.passenger_class_key
    LEFT JOIN {{ ref('dim_embarkation') }} AS de 
        ON f.embarkation_key = de.embarkation_key
    LEFT JOIN {{ ref('dim_cabin') }} AS dc 
        ON f.cabin_key = dc.cabin_key

GROUP BY 
    dpc.class_name,
    dp.gender,
    dp.age_group,
    de.port_name,
    dc.deck_name

ORDER BY 
    dpc.class_name,
    dp.gender,
    dp.age_group 