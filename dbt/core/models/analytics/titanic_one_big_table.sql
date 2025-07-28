{{ config(
    materialized='table',
    schema='analytics',
    description='Denormalized table combining all Titanic data for analytics and GPT queries',
    tags=['analytics', 'denormalized', 'gpt-ready']
) }}

WITH passenger_journey AS (
    SELECT 
        -- Journey identifiers
        f.journey_key,
        f.passenger_key,
        f.survived_flag,
        
        -- Passenger information
        p.passenger_id,
        p.full_name as passenger_name,
        p.gender,
        p.age,
        p.siblings_spouses,
        p.parents_children,
        p.family_size,
        p.is_alone,
        p.title,
        p.age_group,
        
        -- Ticket information
        t.ticket_number,
        f.fare_amount as ticket_fare,
        t.ticket_type,
        t.is_group_ticket as shared_ticket,
        t.passengers_on_ticket as passengers_per_ticket,
        
        -- Passenger class details
        pc.class_number as passenger_class_number,
        pc.class_name as passenger_class_name,
        pc.class_description as class_amenities,
        
        -- Embarkation details
        e.port_code as embarkation_code,
        e.port_name as embarkation_port,
        e.country as embarkation_country,
        
        -- Cabin information
        c.cabin_number,
        c.deck_letter as cabin_deck,
        c.deck_name as deck_description,
        c.has_cabin
        
    FROM {{ ref('fact_passenger_journey') }} f
    LEFT JOIN {{ ref('dim_passenger') }} p ON f.passenger_key = p.passenger_key
    LEFT JOIN {{ ref('dim_ticket') }} t ON f.ticket_key = t.ticket_key
    LEFT JOIN {{ ref('dim_passenger_class') }} pc ON f.passenger_class_key = pc.passenger_class_key
    LEFT JOIN {{ ref('dim_embarkation') }} e ON f.embarkation_key = e.embarkation_key
    LEFT JOIN {{ ref('dim_cabin') }} c ON f.cabin_key = c.cabin_key
),

enriched_data AS (
    SELECT 
        *,
        
        -- Derived analytics fields for easy querying
        CASE 
            WHEN age IS NULL THEN 'Unknown'
            WHEN age < 18 THEN 'Child'
            WHEN age < 30 THEN 'Young Adult'
            WHEN age < 60 THEN 'Adult'
            ELSE 'Senior'
        END as detailed_age_group,
        
        CASE 
            WHEN family_size = 1 THEN 'Traveling Alone'
            WHEN family_size <= 3 THEN 'Small Family'
            WHEN family_size <= 6 THEN 'Large Family'
            ELSE 'Very Large Family'
        END as family_size_category,
        
        CASE 
            WHEN ticket_fare IS NULL THEN 'Unknown'
            WHEN ticket_fare = 0 THEN 'Free/Staff'
            WHEN ticket_fare < 10 THEN 'Low Fare'
            WHEN ticket_fare < 30 THEN 'Medium Fare'
            WHEN ticket_fare < 100 THEN 'High Fare'
            ELSE 'Premium Fare'
        END as fare_category,
        
        -- Survival insights
        CASE 
            WHEN survived_flag = 1 THEN 'Survived'
            ELSE 'Did Not Survive'
        END as survival_status,
        
        -- Deck analysis
        CASE 
            WHEN cabin_deck IN ('A', 'B', 'C') THEN 'Upper Decks'
            WHEN cabin_deck IN ('D', 'E') THEN 'Middle Decks'
            WHEN cabin_deck IN ('F', 'G') THEN 'Lower Decks'
            WHEN cabin_deck = 'T' THEN 'Tank Top'
            ELSE 'Unknown Deck'
        END as deck_category,
        
        -- Socioeconomic indicators
        CASE 
            WHEN passenger_class_number = 1 AND ticket_fare > 100 THEN 'Wealthy Elite'
            WHEN passenger_class_number = 1 THEN 'Upper Class'
            WHEN passenger_class_number = 2 THEN 'Middle Class'
            WHEN passenger_class_number = 3 AND ticket_fare > 15 THEN 'Working Class'
            ELSE 'Lower Class'
        END as socioeconomic_class,
        
        -- Name analysis
        CASE 
            WHEN title IN ('Mr', 'Master', 'Don', 'Sir', 'Capt', 'Col', 'Major', 'Rev', 'Dr', 'Jonkheer') THEN 'Male'
            WHEN title IN ('Mrs', 'Miss', 'Ms', 'Mme', 'Mlle', 'Lady', 'Countess', 'Dona') THEN 'Female'
            ELSE 'Unknown'
        END as title_gender,
        
        CASE 
            WHEN title IN ('Dr', 'Rev', 'Col', 'Major', 'Capt', 'Sir', 'Lady', 'Countess', 'Jonkheer') THEN 'Nobility/Professional'
            WHEN title IN ('Mr', 'Mrs') THEN 'Married/Adult'
            WHEN title IN ('Miss', 'Master', 'Mlle') THEN 'Unmarried/Young'
            ELSE 'Other'
        END as social_status,
        
        -- Geographic analysis
        CASE 
            WHEN embarkation_country = 'England' THEN 'British Isles'
            WHEN embarkation_country = 'Ireland' THEN 'British Isles'
            WHEN embarkation_country = 'France' THEN 'Continental Europe'
            ELSE 'Unknown Region'
        END as embarkation_region,
        
        -- Financial analysis
        CASE 
            WHEN passengers_per_ticket > 1 THEN ticket_fare / passengers_per_ticket
            ELSE ticket_fare
        END as individual_fare_estimate,
        
        -- Survival factors
        CASE 
            WHEN gender = 'female' AND passenger_class_number IN (1, 2) THEN 'High Survival Probability'
            WHEN gender = 'female' THEN 'Medium-High Survival Probability'
            WHEN age < 18 THEN 'Medium Survival Probability'
            WHEN passenger_class_number = 1 THEN 'Medium Survival Probability'
            ELSE 'Low Survival Probability'
        END as survival_probability_category,
        
        -- Journey metadata
        CURRENT_TIMESTAMP() as analysis_timestamp,
        'dbt_analytics' as data_source,
        '1.0' as schema_version
        
    FROM passenger_journey
)

SELECT 
    -- Primary identifiers
    journey_key,
    passenger_key,
    passenger_id,
    
    -- Passenger demographics
    passenger_name,
    gender,
    age,
    age_group,
    detailed_age_group,
    title,
    title_gender,
    social_status,
    
    -- Family information
    siblings_spouses,
    parents_children,
    family_size,
    family_size_category,
    is_alone,
    
    -- Ticket and financial information
    ticket_number,
    ticket_fare,
    fare_category,
    individual_fare_estimate,
    ticket_type,
    shared_ticket,
    passengers_per_ticket,
    
    -- Class and social status
    passenger_class_number,
    passenger_class_name,
    class_amenities,
    socioeconomic_class,
    
    -- Embarkation details
    embarkation_code,
    embarkation_port,
    embarkation_country,
    embarkation_region,
    
    -- Cabin and location
    cabin_number,
    cabin_deck,
    deck_category,
    deck_description,
    has_cabin,
    
    -- Survival information
    survived_flag,
    survival_status,
    survival_probability_category,
    
    -- Metadata
    analysis_timestamp,
    data_source,
    schema_version
    
FROM enriched_data
ORDER BY passenger_id 