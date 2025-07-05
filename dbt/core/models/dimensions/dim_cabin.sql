{{ config(materialized='table') }}

WITH
cabin_data AS (
    SELECT DISTINCT
        CASE 
            WHEN Cabin IS NULL OR Cabin = '' THEN 'Unknown'
            ELSE Cabin 
        END AS cabin_number,
        -- Extract deck letter (first character)
        CASE 
            WHEN Cabin IS NULL OR Cabin = '' THEN 'Unknown'
            ELSE SUBSTR(Cabin, 1, 1)
        END AS deck_letter,
        -- Determine if cabin is known
        CASE 
            WHEN Cabin IS NULL OR Cabin = '' THEN FALSE
            ELSE TRUE
        END AS has_cabin
    FROM 
        {{ ref('titanic') }}
)

SELECT 
    -- Surrogate key
    ROW_NUMBER() OVER (ORDER BY cabin_number) AS cabin_key,
    -- Cabin details
    cabin_number,
    deck_letter,
    CASE 
        WHEN deck_letter = 'A' THEN 'Promenade Deck'
        WHEN deck_letter = 'B' THEN 'Bridge Deck'
        WHEN deck_letter = 'C' THEN 'Shelter Deck'
        WHEN deck_letter = 'D' THEN 'Saloon Deck'
        WHEN deck_letter = 'E' THEN 'Upper Deck'
        WHEN deck_letter = 'F' THEN 'Middle Deck'
        WHEN deck_letter = 'G' THEN 'Lower Deck'
        WHEN deck_letter = 'T' THEN 'Tank Top'
        ELSE 'Unknown Deck'
    END AS deck_name,
    has_cabin,
    -- Metadata
    CURRENT_TIMESTAMP() AS created_at
FROM 
    cabin_data
ORDER BY 
    cabin_number 