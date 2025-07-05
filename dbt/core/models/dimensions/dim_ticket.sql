{{ config(materialized='table') }}

WITH

ticket_data AS (
    SELECT DISTINCT
        Ticket,
        -- Determine ticket type based on format
        CASE 
            WHEN REGEXP_CONTAINS(Ticket, r'^[A-Z]+') THEN 'Prefix Ticket'
            WHEN REGEXP_CONTAINS(Ticket, r'^\d+$') THEN 'Numeric Ticket'
            WHEN REGEXP_CONTAINS(Ticket, r'/') THEN 'Slash Format'
            WHEN REGEXP_CONTAINS(Ticket, r'\.') THEN 'Dot Format'
            ELSE 'Other Format'
        END AS ticket_type,
        -- Check if ticket is shared (appears multiple times)
        COUNT(*) OVER (PARTITION BY Ticket) AS ticket_count
    FROM 
        {{ ref('titanic') }}
    WHERE 
        Ticket IS NOT NULL
)

SELECT 
    -- Surrogate key
    ROW_NUMBER() OVER (ORDER BY Ticket) AS ticket_key,
    -- Ticket details
    Ticket AS ticket_number,
    ticket_type,
    CASE 
        WHEN ticket_count > 1 THEN TRUE
        ELSE FALSE
    END AS is_group_ticket,
    ticket_count AS passengers_on_ticket,
    -- Metadata
    CURRENT_TIMESTAMP() AS created_at
FROM 
    ticket_data
ORDER BY 
    Ticket 