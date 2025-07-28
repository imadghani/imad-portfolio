#!/usr/bin/env python3
"""
Titanic GPT Interface - Natural Language Queries for Titanic Data
Uses OpenAI's GPT to translate natural language questions into SQL queries
"""

import streamlit as st
import pandas as pd
import openai
from google.cloud import bigquery
import os
import json
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="Titanic GPT Interface", 
    page_icon="🚢", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 5px solid #9c27b0;
    }
    .sql-code {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        font-family: 'Courier New', monospace;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

def get_bigquery_client():
    """Initialize BigQuery client with credentials"""
    try:
        credentials_path = "/Users/imadghani/GitHub/imad-portfolio/secrets/bigquery-service-account.json"
        if os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        client = bigquery.Client()
        return client
    except Exception as e:
        st.error(f"Failed to initialize BigQuery client: {str(e)}")
        return None

def get_table_schema():
    """Get the schema of the titanic_one_big_table for GPT context"""
    schema_info = {
        "table_name": "titanic_one_big_table",
        "description": "Comprehensive denormalized table with all Titanic passenger data",
        "columns": {
            # Identifiers
            "journey_key": "Unique identifier for each passenger journey",
            "passenger_key": "Internal passenger key",
            "passenger_id": "Original passenger ID from manifest",
            
            # Demographics
            "passenger_name": "Full passenger name with title",
            "gender": "Passenger gender (male/female)",
            "age": "Passenger age in years",
            "age_group": "Age category (Child, Young Adult, Adult, Senior, Unknown)",
            "title": "Title extracted from name (Mr, Mrs, Miss, Master, Dr, etc.)",
            "title_gender": "Gender derived from title",
            "social_status": "Social status (Nobility/Professional, Married/Adult, Unmarried/Young, Other)",
            
            # Family
            "siblings_spouses": "Number of siblings/spouses aboard (sibsp)",
            "parents_children": "Number of parents/children aboard (parch)",
            "family_size": "Total family size including passenger",
            "family_size_category": "Family size group (Traveling Alone, Small Family, Large Family, Very Large Family)",
            "is_alone": "Boolean indicating if traveling alone",
            
            # Ticket & Class
            "ticket_number": "Ticket identifier",
            "ticket_fare": "Fare paid for ticket",
            "fare_category": "Fare category (Low, Medium, High, Premium)",
            "detailed_fare_category": "Detailed fare analysis",
            "individual_fare_estimate": "Estimated individual fare (for shared tickets)",
            "shared_ticket": "Boolean indicating shared ticket",
            "passengers_per_ticket": "Number of passengers sharing ticket",
            "passenger_class_number": "Class number (1, 2, 3)",
            "passenger_class_name": "Class name (First Class, Second Class, Third Class)",
            "class_amenities": "Description of class amenities",
            "socioeconomic_class": "Derived socioeconomic status (Wealthy Elite, Upper Class, Middle Class, Working Class, Lower Class)",
            
            # Embarkation
            "embarkation_code": "Port code (C, Q, S)",
            "embarkation_port": "Port name (Cherbourg, Queenstown, Southampton)",
            "embarkation_country": "Country (France, Ireland, England)",
            "embarkation_region": "Region (British Isles, Continental Europe)",
            "port_details": "Historical port information",
            
            # Cabin & Location
            "cabin_number": "Cabin identifier",
            "cabin_deck": "Deck letter (A-G, T, Unknown)",
            "cabin_numeric": "Numeric part of cabin",
            "deck_level": "Numeric deck level",
            "deck_category": "Deck category (Upper Decks, Middle Decks, Lower Decks, Boat Deck, Unknown Deck)",
            "deck_description": "Deck characteristics",
            
            # Survival
            "survived": "Survival flag (0/1)",
            "survival_flag": "Boolean survival flag",
            "survival_status": "Human readable (Survived, Did Not Survive)",
            "survival_probability_category": "Predicted survival probability",
            
            # Metadata
            "journey_date": "Date of journey",
            "boarding_sequence": "Estimated boarding sequence",
            "analysis_timestamp": "When record was created",
            "data_source": "Source of the data",
            "schema_version": "Data schema version"
        },
        "sample_queries": [
            "How many passengers survived by class?",
            "What was the average age of female survivors?",
            "Show me passengers who paid the highest fares",
            "Which deck had the highest survival rate?",
            "How many children were on the Titanic?",
            "What percentage of first-class passengers survived?",
            "Show me all passengers from Southampton",
            "Which families had the most members aboard?"
        ]
    }
    return schema_info

def create_gpt_prompt(user_question, schema_info):
    """Create a detailed prompt for GPT to generate SQL"""
    prompt = f"""
You are an expert SQL analyst working with Titanic passenger data. Convert the user's natural language question into a precise BigQuery SQL query.

TABLE INFORMATION:
- Table: `core_analytics.titanic_one_big_table`
- Description: {schema_info['description']}

AVAILABLE COLUMNS:
{json.dumps(schema_info['columns'], indent=2)}

RULES:
1. Always use the full table name: `core_analytics.titanic_one_big_table`
2. Use proper BigQuery SQL syntax
3. Include appropriate aggregations, filters, and sorting
4. Limit results to 100 rows unless specifically asked for more
5. Use descriptive column aliases
6. Handle NULL values appropriately
7. For percentages, multiply by 100 and round to 2 decimal places
8. For survival analysis, use both 'survived' (0/1) and 'survival_status' (text) as appropriate
9. Always include ORDER BY for meaningful sorting
10. Use CASE statements for custom categorizations when helpful

USER QUESTION: {user_question}

Generate ONLY the SQL query without any explanation or markdown formatting:
"""
    return prompt

def execute_query(client, sql_query):
    """Execute BigQuery SQL and return results"""
    try:
        # Clean the SQL query
        sql_query = sql_query.strip()
        if sql_query.startswith('```sql'):
            sql_query = sql_query[6:]
        if sql_query.endswith('```'):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip()
        
        # Execute query
        query_job = client.query(sql_query)
        results = query_job.result()
        
        # Convert to DataFrame
        df = results.to_dataframe()
        return df, None
        
    except Exception as e:
        return None, str(e)

def get_openai_response(prompt, api_key):
    """Get SQL query from OpenAI GPT"""
    try:
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert SQL analyst. Generate only clean SQL queries without any formatting or explanation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        sql_query = response.choices[0].message.content.strip()
        return sql_query, None
        
    except Exception as e:
        return None, str(e)

def main():
    # Header
    st.markdown('<div class="main-header">🚢 Titanic GPT Interface</div>', unsafe_allow_html=True)
    st.markdown("**Ask natural language questions about the Titanic data and get SQL-powered answers!**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to use the GPT interface")
        
        st.header("📊 Quick Examples")
        example_questions = [
            "How many passengers survived by class?",
            "What was the survival rate for women vs men?",
            "Show me the top 10 highest fares paid",
            "Which embarkation port had the most passengers?",
            "How many children under 18 were aboard?",
            "What percentage of first-class passengers survived?",
            "Show me families with more than 5 members",
            "Which deck had the best survival rate?"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"📝 {question}", key=f"example_{i}"):
                st.session_state.current_question = question
        
        st.header("📋 Query History")
        if st.session_state.query_history:
            for i, (q, _) in enumerate(st.session_state.query_history[-5:]):
                if st.button(f"🔄 {q[:30]}...", key=f"history_{i}"):
                    st.session_state.current_question = q
        
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.session_state.messages = []
            st.experimental_rerun()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Question input
        question = st.text_input(
            "Enter your question about the Titanic data:",
            value=st.session_state.get('current_question', ''),
            placeholder="e.g., How many passengers survived by gender and class?",
            key="question_input"
        )
        
        if 'current_question' in st.session_state:
            del st.session_state.current_question
        
        # Submit button
        if st.button("🚀 Ask GPT", disabled=not openai_api_key or not question):
            if not openai_api_key:
                st.error("Please provide an OpenAI API key")
            elif not question:
                st.error("Please enter a question")
            else:
                # Initialize BigQuery client
                client = get_bigquery_client()
                if not client:
                    st.error("Failed to connect to BigQuery")
                    return
                
                # Get schema information
                schema_info = get_table_schema()
                
                # Create GPT prompt
                prompt = create_gpt_prompt(question, schema_info)
                
                # Show progress
                with st.spinner("🤖 GPT is generating SQL query..."):
                    sql_query, error = get_openai_response(prompt, openai_api_key)
                
                if error:
                    st.error(f"Error from OpenAI: {error}")
                    return
                
                # Display generated SQL
                st.subheader("🔍 Generated SQL Query")
                st.code(sql_query, language="sql")
                
                # Execute query
                with st.spinner("📊 Executing query..."):
                    df, query_error = execute_query(client, sql_query)
                
                if query_error:
                    st.error(f"Query execution error: {query_error}")
                    return
                
                # Display results
                st.subheader("📈 Results")
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)
                    
                    # Show summary
                    st.info(f"✅ Found {len(df)} rows")
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"titanic_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Add to history
                    st.session_state.query_history.append((question, sql_query))
                    
                else:
                    st.warning("No results found for this query")
    
    with col2:
        st.header("📖 Data Schema")
        
        schema_info = get_table_schema()
        
        st.subheader("🎯 Key Columns")
        key_columns = {
            "Demographics": ["passenger_name", "gender", "age", "age_group"],
            "Family": ["family_size", "family_size_category", "is_alone"],
            "Class & Fare": ["passenger_class_name", "ticket_fare", "fare_category"],
            "Location": ["embarkation_port", "cabin_deck", "deck_category"],
            "Survival": ["survival_status", "survived"]
        }
        
        for category, columns in key_columns.items():
            with st.expander(f"📂 {category}"):
                for col in columns:
                    if col in schema_info["columns"]:
                        st.write(f"**{col}**: {schema_info['columns'][col]}")
        
        st.subheader("💡 Tips")
        st.info("""
        **Good questions to ask:**
        - Count/percentage queries
        - Comparisons by groups
        - Top/bottom rankings
        - Survival analysis
        - Demographics breakdowns
        
        **Be specific about:**
        - What you want to count/measure
        - How you want to group data
        - Sort order preferences
        """)

if __name__ == "__main__":
    main() 