import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from google.cloud import bigquery
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Titanic Data Explorer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    .section-header {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize BigQuery client
@st.cache_resource
def init_bigquery():
    # Set up credentials path
    project_root = Path(__file__).parent.parent.parent
    credentials_path = project_root / "secrets" / "bigquery-service-account.json"
    if credentials_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
    
    return bigquery.Client(project=os.getenv("GCP_PROJECT"))

# Load data functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_survival_analysis():
    """Load survival analysis data from BigQuery"""
    client = init_bigquery()
    query = """
    SELECT 
        class_name,
        gender,
        age_group,
        port_name,
        deck_name,
        total_passengers,
        survivors,
        survival_rate_pct,
        avg_fare,
        avg_family_size,
        alone_passengers,
        alone_percentage
    FROM `{project}.prototype_data.survival_analysis`
    WHERE class_name IS NOT NULL
    """.format(project=os.getenv("GCP_PROJECT"))
    
    # Use standard BigQuery API to avoid storage permissions issue
    query_job = client.query(query)
    results = query_job.result()
    
    # Convert to pandas DataFrame manually
    rows = [dict(row) for row in results]
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def load_passenger_facts():
    """Load passenger facts from BigQuery"""
    client = init_bigquery()
    query = """
    SELECT 
        f.fare_amount,
        f.age,
        f.family_size,
        f.survived_flag,
        f.is_alone_flag,
        dp.gender,
        dp.age_group,
        dp.title,
        dpc.class_name,
        de.port_name,
        de.country,
        dc.deck_name,
        dt.ticket_type,
        dt.is_group_ticket
    FROM `{project}.prototype_data.fact_passenger_journey` f
    LEFT JOIN `{project}.prototype_data.dim_passenger` dp ON f.passenger_key = dp.passenger_key
    LEFT JOIN `{project}.prototype_data.dim_passenger_class` dpc ON f.passenger_class_key = dpc.passenger_class_key
    LEFT JOIN `{project}.prototype_data.dim_embarkation` de ON f.embarkation_key = de.embarkation_key
    LEFT JOIN `{project}.prototype_data.dim_cabin` dc ON f.cabin_key = dc.cabin_key
    LEFT JOIN `{project}.prototype_data.dim_ticket` dt ON f.ticket_key = dt.ticket_key
    WHERE dpc.class_name IS NOT NULL
    """.format(project=os.getenv("GCP_PROJECT"))
    
    # Use standard BigQuery API to avoid storage permissions issue
    query_job = client.query(query)
    results = query_job.result()
    
    # Convert to pandas DataFrame manually
    rows = [dict(row) for row in results]
    return pd.DataFrame(rows)

# Main dashboard
def main():
    # Header
    st.markdown("<h1 class='main-header'>🚢 Titanic Data Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Interactive analysis of passenger survival on the RMS Titanic</p>", unsafe_allow_html=True)
    
    # Load data
    try:
        with st.spinner("Loading data from BigQuery..."):
            survival_data = load_survival_analysis()
            passenger_facts = load_passenger_facts()
        
        # Sidebar filters
        st.sidebar.header("🔍 Filters")
        
        # Class filter
        classes = ['All'] + sorted(passenger_facts['class_name'].unique().tolist())
        selected_class = st.sidebar.selectbox("Passenger Class", classes)
        
        # Gender filter
        genders = ['All'] + sorted(passenger_facts['gender'].unique().tolist())
        selected_gender = st.sidebar.selectbox("Gender", genders)
        
        # Age group filter
        age_groups = ['All'] + sorted(passenger_facts['age_group'].unique().tolist())
        selected_age_group = st.sidebar.selectbox("Age Group", age_groups)
        
        # Port filter
        ports = ['All'] + sorted(passenger_facts['port_name'].unique().tolist())
        selected_port = st.sidebar.selectbox("Embarkation Port", ports)
        
        # Filter data based on selections
        filtered_data = passenger_facts.copy()
        if selected_class != 'All':
            filtered_data = filtered_data[filtered_data['class_name'] == selected_class]
        if selected_gender != 'All':
            filtered_data = filtered_data[filtered_data['gender'] == selected_gender]
        if selected_age_group != 'All':
            filtered_data = filtered_data[filtered_data['age_group'] == selected_age_group]
        if selected_port != 'All':
            filtered_data = filtered_data[filtered_data['port_name'] == selected_port]
        
        # Key metrics
        st.markdown("<h2 class='section-header'>📊 Key Metrics</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_passengers = len(filtered_data)
            st.metric("Total Passengers", total_passengers)
        
        with col2:
            survivors = filtered_data['survived_flag'].sum()
            st.metric("Survivors", survivors)
        
        with col3:
            survival_rate = (survivors / total_passengers * 100) if total_passengers > 0 else 0
            st.metric("Survival Rate", f"{survival_rate:.1f}%")
        
        with col4:
            avg_fare = filtered_data['fare_amount'].mean()
            st.metric("Average Fare", f"£{avg_fare:.2f}")
        
        # Survival Analysis
        st.markdown("<h2 class='section-header'>🎯 Survival Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Survival by class
            class_survival = filtered_data.groupby('class_name').agg({
                'survived_flag': ['count', 'sum', 'mean']
            }).round(3)
            class_survival.columns = ['Total', 'Survivors', 'Survival_Rate']
            class_survival['Survival_Rate'] = class_survival['Survival_Rate'] * 100
            
            fig = px.bar(
                class_survival.reset_index(),
                x='class_name',
                y='Survival_Rate',
                title='Survival Rate by Passenger Class',
                labels={'class_name': 'Passenger Class', 'Survival_Rate': 'Survival Rate (%)'},
                color='Survival_Rate',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Survival by gender
            gender_survival = filtered_data.groupby('gender').agg({
                'survived_flag': ['count', 'sum', 'mean']
            }).round(3)
            gender_survival.columns = ['Total', 'Survivors', 'Survival_Rate']
            gender_survival['Survival_Rate'] = gender_survival['Survival_Rate'] * 100
            
            fig = px.pie(
                gender_survival.reset_index(),
                values='Survivors',
                names='gender',
                title='Survivors by Gender',
                color_discrete_sequence=['#ff6b6b', '#4ecdc4']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Demographics Analysis
        st.markdown("<h2 class='section-header'>👥 Demographics Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            age_data = filtered_data.dropna(subset=['age'])
            fig = px.histogram(
                age_data,
                x='age',
                color='survived_flag',
                title='Age Distribution by Survival',
                labels={'age': 'Age', 'survived_flag': 'Survived'},
                nbins=20,
                color_discrete_map={0: '#ff6b6b', 1: '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Family size analysis
            family_survival = filtered_data.groupby('family_size').agg({
                'survived_flag': ['count', 'mean']
            }).round(3)
            family_survival.columns = ['Total', 'Survival_Rate']
            family_survival['Survival_Rate'] = family_survival['Survival_Rate'] * 100
            
            fig = px.line(
                family_survival.reset_index(),
                x='family_size',
                y='Survival_Rate',
                title='Survival Rate by Family Size',
                labels={'family_size': 'Family Size', 'Survival_Rate': 'Survival Rate (%)'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Fare Analysis
        st.markdown("<h2 class='section-header'>💰 Fare Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fare by class
            fig = px.box(
                filtered_data,
                x='class_name',
                y='fare_amount',
                title='Fare Distribution by Class',
                labels={'class_name': 'Passenger Class', 'fare_amount': 'Fare Amount (£)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fare vs Survival
            fare_survival = filtered_data.groupby('survived_flag')['fare_amount'].mean()
            fig = px.bar(
                x=['Did not survive', 'Survived'],
                y=fare_survival.values,
                title='Average Fare by Survival Status',
                labels={'x': 'Survival Status', 'y': 'Average Fare (£)'},
                color=fare_survival.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Geographic Analysis
        st.markdown("<h2 class='section-header'>🌍 Geographic Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Embarkation port analysis
            port_survival = filtered_data.groupby('port_name').agg({
                'survived_flag': ['count', 'sum', 'mean']
            }).round(3)
            port_survival.columns = ['Total', 'Survivors', 'Survival_Rate']
            port_survival['Survival_Rate'] = port_survival['Survival_Rate'] * 100
            
            fig = px.bar(
                port_survival.reset_index(),
                x='port_name',
                y='Survival_Rate',
                title='Survival Rate by Embarkation Port',
                labels={'port_name': 'Embarkation Port', 'Survival_Rate': 'Survival Rate (%)'},
                color='Survival_Rate',
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Deck analysis
            deck_data = filtered_data[filtered_data['deck_name'] != 'Unknown Deck']
            if not deck_data.empty:
                deck_survival = deck_data.groupby('deck_name').agg({
                    'survived_flag': ['count', 'sum', 'mean']
                }).round(3)
                deck_survival.columns = ['Total', 'Survivors', 'Survival_Rate']
                deck_survival['Survival_Rate'] = deck_survival['Survival_Rate'] * 100
                
                fig = px.bar(
                    deck_survival.reset_index(),
                    x='deck_name',
                    y='Survival_Rate',
                    title='Survival Rate by Deck',
                    labels={'deck_name': 'Deck', 'Survival_Rate': 'Survival Rate (%)'},
                    color='Survival_Rate',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Data Table
        st.markdown("<h2 class='section-header'>📋 Filtered Data</h2>", unsafe_allow_html=True)
        
        # Show summary statistics
        st.write("### Summary Statistics")
        summary_stats = filtered_data.describe()
        st.dataframe(summary_stats)
        
        # Show raw data with search
        st.write("### Raw Data")
        st.dataframe(
            filtered_data.head(100),
            use_container_width=True
        )
        
        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name='titanic_filtered_data.csv',
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure BigQuery is properly configured and the data tables exist.")

if __name__ == "__main__":
    main() 