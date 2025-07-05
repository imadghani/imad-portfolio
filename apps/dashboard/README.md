# 🚢 Titanic Data Explorer Dashboard

An interactive Streamlit dashboard for exploring and analyzing the Titanic dataset using data stored in BigQuery.

## Features

- **Interactive Filters**: Filter data by passenger class, gender, age group, and embarkation port
- **Key Metrics**: Real-time calculation of survival rates, passenger counts, and fare statistics
- **Survival Analysis**: Visualizations showing survival patterns by different demographics
- **Demographics**: Age distribution and family size analysis
- **Fare Analysis**: Fare distribution and relationship to survival
- **Geographic Analysis**: Analysis by embarkation port and ship deck
- **Data Export**: Download filtered data as CSV

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   - Ensure your `.env` file contains the `GCP_PROJECT` variable
   - Make sure BigQuery authentication is configured

3. **Verify Data**:
   - Run the data check script to ensure tables exist:
   ```bash
   python scripts/check_datasets.py
   ```

## Running the Dashboard

1. **Start the Dashboard**:
   ```bash
   streamlit run apps/dashboard/titanic_dashboard.py
   ```

2. **Access the Dashboard**:
   - Open your browser to `http://localhost:8501`
   - Use the sidebar filters to explore different data segments

## Data Sources

The dashboard connects to the following BigQuery tables:

- `fact_passenger_journey` - Main fact table with passenger journey data
- `dim_passenger` - Passenger dimension (demographics, names, titles)
- `dim_passenger_class` - Passenger class dimension
- `dim_embarkation` - Embarkation port dimension
- `dim_cabin` - Cabin and deck dimension
- `dim_ticket` - Ticket dimension
- `survival_analysis` - Pre-calculated survival analysis view

## Dashboard Sections

### 📊 Key Metrics
- Total passengers (filtered)
- Total survivors
- Overall survival rate
- Average fare

### 🎯 Survival Analysis
- Survival rate by passenger class
- Survivor distribution by gender

### 👥 Demographics Analysis
- Age distribution with survival overlay
- Family size vs survival rate

### 💰 Fare Analysis
- Fare distribution by passenger class
- Average fare by survival status

### 🌍 Geographic Analysis
- Survival rates by embarkation port
- Survival rates by ship deck

### 📋 Data Tables
- Summary statistics
- Raw filtered data (first 100 rows)
- CSV download option

## Customization

To modify the dashboard:

1. **Add New Visualizations**: Add new chart functions in the main() function
2. **Modify Filters**: Update the sidebar filter options
3. **Change Styling**: Modify the CSS in the st.markdown() section
4. **Add New Metrics**: Extend the key metrics section with additional calculations

## Performance

- Data is cached for 1 hour to improve performance
- BigQuery client is cached as a resource
- Filtered data is processed in-memory for fast interactivity

## Troubleshooting

1. **BigQuery Connection Issues**: Ensure your GCP credentials are properly configured
2. **Missing Data**: Verify that the dbt models have been run and tables exist
3. **Performance Issues**: Consider reducing the data cache TTL or limiting data size
4. **Port Conflicts**: Change the Streamlit port with `--server.port` option 