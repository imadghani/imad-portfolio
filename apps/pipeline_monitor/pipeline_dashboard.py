import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import os
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

# Configure page
st.set_page_config(
    page_title="Data Pipeline Monitor",
    page_icon="🔄",
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
    .success {
        color: #27ae60;
        font-weight: bold;
    }
    .failed {
        color: #e74c3c;
        font-weight: bold;
    }
    .running {
        color: #f39c12;
        font-weight: bold;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 5px 10px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .status-failed {
        background-color: #f8d7da;
        color: #721c24;
        padding: 5px 10px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .status-running {
        background-color: #fff3cd;
        color: #856404;
        padding: 5px 10px;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_airflow_db_path():
    """Get the path to Airflow SQLite database"""
    project_root = Path(__file__).parent.parent.parent
    airflow_db_path = project_root / "airflow" / "airflow.db"
    return str(airflow_db_path)

@st.cache_data(ttl=60)
def load_dag_runs():
    """Load DAG runs from Airflow database"""
    try:
        db_path = get_airflow_db_path()
        if not os.path.exists(db_path):
            return pd.DataFrame()
        
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT 
            dag_id,
            run_id,
            state,
            execution_date,
            start_date,
            end_date,
            run_type,
            data_interval_start,
            data_interval_end
        FROM dag_run
        ORDER BY execution_date DESC
        LIMIT 1000
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert datetime columns
        datetime_cols = ['execution_date', 'start_date', 'end_date', 'data_interval_start', 'data_interval_end']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error loading DAG runs: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_task_instances():
    """Load task instances from Airflow database"""
    try:
        db_path = get_airflow_db_path()
        if not os.path.exists(db_path):
            return pd.DataFrame()
        
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT 
            ti.dag_id,
            ti.task_id,
            ti.run_id,
            ti.state,
            ti.start_date,
            ti.end_date,
            ti.duration,
            ti.try_number,
            ti.max_tries,
            dr.execution_date
        FROM task_instance ti
        LEFT JOIN dag_run dr ON ti.dag_id = dr.dag_id AND ti.run_id = dr.run_id
        WHERE dr.execution_date >= datetime('now', '-30 days')
        ORDER BY dr.execution_date DESC
        LIMIT 5000
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert datetime columns
        datetime_cols = ['start_date', 'end_date', 'execution_date']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error loading task instances: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_logs():
    """Load recent logs from Airflow log directory"""
    try:
        project_root = Path(__file__).parent.parent.parent
        logs_dir = project_root / "airflow" / "logs"
        
        if not logs_dir.exists():
            return []
        
        log_files = []
        for log_file in logs_dir.rglob("*.log"):
            if log_file.stat().st_mtime > (time.time() - 86400):  # Last 24 hours
                log_files.append({
                    'file': str(log_file.relative_to(logs_dir)),
                    'size': log_file.stat().st_size,
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime)
                })
        
        return sorted(log_files, key=lambda x: x['modified'], reverse=True)[:100]
    except Exception as e:
        st.error(f"Error loading logs: {str(e)}")
        return []

def format_duration(duration_seconds):
    """Format duration in human readable format"""
    if pd.isna(duration_seconds) or duration_seconds is None:
        return "N/A"
    
    duration = timedelta(seconds=float(duration_seconds))
    
    if duration.days > 0:
        return f"{duration.days}d {duration.seconds//3600}h {(duration.seconds//60)%60}m"
    elif duration.seconds >= 3600:
        return f"{duration.seconds//3600}h {(duration.seconds//60)%60}m {duration.seconds%60}s"
    elif duration.seconds >= 60:
        return f"{(duration.seconds//60)%60}m {duration.seconds%60}s"
    else:
        return f"{duration.seconds}s"

def get_status_color(state):
    """Get color for different states"""
    colors = {
        'success': '#27ae60',
        'failed': '#e74c3c',
        'running': '#f39c12',
        'up_for_retry': '#f39c12',
        'up_for_reschedule': '#f39c12',
        'queued': '#3498db',
        'skipped': '#95a5a6',
        'upstream_failed': '#e74c3c',
        'scheduled': '#3498db'
    }
    return colors.get(state, '#95a5a6')

# Main dashboard
def main():
    # Header
    st.markdown("<h1 class='main-header'>🔄 Data Pipeline Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Real-time monitoring and management of Airflow data pipelines</p>", unsafe_allow_html=True)
    
    # Load data
    dag_runs_df = load_dag_runs()
    task_instances_df = load_task_instances()
    logs = load_logs()
    
    if dag_runs_df.empty and task_instances_df.empty:
        st.warning("No Airflow data found. Make sure Airflow is running and has executed some DAGs.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    if not dag_runs_df.empty:
        min_date = dag_runs_df['execution_date'].min().date()
        max_date = dag_runs_df['execution_date'].max().date()
        
        # Ensure default start date doesn't go below min_date
        default_start = max(min_date, max_date - timedelta(days=7))
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            dag_runs_df = dag_runs_df[
                (dag_runs_df['execution_date'].dt.date >= start_date) &
                (dag_runs_df['execution_date'].dt.date <= end_date)
            ]
            
            # Filter task instances only if dataframe is not empty and has execution_date column
            if not task_instances_df.empty and 'execution_date' in task_instances_df.columns:
                task_instances_df = task_instances_df[
                    (task_instances_df['execution_date'].dt.date >= start_date) &
                    (task_instances_df['execution_date'].dt.date <= end_date)
                ]
    
    # DAG filter
    if not dag_runs_df.empty:
        available_dags = ['All'] + sorted(dag_runs_df['dag_id'].unique().tolist())
        selected_dag = st.sidebar.selectbox("Select DAG", available_dags)
        
        if selected_dag != 'All':
            dag_runs_df = dag_runs_df[dag_runs_df['dag_id'] == selected_dag]
            if not task_instances_df.empty and 'dag_id' in task_instances_df.columns:
                task_instances_df = task_instances_df[task_instances_df['dag_id'] == selected_dag]
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Overview metrics
    st.markdown("<h2 class='section-header'>📊 Pipeline Overview</h2>", unsafe_allow_html=True)
    
    if not dag_runs_df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_runs = len(dag_runs_df)
            st.metric("Total DAG Runs", total_runs)
        
        with col2:
            success_runs = len(dag_runs_df[dag_runs_df['state'] == 'success'])
            success_rate = (success_runs / total_runs * 100) if total_runs > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col3:
            failed_runs = len(dag_runs_df[dag_runs_df['state'] == 'failed'])
            st.metric("Failed Runs", failed_runs, delta=f"{failed_runs} failures")
        
        with col4:
            running_runs = len(dag_runs_df[dag_runs_df['state'] == 'running'])
            st.metric("Currently Running", running_runs)
        
        with col5:
            unique_dags = dag_runs_df['dag_id'].nunique()
            st.metric("Active DAGs", unique_dags)
    
    # DAG Status Overview
    st.markdown("<h2 class='section-header'>🎯 DAG Status Overview</h2>", unsafe_allow_html=True)
    
    if not dag_runs_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution
            status_counts = dag_runs_df['state'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='DAG Run Status Distribution',
                color_discrete_map={
                    'success': '#27ae60',
                    'failed': '#e74c3c',
                    'running': '#f39c12',
                    'queued': '#3498db'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success rate by DAG
            dag_success_rate = dag_runs_df.groupby('dag_id').agg({
                'state': lambda x: (x == 'success').mean() * 100
            }).round(2)
            dag_success_rate.columns = ['Success Rate']
            dag_success_rate = dag_success_rate.sort_values('Success Rate', ascending=False)
            
            fig = px.bar(
                dag_success_rate.reset_index(),
                x='Success Rate',
                y='dag_id',
                title='Success Rate by DAG',
                labels={'dag_id': 'DAG ID', 'Success Rate': 'Success Rate (%)'},
                color='Success Rate',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Timeline Analysis
    st.markdown("<h2 class='section-header'>📈 Pipeline Timeline</h2>", unsafe_allow_html=True)
    
    if not dag_runs_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # DAG runs over time
            dag_runs_daily = dag_runs_df.groupby([
                dag_runs_df['execution_date'].dt.date,
                'state'
            ]).size().reset_index(name='count')
            dag_runs_daily.columns = ['date', 'state', 'count']
            
            fig = px.bar(
                dag_runs_daily,
                x='date',
                y='count',
                color='state',
                title='DAG Runs Over Time',
                color_discrete_map={
                    'success': '#27ae60',
                    'failed': '#e74c3c',
                    'running': '#f39c12'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average duration by DAG
            if not task_instances_df.empty and 'duration' in task_instances_df.columns:
                avg_duration = task_instances_df.groupby('dag_id')['duration'].mean().sort_values(ascending=False)
                
                fig = px.bar(
                    x=avg_duration.values,
                    y=avg_duration.index,
                    title='Average Task Duration by DAG',
                    labels={'x': 'Average Duration (seconds)', 'y': 'DAG ID'},
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Task Performance Analysis
    st.markdown("<h2 class='section-header'>⚡ Task Performance</h2>", unsafe_allow_html=True)
    
    if not task_instances_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Task success rate
            task_success_rate = task_instances_df.groupby('task_id').agg({
                'state': lambda x: (x == 'success').mean() * 100
            }).round(2)
            task_success_rate.columns = ['Success Rate']
            task_success_rate = task_success_rate.sort_values('Success Rate', ascending=True).tail(15)
            
            fig = px.bar(
                task_success_rate.reset_index(),
                x='Success Rate',
                y='task_id',
                title='Task Success Rate (Top 15)',
                labels={'task_id': 'Task ID', 'Success Rate': 'Success Rate (%)'},
                color='Success Rate',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Task retry analysis
            retry_analysis = task_instances_df.groupby('task_id').agg({
                'try_number': 'mean',
                'state': 'count'
            }).round(2)
            retry_analysis.columns = ['Avg Retries', 'Total Runs']
            retry_analysis = retry_analysis[retry_analysis['Total Runs'] >= 5].sort_values('Avg Retries', ascending=False).head(15)
            
            fig = px.scatter(
                retry_analysis.reset_index(),
                x='Total Runs',
                y='Avg Retries',
                hover_data=['task_id'],
                title='Task Retry Analysis',
                labels={'Total Runs': 'Total Task Runs', 'Avg Retries': 'Average Retry Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.markdown("<h2 class='section-header'>🕐 Recent Activity</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Recent DAG Runs")
        if not dag_runs_df.empty:
            recent_runs = dag_runs_df.head(10)[['dag_id', 'run_id', 'state', 'execution_date', 'start_date']].copy()
            recent_runs['execution_date'] = recent_runs['execution_date'].dt.strftime('%Y-%m-%d %H:%M')
            recent_runs['start_date'] = recent_runs['start_date'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Apply styling
            def style_state(val):
                if val == 'success':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'failed':
                    return 'background-color: #f8d7da; color: #721c24'
                elif val == 'running':
                    return 'background-color: #fff3cd; color: #856404'
                return ''
            
            styled_df = recent_runs.style.applymap(style_state, subset=['state'])
            st.dataframe(styled_df, use_container_width=True)
    
    with col2:
        st.write("### Recent Logs")
        if logs:
            log_df = pd.DataFrame(logs[:10])
            log_df['size'] = log_df['size'].apply(lambda x: f"{x/1024:.1f} KB" if x < 1024*1024 else f"{x/(1024*1024):.1f} MB")
            log_df['modified'] = log_df['modified'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info("No recent log files found")
    
    # DAG Details
    if selected_dag != 'All' and not dag_runs_df.empty:
        st.markdown(f"<h2 class='section-header'>📋 {selected_dag} Details</h2>", unsafe_allow_html=True)
        
        dag_specific_runs = dag_runs_df[dag_runs_df['dag_id'] == selected_dag]
        dag_specific_tasks = task_instances_df[task_instances_df['dag_id'] == selected_dag] if not task_instances_df.empty and 'dag_id' in task_instances_df.columns else pd.DataFrame()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Run History")
            run_history = dag_specific_runs[['run_id', 'state', 'execution_date', 'start_date', 'end_date']].copy()
            run_history['duration'] = (run_history['end_date'] - run_history['start_date']).dt.total_seconds()
            run_history['duration'] = run_history['duration'].apply(format_duration)
            run_history['execution_date'] = run_history['execution_date'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(run_history.head(20), use_container_width=True)
        
        with col2:
            st.write("### Task Performance")
            if not dag_specific_tasks.empty:
                task_perf = dag_specific_tasks.groupby('task_id').agg({
                    'duration': 'mean',
                    'state': lambda x: (x == 'success').mean() * 100,
                    'try_number': 'mean'
                }).round(2)
                task_perf.columns = ['Avg Duration (s)', 'Success Rate (%)', 'Avg Retries']
                task_perf['Avg Duration (s)'] = task_perf['Avg Duration (s)'].apply(format_duration)
                st.dataframe(task_perf, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Data Pipeline Monitor | Last updated: {}</p>".format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ), unsafe_allow_html=True)

if __name__ == "__main__":
    main() 