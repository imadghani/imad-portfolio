#!/usr/bin/env python3
"""
Data Catalogue - Comprehensive Data Discovery and Documentation
Provides searchable interface for all tables, columns, tests, and lineage
"""

import streamlit as st
import pandas as pd
import yaml
import os
import json
from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from google.cloud import bigquery
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Data Catalogue", 
    page_icon="📚", 
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
    .table-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
    .dimension-card {
        border-left-color: #28a745;
    }
    .fact-card {
        border-left-color: #dc3545;
    }
    .analytics-card {
        border-left-color: #ffc107;
    }
    .column-info {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #6c757d;
    }
    .test-info {
        background-color: #d1ecf1;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        border-left: 3px solid #0c5460;
        font-size: 0.9rem;
    }
    .stats-metric {
        text-align: center;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .lineage-node {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem;
        border: 2px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dbt_metadata():
    """Load all dbt metadata from schema.yml files"""
    
    # Base paths
    project_root = Path("/Users/imadghani/GitHub/imad-portfolio")
    dbt_models_path = project_root / "dbt" / "core" / "models"
    
    metadata = {
        "models": {},
        "columns": {},
        "tests": {},
        "lineage": {},
        "schemas": {}
    }
    
    # Load schema files
    schema_files = list(dbt_models_path.rglob("schema.yml"))
    
    for schema_file in schema_files:
        try:
            with open(schema_file, 'r') as f:
                schema_data = yaml.safe_load(f)
            
            if 'models' in schema_data:
                for model in schema_data['models']:
                    model_name = model['name']
                    
                    # Determine model type from path
                    relative_path = schema_file.relative_to(dbt_models_path)
                    model_type = str(relative_path.parent)
                    
                    metadata["models"][model_name] = {
                        "name": model_name,
                        "description": model.get('description', ''),
                        "type": model_type,
                        "meta": model.get('meta', {}),
                        "tests": model.get('tests', []),
                        "columns": {},
                        "file_path": str(relative_path)
                    }
                    
                    # Load column information
                    if 'columns' in model:
                        for column in model['columns']:
                            col_name = column['name']
                            metadata["columns"][f"{model_name}.{col_name}"] = {
                                "model": model_name,
                                "name": col_name,
                                "description": column.get('description', ''),
                                "tests": column.get('tests', [])
                            }
                            
                            metadata["models"][model_name]["columns"][col_name] = {
                                "description": column.get('description', ''),
                                "tests": column.get('tests', [])
                            }
                            
        except Exception as e:
            st.warning(f"Could not load {schema_file}: {e}")
    
    # Create lineage information
    lineage_map = {
        "dim_passenger": ["titanic_one_big_table", "fact_passenger_journey", "survival_analysis"],
        "dim_ticket": ["titanic_one_big_table", "fact_passenger_journey", "survival_analysis"],
        "dim_passenger_class": ["titanic_one_big_table", "fact_passenger_journey", "survival_analysis"],
        "dim_embarkation": ["titanic_one_big_table", "fact_passenger_journey", "survival_analysis"],
        "dim_cabin": ["titanic_one_big_table", "fact_passenger_journey", "survival_analysis"],
        "fact_passenger_journey": ["titanic_one_big_table", "survival_analysis"],
        "survival_analysis": [],
        "titanic_one_big_table": []
    }
    
    metadata["lineage"] = lineage_map
    
    return metadata

@st.cache_data
def get_bigquery_stats():
    """Get table statistics from BigQuery"""
    try:
        credentials_path = "/Users/imadghani/GitHub/imad-portfolio/secrets/bigquery-service-account.json"
        if os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        client = bigquery.Client()
        
        # Get dataset information
        datasets = list(client.list_datasets())
        
        stats = {
            "total_datasets": len(datasets),
            "total_tables": 0,
            "table_details": {}
        }
        
        for dataset in datasets:
            dataset_id = dataset.dataset_id
            tables = list(client.list_tables(dataset_id))
            stats["total_tables"] += len(tables)
            
            for table in tables:
                table_ref = client.get_table(table.reference)
                stats["table_details"][f"{dataset_id}.{table.table_id}"] = {
                    "rows": table_ref.num_rows,
                    "size_mb": round(table_ref.num_bytes / (1024 * 1024), 2),
                    "created": table_ref.created.strftime("%Y-%m-%d %H:%M") if table_ref.created else "Unknown",
                    "modified": table_ref.modified.strftime("%Y-%m-%d %H:%M") if table_ref.modified else "Unknown"
                }
        
        return stats
        
    except Exception as e:
        return {"error": str(e), "total_datasets": 0, "total_tables": 0, "table_details": {}}

def create_lineage_graph(metadata, selected_model=None):
    """Create an interactive lineage graph"""
    
    G = nx.DiGraph()
    
    # Add nodes
    for model in metadata["models"].keys():
        model_type = metadata["models"][model]["type"]
        G.add_node(model, type=model_type)
    
    # Add edges based on lineage
    for source, targets in metadata["lineage"].items():
        for target in targets:
            if source in G and target in G:
                G.add_edge(source, target)
    
    # Create plotly network graph
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=2, color='#888'),
                           hoverinfo='none',
                           mode='lines')
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    color_map = {
        "dimensions": "#28a745",
        "facts": "#dc3545", 
        "analytics": "#ffc107"
    }
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Determine node properties
        model_type = metadata["models"][node]["type"]
        node_text.append(f"{node}<br>Type: {model_type}")
        node_colors.append(color_map.get(model_type, "#6c757d"))
        
        # Highlight selected model
        if selected_model and node == selected_model:
            node_sizes.append(25)
        else:
            node_sizes.append(15)
    
    node_trace = go.Scatter(x=node_x, y=node_y,
                           mode='markers+text',
                           hoverinfo='text',
                           text=[node for node in G.nodes()],
                           textposition="middle center",
                           hovertext=node_text,
                           marker=dict(size=node_sizes,
                                     color=node_colors,
                                     line=dict(width=2, color='white')))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(title='Data Model Lineage',
                                  titlefont_size=16,
                                  showlegend=False,
                                  hovermode='closest',
                                  margin=dict(b=20,l=5,r=5,t=40),
                                  annotations=[ dict(text="Data flows from left to right",
                                                   showarrow=False,
                                                   xref="paper", yref="paper",
                                                   x=0.005, y=-0.002,
                                                   xanchor="left", yanchor="bottom",
                                                   font=dict(color="#888", size=12))],
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    return fig

def search_metadata(metadata, search_term):
    """Search across all metadata"""
    results = {
        "models": [],
        "columns": []
    }
    
    search_term = search_term.lower()
    
    # Search models
    for model_name, model_data in metadata["models"].items():
        if (search_term in model_name.lower() or 
            search_term in model_data["description"].lower()):
            results["models"].append({
                "name": model_name,
                "type": model_data["type"],
                "description": model_data["description"]
            })
    
    # Search columns
    for col_key, col_data in metadata["columns"].items():
        if (search_term in col_data["name"].lower() or
            search_term in col_data["description"].lower()):
            results["columns"].append({
                "model": col_data["model"],
                "name": col_data["name"],
                "description": col_data["description"]
            })
    
    return results

def main():
    # Header
    st.markdown('<div class="main-header">📚 Data Catalogue</div>', unsafe_allow_html=True)
    st.markdown("**Comprehensive data discovery and documentation for the Titanic data warehouse**")
    
    # Load metadata
    with st.spinner("Loading metadata..."):
        metadata = load_dbt_metadata()
        bq_stats = get_bigquery_stats()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        
        page = st.selectbox("Choose a view:", [
            "📊 Overview",
            "🏗️ Data Models", 
            "🔍 Search & Explore",
            "📈 Lineage & Dependencies",
            "💾 BigQuery Statistics",
            "📋 Data Quality Tests"
        ])
        
        st.header("🔍 Quick Search")
        search_term = st.text_input("Search models and columns:", placeholder="e.g., passenger, survival, age")
        
        if search_term:
            search_results = search_metadata(metadata, search_term)
            
            if search_results["models"]:
                st.subheader("Models")
                for model in search_results["models"][:5]:
                    st.write(f"**{model['name']}** ({model['type']})")
                    
            if search_results["columns"]:
                st.subheader("Columns")
                for col in search_results["columns"][:5]:
                    st.write(f"**{col['name']}** in {col['model']}")
    
    # Main content based on selected page
    if page == "📊 Overview":
        st.header("📊 Data Warehouse Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="stats-metric">
                <h3>📋 Total Models</h3>
                <h2>{}</h2>
            </div>
            """.format(len(metadata["models"])), unsafe_allow_html=True)
        
        with col2:
            total_columns = len(metadata["columns"])
            st.markdown("""
            <div class="stats-metric">
                <h3>📊 Total Columns</h3>
                <h2>{}</h2>
            </div>
            """.format(total_columns), unsafe_allow_html=True)
        
        with col3:
            total_tests = sum(len(model["tests"]) for model in metadata["models"].values())
            total_tests += sum(len(col["tests"]) for col in metadata["columns"].values())
            st.markdown("""
            <div class="stats-metric">
                <h3>🧪 Total Tests</h3>
                <h2>{}</h2>
            </div>
            """.format(total_tests), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="stats-metric">
                <h3>💾 BQ Tables</h3>
                <h2>{}</h2>
            </div>
            """.format(bq_stats.get("total_tables", "N/A")), unsafe_allow_html=True)
        
        # Model distribution
        st.subheader("📈 Model Distribution")
        
        model_types = {}
        for model in metadata["models"].values():
            model_type = model["type"]
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        if model_types:
            fig = px.pie(values=list(model_types.values()), 
                        names=list(model_types.keys()),
                        title="Models by Type")
            st.plotly_chart(fig)
        
        # Recent activity (mock data)
        st.subheader("⚡ Recent Activity")
        activity_data = [
            {"timestamp": "2025-01-05 09:15", "action": "Schema updated", "object": "titanic_one_big_table"},
            {"timestamp": "2025-01-05 08:30", "action": "Tests passed", "object": "dim_passenger"},
            {"timestamp": "2025-01-04 16:45", "action": "Model refreshed", "object": "survival_analysis"},
            {"timestamp": "2025-01-04 14:20", "action": "New column added", "object": "fact_passenger_journey"}
        ]
        
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df, use_container_width=True)
    
    elif page == "🏗️ Data Models":
        st.header("🏗️ Data Models")
        
        # Filter by model type
        model_type_filter = st.selectbox("Filter by type:", 
                                       ["All"] + list(set(model["type"] for model in metadata["models"].values())))
        
        # Display models
        for model_name, model_data in metadata["models"].items():
            if model_type_filter == "All" or model_data["type"] == model_type_filter:
                
                # Determine card class
                card_class = "table-card"
                if "dimension" in model_data["type"]:
                    card_class += " dimension-card"
                elif "fact" in model_data["type"]:
                    card_class += " fact-card"
                elif "analytics" in model_data["type"]:
                    card_class += " analytics-card"
                
                with st.expander(f"📊 {model_name} ({model_data['type']})"):
                    st.markdown(f"**Description:** {model_data['description']}")
                    
                    # Meta information
                    if model_data["meta"]:
                        st.subheader("📝 Metadata")
                        for key, value in model_data["meta"].items():
                            st.write(f"**{key}:** {value}")
                    
                    # Model tests
                    if model_data["tests"]:
                        st.subheader("🧪 Model Tests")
                        for test in model_data["tests"]:
                            st.markdown(f'<div class="test-info">🔬 {test}</div>', unsafe_allow_html=True)
                    
                    # Columns
                    if model_data["columns"]:
                        st.subheader("📋 Columns")
                        col1, col2 = st.columns([1, 1])
                        
                        for i, (col_name, col_data) in enumerate(model_data["columns"].items()):
                            with col1 if i % 2 == 0 else col2:
                                st.markdown(f"""
                                <div class="column-info">
                                    <strong>{col_name}</strong><br>
                                    {col_data['description']}<br>
                                    <small>Tests: {len(col_data['tests'])}</small>
                                </div>
                                """, unsafe_allow_html=True)
    
    elif page == "🔍 Search & Explore":
        st.header("🔍 Search & Explore")
        
        search_query = st.text_input("Search across all models and columns:", 
                                   placeholder="Enter keywords...")
        
        if search_query:
            results = search_metadata(metadata, search_query)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📊 Models ({len(results['models'])} found)")
                for model in results["models"]:
                    st.markdown(f"""
                    <div class="table-card">
                        <h4>{model['name']}</h4>
                        <p><strong>Type:</strong> {model['type']}</p>
                        <p>{model['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader(f"📋 Columns ({len(results['columns'])} found)")
                for col in results["columns"]:
                    st.markdown(f"""
                    <div class="column-info">
                        <strong>{col['name']}</strong> in {col['model']}<br>
                        {col['description']}
                    </div>
                    """, unsafe_allow_html=True)
    
    elif page == "📈 Lineage & Dependencies":
        st.header("📈 Data Lineage & Dependencies")
        
        # Model selector for highlighting
        selected_model = st.selectbox("Highlight model:", 
                                    ["None"] + list(metadata["models"].keys()))
        
        if selected_model == "None":
            selected_model = None
        
        # Create lineage graph
        fig = create_lineage_graph(metadata, selected_model)
        st.plotly_chart(fig, use_container_width=True)
        
        # Dependency information
        st.subheader("📋 Dependency Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⬆️ Upstream Dependencies")
            for model, dependencies in metadata["lineage"].items():
                if dependencies:
                    st.write(f"**{model}** depends on:")
                    for dep in dependencies:
                        st.write(f"  - {dep}")
        
        with col2:
            st.subheader("⬇️ Downstream Impact")
            downstream = {}
            for model, deps in metadata["lineage"].items():
                for dep in deps:
                    if dep not in downstream:
                        downstream[dep] = []
                    downstream[dep].append(model)
            
            for model, impacts in downstream.items():
                st.write(f"**{model}** impacts:")
                for impact in impacts:
                    st.write(f"  - {impact}")
    
    elif page == "💾 BigQuery Statistics":
        st.header("💾 BigQuery Statistics")
        
        if "error" in bq_stats:
            st.error(f"Could not connect to BigQuery: {bq_stats['error']}")
        else:
            # Overview metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Datasets", bq_stats["total_datasets"])
            with col2:
                st.metric("Total Tables", bq_stats["total_tables"])
            
            # Table details
            if bq_stats["table_details"]:
                st.subheader("📊 Table Statistics")
                
                table_data = []
                for table_name, stats in bq_stats["table_details"].items():
                    table_data.append({
                        "Table": table_name,
                        "Rows": f"{stats['rows']:,}",
                        "Size (MB)": stats['size_mb'],
                        "Created": stats['created'],
                        "Modified": stats['modified']
                    })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
    
    elif page == "📋 Data Quality Tests":
        st.header("📋 Data Quality Tests")
        
        # Aggregate test information
        test_summary = {"total": 0, "by_type": {}, "by_model": {}}
        
        for model_name, model_data in metadata["models"].items():
            model_tests = len(model_data["tests"])
            column_tests = sum(len(col["tests"]) for col in model_data["columns"].values())
            total_tests = model_tests + column_tests
            
            test_summary["total"] += total_tests
            test_summary["by_model"][model_name] = total_tests
            
            # Count test types
            all_tests = model_data["tests"] + [test for col in model_data["columns"].values() for test in col["tests"]]
            for test in all_tests:
                test_type = str(test).split("(")[0] if isinstance(test, str) else str(type(test).__name__)
                test_summary["by_type"][test_type] = test_summary["by_type"].get(test_type, 0) + 1
        
        # Display summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Test Distribution by Model")
            if test_summary["by_model"]:
                fig = px.bar(x=list(test_summary["by_model"].keys()),
                           y=list(test_summary["by_model"].values()),
                           title="Tests per Model")
                st.plotly_chart(fig)
        
        with col2:
            st.subheader("🔍 Test Types")
            for test_type, count in test_summary["by_type"].items():
                st.write(f"**{test_type}:** {count}")
        
        # Detailed test information
        st.subheader("📋 Detailed Test Coverage")
        
        for model_name, model_data in metadata["models"].items():
            with st.expander(f"🧪 {model_name} Tests"):
                if model_data["tests"]:
                    st.write("**Model-level tests:**")
                    for test in model_data["tests"]:
                        st.markdown(f'<div class="test-info">🔬 {test}</div>', unsafe_allow_html=True)
                
                if any(col["tests"] for col in model_data["columns"].values()):
                    st.write("**Column-level tests:**")
                    for col_name, col_data in model_data["columns"].items():
                        if col_data["tests"]:
                            st.write(f"*{col_name}:*")
                            for test in col_data["tests"]:
                                st.markdown(f'<div class="test-info">  🔬 {test}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 