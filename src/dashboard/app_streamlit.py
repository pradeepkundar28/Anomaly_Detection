"""
Streamlit web dashboard for oil rig anomaly detection.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import run_pipeline
from src.config import AppConfig, load_config
from src.utils.logger import setup_logger

# Page configuration
st.set_page_config(
    page_title="Oil Rig Anomaly Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = setup_logger("streamlit_app", log_dir="logs")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(output_dir: str = "output"):
    """Load data from output directory."""
    try:
        sensor_df = pd.read_csv(f"{output_dir}/sensor_data.csv")
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
        
        logs_df = pd.read_csv(f"{output_dir}/operator_logs.csv")
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
        
        anomalies_df = pd.read_csv(f"{output_dir}/anomalies.csv")
        anomalies_df['timestamp'] = pd.to_datetime(anomalies_df['timestamp'])
        
        correlated_df = pd.read_csv(f"{output_dir}/correlated_logs.csv")
        
        with open(f"{output_dir}/summary.txt", 'r') as f:
            summary = f.read()
        
        return sensor_df, logs_df, anomalies_df, correlated_df, summary
    except FileNotFoundError:
        return None, None, None, None, None


def plot_sensor_timeseries(sensor_df, anomalies_df, equipment_id, sensor_type):
    """Create time series plot for specific equipment and sensor."""
    # Filter data
    sensor_data = sensor_df[
        (sensor_df['equipment_id'] == equipment_id) & 
        (sensor_df['sensor_type'] == sensor_type)
    ]
    
    anomaly_data = anomalies_df[
        (anomalies_df['equipment_id'] == equipment_id) & 
        (anomalies_df['is_anomaly'] == True)
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add sensor readings
    fig.add_trace(go.Scatter(
        x=sensor_data['timestamp'],
        y=sensor_data['value'],
        mode='lines',
        name='Sensor Reading',
        line=dict(color='blue', width=1)
    ))
    
    # Add anomaly markers
    if not anomaly_data.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_data['timestamp'],
            y=anomaly_data[sensor_type],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    fig.update_layout(
        title=f"{sensor_type.capitalize()} - {equipment_id}",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_combined_sensor_and_anomaly_scores(sensor_df, anomalies_df, equipment_id):
    """
    Create combined plot showing all sensor values with anomaly markers on a single graph.
    """
    # Filter anomaly data for this equipment
    equip_anomalies = anomalies_df[anomalies_df['equipment_id'] == equipment_id].copy()
    equip_anomalies = equip_anomalies.sort_values('timestamp')
    
    # Create figure
    fig = go.Figure()
    
    # Add each sensor type as a line
    colors = {'pressure': '#1f77b4', 'temperature': '#ff7f0e', 'vibration': '#2ca02c'}
    
    for sensor_type in ['pressure', 'temperature', 'vibration']:
        if sensor_type in equip_anomalies.columns:
            # Add sensor values line
            fig.add_trace(
                go.Scatter(
                    x=equip_anomalies['timestamp'],
                    y=equip_anomalies[sensor_type],
                    mode='lines',
                    name=f'{sensor_type.capitalize()}',
                    line=dict(color=colors.get(sensor_type, 'gray'), width=2)
                )
            )
    
    # Add anomaly markers for each sensor type
    anomaly_points = equip_anomalies[equip_anomalies['is_anomaly']]
    if not anomaly_points.empty:
        for sensor_type in ['pressure', 'temperature', 'vibration']:
            if sensor_type in anomaly_points.columns:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_points['timestamp'],
                        y=anomaly_points[sensor_type],
                        mode='markers',
                        name=f'{sensor_type.capitalize()} Anomalies',
                        marker=dict(
                            size=10,
                            color='red',
                            symbol='x',
                            line=dict(width=2, color='darkred')
                        ),
                        showlegend=False
                    )
                )
    
    # Update layout
    fig.update_layout(
        title=f"Combined Sensor Values with Anomalies - {equipment_id}",
        xaxis_title="Timestamp",
        yaxis_title="Sensor Values",
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_anomaly_distribution(anomalies_df):
    """Create bar chart of anomalies by equipment."""
    anomaly_counts = anomalies_df[anomalies_df['is_anomaly']].groupby('equipment_id').size().reset_index(name='count')
    
    fig = px.bar(
        anomaly_counts,
        x='equipment_id',
        y='count',
        title='Anomaly Distribution by Equipment',
        labels={'equipment_id': 'Equipment ID', 'count': 'Number of Anomalies'},
        color='count',
        color_continuous_scale='Reds'
    )
    
    return fig


def plot_anomaly_scores(anomalies_df):
    """Create histogram of anomaly scores."""
    fig = px.histogram(
        anomalies_df,
        x='anomaly_score',
        nbins=50,
        title='Distribution of Anomaly Scores',
        labels={'anomaly_score': 'Anomaly Score', 'count': 'Frequency'},
        color_discrete_sequence=['steelblue']
    )
    
    fig.add_vline(
        x=anomalies_df[anomalies_df['is_anomaly']]['anomaly_score'].min(),
        line_dash="dash",
        line_color="red",
        annotation_text="Anomaly Threshold"
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🛢️ Oil Rig Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data generation options
    with st.sidebar.expander("🔧 Data Generation Settings", expanded=False):
        contamination = st.slider(
            "Anomaly Contamination Rate",
            min_value=0.01,
            max_value=0.10,
            value=0.03,
            step=0.01,
            help="Expected proportion of anomalies in the dataset"
        )
        
        output_dir = st.text_input("Output Directory", value="output")
    
    # Run pipeline button
    if st.sidebar.button("🚀 Run Detection Pipeline", type="primary"):
        with st.spinner("Running anomaly detection pipeline..."):
            try:
                sensor_df, logs_df, anomalies_df, correlated_df, report_text = run_pipeline(
                    output_dir=output_dir,
                    contamination=contamination
                )
                st.success("✅ Pipeline completed successfully!")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
                logger.error(f"Pipeline error: {e}", exc_info=True)
                return
    
    # Load existing data
    sensor_df, logs_df, anomalies_df, correlated_df, summary = load_data(output_dir)
    
    if sensor_df is None:
        st.info("👈 Click 'Run Detection Pipeline' in the sidebar to generate data and detect anomalies.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Time Series",
        "🔍 Anomalies",
        "📝 Operator Logs",
        "📄 Summary Report"
    ])
    
    with tab1:
        st.header("System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Data Points",
                f"{len(sensor_df):,}",
                help="Total number of sensor readings"
            )
        
        with col2:
            n_anomalies = anomalies_df['is_anomaly'].sum()
            anomaly_rate = (n_anomalies / len(anomalies_df)) * 100
            st.metric(
                "Anomalies Detected",
                f"{n_anomalies:,}",
                f"{anomaly_rate:.2f}%",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Equipment Monitored",
                anomalies_df['equipment_id'].nunique(),
                help="Number of unique equipment types"
            )
        
        with col4:
            st.metric(
                "Operator Logs",
                f"{len(logs_df):,}",
                help="Total number of operator log entries"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_anomaly_distribution(anomalies_df), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_anomaly_scores(anomalies_df), use_container_width=True)
    
    with tab2:
        st.header("Sensor Time Series Analysis")
        
        # View selection
        view_mode = st.radio(
            "Visualization Mode",
            options=["Combined View (All Sensors)", "Individual Sensor View"],
            horizontal=True
        )
        
        # Equipment selection
        equipment_id = st.selectbox(
            "Select Equipment",
            options=sorted(sensor_df['equipment_id'].unique()),
            key="equipment_selector"
        )
        
        if view_mode == "Combined View (All Sensors)":
            # Combined plot showing all sensors
            st.subheader(f"📊 Combined Analysis - {equipment_id}")
            st.markdown("""
            **This plot shows:**
            - 📈 All sensor values (pressure, temperature, vibration) on the same chart
            - 🔴 **Red X markers**: Detected anomalies on each sensor
            """)
            
            fig = plot_combined_sensor_and_anomaly_scores(sensor_df, anomalies_df, equipment_id)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show anomaly statistics for this equipment
            equip_anomalies = anomalies_df[anomalies_df['equipment_id'] == equipment_id]
            n_equip_anomalies = equip_anomalies['is_anomaly'].sum()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Points", len(equip_anomalies))
            col2.metric("Anomalies", n_equip_anomalies)
            col3.metric("Anomaly Rate", f"{(n_equip_anomalies/len(equip_anomalies)*100):.2f}%")
            col4.metric("Max Anomaly Score", f"{equip_anomalies['anomaly_score'].max():.3f}")
            
        else:
            # Individual sensor view
            sensor_type = st.selectbox(
                "Select Sensor Type",
                options=sorted(sensor_df['sensor_type'].unique())
            )
            
            # Plot
            fig = plot_sensor_timeseries(sensor_df, anomalies_df, equipment_id, sensor_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            sensor_stats = sensor_df[
                (sensor_df['equipment_id'] == equipment_id) & 
                (sensor_df['sensor_type'] == sensor_type)
            ]['value'].describe()
            
            st.subheader("Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{sensor_stats['mean']:.2f}")
            col2.metric("Std Dev", f"{sensor_stats['std']:.2f}")
            col3.metric("Min", f"{sensor_stats['min']:.2f}")
            col4.metric("Max", f"{sensor_stats['max']:.2f}")
    
    with tab3:
        st.header("Detected Anomalies")
        
        # Filter controls
        col1, col2 = st.columns(2)
        
        with col1:
            equipment_filter = st.multiselect(
                "Filter by Equipment",
                options=sorted(anomalies_df['equipment_id'].unique()),
                default=sorted(anomalies_df['equipment_id'].unique())
            )
        
        with col2:
            score_threshold = st.slider(
                "Minimum Anomaly Score",
                min_value=float(anomalies_df['anomaly_score'].min()),
                max_value=float(anomalies_df['anomaly_score'].max()),
                value=float(anomalies_df[anomalies_df['is_anomaly']]['anomaly_score'].min())
            )
        
        # Filter data
        filtered_anomalies = anomalies_df[
            (anomalies_df['is_anomaly']) &
            (anomalies_df['equipment_id'].isin(equipment_filter)) &
            (anomalies_df['anomaly_score'] >= score_threshold)
        ].sort_values('anomaly_score', ascending=False)
        
        st.write(f"**Showing {len(filtered_anomalies)} anomalies**")
        
        # Display table
        display_cols = ['timestamp', 'equipment_id', 'anomaly_score', 'pressure', 'temperature', 'vibration']
        st.dataframe(
            filtered_anomalies[display_cols].head(100),
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = filtered_anomalies.to_csv(index=False)
        st.download_button(
            "📥 Download Anomalies CSV",
            csv,
            "anomalies.csv",
            "text/csv",
            key='download-csv'
        )
    
    with tab4:
        st.header("Operator Logs & Correlations")
        
        # Display correlated logs
        if not correlated_df.empty:
            st.subheader("Top Correlated Logs with Anomalies")
            
            for idx, row in correlated_df.head(20).iterrows():
                with st.expander(
                    f"🔗 {row['anomaly_timestamp']} - {row['anomaly_equipment_id']} (Similarity: {row['similarity']:.3f})"
                ):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("**Anomaly Details:**")
                        st.write(f"- Equipment: {row['anomaly_equipment_id']}")
                        st.write(f"- Time: {row['anomaly_timestamp']}")
                        st.write(f"- Score: {row['anomaly_score']:.3f}")
                    
                    with col2:
                        st.write("**Correlated Log:**")
                        st.write(f"- Equipment: {row['log_equipment_id']}")
                        st.write(f"- Time: {row['log_timestamp']}")
                        st.info(row['log_text'])
        else:
            st.info("No strongly correlated logs found for the detected anomalies.")
        
        # All operator logs
        st.subheader("All Operator Logs")
        
        equipment_filter_logs = st.multiselect(
            "Filter by Equipment",
            options=sorted(logs_df['equipment_id'].unique()),
            default=sorted(logs_df['equipment_id'].unique()),
            key='logs_equipment_filter'
        )
        
        filtered_logs = logs_df[logs_df['equipment_id'].isin(equipment_filter_logs)]
        st.dataframe(
            filtered_logs.sort_values('timestamp', ascending=False).head(100),
            use_container_width=True,
            hide_index=True
        )
    
    with tab5:
        st.header("Summary Report")
        st.text(summary)
        
        # Download button
        st.download_button(
            "📥 Download Report",
            summary,
            "anomaly_report.txt",
            "text/plain"
        )


if __name__ == "__main__":
    main()
