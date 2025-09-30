"""
Simplified Streamlit dashboard for AI-powered Intrusion Detection System
Works without heavy dependencies like boto3, torch, etc.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
import random

# Page configuration
st.set_page_config(
    page_title="AI Cybersecurity IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ffebee !important;
        border: 2px solid #f44336 !important;
        border-left: 6px solid #f44336 !important;
        padding: 1.5rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 4px rgba(244, 67, 54, 0.3) !important;
        color: #2c2c2c !important;
    }
    .alert-critical p, .alert-critical strong {
        color: #2c2c2c !important;
    }
    .alert-high {
        background-color: #fff3e0 !important;
        border: 2px solid #ff9800 !important;
        border-left: 6px solid #ff9800 !important;
        padding: 1.5rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 4px rgba(255, 152, 0, 0.3) !important;
        color: #2c2c2c !important;
    }
    .alert-high p, .alert-high strong {
        color: #2c2c2c !important;
    }
    .alert-medium {
        background-color: #fffde7 !important;
        border: 2px solid #ffeb3b !important;
        border-left: 6px solid #ffeb3b !important;
        padding: 1.5rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 4px rgba(255, 235, 59, 0.3) !important;
        color: #2c2c2c !important;
    }
    .alert-medium p, .alert-medium strong {
        color: #2c2c2c !important;
    }
    .alert-low {
        background-color: #e8f5e8 !important;
        border: 2px solid #4caf50 !important;
        border-left: 6px solid #4caf50 !important;
        padding: 1.5rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.3) !important;
        color: #2c2c2c !important;
    }
    .alert-low p, .alert-low strong {
        color: #2c2c2c !important;
    }
</style>
""", unsafe_allow_html=True)

class SimpleDashboard:
    """Simplified dashboard for demonstration"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'detection_active' not in st.session_state:
            st.session_state.detection_active = False
        if 'detection_stats' not in st.session_state:
            st.session_state.detection_stats = {
                'total_processed': 0,
                'total_detected': 0,
                'detection_rate': 0.0,
                'false_positive_rate': 0.02,
                'avg_processing_time': 0.05
            }
        if 'recent_alerts' not in st.session_state:
            st.session_state.recent_alerts = []
    
    def run(self):
        """Run the main dashboard"""
        # Header
        st.markdown('<h1 class="main-header">🛡️ AI-Powered Intrusion Detection System</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content based on page selection
        page = st.session_state.get('current_page', 'Dashboard')
        
        if page == 'Dashboard':
            self.render_dashboard()
        elif page == 'Data Management':
            self.render_data_management()
        elif page == 'Model Training':
            self.render_model_training()
        elif page == 'Real-time Detection':
            self.render_realtime_detection()
        elif page == 'Alerts & Monitoring':
            self.render_alerts_monitoring()
        elif page == 'Analytics':
            self.render_analytics()
        elif page == 'Settings':
            self.render_settings()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("Navigation")
        
        pages = [
            "Dashboard",
            "Data Management", 
            "Model Training",
            "Real-time Detection",
            "Alerts & Monitoring",
            "Analytics",
            "Settings"
        ]
        
        selected_page = st.sidebar.selectbox("Select Page", pages, 
                                           index=pages.index(st.session_state.get('current_page', 'Dashboard')))
        st.session_state.current_page = selected_page
        
        # System status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### System Status")
        
        st.sidebar.success("✅ System Ready")
        
        if st.session_state.detection_active:
            st.sidebar.success("🟢 Detection Active")
        else:
            st.sidebar.warning("🟡 Detection Inactive")
        
        # Quick stats
        stats = st.session_state.detection_stats
        st.sidebar.metric("Total Processed", stats.get('total_processed', 0))
        st.sidebar.metric("Total Detected", stats.get('total_detected', 0))
        st.sidebar.metric("Detection Rate", f"{stats.get('detection_rate', 0):.2%}")
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.markdown("## 📊 System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Models",
                value=8,
                delta=None
            )
        
        with col2:
            st.metric(
                label="Detection Rate",
                value=f"{st.session_state.detection_stats.get('detection_rate', 0):.2%}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="False Positive Rate",
                value=f"{st.session_state.detection_stats.get('false_positive_rate', 0):.2%}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Avg Processing Time",
                value=f"{st.session_state.detection_stats.get('avg_processing_time', 0):.3f}s",
                delta=None
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Detection Trends")
            self.render_detection_trends()
        
        with col2:
            st.markdown("### Alert Levels")
            self.render_alert_levels()
        
        # Recent alerts
        st.markdown("### 🚨 Recent Alerts")
        self.render_recent_alerts()
    
    def render_data_management(self):
        """Render data management page"""
        st.markdown("## 📁 Data Management")
        
        tab1, tab2, tab3 = st.tabs(["Load Datasets", "Generate Data", "Data Preview"])
        
        with tab1:
            st.markdown("### Load Cybersecurity Datasets")
            
            dataset_type = st.selectbox("Select Dataset", ["UNSW-NB15", "CICIDS2017", "Custom"])
            
            if st.button("Load Dataset"):
                with st.spinner("Loading dataset..."):
                    time.sleep(2)  # Simulate loading
                    st.success(f"Loaded {dataset_type} dataset successfully!")
                    st.session_state.dataset_loaded = True
        
        with tab2:
            st.markdown("### Generate Synthetic Traffic Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_samples = st.number_input("Number of Samples", min_value=1000, max_value=100000, value=10000)
                attack_ratio = st.slider("Attack Ratio", 0.0, 0.5, 0.1)
            
            with col2:
                attack_types = st.multiselect("Attack Types", 
                                            ["DDoS", "Port Scan", "Brute Force", "Malware"],
                                            default=["DDoS", "Port Scan"])
            
            if st.button("Generate Data"):
                with st.spinner("Generating synthetic data..."):
                    time.sleep(2)  # Simulate generation
                    st.success(f"Generated {n_samples} samples with {attack_ratio:.1%} attack ratio")
                    st.session_state.generated_data = True
        
        with tab3:
            st.markdown("### Data Preview")
            
            if st.session_state.get('generated_data', False):
                # Generate sample data for preview
                sample_data = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
                    'src_ip': [f"192.168.1.{random.randint(1,255)}" for _ in range(100)],
                    'dst_ip': [f"10.0.0.{random.randint(1,255)}" for _ in range(100)],
                    'protocol': random.choices(['TCP', 'UDP', 'ICMP'], k=100),
                    'packets': np.random.poisson(10, 100),
                    'bytes': np.random.poisson(1000, 100),
                    'attack_type': random.choices(['Normal', 'DDoS', 'Port Scan'], k=100)
                })
                
                st.dataframe(sample_data.head(20))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Data Summary")
                    st.write(f"Shape: {sample_data.shape}")
                    st.write(f"Columns: {list(sample_data.columns)}")
                
                with col2:
                    st.markdown("#### Attack Distribution")
                    attack_counts = sample_data['attack_type'].value_counts()
                    st.bar_chart(attack_counts)
            else:
                st.info("Please generate data first in the 'Generate Data' tab.")
    
    def render_model_training(self):
        """Render model training page"""
        st.markdown("## 🤖 Model Training")
        
        tab1, tab2, tab3 = st.tabs(["Train Models", "Model Performance", "Model Comparison"])
        
        with tab1:
            st.markdown("### Train Machine Learning Models")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Selection")
                ml_models = st.multiselect("ML Models", 
                                         ["Random Forest", "XGBoost", "SVM", "Logistic Regression"],
                                         default=["Random Forest", "XGBoost"])
                
                dl_models = st.multiselect("Deep Learning Models",
                                         ["LSTM", "CNN", "Autoencoder"],
                                         default=["LSTM"])
            
            with col2:
                st.markdown("#### Training Parameters")
                test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
                validation_size = st.slider("Validation Size", 0.1, 0.3, 0.1)
                balance_method = st.selectbox("Balance Method", ["smote", "adasyn", "undersample"])
            
            if st.button("Start Training"):
                with st.spinner("Training models..."):
                    time.sleep(3)  # Simulate training
                    st.success("Model training completed successfully!")
                    st.session_state.models_trained = True
        
        with tab2:
            st.markdown("### Model Performance")
            
            if st.session_state.get('models_trained', False):
                # Display model performance metrics
                st.markdown("#### Performance Metrics")
                
                # Create performance comparison chart
                models = ['Random Forest', 'XGBoost', 'SVM', 'LSTM', 'CNN']
                accuracy = [0.95, 0.93, 0.91, 0.89, 0.87]
                f1_scores = [0.94, 0.92, 0.90, 0.88, 0.86]
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Accuracy', 'F1-Score'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                fig.add_trace(
                    go.Bar(x=models, y=accuracy, name='Accuracy', marker_color='blue'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=models, y=f1_scores, name='F1-Score', marker_color='green'),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please train models first.")
        
        with tab3:
            st.markdown("### Model Comparison")
            
            if st.session_state.get('models_trained', False):
                # Model comparison table
                comparison_data = {
                    'Model': ['Random Forest', 'XGBoost', 'SVM', 'LSTM', 'CNN'],
                    'Accuracy': [0.95, 0.93, 0.91, 0.89, 0.87],
                    'Precision': [0.94, 0.92, 0.90, 0.88, 0.86],
                    'Recall': [0.93, 0.91, 0.89, 0.87, 0.85],
                    'F1-Score': [0.94, 0.92, 0.90, 0.88, 0.86],
                    'FPR': [0.02, 0.03, 0.04, 0.05, 0.06]
                }
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # Best model recommendation
                best_model = df.loc[df['F1-Score'].idxmax(), 'Model']
                st.success(f"🏆 Best Model: {best_model}")
            else:
                st.info("Please train models first.")
    
    def render_realtime_detection(self):
        """Render real-time detection page"""
        st.markdown("## 🔴 Real-time Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Detection Engine Status")
            
            # Detection controls
            col_start, col_stop = st.columns(2)
            
            with col_start:
                if st.button("Start Detection", disabled=st.session_state.detection_active):
                    st.session_state.detection_active = True
                    st.success("Detection started!")
            
            with col_stop:
                if st.button("Stop Detection", disabled=not st.session_state.detection_active):
                    st.session_state.detection_active = False
                    st.warning("Detection stopped!")
            
            # Real-time metrics
            if st.session_state.detection_active:
                # Simulate real-time updates
                if st.button("Update Metrics"):
                    # Simulate processing
                    stats = st.session_state.detection_stats
                    stats['total_processed'] += random.randint(50, 200)
                    stats['total_detected'] += random.randint(1, 10)
                    stats['detection_rate'] = stats['total_detected'] / max(stats['total_processed'], 1)
                    st.rerun()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Processed", st.session_state.detection_stats['total_processed'])
                
                with col2:
                    st.metric("Total Detected", st.session_state.detection_stats['total_detected'])
                
                with col3:
                    st.metric("Detection Rate", f"{st.session_state.detection_stats['detection_rate']:.2%}")
        
        with col2:
            st.markdown("### Model Performance")
            
            model_perf = {
                'Random Forest': {'predictions': 1250, 'detection_rate': 0.12, 'avg_confidence': 0.89},
                'XGBoost': {'predictions': 1180, 'detection_rate': 0.15, 'avg_confidence': 0.91},
                'LSTM': {'predictions': 1100, 'detection_rate': 0.18, 'avg_confidence': 0.87}
            }
            
            for model_name, perf in model_perf.items():
                st.markdown(f"**{model_name}**")
                st.write(f"Predictions: {perf['predictions']}")
                st.write(f"Detection Rate: {perf['detection_rate']:.2%}")
                st.write(f"Avg Confidence: {perf['avg_confidence']:.3f}")
                st.markdown("---")
        
        # Detection log
        st.markdown("### Recent Detections")
        
        # Generate sample detection data
        recent_detections = []
        for i in range(10):
            recent_detections.append({
                'timestamp': datetime.now() - timedelta(minutes=i*5),
                'source_ip': f"192.168.1.{random.randint(1,255)}",
                'destination_ip': f"10.0.0.{random.randint(1,255)}",
                'attack_type': random.choice(['DDoS', 'Port Scan', 'Brute Force', 'Normal']),
                'confidence': random.uniform(0.7, 0.99),
                'severity': random.choice(['Low', 'Medium', 'High', 'Critical'])
            })
        
        detection_df = pd.DataFrame(recent_detections)
        st.dataframe(detection_df, use_container_width=True)
    
    def render_alerts_monitoring(self):
        """Render alerts and monitoring page"""
        st.markdown("## 🚨 Alerts & Monitoring")
        
        tab1, tab2, tab3 = st.tabs(["Active Alerts", "Alert History", "Alert Configuration"])
        
        with tab1:
            st.markdown("### Active Alerts")
            
            # Simulate active alerts
            active_alerts = [
                {
                    'id': 'ALERT_001',
                    'timestamp': datetime.now() - timedelta(minutes=5),
                    'level': 'critical',
                    'description': 'Multiple failed login attempts detected',
                    'source_ip': '192.168.1.100',
                    'confidence': 0.95
                },
                {
                    'id': 'ALERT_002', 
                    'timestamp': datetime.now() - timedelta(minutes=15),
                    'level': 'high',
                    'description': 'Unusual traffic pattern detected',
                    'source_ip': '10.0.0.50',
                    'confidence': 0.87
                }
            ]
            
            for alert in active_alerts:
                if alert['level'] == 'critical':
                    st.markdown(f"""
                    <div class="alert-critical">
                        <h3 style="color: #d32f2f; margin-top: 0; font-weight: bold;">🚨 CRITICAL ALERT</h3>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">ID:</strong> {alert['id']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Time:</strong> {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Description:</strong> {alert['description']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Source IP:</strong> {alert['source_ip']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Confidence:</strong> <span style="color: #d32f2f; font-weight: bold; background-color: rgba(244, 67, 54, 0.1); padding: 2px 6px; border-radius: 3px;">{alert['confidence']:.2%}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                elif alert['level'] == 'high':
                    st.markdown(f"""
                    <div class="alert-high">
                        <h3 style="color: #f57c00; margin-top: 0; font-weight: bold;">⚠️ HIGH ALERT</h3>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">ID:</strong> {alert['id']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Time:</strong> {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Description:</strong> {alert['description']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Source IP:</strong> {alert['source_ip']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Confidence:</strong> <span style="color: #f57c00; font-weight: bold; background-color: rgba(255, 152, 0, 0.1); padding: 2px 6px; border-radius: 3px;">{alert['confidence']:.2%}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                elif alert['level'] == 'medium':
                    st.markdown(f"""
                    <div class="alert-medium">
                        <h3 style="color: #f9a825; margin-top: 0; font-weight: bold;">⚡ MEDIUM ALERT</h3>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">ID:</strong> {alert['id']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Time:</strong> {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Description:</strong> {alert['description']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Source IP:</strong> {alert['source_ip']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Confidence:</strong> <span style="color: #f9a825; font-weight: bold; background-color: rgba(255, 235, 59, 0.1); padding: 2px 6px; border-radius: 3px;">{alert['confidence']:.2%}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                elif alert['level'] == 'low':
                    st.markdown(f"""
                    <div class="alert-low">
                        <h3 style="color: #388e3c; margin-top: 0; font-weight: bold;">ℹ️ LOW ALERT</h3>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">ID:</strong> {alert['id']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Time:</strong> {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Description:</strong> {alert['description']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Source IP:</strong> {alert['source_ip']}</p>
                        <p style="color: #2c2c2c; margin: 0.5rem 0;"><strong style="color: #2c2c2c;">Confidence:</strong> <span style="color: #388e3c; font-weight: bold; background-color: rgba(76, 175, 80, 0.1); padding: 2px 6px; border-radius: 3px;">{alert['confidence']:.2%}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Alert History")
            
            # Alert statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Alerts", 156)
            
            with col2:
                st.metric("Critical Alerts", 12)
            
            with col3:
                st.metric("High Alerts", 34)
            
            with col4:
                st.metric("Medium Alerts", 110)
            
            # Alert trends chart
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
            critical_alerts = np.random.poisson(2, len(dates))
            high_alerts = np.random.poisson(5, len(dates))
            medium_alerts = np.random.poisson(10, len(dates))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=critical_alerts, name='Critical', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=dates, y=high_alerts, name='High', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dates, y=medium_alerts, name='Medium', line=dict(color='yellow')))
            
            fig.update_layout(title='Alert Trends (Last 30 Days)', xaxis_title='Date', yaxis_title='Number of Alerts')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Alert Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Email Notifications")
                email_enabled = st.checkbox("Enable Email Alerts", value=True)
                email_recipients = st.text_area("Email Recipients (one per line)", 
                                              "admin@company.com\nsecurity@company.com")
                
                st.markdown("#### Slack Notifications")
                slack_enabled = st.checkbox("Enable Slack Alerts", value=False)
                slack_webhook = st.text_input("Slack Webhook URL")
            
            with col2:
                st.markdown("#### Alert Thresholds")
                critical_threshold = st.slider("Critical Alert Threshold", 0.8, 1.0, 0.9)
                high_threshold = st.slider("High Alert Threshold", 0.6, 0.9, 0.7)
                medium_threshold = st.slider("Medium Alert Threshold", 0.4, 0.7, 0.5)
                
                st.markdown("#### Alert Rules")
                min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5)
                cooldown_minutes = st.number_input("Alert Cooldown (minutes)", 1, 60, 15)
            
            if st.button("Save Configuration"):
                st.success("Alert configuration saved!")
    
    def render_analytics(self):
        """Render analytics page"""
        st.markdown("## 📈 Analytics & Insights")
        
        tab1, tab2, tab3 = st.tabs(["Attack Patterns", "Performance Metrics", "Threat Intelligence"])
        
        with tab1:
            st.markdown("### Attack Pattern Analysis")
            
            # Attack type distribution
            attack_types = ['DDoS', 'Port Scan', 'Brute Force', 'Malware', 'Normal']
            attack_counts = [45, 32, 28, 15, 890]
            
            fig = px.pie(values=attack_counts, names=attack_types, title='Attack Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
            # Time-based attack patterns
            hours = list(range(24))
            attack_frequency = np.random.poisson(5, 24)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hours, y=attack_frequency, mode='lines+markers', name='Attack Frequency'))
            fig.update_layout(title='Attack Frequency by Hour', xaxis_title='Hour', yaxis_title='Number of Attacks')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Performance Metrics")
            
            # Model performance over time
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
            accuracy = np.random.normal(0.92, 0.02, len(dates))
            f1_score = np.random.normal(0.90, 0.02, len(dates))
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Accuracy Over Time', 'F1-Score Over Time'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=dates, y=accuracy, name='Accuracy', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=dates, y=f1_score, name='F1-Score', line=dict(color='green')),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Threat Intelligence")
            
            # Top source IPs
            source_ips = ['192.168.1.100', '10.0.0.50', '203.0.113.25', '198.51.100.10']
            attack_counts = [25, 18, 12, 8]
            
            fig = px.bar(x=source_ips, y=attack_counts, title='Top Attacking IP Addresses')
            fig.update_layout(xaxis_title='Source IP', yaxis_title='Number of Attacks')
            st.plotly_chart(fig, use_container_width=True)
            
            # Geographic distribution (simulated)
            countries = ['China', 'Russia', 'United States', 'Germany', 'Brazil']
            attack_counts = [45, 32, 28, 15, 12]
            
            fig = px.bar(x=countries, y=attack_counts, title='Attack Distribution by Country')
            fig.update_layout(xaxis_title='Country', yaxis_title='Number of Attacks')
            st.plotly_chart(fig, use_container_width=True)
    
    def render_settings(self):
        """Render settings page"""
        st.markdown("## ⚙️ System Settings")
        
        tab1, tab2, tab3 = st.tabs(["General", "Detection", "Notifications"])
        
        with tab1:
            st.markdown("### General Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### System Configuration")
                system_name = st.text_input("System Name", value="AI Cybersecurity IDS")
                timezone = st.selectbox("Timezone", ["UTC", "EST", "PST", "GMT"])
                log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
            
            with col2:
                st.markdown("#### Data Retention")
                log_retention_days = st.number_input("Log Retention (days)", 1, 365, 30)
                alert_retention_days = st.number_input("Alert Retention (days)", 1, 365, 90)
                model_retention_days = st.number_input("Model Retention (days)", 1, 365, 180)
        
        with tab2:
            st.markdown("### Detection Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Detection Parameters")
                batch_size = st.number_input("Batch Size", 100, 10000, 1000)
                processing_interval = st.number_input("Processing Interval (seconds)", 1, 60, 1)
                max_queue_size = st.number_input("Max Queue Size", 1000, 100000, 10000)
            
            with col2:
                st.markdown("#### Model Settings")
                ensemble_threshold = st.slider("Ensemble Threshold", 0.0, 1.0, 0.5)
                confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8)
                anomaly_threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.5)
        
        with tab3:
            st.markdown("### Notification Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Email Configuration")
                smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", 1, 65535, 587)
                email_username = st.text_input("Email Username")
                email_password = st.text_input("Email Password", type="password")
            
            with col2:
                st.markdown("#### Webhook Configuration")
                webhook_url = st.text_input("Webhook URL")
                webhook_headers = st.text_area("Webhook Headers (JSON format)", value='{"Content-Type": "application/json"}')
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")
    
    def render_detection_trends(self):
        """Render detection trends chart"""
        # Simulate detection trends data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        detections = np.random.poisson(50, len(dates))
        false_positives = np.random.poisson(5, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=detections, name='Detections', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=false_positives, name='False Positives', line=dict(color='red')))
        
        fig.update_layout(title='Detection Trends (Last 30 Days)', xaxis_title='Date', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    def render_alert_levels(self):
        """Render alert levels pie chart"""
        alert_levels = ['Normal', 'Low', 'Medium', 'High', 'Critical']
        alert_counts = [850, 120, 80, 25, 5]
        
        fig = px.pie(values=alert_counts, names=alert_levels, title='Alert Level Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recent_alerts(self):
        """Render recent alerts table"""
        # Simulate recent alerts
        recent_alerts = [
            {
                'Timestamp': datetime.now() - timedelta(minutes=5),
                'Level': 'Critical',
                'Description': 'Multiple failed login attempts',
                'Source IP': '192.168.1.100',
                'Confidence': 0.95
            },
            {
                'Timestamp': datetime.now() - timedelta(minutes=15),
                'Level': 'High', 
                'Description': 'Unusual traffic pattern',
                'Source IP': '10.0.0.50',
                'Confidence': 0.87
            },
            {
                'Timestamp': datetime.now() - timedelta(minutes=30),
                'Level': 'Medium',
                'Description': 'Port scan detected',
                'Source IP': '203.0.113.25',
                'Confidence': 0.72
            }
        ]
        
        df = pd.DataFrame(recent_alerts)
        st.dataframe(df, use_container_width=True)


def main():
    """Main function to run the dashboard"""
    dashboard = SimpleDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
