"""
Utility functions for the dashboard
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class DashboardUtils:
    """Utility functions for dashboard components"""
    
    @staticmethod
    def create_metric_card(title: str, value: Any, delta: Optional[Any] = None, 
                          delta_color: str = "normal") -> str:
        """Create a metric card HTML"""
        delta_html = ""
        if delta is not None:
            delta_color_class = {
                "normal": "",
                "inverse": "delta-down",
                "off": "delta-off"
            }.get(delta_color, "")
            delta_html = f'<div class="delta {delta_color_class}">{delta}</div>'
        
        return f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """
    
    @staticmethod
    def create_alert_card(alert_level: str, title: str, description: str, 
                         timestamp: datetime, confidence: float = None) -> str:
        """Create an alert card HTML"""
        level_classes = {
            "critical": "alert-critical",
            "high": "alert-high", 
            "medium": "alert-medium",
            "low": "alert-low",
            "normal": "alert-normal"
        }
        
        level_icons = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "⚡",
            "low": "ℹ️",
            "normal": "✅"
        }
        
        confidence_html = ""
        if confidence is not None:
            confidence_html = f"<br>Confidence: {confidence:.2%}"
        
        return f"""
        <div class="{level_classes.get(alert_level, 'alert-normal')}">
            <strong>{level_icons.get(alert_level, 'ℹ️')} {alert_level.upper()} ALERT</strong><br>
            <strong>{title}</strong><br>
            {description}<br>
            Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}{confidence_html}
        </div>
        """
    
    @staticmethod
    def create_detection_trends_chart(data: pd.DataFrame) -> go.Figure:
        """Create detection trends chart"""
        fig = go.Figure()
        
        if 'timestamp' in data.columns and 'detections' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['detections'],
                mode='lines+markers',
                name='Detections',
                line=dict(color='blue', width=2)
            ))
        
        if 'false_positives' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['false_positives'],
                mode='lines+markers',
                name='False Positives',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(
            title='Detection Trends',
            xaxis_title='Time',
            yaxis_title='Count',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_attack_distribution_chart(attack_data: Dict[str, int]) -> go.Figure:
        """Create attack distribution pie chart"""
        labels = list(attack_data.keys())
        values = list(attack_data.values())
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors[:len(labels)]
        )])
        
        fig.update_layout(
            title='Attack Type Distribution',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_model_performance_chart(model_data: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create model performance comparison chart"""
        models = list(model_data.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(metrics):
            values = [model_data[model].get(metric, 0) for model in models]
            
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Model Performance Comparison',
            height=600,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_alert_timeline_chart(alert_data: List[Dict[str, Any]]) -> go.Figure:
        """Create alert timeline chart"""
        if not alert_data:
            return go.Figure()
        
        df = pd.DataFrame(alert_data)
        
        # Group alerts by level and time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.floor('H')
        
        alert_counts = df.groupby(['hour', 'alert_level']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        colors = {
            'critical': 'red',
            'high': 'orange',
            'medium': 'yellow',
            'low': 'lightblue',
            'normal': 'green'
        }
        
        for level in alert_counts.columns:
            if level in colors:
                fig.add_trace(go.Scatter(
                    x=alert_counts.index,
                    y=alert_counts[level],
                    mode='lines+markers',
                    name=level.title(),
                    line=dict(color=colors[level], width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title='Alert Timeline',
            xaxis_title='Time',
            yaxis_title='Number of Alerts',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_geographic_attack_chart(geo_data: Dict[str, int]) -> go.Figure:
        """Create geographic attack distribution chart"""
        countries = list(geo_data.keys())
        attack_counts = list(geo_data.values())
        
        fig = go.Figure(data=go.Bar(
            x=countries,
            y=attack_counts,
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Attack Distribution by Country',
            xaxis_title='Country',
            yaxis_title='Number of Attacks',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_network_traffic_chart(traffic_data: pd.DataFrame) -> go.Figure:
        """Create network traffic visualization"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Traffic Volume', 'Packet Size Distribution'),
            vertical_spacing=0.1
        )
        
        # Traffic volume over time
        if 'timestamp' in traffic_data.columns and 'bytes' in traffic_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=traffic_data['timestamp'],
                    y=traffic_data['bytes'],
                    mode='lines',
                    name='Traffic Volume',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # Packet size distribution
        if 'packets' in traffic_data.columns:
            fig.add_trace(
                go.Histogram(
                    x=traffic_data['packets'],
                    name='Packet Size Distribution',
                    marker_color='green'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Network Traffic Analysis',
            height=600,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_threat_intelligence_chart(threat_data: Dict[str, Any]) -> go.Figure:
        """Create threat intelligence visualization"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Top Attacking IPs', 'Threat Categories'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Top attacking IPs
        if 'top_ips' in threat_data:
            ips = list(threat_data['top_ips'].keys())
            counts = list(threat_data['top_ips'].values())
            
            fig.add_trace(
                go.Bar(
                    x=ips,
                    y=counts,
                    name='Attack Count',
                    marker_color='red'
                ),
                row=1, col=1
            )
        
        # Threat categories
        if 'threat_categories' in threat_data:
            categories = list(threat_data['threat_categories'].keys())
            values = list(threat_data['threat_categories'].values())
            
            fig.add_trace(
                go.Pie(
                    labels=categories,
                    values=values,
                    name='Threat Categories'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Threat Intelligence Dashboard',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_performance_metrics_chart(metrics_data: Dict[str, List[float]]) -> go.Figure:
        """Create performance metrics over time chart"""
        fig = go.Figure()
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title='Performance Metrics Over Time',
            xaxis_title='Time Period',
            yaxis_title='Metric Value',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    @staticmethod
    def format_number(value: float, precision: int = 2) -> str:
        """Format number with appropriate units"""
        if value >= 1e9:
            return f"{value/1e9:.{precision}f}B"
        elif value >= 1e6:
            return f"{value/1e6:.{precision}f}M"
        elif value >= 1e3:
            return f"{value/1e3:.{precision}f}K"
        else:
            return f"{value:.{precision}f}"
    
    @staticmethod
    def format_percentage(value: float, precision: int = 1) -> str:
        """Format percentage"""
        return f"{value*100:.{precision}f}%"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    @staticmethod
    def get_alert_color(alert_level: str) -> str:
        """Get color for alert level"""
        colors = {
            'critical': '#f44336',
            'high': '#ff9800',
            'medium': '#ffeb3b',
            'low': '#4caf50',
            'normal': '#2196f3'
        }
        return colors.get(alert_level.lower(), '#9e9e9e')
    
    @staticmethod
    def calculate_uptime(start_time: datetime, current_time: datetime = None) -> str:
        """Calculate system uptime"""
        if current_time is None:
            current_time = datetime.now()
        
        duration = current_time - start_time
        return DashboardUtils.format_duration(duration.total_seconds())
    
    @staticmethod
    def generate_sample_data(data_type: str, n_samples: int = 100) -> pd.DataFrame:
        """Generate sample data for testing"""
        np.random.seed(42)
        
        if data_type == 'detection_trends':
            dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='1H')
            return pd.DataFrame({
                'timestamp': dates,
                'detections': np.random.poisson(10, n_samples),
                'false_positives': np.random.poisson(1, n_samples)
            })
        
        elif data_type == 'attack_distribution':
            attack_types = ['DDoS', 'Port Scan', 'Brute Force', 'Malware', 'Normal']
            counts = np.random.poisson(50, len(attack_types))
            return pd.DataFrame({
                'attack_type': attack_types,
                'count': counts
            })
        
        elif data_type == 'model_performance':
            models = ['Random Forest', 'XGBoost', 'SVM', 'LSTM', 'CNN']
            return pd.DataFrame({
                'model': models,
                'accuracy': np.random.uniform(0.8, 0.95, len(models)),
                'precision': np.random.uniform(0.8, 0.95, len(models)),
                'recall': np.random.uniform(0.8, 0.95, len(models)),
                'f1_score': np.random.uniform(0.8, 0.95, len(models))
            })
        
        else:
            return pd.DataFrame()
