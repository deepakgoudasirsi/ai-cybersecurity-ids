"""
Interactive dashboard for AI-powered intrusion detection system
"""

from .streamlit_dashboard import create_dashboard
from .dashboard_utils import DashboardUtils

__all__ = ['create_dashboard', 'DashboardUtils']
