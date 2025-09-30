#!/usr/bin/env python3
"""
Simple script to run the Streamlit dashboard
"""
import subprocess
import sys
import os

def main():
    """Run the Streamlit dashboard"""
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/dashboard/streamlit_dashboard.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
