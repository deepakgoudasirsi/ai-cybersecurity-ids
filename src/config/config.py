"""
Configuration settings for AI-powered Intrusion Detection System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASETS = {
    "unsw_nb15": {
        "url": "https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/",
        "filename": "UNSW_NB15_training-set.csv",
        "test_filename": "UNSW_NB15_testing-set.csv",
        "features": 49,
        "target_column": "label"
    },
    "cicids2017": {
        "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "filename": "MachineLearningCSV.zip",
        "features": 78,
        "target_column": "Label"
    }
}

# Model configurations
MODEL_CONFIGS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "lstm": {
        "sequence_length": 50,
        "lstm_units": 64,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 32
    },
    "cnn": {
        "filters": 64,
        "kernel_size": 3,
        "pool_size": 2,
        "epochs": 50,
        "batch_size": 32
    },
    "autoencoder": {
        "encoding_dim": 32,
        "epochs": 100,
        "batch_size": 32,
        "validation_split": 0.2
    }
}

# Detection thresholds
DETECTION_THRESHOLDS = {
    "anomaly_threshold": 0.5,
    "intrusion_threshold": 0.7,
    "confidence_threshold": 0.8
}

# Real-time detection settings
REAL_TIME_CONFIG = {
    "batch_size": 1000,
    "processing_interval": 1,  # seconds
    "max_queue_size": 10000
}

# Dashboard settings
DASHBOARD_CONFIG = {
    "host": "0.0.0.0",
    "port": 8501,
    "debug": True,
    "auto_reload": True
}

# Cloud integration settings
CLOUD_CONFIG = {
    "aws": {
        "region": "us-east-1",
        "log_group": "ids-logs",
        "stream_name": "intrusion-detection"
    },
    "azure": {
        "resource_group": "ids-resources",
        "workspace_name": "ids-workspace"
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "ids.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}
