"""
Dataset loader for cybersecurity datasets (UNSW-NB15, CICIDS2017)
"""
import os
import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from urllib.parse import urlparse
from tqdm import tqdm

from ..config.config import DATASETS, RAW_DATA_DIR

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and manage cybersecurity datasets"""
    
    def __init__(self):
        self.datasets_config = DATASETS
        self.raw_data_dir = RAW_DATA_DIR
        
    def download_file(self, url: str, filename: str, chunk_size: int = 8192) -> bool:
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            filepath = self.raw_data_dir / filename
            
            with open(filepath, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = file.write(chunk)
                    progress_bar.update(size)
            
            logger.info(f"Downloaded {filename} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {str(e)}")
            return False
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """Extract zip file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info(f"Extracted {zip_path.name} successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to extract {zip_path.name}: {str(e)}")
            return False
    
    def load_unsw_nb15(self, download: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load UNSW-NB15 dataset"""
        config = self.datasets_config["unsw_nb15"]
        train_file = self.raw_data_dir / config["filename"]
        test_file = self.raw_data_dir / config["test_filename"]
        
        # Download if files don't exist and download is requested
        if download and (not train_file.exists() or not test_file.exists()):
            logger.info("UNSW-NB15 dataset not found. Please download manually from:")
            logger.info(config["url"])
            logger.info("Place the files in the data/raw directory")
        
        try:
            # Load training data
            if train_file.exists():
                train_data = pd.read_csv(train_file)
                logger.info(f"Loaded UNSW-NB15 training data: {train_data.shape}")
            else:
                logger.warning("Training file not found. Creating sample data...")
                train_data = self._create_sample_unsw_data(n_samples=10000, is_training=True)
            
            # Load test data
            if test_file.exists():
                test_data = pd.read_csv(test_file)
                logger.info(f"Loaded UNSW-NB15 test data: {test_data.shape}")
            else:
                logger.warning("Test file not found. Creating sample data...")
                test_data = self._create_sample_unsw_data(n_samples=2000, is_training=False)
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error loading UNSW-NB15 dataset: {str(e)}")
            # Return sample data as fallback
            return self._create_sample_unsw_data(10000, True), self._create_sample_unsw_data(2000, False)
    
    def load_cicids2017(self, download: bool = True) -> pd.DataFrame:
        """Load CICIDS2017 dataset"""
        config = self.datasets_config["cicids2017"]
        zip_file = self.raw_data_dir / config["filename"]
        
        if download and not zip_file.exists():
            logger.info("CICIDS2017 dataset not found. Please download manually from:")
            logger.info(config["url"])
            logger.info("Place the MachineLearningCSV.zip file in the data/raw directory")
        
        try:
            if zip_file.exists():
                # Extract and load the dataset
                extract_dir = self.raw_data_dir / "cicids2017"
                extract_dir.mkdir(exist_ok=True)
                
                if self.extract_zip(zip_file, extract_dir):
                    # Find the main CSV file
                    csv_files = list(extract_dir.glob("*.csv"))
                    if csv_files:
                        data = pd.read_csv(csv_files[0])
                        logger.info(f"Loaded CICIDS2017 data: {data.shape}")
                        return data
            
            # Fallback to sample data
            logger.warning("CICIDS2017 file not found. Creating sample data...")
            return self._create_sample_cicids_data(n_samples=50000)
            
        except Exception as e:
            logger.error(f"Error loading CICIDS2017 dataset: {str(e)}")
            return self._create_sample_cicids_data(50000)
    
    def _create_sample_unsw_data(self, n_samples: int, is_training: bool = True) -> pd.DataFrame:
        """Create sample UNSW-NB15 data for testing"""
        np.random.seed(42)
        
        # UNSW-NB15 features
        features = {
            'srcip': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
            'sport': np.random.randint(1, 65535, n_samples),
            'dstip': [f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
            'dsport': np.random.randint(1, 65535, n_samples),
            'proto': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'state': np.random.choice(['FIN', 'CON', 'INT', 'REQ', 'RST'], n_samples),
            'dur': np.random.exponential(1.0, n_samples),
            'sbytes': np.random.poisson(1000, n_samples),
            'dbytes': np.random.poisson(1000, n_samples),
            'sttl': np.random.randint(1, 255, n_samples),
            'dttl': np.random.randint(1, 255, n_samples),
            'sloss': np.random.poisson(0, n_samples),
            'dloss': np.random.poisson(0, n_samples),
            'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', '-'], n_samples),
            'sload': np.random.exponential(1000, n_samples),
            'dload': np.random.exponential(1000, n_samples),
            'spkts': np.random.poisson(10, n_samples),
            'dpkts': np.random.poisson(10, n_samples),
            'swin': np.random.randint(0, 65535, n_samples),
            'dwin': np.random.randint(0, 65535, n_samples),
            'stcpb': np.random.randint(0, 1000000, n_samples),
            'dtcpb': np.random.randint(0, 1000000, n_samples),
            'smeansz': np.random.exponential(100, n_samples),
            'dmeansz': np.random.exponential(100, n_samples),
            'trans_depth': np.random.randint(0, 10, n_samples),
            'res_bdy_len': np.random.poisson(0, n_samples),
            'sjit': np.random.exponential(0.1, n_samples),
            'djit': np.random.exponential(0.1, n_samples),
            'stime': pd.date_range('2023-01-01', periods=n_samples, freq='1s'),
            'ltime': pd.date_range('2023-01-01', periods=n_samples, freq='1s'),
            'sinpkt': np.random.exponential(0.1, n_samples),
            'dinpkt': np.random.exponential(0.1, n_samples),
            'tcprtt': np.random.exponential(0.1, n_samples),
            'synack': np.random.exponential(0.1, n_samples),
            'ackdat': np.random.exponential(0.1, n_samples),
            'is_sm_ips_ports': np.random.choice([0, 1], n_samples),
            'ct_state_ttl': np.random.randint(0, 10, n_samples),
            'ct_flw_http_mthd': np.random.randint(0, 10, n_samples),
            'is_ftp_login': np.random.choice([0, 1], n_samples),
            'ct_ftp_cmd': np.random.randint(0, 10, n_samples),
            'ct_srv_src': np.random.randint(0, 100, n_samples),
            'ct_srv_dst': np.random.randint(0, 100, n_samples),
            'ct_dst_ltm': np.random.randint(0, 100, n_samples),
            'ct_src_ltm': np.random.randint(0, 100, n_samples),
            'ct_src_dport_ltm': np.random.randint(0, 100, n_samples),
            'ct_dst_sport_ltm': np.random.randint(0, 100, n_samples),
            'ct_dst_src_ltm': np.random.randint(0, 100, n_samples),
            'attack_cat': np.random.choice(['Normal', 'Fuzzers', 'Analysis', 'Backdoors', 'DoS', 
                                          'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'], n_samples),
            'label': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 80% normal, 20% attack
        }
        
        return pd.DataFrame(features)
    
    def _create_sample_cicids_data(self, n_samples: int) -> pd.DataFrame:
        """Create sample CICIDS2017 data for testing"""
        np.random.seed(42)
        
        # CICIDS2017 features (simplified)
        features = {
            'Dst Port': np.random.randint(1, 65535, n_samples),
            'Protocol': np.random.choice([6, 17, 1], n_samples),  # TCP, UDP, ICMP
            'Flow Duration': np.random.exponential(1000, n_samples),
            'Total Fwd Packets': np.random.poisson(10, n_samples),
            'Total Backward Packets': np.random.poisson(10, n_samples),
            'Total Length of Fwd Packets': np.random.poisson(1000, n_samples),
            'Total Length of Bwd Packets': np.random.poisson(1000, n_samples),
            'Fwd Packet Length Max': np.random.exponential(100, n_samples),
            'Fwd Packet Length Min': np.random.exponential(10, n_samples),
            'Fwd Packet Length Mean': np.random.exponential(50, n_samples),
            'Fwd Packet Length Std': np.random.exponential(20, n_samples),
            'Bwd Packet Length Max': np.random.exponential(100, n_samples),
            'Bwd Packet Length Min': np.random.exponential(10, n_samples),
            'Bwd Packet Length Mean': np.random.exponential(50, n_samples),
            'Bwd Packet Length Std': np.random.exponential(20, n_samples),
            'Flow Bytes/s': np.random.exponential(1000, n_samples),
            'Flow Packets/s': np.random.exponential(10, n_samples),
            'Flow IAT Mean': np.random.exponential(100, n_samples),
            'Flow IAT Std': np.random.exponential(50, n_samples),
            'Flow IAT Max': np.random.exponential(1000, n_samples),
            'Flow IAT Min': np.random.exponential(1, n_samples),
            'Fwd IAT Total': np.random.exponential(1000, n_samples),
            'Fwd IAT Mean': np.random.exponential(100, n_samples),
            'Fwd IAT Std': np.random.exponential(50, n_samples),
            'Fwd IAT Max': np.random.exponential(1000, n_samples),
            'Fwd IAT Min': np.random.exponential(1, n_samples),
            'Bwd IAT Total': np.random.exponential(1000, n_samples),
            'Bwd IAT Mean': np.random.exponential(100, n_samples),
            'Bwd IAT Std': np.random.exponential(50, n_samples),
            'Bwd IAT Max': np.random.exponential(1000, n_samples),
            'Bwd IAT Min': np.random.exponential(1, n_samples),
            'Fwd PSH Flags': np.random.choice([0, 1], n_samples),
            'Bwd PSH Flags': np.random.choice([0, 1], n_samples),
            'Fwd URG Flags': np.random.choice([0, 1], n_samples),
            'Bwd URG Flags': np.random.choice([0, 1], n_samples),
            'Fwd Header Length': np.random.randint(0, 100, n_samples),
            'Bwd Header Length': np.random.randint(0, 100, n_samples),
            'Fwd Packets/s': np.random.exponential(10, n_samples),
            'Bwd Packets/s': np.random.exponential(10, n_samples),
            'Min Packet Length': np.random.randint(0, 100, n_samples),
            'Max Packet Length': np.random.randint(100, 1500, n_samples),
            'Packet Length Mean': np.random.exponential(500, n_samples),
            'Packet Length Std': np.random.exponential(200, n_samples),
            'Packet Length Variance': np.random.exponential(40000, n_samples),
            'FIN Flag Count': np.random.choice([0, 1], n_samples),
            'SYN Flag Count': np.random.choice([0, 1], n_samples),
            'RST Flag Count': np.random.choice([0, 1], n_samples),
            'PSH Flag Count': np.random.choice([0, 1], n_samples),
            'ACK Flag Count': np.random.choice([0, 1], n_samples),
            'URG Flag Count': np.random.choice([0, 1], n_samples),
            'CWE Flag Count': np.random.choice([0, 1], n_samples),
            'ECE Flag Count': np.random.choice([0, 1], n_samples),
            'Down/Up Ratio': np.random.exponential(1, n_samples),
            'Average Packet Size': np.random.exponential(500, n_samples),
            'Avg Fwd Segment Size': np.random.exponential(500, n_samples),
            'Avg Bwd Segment Size': np.random.exponential(500, n_samples),
            'Fwd Avg Bytes/Bulk': np.random.exponential(100, n_samples),
            'Fwd Avg Packets/Bulk': np.random.exponential(10, n_samples),
            'Fwd Avg Bulk Rate': np.random.exponential(1000, n_samples),
            'Bwd Avg Bytes/Bulk': np.random.exponential(100, n_samples),
            'Bwd Avg Packets/Bulk': np.random.exponential(10, n_samples),
            'Bwd Avg Bulk Rate': np.random.exponential(1000, n_samples),
            'Subflow Fwd Packets': np.random.poisson(5, n_samples),
            'Subflow Fwd Bytes': np.random.poisson(500, n_samples),
            'Subflow Bwd Packets': np.random.poisson(5, n_samples),
            'Subflow Bwd Bytes': np.random.poisson(500, n_samples),
            'Init_Win_bytes_forward': np.random.randint(0, 65535, n_samples),
            'Init_Win_bytes_backward': np.random.randint(0, 65535, n_samples),
            'act_data_pkt_fwd': np.random.poisson(5, n_samples),
            'min_seg_size_forward': np.random.randint(0, 100, n_samples),
            'Active Mean': np.random.exponential(100, n_samples),
            'Active Std': np.random.exponential(50, n_samples),
            'Active Max': np.random.exponential(1000, n_samples),
            'Active Min': np.random.exponential(1, n_samples),
            'Idle Mean': np.random.exponential(100, n_samples),
            'Idle Std': np.random.exponential(50, n_samples),
            'Idle Max': np.random.exponential(1000, n_samples),
            'Idle Min': np.random.exponential(1, n_samples),
            'Label': np.random.choice(['BENIGN', 'DDoS', 'PortScan', 'Bot', 'Infiltration', 
                                     'Web Attack', 'Brute Force', 'SQL Injection'], n_samples, 
                                    p=[0.7, 0.1, 0.05, 0.05, 0.02, 0.03, 0.03, 0.02])
        }
        
        return pd.DataFrame(features)
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset"""
        if dataset_name in self.datasets_config:
            return self.datasets_config[dataset_name]
        else:
            raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(self.datasets_config.keys())}")
    
    def list_available_datasets(self) -> list:
        """List all available datasets"""
        return list(self.datasets_config.keys())
