"""
Traffic data generator for simulating cloud network traffic
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import random
from faker import Faker

logger = logging.getLogger(__name__)


class TrafficDataGenerator:
    """Generate synthetic network traffic data for testing and training"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)
        
        # Attack patterns
        self.attack_patterns = {
            'ddos': {
                'frequency': 0.05,
                'characteristics': {
                    'high_packet_rate': True,
                    'same_destination': True,
                    'short_duration': True
                }
            },
            'port_scan': {
                'frequency': 0.03,
                'characteristics': {
                    'sequential_ports': True,
                    'same_source': True,
                    'connection_failures': True
                }
            },
            'brute_force': {
                'frequency': 0.02,
                'characteristics': {
                    'repeated_attempts': True,
                    'same_credentials': True,
                    'high_failure_rate': True
                }
            },
            'malware': {
                'frequency': 0.01,
                'characteristics': {
                    'unusual_ports': True,
                    'encrypted_traffic': True,
                    'command_control': True
                }
            }
        }
    
    def generate_normal_traffic(self, n_samples: int, start_time: datetime = None) -> pd.DataFrame:
        """Generate normal network traffic patterns"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        
        data = []
        for i in range(n_samples):
            # Normal traffic characteristics
            timestamp = start_time + timedelta(seconds=i * random.uniform(0.1, 5.0))
            
            # Generate realistic IP addresses
            src_ip = self._generate_ip_address()
            dst_ip = self._generate_ip_address()
            
            # Normal port distribution
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 22, 21, 25, 53, 110, 143, 993, 995])
            
            # Normal packet characteristics
            packets = random.randint(1, 50)
            bytes_transferred = random.randint(64, 1500 * packets)
            duration = random.uniform(0.1, 30.0)
            
            # Protocol distribution
            protocol = random.choices(['TCP', 'UDP', 'ICMP'], weights=[0.7, 0.25, 0.05])[0]
            
            # Normal flags
            flags = self._generate_normal_flags(protocol)
            
            data.append({
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'packets': packets,
                'bytes': bytes_transferred,
                'duration': duration,
                'flags': flags,
                'attack_type': 'Normal',
                'is_attack': 0
            })
        
        return pd.DataFrame(data)
    
    def generate_ddos_traffic(self, n_samples: int, start_time: datetime = None) -> pd.DataFrame:
        """Generate DDoS attack traffic patterns"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        
        data = []
        target_ip = self._generate_ip_address()
        
        for i in range(n_samples):
            timestamp = start_time + timedelta(seconds=i * random.uniform(0.01, 0.1))  # High frequency
            
            # Multiple sources attacking one target
            src_ip = self._generate_ip_address()
            dst_ip = target_ip
            
            # High packet rate
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 53])  # Common attack targets
            
            packets = random.randint(100, 1000)  # High packet count
            bytes_transferred = random.randint(1000, 10000)
            duration = random.uniform(0.01, 1.0)  # Short duration
            
            protocol = random.choices(['TCP', 'UDP'], weights=[0.6, 0.4])[0]
            flags = self._generate_attack_flags('ddos', protocol)
            
            data.append({
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'packets': packets,
                'bytes': bytes_transferred,
                'duration': duration,
                'flags': flags,
                'attack_type': 'DDoS',
                'is_attack': 1
            })
        
        return pd.DataFrame(data)
    
    def generate_port_scan_traffic(self, n_samples: int, start_time: datetime = None) -> pd.DataFrame:
        """Generate port scanning attack traffic patterns"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        
        data = []
        attacker_ip = self._generate_ip_address()
        target_ip = self._generate_ip_address()
        
        # Generate sequential port scans
        ports_to_scan = list(range(1, 1024))  # Common ports
        random.shuffle(ports_to_scan)
        
        for i, port in enumerate(ports_to_scan[:n_samples]):
            timestamp = start_time + timedelta(seconds=i * random.uniform(0.1, 1.0))
            
            src_ip = attacker_ip
            dst_ip = target_ip
            src_port = random.randint(1024, 65535)
            dst_port = port
            
            # Low packet count (connection attempts)
            packets = random.randint(1, 3)
            bytes_transferred = random.randint(40, 100)
            duration = random.uniform(0.1, 5.0)
            
            protocol = 'TCP'
            flags = self._generate_attack_flags('port_scan', protocol)
            
            data.append({
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'packets': packets,
                'bytes': bytes_transferred,
                'duration': duration,
                'flags': flags,
                'attack_type': 'PortScan',
                'is_attack': 1
            })
        
        return pd.DataFrame(data)
    
    def generate_brute_force_traffic(self, n_samples: int, start_time: datetime = None) -> pd.DataFrame:
        """Generate brute force attack traffic patterns"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        
        data = []
        attacker_ip = self._generate_ip_address()
        target_ip = self._generate_ip_address()
        
        # Common brute force targets
        target_ports = [22, 23, 21, 3389, 1433, 3306]
        
        for i in range(n_samples):
            timestamp = start_time + timedelta(seconds=i * random.uniform(1.0, 10.0))
            
            src_ip = attacker_ip
            dst_ip = target_ip
            src_port = random.randint(1024, 65535)
            dst_port = random.choice(target_ports)
            
            # Authentication attempt characteristics
            packets = random.randint(1, 5)
            bytes_transferred = random.randint(100, 500)
            duration = random.uniform(1.0, 10.0)
            
            protocol = 'TCP'
            flags = self._generate_attack_flags('brute_force', protocol)
            
            data.append({
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'packets': packets,
                'bytes': bytes_transferred,
                'duration': duration,
                'flags': flags,
                'attack_type': 'BruteForce',
                'is_attack': 1
            })
        
        return pd.DataFrame(data)
    
    def generate_mixed_traffic(self, n_samples: int, attack_ratio: float = 0.1, start_time: datetime = None) -> pd.DataFrame:
        """Generate mixed normal and attack traffic"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        
        n_attacks = int(n_samples * attack_ratio)
        n_normal = n_samples - n_attacks
        
        # Generate normal traffic
        normal_data = self.generate_normal_traffic(n_normal, start_time)
        
        # Generate attack traffic
        attack_data = []
        attack_types = ['ddos', 'port_scan', 'brute_force']
        
        for attack_type in attack_types:
            n_attack_samples = n_attacks // len(attack_types)
            if attack_type == 'ddos':
                attack_data.append(self.generate_ddos_traffic(n_attack_samples, start_time))
            elif attack_type == 'port_scan':
                attack_data.append(self.generate_port_scan_traffic(n_attack_samples, start_time))
            elif attack_type == 'brute_force':
                attack_data.append(self.generate_brute_force_traffic(n_attack_samples, start_time))
        
        # Combine all data
        all_data = [normal_data] + attack_data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Shuffle the data
        combined_data = combined_data.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return combined_data
    
    def _generate_ip_address(self) -> str:
        """Generate realistic IP addresses"""
        # Mix of private and public IPs
        if random.random() < 0.7:  # 70% private IPs
            if random.random() < 0.5:
                return f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
            else:
                return f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}"
        else:  # 30% public IPs
            return self.fake.ipv4()
    
    def _generate_normal_flags(self, protocol: str) -> str:
        """Generate normal TCP/UDP flags"""
        if protocol == 'TCP':
            return random.choice(['SYN', 'ACK', 'FIN', 'SYN-ACK', 'FIN-ACK'])
        elif protocol == 'UDP':
            return 'UDP'
        else:
            return 'ICMP'
    
    def _generate_attack_flags(self, attack_type: str, protocol: str) -> str:
        """Generate attack-specific flags"""
        if protocol == 'TCP':
            if attack_type == 'ddos':
                return random.choice(['SYN', 'ACK', 'RST'])
            elif attack_type == 'port_scan':
                return random.choice(['SYN', 'RST', 'FIN'])
            elif attack_type == 'brute_force':
                return random.choice(['SYN', 'ACK', 'RST'])
        elif protocol == 'UDP':
            return 'UDP'
        else:
            return 'ICMP'
    
    def add_noise(self, data: pd.DataFrame, noise_level: float = 0.05) -> pd.DataFrame:
        """Add noise to the dataset to make it more realistic"""
        noisy_data = data.copy()
        
        # Add random variations to numerical features
        numerical_cols = ['packets', 'bytes', 'duration']
        for col in numerical_cols:
            if col in noisy_data.columns:
                noise = np.random.normal(0, noise_level, len(noisy_data))
                noisy_data[col] = noisy_data[col] * (1 + noise)
                noisy_data[col] = noisy_data[col].clip(lower=0)  # Ensure non-negative
        
        # Occasionally change attack labels (simulate mislabeling)
        if 'is_attack' in noisy_data.columns:
            n_changes = int(len(noisy_data) * noise_level * 0.1)
            change_indices = np.random.choice(len(noisy_data), n_changes, replace=False)
            noisy_data.loc[change_indices, 'is_attack'] = 1 - noisy_data.loc[change_indices, 'is_attack']
        
        return noisy_data
    
    def generate_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate time series features for the data"""
        enhanced_data = data.copy()
        
        # Sort by timestamp
        enhanced_data = enhanced_data.sort_values('timestamp').reset_index(drop=True)
        
        # Time-based features
        enhanced_data['hour'] = enhanced_data['timestamp'].dt.hour
        enhanced_data['day_of_week'] = enhanced_data['timestamp'].dt.dayofweek
        enhanced_data['is_weekend'] = (enhanced_data['day_of_week'] >= 5).astype(int)
        
        # Rolling statistics
        window_size = 100
        enhanced_data['packets_rolling_mean'] = enhanced_data['packets'].rolling(window=window_size, min_periods=1).mean()
        enhanced_data['bytes_rolling_mean'] = enhanced_data['bytes'].rolling(window=window_size, min_periods=1).mean()
        enhanced_data['duration_rolling_mean'] = enhanced_data['duration'].rolling(window=window_size, min_periods=1).mean()
        
        # Traffic rate features
        enhanced_data['packets_per_second'] = enhanced_data['packets'] / enhanced_data['duration'].clip(lower=0.001)
        enhanced_data['bytes_per_second'] = enhanced_data['bytes'] / enhanced_data['duration'].clip(lower=0.001)
        
        return enhanced_data
