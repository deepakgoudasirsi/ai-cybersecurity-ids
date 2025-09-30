"""
Advanced anomaly detection for zero-day attacks
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Advanced anomaly detection for identifying zero-day attacks"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.detectors = {}
        self.scalers = {}
        self.is_fitted = False
        self.anomaly_history = deque(maxlen=10000)
        
        # Initialize detectors
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Initialize various anomaly detection algorithms"""
        self.detectors = {
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            ),
            'local_outlier_factor': LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.1
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=0.1,
                random_state=42
            ),
            'dbscan': DBSCAN(
                eps=0.5,
                min_samples=5
            ),
            'kmeans': KMeans(
                n_clusters=5,
                random_state=42
            )
        }
        
        logger.info(f"Initialized {len(self.detectors)} anomaly detectors")
    
    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        """Fit anomaly detectors on normal data"""
        logger.info("Fitting anomaly detectors...")
        
        # Use only normal samples for training
        if y is not None:
            normal_mask = y == 0
            X_normal = X[normal_mask]
        else:
            X_normal = X
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_normal)
        self.scalers['main'] = scaler
        
        # Fit each detector
        for name, detector in self.detectors.items():
            try:
                if name == 'local_outlier_factor':
                    # LOF doesn't have a fit method, it's used differently
                    continue
                elif name == 'dbscan':
                    # DBSCAN is unsupervised, fit on all data
                    detector.fit(X_scaled)
                elif name == 'kmeans':
                    # K-means for clustering
                    detector.fit(X_scaled)
                else:
                    # Other detectors
                    detector.fit(X_scaled)
                
                logger.info(f"Fitted {name} detector")
                
            except Exception as e:
                logger.error(f"Error fitting {name} detector: {str(e)}")
        
        self.is_fitted = True
        logger.info("Anomaly detectors fitted successfully")
    
    def detect_anomalies(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using ensemble of detectors"""
        if not self.is_fitted:
            raise ValueError("Anomaly detectors must be fitted before detection")
        
        # Scale the data
        X_scaled = self.scalers['main'].transform(X)
        
        results = {}
        anomaly_scores = []
        
        for name, detector in self.detectors.items():
            try:
                if name == 'isolation_forest':
                    # Isolation Forest
                    predictions = detector.predict(X_scaled)
                    scores = detector.score_samples(X_scaled)
                    anomaly_scores.append(scores)
                    
                elif name == 'local_outlier_factor':
                    # Local Outlier Factor
                    predictions = detector.fit_predict(X_scaled)
                    scores = detector.negative_outlier_factor_
                    anomaly_scores.append(-scores)  # Convert to positive scores
                    
                elif name == 'elliptic_envelope':
                    # Elliptic Envelope
                    predictions = detector.predict(X_scaled)
                    scores = detector.score_samples(X_scaled)
                    anomaly_scores.append(scores)
                    
                elif name == 'dbscan':
                    # DBSCAN
                    predictions = detector.fit_predict(X_scaled)
                    # Convert DBSCAN labels to anomaly scores
                    scores = np.where(predictions == -1, 1.0, 0.0)
                    anomaly_scores.append(scores)
                    
                elif name == 'kmeans':
                    # K-means distance to centroids
                    distances = detector.transform(X_scaled)
                    min_distances = np.min(distances, axis=1)
                    scores = min_distances / np.max(min_distances)  # Normalize
                    predictions = np.where(scores > 0.5, -1, 1)  # Threshold-based
                    anomaly_scores.append(scores)
                
                results[name] = {
                    'predictions': predictions,
                    'scores': scores,
                    'anomaly_count': np.sum(predictions == -1)
                }
                
            except Exception as e:
                logger.error(f"Error in {name} detection: {str(e)}")
                results[name] = {'error': str(e)}
        
        # Ensemble decision
        if anomaly_scores:
            # Combine scores using voting
            ensemble_scores = np.mean(anomaly_scores, axis=0)
            ensemble_predictions = self._ensemble_anomaly_decision(ensemble_scores)
            
            results['ensemble'] = {
                'predictions': ensemble_predictions,
                'scores': ensemble_scores,
                'anomaly_count': np.sum(ensemble_predictions == -1)
            }
        
        return results
    
    def _ensemble_anomaly_decision(self, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make ensemble decision for anomaly detection"""
        # Use percentile-based threshold
        threshold_value = np.percentile(scores, (1 - threshold) * 100)
        predictions = np.where(scores > threshold_value, -1, 1)
        return predictions
    
    def detect_zero_day_attacks(self, X: pd.DataFrame, 
                               historical_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Detect potential zero-day attacks using advanced techniques"""
        logger.info("Detecting potential zero-day attacks...")
        
        # Basic anomaly detection
        anomaly_results = self.detect_anomalies(X)
        
        # Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(X)
        
        # Pattern-based detection
        pattern_anomalies = self._detect_pattern_anomalies(X, historical_data)
        
        # Behavioral anomaly detection
        behavioral_anomalies = self._detect_behavioral_anomalies(X)
        
        # Combine all results
        zero_day_results = {
            'ensemble_anomalies': anomaly_results.get('ensemble', {}),
            'statistical_anomalies': statistical_anomalies,
            'pattern_anomalies': pattern_anomalies,
            'behavioral_anomalies': behavioral_anomalies,
            'zero_day_score': self._calculate_zero_day_score(
                anomaly_results, statistical_anomalies, 
                pattern_anomalies, behavioral_anomalies
            )
        }
        
        return zero_day_results
    
    def _detect_statistical_anomalies(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Detect statistical anomalies using various statistical tests"""
        results = {}
        
        for column in X.select_dtypes(include=[np.number]).columns:
            data = X[column].values
            
            # Z-score based detection
            z_scores = np.abs(stats.zscore(data))
            z_anomalies = z_scores > 3
            
            # Modified Z-score (using median)
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            modified_z_anomalies = np.abs(modified_z_scores) > 3.5
            
            # IQR based detection
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_anomalies = (data < lower_bound) | (data > upper_bound)
            
            results[column] = {
                'z_score_anomalies': z_anomalies,
                'modified_z_anomalies': modified_z_anomalies,
                'iqr_anomalies': iqr_anomalies,
                'anomaly_count': np.sum(z_anomalies | modified_z_anomalies | iqr_anomalies)
            }
        
        return results
    
    def _detect_pattern_anomalies(self, X: pd.DataFrame, 
                                 historical_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Detect pattern-based anomalies"""
        results = {}
        
        if historical_data is not None:
            # Compare with historical patterns
            for column in X.select_dtypes(include=[np.number]).columns:
                if column in historical_data.columns:
                    current_data = X[column].values
                    historical_data_col = historical_data[column].values
                    
                    # Compare distributions using Kolmogorov-Smirnov test
                    ks_statistic, p_value = stats.ks_2samp(historical_data_col, current_data)
                    
                    # Compare means
                    current_mean = np.mean(current_data)
                    historical_mean = np.mean(historical_data_col)
                    mean_diff = abs(current_mean - historical_mean)
                    
                    # Compare variances
                    current_var = np.var(current_data)
                    historical_var = np.var(historical_data_col)
                    var_ratio = current_var / historical_var if historical_var > 0 else 1
                    
                    results[column] = {
                        'ks_statistic': ks_statistic,
                        'p_value': p_value,
                        'mean_difference': mean_diff,
                        'variance_ratio': var_ratio,
                        'is_anomalous': p_value < 0.05 or var_ratio > 2 or var_ratio < 0.5
                    }
        
        return results
    
    def _detect_behavioral_anomalies(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Detect behavioral anomalies"""
        results = {}
        
        # Detect unusual traffic patterns
        if 'packets' in X.columns and 'bytes' in X.columns:
            # Unusual packet-to-byte ratios
            packet_byte_ratio = X['bytes'] / (X['packets'] + 1)
            ratio_anomalies = self._detect_outliers(packet_byte_ratio)
            
            results['packet_byte_ratio'] = {
                'anomalies': ratio_anomalies,
                'anomaly_count': np.sum(ratio_anomalies)
            }
        
        # Detect unusual port usage
        if 'dstport' in X.columns:
            port_anomalies = self._detect_port_anomalies(X['dstport'])
            results['port_anomalies'] = port_anomalies
        
        # Detect unusual protocol usage
        if 'protocol' in X.columns:
            protocol_anomalies = self._detect_protocol_anomalies(X['protocol'])
            results['protocol_anomalies'] = protocol_anomalies
        
        return results
    
    def _detect_outliers(self, data: pd.Series, method: str = 'iqr') -> np.ndarray:
        """Detect outliers in a data series"""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            return z_scores > 3
        else:
            return np.zeros(len(data), dtype=bool)
    
    def _detect_port_anomalies(self, ports: pd.Series) -> Dict[str, Any]:
        """Detect unusual port usage patterns"""
        # Common ports
        common_ports = {80, 443, 22, 21, 25, 53, 110, 143, 993, 995, 3389, 1433, 3306}
        
        # Unusual ports (not in common list)
        unusual_ports = ~ports.isin(common_ports)
        
        # High port numbers (potential dynamic ports)
        high_ports = ports > 49152
        
        # Very low ports (potential system ports)
        low_ports = ports < 1024
        
        return {
            'unusual_ports': unusual_ports,
            'high_ports': high_ports,
            'low_ports': low_ports,
            'unusual_count': np.sum(unusual_ports),
            'high_count': np.sum(high_ports),
            'low_count': np.sum(low_ports)
        }
    
    def _detect_protocol_anomalies(self, protocols: pd.Series) -> Dict[str, Any]:
        """Detect unusual protocol usage"""
        protocol_counts = protocols.value_counts()
        total_count = len(protocols)
        
        # Protocols with very low frequency
        rare_protocols = protocol_counts[protocol_counts / total_count < 0.01]
        
        # Unusual protocol combinations (if multiple protocol columns exist)
        unusual_combinations = []
        
        return {
            'protocol_counts': protocol_counts.to_dict(),
            'rare_protocols': rare_protocols.to_dict(),
            'unusual_combinations': unusual_combinations
        }
    
    def _calculate_zero_day_score(self, ensemble_results: Dict[str, Any],
                                 statistical_anomalies: Dict[str, Any],
                                 pattern_anomalies: Dict[str, Any],
                                 behavioral_anomalies: Dict[str, Any]) -> float:
        """Calculate zero-day attack probability score"""
        score = 0.0
        
        # Ensemble anomaly score (40% weight)
        if 'ensemble' in ensemble_results and 'scores' in ensemble_results['ensemble']:
            ensemble_scores = ensemble_results['ensemble']['scores']
            score += 0.4 * np.mean(ensemble_scores)
        
        # Statistical anomaly score (25% weight)
        stat_score = 0.0
        for column, results in statistical_anomalies.items():
            if 'anomaly_count' in results:
                stat_score += results['anomaly_count'] / len(statistical_anomalies)
        score += 0.25 * stat_score
        
        # Pattern anomaly score (20% weight)
        pattern_score = 0.0
        for column, results in pattern_anomalies.items():
            if 'is_anomalous' in results and results['is_anomalous']:
                pattern_score += 1.0
        if pattern_anomalies:
            pattern_score /= len(pattern_anomalies)
        score += 0.20 * pattern_score
        
        # Behavioral anomaly score (15% weight)
        behavior_score = 0.0
        for category, results in behavioral_anomalies.items():
            if 'anomaly_count' in results:
                behavior_score += results['anomaly_count']
        if behavioral_anomalies:
            behavior_score /= len(behavioral_anomalies)
        score += 0.15 * behavior_score
        
        return min(score, 1.0)  # Cap at 1.0
    
    def update_models(self, X: pd.DataFrame, y: np.ndarray = None):
        """Update anomaly detection models with new data"""
        logger.info("Updating anomaly detection models...")
        
        # Use only normal samples for updating
        if y is not None:
            normal_mask = y == 0
            X_normal = X[normal_mask]
        else:
            X_normal = X
        
        # Update scaler
        X_scaled = self.scalers['main'].transform(X_normal)
        
        # Update each detector (if supported)
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'partial_fit'):
                    detector.partial_fit(X_scaled)
                    logger.info(f"Updated {name} detector")
            except Exception as e:
                logger.error(f"Error updating {name} detector: {str(e)}")
        
        logger.info("Model update completed")
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        if not self.anomaly_history:
            return {'total_anomalies': 0}
        
        recent_anomalies = list(self.anomaly_history)[-1000:]  # Last 1000 detections
        
        stats = {
            'total_anomalies': len(self.anomaly_history),
            'recent_anomalies': len(recent_anomalies),
            'detectors_active': len([d for d in self.detectors.values() if hasattr(d, 'predict')]),
            'is_fitted': self.is_fitted
        }
        
        return stats
    
    def export_anomaly_log(self, filepath: str):
        """Export anomaly detection log"""
        import json
        
        export_data = []
        for anomaly in self.anomaly_history:
            export_anomaly = anomaly.copy()
            if 'timestamp' in export_anomaly:
                export_anomaly['timestamp'] = export_anomaly['timestamp'].isoformat()
            export_data.append(export_anomaly)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} anomaly records to {filepath}")
