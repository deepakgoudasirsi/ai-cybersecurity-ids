"""
Data preprocessing pipeline for intrusion detection
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import logging
from typing import Tuple, Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Main data preprocessing pipeline"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
        self.label_encoder = LabelEncoder()
        self.sampler = None
        self.is_fitted = False
        
    def preprocess_dataset(self, data: pd.DataFrame, target_column: str, 
                          test_size: float = 0.2, validation_size: float = 0.1,
                          balance_method: str = 'smote', 
                          feature_selection_method: str = 'mutual_info',
                          n_features: int = None) -> Dict[str, Any]:
        """Complete preprocessing pipeline"""
        logger.info("Starting data preprocessing pipeline...")
        
        # Step 1: Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        # Step 2: Feature engineering
        X_engineered, feature_info = self.feature_engineer.fit_transform(
            X, y, feature_selection_method, n_features
        )
        
        # Step 3: Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Step 4: Handle class imbalance
        X_balanced, y_balanced = self._balance_classes(X_engineered, y_encoded, balance_method)
        
        # Step 5: Split data
        train_test_data = self._split_data(X_balanced, y_balanced, test_size, validation_size)
        
        # Step 6: Create preprocessing info
        preprocessing_info = {
            'original_shape': data.shape,
            'final_shape': X_balanced.shape,
            'target_classes': list(self.label_encoder.classes_),
            'class_distribution_original': dict(zip(self.label_encoder.classes_, 
                                                  np.bincount(y_encoded))),
            'class_distribution_balanced': dict(zip(self.label_encoder.classes_, 
                                                  np.bincount(y_balanced))),
            'feature_info': feature_info,
            'balance_method': balance_method,
            'sampler_used': self.sampler.__class__.__name__ if self.sampler else None
        }
        
        self.is_fitted = True
        
        logger.info("Data preprocessing completed successfully")
        logger.info(f"Final dataset shape: {X_balanced.shape}")
        logger.info(f"Class distribution after balancing: {preprocessing_info['class_distribution_balanced']}")
        
        return {**train_test_data, 'preprocessing_info': preprocessing_info}
    
    def preprocess_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess new data using fitted transformers"""
        if not self.is_fitted:
            raise ValueError("DataPreprocessor must be fitted before preprocessing new data")
        
        # Apply feature engineering
        X_processed = self.feature_engineer.transform(data)
        
        return X_processed
    
    def _balance_classes(self, X: pd.DataFrame, y: np.ndarray, method: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Handle class imbalance"""
        unique_classes, counts = np.unique(y, return_counts=True)
        logger.info(f"Class distribution before balancing: {dict(zip(unique_classes, counts))}")
        
        # Check if balancing is needed
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio < 2:  # Less than 2:1 ratio, no need to balance
            logger.info("Class imbalance is minimal, skipping balancing")
            return X, y
        
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}, applying {method} balancing")
        
        if method == 'smote':
            self.sampler = SMOTE(random_state=self.random_state, k_neighbors=3)
        elif method == 'adasyn':
            self.sampler = ADASYN(random_state=self.random_state)
        elif method == 'smoteenn':
            self.sampler = SMOTEENN(random_state=self.random_state)
        elif method == 'smotetomek':
            self.sampler = SMOTETomek(random_state=self.random_state)
        elif method == 'undersample':
            self.sampler = RandomUnderSampler(random_state=self.random_state)
        else:
            logger.warning(f"Unknown balancing method: {method}, using SMOTE")
            self.sampler = SMOTE(random_state=self.random_state)
        
        try:
            X_balanced, y_balanced = self.sampler.fit_resample(X, y)
            logger.info(f"Balancing completed. New shape: {X_balanced.shape}")
            return pd.DataFrame(X_balanced, columns=X.columns), y_balanced
        except Exception as e:
            logger.error(f"Error during class balancing: {str(e)}")
            logger.info("Returning original data without balancing")
            return X, y
    
    def _split_data(self, X: pd.DataFrame, y: np.ndarray, test_size: float, validation_size: float) -> Dict[str, Any]:
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {X_train.shape[0]} samples")
        logger.info(f"  Validation: {X_val.shape[0]} samples")
        logger.info(f"  Test: {X_test.shape[0]} samples")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def create_sequences(self, X: pd.DataFrame, y: np.ndarray, sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models (LSTM, etc.)"""
        sequences = []
        targets = []
        
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X.iloc[i:i + sequence_length].values)
            targets.append(y[i + sequence_length - 1])
        
        return np.array(sequences), np.array(targets)
    
    def create_anomaly_dataset(self, X: pd.DataFrame, y: np.ndarray, 
                              anomaly_ratio: float = 0.1) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create dataset for anomaly detection models"""
        # Separate normal and attack samples
        normal_mask = y == 0
        attack_mask = y == 1
        
        X_normal = X[normal_mask]
        y_normal = y[normal_mask]
        X_attack = X[attack_mask]
        y_attack = y[attack_mask]
        
        # Use all normal samples for training
        X_anomaly_train = X_normal.copy()
        y_anomaly_train = np.zeros(len(X_normal))  # All normal (0)
        
        # Create test set with normal + some attacks
        n_anomaly_test = int(len(X_normal) * anomaly_ratio)
        X_anomaly_test = pd.concat([
            X_normal.sample(n=len(X_normal), random_state=self.random_state),
            X_attack.sample(n=min(n_anomaly_test, len(X_attack)), random_state=self.random_state)
        ], ignore_index=True)
        
        y_anomaly_test = np.concatenate([
            np.zeros(len(X_normal)),  # Normal samples
            np.ones(min(n_anomaly_test, len(X_attack)))  # Attack samples
        ])
        
        # Shuffle test data
        shuffle_idx = np.random.permutation(len(X_anomaly_test))
        X_anomaly_test = X_anomaly_test.iloc[shuffle_idx].reset_index(drop=True)
        y_anomaly_test = y_anomaly_test[shuffle_idx]
        
        logger.info(f"Anomaly dataset created:")
        logger.info(f"  Train (normal only): {len(X_anomaly_train)} samples")
        logger.info(f"  Test: {len(X_anomaly_test)} samples ({np.sum(y_anomaly_test)} anomalies)")
        
        return (X_anomaly_train, y_anomaly_train), (X_anomaly_test, y_anomaly_test)
    
    def get_data_summary(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'target_distribution': data[target_column].value_counts().to_dict(),
            'numerical_summary': data.select_dtypes(include=[np.number]).describe().to_dict(),
            'categorical_summary': {}
        }
        
        # Categorical columns summary
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != target_column:
                summary['categorical_summary'][col] = {
                    'unique_values': data[col].nunique(),
                    'most_frequent': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'value_counts': data[col].value_counts().head(10).to_dict()
                }
        
        return summary
    
    def detect_outliers(self, X: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Detect outliers in the dataset"""
        X_clean = X.copy()
        outlier_info = {}
        
        numerical_cols = X_clean.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = X_clean[col].quantile(0.25)
                Q3 = X_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (X_clean[col] < lower_bound) | (X_clean[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((X_clean[col] - X_clean[col].mean()) / X_clean[col].std())
                outliers = z_scores > threshold
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            outlier_count = outliers.sum()
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(X_clean)) * 100,
                'indices': X_clean[outliers].index.tolist()
            }
        
        logger.info(f"Outlier detection completed using {method} method")
        for col, info in outlier_info.items():
            if info['count'] > 0:
                logger.info(f"  {col}: {info['count']} outliers ({info['percentage']:.2f}%)")
        
        return outlier_info
    
    def remove_outliers(self, X: pd.DataFrame, outlier_info: Dict[str, Any], 
                       max_outlier_percentage: float = 5.0) -> pd.DataFrame:
        """Remove outliers based on detection results"""
        X_clean = X.copy()
        removed_indices = set()
        
        for col, info in outlier_info.items():
            if info['percentage'] <= max_outlier_percentage:
                removed_indices.update(info['indices'])
                logger.info(f"Removing {info['count']} outliers from {col}")
            else:
                logger.warning(f"Skipping {col} - too many outliers ({info['percentage']:.2f}%)")
        
        X_clean = X_clean.drop(index=list(removed_indices))
        logger.info(f"Removed {len(removed_indices)} outlier samples. New shape: {X_clean.shape}")
        
        return X_clean
