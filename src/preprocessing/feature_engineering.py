"""
Feature engineering for intrusion detection datasets
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import logging
from typing import Tuple, List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for cybersecurity datasets"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.feature_names = []
        self.is_fitted = False
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None, 
                     feature_selection_method: str = 'mutual_info',
                     n_features: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit and transform features"""
        logger.info("Starting feature engineering...")
        
        # Store original feature names
        self.feature_names = list(X.columns)
        
        # Step 1: Handle missing values
        X_processed = self._handle_missing_values(X)
        
        # Step 2: Encode categorical variables
        X_processed = self._encode_categorical_features(X_processed)
        
        # Step 3: Create new features
        X_processed = self._create_engineered_features(X_processed)
        
        # Step 4: Scale numerical features
        X_processed = self._scale_features(X_processed, fit=True)
        
        # Step 5: Feature selection
        if y is not None and feature_selection_method:
            X_processed = self._select_features(X_processed, y, feature_selection_method, n_features)
        
        # Step 6: Remove highly correlated features
        X_processed = self._remove_correlated_features(X_processed)
        
        self.is_fitted = True
        
        # Create feature info
        feature_info = {
            'original_features': len(self.feature_names),
            'engineered_features': len(X_processed.columns),
            'feature_names': list(X_processed.columns),
            'scalers': list(self.scalers.keys()),
            'encoders': list(self.encoders.keys())
        }
        
        logger.info(f"Feature engineering completed. {feature_info['original_features']} -> {feature_info['engineered_features']} features")
        
        return X_processed, feature_info
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        # Apply the same transformations in the same order
        X_processed = self._handle_missing_values(X)
        X_processed = self._encode_categorical_features(X_processed, fit=False)
        X_processed = self._create_engineered_features(X_processed)
        X_processed = self._scale_features(X_processed, fit=False)
        
        # Select the same features as during training
        if 'feature_selector' in self.feature_selectors:
            X_processed = self._apply_feature_selection(X_processed, fit=False)
        
        # Remove correlated features (use same correlation matrix)
        if 'correlation_threshold' in self.feature_selectors:
            X_processed = self._apply_correlation_removal(X_processed, fit=False)
        
        return X_processed
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        X_processed = X.copy()
        
        # For numerical columns, fill with median
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X_processed[col].isnull().any():
                median_val = X_processed[col].median()
                X_processed[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        # For categorical columns, fill with mode
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X_processed[col].isnull().any():
                mode_val = X_processed[col].mode()[0] if not X_processed[col].mode().empty else 'Unknown'
                X_processed[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        return X_processed
    
    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        X_processed = X.copy()
        
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if fit:
                # Fit label encoder
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.encoders[f'{col}_label'] = le
                
                # For high cardinality categorical features, use one-hot encoding
                if len(X_processed[col].unique()) <= 20:  # Threshold for one-hot encoding
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    ohe_matrix = ohe.fit_transform(X_processed[[col]])
                    
                    # Create column names
                    ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                    
                    # Add one-hot encoded columns
                    ohe_df = pd.DataFrame(ohe_matrix, columns=ohe_cols, index=X_processed.index)
                    X_processed = pd.concat([X_processed, ohe_df], axis=1)
                    
                    # Remove original column
                    X_processed.drop(col, axis=1, inplace=True)
                    
                    self.encoders[f'{col}_onehot'] = ohe
                    logger.info(f"One-hot encoded {col} into {len(ohe_cols)} features")
                else:
                    logger.info(f"Label encoded {col} (high cardinality: {len(X_processed[col].unique())} categories)")
            else:
                # Transform using fitted encoders
                if f'{col}_label' in self.encoders:
                    le = self.encoders[f'{col}_label']
                    # Handle unseen categories
                    unique_values = set(X_processed[col].astype(str).unique())
                    known_values = set(le.classes_)
                    unseen_values = unique_values - known_values
                    
                    if unseen_values:
                        logger.warning(f"Found unseen categories in {col}: {unseen_values}")
                        # Replace unseen categories with the most frequent known category
                        most_frequent = le.classes_[0]
                        X_processed[col] = X_processed[col].astype(str).replace(list(unseen_values), most_frequent)
                    
                    X_processed[col] = le.transform(X_processed[col].astype(str))
                
                if f'{col}_onehot' in self.encoders:
                    ohe = self.encoders[f'{col}_onehot']
                    ohe_matrix = ohe.transform(X_processed[[col]])
                    ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                    ohe_df = pd.DataFrame(ohe_matrix, columns=ohe_cols, index=X_processed.index)
                    X_processed = pd.concat([X_processed, ohe_df], axis=1)
                    X_processed.drop(col, axis=1, inplace=True)
        
        return X_processed
    
    def _create_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create new engineered features"""
        X_processed = X.copy()
        
        # Network traffic features
        if 'srcport' in X_processed.columns and 'dstport' in X_processed.columns:
            X_processed['port_ratio'] = X_processed['srcport'] / (X_processed['dstport'] + 1)
            X_processed['port_sum'] = X_processed['srcport'] + X_processed['dstport']
            X_processed['port_diff'] = abs(X_processed['srcport'] - X_processed['dstport'])
        
        # Packet and byte features
        if 'packets' in X_processed.columns and 'bytes' in X_processed.columns:
            X_processed['avg_packet_size'] = X_processed['bytes'] / (X_processed['packets'] + 1)
            X_processed['packet_byte_ratio'] = X_processed['packets'] / (X_processed['bytes'] + 1)
        
        # Duration features
        if 'duration' in X_processed.columns:
            X_processed['duration_log'] = np.log1p(X_processed['duration'])
            X_processed['duration_sqrt'] = np.sqrt(X_processed['duration'])
        
        # Flow features
        if 'sbytes' in X_processed.columns and 'dbytes' in X_processed.columns:
            X_processed['total_bytes'] = X_processed['sbytes'] + X_processed['dbytes']
            X_processed['byte_ratio'] = X_processed['sbytes'] / (X_processed['dbytes'] + 1)
            X_processed['byte_diff'] = abs(X_processed['sbytes'] - X_processed['dbytes'])
        
        # Statistical features
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['srcport', 'dstport', 'packets', 'bytes', 'duration', 'sbytes', 'dbytes']:
                # Z-score features
                X_processed[f'{col}_zscore'] = (X_processed[col] - X_processed[col].mean()) / (X_processed[col].std() + 1e-8)
                
                # Quantile features
                X_processed[f'{col}_q25'] = (X_processed[col] <= X_processed[col].quantile(0.25)).astype(int)
                X_processed[f'{col}_q75'] = (X_processed[col] >= X_processed[col].quantile(0.75)).astype(int)
        
        # Interaction features
        if 'packets' in X_processed.columns and 'duration' in X_processed.columns:
            X_processed['packets_per_second'] = X_processed['packets'] / (X_processed['duration'] + 1)
        
        if 'bytes' in X_processed.columns and 'duration' in X_processed.columns:
            X_processed['bytes_per_second'] = X_processed['bytes'] / (X_processed['duration'] + 1)
        
        logger.info(f"Created {len(X_processed.columns) - len(X.columns)} new engineered features")
        
        return X_processed
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        X_processed = X.copy()
        
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
        
        if fit:
            # Use StandardScaler for most features
            scaler = StandardScaler()
            X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])
            self.scalers['standard'] = scaler
            
            # Use MinMaxScaler for features that should be bounded
            bounded_cols = [col for col in numerical_cols if 'ratio' in col.lower() or 'percent' in col.lower()]
            if bounded_cols:
                minmax_scaler = MinMaxScaler()
                X_processed[bounded_cols] = minmax_scaler.fit_transform(X_processed[bounded_cols])
                self.scalers['minmax'] = minmax_scaler
        else:
            # Apply fitted scalers
            if 'standard' in self.scalers:
                standard_cols = [col for col in numerical_cols if col not in self.scalers.get('minmax_columns', [])]
                X_processed[standard_cols] = self.scalers['standard'].transform(X_processed[standard_cols])
            
            if 'minmax' in self.scalers:
                minmax_cols = self.scalers.get('minmax_columns', [])
                if minmax_cols:
                    X_processed[minmax_cols] = self.scalers['minmax'].transform(X_processed[minmax_cols])
        
        return X_processed
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, method: str, n_features: int = None) -> pd.DataFrame:
        """Select the most important features"""
        if n_features is None:
            n_features = min(50, X.shape[1] // 2)  # Default to half the features or 50, whichever is smaller
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.feature_selectors['feature_selector'] = selector
        self.feature_selectors['selected_features'] = selected_features
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def _apply_feature_selection(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Apply feature selection to new data"""
        if 'feature_selector' in self.feature_selectors:
            selector = self.feature_selectors['feature_selector']
            selected_features = self.feature_selectors['selected_features']
            
            # Ensure all selected features exist in the new data
            missing_features = set(selected_features) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing features in new data: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    X[feature] = 0
            
            return X[selected_features]
        
        return X
    
    def _remove_correlated_features(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features"""
        X_processed = X.copy()
        
        # Calculate correlation matrix
        corr_matrix = X_processed.corr().abs()
        
        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            X_processed = X_processed.drop(columns=to_drop)
            self.feature_selectors['correlation_threshold'] = threshold
            self.feature_selectors['dropped_correlated'] = to_drop
            logger.info(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
        
        return X_processed
    
    def _apply_correlation_removal(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Apply correlation-based feature removal to new data"""
        if 'dropped_correlated' in self.feature_selectors:
            dropped_features = self.feature_selectors['dropped_correlated']
            return X.drop(columns=[col for col in dropped_features if col in X.columns])
        
        return X
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info') -> pd.Series:
        """Get feature importance scores"""
        if method == 'mutual_info':
            scores = mutual_info_classif(X, y)
        elif method == 'f_classif':
            scores, _ = f_classif(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        feature_importance = pd.Series(scores, index=X.columns).sort_values(ascending=False)
        return feature_importance
    
    def create_time_series_features(self, X: pd.DataFrame, time_column: str = 'timestamp') -> pd.DataFrame:
        """Create time series features"""
        X_processed = X.copy()
        
        if time_column in X_processed.columns:
            # Convert to datetime if not already
            X_processed[time_column] = pd.to_datetime(X_processed[time_column])
            
            # Time-based features
            X_processed['hour'] = X_processed[time_column].dt.hour
            X_processed['day_of_week'] = X_processed[time_column].dt.dayofweek
            X_processed['day_of_month'] = X_processed[time_column].dt.day
            X_processed['month'] = X_processed[time_column].dt.month
            X_processed['is_weekend'] = (X_processed['day_of_week'] >= 5).astype(int)
            X_processed['is_business_hours'] = ((X_processed['hour'] >= 9) & (X_processed['hour'] <= 17)).astype(int)
            
            # Cyclical encoding for time features
            X_processed['hour_sin'] = np.sin(2 * np.pi * X_processed['hour'] / 24)
            X_processed['hour_cos'] = np.cos(2 * np.pi * X_processed['hour'] / 24)
            X_processed['day_sin'] = np.sin(2 * np.pi * X_processed['day_of_week'] / 7)
            X_processed['day_cos'] = np.cos(2 * np.pi * X_processed['day_of_week'] / 7)
        
        return X_processed
