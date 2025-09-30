"""
Dimensionality reduction techniques for intrusion detection
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Tuple, Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """Dimensionality reduction for cybersecurity datasets"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.reducers = {}
        self.is_fitted = False
        
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray = None, 
                     method: str = 'pca', n_components: int = None,
                     **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit and transform data using specified dimensionality reduction method"""
        logger.info(f"Applying {method} dimensionality reduction...")
        
        if n_components is None:
            n_components = min(50, X.shape[1] // 2)
        
        if method == 'pca':
            reducer, X_reduced, info = self._apply_pca(X, n_components, **kwargs)
        elif method == 'ica':
            reducer, X_reduced, info = self._apply_ica(X, n_components, **kwargs)
        elif method == 'svd':
            reducer, X_reduced, info = self._apply_svd(X, n_components, **kwargs)
        elif method == 'lda':
            if y is None:
                raise ValueError("LDA requires target labels")
            reducer, X_reduced, info = self._apply_lda(X, y, n_components, **kwargs)
        elif method == 'tsne':
            reducer, X_reduced, info = self._apply_tsne(X, n_components, **kwargs)
        elif method == 'isomap':
            reducer, X_reduced, info = self._apply_isomap(X, n_components, **kwargs)
        elif method == 'lle':
            reducer, X_reduced, info = self._apply_lle(X, n_components, **kwargs)
        elif method == 'feature_selection':
            if y is None:
                raise ValueError("Feature selection requires target labels")
            reducer, X_reduced, info = self._apply_feature_selection(X, y, n_components, **kwargs)
        elif method == 'autoencoder':
            reducer, X_reduced, info = self._apply_autoencoder(X, n_components, **kwargs)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        self.reducers[method] = reducer
        self.is_fitted = True
        
        # Create column names
        if method in ['pca', 'ica', 'svd', 'lda', 'autoencoder']:
            column_names = [f"{method.upper()}_{i+1}" for i in range(X_reduced.shape[1])]
        elif method == 'feature_selection':
            column_names = info['selected_features']
        else:
            column_names = [f"{method.upper()}_{i+1}" for i in range(X_reduced.shape[1])]
        
        X_reduced_df = pd.DataFrame(X_reduced, columns=column_names, index=X.index)
        
        logger.info(f"Dimensionality reduction completed: {X.shape[1]} -> {X_reduced.shape[1]} features")
        
        return X_reduced_df, info
    
    def transform(self, X: pd.DataFrame, method: str = 'pca') -> pd.DataFrame:
        """Transform new data using fitted reducer"""
        if not self.is_fitted or method not in self.reducers:
            raise ValueError(f"Reducer for method '{method}' not fitted")
        
        reducer = self.reducers[method]
        
        if method == 'feature_selection':
            # For feature selection, just select the same features
            selected_features = reducer.get_support()
            X_transformed = X.iloc[:, selected_features]
            column_names = X.columns[selected_features].tolist()
        else:
            X_transformed = reducer.transform(X)
            column_names = [f"{method.upper()}_{i+1}" for i in range(X_transformed.shape[1])]
        
        return pd.DataFrame(X_transformed, columns=column_names, index=X.index)
    
    def _apply_pca(self, X: pd.DataFrame, n_components: int, **kwargs) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Apply Principal Component Analysis"""
        pca = PCA(n_components=n_components, random_state=self.random_state, **kwargs)
        X_reduced = pca.fit_transform(X)
        
        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        info = {
            'method': 'PCA',
            'n_components': n_components,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'total_variance_explained': cumulative_variance[-1],
            'components': pca.components_,
            'mean': pca.mean_
        }
        
        logger.info(f"PCA: {n_components} components explain {cumulative_variance[-1]:.3f} of variance")
        
        return pca, X_reduced, info
    
    def _apply_ica(self, X: pd.DataFrame, n_components: int, **kwargs) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Apply Independent Component Analysis"""
        ica = FastICA(n_components=n_components, random_state=self.random_state, **kwargs)
        X_reduced = ica.fit_transform(X)
        
        info = {
            'method': 'ICA',
            'n_components': n_components,
            'components': ica.components_,
            'mixing': ica.mixing_
        }
        
        logger.info(f"ICA: {n_components} independent components extracted")
        
        return ica, X_reduced, info
    
    def _apply_svd(self, X: pd.DataFrame, n_components: int, **kwargs) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Apply Singular Value Decomposition"""
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state, **kwargs)
        X_reduced = svd.fit_transform(X)
        
        info = {
            'method': 'SVD',
            'n_components': n_components,
            'explained_variance_ratio': svd.explained_variance_ratio_,
            'singular_values': svd.singular_values_,
            'components': svd.components_
        }
        
        logger.info(f"SVD: {n_components} components extracted")
        
        return svd, X_reduced, info
    
    def _apply_lda(self, X: pd.DataFrame, y: np.ndarray, n_components: int, **kwargs) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Apply Linear Discriminant Analysis"""
        # LDA can have at most min(n_features, n_classes-1) components
        max_components = min(X.shape[1], len(np.unique(y)) - 1)
        n_components = min(n_components, max_components)
        
        lda = LinearDiscriminantAnalysis(n_components=n_components, **kwargs)
        X_reduced = lda.fit_transform(X, y)
        
        info = {
            'method': 'LDA',
            'n_components': n_components,
            'explained_variance_ratio': lda.explained_variance_ratio_,
            'scalings': lda.scalings_,
            'means': lda.means_
        }
        
        logger.info(f"LDA: {n_components} components extracted")
        
        return lda, X_reduced, info
    
    def _apply_tsne(self, X: pd.DataFrame, n_components: int, **kwargs) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Apply t-SNE"""
        # t-SNE is typically used for 2D or 3D visualization
        n_components = min(n_components, 3)
        
        tsne = TSNE(n_components=n_components, random_state=self.random_state, **kwargs)
        X_reduced = tsne.fit_transform(X)
        
        info = {
            'method': 't-SNE',
            'n_components': n_components,
            'kl_divergence': tsne.kl_divergence_
        }
        
        logger.info(f"t-SNE: {n_components} components extracted")
        
        return tsne, X_reduced, info
    
    def _apply_isomap(self, X: pd.DataFrame, n_components: int, **kwargs) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Apply Isomap"""
        isomap = Isomap(n_components=n_components, **kwargs)
        X_reduced = isomap.fit_transform(X)
        
        info = {
            'method': 'Isomap',
            'n_components': n_components,
            'n_neighbors': isomap.n_neighbors,
            'reconstruction_error': isomap.reconstruction_error()
        }
        
        logger.info(f"Isomap: {n_components} components extracted")
        
        return isomap, X_reduced, info
    
    def _apply_lle(self, X: pd.DataFrame, n_components: int, **kwargs) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Apply Locally Linear Embedding"""
        lle = LocallyLinearEmbedding(n_components=n_components, random_state=self.random_state, **kwargs)
        X_reduced = lle.fit_transform(X)
        
        info = {
            'method': 'LLE',
            'n_components': n_components,
            'n_neighbors': lle.n_neighbors,
            'reconstruction_error': lle.reconstruction_error_
        }
        
        logger.info(f"LLE: {n_components} components extracted")
        
        return lle, X_reduced, info
    
    def _apply_feature_selection(self, X: pd.DataFrame, y: np.ndarray, n_features: int, 
                                method: str = 'mutual_info', **kwargs) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Apply feature selection"""
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'random_forest':
            # Use Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, **kwargs)
            rf.fit(X, y)
            feature_importance = rf.feature_importances_
            top_features = np.argsort(feature_importance)[-n_features:]
            selector = type('Selector', (), {
                'get_support': lambda: np.isin(range(len(feature_importance)), top_features),
                'scores_': feature_importance
            })()
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        if method != 'random_forest':
            X_reduced = selector.fit_transform(X, y)
        else:
            X_reduced = X.iloc[:, top_features].values
        
        info = {
            'method': f'Feature Selection ({method})',
            'n_features': n_features,
            'selected_features': X.columns[selector.get_support()].tolist(),
            'feature_scores': selector.scores_ if hasattr(selector, 'scores_') else None
        }
        
        logger.info(f"Feature Selection: {n_features} features selected")
        
        return selector, X_reduced, info
    
    def _apply_autoencoder(self, X: pd.DataFrame, encoding_dim: int, **kwargs) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Apply simple autoencoder for dimensionality reduction"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense
            from tensorflow.keras.optimizers import Adam
            
            # Normalize data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Build autoencoder
            input_dim = X.shape[1]
            input_layer = Input(shape=(input_dim,))
            
            # Encoder
            encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
            encoded = Dense(encoding_dim, activation='relu')(encoded)
            
            # Decoder
            decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
            decoded = Dense(input_dim, activation='sigmoid')(decoded)
            
            # Autoencoder model
            autoencoder = Model(input_layer, decoded)
            encoder = Model(input_layer, encoded)
            
            # Compile and train
            autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train autoencoder
            epochs = kwargs.get('epochs', 50)
            batch_size = kwargs.get('batch_size', 32)
            
            autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, 
                           validation_split=0.1, verbose=0)
            
            # Get encoded representation
            X_reduced = encoder.predict(X_scaled)
            
            info = {
                'method': 'Autoencoder',
                'encoding_dim': encoding_dim,
                'input_dim': input_dim,
                'scaler': scaler,
                'encoder': encoder,
                'autoencoder': autoencoder
            }
            
            logger.info(f"Autoencoder: {input_dim} -> {encoding_dim} dimensions")
            
            return info, X_reduced, info
            
        except ImportError:
            logger.warning("TensorFlow not available, falling back to PCA")
            return self._apply_pca(X, encoding_dim, **kwargs)
    
    def compare_methods(self, X: pd.DataFrame, y: np.ndarray = None, 
                       methods: List[str] = None, n_components: int = 10) -> Dict[str, Any]:
        """Compare different dimensionality reduction methods"""
        if methods is None:
            methods = ['pca', 'ica', 'svd', 'feature_selection']
            if y is not None:
                methods.extend(['lda'])
        
        results = {}
        
        for method in methods:
            try:
                logger.info(f"Comparing {method}...")
                X_reduced, info = self.fit_transform(X, y, method, n_components)
                
                results[method] = {
                    'X_reduced': X_reduced,
                    'info': info,
                    'shape': X_reduced.shape
                }
                
                # Calculate reconstruction error for methods that support it
                if method in ['pca', 'svd']:
                    X_reconstructed = self.transform(X, method)
                    reconstruction_error = np.mean((X.values - X_reconstructed.values) ** 2)
                    results[method]['reconstruction_error'] = reconstruction_error
                
            except Exception as e:
                logger.error(f"Error with {method}: {str(e)}")
                results[method] = {'error': str(e)}
        
        return results
    
    def get_optimal_components(self, X: pd.DataFrame, method: str = 'pca', 
                              variance_threshold: float = 0.95) -> int:
        """Find optimal number of components for PCA/SVD"""
        if method == 'pca':
            pca = PCA()
            pca.fit(X)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        elif method == 'svd':
            svd = TruncatedSVD(n_components=min(X.shape))
            svd.fit(X)
            cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        else:
            raise ValueError(f"Method {method} not supported for optimal component selection")
        
        logger.info(f"Optimal {method.upper()} components for {variance_threshold*100}% variance: {n_components}")
        
        return n_components
