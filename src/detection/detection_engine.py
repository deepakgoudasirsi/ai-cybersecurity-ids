"""
Real-time detection engine for intrusion detection
"""
import numpy as np
import pandas as pd
import asyncio
import queue
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import logging
from collections import deque, defaultdict
import json

from ..config.config import REAL_TIME_CONFIG, DETECTION_THRESHOLDS
from ..models.ml_models import MLModelTrainer
from ..models.dl_models import DeepLearningTrainer
from ..preprocessing.data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class RealTimeDetector:
    """Real-time intrusion detection engine"""
    
    def __init__(self, models: Dict[str, Any] = None, config: Dict[str, Any] = None):
        self.config = config or REAL_TIME_CONFIG
        self.models = models or {}
        self.data_queue = queue.Queue(maxsize=self.config['max_queue_size'])
        self.detection_results = deque(maxlen=10000)  # Store last 10k results
        self.alert_callbacks = []
        self.is_running = False
        self.detection_thread = None
        self.preprocessor = DataPreprocessor()
        
        # Performance metrics
        self.metrics = {
            'total_processed': 0,
            'total_detected': 0,
            'processing_times': deque(maxlen=1000),
            'detection_rates': deque(maxlen=100),
            'false_positive_count': 0,
            'last_update': datetime.now()
        }
        
        # Detection history for pattern analysis
        self.detection_history = defaultdict(list)
        
    def add_model(self, name: str, model: Any, model_type: str = 'ml'):
        """Add a trained model to the detection engine"""
        self.models[name] = {
            'model': model,
            'type': model_type,
            'predictions': deque(maxlen=1000),
            'confidence_scores': deque(maxlen=1000)
        }
        logger.info(f"Added {model_type} model: {name}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
        logger.info("Added alert callback")
    
    def start_detection(self):
        """Start the real-time detection engine"""
        if self.is_running:
            logger.warning("Detection engine is already running")
            return
        
        if not self.models:
            logger.error("No models loaded. Cannot start detection engine.")
            return
        
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        logger.info("Real-time detection engine started")
    
    def stop_detection(self):
        """Stop the real-time detection engine"""
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        logger.info("Real-time detection engine stopped")
    
    def add_data(self, data: pd.DataFrame):
        """Add new data to the detection queue"""
        try:
            self.data_queue.put(data, timeout=1)
            return True
        except queue.Full:
            logger.warning("Detection queue is full, dropping data")
            return False
    
    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        batch_buffer = []
        last_process_time = time.time()
        
        while self.is_running:
            try:
                # Collect data for batch processing
                current_time = time.time()
                
                # Try to get data from queue
                try:
                    data = self.data_queue.get(timeout=0.1)
                    batch_buffer.append(data)
                except queue.Empty:
                    pass
                
                # Process batch if we have enough data or enough time has passed
                if (len(batch_buffer) >= self.config['batch_size'] or 
                    current_time - last_process_time >= self.config['processing_interval']):
                    
                    if batch_buffer:
                        self._process_batch(batch_buffer)
                        batch_buffer = []
                        last_process_time = current_time
                
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in detection loop: {str(e)}")
                time.sleep(1)
    
    def _process_batch(self, data_batch: List[pd.DataFrame]):
        """Process a batch of data"""
        if not data_batch:
            return
        
        # Combine all data in batch
        combined_data = pd.concat(data_batch, ignore_index=True)
        
        start_time = time.time()
        
        try:
            # Preprocess data
            processed_data = self.preprocessor.preprocess_new_data(combined_data)
            
            # Run detection with all models
            detection_results = self._run_detection(processed_data)
            
            # Process results
            self._process_detection_results(detection_results, combined_data)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics['processing_times'].append(processing_time)
            self.metrics['total_processed'] += len(combined_data)
            self.metrics['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
    
    def _run_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run detection using all loaded models"""
        results = {}
        
        for model_name, model_info in self.models.items():
            try:
                model = model_info['model']
                model_type = model_info['type']
                
                if model_type == 'ml':
                    # Traditional ML model
                    predictions = model.predict(data)
                    probabilities = model.predict_proba(data) if hasattr(model, 'predict_proba') else None
                    
                elif model_type == 'dl':
                    # Deep learning model
                    if model_name == 'autoencoder':
                        # Autoencoder returns reconstruction error
                        reconstruction_error = model.predict(data)
                        threshold = DETECTION_THRESHOLDS['anomaly_threshold']
                        predictions = (reconstruction_error > threshold).astype(int)
                        probabilities = reconstruction_error
                    else:
                        # Other DL models
                        predictions = model.predict(data)
                        probabilities = model.predict_proba(data) if hasattr(model, 'predict_proba') else None
                
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                # Store predictions
                model_info['predictions'].extend(predictions)
                if probabilities is not None:
                    model_info['confidence_scores'].extend(probabilities)
                
                results[model_name] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'model_type': model_type
                }
                
            except Exception as e:
                logger.error(f"Error running detection with {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _process_detection_results(self, results: Dict[str, Any], original_data: pd.DataFrame):
        """Process and analyze detection results"""
        # Ensemble decision
        ensemble_predictions = self._ensemble_decision(results)
        
        # Create detection records
        for i, (idx, row) in enumerate(original_data.iterrows()):
            detection_record = {
                'timestamp': datetime.now(),
                'data_index': idx,
                'ensemble_prediction': ensemble_predictions[i] if i < len(ensemble_predictions) else 0,
                'individual_predictions': {},
                'confidence_scores': {},
                'alert_level': 'normal'
            }
            
            # Collect individual model predictions
            for model_name, model_results in results.items():
                if 'error' not in model_results:
                    pred = model_results['predictions'][i] if i < len(model_results['predictions']) else 0
                    prob = model_results['probabilities'][i] if (model_results['probabilities'] is not None and 
                                                               i < len(model_results['probabilities'])) else 0
                    
                    detection_record['individual_predictions'][model_name] = pred
                    detection_record['confidence_scores'][model_name] = prob
            
            # Determine alert level
            detection_record['alert_level'] = self._determine_alert_level(detection_record)
            
            # Store detection result
            self.detection_results.append(detection_record)
            
            # Update detection history
            if detection_record['ensemble_prediction'] == 1:
                self.metrics['total_detected'] += 1
                self.detection_history['detections'].append(detection_record)
            
            # Trigger alerts if necessary
            if detection_record['alert_level'] in ['high', 'critical']:
                self._trigger_alert(detection_record)
    
    def _ensemble_decision(self, results: Dict[str, Any]) -> List[int]:
        """Make ensemble decision based on all model predictions"""
        if not results:
            return []
        
        # Get the length of predictions from the first valid model
        prediction_length = 0
        for model_results in results.values():
            if 'error' not in model_results and 'predictions' in model_results:
                prediction_length = len(model_results['predictions'])
                break
        
        if prediction_length == 0:
            return []
        
        ensemble_predictions = []
        
        for i in range(prediction_length):
            votes = []
            weights = []
            
            for model_name, model_results in results.items():
                if 'error' not in model_results and 'predictions' in model_results:
                    if i < len(model_results['predictions']):
                        votes.append(model_results['predictions'][i])
                        
                        # Weight based on model type and confidence
                        if model_results['probabilities'] is not None and i < len(model_results['probabilities']):
                            confidence = model_results['probabilities'][i]
                            if isinstance(confidence, (list, np.ndarray)):
                                confidence = max(confidence)
                            weight = confidence
                        else:
                            weight = 0.5  # Default weight
                        
                        weights.append(weight)
            
            if votes:
                # Weighted voting
                weighted_vote = np.average(votes, weights=weights)
                ensemble_prediction = 1 if weighted_vote > 0.5 else 0
            else:
                ensemble_prediction = 0
            
            ensemble_predictions.append(ensemble_prediction)
        
        return ensemble_predictions
    
    def _determine_alert_level(self, detection_record: Dict[str, Any]) -> str:
        """Determine alert level based on detection results"""
        ensemble_pred = detection_record['ensemble_prediction']
        confidence_scores = detection_record['confidence_scores']
        
        if ensemble_pred == 0:
            return 'normal'
        
        # Calculate average confidence
        if confidence_scores:
            avg_confidence = np.mean(list(confidence_scores.values()))
        else:
            avg_confidence = 0.5
        
        # Determine alert level based on confidence and thresholds
        if avg_confidence >= DETECTION_THRESHOLDS['confidence_threshold']:
            return 'critical'
        elif avg_confidence >= DETECTION_THRESHOLDS['intrusion_threshold']:
            return 'high'
        else:
            return 'medium'
    
    def _trigger_alert(self, detection_record: Dict[str, Any]):
        """Trigger alert callbacks"""
        alert_data = {
            'timestamp': detection_record['timestamp'],
            'alert_level': detection_record['alert_level'],
            'ensemble_prediction': detection_record['ensemble_prediction'],
            'confidence_scores': detection_record['confidence_scores'],
            'individual_predictions': detection_record['individual_predictions']
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get current detection statistics"""
        current_time = datetime.now()
        
        # Calculate detection rate
        if self.metrics['total_processed'] > 0:
            detection_rate = self.metrics['total_detected'] / self.metrics['total_processed']
        else:
            detection_rate = 0
        
        # Calculate average processing time
        if self.metrics['processing_times']:
            avg_processing_time = np.mean(self.metrics['processing_times'])
        else:
            avg_processing_time = 0
        
        # Recent detection rate (last 100 detections)
        recent_detections = list(self.detection_results)[-100:]
        recent_detection_rate = sum(1 for d in recent_detections if d['ensemble_prediction'] == 1) / max(len(recent_detections), 1)
        
        stats = {
            'total_processed': self.metrics['total_processed'],
            'total_detected': self.metrics['total_detected'],
            'detection_rate': detection_rate,
            'recent_detection_rate': recent_detection_rate,
            'avg_processing_time': avg_processing_time,
            'queue_size': self.data_queue.qsize(),
            'is_running': self.is_running,
            'models_loaded': len(self.models),
            'last_update': self.metrics['last_update'].isoformat(),
            'alert_levels': self._get_alert_level_distribution()
        }
        
        return stats
    
    def _get_alert_level_distribution(self) -> Dict[str, int]:
        """Get distribution of alert levels"""
        recent_detections = list(self.detection_results)[-1000:]  # Last 1000 detections
        
        distribution = defaultdict(int)
        for detection in recent_detections:
            distribution[detection['alert_level']] += 1
        
        return dict(distribution)
    
    def get_recent_detections(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent detection results"""
        return list(self.detection_results)[-limit:]
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for each model"""
        performance = {}
        
        for model_name, model_info in self.models.items():
            predictions = list(model_info['predictions'])
            confidence_scores = list(model_info['confidence_scores'])
            
            if predictions:
                detection_count = sum(predictions)
                avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
                
                performance[model_name] = {
                    'total_predictions': len(predictions),
                    'detection_count': detection_count,
                    'detection_rate': detection_count / len(predictions),
                    'avg_confidence': avg_confidence,
                    'model_type': model_info['type']
                }
            else:
                performance[model_name] = {
                    'total_predictions': 0,
                    'detection_count': 0,
                    'detection_rate': 0,
                    'avg_confidence': 0,
                    'model_type': model_info['type']
                }
        
        return performance
    
    def export_detection_log(self, filepath: str, limit: int = None):
        """Export detection log to file"""
        detections = list(self.detection_results)
        if limit:
            detections = detections[-limit:]
        
        # Convert to JSON-serializable format
        export_data = []
        for detection in detections:
            export_detection = detection.copy()
            export_detection['timestamp'] = detection['timestamp'].isoformat()
            export_data.append(export_detection)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} detection records to {filepath}")
    
    def clear_history(self):
        """Clear detection history"""
        self.detection_results.clear()
        self.detection_history.clear()
        for model_info in self.models.values():
            model_info['predictions'].clear()
            model_info['confidence_scores'].clear()
        
        logger.info("Detection history cleared")
