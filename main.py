"""
Main entry point for AI-powered Intrusion Detection System
"""
import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.config import LOGGING_CONFIG
from src.dashboard.streamlit_dashboard import create_dashboard
from src.data_collection import DatasetLoader, TrafficDataGenerator
from src.preprocessing import DataPreprocessor
from src.models import MLModelTrainer, DeepLearningTrainer, ModelEvaluator
from src.detection import RealTimeDetector, AlertManager, AnomalyDetector
from src.federated_learning import FederatedTrainer

# Configure logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def run_dashboard():
    """Run the Streamlit dashboard"""
    logger.info("Starting Streamlit dashboard...")
    create_dashboard()


def run_training_pipeline():
    """Run the complete training pipeline"""
    logger.info("Starting training pipeline...")
    
    # Load data
    dataset_loader = DatasetLoader()
    train_data, test_data = dataset_loader.load_unsw_nb15()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    preprocessed_data = preprocessor.preprocess_dataset(
        train_data, 'label', test_size=0.2, validation_size=0.1
    )
    
    # Train ML models
    ml_trainer = MLModelTrainer()
    ml_results = ml_trainer.train_models(
        preprocessed_data['X_train'], preprocessed_data['y_train'],
        preprocessed_data['X_val'], preprocessed_data['y_val']
    )
    
    # Train DL models
    dl_trainer = DeepLearningTrainer()
    dl_results = dl_trainer.train_models(
        preprocessed_data['X_train'], preprocessed_data['y_train'],
        preprocessed_data['X_val'], preprocessed_data['y_val']
    )
    
    # Evaluate models
    evaluator = ModelEvaluator()
    
    # Evaluate ML models
    ml_evaluations = {}
    for model_name, model_info in ml_results.items():
        if model_info['status'] == 'success':
            evaluation = evaluator.evaluate_model(
                model_info['model'], 
                preprocessed_data['X_test'], 
                preprocessed_data['y_test'],
                model_name
            )
            ml_evaluations[model_name] = evaluation
    
    # Evaluate DL models
    dl_evaluations = {}
    for model_name, model_info in dl_results.items():
        if model_info['status'] == 'success':
            evaluation = evaluator.evaluate_model(
                model_info['model'], 
                preprocessed_data['X_test'], 
                preprocessed_data['y_test'],
                model_name
            )
            dl_evaluations[model_name] = evaluation
    
    # Generate evaluation report
    all_evaluations = {**ml_evaluations, **dl_evaluations}
    report = evaluator.generate_evaluation_report(all_evaluations)
    
    logger.info("Training pipeline completed successfully")
    logger.info(f"Best model: {report['executive_summary']['best_model']}")
    logger.info(f"Best F1-score: {report['executive_summary']['best_f1_score']:.4f}")
    
    return report


def run_realtime_detection():
    """Run real-time detection system"""
    logger.info("Starting real-time detection system...")
    
    # Initialize components
    detector = RealTimeDetector()
    alert_manager = AlertManager()
    anomaly_detector = AnomalyDetector()
    
    # Load trained models (in a real implementation, these would be loaded from disk)
    ml_trainer = MLModelTrainer()
    dl_trainer = DeepLearningTrainer()
    
    # Add models to detector
    # Note: In a real implementation, you would load pre-trained models
    # detector.add_model('random_forest', trained_model, 'ml')
    # detector.add_model('lstm', trained_dl_model, 'dl')
    
    # Add alert callback
    def alert_callback(alert_data):
        logger.warning(f"Alert triggered: {alert_data['alert_level']}")
        alert_manager.process_alert(alert_data)
    
    detector.add_alert_callback(alert_callback)
    
    # Generate sample data for testing
    data_generator = TrafficDataGenerator()
    sample_data = data_generator.generate_mixed_traffic(n_samples=1000, attack_ratio=0.1)
    
    # Start detection
    detector.start_detection()
    
    # Simulate real-time data processing
    try:
        for i in range(10):  # Process 10 batches
            batch_data = sample_data.iloc[i*100:(i+1)*100]
            detector.add_data(batch_data)
            
            # Get detection stats
            stats = detector.get_detection_stats()
            logger.info(f"Batch {i+1}: Processed {stats['total_processed']}, "
                       f"Detected {stats['total_detected']}, "
                       f"Rate: {stats['detection_rate']:.2%}")
            
            # Sleep to simulate real-time processing
            import time
            time.sleep(2)
    
    except KeyboardInterrupt:
        logger.info("Stopping real-time detection...")
    
    finally:
        detector.stop_detection()
        logger.info("Real-time detection stopped")


def run_federated_learning():
    """Run federated learning training"""
    logger.info("Starting federated learning...")
    
    # Initialize federated trainer
    federated_config = {
        'num_rounds': 5,
        'num_clients_per_round': 3,
        'learning_rate': 0.01,
        'differential_privacy': True,
        'noise_multiplier': 1.1
    }
    
    federated_trainer = FederatedTrainer(federated_config)
    
    # Define model architecture
    model_architecture = {
        'input_size': 50,  # Example input size
        'layers': [
            {'type': 'linear', 'output_size': 128},
            {'type': 'relu'},
            {'type': 'dropout', 'rate': 0.2},
            {'type': 'linear', 'output_size': 64},
            {'type': 'relu'},
            {'type': 'linear', 'output_size': 1}
        ]
    }
    
    # Initialize global model
    federated_trainer.initialize_global_model(model_architecture)
    
    # Add clients
    for i in range(5):
        client_config = {
            'data_size': 1000,
            'cloud_provider': f'cloud_{i}',
            'region': f'region_{i}'
        }
        federated_trainer.add_client(f'client_{i}', client_config)
    
    # Start federated training
    results = federated_trainer.start_federated_training()
    
    logger.info("Federated learning completed")
    logger.info(f"Rounds completed: {results['rounds_completed']}")
    logger.info(f"Final accuracy: {results['global_accuracy'][-1]:.4f}")
    logger.info(f"Convergence achieved: {results['convergence_achieved']}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI-powered Intrusion Detection System")
    parser.add_argument('--mode', choices=['dashboard', 'train', 'detect', 'federated'], 
                       default='dashboard', help='Operation mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        if args.mode == 'dashboard':
            run_dashboard()
        elif args.mode == 'train':
            run_training_pipeline()
        elif args.mode == 'detect':
            run_realtime_detection()
        elif args.mode == 'federated':
            run_federated_learning()
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error running {args.mode}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
