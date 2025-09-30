"""
Federated learning trainer for multi-cloud intrusion detection
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import time
from datetime import datetime
import hashlib

# Optional imports for deep learning and cryptography
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for when torch is not available
    class nn:
        class Module:
            pass
        class Sequential:
            def __init__(self, *args):
                pass
        class Linear:
            def __init__(self, *args, **kwargs):
                pass
        class ReLU:
            def __init__(self, *args, **kwargs):
                pass
        class Dropout:
            def __init__(self, *args, **kwargs):
                pass
        class BatchNorm1d:
            def __init__(self, *args, **kwargs):
                pass

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    # Create dummy Fernet class
    class Fernet:
        def __init__(self, key):
            pass
        def encrypt(self, data):
            return data
        def decrypt(self, data):
            return data
    Fernet.generate_key = lambda: b'dummy_key'

try:
    import asyncio
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

logger = logging.getLogger(__name__)


class FederatedTrainer:
    """Federated learning trainer for privacy-preserving intrusion detection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.clients = {}
        self.global_model = None
        self.training_history = []
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Federated learning parameters
        self.num_rounds = self.config.get('num_rounds', 10)
        self.num_clients_per_round = self.config.get('num_clients_per_round', 3)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs_per_round = self.config.get('epochs_per_round', 5)
        
        # Privacy parameters
        self.differential_privacy = self.config.get('differential_privacy', True)
        self.noise_multiplier = self.config.get('noise_multiplier', 1.1)
        self.l2_norm_clip = self.config.get('l2_norm_clip', 1.0)
        
        # Security parameters
        self.secure_aggregation = self.config.get('secure_aggregation', True)
        self.encryption_enabled = self.config.get('encryption_enabled', True)
        
    def initialize_global_model(self, model_architecture: Dict[str, Any]):
        """Initialize the global model architecture"""
        self.global_model = self._create_model(model_architecture)
        logger.info("Global model initialized")
    
    def _create_model(self, architecture: Dict[str, Any]) -> nn.Module:
        """Create neural network model based on architecture"""
        layers = []
        input_size = architecture['input_size']
        
        for layer_config in architecture['layers']:
            if layer_config['type'] == 'linear':
                layers.append(nn.Linear(input_size, layer_config['output_size']))
                input_size = layer_config['output_size']
            elif layer_config['type'] == 'relu':
                layers.append(nn.ReLU())
            elif layer_config['type'] == 'dropout':
                layers.append(nn.Dropout(layer_config['rate']))
            elif layer_config['type'] == 'batch_norm':
                layers.append(nn.BatchNorm1d(input_size))
        
        return nn.Sequential(*layers)
    
    def add_client(self, client_id: str, client_config: Dict[str, Any]):
        """Add a federated learning client"""
        self.clients[client_id] = {
            'config': client_config,
            'model': None,
            'data_size': 0,
            'last_update': None,
            'participation_count': 0,
            'performance_history': []
        }
        logger.info(f"Added client: {client_id}")
    
    def start_federated_training(self) -> Dict[str, Any]:
        """Start federated learning training process"""
        logger.info("Starting federated learning training...")
        
        training_results = {
            'rounds_completed': 0,
            'global_accuracy': [],
            'client_performances': {},
            'convergence_achieved': False,
            'training_time': 0
        }
        
        start_time = time.time()
        
        for round_num in range(self.num_rounds):
            logger.info(f"Starting federated round {round_num + 1}/{self.num_rounds}")
            
            # Select clients for this round
            selected_clients = self._select_clients_for_round()
            
            # Distribute global model to selected clients
            self._distribute_global_model(selected_clients)
            
            # Train clients locally
            client_updates = self._train_clients_locally(selected_clients)
            
            # Aggregate client updates
            global_update = self._aggregate_updates(client_updates)
            
            # Update global model
            self._update_global_model(global_update)
            
            # Evaluate global model
            global_accuracy = self._evaluate_global_model()
            training_results['global_accuracy'].append(global_accuracy)
            
            # Update training history
            round_info = {
                'round': round_num + 1,
                'selected_clients': selected_clients,
                'global_accuracy': global_accuracy,
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(round_info)
            
            logger.info(f"Round {round_num + 1} completed. Global accuracy: {global_accuracy:.4f}")
            
            # Check for convergence
            if self._check_convergence(training_results['global_accuracy']):
                training_results['convergence_achieved'] = True
                logger.info("Training converged early")
                break
        
        training_results['rounds_completed'] = len(training_results['global_accuracy'])
        training_results['training_time'] = time.time() - start_time
        
        logger.info(f"Federated training completed in {training_results['training_time']:.2f}s")
        
        return training_results
    
    def _select_clients_for_round(self) -> List[str]:
        """Select clients for the current training round"""
        available_clients = list(self.clients.keys())
        
        # Simple random selection (can be improved with more sophisticated strategies)
        num_to_select = min(self.num_clients_per_round, len(available_clients))
        selected = np.random.choice(available_clients, size=num_to_select, replace=False)
        
        return selected.tolist()
    
    def _distribute_global_model(self, client_ids: List[str]):
        """Distribute global model to selected clients"""
        for client_id in client_ids:
            if self.encryption_enabled:
                # Encrypt model parameters
                model_state = self.global_model.state_dict()
                serialized_model = json.dumps({k: v.tolist() for k, v in model_state.items()})
                encrypted_model = self.cipher_suite.encrypt(serialized_model.encode())
                
                # Send encrypted model to client
                self._send_to_client(client_id, 'model_update', encrypted_model)
            else:
                # Send model directly
                self._send_to_client(client_id, 'model_update', self.global_model.state_dict())
    
    def _train_clients_locally(self, client_ids: List[str]) -> Dict[str, Any]:
        """Train clients locally and collect updates"""
        client_updates = {}
        
        for client_id in client_ids:
            logger.info(f"Training client {client_id} locally...")
            
            # Request local training from client
            training_request = {
                'epochs': self.epochs_per_round,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate
            }
            
            # Simulate local training (in real implementation, this would be done on client side)
            client_update = self._simulate_local_training(client_id, training_request)
            
            # Apply differential privacy if enabled
            if self.differential_privacy:
                client_update = self._apply_differential_privacy(client_update)
            
            client_updates[client_id] = client_update
            
            # Update client participation count
            self.clients[client_id]['participation_count'] += 1
            self.clients[client_id]['last_update'] = datetime.now()
        
        return client_updates
    
    def _simulate_local_training(self, client_id: str, training_request: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate local training on client (placeholder for actual implementation)"""
        # In a real implementation, this would be done on the client side
        # For simulation, we'll generate random updates
        
        model_params = {}
        for name, param in self.global_model.named_parameters():
            # Generate random update (simulating local training)
            update = torch.randn_like(param) * 0.01
            model_params[name] = update
        
        return {
            'model_params': model_params,
            'data_size': np.random.randint(100, 1000),  # Simulated data size
            'training_loss': np.random.uniform(0.1, 0.5),
            'training_accuracy': np.random.uniform(0.8, 0.95)
        }
    
    def _apply_differential_privacy(self, client_update: Dict[str, Any]) -> Dict[str, Any]:
        """Apply differential privacy to client update"""
        if not self.differential_privacy:
            return client_update
        
        # Clip gradients
        clipped_params = {}
        for name, param in client_update['model_params'].items():
            param_norm = torch.norm(param)
            if param_norm > self.l2_norm_clip:
                clipped_param = param * (self.l2_norm_clip / param_norm)
            else:
                clipped_param = param
            clipped_params[name] = clipped_param
        
        # Add Gaussian noise
        noisy_params = {}
        for name, param in clipped_params.items():
            noise = torch.randn_like(param) * self.noise_multiplier * self.l2_norm_clip
            noisy_params[name] = param + noise
        
        client_update['model_params'] = noisy_params
        return client_update
    
    def _aggregate_updates(self, client_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate client updates using FedAvg algorithm"""
        if not client_updates:
            return {}
        
        # Calculate total data size
        total_data_size = sum(update['data_size'] for update in client_updates.values())
        
        # Weighted average of model parameters
        aggregated_params = {}
        
        for client_id, update in client_updates.items():
            weight = update['data_size'] / total_data_size
            
            for name, param in update['model_params'].items():
                if name not in aggregated_params:
                    aggregated_params[name] = torch.zeros_like(param)
                aggregated_params[name] += weight * param
        
        return {
            'model_params': aggregated_params,
            'total_data_size': total_data_size,
            'num_clients': len(client_updates)
        }
    
    def _update_global_model(self, global_update: Dict[str, Any]):
        """Update global model with aggregated updates"""
        if not global_update or 'model_params' not in global_update:
            return
        
        # Update global model parameters
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in global_update['model_params']:
                    param.data += global_update['model_params'][name]
    
    def _evaluate_global_model(self) -> float:
        """Evaluate global model performance"""
        # In a real implementation, this would use a test dataset
        # For simulation, we'll return a random accuracy
        return np.random.uniform(0.85, 0.95)
    
    def _check_convergence(self, accuracy_history: List[float], window_size: int = 5) -> bool:
        """Check if training has converged"""
        if len(accuracy_history) < window_size:
            return False
        
        recent_accuracies = accuracy_history[-window_size:]
        accuracy_std = np.std(recent_accuracies)
        
        # Consider converged if standard deviation is very low
        return accuracy_std < 0.001
    
    def _send_to_client(self, client_id: str, message_type: str, data: Any):
        """Send message to client (placeholder for actual communication)"""
        # In a real implementation, this would use actual network communication
        logger.info(f"Sending {message_type} to client {client_id}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get federated training statistics"""
        if not self.training_history:
            return {'error': 'No training history available'}
        
        stats = {
            'total_rounds': len(self.training_history),
            'total_clients': len(self.clients),
            'active_clients': len([c for c in self.clients.values() if c['last_update']]),
            'final_accuracy': self.training_history[-1]['global_accuracy'] if self.training_history else 0,
            'convergence_achieved': self._check_convergence([r['global_accuracy'] for r in self.training_history]),
            'client_participation': {client_id: client['participation_count'] 
                                   for client_id, client in self.clients.items()}
        }
        
        return stats
    
    def export_global_model(self, filepath: str):
        """Export the trained global model"""
        if self.global_model is None:
            raise ValueError("No global model to export")
        
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }, filepath)
        
        logger.info(f"Global model exported to {filepath}")
    
    def load_global_model(self, filepath: str):
        """Load a trained global model"""
        checkpoint = torch.load(filepath)
        
        if self.global_model is None:
            raise ValueError("Global model not initialized")
        
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Global model loaded from {filepath}")
    
    def add_privacy_budget_tracking(self, epsilon: float = 1.0, delta: float = 1e-5):
        """Add privacy budget tracking for differential privacy"""
        self.privacy_budget = {
            'epsilon': epsilon,
            'delta': delta,
            'used_epsilon': 0.0,
            'remaining_epsilon': epsilon
        }
        
        logger.info(f"Privacy budget initialized: ε={epsilon}, δ={delta}")
    
    def update_privacy_budget(self, epsilon_used: float):
        """Update privacy budget after training round"""
        if hasattr(self, 'privacy_budget'):
            self.privacy_budget['used_epsilon'] += epsilon_used
            self.privacy_budget['remaining_epsilon'] -= epsilon_used
            
            if self.privacy_budget['remaining_epsilon'] <= 0:
                logger.warning("Privacy budget exhausted!")
    
    def get_privacy_budget_status(self) -> Dict[str, Any]:
        """Get current privacy budget status"""
        if not hasattr(self, 'privacy_budget'):
            return {'error': 'Privacy budget not initialized'}
        
        return {
            'total_epsilon': self.privacy_budget['epsilon'],
            'used_epsilon': self.privacy_budget['used_epsilon'],
            'remaining_epsilon': self.privacy_budget['remaining_epsilon'],
            'budget_exhausted': self.privacy_budget['remaining_epsilon'] <= 0
        }
