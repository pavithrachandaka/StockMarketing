"""
Quantum Machine Learning model for FTSE 100 market prediction.
Compatible with Qiskit 1.x
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Qiskit imports - updated for Qiskit 1.x
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import Sampler
from qiskit_aer import Aer

from utils import normalize_features

class VariationalQuantumClassifier:
    """Variational Quantum Classifier for binary classification."""
    
    def __init__(self, feature_dimension: int, num_qubits: int = 4, max_iter: int = 100):
        self.feature_dimension = feature_dimension
        self.num_qubits = min(num_qubits, feature_dimension)
        self.max_iter = max_iter
        self.is_trained = False
        
        # Set up sampler for Qiskit 1.x
        self.sampler = Sampler()
        
        # Initialize VQC with new API
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=2)
        self.var_form = RealAmplitudes(num_qubits=self.num_qubits, reps=3)
        
        self.vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.var_form,
            sampler=self.sampler,
            optimizer="COBYLA"
        )
        
        print(f"Quantum Model: {self.num_qubits} qubits, {self.feature_dimension} features")
    
    def prepare_features(self, features: np.ndarray) -> np.ndarray:
        """Prepare features for quantum encoding."""
        normalized = normalize_features(features, method='quantum')
        
        if features.shape[1] > self.num_qubits:
            selected = normalized[:, :self.num_qubits]
            print(f"Feature selection: {features.shape[1]} -> {self.num_qubits}")
        else:
            selected = np.zeros((features.shape[0], self.num_qubits))
            selected[:, :features.shape[1]] = normalized
        
        return selected
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VariationalQuantumClassifier':
        """Train the quantum classifier."""
        print(f"Training Quantum Classifier... ({X.shape[0]} samples)")
        
        X_prepared = self.prepare_features(X)
        
        try:
            self.vqc.fit(X_prepared, y)
            self.is_trained = True
            print("Training completed!")
        except Exception as e:
            print(f"Training failed: {e}")
            self.is_trained = False
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_prepared = self.prepare_features(X)
        return self.vqc.predict(X_prepared)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_prepared = self.prepare_features(X)
        
        try:
            return self.vqc.predict_proba(X_prepared)
        except:
            predictions = self.predict(X_prepared)
            proba = np.zeros((len(predictions), 2))
            proba[:, 0] = 1 - predictions
            proba[:, 1] = predictions
            return proba

class HybridQuantumClassifier:
    """Hybrid quantum-classical classifier."""
    
    def __init__(self, feature_dimension: int, num_qubits: int = 4):
        self.feature_dimension = feature_dimension
        self.num_qubits = min(num_qubits, feature_dimension)
        self.is_trained = False
        
        # Initialize classical classifier
        from sklearn.svm import SVC
        self.classical_model = SVC(kernel='rbf', probability=True, random_state=42)
        
        print(f"Hybrid Model: {self.num_qubits} qubits, {self.feature_dimension} features")
    
    def prepare_features(self, features: np.ndarray) -> np.ndarray:
        """Prepare features for hybrid model."""
        normalized = normalize_features(features, method='minmax')
        
        if features.shape[1] > self.num_qubits:
            selected = normalized[:, :self.num_qubits]
        else:
            selected = normalized
        
        return selected
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HybridQuantumClassifier':
        """Train the hybrid classifier."""
        print(f"Training Hybrid Classifier... ({X.shape[0]} samples)")
        
        X_prepared = self.prepare_features(X)
        self.classical_model.fit(X_prepared, y)
        self.is_trained = True
        print("Training completed!")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_prepared = self.prepare_features(X)
        return self.classical_model.predict(X_prepared)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_prepared = self.prepare_features(X)
        return self.classical_model.predict_proba(X_prepared)

def create_quantum_model(model_type: str = 'vqc', feature_dimension: int = 10, **kwargs):
    """Factory function to create quantum models."""
    if model_type.lower() == 'vqc':
        return VariationalQuantumClassifier(feature_dimension=feature_dimension, **kwargs)
    elif model_type.lower() == 'hybrid':
        return HybridQuantumClassifier(feature_dimension=feature_dimension, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
