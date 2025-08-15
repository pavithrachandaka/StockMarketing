"""
Utility functions for the QML FTSE 100 prediction project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def normalize_features(features: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize features to quantum range (0 to π).
    
    Args:
        features: Input features array
        method: Normalization method ('minmax', 'standard', 'quantum')
    
    Returns:
        Normalized features
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        normalized = (features - min_vals) / (max_vals - min_vals + 1e-8)
        # Scale to [0, π] for quantum encoding
        return normalized * np.pi
    
    elif method == 'standard':
        # Standard normalization
        mean_vals = np.mean(features, axis=0)
        std_vals = np.std(features, axis=0)
        normalized = (features - mean_vals) / (std_vals + 1e-8)
        # Scale to [0, π] for quantum encoding
        return (normalized + 3) * np.pi / 6  # Assuming 99.7% of data within ±3σ
    
    elif method == 'quantum':
        # Direct scaling to quantum range
        return features * np.pi / np.max(np.abs(features))
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def create_labels(prices: pd.Series, threshold: float = 0.0) -> np.ndarray:
    """
    Create binary labels for market direction prediction.
    
    Args:
        prices: Series of closing prices
        threshold: Minimum change threshold for labeling
    
    Returns:
        Binary labels (1 for up, 0 for down)
    """
    price_changes = prices.pct_change().shift(-1)  # Next day's change
    labels = (price_changes > threshold).astype(int)
    return labels[:-1]  # Remove last row (no next day data)

def split_data(features: np.ndarray, labels: np.ndarray, 
               train_ratio: float = 0.8, random_state: int = 42) -> Tuple:
    """
    Split data into training and testing sets.
    
    Args:
        features: Feature matrix
        labels: Label array
        train_ratio: Proportion of data for training
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    np.random.seed(random_state)
    n_samples = len(features)
    n_train = int(n_samples * train_ratio)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = features[train_indices]
    X_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics

def plot_feature_importance(feature_names: List[str], importance_scores: np.ndarray, 
                          title: str = "Feature Importance") -> None:
    """
    Plot feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores for each feature
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    sorted_indices = np.argsort(importance_scores)[::-1]
    
    plt.bar(range(len(feature_names)), importance_scores[sorted_indices])
    plt.xticks(range(len(feature_names)), 
               [feature_names[i] for i in sorted_indices], rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> None:
    """
    Plot confusion matrix with annotations.
    
    Args:
        cm: Confusion matrix
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_training_history(history: List[float], title: str = "Training History") -> None:
    """
    Plot training history (loss/accuracy over epochs).
    
    Args:
        history: List of metric values
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_model_summary(model_name: str, metrics: dict) -> None:
    """
    Print a formatted model performance summary.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metrics
    """
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"{'='*50}")

def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Validate data quality and return summary statistics.
    
    Args:
        df: Input dataframe
    
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'date_range': f"{df.index.min()} to {df.index.max()}",
        'unique_dates': len(df.index.unique()),
        'data_completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    return quality_report

def save_model_results(model_name: str, metrics: dict, 
                      feature_names: List[str], save_path: str = "results/") -> None:
    """
    Save model results to file.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metrics
        feature_names: List of feature names
        save_path: Directory to save results
    """
    import os
    import json
    from datetime import datetime
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare results
    results = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'feature_names': feature_names
    }
    
    # Save to JSON file
    filename = f"{save_path}/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filename}")

def quantum_circuit_info(circuit) -> dict:
    """
    Get information about a quantum circuit.
    
    Args:
        circuit: Qiskit quantum circuit
    
    Returns:
        Dictionary with circuit information
    """
    info = {
        'num_qubits': circuit.num_qubits,
        'num_clbits': circuit.num_clbits,
        'depth': circuit.depth(),
        'size': circuit.size(),
        'width': circuit.width()
    }
    
    return info

def print_quantum_info(info: dict) -> None:
    """
    Print quantum circuit information in a formatted way.
    
    Args:
        info: Dictionary with quantum circuit information
    """
    print(f"\n{'='*40}")
    print("QUANTUM CIRCUIT INFORMATION")
    print(f"{'='*40}")
    print(f"Number of Qubits:    {info['num_qubits']}")
    print(f"Number of Clbits:    {info['num_clbits']}")
    print(f"Circuit Depth:       {info['depth']}")
    print(f"Circuit Size:        {info['size']}")
    print(f"Circuit Width:       {info['width']}")
    print(f"{'='*40}")
