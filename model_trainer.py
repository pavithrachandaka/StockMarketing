"""
Model training and evaluation for FTSE 100 market prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from data_collector import FTSE100DataCollector, collect_sample_data
from feature_engineering import FeatureEngineer, create_sample_features
from quantum_model import VariationalQuantumClassifier, HybridQuantumClassifier, create_quantum_model
from utils import (
    create_labels, split_data, calculate_metrics, 
    plot_confusion_matrix, plot_training_history,
    print_model_summary, save_model_results
)

class ModelTrainer:
    """
    Trains and evaluates quantum models for market prediction.
    """
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize the model trainer.
        
        Args:
            data: Input OHLCV data (optional)
        """
        self.data = data
        self.features = None
        self.labels = None
        self.feature_names = []
        self.models = {}
        self.results = {}
        
    def prepare_data(self, threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            threshold: Minimum change threshold for labeling
        
        Returns:
            Tuple of (features, labels, feature_names)
        """
        if self.data is None:
            print("Collecting FTSE 100 data...")
            self.data = collect_sample_data()
        
        if self.data is None:
            raise ValueError("Failed to collect data")
        
        # Create features
        print("Creating features...")
        self.features, self.feature_names = create_sample_features(self.data)
        
        # Create labels
        print("Creating labels...")
        self.labels = create_labels(self.data['Close'], threshold=threshold)
        
        # Align features and labels
        min_length = min(len(self.features), len(self.labels))
        self.features = self.features.iloc[:min_length]
        self.labels = self.labels[:min_length]
        
        print(f"Data prepared: {self.features.shape[0]} samples, {self.features.shape[1]} features")
        print(f"Label distribution: {np.bincount(self.labels)}")
        
        return self.features.values, self.labels, self.feature_names
    
    def train_models(self, 
                    features: np.ndarray, 
                    labels: np.ndarray,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Dict:
        """
        Train multiple quantum models.
        
        Args:
            features: Input features
            labels: Input labels
            test_size: Proportion of data for testing
            random_state: Random seed
        
        Returns:
            Dictionary with training results
        """
        # Split data
        X_train, X_test, y_train, y_test = split_data(
            features, labels, train_ratio=1-test_size, random_state=random_state
        )
        
        print(f"\nTraining Models:")
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        
        results = {}
        
        # Train Variational Quantum Classifier
        print("\n" + "="*50)
        print("TRAINING VARIATIONAL QUANTUM CLASSIFIER")
        print("="*50)
        
        try:
            vqc = VariationalQuantumClassifier(
                feature_dimension=X_train.shape[1],
                num_qubits=min(4, X_train.shape[1]),
                max_iter=50  # Reduced for faster training
            )
            
            vqc.fit(X_train, y_train)
            y_pred_vqc = vqc.predict(X_test)
            metrics_vqc = calculate_metrics(y_test, y_pred_vqc)
            
            self.models['vqc'] = vqc
            results['vqc'] = {
                'model': vqc,
                'metrics': metrics_vqc,
                'predictions': y_pred_vqc,
                'probabilities': vqc.predict_proba(X_test)
            }
            
            print_model_summary("Variational Quantum Classifier", metrics_vqc)
            
        except Exception as e:
            print(f"VQC training failed: {e}")
            results['vqc'] = None
        
        # Train Hybrid Quantum-Classical Classifier
        print("\n" + "="*50)
        print("TRAINING HYBRID QUANTUM-CLASSICAL CLASSIFIER")
        print("="*50)
        
        try:
            hybrid = HybridQuantumClassifier(
                feature_dimension=X_train.shape[1],
                num_qubits=min(4, X_train.shape[1])
            )
            
            hybrid.fit(X_train, y_train)
            y_pred_hybrid = hybrid.predict(X_test)
            metrics_hybrid = calculate_metrics(y_test, y_pred_hybrid)
            
            self.models['hybrid'] = hybrid
            results['hybrid'] = {
                'model': hybrid,
                'metrics': metrics_hybrid,
                'predictions': y_pred_hybrid,
                'probabilities': hybrid.predict_proba(X_test)
            }
            
            print_model_summary("Hybrid Quantum-Classical", metrics_hybrid)
            
        except Exception as e:
            print(f"Hybrid training failed: {e}")
            results['hybrid'] = None
        
        # Train classical baseline
        print("\n" + "="*50)
        print("TRAINING CLASSICAL BASELINE")
        print("="*50)
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            metrics_rf = calculate_metrics(y_test, y_pred_rf)
            
            self.models['random_forest'] = rf
            results['random_forest'] = {
                'model': rf,
                'metrics': metrics_rf,
                'predictions': y_pred_rf,
                'probabilities': rf.predict_proba(X_test)
            }
            
            print_model_summary("Random Forest", metrics_rf)
            
            # SVM
            svm = SVC(kernel='rbf', probability=True, random_state=random_state)
            svm.fit(X_train, y_train)
            y_pred_svm = svm.predict(X_test)
            metrics_svm = calculate_metrics(y_test, y_pred_svm)
            
            self.models['svm'] = svm
            results['svm'] = {
                'model': svm,
                'metrics': metrics_svm,
                'predictions': y_pred_svm,
                'probabilities': svm.predict_proba(X_test)
            }
            
            print_model_summary("Support Vector Machine", metrics_svm)
            
        except Exception as e:
            print(f"Classical training failed: {e}")
        
        self.results = results
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all trained models.
        
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            print("No models trained yet. Run train_models() first.")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            if result is not None:
                metrics = result['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_results(self) -> None:
        """Plot training results and comparisons."""
        if not self.results:
            print("No results to plot. Run train_models() first.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FTSE 100 Market Prediction Results', fontsize=16)
        
        # 1. Model Accuracy Comparison
        model_names = []
        accuracies = []
        
        for model_name, result in self.results.items():
            if result is not None:
                model_names.append(model_name.replace('_', ' ').title())
                accuracies.append(result['metrics']['accuracy'])
        
        axes[0, 0].bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Confusion Matrix for best model
        best_model = max(self.results.items(), 
                        key=lambda x: x[1]['metrics']['accuracy'] if x[1] else 0)
        
        if best_model[1] is not None:
            cm = best_model[1]['metrics']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                       ax=axes[0, 1])
            axes[0, 1].set_title(f'Confusion Matrix - {best_model[0].replace("_", " ").title()}')
        
        # 3. Feature importance (if available)
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                top_features = np.argsort(importances)[-10:]  # Top 10 features
                
                axes[1, 0].barh(range(len(top_features)), importances[top_features])
                axes[1, 0].set_yticks(range(len(top_features)))
                axes[1, 0].set_yticklabels([self.feature_names[i] for i in top_features])
                axes[1, 0].set_title('Top 10 Feature Importances (Random Forest)')
                axes[1, 0].set_xlabel('Importance')
        
        # 4. Prediction probabilities distribution
        if best_model[1] is not None:
            probabilities = best_model[1]['probabilities']
            axes[1, 1].hist(probabilities[:, 1], bins=20, alpha=0.7, color='skyblue')
            axes[1, 1].set_title(f'Prediction Probabilities - {best_model[0].replace("_", " ").title()}')
            axes[1, 1].set_xlabel('Probability of Up')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, save_path: str = "results/") -> None:
        """Save training results to files."""
        if not self.results:
            print("No results to save. Run train_models() first.")
            return
        
        # Save comparison table
        comparison_df = self.compare_models()
        comparison_df.to_csv(f"{save_path}model_comparison.csv", index=False)
        
        # Save individual model results
        for model_name, result in self.results.items():
            if result is not None:
                save_model_results(
                    model_name=model_name,
                    metrics=result['metrics'],
                    feature_names=self.feature_names,
                    save_path=save_path
                )
        
        print(f"Results saved to {save_path}")
    
    def predict_latest(self, model_name: str = None) -> Dict:
        """
        Make prediction for the latest market data.
        
        Args:
            model_name: Name of the model to use (if None, uses best model)
        
        Returns:
            Dictionary with prediction results
        """
        if not self.models:
            print("No models available. Run train_models() first.")
            return {}
        
        # Select model
        if model_name is None:
            # Use best performing model
            best_model = max(self.results.items(), 
                           key=lambda x: x[1]['metrics']['accuracy'] if x[1] else 0)
            model_name = best_model[0]
            model = best_model[1]['model']
        else:
            if model_name not in self.models:
                print(f"Model '{model_name}' not found.")
                return {}
            model = self.models[model_name]
        
        # Get latest features
        if self.features is None:
            print("No features available. Run prepare_data() first.")
            return {}
        
        latest_features = self.features.iloc[-1:].values
        
        # Make prediction
        try:
            prediction = model.predict(latest_features)[0]
            probability = model.predict_proba(latest_features)[0]
            
            result = {
                'model': model_name,
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': max(probability),
                'probabilities': {
                    'DOWN': probability[0],
                    'UP': probability[1]
                },
                'timestamp': pd.Timestamp.now(),
                'latest_price': self.data['Close'].iloc[-1] if self.data is not None else None
            }
            
            print(f"\nLatest Prediction ({model_name}):")
            print(f"  Direction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Probabilities: DOWN {result['probabilities']['DOWN']:.2%}, UP {result['probabilities']['UP']:.2%}")
            
            return result
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return {}

def run_training_pipeline() -> ModelTrainer:
    """
    Run the complete training pipeline.
    
    Returns:
        Trained ModelTrainer instance
    """
    print("FTSE 100 Quantum Machine Learning Pipeline")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    features, labels, feature_names = trainer.prepare_data(threshold=0.0)
    
    # Train models
    results = trainer.train_models(features, labels)
    
    # Compare models
    comparison = trainer.compare_models()
    
    # Plot results
    trainer.plot_results()
    
    # Save results
    trainer.save_results()
    
    # Make latest prediction
    latest_prediction = trainer.predict_latest()
    
    return trainer

if __name__ == "__main__":
    # Run the complete pipeline
    trainer = run_training_pipeline()
