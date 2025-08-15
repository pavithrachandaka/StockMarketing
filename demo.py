"""
Demo script for FTSE 100 Quantum Machine Learning Prediction
Quick test of the system without the full dashboard.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_collector import collect_sample_data
from feature_engineering import create_sample_features
from model_trainer import ModelTrainer

def run_demo():
    """Run a quick demo of the QML system."""
    
    print("🚀 FTSE 100 Quantum ML Demo")
    print("=" * 50)
    
    # Step 1: Collect data
    print("\n📊 Step 1: Collecting FTSE 100 data...")
    data = collect_sample_data()
    
    if data is None:
        print("❌ Failed to collect data. Exiting.")
        return
    
    print(f"✅ Collected {len(data)} data points")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    print(f"   Current price: £{data['Close'].iloc[-1]:.2f}")
    
    # Step 2: Create features
    print("\n🔧 Step 2: Creating features...")
    features, feature_names = create_sample_features(data)
    
    if features is None:
        print("❌ Failed to create features. Exiting.")
        return
    
    print(f"✅ Created {len(feature_names)} features")
    print(f"   Feature categories: Moving Averages, Volatility, Momentum, Technical Indicators, Volume")
    
    # Step 3: Train models
    print("\n🤖 Step 3: Training models...")
    
    trainer = ModelTrainer(data)
    trainer.features = features
    trainer.feature_names = feature_names
    
    # Prepare data
    features_array, labels, feature_names = trainer.prepare_data(threshold=0.0)
    
    print(f"✅ Prepared dataset: {len(features_array)} samples, {len(feature_names)} features")
    print(f"   Label distribution: {np.sum(labels)} UP, {len(labels) - np.sum(labels)} DOWN")
    
    # Train models (with reduced iterations for demo)
    try:
        results = trainer.train_models(features_array, labels)
        
        if results:
            print("\n📊 Model Results:")
            print("-" * 30)
            
            for model_name, result in results.items():
                if result is not None:
                    metrics = result['metrics']
                    print(f"{model_name.replace('_', ' ').title():<25} "
                          f"Accuracy: {metrics['accuracy']:.3f}")
            
            # Find best model
            best_model = max(results.items(), 
                           key=lambda x: x[1]['metrics']['accuracy'] if x[1] else 0)
            
            print(f"\n🏆 Best Model: {best_model[0].replace('_', ' ').title()}")
            print(f"   Accuracy: {best_model[1]['metrics']['accuracy']:.3f}")
            
            # Make prediction
            print("\n🔮 Making prediction...")
            prediction_result = trainer.predict_latest()
            
            if prediction_result:
                print(f"📈 Prediction: {prediction_result['prediction']}")
                print(f"   Confidence: {prediction_result['confidence']:.1%}")
                print(f"   Probabilities: UP {prediction_result['probabilities']['UP']:.1%}, "
                      f"DOWN {prediction_result['probabilities']['DOWN']:.1%}")
                print(f"   Model: {prediction_result['model']}")
                print(f"   Timestamp: {prediction_result['timestamp']}")
            
        else:
            print("❌ No models were successfully trained.")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("💡 Run 'streamlit run dashboard.py' for the full interactive experience")
    print("⚠️  Remember: This is for educational purposes only!")

if __name__ == "__main__":
    run_demo()
