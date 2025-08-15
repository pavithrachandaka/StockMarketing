# Getting Started with FTSE 100 Quantum ML Predictor

## 🚀 Quick Start Guide

This guide will help you get the Quantum Machine Learning FTSE 100 prediction system up and running.

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection (for data fetching)
- At least 4GB RAM (for quantum simulations)

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The installation may take a few minutes due to Qiskit and its dependencies.

### 2. Verify Installation

```bash
python -c "import qiskit; print('Qiskit version:', qiskit.__version__)"
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
```

## 🎯 Running the Application

### Option 1: Quick Demo (Recommended for first run)

```bash
python demo.py
```

This will:
- Fetch FTSE 100 data
- Create features
- Train quantum and classical models
- Make a prediction
- Show results in the console

### Option 2: Interactive Dashboard

```bash
streamlit run dashboard.py
```

This will:
- Open a web browser with the interactive dashboard
- Allow you to control all parameters
- Provide real-time visualizations
- Make predictions interactively

### Option 3: Full Training Pipeline

```bash
python model_trainer.py
```

This will:
- Run the complete training pipeline
- Generate detailed reports and plots
- Save results to files

## 📊 Understanding the Output

### Model Performance
- **Accuracy**: Percentage of correct predictions
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all positive cases
- **F1-Score**: Harmonic mean of precision and recall

### Prediction Results
- **UP/DOWN**: Market direction prediction
- **Confidence**: Model's confidence in the prediction
- **Probabilities**: Detailed probability breakdown

## 🔧 Configuration Options

### Data Parameters
- **Data Period**: 1y, 2y, 5y, 10y (default: 5y)
- **Prediction Threshold**: Minimum change to consider as "up" movement

### Model Parameters
- **Number of Qubits**: 2-4 (default: 4)
- **Training Iterations**: 10-100 (default: 50)
- **Feature Selection**: All features or correlation-based selection

## 🎓 Learning Path

### Beginner Level
1. Run `demo.py` to see the system in action
2. Explore the dashboard to understand the interface
3. Read the README.md for project overview

### Intermediate Level
1. Modify feature engineering in `feature_engineering.py`
2. Experiment with different quantum circuits in `quantum_model.py`
3. Try different classical models in `model_trainer.py`

### Advanced Level
1. Implement custom quantum feature maps
2. Optimize quantum circuit parameters
3. Add new technical indicators
4. Implement ensemble methods

## 🐛 Troubleshooting

### Common Issues

**1. Installation Problems**
```bash
# If you get dependency conflicts
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**2. Qiskit Installation Issues**
```bash
# Install Qiskit separately
pip install qiskit qiskit-machine-learning qiskit-aer
```

**3. Data Fetching Issues**
- Check your internet connection
- Verify Yahoo Finance is accessible
- Try using a VPN if needed

**4. Memory Issues**
- Reduce the number of features
- Use smaller data periods
- Reduce training iterations

**5. Quantum Simulation Issues**
- Ensure you have enough RAM (4GB+)
- Reduce the number of qubits
- Use fewer training iterations

### Error Messages

**"No module named 'qiskit'"**
```bash
pip install qiskit
```

**"Failed to collect data"**
- Check internet connection
- Try running again later
- Verify Yahoo Finance API access

**"Training failed"**
- Reduce training iterations
- Check available memory
- Try with fewer features

## 📈 Expected Performance

### Model Accuracy
- **Random Forest**: 55-65%
- **SVM**: 50-60%
- **VQC**: 45-55%
- **Hybrid**: 55-65%

*Note: Financial predictions are inherently difficult. These are educational results.*

### Training Time
- **Demo**: 2-5 minutes
- **Full Training**: 10-30 minutes
- **Dashboard**: 1-3 minutes (with caching)

## 🔬 Understanding Quantum ML

### Key Concepts
1. **Qubits**: Quantum bits that can be in superposition
2. **Feature Map**: Encodes classical data into quantum states
3. **Variational Circuit**: Trainable quantum operations
4. **Hybrid Optimization**: Classical optimizer with quantum evaluation

### Why Quantum ML?
- **Feature Space**: Quantum computers can explore larger feature spaces
- **Non-linearity**: Quantum circuits can capture complex patterns
- **Future Potential**: As quantum hardware improves, so will performance

## 📚 Next Steps

### Enhancements
1. **More Features**: Add sentiment analysis, economic indicators
2. **Better Models**: Implement quantum neural networks
3. **Real-time Data**: Connect to live market feeds
4. **Portfolio Optimization**: Extend to multi-asset prediction

### Research Areas
1. **Quantum Feature Selection**: Use quantum algorithms for feature selection
2. **Quantum Ensembles**: Combine multiple quantum models
3. **Error Mitigation**: Implement quantum error correction
4. **Hardware Optimization**: Optimize for specific quantum hardware

## ⚠️ Important Notes

### Educational Purpose
- This project is for **educational purposes only**
- **Do not use for actual investment decisions**
- Always consult financial professionals

### Limitations
- Quantum computers are still in development
- Current simulations are limited
- Financial markets are inherently unpredictable
- Past performance doesn't guarantee future results

### Data Disclaimer
- Data comes from Yahoo Finance
- May have delays or inaccuracies
- Use for educational purposes only

## 🤝 Getting Help

### Documentation
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Yahoo Finance API](https://finance.yahoo.com/)

### Community
- [Qiskit Community](https://qiskit.org/ecosystem/)
- [Streamlit Community](https://discuss.streamlit.io/)

## 🎉 Congratulations!

You've successfully set up a Quantum Machine Learning system for financial prediction! 

Remember:
- Start with the demo
- Experiment with parameters
- Learn from the code
- Have fun exploring quantum computing!

---

**Happy Quantum Computing! 🚀**
