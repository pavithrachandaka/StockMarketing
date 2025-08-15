# Quantum Machine Learning for FTSE 100 Market Prediction

This project demonstrates the use of Quantum Machine Learning (QML) to predict whether the FTSE 100 index will go up or down using a Variational Quantum Classifier (VQC).

## 🎯 Problem Statement

Predict whether the FTSE 100 index (UK's top 100 companies) will go up (📈) or down (📉) in the future using quantum computing techniques.

## 🚀 Features

- **Quantum Machine Learning**: Uses Variational Quantum Classifier (VQC) for predictions
- **Real-time Data**: Fetches live FTSE 100 data using Yahoo Finance API
- **Feature Engineering**: Calculates technical indicators (moving averages, volatility, etc.)
- **Interactive Dashboard**: Streamlit web interface for easy predictions
- **Model Evaluation**: Comprehensive metrics and visualizations

## 🛠️ Technology Stack

- **Python**: Main programming language
- **Qiskit**: Quantum computing framework
- **yfinance**: Financial data fetching
- **Streamlit**: Web dashboard
- **Pandas/NumPy**: Data processing
- **Matplotlib/Seaborn**: Visualizations

## 📁 Project Structure

```
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── data_collector.py        # Data fetching and preprocessing
├── feature_engineering.py   # Technical indicators calculation
├── quantum_model.py         # VQC implementation
├── model_trainer.py         # Training and evaluation
├── dashboard.py             # Streamlit web interface
└── utils.py                 # Utility functions
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

3. **Train the Model**:
   ```bash
   python model_trainer.py
   ```

## 🔬 How It Works

### 1. Data Collection
- Fetches historical FTSE 100 data (^FTSE) from Yahoo Finance
- Collects 5+ years of daily closing prices
- Handles missing data and outliers

### 2. Feature Engineering
- **Moving Averages**: 10-day and 50-day trends
- **Volatility**: Rolling standard deviation
- **Price Changes**: Daily and weekly percentage changes
- **Technical Indicators**: RSI, MACD, Bollinger Bands

### 3. Quantum Model
- **Feature Map**: Encodes classical data into quantum states
- **Variational Circuit**: Trainable quantum gates
- **Hybrid Optimization**: Classical optimizer with quantum evaluation

### 4. Prediction Pipeline
- Normalizes features to quantum range (0 to π)
- Runs quantum circuit for feature transformation
- Applies classical classifier for final prediction

## 📊 Model Performance

The VQC model typically achieves:
- **Accuracy**: 55-65% (better than random guessing)
- **Precision**: Varies based on market conditions
- **Recall**: Balanced for both up/down predictions

*Note: Financial predictions are inherently difficult and should not be used as sole investment advice.*

## 🎓 Learning Objectives

- Understanding quantum computing basics
- Implementing quantum machine learning
- Feature engineering for financial data
- Building hybrid classical-quantum systems
- Creating interactive dashboards

## ⚠️ Disclaimer

This project is for educational purposes only. Financial predictions are inherently uncertain and should not be used as investment advice. Always consult with financial professionals before making investment decisions.

## 🤝 Contributing

Feel free to contribute by:
- Improving feature engineering
- Optimizing quantum circuits
- Adding new technical indicators
- Enhancing the dashboard

## 📚 Resources

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Variational Quantum Classifier](https://qiskit.org/documentation/machine-learning/tutorials/01_vqc.html)
- [FTSE 100 Information](https://www.londonstockexchange.com/indices/ftse-100)
- [Yahoo Finance API](https://finance.yahoo.com/)
