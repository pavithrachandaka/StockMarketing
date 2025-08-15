"""
Feature engineering for FTSE 100 market prediction.
Calculates technical indicators and prepares features for quantum machine learning.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Calculates technical indicators and prepares features for market prediction.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the feature engineer.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.features = None
        self.feature_names = []
        
    def calculate_moving_averages(self, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Calculate moving averages for different time windows.
        
        Args:
            windows: List of window sizes for moving averages
        
        Returns:
            DataFrame with moving average features
        """
        features = pd.DataFrame(index=self.data.index)
        
        for window in windows:
            # Simple Moving Average
            features[f'sma_{window}'] = self.data['Close'].rolling(window=window).mean()
            
            # Exponential Moving Average
            features[f'ema_{window}'] = self.data['Close'].ewm(span=window).mean()
            
            # Price relative to moving average
            features[f'price_sma_{window}_ratio'] = self.data['Close'] / features[f'sma_{window}']
            features[f'price_ema_{window}_ratio'] = self.data['Close'] / features[f'ema_{window}']
            
            self.feature_names.extend([
                f'sma_{window}', f'ema_{window}', 
                f'price_sma_{window}_ratio', f'price_ema_{window}_ratio'
            ])
        
        return features
    
    def calculate_volatility_features(self, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate volatility-based features.
        
        Args:
            windows: List of window sizes for volatility calculation
        
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=self.data.index)
        
        # Calculate returns
        returns = self.data['Close'].pct_change()
        
        for window in windows:
            # Rolling volatility (standard deviation of returns)
            features[f'volatility_{window}'] = returns.rolling(window=window).std()
            
            # Rolling variance
            features[f'variance_{window}'] = returns.rolling(window=window).var()
            
            # True Range (for intraday volatility)
            high_low = self.data['High'] - self.data['Low']
            high_close = np.abs(self.data['High'] - self.data['Close'].shift())
            low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features[f'true_range_{window}'] = true_range.rolling(window=window).mean()
            
            self.feature_names.extend([
                f'volatility_{window}', f'variance_{window}', f'true_range_{window}'
            ])
        
        return features
    
    def calculate_momentum_features(self) -> pd.DataFrame:
        """
        Calculate momentum-based features.
        
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=self.data.index)
        
        # Price changes over different periods
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}d'] = self.data['Close'].pct_change(period)
            features[f'log_return_{period}d'] = np.log(self.data['Close'] / self.data['Close'].shift(period))
            
            self.feature_names.extend([f'return_{period}d', f'log_return_{period}d'])
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = ((self.data['Close'] - self.data['Close'].shift(period)) / 
                                       self.data['Close'].shift(period)) * 100
            
            self.feature_names.append(f'roc_{period}')
        
        return features
    
    def calculate_technical_indicators(self) -> pd.DataFrame:
        """
        Calculate common technical indicators.
        
        Returns:
            DataFrame with technical indicators
        """
        features = pd.DataFrame(index=self.data.index)
        
        # Relative Strength Index (RSI)
        features['rsi_14'] = self._calculate_rsi(14)
        features['rsi_21'] = self._calculate_rsi(21)
        
        # MACD
        macd_line, signal_line, macd_histogram = self._calculate_macd()
        features['macd_line'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = macd_histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands()
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features['bb_position'] = (self.data['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic()
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        
        # Williams %R
        features['williams_r'] = self._calculate_williams_r()
        
        # Average True Range (ATR)
        features['atr_14'] = self._calculate_atr(14)
        
        self.feature_names.extend([
            'rsi_14', 'rsi_21', 'macd_line', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'stoch_k', 'stoch_d', 'williams_r', 'atr_14'
        ])
        
        return features
    
    def _calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = self.data['Close'].ewm(span=fast).mean()
        ema_slow = self.data['Close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram
    
    def _calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2) -> Tuple:
        """Calculate Bollinger Bands."""
        sma = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _calculate_stochastic(self, k_period: int = 14, d_period: int = 3) -> Tuple:
        """Calculate Stochastic Oscillator."""
        lowest_low = self.data['Low'].rolling(window=k_period).min()
        highest_high = self.data['High'].rolling(window=k_period).max()
        k_percent = 100 * ((self.data['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = self.data['High'].rolling(window=period).max()
        lowest_low = self.data['Low'].rolling(window=period).min()
        williams_r = -100 * ((highest_high - self.data['Close']) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_volume_features(self) -> pd.DataFrame:
        """
        Calculate volume-based features.
        
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=self.data.index)
        
        # Volume moving averages
        for window in [5, 10, 20]:
            features[f'volume_sma_{window}'] = self.data['Volume'].rolling(window=window).mean()
            features[f'volume_ratio_{window}'] = self.data['Volume'] / features[f'volume_sma_{window}']
            
            self.feature_names.extend([f'volume_sma_{window}', f'volume_ratio_{window}'])
        
        # Volume-price trend
        features['volume_price_trend'] = (self.data['Volume'] * 
                                        self.data['Close'].pct_change()).cumsum()
        
        # On-Balance Volume (OBV)
        features['obv'] = self._calculate_obv()
        
        self.feature_names.extend(['volume_price_trend', 'obv'])
        
        return features
    
    def _calculate_obv(self) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=self.data.index, dtype=float)
        obv.iloc[0] = self.data['Volume'].iloc[0]
        
        for i in range(1, len(self.data)):
            if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + self.data['Volume'].iloc[i]
            elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - self.data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def create_all_features(self) -> pd.DataFrame:
        """
        Create all feature categories and combine them.
        
        Returns:
            DataFrame with all features
        """
        print("Creating features...")
        
        # Calculate all feature categories
        ma_features = self.calculate_moving_averages()
        vol_features = self.calculate_volatility_features()
        mom_features = self.calculate_momentum_features()
        tech_features = self.calculate_technical_indicators()
        vol_features_2 = self.calculate_volume_features()
        
        # Combine all features
        all_features = pd.concat([
            ma_features, vol_features, mom_features, tech_features, vol_features_2
        ], axis=1)
        
        # Remove any remaining NaN values
        all_features = all_features.dropna()
        
        # Align with original data
        self.data = self.data.loc[all_features.index]
        
        self.features = all_features
        self.feature_names = list(all_features.columns)
        
        print(f"Created {len(self.feature_names)} features")
        print(f"Final dataset shape: {self.features.shape}")
        
        return self.features
    
    def get_feature_summary(self) -> dict:
        """
        Get summary statistics for all features.
        
        Returns:
            Dictionary with feature summary
        """
        if self.features is None:
            return {"error": "No features available"}
        
        summary = {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'data_shape': self.features.shape,
            'missing_values': self.features.isnull().sum().sum(),
            'feature_categories': {
                'Moving Averages': len([f for f in self.feature_names if 'sma' in f or 'ema' in f]),
                'Volatility': len([f for f in self.feature_names if 'volatility' in f or 'variance' in f]),
                'Momentum': len([f for f in self.feature_names if 'return' in f or 'roc' in f]),
                'Technical Indicators': len([f for f in self.feature_names if f in ['rsi_14', 'rsi_21', 'macd_line', 'macd_signal', 'macd_histogram', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position', 'stoch_k', 'stoch_d', 'williams_r', 'atr_14']]),
                'Volume': len([f for f in self.feature_names if 'volume' in f or 'obv' in f])
            }
        }
        
        return summary
    
    def select_features(self, method: str = 'correlation', threshold: float = 0.8) -> List[str]:
        """
        Select the most important features.
        
        Args:
            method: Feature selection method ('correlation', 'variance', 'all')
            threshold: Threshold for feature selection
        
        Returns:
            List of selected feature names
        """
        if self.features is None:
            return []
        
        if method == 'all':
            return self.feature_names
        
        elif method == 'correlation':
            # Remove highly correlated features
            corr_matrix = self.features.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            selected_features = [f for f in self.feature_names if f not in to_drop]
            
            print(f"Feature selection: {len(self.feature_names)} -> {len(selected_features)} features")
            return selected_features
        
        elif method == 'variance':
            # Remove low variance features
            variances = self.features.var()
            selected_features = variances[variances > threshold].index.tolist()
            
            print(f"Feature selection: {len(self.feature_names)} -> {len(selected_features)} features")
            return selected_features
        
        else:
            return self.feature_names

def create_sample_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create sample features for demonstration.
    
    Args:
        data: Input OHLCV data
    
    Returns:
        Tuple of (features DataFrame, feature names list)
    """
    engineer = FeatureEngineer(data)
    features = engineer.create_all_features()
    feature_names = engineer.feature_names
    
    # Print feature summary
    summary = engineer.get_feature_summary()
    print("\nFeature Summary:")
    for key, value in summary.items():
        if key != 'feature_names':
            print(f"  {key}: {value}")
    
    return features, feature_names
