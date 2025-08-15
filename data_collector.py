"""
Data collection and preprocessing for FTSE 100 market prediction.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils import validate_data_quality

class FTSE100DataCollector:
    """
    Collects and preprocesses FTSE 100 market data from Yahoo Finance.
    """
    
    def __init__(self, symbol: str = "^FTSE", period: str = "5y"):
        """
        Initialize the data collector.
        
        Args:
            symbol: Stock symbol (default: ^FTSE for FTSE 100)
            period: Data period (default: 5y for 5 years)
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.ticker = None
        
    def fetch_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch FTSE 100 data from Yahoo Finance.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
        
        Returns:
            DataFrame with market data
        """
        print(f"Fetching {self.symbol} data...")
        
        try:
            # Initialize ticker
            self.ticker = yf.Ticker(self.symbol)
            
            # Fetch data
            if start_date and end_date:
                self.data = self.ticker.history(start=start_date, end=end_date)
            else:
                self.data = self.ticker.history(period=self.period)
            
            # Basic data cleaning
            self._clean_data()
            
            print(f"Successfully fetched {len(self.data)} data points")
            print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def _clean_data(self) -> None:
        """
        Clean and preprocess the raw data.
        """
        if self.data is None:
            return
        
        # Remove rows with missing values
        initial_rows = len(self.data)
        self.data = self.data.dropna()
        
        # Remove duplicate dates
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        
        # Sort by date
        self.data = self.data.sort_index()
        
        # Keep only essential columns
        essential_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in essential_columns if col in self.data.columns]
        self.data = self.data[available_columns]
        
        print(f"Data cleaning: {initial_rows} -> {len(self.data)} rows")
    
    def get_data_info(self) -> dict:
        """
        Get information about the collected data.
        
        Returns:
            Dictionary with data information
        """
        if self.data is None:
            return {"error": "No data available"}
        
        info = {
            'symbol': self.symbol,
            'total_rows': len(self.data),
            'date_range': f"{self.data.index.min()} to {self.data.index.max()}",
            'columns': list(self.data.columns),
            'data_types': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'price_range': {
                'min_close': self.data['Close'].min(),
                'max_close': self.data['Close'].max(),
                'current_close': self.data['Close'].iloc[-1]
            }
        }
        
        return info
    
    def get_recent_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get the most recent data points.
        
        Args:
            days: Number of recent days to retrieve
        
        Returns:
            DataFrame with recent data
        """
        if self.data is None:
            return None
        
        return self.data.tail(days)
    
    def save_data(self, filename: str = None) -> str:
        """
        Save the collected data to a CSV file.
        
        Args:
            filename: Output filename (optional)
        
        Returns:
            Path to saved file
        """
        if self.data is None:
            print("No data to save")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ftse100_data_{timestamp}.csv"
        
        self.data.to_csv(filename)
        print(f"Data saved to: {filename}")
        return filename
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filename: Path to CSV file
        
        Returns:
            DataFrame with loaded data
        """
        try:
            self.data = pd.read_csv(filename, index_col=0, parse_dates=True)
            print(f"Data loaded from: {filename}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

def collect_sample_data() -> pd.DataFrame:
    """
    Collect sample FTSE 100 data for demonstration.
    
    Returns:
        DataFrame with sample data
    """
    collector = FTSE100DataCollector()
    data = collector.fetch_data()
    
    if data is not None:
        # Print data quality report
        quality_report = validate_data_quality(data)
        print("\nData Quality Report:")
        for key, value in quality_report.items():
            print(f"  {key}: {value}")
    
    return data

def get_market_summary() -> dict:
    """
    Get a summary of current market conditions.
    
    Returns:
        Dictionary with market summary
    """
    try:
        ticker = yf.Ticker("^FTSE")
        info = ticker.info
        
        # Get recent data
        recent_data = ticker.history(period="1mo")
        
        summary = {
            'current_price': recent_data['Close'].iloc[-1],
            'previous_close': recent_data['Close'].iloc[-2],
            'daily_change': recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[-2],
            'daily_change_pct': ((recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[-2]) - 1) * 100,
            'volume': recent_data['Volume'].iloc[-1],
            'high_52w': info.get('fiftyTwoWeekHigh', 'N/A'),
            'low_52w': info.get('fiftyTwoWeekLow', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A')
        }
        
        return summary
        
    except Exception as e:
        print(f"Error getting market summary: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    print("FTSE 100 Data Collector")
    print("=" * 50)
    
    # Collect sample data
    data = collect_sample_data()
    
    if data is not None:
        # Get market summary
        summary = get_market_summary()
        print("\nMarket Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Save data
        collector = FTSE100DataCollector()
        collector.data = data
        collector.save_data()
