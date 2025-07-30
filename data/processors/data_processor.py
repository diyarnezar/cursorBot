"""
Data Processor for Real-time Data Processing and Feature Engineering
Part of Project Hyperion - Ultimate Autonomous Trading Bot
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Advanced Data Processor for Real-time Data Processing
    
    Features:
    - Real-time data cleaning and validation
    - Feature engineering pipeline
    - Data normalization and scaling
    - Outlier detection and handling
    - Missing data imputation
    - Data quality monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_buffers = {}
        self.feature_pipelines = {}
        self.scalers = {}
        self.quality_metrics = {}
        self.processing_stats = {}
        
        # Processing parameters
        self.buffer_size = config.get('buffer_size', 10000)
        self.quality_threshold = config.get('quality_threshold', 0.95)
        self.outlier_threshold = config.get('outlier_threshold', 3.0)
        self.missing_data_strategy = config.get('missing_data_strategy', 'forward_fill')
        
        logger.info("Data Processor initialized")

    def create_data_buffer(self, symbol: str, max_size: int = None) -> deque:
        """Create a data buffer for a symbol"""
        if max_size is None:
            max_size = self.buffer_size
        
        self.data_buffers[symbol] = deque(maxlen=max_size)
        logger.info(f"Created data buffer for {symbol} with max size {max_size}")
        return self.data_buffers[symbol]

    def add_data(self, symbol: str, data: Dict[str, Any]):
        """Add data to the buffer for a symbol"""
        if symbol not in self.data_buffers:
            self.create_data_buffer(symbol)
        
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now()
        
        self.data_buffers[symbol].append(data)
        
        # Update processing stats
        if symbol not in self.processing_stats:
            self.processing_stats[symbol] = {'total_records': 0, 'processed_records': 0}
        
        self.processing_stats[symbol]['total_records'] += 1

    def get_buffer_data(self, symbol: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get data from buffer for a symbol"""
        if symbol not in self.data_buffers:
            return []
        
        buffer_data = list(self.data_buffers[symbol])
        
        if limit:
            buffer_data = buffer_data[-limit:]
        
        return buffer_data

    def convert_to_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of dictionaries to DataFrame"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime if it's a string
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df

    def clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate data"""
        if df.empty:
            return df
        
        original_length = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Validate data types
        df = self._validate_data_types(df)
        
        cleaned_length = len(df)
        removed_count = original_length - cleaned_length
        
        if removed_count > 0:
            logger.info(f"Cleaned {symbol} data: removed {removed_count} records")
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data"""
        if df.empty:
            return df
        
        # Count missing values
        missing_counts = df.isnull().sum()
        
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Apply missing data strategy
        if self.missing_data_strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        elif self.missing_data_strategy == 'backward_fill':
            df = df.fillna(method='bfill')
        elif self.missing_data_strategy == 'interpolate':
            df = df.interpolate()
        elif self.missing_data_strategy == 'drop':
            df = df.dropna()
        else:
            # Default to forward fill
            df = df.fillna(method='ffill')
        
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using statistical methods"""
        if df.empty:
            return df
        
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return df
        
        # Calculate z-scores for outlier detection
        z_scores = np.abs((df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std())
        
        # Create outlier mask
        outlier_mask = (z_scores > self.outlier_threshold).any(axis=1)
        
        # Remove outliers
        df_clean = df[~outlier_mask]
        
        removed_count = len(df) - len(df_clean)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outliers")
        
        return df_clean

    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types"""
        if df.empty:
            return df
        
        # Expected data types for common columns
        expected_types = {
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64',
            'price': 'float64',
            'quantity': 'float64'
        }
        
        for col, expected_type in expected_types.items():
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to numeric: {e}")
        
        return df

    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical features"""
        if df.empty or 'close' not in df.columns:
            return df
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Price relative to moving averages
        df['price_vs_sma_5'] = df['close'] / df['sma_5'] - 1
        df['price_vs_sma_20'] = df['close'] / df['sma_20'] - 1
        df['price_vs_sma_50'] = df['close'] / df['sma_50'] - 1
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        return df

    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical features"""
        if df.empty or 'close' not in df.columns:
            return df
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        if 'high' in df.columns and 'low' in df.columns:
            df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df['high'], df['low'], df['close'])
        
        # ATR (Average True Range)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
        
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ATR (Average True Range)"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def normalize_features(self, df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """Normalize features using specified method"""
        if df.empty:
            return df
        
        # Select numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return df
        
        df_normalized = df.copy()
        
        if method == 'minmax':
            # Min-Max normalization
            for col in numeric_columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            # Z-score normalization
            for col in numeric_columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
        
        elif method == 'robust':
            # Robust scaling using median and IQR
            for col in numeric_columns:
                median_val = df[col].median()
                q75 = df[col].quantile(0.75)
                q25 = df[col].quantile(0.25)
                iqr = q75 - q25
                if iqr > 0:
                    df_normalized[col] = (df[col] - median_val) / iqr
        
        return df_normalized

    def create_feature_pipeline(self, symbol: str, features: List[str]) -> Dict[str, Callable]:
        """Create a feature engineering pipeline for a symbol"""
        pipeline = {}
        
        for feature in features:
            if feature == 'basic':
                pipeline[feature] = self.calculate_basic_features
            elif feature == 'advanced':
                pipeline[feature] = self.calculate_advanced_features
            elif feature == 'normalize':
                pipeline[feature] = lambda df: self.normalize_features(df, 'minmax')
            elif feature == 'clean':
                pipeline[feature] = lambda df: self.clean_data(df, symbol)
        
        self.feature_pipelines[symbol] = pipeline
        logger.info(f"Created feature pipeline for {symbol} with {len(pipeline)} steps")
        
        return pipeline

    def process_data(self, symbol: str, pipeline_steps: List[str] = None) -> pd.DataFrame:
        """Process data using the feature pipeline"""
        if symbol not in self.data_buffers:
            logger.warning(f"No data buffer found for {symbol}")
            return pd.DataFrame()
        
        # Get data from buffer
        buffer_data = self.get_buffer_data(symbol)
        
        if not buffer_data:
            logger.warning(f"No data in buffer for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = self.convert_to_dataframe(buffer_data)
        
        if df.empty:
            return df
        
        # Apply pipeline steps
        if pipeline_steps is None:
            pipeline_steps = ['clean', 'basic', 'advanced']
        
        if symbol in self.feature_pipelines:
            pipeline = self.feature_pipelines[symbol]
            
            for step in pipeline_steps:
                if step in pipeline:
                    df = pipeline[step](df)
        
        # Update processing stats
        if symbol in self.processing_stats:
            self.processing_stats[symbol]['processed_records'] = len(df)
        
        return df

    def monitor_data_quality(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Monitor data quality metrics"""
        if df.empty:
            return {}
        
        quality_metrics = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'total_records': len(df),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_records': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns)
        }
        
        # Check for outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            z_scores = np.abs((df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std())
            outlier_count = (z_scores > self.outlier_threshold).sum().sum()
            quality_metrics['outlier_count'] = outlier_count
            quality_metrics['outlier_percentage'] = (outlier_count / (len(df) * len(numeric_columns))) * 100
        
        # Calculate overall quality score
        quality_score = 1.0
        quality_score -= quality_metrics['missing_percentage'] / 100
        quality_score -= quality_metrics['duplicate_percentage'] / 100
        if 'outlier_percentage' in quality_metrics:
            quality_score -= quality_metrics['outlier_percentage'] / 100
        
        quality_metrics['quality_score'] = max(0, quality_score)
        quality_metrics['quality_status'] = 'good' if quality_score >= self.quality_threshold else 'poor'
        
        # Store quality metrics
        self.quality_metrics[symbol] = quality_metrics
        
        return quality_metrics

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics"""
        return self.quality_metrics

    def save_processed_data(self, df: pd.DataFrame, filepath: str, format: str = 'parquet'):
        """Save processed data to file"""
        try:
            if format == 'parquet':
                df.to_parquet(filepath)
            elif format == 'csv':
                df.to_csv(filepath, index=True)
            elif format == 'json':
                df.to_json(filepath, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved processed data to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")

    def load_processed_data(self, filepath: str, format: str = 'parquet') -> pd.DataFrame:
        """Load processed data from file"""
        try:
            if format == 'parquet':
                df = pd.read_parquet(filepath)
            elif format == 'csv':
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif format == 'json':
                df = pd.read_json(filepath, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Loaded processed data from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            return pd.DataFrame() 