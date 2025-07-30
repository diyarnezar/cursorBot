import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import os
import json

# Import crypto features
try:
    from .crypto_features import CryptoFeatures
    CRYPTO_FEATURES_AVAILABLE = True
except ImportError:
    CRYPTO_FEATURES_AVAILABLE = False
    logging.warning("Crypto features module not available. Crypto-specific features will be limited.")

# Try to import sklearn for preprocessing, with fallback
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Feature normalization will be limited.")

# Try to import talib for advanced indicators, with fallback
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Will use manual implementations of indicators.")

# Try to import shap for SHAP explainability, with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Explainability will be limited.")


class FeatureEngineer:
    """
    Base feature engineering class that provides fundamental technical indicators.
    """
    
    def __init__(self):
        """Initialize the base Feature Engineer."""
        pass
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to the DataFrame for maximum profitability.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        if df.empty:
            return df
        
        try:
            # --- COMPREHENSIVE MOVING AVERAGES ---
            # Simple Moving Averages (SMA)
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(3)  # Slope for trend strength
            
            # Exponential Moving Averages (EMA)
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff(3)
            
            # Weighted Moving Average (WMA)
            for period in [10, 20, 50]:
                weights = np.arange(1, period + 1)
                df[f'wma_{period}'] = df['close'].rolling(window=period).apply(
                    lambda x: np.dot(x, weights) / weights.sum(), raw=True
                )
            
            # Hull Moving Average (HMA) - Very responsive
            for period in [10, 20, 50]:
                wma_half = df['close'].rolling(window=period//2).apply(
                    lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
                )
                wma_full = df['close'].rolling(window=period).apply(
                    lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
                )
                raw_hma = 2 * wma_half - wma_full
                df[f'hma_{period}'] = raw_hma.rolling(window=int(np.sqrt(period))).mean()
            
            # Triple Exponential Moving Average (TEMA)
            for period in [10, 20, 50]:
                ema1 = df['close'].ewm(span=period, adjust=False).mean()
                ema2 = ema1.ewm(span=period, adjust=False).mean()
                ema3 = ema2.ewm(span=period, adjust=False).mean()
                df[f'tema_{period}'] = 3 * ema1 - 3 * ema2 + ema3
            
            # Kaufman Adaptive Moving Average (KAMA)
            for period in [10, 20, 50]:
                df[f'kama_{period}'] = self._calculate_kama(df['close'], period)
            
            # --- MOMENTUM INDICATORS ---
            # Relative Strength Index (RSI)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
            
            # Multiple RSI periods for different timeframes
            for period in [7, 14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100.0 - (100.0 / (1.0 + rs))

            # Stochastic Oscillator
            for period in [14, 21]:
                low_min = df['low'].rolling(window=period).min()
                high_max = df['high'].rolling(window=period).max()
                df[f'stochastic_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
                df[f'stochastic_d_{period}'] = df[f'stochastic_k_{period}'].rolling(window=3).mean()
            
            # Williams %R
            for period in [14, 21]:
                low_min = df['low'].rolling(window=period).min()
                high_max = df['high'].rolling(window=period).max()
                df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)

            # Moving Average Convergence Divergence (MACD)
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp12 - exp26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # MACD with different periods
            for fast, slow in [(8, 21), (12, 26), (21, 55)]:
                fast_ema = df['close'].ewm(span=fast, adjust=False).mean()
                slow_ema = df['close'].ewm(span=slow, adjust=False).mean()
                macd_line = fast_ema - slow_ema
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                df[f'macd_{fast}_{slow}'] = macd_line
                df[f'macd_signal_{fast}_{slow}'] = signal_line
                df[f'macd_hist_{fast}_{slow}'] = macd_line - signal_line

            # --- VOLATILITY INDICATORS ---
            # Bollinger Bands with multiple periods
            for period in [20, 50]:
                df[f'bollinger_mid_{period}'] = df['close'].rolling(window=period).mean()
                df[f'bollinger_std_{period}'] = df['close'].rolling(window=period).std()
                df[f'bollinger_upper_{period}'] = df[f'bollinger_mid_{period}'] + (df[f'bollinger_std_{period}'] * 2)
                df[f'bollinger_lower_{period}'] = df[f'bollinger_mid_{period}'] - (df[f'bollinger_std_{period}'] * 2)
                df[f'bollinger_width_{period}'] = (df[f'bollinger_upper_{period}'] - df[f'bollinger_lower_{period}']) / df[f'bollinger_mid_{period}']
                df[f'bollinger_position_{period}'] = (df['close'] - df[f'bollinger_lower_{period}']) / (df[f'bollinger_upper_{period}'] - df[f'bollinger_lower_{period}'])

            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            for period in [14, 20, 21, 50]:
                df[f'atr_{period}'] = tr.ewm(span=period, adjust=False).mean()

            # Keltner Channels
            for period in [20, 50]:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                atr = df[f'atr_{period}']
                df[f'keltner_upper_{period}'] = typical_price + (2 * atr)
                df[f'keltner_lower_{period}'] = typical_price - (2 * atr)
                df[f'keltner_width_{period}'] = (df[f'keltner_upper_{period}'] - df[f'keltner_lower_{period}']) / typical_price

            # --- TREND STRENGTH INDICATORS ---
            # Average Directional Index (ADX)
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            for period in [14, 21]:
                atr = df[f'atr_{period}']
                plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=0, adjust=False).mean() / atr)
                minus_di = 100 * (abs(minus_dm.ewm(alpha=1/period, min_periods=0, adjust=False).mean()) / atr)
                
                dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
                df[f'adx_{period}'] = dx.ewm(alpha=1/period, min_periods=0, adjust=False).mean()
                df[f'plus_di_{period}'] = plus_di
                df[f'minus_di_{period}'] = minus_di

            # Parabolic SAR
            df['psar'] = self._calculate_psar(df)
            
            # --- VOLUME INDICATORS ---
            # On-Balance Volume (OBV)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # Volume Weighted Average Price (VWAP)
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
            
            # Money Flow Index (MFI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            mfi_ratio = positive_flow / negative_flow
            df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
            # Chaikin Money Flow
            mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mfm = mfm.replace([np.inf, -np.inf], 0)
            mfv = mfm * df['volume']
            df['cmf'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()
            
            # --- ADVANCED MOMENTUM ---
            # Commodity Channel Index (CCI)
            for period in [14, 20]:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                sma_tp = typical_price.rolling(window=period).mean()
                mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
                df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # Rate of Change (ROC)
            for period in [10, 14, 20]:
                df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
            
            # Momentum - optimized to avoid DataFrame fragmentation
            momentum_features = {}
            for period in [10, 14, 20]:
                momentum_features[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            df = pd.concat([df, pd.DataFrame(momentum_features, index=df.index)], axis=1)
            
            # --- PRICE ACTION PATTERNS ---
            # Price action patterns - optimized to avoid DataFrame fragmentation
            body_size = abs(df['close'] - df['open'])
            total_range = df['high'] - df['low']
            lower_shadow = np.minimum(df['open'], df['close']) - df['low']
            upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
            
            pattern_features = {
                'doji': np.where(body_size <= (total_range * 0.1), 1, 0),
                'hammer': np.where((lower_shadow > 2 * body_size) & (upper_shadow < body_size), 1, 0),
                'shooting_star': np.where((upper_shadow > 2 * body_size) & (lower_shadow < body_size), 1, 0)
            }
            df = pd.concat([df, pd.DataFrame(pattern_features, index=df.index)], axis=1)
            
            # --- CROSSOVER SIGNALS ---
            # Crossover signals - optimized to avoid DataFrame fragmentation
            crossover_features = {
                'golden_cross': np.where((df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1)), 1, 0),
                'death_cross': np.where((df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1)), 1, 0)
            }
            
            # Price vs EMA crossovers
            for period in [20, 50, 200]:
                crossover_features[f'price_above_ema_{period}'] = np.where(df['close'] > df[f'ema_{period}'], 1, 0)
                crossover_features[f'price_cross_ema_{period}'] = np.where(
                    (df['close'] > df[f'ema_{period}']) & (df['close'].shift(1) <= df[f'ema_{period}'].shift(1)), 1, 0
                )
            
            df = pd.concat([df, pd.DataFrame(crossover_features, index=df.index)], axis=1)
            
            # --- VOLATILITY RATIOS ---
            # Historical volatility
            for period in [20, 50]:
                returns = df['close'].pct_change()
                df[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
            
            # Volatility ratio (current vs historical)
            df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']
            
            # --- SUPPORT/RESISTANCE ---
            # Pivot points
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['r1'] = 2 * df['pivot'] - df['low']
            df['s1'] = 2 * df['pivot'] - df['high']
            df['r2'] = df['pivot'] + (df['high'] - df['low'])
            df['s2'] = df['pivot'] - (df['high'] - df['low'])
            
            # Distance from support/resistance - optimized to avoid DataFrame fragmentation
            distance_features = {
                'distance_from_r1': (df['close'] - df['r1']) / df['close'],
                'distance_from_s1': (df['close'] - df['s1']) / df['close']
            }
            df = pd.concat([df, pd.DataFrame(distance_features, index=df.index)], axis=1)

            # For small datasets, fill NaN values instead of dropping rows
            if len(df) < 50:
                # Fill NaN values with forward fill then backward fill
                df = df.fillna(method='ffill').fillna(method='bfill')
                # Fill any remaining NaN with 0
                df = df.fillna(0)
            else:
                # Drop any rows with NaN values that were created by the rolling calculations
                df.dropna(inplace=True)
            
        except Exception as e:
            logging.error(f"Error during feature engineering: {e}", exc_info=True)
            return pd.DataFrame() 

        return df
    
    def _calculate_kama(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Kaufman Adaptive Moving Average (KAMA)
        
        Args:
            prices: Price series
            period: Period for KAMA calculation
            
        Returns:
            KAMA series
        """
        try:
            # Calculate change in price
            change = abs(prices - prices.shift(period))
            
            # Calculate volatility
            volatility = change.rolling(window=period).sum()
            
            # Calculate efficiency ratio
            efficiency_ratio = change / volatility
            efficiency_ratio = efficiency_ratio.fillna(0)
            
            # Calculate smoothing constant
            fast_constant = 2 / (2 + 1)  # 2/(2+1) = 0.6667
            slow_constant = 2 / (30 + 1)  # 2/(30+1) = 0.0645
            smoothing_constant = (efficiency_ratio * (fast_constant - slow_constant) + slow_constant) ** 2
            
            # Calculate KAMA
            kama = prices.copy()
            for i in range(1, len(prices)):
                kama.iloc[i] = kama.iloc[i-1] + smoothing_constant.iloc[i] * (prices.iloc[i] - kama.iloc[i-1])
            
            return kama
            
        except Exception as e:
            logging.error(f"Error calculating KAMA: {e}")
            return prices.rolling(window=period).mean()
    
    def _calculate_psar(self, df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """
        Calculate Parabolic SAR (Stop and Reverse)
        
        Args:
            df: DataFrame with high, low, close columns
            acceleration: Acceleration factor
            maximum: Maximum acceleration factor
            
        Returns:
            PSAR series
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            psar = np.zeros(len(df))
            af = acceleration  # Acceleration factor
            ep = low[0]  # Extreme point
            long = True  # Long position
            
            for i in range(1, len(df)):
                if long:
                    psar[i] = psar[i-1] + af * (ep - psar[i-1])
                    
                    if low[i] < psar[i]:
                        long = False
                        psar[i] = ep
                        ep = high[i]
                        af = acceleration
                    else:
                        if high[i] > ep:
                            ep = high[i]
                            af = min(af + acceleration, maximum)
                else:
                    psar[i] = psar[i-1] + af * (ep - psar[i-1])
                    
                    if high[i] > psar[i]:
                        long = True
                        psar[i] = ep
                        ep = low[i]
                        af = acceleration
                    else:
                        if low[i] < ep:
                            ep = low[i]
                            af = min(af + acceleration, maximum)
            
            return pd.Series(psar, index=df.index)
            
        except Exception as e:
            logging.error(f"Error calculating PSAR: {e}")
            return pd.Series(df['close'].rolling(window=20).mean(), index=df.index)


class EnhancedFeatureEngineer(FeatureEngineer):
    """
    Enhanced feature engineering module that extends the base FeatureEngineer with:
    
    1. Market microstructure features (order book analysis, trade flow)
    2. Advanced momentum and volatility indicators
    3. Alternative data integration
    4. Adaptive feature calculation based on market regime
    5. Feature preprocessing and normalization
    
    This class maintains backward compatibility with the original FeatureEngineer.
    """
    
    def __init__(self, 
                 use_microstructure: bool = True,
                 use_alternative_data: bool = True,
                 use_advanced_indicators: bool = True,
                 use_adaptive_features: bool = True,
                 use_normalization: bool = True,
                 use_crypto_features: bool = True,
                 scaler_type: str = 'standard',
                 history_window: int = 100,
                 alternative_data_path: str = 'data/alternative'):
        """
        Initialize the Enhanced Feature Engineer.
        
        Args:
            use_microstructure: Whether to calculate market microstructure features
            use_alternative_data: Whether to incorporate alternative data
            use_advanced_indicators: Whether to calculate advanced technical indicators
            use_adaptive_features: Whether to adapt feature calculation to market regime
            use_normalization: Whether to normalize features
            scaler_type: Type of scaler to use ('standard', 'minmax', or 'robust')
            history_window: Size of rolling window for historical calculations
            alternative_data_path: Path to alternative data files
        """
        # Initialize parent class
        super().__init__()
        
        # Feature calculation flags
        self.use_microstructure = use_microstructure
        self.use_alternative_data = use_alternative_data
        self.use_advanced_indicators = use_advanced_indicators
        self.use_adaptive_features = use_adaptive_features
        self.use_normalization = use_normalization and SKLEARN_AVAILABLE
        
        # Configuration
        self.history_window = history_window
        self.alternative_data_path = alternative_data_path
        self.talib_available = TALIB_AVAILABLE
        
        # Initialize scalers if normalization is enabled
        self.scalers = {}
        self.scaler_type = scaler_type
        if self.use_normalization:
            self._initialize_scalers()
        
        # Buffers for historical data
        self.price_history = deque(maxlen=self.history_window)
        self.volume_history = deque(maxlen=self.history_window)
        
        # Feature sets
        self.base_features = ['rsi', 'macd_hist', 'bollinger_width', 'atr', 'adx', 'obv']
        
        self.microstructure_features = [
            'bid_ask_spread', 'order_book_imbalance', 'trade_flow_imbalance',
            'vwap', 'vwap_deviation', 'market_impact', 'effective_spread'
        ]
        
        self.advanced_indicators = [
            'kst', 'ichimoku_a', 'ichimoku_b', 'fisher_transform', 
            'wavetrend', 'chande_momentum', 'elder_ray_bull', 'elder_ray_bear',
            'hurst_exponent', 'fractals_up', 'fractals_down', 'volatility_ratio'
        ]
        
        self.alternative_features = [
            'sentiment_score', 'news_impact', 'social_volume', 'funding_rate',
            'liquidations', 'open_interest_change', 'whale_activity', 'network_value'
        ]
        
        # Initialize crypto features
        self.use_crypto_features = use_crypto_features
        if use_crypto_features and CRYPTO_FEATURES_AVAILABLE:
            try:
                self.crypto_features = CryptoFeatures()
                logging.info("✅ Crypto features initialized for ETH/FDUSD")
            except Exception as e:
                logging.warning(f"❌ Failed to initialize crypto features: {e}")
                self.crypto_features = None
        else:
            self.crypto_features = None
        
        # Crypto-specific feature names
        self.crypto_features_list = [
            'funding_rate', 'funding_rate_impact', 'open_interest', 'open_interest_change',
            'long_liquidations', 'short_liquidations', 'liquidation_imbalance',
            'order_book_imbalance', 'whale_activity_score', 'taker_ratio', 'maker_ratio',
            'funding_oi_pressure', 'liquidation_whale_impact', 'volatility_predictor'
        ]
        
        logging.info("Enhanced Feature Engineer initialized with advanced capabilities.")
        
    def _initialize_scalers(self):
        """
        Initialize the appropriate scalers based on configuration.
        """
        if not SKLEARN_AVAILABLE:
            logging.warning("Sklearn not available. Feature normalization disabled.")
            self.use_normalization = False
            return
            
        try:
            # Initialize different scalers for different feature types
            if self.scaler_type == 'standard':
                self.scalers['price'] = StandardScaler()
                self.scalers['volume'] = StandardScaler()
                self.scalers['indicators'] = StandardScaler()
                self.scalers['microstructure'] = StandardScaler()
                self.scalers['alternative'] = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scalers['price'] = MinMaxScaler(feature_range=(-1, 1))
                self.scalers['volume'] = MinMaxScaler(feature_range=(0, 1))
                self.scalers['indicators'] = MinMaxScaler(feature_range=(-1, 1))
                self.scalers['microstructure'] = MinMaxScaler(feature_range=(-1, 1))
                self.scalers['alternative'] = MinMaxScaler(feature_range=(-1, 1))
            elif self.scaler_type == 'robust':
                self.scalers['price'] = RobustScaler()
                self.scalers['volume'] = RobustScaler()
                self.scalers['indicators'] = RobustScaler()
                self.scalers['microstructure'] = RobustScaler()
                self.scalers['alternative'] = RobustScaler()
            else:
                logging.warning(f"Unknown scaler type: {self.scaler_type}. Using StandardScaler.")
                self.scalers['price'] = StandardScaler()
                self.scalers['volume'] = StandardScaler()
                self.scalers['indicators'] = StandardScaler()
                self.scalers['microstructure'] = StandardScaler()
                self.scalers['alternative'] = StandardScaler()
                
            logging.info(f"Initialized {self.scaler_type} scalers for feature normalization.")
        except Exception as e:
            logging.error(f"Error initializing scalers: {e}")
            self.use_normalization = False
            
    def add_enhanced_features(self, df: pd.DataFrame, market_regime: str = None, order_book: Dict = None, whale_features: Dict = None) -> pd.DataFrame:
        """
        Add enhanced features, including whale activity features, to the DataFrame.
        Args:
            df: DataFrame with OHLCV data
            market_regime: Optional market regime string
            order_book: Optional order book data
            whale_features: Optional dict of whale features (large trades, whale alerts, order book imbalance, onchain flows)
        Returns:
            DataFrame with enhanced features added
        """
        if df.empty:
            return df
        # Add whale features
        whale_keys = [
            'large_trade_count', 'large_trade_volume', 'large_buy_count', 'large_sell_count',
            'large_buy_volume', 'large_sell_volume', 'whale_alert_count', 'whale_alert_flag',
            'order_book_imbalance', 'onchain_whale_inflow', 'onchain_whale_outflow'
        ]
        if whale_features is None:
            whale_features = {k: 0.0 for k in whale_keys}
        for k in whale_keys:
            df[k] = whale_features.get(k, 0.0)
        
        # Continue with existing enhanced features
        df = super().add_features(df)
        
        # (Other enhanced features: microstructure, alternative data, etc.)
        return df
            
    def _update_history(self, df: pd.DataFrame) -> None:
        """
        Update historical data buffers with the latest data.
        
        Args:
            df: DataFrame with latest market data
        """
        try:
            if not df.empty:
                # Update price history
                latest_price = df['close'].iloc[-1]
                self.price_history.append(latest_price)
                
                # Update volume history
                latest_volume = df['volume'].iloc[-1]
                self.volume_history.append(latest_volume)
                
        except Exception as e:
            logging.warning(f"Error updating history: {e}")
            
    def _add_microstructure_features(self, df: pd.DataFrame, order_book: Dict) -> pd.DataFrame:
        """
        Add market microstructure features based on order book data.
        
        Args:
            df: DataFrame with market data
            order_book: Order book data from exchange
            
        Returns:
            DataFrame with microstructure features added
        """
        try:
            # Initialize default values
            df['bid_ask_spread'] = 0.001
            df['order_book_imbalance'] = 0.0
            df['vwap'] = df['close']
            df['vwap_deviation'] = 0.0
            df['market_impact'] = 0.0
            df['effective_spread'] = 0.001
            df['trade_flow_imbalance'] = 0.0
            
            # Check if order_book is valid
            if order_book and isinstance(order_book, dict) and 'bids' in order_book and 'asks' in order_book:
                try:
                    best_bid = float(order_book['bids'][0][0]) if order_book['bids'] else df['close'].iloc[-1]
                    best_ask = float(order_book['asks'][0][0]) if order_book['asks'] else df['close'].iloc[-1]
                    
                    # Bid-ask spread
                    df['bid_ask_spread'] = (best_ask - best_bid) / best_bid
                    
                    # Order book imbalance
                    bid_volume = sum(float(bid[1]) for bid in order_book['bids'][:5])
                    ask_volume = sum(float(ask[1]) for ask in order_book['asks'][:5])
                    total_volume = bid_volume + ask_volume
                    df['order_book_imbalance'] = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
                    
                except (IndexError, ValueError, TypeError) as e:
                    logging.warning(f"Error processing order book data: {e}")
                    # Keep default values
            
            # VWAP calculation (ensure we're working with pandas Series)
            if 'volume' in df.columns and 'close' in df.columns:
                try:
                    # Calculate VWAP using pandas operations
                    volume_series = pd.Series(df['volume'].values, index=df.index)
                    close_series = pd.Series(df['close'].values, index=df.index)
                    
                    # Calculate rolling VWAP
                    rolling_volume = volume_series.rolling(window=20, min_periods=1).sum()
                    rolling_price_volume = (close_series * volume_series).rolling(window=20, min_periods=1).sum()
                    
                    df['vwap'] = rolling_price_volume / rolling_volume
                    df['vwap_deviation'] = (close_series - df['vwap']) / df['vwap']
                    
                    # Market impact (simplified)
                    volume_std = volume_series.rolling(window=10, min_periods=1).std()
                    volume_mean = volume_series.rolling(window=10, min_periods=1).mean()
                    df['market_impact'] = volume_std / volume_mean
                    
                except Exception as e:
                    logging.warning(f"Error calculating VWAP: {e}")
                    # Keep default values
            
            # Effective spread
            df['effective_spread'] = df['bid_ask_spread'] * (1 + abs(df['order_book_imbalance']))
            
            # Trade flow imbalance (based on volume and price movement)
            if 'volume' in df.columns and 'close' in df.columns:
                try:
                    # Calculate price changes
                    price_change = df['close'].diff()
                    
                    # Create trade flow series
                    trade_flow = pd.Series(0.0, index=df.index)
                    trade_flow[price_change > 0] = df['volume'][price_change > 0]
                    trade_flow[price_change < 0] = -df['volume'][price_change < 0]
                    
                    # Calculate rolling sum
                    df['trade_flow_imbalance'] = trade_flow.rolling(window=10, min_periods=1).sum()
                    
                except Exception as e:
                    logging.warning(f"Error calculating trade flow: {e}")
            
            return df
            
        except Exception as e:
            logging.warning(f"Error adding microstructure features: {e}")
            # Return DataFrame with default values
            df['bid_ask_spread'] = 0.001
            df['order_book_imbalance'] = 0.0
            df['vwap'] = df['close']
            df['vwap_deviation'] = 0.0
            df['market_impact'] = 0.0
            df['effective_spread'] = 0.001
            df['trade_flow_imbalance'] = 0.0
            return df
    
    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive advanced technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with advanced indicators added
        """
        try:
            # 🚀 EXISTING ADVANCED INDICATORS
            # KST (Know Sure Thing) indicator
            df['kst'] = self._calculate_kst(df)
            
            # Ichimoku Cloud components
            df['ichimoku_a'] = self._calculate_ichimoku_a(df)
            df['ichimoku_b'] = self._calculate_ichimoku_b(df)
            
            # Fisher Transform
            df['fisher_transform'] = self._calculate_fisher_transform(df)
            
            # Wave Trend
            df['wavetrend'] = self._calculate_wavetrend(df)
            
            # Chande Momentum Oscillator
            df['chande_momentum'] = self._calculate_chande_momentum(df)
            
            # Elder Ray Index
            df['elder_ray_bull'] = df['high'] - df['close'].rolling(window=13).mean()
            df['elder_ray_bear'] = df['low'] - df['close'].rolling(window=13).mean()
            
            # Hurst Exponent (simplified)
            df['hurst_exponent'] = self._calculate_hurst_exponent(df)
            
            # Fractals
            df['fractals_up'] = self._calculate_fractals_up(df)
            df['fractals_down'] = self._calculate_fractals_down(df)
            
            # Volatility Ratio (ensure ATR exists)
            if 'atr' in df.columns:
                df['volatility_ratio'] = df['atr'] / df['atr'].rolling(window=20).mean()
            else:
                # Calculate ATR if not present
                df['atr'] = self._calculate_atr(df)
                df['volatility_ratio'] = df['atr'] / df['atr'].rolling(window=20).mean()
            
            # 🚀 NEW: SUPER ADVANCED MOVING AVERAGES
            # ZLEMA (Zero Lag Exponential Moving Average)
            for period in [10, 20, 50]:
                lag = (period - 1) / 2
                df[f'zlema_{period}'] = df['close'].ewm(span=period).mean() + (df['close'] - df['close'].shift(int(lag)))
            
            # DEMA (Double Exponential Moving Average)
            for period in [10, 20, 50]:
                ema1 = df['close'].ewm(span=period).mean()
                ema2 = ema1.ewm(span=period).mean()
                df[f'dema_{period}'] = 2 * ema1 - ema2
            
            # FRAMA (Fractal Adaptive Moving Average)
            for period in [10, 20, 50]:
                df[f'frama_{period}'] = self._calculate_frama(df['close'], period)
            
            # JMA (Jurik Moving Average)
            for period in [10, 20, 50]:
                df[f'jma_{period}'] = self._calculate_jma(df['close'], period)
            
            # ALMA (Arnaud Legoux Moving Average)
            for period in [10, 20, 50]:
                df[f'alma_{period}'] = self._calculate_alma(df['close'], period)
            
            # VWMA (Volume Weighted Moving Average)
            for period in [10, 20, 50]:
                df[f'vwma_{period}'] = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
            
            # 🚀 NEW: SUPER ADVANCED OSCILLATORS
            # PPO (Percentage Price Oscillator)
            for fast, slow in [(12, 26), (8, 21), (21, 55)]:
                fast_ema = df['close'].ewm(span=fast).mean()
                slow_ema = df['close'].ewm(span=slow).mean()
                df[f'ppo_{fast}_{slow}'] = ((fast_ema - slow_ema) / slow_ema) * 100
                df[f'ppo_signal_{fast}_{slow}'] = df[f'ppo_{fast}_{slow}'].ewm(span=9).mean()
                df[f'ppo_hist_{fast}_{slow}'] = df[f'ppo_{fast}_{slow}'] - df[f'ppo_signal_{fast}_{slow}']
            
            # DPO (Detrended Price Oscillator)
            for period in [20, 50]:
                shift_period = period // 2 + 1
                sma = df['close'].rolling(period).mean()
                df[f'dpo_{period}'] = df['close'] - sma.shift(shift_period)
            
            # TRIX
            for period in [15, 30]:
                ema1 = df['close'].ewm(span=period).mean()
                ema2 = ema1.ewm(span=period).mean()
                ema3 = ema2.ewm(span=period).mean()
                df[f'trix_{period}'] = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
                df[f'trix_signal_{period}'] = df[f'trix_{period}'].ewm(span=9).mean()
            
            # 🚀 NEW: SUPER ADVANCED VOLATILITY INDICATORS
            # SuperTrend
            for period in [10, 20]:
                df[f'supertrend_{period}'] = self._calculate_supertrend(df, period)
            
            # Donchian Channels
            for period in [20, 50]:
                df[f'donchian_upper_{period}'] = df['high'].rolling(period).max()
                df[f'donchian_lower_{period}'] = df['low'].rolling(period).min()
                df[f'donchian_mid_{period}'] = (df[f'donchian_upper_{period}'] + df[f'donchian_lower_{period}']) / 2
                df[f'donchian_width_{period}'] = (df[f'donchian_upper_{period}'] - df[f'donchian_lower_{period}']) / df[f'donchian_mid_{period}']
            
            # Chandelier Exit
            for period in [22]:
                df[f'chandelier_long_{period}'] = self._calculate_chandelier_exit(df, period, 'long')
                df[f'chandelier_short_{period}'] = self._calculate_chandelier_exit(df, period, 'short')
            
            # 🚀 NEW: SUPER ADVANCED VOLUME INDICATORS
            # Force Index
            for period in [13, 21]:
                force_index = df['close'].diff() * df['volume']
                df[f'force_index_{period}'] = force_index.ewm(span=period).mean()
            
            # Ease of Movement
            for period in [14]:
                high_low = df['high'] - df['low']
                high_low_prev = df['high'].shift(1) - df['low'].shift(1)
                box_ratio = df['volume'] / (high_low + 1e-8)
                distance_moved = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
                ease_of_movement = distance_moved / (box_ratio + 1e-8)
                df[f'ease_of_movement_{period}'] = ease_of_movement.ewm(span=period).mean()
            
            # Accumulation/Distribution Line
            clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)
            df['accumulation_distribution'] = (clv * df['volume']).cumsum()
            
            # 🚀 NEW: SUPER ADVANCED PATTERN RECOGNITION
            # Engulfing Patterns
            df['bullish_engulfing'] = np.where(
                (df['close'] > df['open']) &  # Current candle is bullish
                (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish
                (df['open'] < df['close'].shift(1)) &  # Current open below previous close
                (df['close'] > df['open'].shift(1)),  # Current close above previous open
                1, 0
            )
            
            df['bearish_engulfing'] = np.where(
                (df['close'] < df['open']) &  # Current candle is bearish
                (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
                (df['open'] > df['close'].shift(1)) &  # Current open above previous close
                (df['close'] < df['open'].shift(1)),  # Current close below previous open
                1, 0
            )
            
            # 🚀 NEW: SUPER ADVANCED MOMENTUM INDICATORS
            # Williams Alligator
            df['alligator_jaw'] = (df['high'].rolling(13).max() + df['low'].rolling(13).min()) / 2
            df['alligator_teeth'] = (df['high'].rolling(8).max() + df['low'].rolling(8).min()) / 2
            df['alligator_lips'] = (df['high'].rolling(5).max() + df['low'].rolling(5).min()) / 2
            
            # 🚀 NEW: SUPER ADVANCED TREND INDICATORS
            # Linear Regression Slope
            for period in [20, 50]:
                df[f'linear_regression_slope_{period}'] = df['close'].rolling(period).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
            
            # 🚀 NEW: SUPER ADVANCED RISK METRICS
            # Z-Score
            for period in [20, 50]:
                returns = df['close'].pct_change()
                df[f'z_score_{period}'] = (returns - returns.rolling(period).mean()) / returns.rolling(period).std()
            
            # VIX-like Volatility Index
            returns = df['close'].pct_change()
            df['vix_like'] = returns.rolling(20).std() * np.sqrt(252) * 100
            
            # 🚀 NEW: SUPER ADVANCED MICROSTRUCTURE
            # Price Velocity
            df['price_velocity'] = df['close'].diff(5) / 5
            
            # Volume Price Trend (Enhanced)
            df['vpt_enhanced'] = (df['close'].pct_change() * df['volume']).cumsum()
            df['vpt_ma'] = df['vpt_enhanced'].rolling(20).mean()
            df['vpt_signal'] = np.where(df['vpt_enhanced'] > df['vpt_ma'], 1, -1)
            
            # 🚀 NEW: SUPER ADVANCED CROSSOVER SIGNALS
            # Multiple MA Crossovers
            for short_ma in ['sma_5', 'ema_10', 'hma_10']:
                for long_ma in ['sma_50', 'ema_50', 'hma_50']:
                    if short_ma in df.columns and long_ma in df.columns:
                        df[f'crossover_{short_ma}_{long_ma}'] = np.where(
                            df[short_ma] > df[long_ma], 1, -1
                        )
                        df[f'crossover_signal_{short_ma}_{long_ma}'] = np.where(
                            (df[short_ma] > df[long_ma]) & (df[short_ma].shift(1) <= df[long_ma].shift(1)), 1, 0
                        )
            
            return df
            
        except Exception as e:
            logging.warning(f"Error adding advanced indicators: {e}")
            # Continue with basic features if advanced indicators fail
            return df
    
    def _calculate_kst(self, df: pd.DataFrame) -> pd.Series:
        """Calculate KST (Know Sure Thing) indicator."""
        try:
            # Simplified KST calculation
            rcm1 = df['close'].pct_change(10).rolling(window=10).mean()
            rcm2 = df['close'].pct_change(15).rolling(window=10).mean()
            rcm3 = df['close'].pct_change(20).rolling(window=10).mean()
            rcm4 = df['close'].pct_change(30).rolling(window=15).mean()
            
            kst = (rcm1 * 1) + (rcm2 * 2) + (rcm3 * 3) + (rcm4 * 4)
            return kst
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_ichimoku_a(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Ichimoku A line."""
        try:
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            return (high_9 + low_9) / 2
        except:
            return pd.Series(df['close'], index=df.index)
                    
    def _calculate_ichimoku_b(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Ichimoku B line."""
        try:
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            return (high_26 + low_26) / 2
        except:
            return pd.Series(df['close'], index=df.index)
    
    def _calculate_fisher_transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Fisher Transform."""
        try:
            high_low = df['high'] - df['low']
            high_close = df['high'] - df['close'].shift(1)
            low_close = df['low'] - df['close'].shift(1)
            
            value1 = np.where(high_low != 0, (high_close + low_close) / high_low, 0)
            value2 = value1.rolling(window=10).mean()
            
            fisher = 0.5 * np.log((1 + value2) / (1 - value2))
            return fisher
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_wavetrend(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Wave Trend oscillator."""
        try:
            # Simplified Wave Trend calculation
            ap = (df['high'] + df['low'] + df['close']) / 3
            ema1 = ap.ewm(span=10).mean()
            ema2 = ap.ewm(span=21).mean()
            
            wavetrend = (ema1 - ema2) / (ema1 + ema2) * 100
            return wavetrend
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_chande_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Chande Momentum Oscillator."""
        try:
            up_sum = df['close'].diff().where(df['close'].diff() > 0, 0).rolling(window=14).sum()
            down_sum = abs(df['close'].diff().where(df['close'].diff() < 0, 0)).rolling(window=14).sum()
            
            cmo = 100 * (up_sum - down_sum) / (up_sum + down_sum)
            return cmo
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_hurst_exponent(self, df: pd.DataFrame) -> pd.Series:
        """Calculate simplified Hurst Exponent."""
        try:
            # Simplified calculation
            returns = df['close'].pct_change()
            hurst = returns.rolling(window=20).std() / returns.rolling(window=60).std()
            return hurst
        except:
            return pd.Series(0.5, index=df.index)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            return atr
        except:
            return pd.Series(df['close'] * 0.02, index=df.index)  # Default 2% volatility
    
    def _calculate_fractals_up(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bullish fractals."""
        try:
            fractals = pd.Series(0, index=df.index)
            for i in range(2, len(df) - 2):
                if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                    df['high'].iloc[i] > df['high'].iloc[i-2] and
                    df['high'].iloc[i] > df['high'].iloc[i+1] and
                    df['high'].iloc[i] > df['high'].iloc[i+2]):
                    fractals.iloc[i] = 1
            return fractals
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_fractals_down(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bearish fractals."""
        try:
            fractals = pd.Series(0, index=df.index)
            for i in range(2, len(df) - 2):
                if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                    df['low'].iloc[i] < df['low'].iloc[i-2] and
                    df['low'].iloc[i] < df['low'].iloc[i+1] and
                    df['low'].iloc[i] < df['low'].iloc[i+2]):
                    fractals.iloc[i] = 1
            return fractals
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_frama(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate FRAMA (Fractal Adaptive Moving Average).
        
        Args:
            prices: Price series
            period: Period for calculation
            
        Returns:
            FRAMA series
        """
        try:
            # Simplified FRAMA implementation
            alpha = 0.5  # Default alpha
            frama = prices.ewm(alpha=alpha, adjust=False).mean()
            return frama
            
        except Exception as e:
            logging.warning(f"Error calculating FRAMA: {e}")
            return prices.rolling(period).mean()
    
    def _calculate_jma(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate JMA (Jurik Moving Average).
        
        Args:
            prices: Price series
            period: Period for calculation
            
        Returns:
            JMA series
        """
        try:
            # Simplified JMA implementation
            jma = prices.ewm(span=period, adjust=False).mean()
            return jma
            
        except Exception as e:
            logging.warning(f"Error calculating JMA: {e}")
            return prices.rolling(period).mean()
    
    def _calculate_alma(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate ALMA (Arnaud Legoux Moving Average).
        
        Args:
            prices: Price series
            period: Period for calculation
            
        Returns:
            ALMA series
        """
        try:
            # Simplified ALMA implementation
            sigma = 0.85
            offset = 0.85
            
            weights = np.exp(-((np.arange(period) - offset * (period - 1)) ** 2) / (2 * sigma * sigma))
            weights = weights / weights.sum()
            
            alma = prices.rolling(period).apply(
                lambda x: np.dot(x, weights) if len(x) == period else np.nan
            )
            return alma
            
        except Exception as e:
            logging.warning(f"Error calculating ALMA: {e}")
            return prices.rolling(period).mean()
    
    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        Calculate SuperTrend indicator.
        
        Args:
            df: DataFrame with high, low, close columns
            period: Period for ATR calculation
            
        Returns:
            SuperTrend series
        """
        try:
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            # Calculate SuperTrend
            hl2 = (df['high'] + df['low']) / 2
            upperband = hl2 + (2 * atr)
            lowerband = hl2 - (2 * atr)
            
            supertrend = pd.Series(index=df.index, dtype=float)
            supertrend.iloc[0] = lowerband.iloc[0]
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > supertrend.iloc[i-1]:
                    supertrend.iloc[i] = max(lowerband.iloc[i], supertrend.iloc[i-1])
                else:
                    supertrend.iloc[i] = min(upperband.iloc[i], supertrend.iloc[i-1])
            
            return supertrend
            
        except Exception as e:
            logging.warning(f"Error calculating SuperTrend: {e}")
            return df['close'].rolling(period).mean()
    
    def _calculate_chandelier_exit(self, df: pd.DataFrame, period: int = 22, direction: str = 'long') -> pd.Series:
        """
        Calculate Chandelier Exit indicator.
        
        Args:
            df: DataFrame with high, low, close columns
            period: Period for ATR calculation
            direction: 'long' or 'short'
            
        Returns:
            Chandelier Exit series
        """
        try:
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            if direction == 'long':
                # Long exit
                highest_high = df['high'].rolling(period).max()
                chandelier_exit = highest_high - (3 * atr)
            else:
                # Short exit
                lowest_low = df['low'].rolling(period).min()
                chandelier_exit = lowest_low + (3 * atr)
            
            return chandelier_exit
            
        except Exception as e:
            logging.warning(f"Error calculating Chandelier Exit: {e}")
            return df['close'].rolling(period).mean()
            
    def _add_crypto_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive crypto-specific features for ETH/FDUSD trading.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with crypto features added
        """
        if df.empty or not self.crypto_features:
            return df
        
        try:
            # Get all crypto features
            crypto_data = self.crypto_features.get_all_crypto_features("ETHFDUSD")
            
            # Add crypto features to DataFrame
            for feature_name, value in crypto_data.items():
                df[feature_name] = value
            
            logging.info(f"Added {len(crypto_data)} crypto features to DataFrame")
            return df
            
        except Exception as e:
            logging.warning(f"Failed to add crypto features: {e}")
            # Add default values for crypto features
            for feature_name in self.crypto_features_list:
                df[feature_name] = 0.0
            return df
    
    def _add_alternative_data_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add alternative data features, including external news, market, and sentiment APIs.
        Features: news_sentiment_score, news_volume, breaking_news_flag, news_volatility,
        external_market_cap, external_supply, external_rank, external_price, external_volume_24h,
        fear_greed_index, fear_greed_trend, etc.
        """
        try:
            # --- Existing placeholders (keep for backward compatibility) ---
            df['sentiment_score'] = 0
            df['news_impact'] = 0
            df['social_volume'] = 0
            df['funding_rate'] = 0
            df['liquidations'] = 0
            df['open_interest_change'] = 0
            df['whale_activity'] = 0
            df['network_value'] = 0
            
            # --- NEW: External API features ---
            from modules.alternative_data import EnhancedAlternativeData
            import json
            # Load API keys from config
            try:
                with open('config.json', 'r') as f:
                    config = json.load(f)
                api_keys = config.get('api_keys', {})
            except Exception:
                api_keys = {}
            alt = EnhancedAlternativeData(api_keys=api_keys)

            # News & Sentiment (NewsData.io, MediaStack, GNews, Guardian)
            try:
                news = alt.get_news_impact()
                df['news_sentiment_score'] = news.get('news_sentiment_score', 0)  # Avg sentiment, all news APIs
                df['news_volume'] = news.get('news_volume', 0)                    # Number of articles
                df['breaking_news_flag'] = int(news.get('breaking_news_flag', False))  # 1 if breaking news
                df['news_volatility'] = {'low': 0, 'medium': 1, 'high': 2}.get(news.get('news_volatility', 'low'), 0)
            except Exception as e:
                df['news_sentiment_score'] = 0
                df['news_volume'] = 0
                df['breaking_news_flag'] = 0
                df['news_volatility'] = 0

            # Market Data (CoinMarketCap, CoinRanking, FreeCryptoAPI)
            try:
                market = alt.get_external_market_data()
                df['external_market_cap'] = market.get('cmc_market_cap', 0) or market.get('cr_market_cap', 0) or market.get('fca_market_cap', 0)
                df['external_supply'] = market.get('cmc_supply', 0) or market.get('cr_supply', 0) or market.get('fca_supply', 0)
                df['external_rank'] = market.get('cmc_rank', 0) or market.get('cr_rank', 0) or market.get('fca_rank', 0)
                df['external_price'] = market.get('cmc_price', 0) or market.get('cr_price', 0) or market.get('fca_price', 0)
                df['external_volume_24h'] = market.get('cmc_volume_24h', 0) or market.get('cr_volume_24h', 0) or market.get('fca_volume_24h', 0)
            except Exception as e:
                df['external_market_cap'] = 0
                df['external_supply'] = 0
                df['external_rank'] = 0
                df['external_price'] = 0
                df['external_volume_24h'] = 0

            # Fear & Greed (CoinyBubble, fallback alternative.me)
            try:
                fg = alt._get_fear_and_greed()
                df['fear_greed_index'] = fg.get('value', 50)
                df['fear_greed_trend'] = {'rising': 1, 'falling': -1, 'unknown': 0}.get(fg.get('trend', 'unknown'), 0)
            except Exception as e:
                df['fear_greed_index'] = 50
                df['fear_greed_trend'] = 0

            # Finnhub features
            try:
                finnhub_data = alt._get_finnhub_data()
                df['finnhub_sentiment_score'] = finnhub_data.get('finnhub_sentiment_score', 0)
                df['finnhub_news_count'] = finnhub_data.get('finnhub_news_count', 0)
                df['finnhub_company_country'] = finnhub_data.get('finnhub_company_country', '')
                df['finnhub_price'] = finnhub_data.get('finnhub_price', 0)
                df['finnhub_volume'] = finnhub_data.get('finnhub_volume', 0)
                df['finnhub_rsi'] = finnhub_data.get('finnhub_rsi', 0)
            except Exception:
                df['finnhub_sentiment_score'] = 0
                df['finnhub_news_count'] = 0
                df['finnhub_company_country'] = ''
                df['finnhub_price'] = 0
                df['finnhub_volume'] = 0
                df['finnhub_rsi'] = 0
            # Twelve Data features
            try:
                twelvedata_data = alt._get_twelvedata_data()
                df['twelvedata_price'] = twelvedata_data.get('twelvedata_price', 0)
                df['twelvedata_volume'] = twelvedata_data.get('twelvedata_volume', 0)
                df['twelvedata_rsi'] = twelvedata_data.get('twelvedata_rsi', 0)
            except Exception:
                df['twelvedata_price'] = 0
                df['twelvedata_volume'] = 0
                df['twelvedata_rsi'] = 0

            return df
        except Exception as e:
            logging.warning(f"Error adding alternative data features: {e}")
            return df
            
    def _adapt_features_to_regime(self, df: pd.DataFrame, market_regime: str) -> pd.DataFrame:
        """
        Adapt feature calculations based on market regime.
        
        Args:
            df: DataFrame with features
            market_regime: Current market regime
            
        Returns:
            DataFrame with regime-adapted features
        """
        try:
            if market_regime == 'TRENDING':
                # Emphasize trend-following indicators
                df['regime_adjusted_rsi'] = df['rsi'] * 1.2
                df['regime_adjusted_macd'] = df['macd'] * 1.1
            elif market_regime == 'RANGING':
                # Emphasize mean-reversion indicators
                df['regime_adjusted_rsi'] = df['rsi'] * 0.8
                df['regime_adjusted_bollinger'] = df['bollinger_width'] * 1.3
            elif market_regime == 'VOLATILE':
                # Emphasize volatility indicators
                df['regime_adjusted_atr'] = df['atr'] * 1.5
                df['regime_adjusted_volatility'] = df['volatility_ratio'] * 1.2
            else:
                # Normal regime - no adjustments
                df['regime_adjusted_rsi'] = df['rsi']
                df['regime_adjusted_macd'] = df['macd']
                if 'bollinger_width' in df.columns:
                    df['regime_adjusted_bollinger'] = df['bollinger_width']
                if 'atr' in df.columns:
                    df['regime_adjusted_atr'] = df['atr']
                if 'volatility_ratio' in df.columns:
                    df['regime_adjusted_volatility'] = df['volatility_ratio']
            
            return df
            
        except Exception as e:
            logging.warning(f"Error adapting features to regime: {e}")
            # Continue with original features if regime adaptation fails
            return df
            
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using the appropriate scalers.
        
        Args:
            df: DataFrame with features to normalize
            
        Returns:
            DataFrame with normalized features
        """
        try:
            if not self.use_normalization or not self.scalers:
                return df
            
            # Define feature groups
            price_features = ['open', 'high', 'low', 'close']
            volume_features = ['volume', 'obv']
            indicator_features = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bollinger_width', 'atr', 'adx']
            microstructure_features = ['bid_ask_spread', 'order_book_imbalance', 'vwap_deviation', 'market_impact']
            
            # Normalize each group
            for feature_group, scaler_key in [
                (price_features, 'price'),
                (volume_features, 'volume'),
                (indicator_features, 'indicators'),
                (microstructure_features, 'microstructure')
            ]:
                if scaler_key in self.scalers:
                    available_features = [f for f in feature_group if f in df.columns]
                    if available_features:
                        df[available_features] = self.scalers[scaler_key].fit_transform(df[available_features])
            
            return df
            
        except Exception as e:
            logging.warning(f"Error normalizing features: {e}")
            return df
            
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        try:
            # Forward fill for most features
            df = df.ffill()
            
            # Backward fill for any remaining NaNs
            df = df.bfill()
            
            # Fill any remaining NaNs with 0
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logging.warning(f"Error handling missing values: {e}")
            return df
            
    def get_feature_importance(self, model) -> Dict[str, float]:
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            if hasattr(model, 'feature_importances_'):
                feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
                importances = model.feature_importances_
                
                if len(feature_names) == len(importances):
                    return dict(zip(feature_names, importances))
                else:
                    return {f'feature_{i}': imp for i, imp in enumerate(importances)}
            else:
                logging.warning("Model does not have feature_importances_ attribute")
                return {}
                
        except Exception as e:
            logging.error(f"Error getting feature importance: {e}")
            return {}

    def enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced feature engineering method that adds all advanced features.
        This is the main method called by the training script.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with all enhanced features added
        """
        try:
            if df.empty:
                logging.warning("Input DataFrame is empty. Skipping feature engineering.")
                return df
            
            # Update historical data
            self._update_history(df)
            
            # Add base features using parent class method
            df = super().add_features(df)
            if df.empty:
                return df
            
            # Add advanced technical indicators
            if self.use_advanced_indicators:
                df = self._add_advanced_indicators(df)
            
            # Add market microstructure features (without order book for now)
            if self.use_microstructure:
                df = self._add_microstructure_features_simple(df)
            
            # Add alternative data features
            if self.use_alternative_data:
                df = self._add_alternative_data_features(df)
            
            # Add crypto-specific features
            if self.use_crypto_features:
                df = self._add_crypto_features(df)
            
            # Apply adaptive feature calculations
            if self.use_adaptive_features:
                df = self._adapt_features_to_regime(df, 'NORMAL')  # Default regime
            
            # Apply feature normalization if enabled
            if self.use_normalization:
                df = self._normalize_features(df)
            
            # Add MAXIMUM INTELLIGENCE features
            df = self._add_maximum_intelligence_features(df)
            
            # Ensure there are no NaN values
            df = self._handle_missing_values(df)
            
            logging.info(f"Enhanced features added successfully. Total features: {len(df.columns)}")
            return df
            
        except Exception as e:
            logging.error(f"Error in enhanced feature engineering: {e}")
            return df
    
    def _add_microstructure_features_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features without order book data.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with microstructure features added
        """
        try:
            # Initialize default values
            df['bid_ask_spread'] = 0.001
            df['order_book_imbalance'] = 0.0
            df['vwap'] = df['close']
            df['vwap_deviation'] = 0.0
            df['market_impact'] = 0.0
            df['effective_spread'] = 0.001
            df['trade_flow_imbalance'] = 0.0
            
            # Calculate VWAP
            if 'volume' in df.columns and 'close' in df.columns:
                try:
                    volume_series = pd.Series(df['volume'].values, index=df.index)
                    close_series = pd.Series(df['close'].values, index=df.index)
                    
                    # Calculate rolling VWAP
                    rolling_volume = volume_series.rolling(window=20, min_periods=1).sum()
                    rolling_price_volume = (close_series * volume_series).rolling(window=20, min_periods=1).sum()
                    
                    df['vwap'] = rolling_price_volume / rolling_volume
                    df['vwap_deviation'] = (close_series - df['vwap']) / df['vwap']
                    
                    # Market impact (simplified)
                    volume_std = volume_series.rolling(window=10, min_periods=1).std()
                    volume_mean = volume_series.rolling(window=10, min_periods=1).mean()
                    df['market_impact'] = volume_std / volume_mean
                    
                except Exception as e:
                    logging.warning(f"Error calculating VWAP: {e}")
            
            # Effective spread
            df['effective_spread'] = df['bid_ask_spread'] * (1 + abs(df['order_book_imbalance']))
            
            # Trade flow imbalance (based on volume and price movement)
            if 'volume' in df.columns and 'close' in df.columns:
                try:
                    # Calculate price changes
                    price_change = df['close'].diff()
                    
                    # Create trade flow series
                    trade_flow = pd.Series(0.0, index=df.index)
                    trade_flow[price_change > 0] = df['volume'][price_change > 0]
                    trade_flow[price_change < 0] = -df['volume'][price_change < 0]
                    
                    # Calculate rolling sum
                    df['trade_flow_imbalance'] = trade_flow.rolling(window=10, min_periods=1).sum()
                    
                except Exception as e:
                    logging.warning(f"Error calculating trade flow: {e}")

            return df
            
        except Exception as e:
            logging.warning(f"Error adding microstructure features: {e}")
            # Return DataFrame with default values
            df['bid_ask_spread'] = 0.001
            df['order_book_imbalance'] = 0.0
            df['vwap'] = df['close']
            df['vwap_deviation'] = 0.0
            df['market_impact'] = 0.0
            df['effective_spread'] = 0.001
            df['trade_flow_imbalance'] = 0.0
            return df
    
    def _add_maximum_intelligence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add maximum intelligence features for ultimate profitability"""
        try:
            # Quantum-inspired features
            df['quantum_momentum'] = df['close'].pct_change().rolling(5).apply(
                lambda x: np.exp(-np.sum(x**2)) if len(x) > 0 else 0
            )
            
            # Quantum entanglement (correlation between price and volume changes)
            price_changes = df['close'].pct_change().rolling(10).mean()
            volume_changes = df['volume'].pct_change().rolling(10).mean()
            df['quantum_entanglement'] = (price_changes * volume_changes).abs()
            
            df['quantum_tunneling'] = (
                (df['high'] - df['low']) / df['close']
            ).rolling(20).quantile(0.95)
            
            # AI-enhanced features
            df['ai_volatility_forecast'] = (
                df['close'].pct_change().rolling(50).std() * 
                (1 + df['volume'].pct_change().rolling(20).mean())
            )
            df['ai_trend_strength'] = (
                df['close'].rolling(20).mean() - df['close'].rolling(50).mean()
            ) / df['close'].rolling(50).std()
            df['ai_market_efficiency'] = (
                df['close'].pct_change().abs().rolling(30).mean() /
                df['close'].pct_change().rolling(30).std()
            )
            
            # Psychology features
            df['fear_greed_index'] = (
                (df['close'].pct_change().rolling(10).std() * 100) +
                (df['volume'].pct_change().rolling(10).mean() * 50)
            ).clip(0, 100)
            
            price_momentum = df['close'].pct_change().rolling(5).mean()
            volume_momentum = df['volume'].pct_change().rolling(5).mean()
            df['sentiment_momentum'] = price_momentum * volume_momentum
            
            df['herd_behavior'] = (
                df['volume'].rolling(10).std() / 
                df['volume'].rolling(10).mean()
            )
            
            # Advanced patterns
            df['elliott_wave'] = (
                df['close'].rolling(21).max() - df['close'].rolling(21).min()
            ) / df['close'].rolling(21).mean()
            
            price_change_8 = df['close'].pct_change().rolling(8).sum()
            price_change_13 = df['close'].pct_change().rolling(13).sum()
            df['harmonic_pattern'] = price_change_8 * price_change_13
            
            # Advanced microstructure
            volume_std = df['volume'].rolling(10).std()
            volume_mean = df['volume'].rolling(10).mean()
            price_change_abs = df['close'].pct_change().abs()
            df['order_flow_toxicity'] = (volume_std / volume_mean) * price_change_abs
            
            volume_change = df['volume'].pct_change().rolling(5).mean()
            df['market_impact_prediction'] = volume_change * price_change_abs.rolling(5).mean()
            
            df['liquidity_stress'] = (
                (df['high'] - df['low']) / df['close']
            ).rolling(20).quantile(0.9)
            
            # Regime switching features
            vol_20 = df['close'].pct_change().rolling(20).std()
            vol_50 = df['close'].pct_change().rolling(50).std()
            df['volatility_regime'] = np.where(vol_20 > vol_50, 1, 0)
            
            ma_20 = df['close'].rolling(20).mean()
            ma_50 = df['close'].rolling(50).mean()
            df['trend_regime'] = np.where(ma_20 > ma_50, 1, -1)
            
            vol_avg = df['volume'].rolling(20).mean()
            df['volume_regime'] = np.where(df['volume'] > vol_avg * 1.5, 1, 0)
            
            df['combined_regime'] = (
                df['volatility_regime'] + 
                df['trend_regime'] + 
                df['volume_regime']
            )
            
            # 🚀 NEW: Advanced Moving Average Crossovers
            # Multiple timeframe MA crossovers
            for short_period in [5, 10, 20]:
                for long_period in [20, 50, 100]:
                    if short_period < long_period:
                        short_ma = df['close'].rolling(short_period).mean()
                        long_ma = df['close'].rolling(long_period).mean()
                        df[f'ma_cross_{short_period}_{long_period}'] = np.where(short_ma > long_ma, 1, -1)
                        df[f'ma_cross_signal_{short_period}_{long_period}'] = np.where(
                            (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1)), 1, 0
                        )
            
            # 🚀 NEW: Advanced RSI Divergence Detection
            # RSI divergence with price
            for period in [14, 21]:
                rsi = df[f'rsi_{period}']
                price_highs = df['close'].rolling(10).max()
                rsi_highs = rsi.rolling(10).max()
                
                # Bullish divergence: price makes lower low, RSI makes higher low
                price_lower_low = (df['close'] < price_highs.shift(5)) & (price_highs.shift(5) < price_highs.shift(10))
                rsi_higher_low = (rsi > rsi_highs.shift(5)) & (rsi_highs.shift(5) > rsi_highs.shift(10))
                df[f'bullish_divergence_{period}'] = np.where(price_lower_low & rsi_higher_low, 1, 0)
                
                # Bearish divergence: price makes higher high, RSI makes lower high
                price_higher_high = (df['close'] > price_highs.shift(5)) & (price_highs.shift(5) > price_highs.shift(10))
                rsi_lower_high = (rsi < rsi_highs.shift(5)) & (rsi_highs.shift(5) < rsi_highs.shift(10))
                df[f'bearish_divergence_{period}'] = np.where(price_higher_high & rsi_lower_high, 1, 0)
            
            # 🚀 NEW: Advanced Volume Analysis
            # Volume price trend (VPT)
            df['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()
            df['vpt_ma'] = df['vpt'].rolling(20).mean()
            df['vpt_signal'] = np.where(df['vpt'] > df['vpt_ma'], 1, -1)
            
            # Volume rate of change
            df['volume_roc'] = df['volume'].pct_change(10) * 100
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # 🚀 NEW: Advanced Momentum Oscillators
            # Ultimate Oscillator
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            bp1 = df['close'] - df['low']
            bp2 = df['close'] - df['close'].shift(1)
            bp = pd.concat([bp1, bp2], axis=1).min(axis=1)
            bp = bp.where(bp > 0, 0)
            
            avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
            avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
            avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
            
            df['ultimate_oscillator'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
            
            # 🚀 NEW: Advanced Volatility Indicators
            # Chaikin Volatility
            df['chaikin_volatility'] = (
                (df['high'] - df['low']).rolling(10).mean() / 
                (df['high'] - df['low']).rolling(10).mean().rolling(10).mean()
            ) * 100
            
            # Volatility Index (VIX-like)
            returns = df['close'].pct_change()
            df['volatility_index'] = returns.rolling(20).std() * np.sqrt(252) * 100
            
            # 🚀 NEW: Advanced Trend Strength
            # Linear Regression Slope
            for period in [20, 50]:
                df[f'linear_regression_slope_{period}'] = df['close'].rolling(period).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
            
            # Trend Strength Index
            df['trend_strength_index'] = (
                (df['close'] - df['close'].rolling(50).mean()) / 
                df['close'].rolling(50).std()
            ).abs()
            
            # 🚀 NEW: Advanced Support/Resistance
            # Dynamic support and resistance
            for period in [20, 50]:
                df[f'dynamic_support_{period}'] = df['low'].rolling(period).min()
                df[f'dynamic_resistance_{period}'] = df['high'].rolling(period).max()
                df[f'price_position_{period}'] = (df['close'] - df[f'dynamic_support_{period}']) / (df[f'dynamic_resistance_{period}'] - df[f'dynamic_support_{period}'])
            
            # 🚀 NEW: Advanced Market Microstructure
            # Order flow imbalance (simplified)
            df['order_flow_imbalance'] = (
                df['volume'] * np.sign(df['close'].diff())
            ).rolling(10).sum()
            
            # Market efficiency ratio
            df['market_efficiency_ratio'] = (
                df['close'].pct_change().abs().rolling(20).mean() /
                df['close'].pct_change().rolling(20).std()
            )
            
            # 🚀 NEW: Advanced Pattern Recognition
            # Double top/bottom detection
            highs = df['high'].rolling(10).max()
            lows = df['low'].rolling(10).min()
            
            df['double_top'] = np.where(
                (highs == highs.shift(5)) & (highs > highs.shift(10)), 1, 0
            )
            df['double_bottom'] = np.where(
                (lows == lows.shift(5)) & (lows < lows.shift(10)), 1, 0
            )
            
            # 🚀 NEW: Advanced Risk Metrics
            # Maximum drawdown
            rolling_max = df['close'].rolling(50).max()
            df['drawdown'] = (df['close'] - rolling_max) / rolling_max
            
            # Risk-adjusted return
            returns = df['close'].pct_change()
            df['sharpe_ratio'] = returns.rolling(20).mean() / returns.rolling(20).std()
            
            # 🚀 NEW: Advanced Market Regime Detection
            # Volatility clustering
            df['volatility_cluster'] = (
                df['close'].pct_change().abs().rolling(5).mean() /
                df['close'].pct_change().abs().rolling(20).mean()
            )
            
            # Trend persistence
            df['trend_persistence'] = (
                df['close'].rolling(10).apply(lambda x: np.sum(np.sign(x.diff().dropna()))) / 9
            ).abs()
            
            logging.info("Added maximum intelligence features with advanced indicators")
            return df
            
        except Exception as e:
            logging.warning(f"Error adding maximum intelligence features: {e}")
            return df

    def compute_shap_values(self, model, X):
        """
        Compute SHAP values for the given model and features X.
        Returns SHAP values or None if not available.
        """
        if not SHAP_AVAILABLE:
            logging.warning("SHAP not available. Returning None.")
            return None
        try:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            return shap_values
        except Exception as e:
            logging.warning(f"SHAP computation failed: {e}")
            return None
