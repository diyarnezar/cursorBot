"""
Maximum Intelligence Feature Engineering
=======================================

Part 1: Advanced Feature Engineering & Selection
Focus: Quality over Quantity - Smart Feature Selection

This module implements the smartest possible feature engineering that:
- Keeps the BEST features (not just removes bad ones)
- Creates NEW intelligent features from existing data
- Uses advanced selection methods for maximum predictive power
- Prioritizes features that actually improve trading performance
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import talib
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MaximumIntelligenceFeatureEngineer:
    """
    Maximum Intelligence Feature Engineer
    Focus: Create the smartest possible features for maximum trading performance
    """
    
    def __init__(self):
        self.feature_importance_scores = {}
        self.selected_features = []
        self.feature_metadata = {}
        
    def create_intelligent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create intelligent features that maximize predictive power
        Focus: Quality features that actually improve trading performance
        """
        logger.info("🧠 Creating Maximum Intelligence Features...")
        
        # 1. Advanced Technical Indicators (Smart Selection)
        df = self._add_smart_technical_indicators(df)
        
        # 2. Market Microstructure Features (High Predictive Value)
        df = self._add_microstructure_intelligence(df)
        
        # 3. Regime Detection Features (Market Context)
        df = self._add_regime_intelligence(df)
        
        # 4. Volatility Intelligence (Risk Management)
        df = self._add_volatility_intelligence(df)
        
        # 5. Momentum Intelligence (Trend Following)
        df = self._add_momentum_intelligence(df)
        
        # 6. Volume Intelligence (Market Participation)
        df = self._add_volume_intelligence(df)
        
        # 7. Price Action Intelligence (Pattern Recognition)
        df = self._add_price_action_intelligence(df)
        
        # 8. Cross-Asset Intelligence (Market Relationships)
        df = self._add_cross_asset_intelligence(df)
        
        logger.info(f"🧠 Created {len(df.columns)} intelligent features")
        return df
    
    def _add_smart_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add only the most predictive technical indicators"""
        
        # RSI with multiple timeframes (most predictive)
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
        df['rsi_divergence'] = self._calculate_rsi_divergence(df)
        
        # MACD with signal analysis
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['macd_strength'] = abs(df['macd'] - df['macd_signal'])
        
        # Bollinger Bands with squeeze detection
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic with smart signals
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['stoch_signal'] = np.where(df['stoch_k'] > df['stoch_d'], 1, -1)
        
        # Williams %R (reversal indicator)
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
        
        # CCI (trend strength)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'])
        
        return df
    
    def _add_microstructure_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features (high predictive value)"""
        
        # VWAP with deviation
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Volume-weighted features
        df['volume_price_trend'] = (df['close'] - df['close'].shift(1)) * df['volume']
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Order flow indicators
        df['buy_pressure'] = np.where(df['close'] > df['open'], df['volume'], 0)
        df['sell_pressure'] = np.where(df['close'] < df['open'], df['volume'], 0)
        df['pressure_ratio'] = df['buy_pressure'].rolling(10).sum() / df['sell_pressure'].rolling(10).sum()
        
        # Spread estimation (if not available)
        df['spread_estimate'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _add_regime_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        
        # Volatility regime
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        df['volatility_regime'] = pd.cut(df['volatility_20'], 
                                       bins=[0, 0.01, 0.02, 0.05, 1], 
                                       labels=['low', 'medium', 'high', 'extreme'])
        
        # Trend regime
        df['trend_20'] = df['close'].rolling(20).mean()
        df['trend_strength'] = abs(df['close'] - df['trend_20']) / df['trend_20']
        df['trend_regime'] = np.where(df['trend_strength'] > 0.02, 'trending', 'sideways')
        
        # Volume regime
        df['volume_regime'] = np.where(df['volume'] > df['volume'].rolling(50).quantile(0.8), 
                                     'high_volume', 'normal_volume')
        
        return df
    
    def _add_volatility_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features for risk management"""
        
        # GARCH-like volatility estimation
        returns = df['close'].pct_change()
        df['volatility_garch'] = returns.rolling(20).std()
        df['volatility_clustering'] = returns.rolling(10).std() / returns.rolling(50).std()
        
        # Volatility of volatility
        df['vol_of_vol'] = df['volatility_garch'].rolling(10).std()
        
        # Realized volatility
        df['realized_vol'] = np.sqrt((returns**2).rolling(20).sum())
        
        return df
    
    def _add_momentum_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features for trend following"""
        
        # Price momentum
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'momentum_ma_{period}'] = df[f'momentum_{period}'].rolling(10).mean()
        
        # Momentum acceleration
        df['momentum_accel'] = df['momentum_10'].diff()
        
        # Momentum divergence
        df['momentum_divergence'] = df['momentum_10'] - df['momentum_20']
        
        return df
    
    def _add_volume_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based intelligence"""
        
        # Volume momentum
        df['volume_momentum'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volume-price relationship
        df['volume_price_corr'] = df['volume'].rolling(10).corr(df['close'])
        
        # On-balance volume
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['obv_momentum'] = df['obv'].pct_change()
        
        return df
    
    def _add_price_action_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action patterns"""
        
        # Candlestick patterns
        df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        
        # Support/resistance levels
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        return df
    
    def _add_cross_asset_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset relationship features"""
        
        # Correlation with major assets (if available)
        # For now, create synthetic correlation features
        df['correlation_btc'] = np.random.normal(0.7, 0.1, len(df))  # Placeholder
        df['correlation_sp500'] = np.random.normal(0.3, 0.2, len(df))  # Placeholder
        
        # Market sentiment proxy
        df['fear_greed_proxy'] = self._calculate_fear_greed_proxy(df)
        
        return df
    
    def _calculate_rsi_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI divergence (price vs RSI)"""
        rsi = talib.RSI(df['close'], timeperiod=14)
        
        # Find peaks and troughs
        price_peaks = df['close'].rolling(5, center=True).apply(lambda x: 1 if x.iloc[2] == max(x) else 0)
        rsi_peaks = rsi.rolling(5, center=True).apply(lambda x: 1 if x.iloc[2] == max(x) else 0)
        
        # Calculate divergence
        divergence = np.where((price_peaks == 1) & (rsi_peaks == 0), -1, 0)  # Bearish divergence
        divergence = np.where((price_peaks == 0) & (rsi_peaks == 1), 1, divergence)  # Bullish divergence
        
        return pd.Series(divergence, index=df.index)
    
    def _calculate_fear_greed_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate fear/greed proxy based on volatility and momentum"""
        volatility = df['close'].pct_change().rolling(20).std()
        momentum = df['close'].pct_change(20)
        
        # High volatility + negative momentum = fear
        # Low volatility + positive momentum = greed
        fear_greed = (momentum - volatility * 10) / 2
        return fear_greed.clip(-1, 1)  # Normalize to [-1, 1]
    
    def select_best_features(self, df: pd.DataFrame, target: pd.Series, max_features: int = 200) -> pd.DataFrame:
        """
        Select the best features using multiple intelligent methods
        Focus: Keep only features that actually improve predictions
        """
        logger.info(f"🧠 Selecting best {max_features} features from {len(df.columns)} candidates...")
        
        # Remove non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Method 1: Mutual Information (captures non-linear relationships)
        mi_scores = mutual_info_regression(numeric_df, target, random_state=42)
        mi_features = pd.Series(mi_scores, index=numeric_df.columns).sort_values(ascending=False)
        
        # Method 2: Correlation with target
        corr_scores = numeric_df.corrwith(target).abs().sort_values(ascending=False)
        
        # Method 3: F-statistic (linear relationships)
        f_scores = f_regression(numeric_df, target)[0]
        f_features = pd.Series(f_scores, index=numeric_df.columns).sort_values(ascending=False)
        
        # Combine scores intelligently
        combined_scores = pd.DataFrame({
            'mutual_info': mi_features,
            'correlation': corr_scores,
            'f_statistic': f_features
        }).fillna(0)
        
        # Weighted score (prioritize mutual information for non-linear relationships)
        combined_scores['final_score'] = (
            combined_scores['mutual_info'] * 0.5 +
            combined_scores['correlation'] * 0.3 +
            combined_scores['f_statistic'] * 0.2
        )
        
        # Select top features
        top_features = combined_scores['final_score'].sort_values(ascending=False).head(max_features)
        
        # Store feature importance for later use
        self.feature_importance_scores = top_features.to_dict()
        self.selected_features = top_features.index.tolist()
        
        # Select only the best features
        selected_df = df[top_features.index]
        
        logger.info(f"🧠 Selected {len(selected_df.columns)} best features")
        logger.info(f"   • Top 5 features: {list(top_features.head().index)}")
        
        return selected_df
    
    def get_feature_importance_report(self) -> Dict:
        """Get detailed feature importance report"""
        return {
            'feature_scores': self.feature_importance_scores,
            'selected_features': self.selected_features,
            'total_features': len(self.selected_features)
        } 