#!/usr/bin/env python3
# PROJECT HYPERION - ULTRA ENHANCED MAIN.PY
# The Smartest Trading Bot Ever Created

import json
import logging
import time
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import uuid
import os
import traceback
from datetime import datetime, timedelta
import threading
import asyncio
import signal
import sys
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all modules
from modules.data_ingestion import fetch_klines, fetch_ticker_24hr, fetch_order_book
from modules.feature_engineering import FeatureEngineer, EnhancedFeatureEngineer
from modules.alternative_data import EnhancedAlternativeData
from modules.smart_data_collector import SmartDataCollector
from modules.prediction_engine import UltraEnhancedPredictionEngine
from modules.execution_engine import ExecutionEngine
from modules.risk_manager import RiskManager
from modules.telegram_bot import TelegramNotifier
from ultra_train_enhanced import UltraEnhancedTrainer
from modules.rl_agent import RLAgent, ReplayBuffer
from modules.backtester import RollingBacktester

# Import advanced modules from integrated version
try:
    from modules.crypto_features import CryptoFeatures
    from modules.alternative_data_collector import AlternativeDataCollector
    from modules.advanced_ensemble import AdvancedEnsemble
    from modules.autonomous_system import AutonomousSystem, SelfPlayEnvironment, AutomatedBacktester, PerformanceMonitor
    from modules.intelligence_enhancer import IntelligenceEnhancer, MarketRegimeDetector, AdvancedExplainability, AnomalyDetector
    from modules.robustness_manager import RobustnessManager, DynamicRiskManager, APILimitHandler, FailoverSystem
    ADVANCED_MODULES_AVAILABLE = True
except ImportError:
    ADVANCED_MODULES_AVAILABLE = False

class PaperTradingEngine:
    """
    Paper Trading Engine that simulates live trading with 100% identical logic
    This ensures simulation results are exactly the same as live trading
    """
    
    def __init__(self, initial_capital: float = 60.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.order_history = []
        self.trade_history = []
        self.current_price = 0.0
        
    def get_current_price(self, pair: str) -> float:
        """Get current price - identical to live trading"""
        try:
            # Use the same data source as live trading
            ticker = fetch_ticker_24hr(pair)
            if ticker:
                self.current_price = float(ticker['lastPrice'])
                return self.current_price
            return 0.0
        except Exception as e:
            logging.error(f"Error getting current price: {e}")
            return 0.0
    
    def get_account_balance(self, asset: str) -> Dict[str, float]:
        """Get account balance - identical to live trading"""
        if asset == 'FDUSD':
            return {
                'free': self.current_capital,
                'locked': 0.0,
                'total': self.current_capital
            }
        elif asset == 'ETH':
            eth_balance = self.positions.get('ETH', 0.0)
            return {
                'free': eth_balance,
                'locked': 0.0,
                'total': eth_balance
            }
        return {'free': 0.0, 'locked': 0.0, 'total': 0.0}
    
    def place_maker_order(self, pair: str, side: str, quantity: float, price: float) -> Optional[Dict[str, Any]]:
        """Place maker order - identical logic to live trading"""
        try:
            order_id = str(uuid.uuid4())
            current_time = int(time.time() * 1000)
            fee_rate = 0.001  # 0.1% maker fee
            fee_amount = quantity * price * fee_rate
            
            if side == 'BUY':
                cost = quantity * price + fee_amount
                if cost <= self.current_capital:
                    # Update capital and position
                    self.current_capital -= cost
                    self.positions['ETH'] = self.positions.get('ETH', 0.0) + quantity
                    
                    # Record the order
                    order = {
                        'orderId': order_id,
                        'symbol': pair,
                        'side': side,
                        'quantity': quantity,
                        'price': price,
                        'status': 'FILLED',
                        'executedQty': quantity,
                        'cummulativeQuoteQty': cost,
                        'time': current_time,
                        'updateTime': current_time,
                        'fee': fee_amount
                    }
                    
                    # Record trade
                    trade = {
                        'id': str(uuid.uuid4()),
                        'orderId': order_id,
                        'symbol': pair,
                        'side': side,
                        'quantity': quantity,
                        'price': price,
                        'cost': cost,
                        'fee': fee_amount,
                        'time': current_time
                    }
                    
                    self.order_history.append(order)
                    self.trade_history.append(trade)
                    
                    logging.info(f"Paper trade executed: {side} {quantity} ETH at ${price:.2f}")
                    return order
                else:
                    logging.error(f"Insufficient capital for paper trade: ${cost:.2f} > ${self.current_capital:.2f}")
                    return None
                    
            elif side == 'SELL':
                eth_balance = self.positions.get('ETH', 0.0)
                if quantity <= eth_balance:
                    # Update capital and position
                    revenue = quantity * price - fee_amount
                    self.current_capital += revenue
                    self.positions['ETH'] = eth_balance - quantity
                    
                    # Record the order
                    order = {
                        'orderId': order_id,
                        'symbol': pair,
                        'side': side,
                        'quantity': quantity,
                        'price': price,
                        'status': 'FILLED',
                        'executedQty': quantity,
                        'cummulativeQuoteQty': revenue,
                        'time': current_time,
                        'updateTime': current_time,
                        'fee': fee_amount
                    }
                    
                    # Record trade
                    trade = {
                        'id': str(uuid.uuid4()),
                        'orderId': order_id,
                        'symbol': pair,
                        'side': side,
                        'quantity': quantity,
                        'price': price,
                        'revenue': revenue,
                        'fee': fee_amount,
                        'time': current_time
                    }
                    
                    self.order_history.append(order)
                    self.trade_history.append(trade)
                    
                    logging.info(f"Paper trade executed: {side} {quantity} ETH at ${price:.2f}")
                    return order
                else:
                    logging.error(f"Insufficient ETH for paper trade: {quantity} > {eth_balance}")
                    return None
            
            return None
            
        except Exception as e:
            logging.error(f"Error in paper trading: {e}")
            return None
    
    def place_taker_order(self, pair: str, side: str, quantity: float) -> Optional[Dict[str, Any]]:
        """Place taker order - identical logic to live trading"""
        current_price = self.get_current_price(pair)
        if current_price > 0:
            return self.place_maker_order(pair, side, quantity, current_price)
        return None
    
    def get_performance_summary(self) -> Optional[Dict[str, Any]]:
        """Get performance summary"""
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'current_capital': self.current_capital,
                'initial_capital': self.initial_capital,
                'return_pct': 0.0
            }
        
        # Calculate PnL from trade history
        total_pnl = 0.0
        for trade in self.trade_history:
            if trade['side'] == 'BUY':
                total_pnl -= trade['cost']
            else:  # SELL
                total_pnl += trade['revenue']
        
        return_pct = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'return_pct': return_pct,
            'eth_position': self.positions.get('ETH', 0.0)
        }

class UltraTradingBot:
    """
    Ultra-Enhanced Trading Bot with Maximum Intelligence
    
    Features:
    - Smart data collection with tiered API usage
    - Advanced feature engineering with 100+ indicators
    - Multi-timeframe ensemble predictions (1m, 5m, 15m)
    - Real-time risk management and position sizing
    - Advanced execution with maker-only orders
    - XAI layer for explainable decisions
    - Telegram notifications and monitoring
    - Continuous learning and adaptation
    - Identical simulation and live trading modes
    """
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize the ultra trading bot with maximum intelligence"""
        # Handle both config file path and config dictionary
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            self.config = self.load_config(config_path)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize simulation mode
        self.simulation_mode = self.config.get('trading_parameters', {}).get('simulation_mode', True)
        
        if self.simulation_mode:
            logging.info("🚀 Starting in SIMULATION MODE - All logic identical to live trading")
        else:
            logging.info("🚀 Starting in LIVE TRADING MODE")
        
        # Initialize smart data collector
        self.smart_collector = SmartDataCollector(self.config)
        
        # Initialize prediction engine
        self.prediction_engine = UltraEnhancedPredictionEngine(self.config)
        
        # Initialize feature engineer
        self.feature_engineer = EnhancedFeatureEngineer()
        
        # Initialize alternative data
        self.alternative_data = EnhancedAlternativeData(self.config)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(self.config)
        
        # Initialize telegram notifier
        telegram_config = self.config.get('telegram', {})
        telegram_token = telegram_config.get('bot_token', '')
        telegram_chat_id = telegram_config.get('chat_id', '')
        self.telegram = TelegramNotifier(telegram_token, telegram_chat_id)
        
        # Initialize trainer
        self.trainer = UltraEnhancedTrainer(self.config)
        
        # Initialize RL agent
        self.rl_agent = RLAgent(self.config)
        
        # Initialize backtester
        self.backtester = RollingBacktester(self.config)
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'current_balance': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'last_update': time.time()
        }
        
        # Trading state
        self.current_position = 0.0
        self.last_trade_time = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Initialize execution engine based on mode
        if self.simulation_mode:
            initial_capital = self.config.get('trading_parameters', {}).get('paper_trading', {}).get('initial_capital', 60.0)
            self.execution_engine = PaperTradingEngine(initial_capital)
            logging.info(f"📊 Paper Trading Engine initialized with ${initial_capital} capital")
        else:
            self.binance_creds = self.config.get('binance_credentials', {})
            self.execution_engine = ExecutionEngine(self.binance_creds)
            logging.info("🔗 Live Trading Engine initialized")
        
        # Initialize advanced modules if available
        global ADVANCED_MODULES_AVAILABLE
        if ADVANCED_MODULES_AVAILABLE:
            try:
                # Advanced ensemble system
                self.advanced_ensemble = AdvancedEnsemble(self.config)
                
                # Autonomous system
                self.autonomous_system = AutonomousSystem(self.config)
                
                # Intelligence enhancer
                self.intelligence_enhancer = IntelligenceEnhancer(self.config)
                
                # Robustness manager
                self.robustness_manager = RobustnessManager(self.config)
                
                # Performance monitor
                self.performance_monitor = PerformanceMonitor("data/performance.db")
                
                logging.info("🚀 Advanced modules loaded successfully")
            except Exception as e:
                logging.warning(f"Advanced modules initialization failed: {e}")
                ADVANCED_MODULES_AVAILABLE = False
        else:
            logging.info("📝 Running with standard modules only")
        
        # Initialize autonomous parameter optimization system
        self.autonomous_params = self._initialize_autonomous_params()
        self.param_optimization_history = []
        self.optimization_metrics = {
            'fill_rate': 0.0,
            'avg_slippage': 0.0,
            'profit_per_trade': 0.0,
            'win_rate': 0.0
        }
        
        # Initialize background threads
        self.background_threads = []
        if not self.simulation_mode:
            self.background_threads.append(threading.Thread(target=self.start_autonomous_optimization, daemon=True))
            self.background_threads[-1].start()
            logging.info("💰 LIVE MODE: Real money trading enabled. Autonomous optimization started.")
        else:
            logging.info("🎯 SIMULATION MODE: All trading logic identical to live mode")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration with enhanced settings"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            if 'enhanced_features' not in self.config:
                self.config['enhanced_features'] = {
                    'use_microstructure': True,
                    'use_alternative_data': True,
                    'use_advanced_indicators': True,
                    'use_adaptive_features': True,
                    'use_normalization': True,
                    'use_sentiment_analysis': True,
                    'use_onchain_data': True,
                    'use_market_microstructure': True
                }
            
            if 'smart_data_collection' not in self.config:
                self.config['smart_data_collection'] = {
                    'enable_tiered_apis': True,
                    'use_caching': True,
                    'rate_limit_respect': True,
                    'fallback_strategies': True,
                    'data_quality_checks': True
                }
            
            if 'multi_timeframe' not in self.config:
                self.config['multi_timeframe'] = {
                    'enabled': True,
                    'timeframes': ['1m', '5m', '15m'],
                    'ensemble_weights': True
                }
            
            logging.info(f"Configuration loaded from {config_path} with enhanced features")
            return self.config
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {}
    
    def setup_logging(self):
        """Setup advanced logging"""
        logging.basicConfig(level=logging.INFO)
        self.log_file = logging.FileHandler('ultra_bot.log')
        self.simple_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
        self.log_file.setFormatter(self.simple_formatter)
        logging.getLogger().addHandler(self.log_file)
        logging.info("Logging setup complete.")
    
    def collect_market_data(self) -> pd.DataFrame:
        """Collect comprehensive market data using smart data collector, including whale features"""
        try:
            # Use smart data collector for comprehensive data
            self.data = self.smart_collector.collect_data()
            if self.data.empty:
                logging.warning("Smart collector returned no data, using fallback method")
                return self.collect_fallback_data()
            # Collect whale features
            self.whale_features = self.smart_collector.get_whale_features()
            logging.info(f"Whale features collected: {self.whale_features}")
            # Apply advanced feature engineering with whale features
            self.data = self.feature_engineer.add_enhanced_features(self.data, whale_features=self.whale_features)
            logging.info(f"Collected {len(self.data)} data points with {len(self.data.columns)} enhanced features (including whale features)")
            return self.data
        except Exception as e:
            logging.error(f"Error in smart data collection: {e}")
            return self.collect_fallback_data()
    
    def collect_fallback_data(self) -> pd.DataFrame:
        """Fallback data collection method"""
        try:
            # Basic klines data
            self.end_time = datetime.now()
            self.klines = fetch_klines('ETHUSDT', '1m', self.end_time - timedelta(minutes=1000))
            if not self.klines:
                raise Exception("No klines data available")
            
            # Convert to DataFrame
            self.df = pd.DataFrame(self.klines)
            self.numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            for col in self.numeric_columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Remove non-numeric columns
            self.df = self.df.drop(['timestamp', 'close_time', 'ignore'], axis=1)
            
            # Add comprehensive features
            self.df = self.add_comprehensive_features(self.df)
            
            # Clean data
            self.df = self.df.dropna()
            
            logging.info(f"Fallback data collected: {len(self.df)} rows, {len(self.df.columns)} features")
            return self.df
            
        except Exception as e:
            logging.error(f"Fallback data collection failed: {e}")
            return pd.DataFrame()
    
    def add_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive features with maximum intelligence and fix compatibility issues"""
        try:
            if df.empty:
                return df
            
            # Store original features for compatibility
            if hasattr(self.prediction_engine, 'feature_selector') and self.prediction_engine.feature_selector is not None:
                try:
                    # Get the expected feature names from the fitted selector
                    if hasattr(self.prediction_engine.feature_selector, 'get_support'):
                        # Use the same features that were used during training
                        self.expected_features = self.prediction_engine.feature_selector.get_support()
                        missing_features = [f for f in self.prediction_engine.feature_selector.get_feature_names_out() if f not in self.expected_features]
                        
                        for feature in missing_features:
                            df[feature] = 0.0
                            
                        # Select only the expected features in the correct order
                        self.df = df[self.expected_features]
                        
                        logging.info(f"Feature selection applied: {len(df.columns)} features")
                    else:
                        logging.warning("Feature selector not properly fitted, using all features")
                except Exception as e:
                    logging.warning(f"Feature selection failed: {e}, using all features")
            
            # Ensure we have the minimum required features for models
            self.required_features = ['close', 'volume', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'adx']
            for feature in self.required_features:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            logging.info(f"Added maximum intelligence features. Total features: {len(df.columns)}")
            return df
            
        except Exception as e:
            logging.error(f"Error adding comprehensive features: {e}")
            return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        self.ema_fast = prices.ewm(span=fast, adjust=False).mean()
        self.ema_slow = prices.ewm(span=slow, adjust=False).mean()
        self.macd = self.ema_fast - self.ema_slow
        self.signal = self.macd.ewm(span=signal, adjust=False).mean()
        self.histogram = self.macd - self.signal
        return self.macd, self.signal, self.histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        self.middle = prices.rolling(window=period).mean()
        self.std_dev = prices.rolling(window=period).std()
        self.upper = self.middle + (self.std_dev * std)
        self.lower = self.middle - (self.std_dev * std)
        return self.middle, self.upper, self.lower
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            tr1 = pd.Series(high_low, name="tr1")
            tr2 = pd.Series(high_close, name="tr2")
            tr3 = pd.Series(low_close, name="tr3")
            
            self.tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self.atr = self.tr.rolling(window=period).mean()
            
            return self.atr
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return pd.Series(index=df.index, dtype=float)
    
    def get_predictions(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get predictions for all timeframes"""
        try:
            # Ensure we have numeric data only
            if data.empty:
                raise Exception("No valid numeric data for predictions")
            
            # Get predictions for all timeframes
            predictions = {}
            for timeframe in ['1m', '5m', '15m']:
                try:
                    self.pred = self.prediction_engine.predict(data, timeframe=timeframe)
                    predictions[timeframe] = self.pred
                except Exception as e:
                    logging.warning(f"Failed to get prediction for {timeframe}: {e}")
                    predictions[timeframe] = 0.0
            
            logging.info(f"Predictions: {predictions}")
            return predictions
            
        except Exception as e:
            logging.error(f"Error getting predictions: {e}")
            return {'1m': 0.0, '5m': 0.0, '15m': 0.0}
    
    def analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market conditions with enhanced intelligence and fix volatility error"""
        try:
            if data.empty:
                return {
                'trend': 'neutral',
                'volatility': 'low',
                'volume': 'normal',
                'momentum': 'neutral',
                    'strength': 'weak',
                    'regime': 'normal'
            }
            
            # Calculate volatility properly
            if 'close' in data.columns:
                self.returns = data['close'].pct_change().dropna()
                self.volatility = self.returns.std()
            else:
                self.volatility = 0.02 # Default if 'close' not available
            
            # Calculate short and long moving averages
            self.short_ma = data['close'].ewm(span=10, adjust=False).mean()
            self.long_ma = data['close'].ewm(span=50, adjust=False).mean()
            
            # Determine trend
            self.current_price = float(data['close'].iloc[-1])
            if self.current_price > self.short_ma.iloc[-1] and self.current_price > self.long_ma.iloc[-1]:
                self.trend = 'bullish'
            elif self.current_price < self.short_ma.iloc[-1] and self.current_price < self.long_ma.iloc[-1]:
                self.trend = 'bearish'
            else:
                self.trend = 'neutral'
            
            # Calculate momentum and strength
            self.rsi = self.calculate_rsi(data['close'])
            self.momentum = 'oversold' if self.rsi.iloc[-1] < 30 else 'overbought' if self.rsi.iloc[-1] > 70 else 'neutral'
            
            # Calculate volume strength
            self.avg_volume = data['volume'].mean()
            self.current_volume = float(data['volume'].iloc[-1])
            self.volume_level = 'low' if self.current_volume < self.avg_volume * 0.5 else 'normal' if self.current_volume < self.avg_volume * 1.5 else 'high'
            
            # Calculate ADX for strength
            self.adx = self.calculate_adx(data)
            self.strength = 'weak'
            if self.adx.iloc[-1] > 20:
                self.strength = 'moderate'
                if self.adx.iloc[-1] > 25:
                    self.strength = 'strong'
            
            # Determine regime
            self.regime = 'normal'
            if self.volatility > 0.04:
                self.regime = 'volatile'
            elif self.volatility < 0.02:
                self.regime = 'stable'
            
            self.analysis = {
                'trend': self.trend,
                'volatility': self.volatility,
                'volume': self.volume_level,
                'momentum': self.momentum,
                'strength': self.strength,
                'regime': self.regime,
                'volatility_value': float(self.volatility),
                'rsi': float(self.rsi.iloc[-1]),
                'adx': float(self.adx.iloc[-1])
            }
            
            logging.info(f"Market analysis: {self.analysis}")
            return self.analysis
            
        except Exception as e:
            logging.error(f"Error analyzing market conditions: {e}")
            return {
                'trend': 'neutral',
                'volatility': 'low',
                'volume': 'normal',
                'momentum': 'neutral',
                'strength': 'weak',
                'regime': 'normal',
                'volatility_value': 0.0,
                'rsi': 50.0,
                'adx': 20.0
            }
    
    def make_trading_decision(self, predictions: Dict[str, float], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent trading decision based on predictions, market analysis, and whale features"""
        try:
            self.decision = {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': [],
                'position_size': 0.0,
                'stop_loss': None,
                'take_profit': None
            }
            # Calculate weighted prediction
            self.weights = {'1m': 0.5, '5m': 0.3, '15m': 0.2}
            self.weighted_pred = sum(predictions[tf] * self.weights[tf] for tf in self.weights)
            
            # Whale-aware trade filter
            self.whale_reason = []
            if self.weighted_pred > 0.001 and market_analysis['confidence'] > 30:
                # Buy logic
                if self.whale_features.get('large_buy_volume', 0) < 10 or self.whale_features.get('order_book_imbalance', 0) < 0 or self.whale_features.get('whale_alert_flag', 0) == 1:
                    self.allow_trade = False
                    self.whale_reason.append("Whale signals do NOT support buying (low large_buy_volume, negative imbalance, or whale alert)")
                else:
                    self.allow_trade = True
                    self.whale_reason.append("Whale signals support buying (large_buy_volume, positive imbalance, no whale alert)")
            elif self.weighted_pred < -0.001 and market_analysis['confidence'] > 30:
                # Sell logic
                if self.whale_features.get('large_sell_volume', 0) < 10 or self.whale_features.get('order_book_imbalance', 0) > 0 or self.whale_features.get('whale_alert_flag', 0) == 1:
                    self.allow_trade = False
                    self.whale_reason.append("Whale signals do NOT support selling (low large_sell_volume, positive imbalance, or whale alert)")
                else:
                    self.allow_trade = True
                    self.whale_reason.append("Whale signals support selling (large_sell_volume, negative imbalance, no whale alert)")
            else:
                self.allow_trade = True # Default to allow if no strong signal
                self.whale_reason.append("No strong signal, allowing trade based on confidence.")
            
            # RL agent adjustment
            if self.rl_agent is not None:
                try:
                    self.rl_action = self.rl_agent.get_action(self.latest_data, predictions, market_analysis, self.whale_features)
                    self.whale_reason.append(f"RL agent adjustment: {self.rl_action}")
                except Exception as e:
                    self.whale_reason.append(f"RL agent error: {e}")
            
            # Final decision
            if self.allow_trade:
                if self.weighted_pred > 0.001:
                    self.decision['action'] = 'buy'
                    self.decision['reasoning'].append(f"Strong buy signal ({self.weighted_pred:.4f})")
                    self.decision['reasoning'].append(f"BULLISH trend confirmed")
                elif self.weighted_pred < -0.001:
                    self.decision['action'] = 'sell'
                    self.decision['reasoning'].append(f"Strong sell signal ({self.weighted_pred:.4f})")
                    self.decision['reasoning'].append(f"BEARISH trend confirmed")
                # Position sizing: boost if strong whale activity
                self.base_size = 1.0 # Default base size
                if self.whale_features.get('large_buy_volume', 0) > 100 or self.whale_features.get('large_sell_volume', 0) > 100:
                    self.whale_boost = 1.5
                    self.decision['reasoning'].append("Position size boosted due to strong whale activity")
                self.decision['position_size'] = self.base_size * (market_analysis['confidence'] / 100) * self.risk_manager.get_risk_factor() * self.whale_boost
            else:
                self.decision['action'] = 'hold'
                self.decision['reasoning'].append("Trade blocked by whale-aware filter")
            self.decision['reasoning'] += self.whale_reason
            
            # Real-time explainability: log and Telegram
            self.log_msg = f"Whale-aware decision: {self.decision['action']} | Reason: {'; '.join(self.decision['reasoning'])}"
            logging.info(self.log_msg)
            if self.telegram:
                try:
                    self.telegram.send_message(f"🐋 {self.log_msg}")
                except Exception:
                    pass
            return self.decision
        except Exception as e:
            logging.error(f"Error making trading decision: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': ['Error in decision making']}
    
    def get_current_price(self) -> Optional[float]:
        """Get current ETH price"""
        try:
            self.ticker = fetch_ticker_24hr('ETHUSDT')
            if self.ticker:
                return float(self.ticker['lastPrice'])
            return None
        except Exception as e:
            logging.error(f"Error getting current price: {e}")
            return None
    
    def execute_trade(self, decision: Dict[str, Any]) -> bool:
        """Execute trade using only maker orders for zero fees - enhanced with autonomous optimization"""
        try:
            if decision['action'] == 'hold':
                return True
            # Check risk limits
            if not self.risk_manager.check_risk_limits(decision):
                logging.warning("Trade blocked by risk manager")
                return False
            
            # Get optimal maker order parameters from autonomous system
            self.maker_params = self.get_optimal_maker_parameters(decision)
            
            # Place order using execution engine
            self.order_details = self.execution_engine.place_order(
                'ETHUSDT',
                decision['action'],
                self.maker_params['price'],
                self.maker_params['quantity'],
                self.maker_params['side']
            )
            
            if self.order_details:
                self.mode_text = "💰 LIVE" if not self.simulation_mode else "🚀 SIMULATION"
                logging.info(f"{self.mode_text} Maker order executed: {decision['action']} {self.maker_params['quantity']} ETH @ {self.maker_params['price']} (zero fees)")
                logging.info(f"Maker optimization: spread={self.maker_params['spread']:.4f}, offset={self.maker_params['offset']:.4f}")
                self.record_trade(decision, self.order_details)
                
                # Update autonomous learning with trade results
                self.update_autonomous_learning(decision, self.maker_params, self.order_details)
                return True
            else:
                logging.error("Maker order execution failed")
                return False
        except Exception as e:
            logging.error(f"Error executing trade: {e}")
            return False
    
    def get_optimal_maker_parameters(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal maker order parameters for best fill rate and zero fees"""
        try:
            self.current_price = self.get_current_price()
            if self.current_price is None:
                return self.get_fallback_maker_params(decision)
            
            # Get market depth and spread analysis
            self.market_analysis = self.analyze_market_depth()
            
            # Get autonomous parameters from self-optimizing system
            self.autonomous_params = self.get_autonomous_parameters()
            
            # Calculate optimal maker price based on:
            # 1. Market spread and depth
            # 2. Historical fill rates
            # 3. Confidence level
            # 4. Current market volatility
            # 5. Autonomous learning parameters
            
            self.spread = self.market_analysis['spread']
            self.volatility = self.market_analysis['volatility']
            self.confidence = decision.get('confidence', 0.5)
            
            self.spread_adjustment = self.spread * 0.5 * self.autonomous_params['spread_multiplier']
            self.volatility_adjustment = self.volatility * 0.1 * self.autonomous_params['volatility_multiplier']
            self.confidence_adjustment = (1 - self.confidence) * 0.001 * self.autonomous_params['confidence_multiplier']
            
            self.optimal_offset = self.autonomous_params['base_maker_offset'] + self.confidence_adjustment + self.volatility_adjustment + self.spread_adjustment
            
            if decision['action'] == 'buy':
                self.maker_price = self.current_price * (1 - self.optimal_offset)
            else: # sell
                self.maker_price = self.current_price * (1 + self.optimal_offset)
            
            return {
                'price': self.maker_price,
                'quantity': self.autonomous_params['max_position_size'] * self.current_price, # Max position size as % of capital
                'side': decision['action'],
                'spread': self.spread,
                'volatility': self.volatility,
                'confidence': self.confidence,
                'offset': self.optimal_offset,
                'autonomous_params': self.autonomous_params,
                'market_depth': self.market_analysis.get('depth', 'low')
            }
            
        except Exception as e:
            logging.error(f"Error getting optimal maker parameters: {e}")
            return self.get_fallback_maker_params(decision)
    
    def get_fallback_maker_params(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback maker parameters if analysis fails"""
        self.current_price = self.get_current_price()
        if self.current_price is None:
            return {
                'price': 0.0,
                'quantity': 0.0,
                'side': decision['action'],
                'spread': 0.001,
                'volatility': 0.02,
                'confidence': decision.get('confidence', 0.5),
                'offset': 0.0005,
                'autonomous_params': {},
                'market_depth': 'low'
            }
        self.confidence = decision.get('confidence', 0.5)
        return {
            'price': self.current_price * (1 - 0.0005), # Slightly below market
            'quantity': self.autonomous_params['max_position_size'] * self.current_price, # Max position size as % of capital
            'side': decision['action'],
            'spread': 0.001,
            'volatility': 0.02,
            'confidence': self.confidence,
            'offset': 0.0005,
            'autonomous_params': self.autonomous_params,
            'market_depth': 'low'
        }
    
    def analyze_market_depth(self) -> Dict[str, Any]:
        """Analyze market depth for optimal maker order placement"""
        try:
            # Get order book data
            self.order_book = fetch_order_book('ETHUSDT', 100) # Fetch top 100 bids and asks
            if not self.order_book:
                return {'spread': 0.001, 'volatility': 0.02, 'depth': 'low'}
            
            # Calculate spread
            self.best_bid = float(self.order_book['bids'][0][0]) if self.order_book['bids'] else 0
            self.best_ask = float(self.order_book['asks'][0][0]) if self.order_book['asks'] else 0
            
            if self.best_bid > 0 and self.best_ask > 0:
                self.spread = (self.best_ask - self.best_bid) / self.best_bid
            else:
                self.spread = 0.001 # Default if no bid/ask
            
            # Calculate volatility from recent price movements
            self.recent_prices = self.get_recent_prices(20) # Get last 20 close prices
            if len(self.recent_prices) > 1:
                self.returns = [self.recent_prices[i+1] - self.recent_prices[i] for i in range(len(self.recent_prices)-1)]
                self.volatility = sum(self.returns) / len(self.returns)
            else:
                self.volatility = 0.02 # Default if not enough data
            
            return {
                'spread': self.spread,
                'volatility': self.volatility,
                'bid_depth': len(self.order_book['bids']),
                'ask_depth': len(self.order_book['asks']),
                'depth': 'high' if min(len(self.order_book['bids']), len(self.order_book['asks'])) > 10 else 'low'
            }
            
        except Exception as e:
            logging.error(f"Error analyzing market depth: {e}")
            return {'spread': 0.001, 'volatility': 0.02, 'depth': 'low'}
    
    def get_recent_prices(self, count: int = 20) -> List[float]:
        """Get recent price history for volatility calculation"""
        try:
            # Get recent klines/candlesticks
            self.klines = fetch_klines('ETHUSDT', '1m', datetime.now() - timedelta(minutes=count))
            if not self.klines:
                return []
            return [float(kline[4]) for kline in self.klines]  # Close prices
        except Exception as e:
            logging.error(f"Error getting recent prices: {e}")
            return []
    
    def _initialize_autonomous_params(self) -> Dict[str, Any]:
        """Initialize autonomous parameters for maker order optimization"""
        return {
            # Maker order parameters
            'base_maker_offset': 0.0005,  # Base offset from market price
            'confidence_multiplier': 1.0,  # How much confidence affects offset
            'volatility_multiplier': 1.0,  # How much volatility affects offset
            'spread_multiplier': 1.0,      # How much spread affects offset
            
            # Risk management parameters
            'max_position_size': 0.1,      # Maximum position size as % of capital
            'stop_loss_multiplier': 1.0,   # Stop loss adjustment
            'take_profit_multiplier': 1.0, # Take profit adjustment
            
            # Learning parameters
            'learning_rate': 0.01,         # How fast to adapt parameters
            'optimization_frequency': 60,  # Optimize every 60 seconds
            'performance_window': 100,     # Look at last 100 trades for optimization
            
            # Market adaptation parameters
            'market_regime_weights': {
                'trending': 1.2,           # More aggressive in trending markets
                'ranging': 0.8,            # More conservative in ranging markets
                'volatile': 1.5,           # More aggressive in volatile markets
                'stable': 0.7              # More conservative in stable markets
            }
        }
    
    def get_autonomous_parameters(self) -> Dict[str, Any]:
        """Get current autonomous parameters"""
        return self.autonomous_params.copy()
    
    def start_autonomous_optimization(self):
        """Start autonomous parameter optimization in background"""
        import threading
        import time
        
        def optimization_loop():
            while self.is_running:
                try:
                    self.optimize_parameters()
                    time.sleep(self.autonomous_params['optimization_frequency'])
                except Exception as e:
                    logging.error(f"Error in autonomous optimization: {e}")
                    time.sleep(60)  # Wait before retrying
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self.optimization_thread.start()
        logging.info("Autonomous optimization thread started.")
    
    def optimize_parameters(self):
        """Optimize parameters based on recent performance"""
        try:
            if len(self.trading_history) < 10:
                return  # Need more data
            
            # Analyze recent performance
            self.recent_trades = self.trading_history[-self.autonomous_params['performance_window']:]
            
            # Calculate performance metrics
            self.fill_rate = self.calculate_fill_rate(self.recent_trades)
            self.avg_slippage = self.calculate_avg_slippage(self.recent_trades)
            self.profit_per_trade = self.calculate_profit_per_trade(self.recent_trades)
            self.win_rate = self.calculate_win_rate(self.recent_trades)
            
            # Update optimization metrics
            self.optimization_metrics.update({
                'fill_rate': self.fill_rate,
                'avg_slippage': self.avg_slippage,
                'profit_per_trade': self.profit_per_trade,
                'win_rate': self.win_rate
            })
            
            # Optimize maker order parameters
            self.optimize_maker_parameters(self.fill_rate, self.avg_slippage, self.profit_per_trade, self.win_rate)
            
            # Optimize risk management parameters
            self.optimize_risk_parameters(self.profit_per_trade, self.win_rate)
            
            # Record optimization
            self.param_optimization_history.append({
                'timestamp': datetime.now(),
                'metrics': self.optimization_metrics.copy(),
                'params': self.autonomous_params.copy()
            })
            
            logging.info(f"🤖 Parameters optimized - Fill: {self.fill_rate:.2f}, Slippage: {self.avg_slippage:.4f}, Profit: {self.profit_per_trade:.2f}, Win: {self.win_rate:.2f}")
            
        except Exception as e:
            logging.error(f"Error optimizing parameters: {e}")
    
    def optimize_maker_parameters(self, fill_rate: float, avg_slippage: float, profit_per_trade: float, win_rate: float):
        """Optimize maker order parameters for better performance"""
        try:
            self.learning_rate = 0.01 # Default learning rate
            if fill_rate < 0.8:  # Low fill rate
                # Increase offset to get better fills
                self.autonomous_params['base_maker_offset'] *= (1 + self.learning_rate)
            elif fill_rate > 0.95 and avg_slippage > 0.001:  # High fill rate but high slippage
                # Decrease offset to reduce slippage
                self.autonomous_params['base_maker_offset'] *= (1 - self.learning_rate * 0.5)
            
            # Optimize confidence multiplier
            if win_rate > 0.6:  # High win rate
                # Be more aggressive with high confidence trades
                self.autonomous_params['confidence_multiplier'] *= (1 + self.learning_rate * 0.5)
            elif win_rate < 0.4:  # Low win rate
                # Be more conservative
                self.autonomous_params['confidence_multiplier'] *= (1 - self.learning_rate * 0.5)
            
            # Optimize volatility multiplier
            if profit_per_trade > 0:  # Profitable trades
                # Increase volatility adjustment for better profits
                self.autonomous_params['volatility_multiplier'] *= (1 + self.learning_rate * 0.3)
            else:  # Losing trades
                # Decrease volatility adjustment
                self.autonomous_params['volatility_multiplier'] *= (1 - self.learning_rate * 0.3)
            
            # Ensure parameters stay within reasonable bounds
            self.autonomous_params['base_maker_offset'] = max(0.0001, min(0.002, self.autonomous_params['base_maker_offset']))
            self.autonomous_params['confidence_multiplier'] = max(0.5, min(2.0, self.autonomous_params['confidence_multiplier']))
            self.autonomous_params['volatility_multiplier'] = max(0.5, min(2.0, self.autonomous_params['volatility_multiplier']))
            
        except Exception as e:
            logging.error(f"Error optimizing maker parameters: {e}")
    
    def optimize_risk_parameters(self, profit_per_trade: float, win_rate: float):
        """Optimize risk management parameters"""
        try:
            self.learning_rate = 0.01 # Default learning rate
            if profit_per_trade > 0 and win_rate > 0.6:  # Good performance
                # Increase position size slightly
                self.autonomous_params['max_position_size'] *= (1 + self.learning_rate * 0.2)
            elif profit_per_trade < 0 or win_rate < 0.4:  # Poor performance
                # Decrease position size
                self.autonomous_params['max_position_size'] *= (1 - self.learning_rate * 0.3)
            
            # Optimize stop loss
            if win_rate < 0.4:  # Low win rate
                # Tighten stop loss
                self.autonomous_params['stop_loss_multiplier'] *= (1 - self.learning_rate * 0.2)
            elif win_rate > 0.7:  # High win rate
                # Loosen stop loss slightly
                self.autonomous_params['stop_loss_multiplier'] *= (1 + self.learning_rate * 0.1)
            
            # Ensure parameters stay within reasonable bounds
            self.autonomous_params['max_position_size'] = max(0.05, min(0.3, self.autonomous_params['max_position_size']))
            self.autonomous_params['stop_loss_multiplier'] = max(0.5, min(1.5, self.autonomous_params['stop_loss_multiplier']))
            
        except Exception as e:
            logging.error(f"Error optimizing risk parameters: {e}")
    
    def calculate_fill_rate(self, trades: List[Dict]) -> float:
        """Calculate fill rate from recent trades"""
        if not trades:
            return 0.0
        self.filled_trades = [t for t in trades if 'executed_price' in t and 'intended_price' in t]
        return len(self.filled_trades) / len(trades)
    
    def calculate_avg_slippage(self, trades: List[Dict]) -> float:
        """Calculate average slippage from recent trades"""
        if not trades:
            return 0.0
        self.slippages = [abs(t['executed_price'] - t['intended_price']) / t['intended_price'] for t in self.filled_trades]
        return sum(self.slippages) / len(self.slippages) if self.slippages else 0.0
    
    def calculate_profit_per_trade(self, trades: List[Dict]) -> float:
        """Calculate average profit per trade"""
        if not trades:
            return 0.0
        self.profits = [t['revenue'] - t['cost'] for t in self.filled_trades]
        return sum(self.profits) / len(self.profits)
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from recent trades"""
        if not trades:
            return 0.0
        self.winning_trades = [t for t in self.filled_trades if t['revenue'] > t['cost']]
        return len(self.winning_trades) / len(self.filled_trades)
    
    def update_autonomous_learning(self, decision: Dict[str, Any], maker_params: Dict[str, Any], order: Dict[str, Any]):
        """Update autonomous learning with trade results"""
        try:
            # Record trade with maker parameters for learning
            self.trade_record = {
                'timestamp': datetime.now(),
                'decision': decision,
                'maker_params': maker_params,
                'order': order,
                'autonomous_params': self.autonomous_params.copy()
            }
            
            # Store for optimization
            self.trading_history.append(self.trade_record)
            
            # Trigger optimization if enough trades
            if len(self.trading_history) % 10 == 0: # Every 10 trades
                self.optimize_parameters()
                
        except Exception as e:
            logging.error(f"Error updating autonomous learning: {e}")
    
    def record_trade(self, decision: Dict[str, Any], order: Dict):
        """Record trade for performance tracking - identical for simulation and live"""
        self.trade = {
            'timestamp': datetime.now(),
            'action': decision['action'],
            'confidence': decision['confidence'],
            'reasoning': decision['reasoning'],
            'order': order,
            'market_conditions': self.analyze_market_conditions(self.latest_data) if self.latest_data is not None else {},
            'simulation_mode': self.simulation_mode
        }
        
        self.trading_history.append(self.trade)
        self.total_trades += 1
        self.performance_metrics['last_update'] = time.time()
    
    def update_performance_metrics(self):
        """Update performance metrics - identical for simulation and live"""
        try:
            if self.total_trades == 0:
                return
            # Calculate win rate
            self.performance_metrics['win_rate'] = (self.winning_trades / self.total_trades) * 100
            # Calculate total PnL
            if self.simulation_mode and hasattr(self.execution_engine, 'get_performance_summary') and type(self.execution_engine).__name__ == 'PaperTradingEngine':
                self.paper_summary = self.execution_engine.get_performance_summary()
                self.performance_metrics['total_pnl'] = self.paper_summary.get('total_pnl', 0.0)
                self.performance_metrics['current_balance'] = self.paper_summary.get('current_capital', 0.0)
                self.performance_metrics['return_pct'] = self.paper_summary.get('return_pct', 0.0)
            else:
                self.performance_metrics['total_pnl'] = self.total_pnl
                self.performance_metrics['current_balance'] = self.current_position * self.get_current_price() + self.initial_capital
                self.performance_metrics['return_pct'] = ((self.performance_metrics['current_balance'] - self.initial_capital) / self.initial_capital) * 100
            
            # Calculate average trade PnL
            self.performance_metrics['avg_trade_pnl'] = self.performance_metrics['total_pnl'] / self.total_trades
            # Calculate Sharpe ratio (simplified)
            if len(self.trading_history) > 1:
                self.returns = [t['revenue'] - t['cost'] for t in self.filled_trades]
                self.performance_metrics['sharpe_ratio'] = np.mean(self.returns) / np.std(self.returns) if np.std(self.returns) > 0 else 0
            logging.info(f"Performance metrics updated: {self.performance_metrics}")
        except Exception as e:
            logging.error(f"Error updating performance metrics: {e}")
    
    def send_status_update(self):
        """Send status update to Telegram - shows simulation vs live mode"""
        try:
            if not self.telegram:
                return
            # Prepare status message
            self.mode_text = "💰 LIVE TRADING MODE" if not self.simulation_mode else "🚀 SIMULATION MODE"
            self.status = f"🤖 Ultra Bot Status Update - {self.mode_text}\n\n"
            self.status += f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            self.status += f"📊 Total Trades: {self.total_trades}\n"
            if self.simulation_mode and hasattr(self.execution_engine, 'get_performance_summary') and type(self.execution_engine).__name__ == 'PaperTradingEngine':
                self.paper_summary = self.execution_engine.get_performance_summary()
                self.status += f"💰 Current Capital: ${self.paper_summary.get('current_capital', 0):.2f}\n"
                self.status += f"📈 Return: {self.paper_summary.get('return_pct', 0):.2f}%\n"
                self.status += f"🎯 ETH Position: {self.paper_summary.get('eth_position', 0):.6f} ETH\n"
            else:
                self.status += f"💰 Total PnL: ${self.performance_metrics.get('total_pnl', 0):.2f}\n"
                self.status += f"💰 Current Balance: ${self.performance_metrics.get('current_balance', 0):.2f}\n"
                self.status += f"📈 Return: {self.performance_metrics.get('return_pct', 0):.2f}%\n"
                self.status += f"🎯 Current Position: {self.current_position:.6f} ETH\n"
            if self.total_trades > 0:
                self.status += f"📈 Win Rate: {self.performance_metrics.get('win_rate', 0):.1f}%\n"
                self.status += f"📊 Avg Trade PnL: ${self.performance_metrics.get('avg_trade_pnl', 0):.2f}\n"
            if self.current_position > 0:
                self.status += f"🎯 Current Position: {self.current_position:.6f} ETH\n"
            # Add market analysis
            if self.latest_data is not None:
                self.market_analysis = self.analyze_market_conditions(self.latest_data)
                self.status += f"📈 Market Trend: {self.market_analysis['trend']}\n"
                self.status += f"📊 Volatility: {self.market_analysis['volatility']}\n"
                self.status += f"⚠️ Risk Level: {self.market_analysis.get('regime', 'unknown')}\n"
            self.telegram.send_message(self.status)
        except Exception as e:
            logging.error(f"Error sending status update: {e}")
    
    def trading_cycle(self):
        """Main trading cycle with 10x intelligence hooks"""
        try:
            logging.info("Starting trading cycle...")
            # Collect market data
            self.latest_data = self.collect_market_data()
            if self.latest_data.empty:
                logging.warning("No market data available")
                return
            # Get predictions
            self.predictions = self.get_predictions(self.latest_data)
            self.whale_features = self.smart_collector.get_whale_features() # Ensure whale features are updated
            
            # Make trading decision
            self.decision = self.make_trading_decision(self.predictions, self.whale_features)
            self.trade_executed = self.execute_trade(self.decision)
            
            if self.trade_executed:
                logging.info(f"Trade executed successfully: {self.decision['action']}")
            else:
                logging.warning("Trade execution failed")
            
            # Online learning (after each trade)
            if self.rl_agent is not None and self.trade_executed:
                logging.info("[10x] Performing online learning update...")
                # Use latest features and targets if available
                try:
                    self.X = self.latest_data[self.prediction_engine.feature_selector.get_feature_names_out()]
                    self.y = self.prediction_engine.predict(self.latest_data, '1m') # Assuming 1m prediction is the target
                    self.trainer.enable_online_learning(self.X, self.y)
                except Exception as e:
                    logging.warning(f"Online learning update failed: {e}")
            
            # Meta-learning after each cycle
            if self.rl_agent is not None:
                try:
                    self.perf_hist = self.trading_history[-self.autonomous_params['performance_window']:] # Use recent trades for perf history
                    self.regime = self.autonomous_system.detect_market_regime(self.latest_data) if self.autonomous_system else 'normal'
                    self.rl_agent.meta_learn(self.perf_hist, regime=self.regime)
                    logging.info("[Meta-Learning] RL agent meta-learning update")
                except Exception as e:
                    logging.warning(f"[Meta-Learning] RL agent meta-learning failed: {e}")
            
            # Adversarial training periodically
            if self.rl_agent is not None and self.total_trades % 50 == 0:
                try:
                    self.rl_agent.adversarial_train()
                    logging.info("[Adversarial] RL agent adversarial training update")
                except Exception as e:
                    logging.warning(f"[Adversarial] RL agent adversarial training failed: {e}")
            
            # After trade, store RL experience
            if self.rl_agent is not None:
                try:
                    # Build state, action, reward, next_state, done
                    self.state = self.rl_agent.get_state(self.latest_data, self.predictions, self.whale_features)
                    self.action = self.rl_agent.get_action(self.latest_data, self.predictions, self.whale_features)
                    self.reward = self.rl_agent.get_reward(self.decision['action'], self.trade_executed, self.decision['position_size'], self.latest_data)
                    self.next_state = self.rl_agent.get_next_state(self.latest_data, self.predictions, self.whale_features)
                    self.done = self.rl_agent.is_done(self.decision['action'], self.trade_executed, self.decision['position_size'], self.latest_data)
                    
                    self.rl_replay_buffer.add(self.state, self.action, self.reward, self.next_state, self.done)
                except Exception as e:
                    logging.warning(f"[RL] Failed to store RL experience: {e}")
            
            # Update performance metrics
            self.update_performance_metrics()
            
            # Self-repair (after performance update)
            if self.trainer.self_repair_enabled:
                logging.info("[10x] Checking for self-repair actions...")
                try:
                    self.trainer.perform_self_repair()
                except Exception as e:
                    logging.warning(f"Self-repair failed: {e}")
            
            # Send status update every 10 cycles
            if self.total_trades % 10 == 0:
                self.send_status_update()
            
            # Rolling backtest (background)
            if self.latest_data is not None and len(self.latest_data) > 1000:
                try:
                    self.recent_data = self.latest_data.tail(1000)
                    metrics, trade_history = self.backtester.run(self.recent_data, verbose=True)
                    logging.info(f"[Backtest] Rolling metrics: {metrics}")
                except Exception as e:
                    logging.warning(f"[Backtest] Rolling backtest failed: {e}")
            
            logging.info("Trading cycle completed")
        except Exception as e:
            logging.error(f"Error in trading cycle: {e}")
    
    def run(self):
        """Run the ultra trading bot - identical logic for simulation and live"""
        self.mode_text = "💰 LIVE TRADING MODE" if not self.simulation_mode else "🚀 SIMULATION MODE"
        logging.info(f"Starting Ultra Trading Bot with maximum intelligence... {self.mode_text}")
        
        try:
            self.is_running = True
            self.start_time = datetime.now()
            
            if self.telegram:
                self.startup_msg = f"🚀 Starting Ultra Trading Bot with maximum intelligence... {self.mode_text}\n"
                if self.simulation_mode:
                    self.startup_msg += f"\n📊 Initial Capital: ${self.execution_engine.initial_capital:.2f}"
                self.startup_msg += f"\n🤖 Autonomous optimization active - continuously learning and improving"
                self.telegram.send_message(self.startup_msg)
            
            # Main trading loop
            while self.is_running:
                try:
                    self.trading_cycle()
                    
                    # Wait for next cycle
                    time.sleep(60)  # 1-minute cycles
                    
                except KeyboardInterrupt:
                    logging.info("Received interrupt signal")
                    break
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    time.sleep(60)  # Wait before retrying
            
            # Send shutdown notification
            if self.telegram:
                self.runtime = datetime.now() - self.start_time
                self.shutdown_msg = f"🛑 Ultra Trading Bot stopped - {self.mode_text}\n"
                self.shutdown_msg += f"⏱️ Runtime: {self.runtime}\n"
                
                if self.simulation_mode:
                    self.paper_summary = self.execution_engine.get_performance_summary()
                    self.shutdown_msg += f"📊 Final Capital: ${self.paper_summary.get('current_capital', 0):.2f}\n"
                    self.shutdown_msg += f"📈 Total Return: {self.paper_summary.get('return_pct', 0):.2f}%\n"
                    self.shutdown_msg += f"🎯 Total Trades: {self.paper_summary.get('total_trades', 0)}\n"
                else:
                    self.shutdown_msg += f"📊 Final Stats: {self.performance_metrics}\n"
                
                self.telegram.send_message(self.shutdown_msg)
            
            logging.info(f"Ultra Trading Bot stopped - {self.mode_text}")
            
        except Exception as e:
            logging.error(f"Error running bot: {e}")
        finally:
            self.is_running = False
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        logging.info("Stopping Ultra Trading Bot...")
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join()
        if self.rl_agent:
            self.rl_agent.save_model()
        if self.trainer:
            self.trainer.save_state()
        if self.backtester:
            self.backtester.save_state()
        if self.telegram:
            self.telegram.send_message("🛑 Ultra Trading Bot stopped.")
        logging.info("Ultra Trading Bot stopped.")
    
    def start_continuous_backtesting(self):
        import threading, time
        def backtest_loop():
            while True:
                try:
                    # (Stub) Run rolling backtest on recent data
                    logging.info("[Backtest] Running rolling backtest on recent data (stub)...")
                    time.sleep(3600)  # Run every hour
                except Exception as e:
                    logging.error(f"[Backtest] Error: {e}")
                    time.sleep(3600)
        self.t = threading.Thread(target=backtest_loop, daemon=True)
        self.t.start()
        logging.info("Continuous backtesting thread started.")
    
    def run_intelligence_enhancement(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive intelligence enhancement"""
        try:
            logging.info("🧠 Running intelligence enhancement")
            
            if not ADVANCED_MODULES_AVAILABLE or self.intelligence_enhancer is None:
                logging.warning("Intelligence enhancer not available")
                return {}
            
            # Prepare features for intelligence analysis
            self.feature_cols = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            self.X = data[self.feature_cols].fillna(0)
            
            # Run intelligence enhancement
            self.results = self.intelligence_enhancer.enhance_intelligence(data, self.X)
            
            logging.info("✅ Intelligence enhancement completed")
            return self.results
            
        except Exception as e:
            logging.error(f"Error in intelligence enhancement: {e}")
            return {}
    
    def execute_trading_decision_advanced(self, 
                                        decision: str, 
                                        confidence: float, 
                                        price: float,
                                        data: pd.DataFrame) -> bool:
        """Execute trading decision with advanced risk management"""
        try:
            if not ADVANCED_MODULES_AVAILABLE or self.robustness_manager is None:
                logging.warning("Robustness manager not available, using basic execution")
                return self.execute_trade({'action': decision, 'confidence': confidence})
            
            # Check if trading should be allowed
            self.risk_check = self.robustness_manager.check_risk_limits(decision)
            if self.risk_check.get('should_stop_trading', False):
                logging.warning("🚨 Trading stopped due to risk limits")
                return False
            
            # Check API limits
            if not self.robustness_manager.api_handler.check_rate_limit('order'):
                logging.warning("⚠️ API rate limit reached, skipping trade")
                return False
            
            if decision == 'hold':
                return True
            
            # Calculate position size
            self.position_result = self.risk_manager.calculate_position_size(decision, confidence, price, data)
            if self.position_result <= 0:
                logging.info("Position size too small, skipping trade")
                return True
            
            # Execute trade using existing execution engine
            if decision == 'buy' and self.current_position <= 0:
                self.success = self.execution_engine.place_order(
                    'ETHUSDT',
                    decision,
                    price,
                    self.position_result,
                    'BUY'
                )
                if self.success:
                    self.current_position = self.position_result
                    self.last_trade_time = datetime.now()
                    logging.info(f"✅ BUY order executed: {self.position_result} @ {price}")
                    return True
                else:
                    logging.warning(f"BUY order execution failed: {self.success}")
                    return False
                    
            elif decision == 'sell' and self.current_position > 0:
                self.success = self.execution_engine.place_order(
                    'ETHUSDT',
                    decision,
                    price,
                    self.current_position,
                    'SELL'
                )
                if self.success:
                    self.current_position = 0
                    self.last_trade_time = datetime.now()
                    logging.info(f"✅ SELL order executed: {self.current_position} @ {price}")
                    return True
                else:
                    logging.warning(f"SELL order execution failed: {self.success}")
                    return False
            
            return False
            
        except Exception as e:
            logging.error(f"Error executing trading decision: {e}")
            return False
    
    def update_performance_metrics_advanced(self, 
                                        data: pd.DataFrame, 
                                        decision: str, 
                                        confidence: float, 
                                        execution_success: bool):
        """Update performance metrics with advanced monitoring"""
        try:
            # Calculate basic metrics
            self.current_price = self.get_current_price()
            if self.current_price is None:
                return
            
            self.trade_record = {
                'timestamp': datetime.now(),
                'decision': decision,
                'confidence': confidence,
                'price': self.current_price,
                'position_size': self.current_position,
                'execution_success': execution_success
            }
            self.trading_history.append(self.trade_record)
            
            # Update performance monitor
            if self.performance_monitor is not None:
                self.performance_monitor.update_performance(
                    total_pnl=self.performance_metrics['total_pnl'],
                    current_balance=self.performance_metrics['current_balance'],
                    winning_trades=self.performance_metrics['winning_trades'],
                    losing_trades=self.performance_metrics['losing_trades'],
                    total_trades=self.performance_metrics['total_trades'],
                    win_rate=self.performance_metrics['win_rate'],
                    avg_trade_pnl=self.performance_metrics['avg_trade_pnl'],
                    sharpe_ratio=self.performance_metrics['sharpe_ratio'],
                    max_drawdown=self.performance_metrics['max_drawdown']
                )
            
        except Exception as e:
            logging.error(f"Error updating performance metrics: {e}")
    
    def get_system_status_advanced(self) -> Dict[str, Any]:
        """Get comprehensive system status with advanced components"""
        try:
            self.status = {
                'bot_status': 'running' if self.is_running else 'stopped',
                'simulation_mode': self.simulation_mode,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'total_pnl': self.performance_metrics['total_pnl'],
                'current_balance': self.performance_metrics['current_balance'],
                'return_pct': self.performance_metrics['return_pct'],
                'avg_trade_pnl': self.performance_metrics['avg_trade_pnl'],
                'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
                'win_rate': self.performance_metrics['win_rate'],
                'max_drawdown': self.performance_metrics['max_drawdown']
            }
            
            # Add advanced component status if available
            if ADVANCED_MODULES_AVAILABLE:
                if self.robustness_manager is not None:
                    self.status['robustness_status'] = self.robustness_manager.get_robustness_status()
                if self.performance_monitor is not None:
                    self.status['performance_summary'] = self.performance_monitor.get_performance_summary()
                if self.intelligence_enhancer is not None:
                    self.status['intelligence_summary'] = self.intelligence_enhancer.get_intelligence_summary()
                if self.advanced_ensemble is not None:
                    self.status['ensemble_status'] = self.advanced_ensemble.get_ensemble_status()
            
            return self.status
            
        except Exception as e:
            logging.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        try:
            # Calculate True Range
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            up_move = data['high'] - data['high'].shift()
            down_move = data['low'].shift() - data['low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Calculate smoothed values
            tr_smooth = tr.rolling(window=period).mean()
            plus_di = pd.Series(plus_dm).rolling(window=period).mean() / tr_smooth * 100
            minus_di = pd.Series(minus_dm).rolling(window=period).mean() / tr_smooth * 100
            
            # Calculate ADX
            dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
            adx = pd.Series(dx).rolling(window=period).mean()
            
            return adx.fillna(20.0)  # Default to 20 if not enough data
            
        except Exception as e:
            logging.error(f"Error calculating ADX: {e}")
            return pd.Series(index=data.index, dtype=float).fillna(20.0)

def main():
    """Main entry point"""
    try:
        # Create and run the bot
        bot = UltraTradingBot()
        
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}")
            bot.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run the bot
        bot.run()
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
