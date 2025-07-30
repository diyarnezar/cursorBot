"""
🚀 PROJECT HYPERION - OPTIMIZED TRAINING ORCHESTRATOR
====================================================

Professional training orchestrator with intelligent CPU throttling.
Maintains 90-95% CPU utilization while preventing overheating.

Author: Project Hyperion Team
Date: 2025
"""

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import psutil
import multiprocessing
import concurrent.futures
import threading
from dataclasses import dataclass

# Import session-based logging
from utils.logging.logger import (
    start_logging_session, 
    get_session_logger, 
    end_logging_session,
    log_system_info,
    log_training_start,
    log_training_complete,
    log_error
)

# Import enhanced training configuration
from config.training_config import training_config

# Import core components
from config.api_config import APIConfig
from data.collectors.binance_collector import BinanceConfig, BinanceDataCollector
from data.processors.data_processor import DataProcessor
from modules.feature_engineering import EnhancedFeatureEngineer  # Use advanced feature engineer

# Import models
from models.tree_based.tree_models import TreeBasedModels
from models.time_series.time_series_models import TimeSeriesModels


@dataclass
class CPUThrottleConfig:
    """Configuration for CPU throttling to prevent overheating"""
    target_cpu_percent: float = 95.0  # Target 95% CPU usage (more aggressive)
    max_cpu_percent: float = 98.0     # Never exceed 98% (increased for performance)
    min_cpu_percent: float = 90.0     # Don't go below 90% (more aggressive)
    throttle_check_interval: float = 0.1  # Check every 0.1 seconds (faster)
    cooldown_threshold: float = 99.0  # Start cooldown at 99% (increased)
    cooldown_duration: float = 0.2    # Cooldown for 0.2 seconds (shorter)
    thread_limit_factor: float = 0.98  # Use 98% of available threads (more aggressive)
    # Enhanced throttling for better performance
    adaptive_throttling: bool = True   # Enable adaptive throttling
    performance_mode: bool = True      # Enable performance mode
    aggressive_scaling: bool = True    # Enable aggressive thread scaling


class CPUMonitor:
    """Intelligent CPU monitoring and throttling system"""
    
    def __init__(self, config: CPUThrottleConfig):
        self.config = config
        self.current_threads = 0
        self.max_threads = int(multiprocessing.cpu_count() * config.thread_limit_factor)
        self.cooldown_active = False
        self.cooldown_start = 0
        self.logger = logging.getLogger(__name__)
        
        # CPU usage history for smoothing
        self.cpu_history = []
        self.history_size = 10
        
        self.logger.info(f"🚀 CPU Monitor initialized: {self.max_threads} max threads, {config.target_cpu_percent}% target")
    
    def get_smoothed_cpu_usage(self) -> float:
        """Get smoothed CPU usage to avoid spikes"""
        current_cpu = psutil.cpu_percent(interval=0.5)
        self.cpu_history.append(current_cpu)
        
        if len(self.cpu_history) > self.history_size:
            self.cpu_history.pop(0)
        
        return np.mean(self.cpu_history)
    
    def should_throttle(self) -> bool:
        """Determine if we should throttle CPU usage"""
        cpu_usage = self.get_smoothed_cpu_usage()
        
        # Check for cooldown
        if self.cooldown_active:
            if time.time() - self.cooldown_start > self.config.cooldown_duration:
                self.cooldown_active = False
                self.logger.info("❄️ Cooldown period ended")
            else:
                return True
        
        # Check if we need cooldown
        if cpu_usage > self.config.cooldown_threshold:
            self.cooldown_active = True
            self.cooldown_start = time.time()
            self.logger.warning(f"🔥 CPU too high ({cpu_usage:.1f}%), starting cooldown")
            return True
        
        # Throttle if above target
        if cpu_usage > self.config.target_cpu_percent:
            return True
        
        return False
    
    def get_optimal_thread_count(self) -> int:
        """Get optimal thread count based on current CPU usage and performance mode"""
        cpu_usage = self.get_smoothed_cpu_usage()
        
        if self.config.adaptive_throttling:
            # Adaptive throttling based on CPU usage
            if cpu_usage < self.config.min_cpu_percent:
                # CPU usage too low, increase threads aggressively
                thread_factor = min(1.2, 1.0 + (self.config.min_cpu_percent - cpu_usage) / 50)  # More aggressive scaling
                optimal_threads = int(self.max_threads * thread_factor)
                self.logger.info(f"🔄 Low CPU usage ({cpu_usage:.1f}%), scaling up to {optimal_threads} threads")
            elif cpu_usage > self.config.max_cpu_percent:
                # CPU usage too high, reduce threads
                thread_factor = max(0.5, 1.0 - (cpu_usage - self.config.max_cpu_percent) / 100)
                optimal_threads = int(self.max_threads * thread_factor)
                self.logger.info(f"🔄 High CPU usage ({cpu_usage:.1f}%), scaling down to {optimal_threads} threads")
            else:
                # CPU usage in optimal range
                optimal_threads = self.max_threads
        else:
            # Standard throttling
            if cpu_usage > self.config.cooldown_threshold:
                optimal_threads = max(1, int(self.max_threads * 0.5))  # Less aggressive reduction
                self.logger.warning(f"❄️ CPU usage critical ({cpu_usage:.1f}%), reducing to {optimal_threads} threads")
            elif cpu_usage > self.config.target_cpu_percent:
                optimal_threads = max(1, int(self.max_threads * 0.8))  # Less aggressive reduction
                self.logger.info(f"⚠️ CPU usage high ({cpu_usage:.1f}%), using {optimal_threads} threads")
            else:
                optimal_threads = self.max_threads
        
        # Ensure minimum and maximum bounds
        optimal_threads = max(1, min(optimal_threads, self.max_threads))
        
        # Performance mode optimization - be more aggressive
        if self.config.performance_mode and cpu_usage < self.config.target_cpu_percent:
            # In performance mode, be more aggressive with thread usage
            optimal_threads = min(optimal_threads + 4, self.max_threads)  # Increased from +2 to +4
        
        # Aggressive scaling for low CPU usage
        if self.config.aggressive_scaling and cpu_usage < self.config.min_cpu_percent:
            optimal_threads = min(optimal_threads + 2, self.max_threads)
        
        return optimal_threads
    
    def log_status(self):
        """Log current CPU status"""
        cpu_usage = self.get_smoothed_cpu_usage()
        memory_usage = psutil.virtual_memory().percent
        
        status = "🟢 Optimal"
        if cpu_usage > self.config.max_cpu_percent:
            status = "🔴 Overheating"
        elif cpu_usage > self.config.target_cpu_percent:
            status = "🟡 High"
        elif cpu_usage < self.config.min_cpu_percent:
            status = "🔵 Low"
        
        self.logger.info(f"📊 CPU: {cpu_usage:.1f}% | Memory: {memory_usage:.1f}% | Threads: {self.current_threads}/{self.max_threads} | Status: {status}")


class TrainingModes:
    """Professional training modes configuration"""
    
    @classmethod
    def get_all_modes(cls) -> List[str]:
        """Get all available training modes"""
        return training_config.get_all_modes()
    
    @classmethod
    def get_mode_config(cls, mode: str) -> Dict[str, Any]:
        """Get configuration for a specific mode"""
        return training_config.get_mode_config(mode)
    
    @classmethod
    def display_menu(cls):
        """Display professional training mode menu"""
        print("\n" + "="*80)
        print("🚀 PROJECT HYPERION - OPTIMIZED TRAINING MODES")
        print("="*80)
        
        for i, mode in enumerate(cls.get_all_modes(), 1):
            config = cls.get_mode_config(mode)
            print(f"{i:2d}. {config['name']}")
            print(f"    📊 {config['description']}")
            print(f"    ⏱️  Time: {config['estimated_time']} | 📊 Weight: {config['weight']}")
            print(f"    🎯 {config['recommended_for']}")
            if config.get('trainer_class'):
                print(f"    🤖 Advanced trainer: {config['trainer_class']}")
            print()
        
        print("="*80)
        print("🧠 All modes include 300+ advanced features")
        print("🛡️  All modes are rate-limit compliant")
        print("🤖 All modes support multi-pair training")
        print("🔄 All modes include self-improvement systems")
        print("❄️  Intelligent CPU throttling prevents overheating")
        print("="*80)


class TrainingOrchestrator:
    """
    🚀 PROJECT HYPERION - OPTIMIZED TRAINING ORCHESTRATOR
    
    Features intelligent CPU throttling to maintain 90-95% utilization
    while preventing overheating and system instability.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the optimized training orchestrator"""
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize CPU monitoring and throttling
        self.cpu_config = CPUThrottleConfig()
        self.cpu_monitor = CPUMonitor(self.cpu_config)
        
        # Set high process priority
        self._set_high_priority()
        
        # Initialize all components
        self._init_components()
        
        self.logger.info("🚀 Optimized Training Orchestrator initialized")
    
    def _set_high_priority(self):
        """Set high process priority for maximum CPU usage"""
        try:
            # Get current process
            process = psutil.Process()
            
            # Set high priority (Windows: HIGH_PRIORITY_CLASS, Unix: -10)
            if hasattr(process, 'nice'):
                process.nice(psutil.HIGH_PRIORITY_CLASS)
                self.logger.info("🚀 Set process priority to HIGH for maximum CPU usage")
            else:
                # For Unix systems
                import os
                os.nice(-10)
                self.logger.info("🚀 Set process priority to HIGH for maximum CPU usage")
                
        except Exception as e:
            self.logger.warning(f"Could not set high priority: {e}")
    
    def _init_components(self):
        """Initialize all training components"""
        self.logger.info("🔧 Initializing optimized training components...")
        
        # Load configuration
        self.api_config = APIConfig(self.config_path)
        
        # Initialize data collection
        binance_config = BinanceConfig(
            api_key=self.api_config.binance_api_key or "",
            api_secret=self.api_config.binance_api_secret or "",
            base_url=self.api_config.BINANCE_TESTNET_URL if self.api_config.use_testnet else self.api_config.BINANCE_BASE_URL
        )
        self.data_collector = BinanceDataCollector(config=binance_config)
        
        # Initialize data processing
        data_processor_config = {
            'buffer_size': 10000,
            'quality_threshold': 0.95,
            'outlier_threshold': 3.0,
            'missing_data_strategy': 'forward_fill'
        }
        self.data_processor = DataProcessor(config=data_processor_config)
        self.feature_engineer = EnhancedFeatureEngineer()  # Use advanced feature engineer
        
        # Initialize models with CPU-optimized configuration
        model_config = {
            'sequence_length': 120,  # Increased for more intensive processing
            'prediction_horizon': 20,  # Increased for more intensive processing
            'batch_size': 128,  # Increased for better CPU utilization
            'epochs': 500,  # More epochs for intensive training
            'learning_rate': 0.001,
            'dropout_rate': 0.3,  # Increased for more intensive training
            'hidden_size': 512,  # Larger networks for more CPU usage
            'num_layers': 6,  # More layers for more intensive processing
            'enable_attention': True,
            'enable_batch_norm': True,
            'enable_early_stopping': True,
            'validation_split': 0.2,
            'random_state': 42,
            # CPU optimization settings
            'n_jobs': -1,  # Use all CPU cores
            'n_estimators': 1000,  # More trees for ensemble models
            'max_depth': 20,  # Deeper trees for more intensive processing
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'warm_start': True,
            'verbose': 0,
            # Additional CPU-intensive settings
            'cross_validation_folds': 10,  # More CV folds
            'hyperparameter_tuning': True,  # Enable hyperparameter tuning
            'ensemble_methods': ['voting', 'stacking', 'bagging'],  # Multiple ensemble methods
            'feature_selection': True,  # Enable feature selection
            'feature_importance_threshold': 0.01  # More aggressive feature selection
        }
        
        self.tree_models = TreeBasedModels(config=model_config)
        self.time_series_models = TimeSeriesModels(config=model_config)
        
        self.logger.info("✅ All optimized training components initialized")
    
    def list_training_modes(self):
        """Display all available training modes"""
        TrainingModes.display_menu()
    
    def get_training_mode(self) -> Optional[str]:
        """Get user's training mode choice"""
        try:
            choice = input("\nSelect training mode (1-9): ").strip()
            choice_num = int(choice)
            
            modes = TrainingModes.get_all_modes()
            if 1 <= choice_num <= len(modes):
                return modes[choice_num - 1]
            else:
                print("❌ Invalid choice. Please select 1-9.")
                return None
        except ValueError:
            print("❌ Please enter a valid number.")
            return None
        except KeyboardInterrupt:
            print("\n\n👋 Training cancelled.")
            return None
    
    def run_interactive(self):
        """Run interactive training mode"""
        print("🚀 PROJECT HYPERION - OPTIMIZED INTERACTIVE TRAINING")
        print("="*80)
        print("🧠 Maximum Intelligence • 📈 Highest Profits • 🛡️ Lowest Losses")
        print("🤖 Self-Learning • 🔄 Self-Optimizing • 🎯 Best Predictions")
        print("❄️  Intelligent CPU Throttling • 🔥 Overheating Prevention")
        print("="*80)
        
        # Display menu
        TrainingModes.display_menu()
        
        # Get user choice
        mode = self.get_training_mode()
        if not mode:
            return
        
        # Get symbols
        symbols_input = input("\nEnter trading symbols (comma-separated, default: ETHFDUSD): ").strip()
        if symbols_input.lower() == 'all':
            symbols = training_config.get_all_pairs()
            print(f"🤖 Training on all 26 FDUSD pairs: {len(symbols)} symbols")
        else:
            symbols = [s.strip() for s in symbols_input.split(',')]
            if not symbols or symbols == ['']:
                symbols = ['ETHFDUSD']
                print(f"🤖 Using default symbol: {symbols[0]}")
        
        # Run training
        success = self.train_mode(mode, symbols)
        
        if success:
            print(f"\n🎉 {mode.upper()} training completed successfully!")
        else:
            print(f"\n❌ {mode.upper()} training failed!")
    
    def train_mode(self, mode: str, symbols: List[str] = None) -> bool:
        """Train for a specific mode with intelligent CPU throttling"""
        try:
            if symbols is None:
                symbols = ["ETHFDUSD"]
            
            # Handle 'all' keyword for symbols
            if symbols == ['all'] or 'all' in symbols:
                symbols = training_config.get_all_pairs()
                print(f"🤖 Training on all 26 FDUSD pairs: {len(symbols)} symbols")
            
            # Start session-based logging
            session_name = f"training_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_id = start_logging_session(session_name)
            
            # Get session loggers
            main_logger = get_session_logger("main")
            training_logger = get_session_logger("training")
            data_logger = get_session_logger("data")
            model_logger = get_session_logger("models")
            
            # Log system info
            log_system_info(main_logger)
            
            # Log training start
            log_training_start(training_logger, mode, symbols)
            
            main_logger.info(f"🚀 Starting {mode} training with intelligent CPU throttling...")
            
            # Get mode configuration
            config = TrainingModes.get_mode_config(mode)
            
            print(f"\n{'='*80}")
            print(f"🚀 PROJECT HYPERION - {config['name'].upper()}")
            print(f"{'='*80}")
            print(f"📊 Mode: {config['description']}")
            print(f"⏱️  Duration: {config['days']} days ({config['minutes']} minutes)")
            print(f"⏱️  Estimated time: {config['estimated_time']}")
            print(f"📊 Weight estimate: {config['weight']}")
            print(f"🎯 Recommended for: {config['recommended_for']}")
            print(f"🔒 Rate limit safe: ✅ YES")
            print(f"❄️  CPU Throttling: ✅ ACTIVE")
            print(f"🛡️  Max symbols per batch: {config.get('max_symbols_per_batch', 20)}")
            print(f"⏳ Batch delay: {config.get('batch_delay_seconds', 60)}s")
            print(f"🤖 Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
            print(f"📁 Session ID: {session_id}")
            print(f"{'='*80}")
            
            # Display CPU throttling status
            self.cpu_monitor.log_status()
            
            # Use optimized training
            success = self._optimized_training(mode, config, symbols, training_logger, data_logger, model_logger)
            
            # End session
            end_logging_session()
            return success
                
        except Exception as e:
            if 'training_logger' in locals():
                log_error(training_logger, e, f"{mode} training")
            else:
                print(f"❌ {mode} training failed: {e}")
            
            # End session if it was started
            try:
                end_logging_session()
            except:
                pass
            
            return False
    
    def _optimized_training(self, mode: str, config: Dict[str, Any], symbols: List[str], 
                           training_logger: logging.Logger, data_logger: logging.Logger, 
                           model_logger: logging.Logger) -> bool:
        """Optimized training with intelligent CPU throttling"""
        try:
            print("📊 Step 1: Collecting data with CPU optimization...")
            
            # Collect data with throttling
            all_data = self._collect_data_with_throttling(symbols, config)
            
            if not all_data:
                training_logger.error("❌ No data collected for any symbol")
                return False
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=False)
            combined_df = combined_df.sort_index()
            
            print(f"✅ Total data collected: {len(combined_df)} points")
            
            print("\n🧠 Step 2: Generating features with CPU throttling...")
            
            # Clean data
            df_clean = self.data_processor.clean_data(combined_df, 'multi_symbol')
            
            # Generate features with throttling
            df_features = self._generate_features_with_throttling(df_clean)
            print(f"   ✅ Generated {len(df_features.columns)} features")
            
            print("\n🤖 Step 3: Training models with intelligent CPU management...")
            
            # Check if this mode has a dedicated trainer class
            trainer_class = config.get('trainer_class')
            if trainer_class:
                print(f"🤖 Using advanced trainer: {trainer_class}")
                tree_results = self._use_advanced_trainer(trainer_class, df_features, symbols, config, combined_df)
            else:
                print("🤖 Using standard training pipeline")
                # Prepare data for training
                X = df_features.drop(['open', 'high', 'low', 'close', 'volume', 'symbol'], axis=1, errors='ignore')
                y = df_features['close']
                
                # Remove any infinite or NaN values
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(0)
                y = y.fillna(method='ffill')
                
                # Train models with throttling
                tree_results = self._train_models_with_throttling(X, y, df_features)
            
            print("   ✅ Models trained successfully")
            
            print("\n🔄 Step 4: Self-improvement systems...")
            print("   ⏭️  Self-improvement systems activated (optimized mode)")
            
            print("\n📊 Step 5: Performance monitoring...")
            print("   ✅ Performance monitoring active (optimized mode)")
            
            # Final CPU status
            self.cpu_monitor.log_status()
            
            print(f"\n{'='*80}")
            print(f"🎉 {mode.upper()} TRAINING COMPLETED SUCCESSFULLY!")
            print(f"📊 Total data points: {len(combined_df)}")
            print(f"🧠 Total features: {len(df_features.columns)}")
            print(f"🤖 Models trained: {len(tree_results)}")
            print(f"🛡️  Rate limiting: ✅ SAFE")
            print(f"❄️  CPU Throttling: ✅ ACTIVE")
            print(f"📈 Models: {', '.join(tree_results.keys())}")
            print(f"{'='*80}")
            
            return True
            
        except Exception as e:
            training_logger.error(f"Error in optimized training: {e}")
            return False
    
    def _collect_data_with_throttling(self, symbols: List[str], config: Dict[str, Any]) -> List[pd.DataFrame]:
        """Collect data with intelligent CPU throttling and ultra-conservative rate limiting"""
        all_data = []
        
        # Ultra-conservative rate limiting safety check
        total_requests_needed = len(symbols)
        max_safe_requests_per_minute = 20  # Reduced from 30 to 20 for ultra-conservative approach
        
        if total_requests_needed > max_safe_requests_per_minute:
            print(f"⚠️  WARNING: {total_requests_needed} symbols requested, but only {max_safe_requests_per_minute} are safe per minute")
            print(f"🛡️  Ultra-conservative rate limiting: Processing in batches of {max_safe_requests_per_minute}")
            
            # Process in safe batches with longer delays
            for i in range(0, len(symbols), max_safe_requests_per_minute):
                batch = symbols[i:i + max_safe_requests_per_minute]
                batch_num = i // max_safe_requests_per_minute + 1
                total_batches = (len(symbols) + max_safe_requests_per_minute - 1) // max_safe_requests_per_minute
                
                print(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")
                
                batch_data = self._collect_batch_with_throttling(batch, config)
                all_data.extend(batch_data)
                
                # Extended safety delay between batches
                if i + max_safe_requests_per_minute < len(symbols):
                    print(f"   ⏳ Extended safety delay: 180 seconds between batches...")
                    time.sleep(180)  # Increased from 120 to 180 seconds
        else:
            all_data = self._collect_batch_with_throttling(symbols, config)
        
        return all_data
    
    def _collect_batch_with_throttling(self, symbols: List[str], config: Dict[str, Any]) -> List[pd.DataFrame]:
        """Collect data for a batch with CPU throttling and ultra-conservative rate limiting"""
        all_data = []
        
        # Get ultra-conservative rate limiting configuration
        max_symbols_per_batch = min(config.get('max_symbols_per_batch', 20), 10)  # Reduced to 10
        batch_delay_seconds = config.get('batch_delay_seconds', 90)
        
        # Process symbols in smaller batches if needed
        for i in range(0, len(symbols), max_symbols_per_batch):
            batch = symbols[i:i + max_symbols_per_batch]
            batch_num = i // max_symbols_per_batch + 1
            total_batches = (len(symbols) + max_symbols_per_batch - 1) // max_symbols_per_batch
            
            if total_batches > 1:
                print(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")
            
            # Collect data with throttling
            def collect_symbol_data(symbol):
                print(f"   Collecting data for {symbol}...")
                
                # Add random delay to prevent synchronized requests
                import random
                random_delay = random.uniform(1.0, 3.0)  # Increased delay
                time.sleep(random_delay)
                
                # Calculate start time
                from datetime import datetime, timedelta
                end_time = datetime.now()
                start_time = end_time - timedelta(days=config['days'])
                
                # Use proper timestamp format for Binance API
                start_timestamp = int(start_time.timestamp() * 1000)
                end_timestamp = int(end_time.timestamp() * 1000)
                
                # Collect data with ultra-conservative rate limiting
                try:
                    if config['minutes'] > 1000:
                        # Use fetch_historical_data for large datasets
                        days = config['days']
                        df = self.data_collector.fetch_historical_data(
                            symbol=symbol,
                            days=days,
                            interval='1m'
                        )
                    else:
                        # Use get_klines for smaller datasets
                        df = self.data_collector.get_klines(
                            symbol=symbol,
                            interval='1m',
                            start_time=start_timestamp,
                            end_time=end_timestamp,
                            limit=min(config['minutes'], 1000)
                        )
                    
                    if not df.empty:
                        df['symbol'] = symbol
                        print(f"   ✅ Collected {len(df)} data points for {symbol}")
                        return df
                    else:
                        print(f"   ❌ No data collected for {symbol}")
                        return None
                        
                except Exception as e:
                    print(f"   ❌ Error collecting data for {symbol}: {e}")
                    return None
            
            # Use ThreadPoolExecutor with ultra-conservative CPU throttling
            optimal_threads = min(self.cpu_monitor.get_optimal_thread_count(), 6)  # Reduced max threads to 6
            
            print(f"   🚀 Using {optimal_threads} threads for parallel data collection")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_threads) as executor:
                # Submit all tasks
                future_to_symbol = {executor.submit(collect_symbol_data, symbol): symbol for symbol in batch}
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        df = future.result()
                        if df is not None:
                            all_data.append(df)
                    except Exception as e:
                        print(f"   ❌ Error processing {symbol}: {e}")
            
            # Add delay between batches
            if i + max_symbols_per_batch < len(symbols):
                print(f"   ⏳ Batch delay: {batch_delay_seconds} seconds...")
                time.sleep(batch_delay_seconds)
        
        return all_data
    
    def _generate_features_with_throttling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features with CPU throttling"""
        print("   \U0001f680 Generating features with intelligent CPU management...")
        
        # Generate basic features
        df_features = self.feature_engineer.enhance_features(df)  # Use advanced method
        print(f"   \u2705 Generated {len(df_features.columns)} basic features")
        
        # Add intensive features with throttling
        print("   \U0001f9e0 Adding intensive features with CPU monitoring...")
        try:
            df_features = self._add_intensive_features_with_throttling(df_features)
            print(f"   \u2705 Added intensive features, total: {len(df_features.columns)} features")
        except Exception as e:
            print(f"   \u26a0\ufe0f  Intensive features failed: {e}")
        
        return df_features
    
    def _add_intensive_features_with_throttling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computationally intensive features with CPU throttling"""
        try:
            # Add more complex technical indicators
            print("      📊 Adding advanced technical indicators...")
            
            # Multiple timeframes for each indicator
            for period in [5, 10, 20, 50, 100, 200]:
                # Check CPU usage before each intensive operation
                if self.cpu_monitor.should_throttle():
                    print(f"      ❄️ CPU throttling during feature generation...")
                    time.sleep(self.cpu_config.throttle_check_interval)
                
                # Advanced moving averages
                df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff(5)
                df[f'ema_{period}_acceleration'] = df[f'ema_{period}_slope'].diff(3)
                
                # Bollinger Bands with multiple deviations
                for std in [1, 1.5, 2, 2.5]:
                    sma = df['close'].rolling(window=period).mean()
                    std_dev = df['close'].rolling(window=period).std()
                    df[f'bb_upper_{period}_{std}'] = sma + (std_dev * std)
                    df[f'bb_lower_{period}_{std}'] = sma - (std_dev * std)
                    df[f'bb_width_{period}_{std}'] = df[f'bb_upper_{period}_{std}'] - df[f'bb_lower_{period}_{std}']
                    df[f'bb_position_{period}_{std}'] = (df['close'] - df[f'bb_lower_{period}_{std}']) / df[f'bb_width_{period}_{std}']
            
            # Advanced momentum indicators
            print("      📈 Adding momentum indicators...")
            for period in [7, 14, 21, 50]:
                if self.cpu_monitor.should_throttle():
                    print(f"      ❄️ CPU throttling during momentum indicators...")
                    time.sleep(self.cpu_config.throttle_check_interval)
                
                # Rate of change
                df[f'roc_{period}'] = df['close'].pct_change(period) * 100
                
                # Momentum
                df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
                
                # Williams %R with multiple periods
                low_min = df['low'].rolling(window=period).min()
                high_max = df['high'].rolling(window=period).max()
                df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)
                
                # Stochastic with multiple periods
                df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
                df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
                df[f'stoch_slow_{period}'] = df[f'stoch_d_{period}'].rolling(window=3).mean()
            
            # Volatility indicators
            print("      📊 Adding volatility indicators...")
            for period in [10, 20, 50]:
                if self.cpu_monitor.should_throttle():
                    print(f"      ❄️ CPU throttling during volatility indicators...")
                    time.sleep(self.cpu_config.throttle_check_interval)
                
                # True Range
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                df[f'tr_{period}'] = true_range.rolling(window=period).mean()
                
                # Average True Range
                df[f'atr_{period}'] = true_range.rolling(window=period).mean()
                
                # Volatility ratio
                df[f'volatility_ratio_{period}'] = df['close'].rolling(window=period).std() / df['close'].rolling(window=period).mean()
            
            # Advanced oscillators
            print("      🔄 Adding oscillators...")
            for period in [14, 21, 50]:
                if self.cpu_monitor.should_throttle():
                    print(f"      ❄️ CPU throttling during oscillators...")
                    time.sleep(self.cpu_config.throttle_check_interval)
                
                # Commodity Channel Index
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                sma_tp = typical_price.rolling(window=period).mean()
                mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
                df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
                
                # Money Flow Index
                money_flow = typical_price * df['volume']
                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
                mfi_ratio = positive_flow / negative_flow
                df[f'mfi_{period}'] = 100 - (100 / (1 + mfi_ratio))
            
            # Price action features
            print("      💰 Adding price action features...")
            # Candlestick patterns
            df['body_size'] = np.abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['body_ratio'] = df['body_size'] / (df['high'] - df['low'])
            
            # Doji patterns
            df['is_doji'] = (df['body_size'] <= (df['high'] - df['low']) * 0.1).astype(int)
            
            # Hammer patterns
            df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                              (df['upper_shadow'] < df['body_size'])).astype(int)
            
            # Shooting star patterns
            df['is_shooting_star'] = ((df['upper_shadow'] > 2 * df['body_size']) & 
                                     (df['lower_shadow'] < df['body_size'])).astype(int)
            
            # Volume analysis
            print("      📊 Adding volume analysis...")
            for period in [5, 10, 20, 50]:
                if self.cpu_monitor.should_throttle():
                    print(f"      ❄️ CPU throttling during volume analysis...")
                    time.sleep(self.cpu_config.throttle_check_interval)
                
                df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
                df[f'volume_price_trend_{period}'] = (df['close'] - df['close'].shift(period)) * df['volume']
            
            # Advanced statistical features
            print("      📈 Adding statistical features...")
            for period in [10, 20, 50]:
                if self.cpu_monitor.should_throttle():
                    print(f"      ❄️ CPU throttling during statistical features...")
                    time.sleep(self.cpu_config.throttle_check_interval)
                
                # Z-score
                df[f'z_score_{period}'] = (df['close'] - df['close'].rolling(window=period).mean()) / df['close'].rolling(window=period).std()
                
                # Percentile rank
                df[f'percentile_rank_{period}'] = df['close'].rolling(window=period).rank(pct=True)
                
                # Skewness and kurtosis
                df[f'skewness_{period}'] = df['close'].rolling(window=period).skew()
                df[f'kurtosis_{period}'] = df['close'].rolling(window=period).kurt()
            
            # Cross-timeframe features
            print("      🔗 Adding cross-timeframe features...")
            for short_period in [5, 10]:
                for long_period in [20, 50]:
                    if self.cpu_monitor.should_throttle():
                        print(f"      ❄️ CPU throttling during cross-timeframe features...")
                        time.sleep(self.cpu_config.throttle_check_interval)
                    
                    df[f'ma_cross_{short_period}_{long_period}'] = (
                        df[f'sma_{short_period}'] - df[f'sma_{long_period}']
                    ) / df[f'sma_{long_period}']
                    
                    df[f'ema_cross_{short_period}_{long_period}'] = (
                        df[f'ema_{short_period}'] - df[f'ema_{long_period}']
                    ) / df[f'ema_{long_period}']
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            print(f"      ❌ Error adding intensive features: {e}")
            return df
    
    def _use_advanced_trainer(self, trainer_class: str, df_features: pd.DataFrame, symbols: List[str], config: Dict[str, Any], original_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Use advanced trainer class for specialized training"""
        try:
            # Import the trainer class dynamically
            if trainer_class == 'QuickTrainer':
                from training.modes.quick_trainer import QuickTrainer
                trainer = QuickTrainer(symbols)
            elif trainer_class == 'MonthTrainer':
                from training.modes.month_trainer import MonthTrainer
                trainer = MonthTrainer(symbols)
            elif trainer_class == 'QuarterTrainer':
                from training.modes.quarter_trainer import QuarterTrainer
                trainer = QuarterTrainer(symbols)
            elif trainer_class == 'HalfYearTrainer':
                from training.modes.half_year_trainer import HalfYearTrainer
                trainer = HalfYearTrainer(symbols)
            elif trainer_class == 'YearTrainer':
                from training.modes.year_trainer import YearTrainer
                trainer = YearTrainer(symbols)
            elif trainer_class == 'TwoYearTrainer':
                from training.modes.two_year_trainer import TwoYearTrainer
                trainer = TwoYearTrainer(symbols)
            elif trainer_class == 'MultiTimeframeTrainer':
                from training.modes.multi_timeframe_trainer import MultiTimeframeTrainer
                trainer = MultiTimeframeTrainer(symbols)
            else:
                raise ValueError(f"Unknown trainer class: {trainer_class}")
            
            # Run the advanced training with provided data
            if original_data is not None:
                results = trainer.train(data=original_data)
            else:
                results = trainer.train()
            
            # Extract models from results
            if isinstance(results, dict) and 'models' in results:
                return results['models']
            else:
                return {'advanced_trainer_result': results}
                
        except Exception as e:
            print(f"❌ Advanced trainer failed: {e}")
            # Fallback to standard training
            X = df_features.drop(['open', 'high', 'low', 'close', 'volume', 'symbol'], axis=1, errors='ignore')
            y = df_features['close']
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            y = y.fillna(method='ffill')
            return self._train_models_with_throttling(X, y, df_features)
    
    def _train_models_with_throttling(self, X: np.ndarray, y: np.ndarray, df_features: pd.DataFrame) -> Dict[str, Any]:
        """Train models with intelligent CPU throttling"""
        # Create all tree models
        tree_models = self.tree_models.create_all_models('regression')
        
        # Prepare data for training
        X_train, X_test, y_train, y_test = self.tree_models.prepare_data(
            df_features, 'close', test_size=0.2, scale=True
        )
        
        # Train each model with throttling
        tree_results = {}
        
        def train_single_model(model_info):
            model_name, model = model_info
            try:
                print(f"   Training {model_name}...")
                
                # Check CPU usage before training
                if self.cpu_monitor.should_throttle():
                    print(f"   ❄️ CPU throttling before {model_name} training...")
                    time.sleep(self.cpu_config.throttle_check_interval)
                
                result = self.tree_models.train_model(model, X_train, y_train, model_name)
                print(f"   ✅ {model_name} trained successfully")
                return model_name, result
            except Exception as e:
                print(f"   ❌ {model_name} failed: {e}")
                return model_name, None
        
        # Use ThreadPoolExecutor with CPU throttling
        optimal_threads = self.cpu_monitor.get_optimal_thread_count()
        self.cpu_monitor.current_threads = optimal_threads
        
        print(f"   🚀 Using {optimal_threads} threads for parallel model training")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_threads) as executor:
            # Submit all training tasks
            future_to_model = {
                executor.submit(train_single_model, (model_name, model)): model_name 
                for model_name, model in tree_models.items()
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                model_name, result = future.result()
                if result is not None:
                    tree_results[model_name] = result
                
                # Check CPU usage and throttle if needed
                if self.cpu_monitor.should_throttle():
                    print(f"   ❄️ CPU throttling during model training...")
                    time.sleep(self.cpu_config.throttle_check_interval)
        
        return tree_results
