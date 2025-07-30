#!/usr/bin/env python3
"""
Simple test for the training system
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import only the basic components
from config.api_config import APIConfig
from data.collectors.binance_collector import BinanceConfig, BinanceDataCollector
from data.processors.data_processor import DataProcessor
from modules.feature_engineering import EnhancedFeatureEngineer

def test_basic_training():
    """Test basic training functionality"""
    print("🧪 Testing basic training functionality...")
    
    try:
        # Initialize components
        api_config = APIConfig("config.json")
        
        binance_config = BinanceConfig(
            api_key=api_config.binance_api_key or "",
            api_secret=api_config.binance_api_secret or "",
            base_url=api_config.BINANCE_TESTNET_URL if api_config.use_testnet else api_config.BINANCE_BASE_URL
        )
        data_collector = BinanceDataCollector(config=binance_config)
        
        data_processor_config = {
            'buffer_size': 1000,
            'quality_threshold': 0.95,
            'outlier_threshold': 3.0,
            'missing_data_strategy': 'forward_fill'
        }
        data_processor = DataProcessor(config=data_processor_config)
        feature_engineer = EnhancedFeatureEngineer()
        
        # Test data collection
        print("📊 Testing data collection...")
        df = data_collector.get_klines("ETHFDUSD", "1m", limit=100)
        print(f"✅ Collected {len(df)} data points")
        
        # Test feature engineering
        print("🧠 Testing feature engineering...")
        df_clean = data_processor.clean_data(df, "ETHFDUSD")
        df_features = feature_engineer.enhance_features(df_clean)
        print(f"✅ Generated {len(df_features.columns)} features")
        
        print("🎉 Basic training test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_basic_training() 