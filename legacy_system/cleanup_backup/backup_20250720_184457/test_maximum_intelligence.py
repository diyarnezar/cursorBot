"""
Test Maximum Intelligence Enhancements
=====================================

This script tests all three parts of the maximum intelligence enhancements:
1. Advanced Feature Engineering & Selection
2. Advanced Model Training & Optimization  
3. Advanced Risk Management & Psychology

Focus: Verify that all components work together for maximum trading performance
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# Add modules to path
sys.path.append('modules')

# Import maximum intelligence modules
from maximum_intelligence_features import MaximumIntelligenceFeatureEngineer
from maximum_intelligence_models import MaximumIntelligenceModelTrainer
from maximum_intelligence_risk import MaximumIntelligenceRiskManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing"""
    logger.info("📊 Creating sample market data...")
    
    np.random.seed(42)
    
    # Generate price data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Generate realistic price movements
    returns = np.random.normal(0.0001, 0.02, n_samples)  # Small positive drift, 2% volatility
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + abs(np.random.normal(0, 0.002, n_samples))),
        'low': prices * (1 - abs(np.random.normal(0, 0.002, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Ensure OHLC relationships are correct
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    logger.info(f"📊 Created {len(data)} samples of market data")
    return data

def test_feature_engineering():
    """Test Part 1: Advanced Feature Engineering & Selection"""
    logger.info("\n" + "="*60)
    logger.info("🧠 TESTING PART 1: ADVANCED FEATURE ENGINEERING")
    logger.info("="*60)
    
    try:
        # Create sample data
        data = create_sample_data(500)
        
        # Initialize feature engineer
        feature_engineer = MaximumIntelligenceFeatureEngineer()
        
        # Create intelligent features
        enhanced_data = feature_engineer.create_intelligent_features(data)
        
        # Create target variable (next period return)
        target = enhanced_data['close'].pct_change().shift(-1)
        
        # Drop NaN values from both features and target
        valid_indices = ~(enhanced_data.isnull().any(axis=1) | target.isnull())
        enhanced_data = enhanced_data[valid_indices]
        target = target[valid_indices]
        
        # Select best features
        selected_data = feature_engineer.select_best_features(
            enhanced_data, target, max_features=50
        )
        
        # Get feature importance report
        importance_report = feature_engineer.get_feature_importance_report()
        
        logger.info("✅ Feature Engineering Test PASSED")
        logger.info(f"   • Original features: {len(data.columns)}")
        logger.info(f"   • Enhanced features: {len(enhanced_data.columns)}")
        logger.info(f"   • Selected features: {len(selected_data.columns)}")
        logger.info(f"   • Top 5 features: {list(importance_report['feature_scores'].keys())[:5]}")
        
        return selected_data, target
        
    except Exception as e:
        logger.error(f"❌ Feature Engineering Test FAILED: {str(e)}")
        raise

def test_model_training(X: pd.DataFrame, y: pd.Series):
    """Test Part 2: Advanced Model Training & Optimization"""
    logger.info("\n" + "="*60)
    logger.info("🧠 TESTING PART 2: ADVANCED MODEL TRAINING")
    logger.info("="*60)
    
    try:
        # Initialize model trainer
        model_trainer = MaximumIntelligenceModelTrainer()
        
        # Train maximum intelligence models
        training_results = model_trainer.train_maximum_intelligence_models(X, y)
        
        # Get model summary
        model_summary = model_trainer.get_model_summary()
        
        # Test ensemble prediction
        ensemble_pred = model_trainer.predict_ensemble(X.iloc[:10])  # Test on first 10 samples
        
        logger.info("✅ Model Training Test PASSED")
        logger.info(f"   • Models trained: {len(training_results['models'])}")
        logger.info(f"   • Best model: {model_summary['best_model']}")
        logger.info(f"   • Best score: {model_summary['average_score']:.3f}")
        logger.info(f"   • Ensemble weights: {list(training_results['ensemble_weights'].keys())}")
        logger.info(f"   • Ensemble prediction shape: {ensemble_pred.shape}")
        
        return model_trainer, training_results
        
    except Exception as e:
        logger.error(f"❌ Model Training Test FAILED: {str(e)}")
        raise

def test_risk_management():
    """Test Part 3: Advanced Risk Management & Psychology"""
    logger.info("\n" + "="*60)
    logger.info("🧠 TESTING PART 3: ADVANCED RISK MANAGEMENT")
    logger.info("="*60)
    
    try:
        # Initialize risk manager
        risk_manager = MaximumIntelligenceRiskManager(initial_capital=10000.0)
        
        # Test position sizing
        position_size = risk_manager.calculate_optimal_position_size(
            signal_strength=0.8,
            volatility=0.02,
            confidence=0.7,
            current_drawdown=0.05
        )
        
        # Test stop loss calculation
        stop_loss = risk_manager.calculate_dynamic_stop_loss(
            entry_price=100.0,
            signal_direction=1,  # Long position
            volatility=0.02,
            atr=2.0
        )
        
        # Test take profit calculation
        take_profit = risk_manager.calculate_dynamic_take_profit(
            entry_price=100.0,
            signal_strength=0.8,
            volatility=0.02,
            atr=2.0
        )
        
        # Test psychology state update
        recent_trades = [
            {'return': 0.02}, {'return': -0.01}, {'return': 0.03},
            {'return': 0.01}, {'return': -0.005}, {'return': 0.015}
        ]
        psychology_state = risk_manager.update_psychology_state(
            recent_trades, {'volatility': 0.02}
        )
        
        # Test market regime update
        market_data = create_sample_data(100)
        regime = risk_manager.update_market_regime(market_data)
        
        # Test trading decision
        should_trade = risk_manager.should_trade(
            signal_strength=0.8,
            market_conditions={'volatility': 0.02},
            current_drawdown=0.05
        )
        
        # Get risk summary
        risk_summary = risk_manager.get_risk_summary()
        
        logger.info("✅ Risk Management Test PASSED")
        logger.info(f"   • Position size: {position_size:.3f}")
        logger.info(f"   • Stop loss: {stop_loss:.2f}")
        logger.info(f"   • Take profit: {take_profit:.2f}")
        logger.info(f"   • Psychology state: {psychology_state['state']}")
        logger.info(f"   • Market regime: {regime}")
        logger.info(f"   • Should trade: {should_trade}")
        
        return risk_manager
        
    except Exception as e:
        logger.error(f"❌ Risk Management Test FAILED: {str(e)}")
        raise

def test_integration():
    """Test integration of all three parts"""
    logger.info("\n" + "="*60)
    logger.info("🧠 TESTING INTEGRATION OF ALL PARTS")
    logger.info("="*60)
    
    try:
        # Test all parts in sequence
        X, y = test_feature_engineering()
        model_trainer, training_results = test_model_training(X, y)
        risk_manager = test_risk_management()
        
        # Test end-to-end workflow
        logger.info("\n🔄 Testing end-to-end workflow...")
        
        # 1. Get prediction from ensemble
        ensemble_pred = model_trainer.predict_ensemble(X.iloc[:1])
        signal_strength = abs(ensemble_pred[0])
        signal_direction = 1 if ensemble_pred[0] > 0 else -1
        
        # 2. Calculate position size
        position_size = risk_manager.calculate_optimal_position_size(
            signal_strength=signal_strength,
            volatility=0.02,
            confidence=0.7
        )
        
        # 3. Calculate stop loss and take profit
        entry_price = 100.0
        stop_loss = risk_manager.calculate_dynamic_stop_loss(
            entry_price=entry_price,
            signal_direction=signal_direction,
            volatility=0.02,
            atr=2.0
        )
        
        take_profit = risk_manager.calculate_dynamic_take_profit(
            entry_price=entry_price,
            signal_strength=signal_strength,
            volatility=0.02,
            atr=2.0
        )
        
        # 4. Check if we should trade
        should_trade = risk_manager.should_trade(
            signal_strength=signal_strength,
            market_conditions={'volatility': 0.02},
            current_drawdown=0.0
        )
        
        logger.info("✅ Integration Test PASSED")
        logger.info(f"   • Ensemble prediction: {ensemble_pred[0]:.4f}")
        logger.info(f"   • Signal strength: {signal_strength:.3f}")
        logger.info(f"   • Position size: {position_size:.3f}")
        logger.info(f"   • Stop loss: {stop_loss:.2f}")
        logger.info(f"   • Take profit: {take_profit:.2f}")
        logger.info(f"   • Should trade: {should_trade}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration Test FAILED: {str(e)}")
        raise

def main():
    """Run all tests"""
    logger.info("🚀 STARTING MAXIMUM INTELLIGENCE ENHANCEMENTS TEST")
    logger.info("="*60)
    
    try:
        # Test individual parts
        test_feature_engineering()
        X, y = test_feature_engineering()  # Get data for model training
        test_model_training(X, y)
        test_risk_management()
        
        # Test integration
        test_integration()
        
        logger.info("\n" + "="*60)
        logger.info("🎉 ALL TESTS PASSED! MAXIMUM INTELLIGENCE ENHANCEMENTS WORKING")
        logger.info("="*60)
        logger.info("✅ Part 1: Advanced Feature Engineering & Selection")
        logger.info("✅ Part 2: Advanced Model Training & Optimization")
        logger.info("✅ Part 3: Advanced Risk Management & Psychology")
        logger.info("✅ Integration: All parts work together seamlessly")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {str(e)}")
        logger.error("Please check the error and fix the issue.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 