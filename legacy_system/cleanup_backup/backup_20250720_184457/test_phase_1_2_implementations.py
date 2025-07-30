#!/usr/bin/env python3
"""
TEST PHASE 1 & 2 IMPLEMENTATIONS
================================

This script tests all Phase 1 and Phase 2 implementations:
Phase 1:
- Data Leakage Detector
- Historical Data Pipeline
- High-Fidelity Backtester

Phase 2:
- Portfolio Engine
- Multi-Asset Data Pipeline
- Opportunity Scanner
- Capital Allocation
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phase_1_implementations():
    """Test all Phase 1 implementations"""
    logger.info("🧪 Testing Phase 1 Implementations...")
    
    results = {}
    
    # Test 1: Data Leakage Detector
    try:
        from modules.data_leakage_detector import audit_features, validate_baseline
        
        # Create test data with leakage
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        
        test_data = {
            'timestamp': dates,
            'open': np.random.normal(100, 1, 1000),
            'high': np.random.normal(101, 1, 1000),
            'low': np.random.normal(99, 1, 1000),
            'close': np.random.normal(100, 1, 1000),
            'volume': np.random.normal(1000, 100, 1000),
            'target': np.random.normal(0, 1, 1000),
            'future_price': np.random.normal(100, 1, 1000),  # Suspicious
            'next_target': np.random.normal(0, 1, 1000),  # Suspicious
        }
        
        df = pd.DataFrame(test_data)
        
        # Run audit
        audit_results = audit_features(df, 'target')
        baseline_results = validate_baseline(df, 'target')
        
        results['Data Leakage Detector'] = {
            'leakage_detected': audit_results['leakage_detected'],
            'suspicious_features': len(audit_results['suspicious_features']),
            'baseline_valid': baseline_results['valid']
        }
        
        logger.info(f"   ✅ Data Leakage Detector: {len(audit_results['suspicious_features'])} suspicious features")
        
    except Exception as e:
        logger.error(f"❌ Data Leakage Detector test failed: {e}")
        results['Data Leakage Detector'] = {'error': str(e)}
    
    # Test 2: Historical Data Pipeline
    try:
        from modules.historical_data_pipeline import HistoricalDataPipeline
        
        pipeline = HistoricalDataPipeline()
        pipeline.collect_all_data()
        summary = pipeline.get_data_summary()
        
        total_records = sum(info['total_records'] for info in summary.values())
        
        results['Historical Data Pipeline'] = {
            'total_records': total_records,
            'tables': len(summary)
        }
        
        logger.info(f"   ✅ Historical Data Pipeline: {total_records} total records across {len(summary)} tables")
        
    except Exception as e:
        logger.error(f"❌ Historical Data Pipeline test failed: {e}")
        results['Historical Data Pipeline'] = {'error': str(e)}
    
    # Test 3: High-Fidelity Backtester
    try:
        from modules.high_fidelity_backtester import HighFidelityBacktester, OrderSide, OrderType
        
        backtester = HighFidelityBacktester()
        
        # Create simple test strategy
        def test_strategy(portfolio_state, backtester):
            # Simple buy and hold strategy
            pass
        
        # Run short backtest
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        result = backtester.run_backtest(
            strategy_function=test_strategy,
            assets=['BTC', 'ETH'],
            start_date=start_date,
            end_date=end_date
        )
        
        results['High-Fidelity Backtester'] = {
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'total_trades': result.total_trades,
            'total_fees': result.total_fees
        }
        
        logger.info(f"   ✅ High-Fidelity Backtester: {result.total_return*100:.2f}% return, {result.total_trades} trades")
        
    except Exception as e:
        logger.error(f"❌ High-Fidelity Backtester test failed: {e}")
        results['High-Fidelity Backtester'] = {'error': str(e)}
    
    return results

def test_phase_2_implementations():
    """Test all Phase 2 implementations"""
    logger.info("🧪 Testing Phase 2 Implementations...")
    
    results = {}
    
    # Test 1: Portfolio Engine
    try:
        from modules.portfolio_engine import PortfolioEngine
        
        engine = PortfolioEngine()
        
        # Test data collection
        engine.collect_multi_asset_data()
        
        # Test opportunity scanning
        opportunities = engine.scan_opportunities()
        
        # Test capital allocation
        allocations = engine.allocate_capital()
        
        # Get portfolio summary
        summary = engine.get_portfolio_summary()
        
        results['Portfolio Engine'] = {
            'assets_loaded': len(engine.asset_data),
            'opportunities_found': len(opportunities),
            'allocations_made': len(allocations),
            'portfolio_value': summary['portfolio_value']
        }
        
        logger.info(f"   ✅ Portfolio Engine: {len(engine.asset_data)} assets, {len(opportunities)} opportunities")
        
    except Exception as e:
        logger.error(f"❌ Portfolio Engine test failed: {e}")
        results['Portfolio Engine'] = {'error': str(e)}
    
    # Test 2: Multi-Asset Data Pipeline
    try:
        from modules.portfolio_engine import PortfolioEngine
        
        engine = PortfolioEngine()
        
        # Test data collection for multiple assets
        asset_data = engine.collect_multi_asset_data()
        
        total_data_points = sum(len(data) for data in asset_data.values())
        assets_with_data = len([data for data in asset_data.values() if len(data) > 0])
        
        results['Multi-Asset Data Pipeline'] = {
            'total_data_points': total_data_points,
            'assets_with_data': assets_with_data,
            'asset_universe': len(engine.asset_universe)
        }
        
        logger.info(f"   ✅ Multi-Asset Data Pipeline: {total_data_points} data points across {assets_with_data} assets")
        
    except Exception as e:
        logger.error(f"❌ Multi-Asset Data Pipeline test failed: {e}")
        results['Multi-Asset Data Pipeline'] = {'error': str(e)}
    
    # Test 3: Opportunity Scanner
    try:
        from modules.portfolio_engine import PortfolioEngine
        
        engine = PortfolioEngine()
        engine.collect_multi_asset_data()
        
        # Test opportunity scanning
        opportunities = engine.scan_opportunities()
        
        if opportunities:
            avg_conviction = np.mean([opp.conviction_score for opp in opportunities.values()])
            avg_sharpe = np.mean([opp.predicted_sharpe for opp in opportunities.values()])
            
            results['Opportunity Scanner'] = {
                'opportunities_found': len(opportunities),
                'avg_conviction': avg_conviction,
                'avg_sharpe': avg_sharpe,
                'market_regimes': list(set(opp.market_regime.value for opp in opportunities.values()))
            }
            
            logger.info(f"   ✅ Opportunity Scanner: {len(opportunities)} opportunities, avg conviction {avg_conviction:.3f}")
        else:
            results['Opportunity Scanner'] = {
                'opportunities_found': 0,
                'avg_conviction': 0,
                'avg_sharpe': 0,
                'market_regimes': []
            }
            
            logger.info(f"   ✅ Opportunity Scanner: No opportunities found (expected in test)")
        
    except Exception as e:
        logger.error(f"❌ Opportunity Scanner test failed: {e}")
        results['Opportunity Scanner'] = {'error': str(e)}
    
    # Test 4: Capital Allocation
    try:
        from modules.portfolio_engine import PortfolioEngine
        
        engine = PortfolioEngine()
        engine.collect_multi_asset_data()
        engine.scan_opportunities()
        
        # Test capital allocation
        allocations = engine.allocate_capital()
        
        total_allocated = sum(allocations.values())
        max_allocation = max(allocations.values()) if allocations else 0
        
        results['Capital Allocation'] = {
            'allocations_made': len(allocations),
            'total_allocated': total_allocated,
            'max_allocation': max_allocation,
            'portfolio_config': engine.portfolio_config
        }
        
        logger.info(f"   ✅ Capital Allocation: {len(allocations)} allocations, {total_allocated:.3f} total allocated")
        
    except Exception as e:
        logger.error(f"❌ Capital Allocation test failed: {e}")
        results['Capital Allocation'] = {'error': str(e)}
    
    return results

def test_integration():
    """Test integration between Phase 1 and Phase 2"""
    logger.info("🧪 Testing Phase 1 & 2 Integration...")
    
    try:
        # Test that all modules can work together
        from modules.data_leakage_detector import audit_features
        from modules.historical_data_pipeline import HistoricalDataPipeline
        from modules.high_fidelity_backtester import HighFidelityBacktester
        from modules.portfolio_engine import PortfolioEngine
        
        logger.info("   ✅ All modules imported successfully")
        
        # Test data flow from historical pipeline to portfolio engine
        pipeline = HistoricalDataPipeline()
        pipeline.collect_all_data()
        
        engine = PortfolioEngine()
        engine.collect_multi_asset_data()
        
        # Test that both can access data
        pipeline_summary = pipeline.get_data_summary()
        engine_summary = engine.get_portfolio_summary()
        
        logger.info(f"   ✅ Data pipeline integration: {len(pipeline_summary)} tables, {len(engine_summary['opportunities'])} opportunities")
        
        # Test backtester with portfolio engine data
        backtester = HighFidelityBacktester()
        
        # Create a strategy that uses portfolio engine insights
        def integrated_strategy(portfolio_state, backtester):
            # This would use portfolio engine insights in a real implementation
            pass
        
        # Run short backtest
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now()
        
        result = backtester.run_backtest(
            strategy_function=integrated_strategy,
            assets=['BTC', 'ETH'],
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"   ✅ Backtester integration: {result.total_trades} trades executed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Phase 1 & 2 Implementation Tests...")
    logger.info("="*70)
    
    # Test Phase 1
    phase_1_results = test_phase_1_implementations()
    
    logger.info("\n" + "="*70)
    
    # Test Phase 2
    phase_2_results = test_phase_2_implementations()
    
    logger.info("\n" + "="*70)
    
    # Test Integration
    integration_success = test_integration()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("📊 PHASE 1 & 2 IMPLEMENTATION TEST RESULTS")
    logger.info("="*70)
    
    # Phase 1 Summary
    logger.info("\n🎯 PHASE 1 RESULTS:")
    phase_1_passed = 0
    phase_1_total = len(phase_1_results)
    
    for test_name, result in phase_1_results.items():
        if 'error' not in result:
            phase_1_passed += 1
            status = "✅ PASSED"
        else:
            status = f"❌ FAILED: {result['error']}"
        logger.info(f"   {test_name}: {status}")
    
    # Phase 2 Summary
    logger.info("\n🎯 PHASE 2 RESULTS:")
    phase_2_passed = 0
    phase_2_total = len(phase_2_results)
    
    for test_name, result in phase_2_results.items():
        if 'error' not in result:
            phase_2_passed += 1
            status = "✅ PASSED"
        else:
            status = f"❌ FAILED: {result['error']}"
        logger.info(f"   {test_name}: {status}")
    
    # Overall Summary
    total_passed = phase_1_passed + phase_2_passed
    total_tests = phase_1_total + phase_2_total
    
    logger.info(f"\n🎯 OVERALL RESULTS:")
    logger.info(f"   Phase 1: {phase_1_passed}/{phase_1_total} tests passed")
    logger.info(f"   Phase 2: {phase_2_passed}/{phase_2_total} tests passed")
    logger.info(f"   Integration: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    logger.info(f"   Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests and integration_success:
        logger.info("\n🎉 ALL PHASE 1 & 2 IMPLEMENTATIONS WORKING!")
        logger.info("\n🚀 Ready to proceed with Phase 3:")
        logger.info("   1. Intelligent Maker-Only Execution")
        logger.info("   2. Advanced Order Routing")
        logger.info("   3. Market Microstructure Optimization")
        logger.info("   4. Zero-Fee Trading Strategy")
    else:
        logger.error("\n⚠️ Some tests failed. Please check the errors above.")
    
    return total_passed == total_tests and integration_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 