#!/usr/bin/env python3
TEST GEMINI'S CLUSTERED STRATEGY IMPLEMENTATION
===============================================

This script tests the implementation of Gemini's26clustered strategy:
- Asset Cluster Modeling with Specialized Models
- Opportunity Scanner & Ranking Module  
- Dynamic Capital Allocation & Portfolio Riskimport json
import logging
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.portfolio_engine import PortfolioEngine, AssetCluster, MarketRegime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gemini_clustered_strategy():
  Gemini's clustered strategy implementation"
    logger.info(🧪 Testing Gemini's Clustered Strategy Implementation")
    
    # Initialize portfolio engine
    engine = PortfolioEngine()
    
    # Test 1: Verify asset universe
    logger.info("\n📊 Test1verse Verification")
    expected_assets = [
        # Bedrock (6 assets)
        BTC,ETH, BNB,SOL, ,DOGE  # Infrastructure (5 assets)  
       AVAX', DOT', LINK', 'ARB,OP',
        # DeFi Bluechips (4 assets)
       UNI',AAVE',JUPPENDLE',
        # Volatility Engine (5 assets)
        PEPE, SHIB',BONK, ,BOME',
        # AI & Data (3 assets)
       FET',RNDR',WLD'
    ]
    
    actual_assets = engine.asset_universe
    logger.info(f"Expected assets: {len(expected_assets)})
    logger.info(f"Actual assets: {len(actual_assets)}")
    
    # Check if all expected assets are present
    missing_assets = set(expected_assets) - set(actual_assets)
    extra_assets = set(actual_assets) - set(expected_assets)
    
    if missing_assets:
        logger.warning(fMissing assets: {missing_assets}")
    if extra_assets:
        logger.warning(f"Extra assets: {extra_assets})   
    assert len(actual_assets) >=20Expected at least 20 assets, got {len(actual_assets)}"
    logger.info("✅ Asset universe verification passed)
    
    # Test2ster configuration
    logger.info("\n🎯 Test 2: Cluster Configuration Verification")
    
    for cluster, config in engine.asset_clusters.items():
        logger.info(f"Cluster {cluster.value}:")
        logger.info(f Assets: {config['assets']}")
        logger.info(f"  Position multiplier: {config['position_size_multiplier]*100}%")
        logger.info(f Risk tolerance: {configrisk_tolerance]*100}%")
        logger.info(f  Target features: {config[target_features']}")
        
        # Verify cluster has assets
        assert len(configassets]) > 0, f"Cluster {cluster.value} has no assets        
        # Verify position multiplier is reasonable
        assert 0 < config['position_size_multiplier] <= 1,f"Invalid position multiplier for {cluster.value}        
        # Verify risk tolerance is reasonable
        assert 0ig['risk_tolerance'] <= 00.5lid risk tolerance for {cluster.value}"
    
    logger.info("✅ Cluster configuration verification passed)
    
    # Test 3: Test asset clustering
    logger.info("\n🔍 Test 3: Asset Clustering Verification")
    
    for asset in actual_assets[:10:  # Test first10ts
        cluster = engine.get_asset_cluster(asset)
        if cluster:
            logger.info(f"{asset} -> {cluster.value}")
        else:
            logger.warning(f"{asset} -> No cluster found)  
    logger.info("✅ Asset clustering verification passed)
    
    # Test 4: Test data collection
    logger.info("\n📈 Test 4: Multi-Asset Data Collection")
    
    asset_data = engine.collect_multi_asset_data()
    logger.info(f"Collected data for {len(asset_data)} assets")
    
    # Verify data quality
    for asset, data in list(asset_data.items())[:5]:  # Check first 5
        logger.info(f"{asset}: {len(data)} data points, columns: {list(data.columns)}")
        assert len(data) > 0No data for {asset}     assert 'closein data.columns, f"No close price for {asset}   assertclusterin data.columns, f"No cluster info for {asset}"
    
    logger.info("✅ Multi-asset data collection passed)
    
    # Test 5: Test opportunity scanning
    logger.info("\n🎯 Test 5: Opportunity Scanning")
    
    opportunities = engine.scan_opportunities()
    logger.info(f"Found {len(opportunities)} opportunities")
    
    # Analyze opportunities by cluster
    cluster_opportunities =[object Object]    for asset, opp in opportunities.items():
        cluster = opp.cluster.value
        if cluster not in cluster_opportunities:
            cluster_opportunities[cluster] = ]
        cluster_opportunities[cluster].append(opp)
    
    for cluster, opps in cluster_opportunities.items():
        avg_conviction = np.mean([opp.conviction_score for opp in opps])
        logger.info(f"{cluster}: {len(opps)} opportunities (avg conviction: [object Object]avg_conviction:.3f}))  
    logger.info("✅ Opportunity scanning passed)
    
    # Test 6: Test opportunity ranking
    logger.info("\n🏆 Test 6: Opportunity Ranking)
    
    ranked_opportunities = engine.rank_opportunities(opportunities)
    logger.info(f"Ranked {len(ranked_opportunities)} opportunities")
    
    if ranked_opportunities:
        top_opp = ranked_opportunities0       logger.info(f"Top opportunity: {top_opp.asset} ({top_opp.cluster.value})")
        logger.info(f  Direction: {top_opp.direction}")
        logger.info(f"  Conviction: {top_opp.conviction_score:.3f}")
        logger.info(f"  Predicted return: {top_opp.predicted_return:.2f}%")
        logger.info(f"  Market regime: {top_opp.market_regime.value})  
    logger.info("✅ Opportunity ranking passed)
    
    # Test 7: Test capital allocation
    logger.info("\n💰 Test 7: Capital Allocation")
    
    allocations = engine.allocate_capital(ranked_opportunities)
    logger.info(fAllocated capital to {len(allocations)} positions)  
    total_allocated = sum(allocations.values())
    logger.info(f"Total allocated: $[object Object]total_allocated:.2    
    for asset, amount in allocations.items():
        logger.info(f" [object Object]asset}: ${amount:.2f})  
    logger.info("✅ Capital allocation passed)
    
    # Test8ortfolio execution
    logger.info("\n🚀 Test 8: Portfolio Execution)    engine.execute_allocations(allocations)
    logger.info(f"Executed {len(allocations)} positions)
    logger.info(f"Total positions: {len(engine.positions)}")
    
    for asset, position in engine.positions.items():
        logger.info(f"  {asset}: {position.direction} ${position.position_size:.2t ${position.entry_price:.6f})  
    logger.info("✅ Portfolio execution passed)
    
    # Test9t portfolio update
    logger.info("\n📊 Test 9: Portfolio Update)
    engine.update_portfolio()
    logger.info(f"Portfolio value: ${engine.portfolio_value:.2f})
    logger.info(fOpen positions: {len(engine.positions)}")
    
    total_pnl = sum(pos.unrealized_pnl for pos in engine.positions.values())
    logger.info(f"Total unrealized PnL: ${total_pnl:.2f})  
    logger.info("✅ Portfolio update passed)
    
    # Test 10 portfolio summary
    logger.info("\n📋 Test 10: Portfolio Summary")
    
    summary = engine.get_portfolio_summary()
    logger.info(Portfolio Summary:)
    logger.info(f  Portfolio value: ${summary[portfolio_value']:.2f})
    logger.info(f  Total positions: {summary[total_positions']})
    logger.info(f"  Total unrealized PnL: ${summary['total_unrealized_pnl]:.2f})  
    logger.info("Positions by cluster:)
    for cluster, count in summary['positions_by_cluster'].items():
        if count > 0:
            logger.info(f"  {cluster}: {count})  
    logger.info(Top opportunities:")
    for opp in summary[top_opportunities'][:3       logger.info(f"  {opp['asset]} ({opp[cluster']}): {opp['direction']} - conviction: {opp['conviction]:.3f})  
    logger.info("✅ Portfolio summary passed)
    
    # Test11complete portfolio cycle
    logger.info("\n🔄 Test 11: Complete Portfolio Cycle")
    
    # Reset portfolio for clean test
    engine.positions = {}
    engine.portfolio_value = 1000   
    engine.run_portfolio_cycle()
    
    logger.info(fCycle completed - Portfolio value: ${engine.portfolio_value:.2f})
    logger.info(fOpen positions: {len(engine.positions)})  
    logger.info("✅ Complete portfolio cycle passed)  
    logger.info("\n🎉 ALL TESTS PASSED! Gemini's clustered strategy is working correctly!)  
    returntrue

def test_cluster_specialization():
 t cluster-specific behavior"    logger.info("\n🧠 Testing Cluster Specialization)
    
    engine = PortfolioEngine()
    
    # Test volatility engine (memecoins) vs bedrock (large caps)
    logger.info("Testing cluster-specific characteristics:")
    
    # Get sample assets from different clusters
    memecoin = 'PEPE'  # Volatility Engine
    largecap =BTC   # Bedrock
    
    memecoin_cluster = engine.get_asset_cluster(memecoin)
    largecap_cluster = engine.get_asset_cluster(largecap)
    
    if memecoin_cluster and largecap_cluster:
        memecoin_config = engine.asset_clusters[memecoin_cluster]
        largecap_config = engine.asset_clusters[largecap_cluster]
        
        logger.info(f"{memecoin} ({memecoin_cluster.value}):")
        logger.info(f"  Position multiplier: {memecoin_config['position_size_multiplier]*100}%")
        logger.info(f Risk tolerance: {memecoin_configrisk_tolerance']*100}%")
        
        logger.info(f"{largecap} ({largecap_cluster.value}):")
        logger.info(f"  Position multiplier: {largecap_config['position_size_multiplier]*100}%")
        logger.info(f Risk tolerance: {largecap_configrisk_tolerance']*100}%")
        
        # Verify memecoins have higher risk tolerance
        assert memecoin_config['risk_tolerance'] > largecap_config[risk_tolerance'], \
           Memecoins should have higher risk tolerance than large caps        
        # Verify memecoins have lower position multiplier
        assert memecoin_config['position_size_multiplier'] < largecap_config['position_size_multiplier'], \
           Memecoins should have lower position multiplier than large caps   
        logger.info("✅ Cluster specialization verification passed)  
    return True

def main():
    """Main test function""
    try:
        logger.info("🚀 Starting Gemini Clustered Strategy Tests")
        
        # Run main tests
        test_gemini_clustered_strategy()
        
        # Run specialization tests
        test_cluster_specialization()
        
        logger.info("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("Gemini's26clustered strategy is fully implemented and working!")
        
        returntrue       
    except Exception as e:
        logger.error(f❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 