#!/usr/bin/env python3
"le Test for Gemini's Clustered Strategy"

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.portfolio_engine import PortfolioEngine, AssetCluster

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_clustered_strategy():
    """Test the clustered strategy implementation"
    logger.info(🧪 Testing Gemini's Clustered Strategy Implementation")
    
    # Initialize portfolio engine
    engine = PortfolioEngine()
    
    # Test 1: Verify asset universe
    logger.info(f📊 Asset universe: {len(engine.asset_universe)} assets)
    logger.info(f"Assets: {engine.asset_universe})
    
    # Test 2: Verify clusters
    logger.info("\n🎯 Asset Clusters:)
    for cluster, config in engine.asset_clusters.items():
        logger.info(f"  {cluster.value}: {len(config[assets'])} assets")
        logger.info(f"    Assets: {config['assets']}")
        logger.info(f"    Position multiplier: {config['position_size_multiplier]*100}%")
        logger.info(f"    Risk tolerance: {configrisk_tolerance']*100)
    
    # Test 3: Test asset clustering
    logger.info("\n🔍 Asset Clustering:")
    for asset in engine.asset_universe[:5]:  # Test first5
        cluster = engine.get_asset_cluster(asset)
        if cluster:
            logger.info(f"  {asset} -> {cluster.value}")
        else:
            logger.warning(f"  {asset} -> No cluster found)
    
    # Test 4: Test data collection
    logger.info("\n📈 Data Collection:")
    asset_data = engine.collect_multi_asset_data()
    logger.info(f"  Collected data for {len(asset_data)} assets)
    
    # Test 5: Test opportunity scanning
    logger.info("\n🎯 Opportunity Scanning:")
    opportunities = engine.scan_opportunities()
    logger.info(f"  Found {len(opportunities)} opportunities")
    
    if opportunities:
        # Show top opportunities by cluster
        cluster_opps = {}
        for asset, opp in opportunities.items():
            cluster = opp.cluster.value
            if cluster not in cluster_opps:
                cluster_opps[cluster] = []
            cluster_opps[cluster].append(opp)
        
        for cluster, opps in cluster_opps.items():
            avg_conviction = sum(opp.conviction_score for opp in opps) / len(opps)
            logger.info(f[object Object]cluster}: {len(opps)} opportunities (avg conviction: [object Object]avg_conviction:0.3)
    
    # Test 6: Portfolio cycle
    logger.info("\n🔄 Portfolio Cycle:")
    engine.run_portfolio_cycle()
    logger.info(f  Portfolio value: ${engine.portfolio_value:.2f})
    logger.info(f Open positions: {len(engine.positions)})
    
    # Test 7: Portfolio summary
    logger.info("\n📋 Portfolio Summary:")
    summary = engine.get_portfolio_summary()
    logger.info(f  Portfolio value: ${summary[portfolio_value']:.2f})
    logger.info(f  Total positions: {summary[total_positions']})
    logger.info(f"  Total unrealized PnL: ${summary['total_unrealized_pnl']:.2f})    logger.info("\n✅ Basic functionality test completed successfully!)
    return True

if __name__ == "__main__":
    try:
        test_clustered_strategy()
        print("\n🎉 SUCCESS: Gemini's clustered strategy is working!")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 