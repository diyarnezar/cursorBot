#!/usr/bin/env python3le test of clustered strategy"

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.portfolio_engine import PortfolioEngine, AssetCluster

def main():
    print("Testing Gemini's Clustered Strategy Implementation")
    
    # Initialize portfolio engine
    engine = PortfolioEngine()
    
    # Show asset universe
    print(fundefinednAsset Universe: {len(engine.asset_universe)} assets")
    print(f"Assets: {engine.asset_universe})   # Show clusters
    print("\nAsset Clusters:)
    for cluster, config in engine.asset_clusters.items():
        print(f"  {cluster.value}: {len(config[assets'])} assets)
        print(f"    Assets: {config['assets']})
        print(f"    Position multiplier: {config['position_size_multiplier]*100}%)
        print(f"    Risk tolerance: {configrisk_tolerance']*100) 
    # Test asset clustering
    print("\nAsset Clustering:")
    for asset in engine.asset_universe[:5]:
        cluster = engine.get_asset_cluster(asset)
        if cluster:
            print(f"  {asset} -> {cluster.value})  
    # Test data collection
    print(nData Collection:")
    asset_data = engine.collect_multi_asset_data()
    print(f"  Collected data for {len(asset_data)} assets)# Test opportunity scanning
    print("\nOpportunity Scanning:")
    opportunities = engine.scan_opportunities()
    print(f"  Found {len(opportunities)} opportunities")
    
    if opportunities:
        cluster_opps = {}
        for asset, opp in opportunities.items():
            cluster = opp.cluster.value
            if cluster not in cluster_opps:
                cluster_opps[cluster] = []
            cluster_opps[cluster].append(opp)
        
        for cluster, opps in cluster_opps.items():
            avg_conviction = sum(opp.conviction_score for opp in opps) / len(opps)
            print(f[object Object]cluster}: {len(opps)} opportunities (avg conviction: [object Object]avg_conviction:0.3)  # Test portfolio cycle
    print(nPortfolio Cycle:")
    engine.run_portfolio_cycle()
    print(f  Portfolio value: ${engine.portfolio_value:.2f}")
    print(f Open positions: {len(engine.positions)})  # Show portfolio summary
    print("\nPortfolio Summary:")
    summary = engine.get_portfolio_summary()
    print(f  Portfolio value: ${summary[portfolio_value']:.2f}")
    print(f  Total positions: {summary[total_positions]}")
    print(f"  Total unrealized PnL: ${summary['total_unrealized_pnl]:.2f}")
    
    print(undefinednSUCCESS: Gemini's clustered strategy is working!)
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 