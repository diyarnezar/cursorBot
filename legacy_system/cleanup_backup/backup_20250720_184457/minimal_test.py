#!/usr/bin/env python3le test"

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.portfolio_engine import PortfolioEngine

def main():
    print(TestingClustered Strategy)
    
    engine = PortfolioEngine()
    
    print(Asset Universe:", len(engine.asset_universe), "assets")
    print(Assets:, engine.asset_universe)
    
    print("\nAsset Clusters:)
    for cluster, config in engine.asset_clusters.items():
        print(" , cluster.value,:len(config['assets']),assets)        print("    Assets:", config[assets'])
        print("    Position multiplier:", config['position_size_multiplier']*100, %)        print("    Risk tolerance:", configrisk_tolerance]*100%")
    
    print("\nAsset Clustering:")
    for asset in engine.asset_universe[:5]:
        cluster = engine.get_asset_cluster(asset)
        if cluster:
            print(  set, "->, cluster.value)
    
    print(nData Collection:")
    asset_data = engine.collect_multi_asset_data()
    print("  Collected data for", len(asset_data), assets")
    
    print("\nOpportunity Scanning:")
    opportunities = engine.scan_opportunities()
    print("  Found", len(opportunities),opportunities")
    
    if opportunities:
        cluster_opps = {}
        for asset, opp in opportunities.items():
            cluster = opp.cluster.value
            if cluster not in cluster_opps:
                cluster_opps[cluster] = []
            cluster_opps[cluster].append(opp)
        
        for cluster, opps in cluster_opps.items():
            avg_conviction = sum(opp.conviction_score for opp in opps) / len(opps)
            print(", cluster,:n(opps), "opportunities (avg conviction:", f[object Object]avg_conviction:.3f})")
    
    print(nPortfolio Cycle:")
    engine.run_portfolio_cycle()
    print(  Portfolio value: $", f{engine.portfolio_value:.2
    print("  Open positions:", len(engine.positions))
    
    print("\nPortfolio Summary:")
    summary = engine.get_portfolio_summary()
    print(  Portfolio value: $", f"{summary[portfolio_value']:.2    print(Total positions:", summary['total_positions'])
    print("  Total unrealized PnL: $", f"{summary['total_unrealized_pnl]:.2f}")
    
    print("\nSUCCESS: Clustered strategy is working!)
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1) 