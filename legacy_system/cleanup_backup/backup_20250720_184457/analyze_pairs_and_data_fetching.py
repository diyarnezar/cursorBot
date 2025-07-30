#!/usr/bin/env python3
"""
COMPREHENSIVE PAIRS & DATA FETCHING ANALYSIS
===========================================

This script analyzes:
1. All 26 FDUSD pairs integration status
2. Optimal data fetching strategy for 15 days training
3. Binance API limits compliance
4. Rate limiting calculations
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.portfolio_engine import PortfolioEngine, AssetCluster

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PairsAndDataAnalyzer:
    """Comprehensive analyzer for pairs integration and data fetching strategy"""
    
    def __init__(self):
        self.portfolio_engine = PortfolioEngine()
        
        # Binance API limits (conservative)
        self.binance_limits = {
            'requests_per_minute': 1000,  # Conservative: 1000 instead of 1200
            'requests_per_second': 16,    # Conservative: 16 instead of 20
            'klines_per_request': 1000,   # Max klines per request
            'safety_margin': 0.8          # Use 80% of limits
        }
        
        # Training parameters
        self.training_days = 15
        self.timeframes = ['1m', '5m', '15m']  # Multi-timeframe training
        
        logger.info("🔍 Pairs and Data Fetching Analyzer initialized")
    
    def analyze_pairs_integration(self) -> Dict:
        """Analyze the integration status of all 26 pairs"""
        logger.info("\n" + "="*60)
        logger.info("📊 PAIRS INTEGRATION ANALYSIS")
        logger.info("="*60)
        
        # Get current asset universe
        asset_universe = self.portfolio_engine.asset_universe
        asset_clusters = self.portfolio_engine.asset_clusters
        
        # Expected 26 pairs from the plan
        expected_pairs = [
            # Bedrock (6 pairs)
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE',
            # Infrastructure (5 pairs)
            'AVAX', 'DOT', 'LINK', 'ARB', 'OP',
            # DeFi Bluechips (4 pairs)
            'UNI', 'AAVE', 'JUP', 'PENDLE',
            # Volatility Engine (5 pairs)
            'PEPE', 'SHIB', 'BONK', 'WIF', 'BOME',
            # AI & Data (3 pairs)
            'FET', 'RNDR', 'WLD'
        ]
        
        # Analysis results
        analysis = {
            'total_pairs': len(asset_universe),
            'expected_pairs': len(expected_pairs),
            'missing_pairs': [],
            'extra_pairs': [],
            'cluster_breakdown': {},
            'integration_status': 'UNKNOWN',
            'fdusd_pairs': []
        }
        
        # Check for missing/extra pairs
        missing_pairs = set(expected_pairs) - set(asset_universe)
        extra_pairs = set(asset_universe) - set(expected_pairs)
        
        analysis['missing_pairs'] = list(missing_pairs)
        analysis['extra_pairs'] = list(extra_pairs)
        
        # Generate FDUSD pairs
        fdusd_pairs = [f"{asset}FDUSD" for asset in asset_universe]
        analysis['fdusd_pairs'] = fdusd_pairs
        
        # Cluster breakdown
        for cluster, config in asset_clusters.items():
            analysis['cluster_breakdown'][cluster.value] = {
                'assets': config['assets'],
                'count': len(config['assets']),
                'fdusd_pairs': [f"{asset}FDUSD" for asset in config['assets']]
            }
        
        # Integration status
        if len(missing_pairs) == 0 and len(extra_pairs) == 0:
            analysis['integration_status'] = 'PERFECT'
        elif len(missing_pairs) == 0:
            analysis['integration_status'] = 'EXTENDED'
        elif len(missing_pairs) <= 2:
            analysis['integration_status'] = 'GOOD'
        else:
            analysis['integration_status'] = 'INCOMPLETE'
        
        # Log results
        logger.info(f"✅ Total pairs integrated: {analysis['total_pairs']}")
        logger.info(f"📋 Expected pairs: {analysis['expected_pairs']}")
        logger.info(f"🎯 Integration status: {analysis['integration_status']}")
        
        if missing_pairs:
            logger.warning(f"❌ Missing pairs: {missing_pairs}")
        if extra_pairs:
            logger.info(f"➕ Extra pairs: {extra_pairs}")
        
        logger.info("\n📊 Cluster Breakdown:")
        for cluster, data in analysis['cluster_breakdown'].items():
            logger.info(f"   {cluster}: {data['count']} pairs - {data['assets']}")
        
        return analysis
    
    def calculate_data_fetching_strategy(self) -> Dict:
        """Calculate optimal data fetching strategy for 15 days training"""
        logger.info("\n" + "="*60)
        logger.info("📈 DATA FETCHING STRATEGY ANALYSIS")
        logger.info("="*60)
        
        # Get number of pairs
        num_pairs = len(self.portfolio_engine.asset_universe)
        
        # Calculate data requirements for 15 days
        data_requirements = {}
        
        for timeframe in self.timeframes:
            # Calculate data points needed
            if timeframe == '1m':
                minutes_per_day = 24 * 60
                data_points_per_day = minutes_per_day
            elif timeframe == '5m':
                minutes_per_day = 24 * 60
                data_points_per_day = minutes_per_day // 5
            elif timeframe == '15m':
                minutes_per_day = 24 * 60
                data_points_per_day = minutes_per_day // 15
            
            total_data_points = data_points_per_day * self.training_days
            
            # Calculate number of API requests needed
            # Binance allows max 1000 klines per request
            requests_per_pair = (total_data_points + 999) // 1000  # Ceiling division
            
            data_requirements[timeframe] = {
                'data_points_per_day': data_points_per_day,
                'total_data_points': total_data_points,
                'requests_per_pair': requests_per_pair,
                'total_requests': requests_per_pair * num_pairs
            }
        
        # Calculate optimal fetching strategy
        strategy = {
            'num_pairs': num_pairs,
            'training_days': self.training_days,
            'data_requirements': data_requirements,
            'fetching_strategy': {},
            'api_limits_compliance': {},
            'recommended_approach': {}
        }
        
        # Analyze each timeframe
        for timeframe, reqs in data_requirements.items():
            total_requests = reqs['total_requests']
            
            # Calculate time needed with rate limiting
            requests_per_minute = self.binance_limits['requests_per_minute'] * self.binance_limits['safety_margin']
            minutes_needed = total_requests / requests_per_minute
            
            # Calculate parallel fetching capability
            max_parallel_pairs = int(requests_per_minute / reqs['requests_per_pair']) if reqs['requests_per_pair'] > 0 else 0
            
            strategy['fetching_strategy'][timeframe] = {
                'total_requests': total_requests,
                'minutes_needed_sequential': minutes_needed,
                'max_parallel_pairs': max_parallel_pairs,
                'parallel_fetching_possible': max_parallel_pairs >= num_pairs,
                'optimal_batch_size': min(max_parallel_pairs, num_pairs)
            }
            
            # API limits compliance
            strategy['api_limits_compliance'][timeframe] = {
                'requests_per_minute_used': total_requests / minutes_needed if minutes_needed > 0 else 0,
                'percentage_of_limit': (total_requests / minutes_needed) / self.binance_limits['requests_per_minute'] if minutes_needed > 0 else 0,
                'within_limits': (total_requests / minutes_needed) <= self.binance_limits['requests_per_minute'] if minutes_needed > 0 else True
            }
        
        # Recommended approach
        strategy['recommended_approach'] = self._calculate_recommended_approach(strategy)
        
        # Log results
        logger.info(f"📊 Data Requirements for {self.training_days} days training:")
        for timeframe, reqs in data_requirements.items():
            logger.info(f"   {timeframe}: {reqs['total_data_points']:,} data points, {reqs['requests_per_pair']} requests/pair")
        
        logger.info(f"\n🎯 Fetching Strategy:")
        for timeframe, fetch_strat in strategy['fetching_strategy'].items():
            logger.info(f"   {timeframe}:")
            logger.info(f"     Total requests: {fetch_strat['total_requests']:,}")
            logger.info(f"     Time needed: {fetch_strat['minutes_needed_sequential']:.1f} minutes")
            logger.info(f"     Parallel possible: {fetch_strat['parallel_fetching_possible']}")
            logger.info(f"     Optimal batch size: {fetch_strat['optimal_batch_size']}")
        
        return strategy
    
    def _calculate_recommended_approach(self, strategy: Dict) -> Dict:
        """Calculate the recommended data fetching approach"""
        
        # Find the most efficient timeframe
        best_timeframe = None
        best_efficiency = float('inf')
        
        for timeframe, fetch_strat in strategy['fetching_strategy'].items():
            efficiency = fetch_strat['total_requests'] / fetch_strat['optimal_batch_size']
            if efficiency < best_efficiency:
                best_efficiency = efficiency
                best_timeframe = timeframe
        
        # Calculate optimal batch processing
        num_pairs = strategy['num_pairs']
        optimal_batch_size = strategy['fetching_strategy'][best_timeframe]['optimal_batch_size']
        num_batches = (num_pairs + optimal_batch_size - 1) // optimal_batch_size
        
        # Calculate timing - get requests_per_pair from data_requirements
        requests_per_pair = strategy['data_requirements'][best_timeframe]['requests_per_pair']
        requests_per_batch = requests_per_pair * optimal_batch_size
        requests_per_minute = self.binance_limits['requests_per_minute'] * self.binance_limits['safety_margin']
        time_per_batch = requests_per_batch / requests_per_minute
        total_time = time_per_batch * num_batches
        
        return {
            'recommended_timeframe': best_timeframe,
            'batch_size': optimal_batch_size,
            'num_batches': num_batches,
            'time_per_batch_minutes': time_per_batch,
            'total_time_minutes': total_time,
            'total_time_hours': total_time / 60,
            'parallel_processing': optimal_batch_size > 1,
            'rate_limit_utilization': (requests_per_batch / time_per_batch) / self.binance_limits['requests_per_minute'] if time_per_batch > 0 else 0
        }
    
    def analyze_api_limits_compliance(self) -> Dict:
        """Analyze compliance with Binance API limits"""
        logger.info("\n" + "="*60)
        logger.info("🔒 API LIMITS COMPLIANCE ANALYSIS")
        logger.info("="*60)
        
        # Current usage analysis
        num_pairs = len(self.portfolio_engine.asset_universe)
        
        # Different scenarios
        scenarios = {
            'real_time_monitoring': {
                'requests_per_minute': num_pairs * 2,  # 2 requests per pair per minute
                'description': 'Real-time monitoring (2 req/pair/min)'
            },
            'opportunity_scanning': {
                'requests_per_minute': num_pairs * 1,  # 1 request per pair per minute
                'description': 'Opportunity scanning (1 req/pair/min)'
            },
            'historical_data_15days': {
                'requests_per_minute': 50,  # Conservative batch processing
                'description': 'Historical data collection (15 days)'
            },
            'order_execution': {
                'requests_per_minute': num_pairs * 0.1,  # 1 order per 10 minutes per pair
                'description': 'Order execution (0.1 req/pair/min)'
            }
        }
        
        compliance_analysis = {
            'binance_limits': self.binance_limits,
            'scenarios': {},
            'total_usage': {},
            'recommendations': []
        }
        
        # Analyze each scenario
        total_usage = 0
        for scenario_name, scenario_data in scenarios.items():
            usage = scenario_data['requests_per_minute']
            percentage = usage / self.binance_limits['requests_per_minute']
            
            compliance_analysis['scenarios'][scenario_name] = {
                'usage_per_minute': usage,
                'percentage_of_limit': percentage,
                'within_limits': percentage <= 1.0,
                'safety_margin': 1.0 - percentage if percentage <= 1.0 else 0
            }
            
            total_usage += usage
        
        # Total usage analysis
        compliance_analysis['total_usage'] = {
            'total_requests_per_minute': total_usage,
            'percentage_of_limit': total_usage / self.binance_limits['requests_per_minute'],
            'within_limits': total_usage <= self.binance_limits['requests_per_minute'],
            'safety_margin': 1.0 - (total_usage / self.binance_limits['requests_per_minute']) if total_usage <= self.binance_limits['requests_per_minute'] else 0
        }
        
        # Generate recommendations
        if compliance_analysis['total_usage']['within_limits']:
            compliance_analysis['recommendations'].append("✅ All scenarios fit within API limits")
        else:
            compliance_analysis['recommendations'].append("⚠️ Total usage exceeds API limits - need optimization")
        
        # Log results
        logger.info(f"📊 API Usage Analysis:")
        for scenario_name, scenario_data in scenarios.items():
            analysis = compliance_analysis['scenarios'][scenario_name]
            status = "✅" if analysis['within_limits'] else "❌"
            logger.info(f"   {status} {scenario_data['description']}: {analysis['usage_per_minute']} req/min ({analysis['percentage_of_limit']:.1%})")
        
        total_analysis = compliance_analysis['total_usage']
        total_status = "✅" if total_analysis['within_limits'] else "❌"
        logger.info(f"\n{total_status} Total usage: {total_analysis['total_requests_per_minute']} req/min ({total_analysis['percentage_of_limit']:.1%})")
        
        return compliance_analysis
    
    def generate_implementation_plan(self) -> Dict:
        """Generate implementation plan for optimal data fetching"""
        logger.info("\n" + "="*60)
        logger.info("🚀 IMPLEMENTATION PLAN")
        logger.info("="*60)
        
        # Get analysis results
        pairs_analysis = self.analyze_pairs_integration()
        data_strategy = self.calculate_data_fetching_strategy()
        api_compliance = self.analyze_api_limits_compliance()
        
        # Generate implementation plan
        plan = {
            'pairs_status': pairs_analysis,
            'data_strategy': data_strategy,
            'api_compliance': api_compliance,
            'implementation_steps': [],
            'code_improvements': [],
            'monitoring_recommendations': []
        }
        
        # Implementation steps
        if pairs_analysis['integration_status'] != 'PERFECT':
            plan['implementation_steps'].append({
                'priority': 'HIGH',
                'action': 'Complete pairs integration',
                'details': f"Add missing pairs: {pairs_analysis['missing_pairs']}"
            })
        
        plan['implementation_steps'].append({
            'priority': 'HIGH',
            'action': 'Implement batch data fetching',
            'details': f"Use {data_strategy['recommended_approach']['batch_size']} pairs per batch"
        })
        
        plan['implementation_steps'].append({
            'priority': 'MEDIUM',
            'action': 'Add rate limiting monitoring',
            'details': 'Real-time API usage tracking and alerts'
        })
        
        # Code improvements
        plan['code_improvements'].append({
            'file': 'modules/data_ingestion.py',
            'improvement': 'Add batch processing for multiple pairs',
            'benefit': 'Reduce API calls and improve efficiency'
        })
        
        plan['code_improvements'].append({
            'file': 'modules/portfolio_engine.py',
            'improvement': 'Add real-time data validation',
            'benefit': 'Ensure data quality and completeness'
        })
        
        # Monitoring recommendations
        plan['monitoring_recommendations'].append({
            'metric': 'API usage per minute',
            'threshold': self.binance_limits['requests_per_minute'] * 0.8,
            'action': 'Alert when approaching 80% of limit'
        })
        
        plan['monitoring_recommendations'].append({
            'metric': 'Data collection success rate',
            'threshold': 0.95,
            'action': 'Alert when success rate drops below 95%'
        })
        
        # Log plan
        logger.info("📋 Implementation Steps:")
        for step in plan['implementation_steps']:
            logger.info(f"   {step['priority']}: {step['action']}")
            logger.info(f"      {step['details']}")
        
        logger.info("\n🔧 Code Improvements:")
        for improvement in plan['code_improvements']:
            logger.info(f"   {improvement['file']}: {improvement['improvement']}")
        
        return plan
    
    def run_comprehensive_analysis(self):
        """Run the complete analysis"""
        logger.info("🚀 Starting comprehensive pairs and data fetching analysis...")
        
        # Run all analyses
        pairs_analysis = self.analyze_pairs_integration()
        data_strategy = self.calculate_data_fetching_strategy()
        api_compliance = self.analyze_api_limits_compliance()
        implementation_plan = self.generate_implementation_plan()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 ANALYSIS SUMMARY")
        logger.info("="*60)
        
        logger.info(f"✅ Pairs Integration: {pairs_analysis['integration_status']}")
        logger.info(f"📈 Data Strategy: {data_strategy['recommended_approach']['recommended_timeframe']} timeframe")
        logger.info(f"🔒 API Compliance: {'✅' if api_compliance['total_usage']['within_limits'] else '❌'}")
        logger.info(f"⏱️  Data Collection Time: {data_strategy['recommended_approach']['total_time_hours']:.1f} hours")
        
        # Key findings
        logger.info("\n🎯 KEY FINDINGS:")
        
        if pairs_analysis['integration_status'] == 'PERFECT':
            logger.info("   ✅ All 26 pairs are perfectly integrated")
        else:
            logger.info(f"   ⚠️  {len(pairs_analysis['missing_pairs'])} pairs missing from integration")
        
        if api_compliance['total_usage']['within_limits']:
            logger.info("   ✅ API usage well within Binance limits")
        else:
            logger.info("   ⚠️  API usage exceeds limits - optimization needed")
        
        if data_strategy['recommended_approach']['parallel_processing']:
            logger.info("   ✅ Parallel processing possible for data collection")
        else:
            logger.info("   ⚠️  Sequential processing required for data collection")
        
        return {
            'pairs_analysis': pairs_analysis,
            'data_strategy': data_strategy,
            'api_compliance': api_compliance,
            'implementation_plan': implementation_plan
        }

def main():
    """Main function to run the analysis"""
    analyzer = PairsAndDataAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)
    
    # Save results to file
    with open('pairs_and_data_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("📄 Results saved to 'pairs_and_data_analysis.json'")
    
    return results

if __name__ == "__main__":
    main() 