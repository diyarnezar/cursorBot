"""
🚀 PROJECT HYPERION - OPPORTUNITY SCANNER & RANKING ENGINE
=========================================================

Implements Phase 2 from gemini_plan_new.md
Generates predictions for all 26 assets and ranks potential trades using conviction scores.

Author: Project Hyperion Team
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
import time

from config.training_config import training_config
from core.asset_cluster_manager import AssetClusterManager
from data.collectors.binance_collector import BinanceDataCollector
from data.processors.data_processor import DataProcessor
from modules.feature_engineering import EnhancedFeatureEngineer
from features.regime_detection.regime_detection_features import RegimeDetectionFeatures
from risk.maximum_intelligence_risk import MaximumIntelligenceRisk


class OpportunityScanner:
    """
    Opportunity Scanner & Ranking Engine
    Runs every minute to scan all 26 assets and rank trading opportunities
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the Opportunity Scanner"""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
        
        # Initialize components
        self.asset_cluster_manager = AssetClusterManager(config_path)
        self.data_collector = BinanceDataCollector()
        self.data_processor = DataProcessor(config=self.config)
        self.feature_engineer = EnhancedFeatureEngineer()
        self.regime_detector = RegimeDetectionFeatures()
        self.risk_manager = MaximumIntelligenceRisk(config=self.config)
        
        # Scanner state
        self.last_scan_time = None
        self.scan_interval = 60  # 60 seconds
        self.opportunities = {}
        self.ranked_trades = []
        self.market_regime = 'normal'
        
        # Performance tracking
        self.scan_count = 0
        self.opportunities_found = 0
        
        self.logger.info("🚀 Opportunity Scanner initialized")
    
    async def start_scanning(self):
        """Start continuous opportunity scanning"""
        self.logger.info("🚀 Starting continuous opportunity scanning...")
        
        while True:
            try:
                await self.scan_opportunities()
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in opportunity scanning: {e}")
                await asyncio.sleep(10)  # Short delay on error
    
    async def scan_opportunities(self):
        """Scan all 26 assets for trading opportunities"""
        try:
            start_time = time.time()
            self.scan_count += 1
            
            self.logger.info(f"🔍 Scanning opportunities (scan #{self.scan_count})")
            
            # Get all FDUSD pairs
            all_pairs = training_config.get_all_pairs()
            self.logger.info(f"📊 Scanning {len(all_pairs)} pairs: {all_pairs}")
            
            # Update market regime
            self.market_regime = await self._detect_market_regime()
            self.logger.info(f"📈 Market regime detected: {self.market_regime}")
            
            # Collect recent data for all pairs
            opportunities = {}
            data_collected = 0
            predictions_generated = 0
            
            for pair in all_pairs:
                try:
                    self.logger.debug(f"🔍 Scanning {pair}...")
                    
                    # Get recent data (last 1000 candles)
                    data = await self._get_recent_data(pair, limit=1000)
                    
                    if data is not None and not data.empty:
                        data_collected += 1
                        self.logger.debug(f"✅ Data collected for {pair}: {len(data)} rows")
                        
                        # Generate features
                        features = self.feature_engineer.enhance_features(data)
                        self.logger.debug(f"🔧 Features generated for {pair}: {len(features.columns)} features")
                        
                        # Get cluster for this asset
                        cluster_name = self.asset_cluster_manager.get_cluster_for_asset(pair)
                        
                        if cluster_name:
                            self.logger.debug(f"🏷️ {pair} assigned to cluster: {cluster_name}")
                            
                            # Generate predictions using cluster-specific models
                            predictions = self.asset_cluster_manager.predict_cluster(cluster_name, features)
                            
                            if predictions:
                                predictions_generated += 1
                                self.logger.debug(f"📊 Predictions generated for {pair}: {predictions}")
                                
                                # Calculate conviction score
                                market_data = {
                                    'market_regime': self.market_regime,
                                    'volatility': data['volatility'].iloc[-1] if 'volatility' in data.columns else 0.02,
                                    'volume': data['volume'].iloc[-1] if 'volume' in data.columns else 0
                                }
                                
                                conviction_score = self.asset_cluster_manager.get_cluster_conviction_score(
                                    cluster_name, predictions, market_data
                                )
                                
                                self.logger.debug(f"🎯 Conviction score for {pair}: {conviction_score:.4f}")
                                
                                # Create opportunity
                                opportunity = {
                                    'pair': pair,
                                    'cluster': cluster_name,
                                    'predictions': predictions,
                                    'conviction_score': conviction_score,
                                    'current_price': data['close'].iloc[-1],
                                    'volume_24h': data['volume'].sum() if 'volume' in data.columns else 0,
                                    'volatility': data['volatility'].iloc[-1] if 'volatility' in data.columns else 0.02,
                                    'timestamp': datetime.now(),
                                    'market_regime': self.market_regime
                                }
                                
                                opportunities[pair] = opportunity
                                self.logger.info(f"🎯 Opportunity found for {pair}: conviction={conviction_score:.4f}, cluster={cluster_name}")
                            else:
                                self.logger.debug(f"❌ No predictions generated for {pair}")
                        else:
                            self.logger.warning(f"⚠️ No cluster found for {pair}")
                    else:
                        self.logger.warning(f"⚠️ No data available for {pair}")
                
                except Exception as e:
                    self.logger.error(f"❌ Error scanning {pair}: {e}")
                    continue
            
            # Rank opportunities
            ranked_opportunities = self._rank_opportunities(opportunities)
            
            # Update state
            self.opportunities = opportunities
            self.ranked_trades = ranked_opportunities
            self.last_scan_time = datetime.now()
            self.opportunities_found = len(opportunities)
            
            scan_duration = time.time() - start_time
            
            self.logger.info(f"✅ Scan completed in {scan_duration:.2f}s")
            self.logger.info(f"📊 Data collected: {data_collected}/{len(all_pairs)} pairs")
            self.logger.info(f"📊 Predictions generated: {predictions_generated}/{len(all_pairs)} pairs")
            self.logger.info(f"🎯 Opportunities found: {len(opportunities)}")
            
            # Log top opportunities
            if ranked_opportunities:
                self._log_top_opportunities(ranked_opportunities[:5])
            else:
                self.logger.warning("⚠️ No opportunities found in this scan")
            
        except Exception as e:
            self.logger.error(f"❌ Error in opportunity scanning: {e}")
    
    def _rank_opportunities(self, opportunities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank opportunities by conviction score and risk/reward"""
        try:
            ranked = []
            
            for pair, opp in opportunities.items():
                # Calculate risk/reward ratio
                risk_reward = self._calculate_risk_reward(opp)
                
                # Calculate final score (conviction * risk/reward * cluster multiplier)
                cluster_multiplier = self.asset_cluster_manager.get_position_size_multiplier(pair)
                
                final_score = (
                    opp['conviction_score'] * 
                    risk_reward * 
                    cluster_multiplier
                )
                
                ranked_opp = opp.copy()
                ranked_opp['risk_reward_ratio'] = risk_reward
                ranked_opp['final_score'] = final_score
                ranked_opp['recommended_position_size'] = self._calculate_position_size(opp, final_score)
                
                ranked.append(ranked_opp)
            
            # Sort by final score (descending)
            ranked.sort(key=lambda x: x['final_score'], reverse=True)
            
            return ranked
            
        except Exception as e:
            self.logger.error(f"❌ Error ranking opportunities: {e}")
            return []
    
    def _calculate_risk_reward(self, opportunity: Dict[str, Any]) -> float:
        """Calculate risk/reward ratio for an opportunity"""
        try:
            # Get cluster characteristics
            cluster_name = opportunity['cluster']
            characteristics = self.asset_cluster_manager.get_cluster_characteristics(cluster_name)
            
            # Base risk/reward from volatility
            volatility = opportunity['volatility']
            base_risk_reward = 1.0 / (1.0 + volatility * 10)  # Higher volatility = lower risk/reward
            
            # Adjust for market regime
            regime_adjustment = 1.0
            if opportunity['market_regime'] == 'high_volatility':
                regime_adjustment = 0.7
            elif opportunity['market_regime'] == 'low_volatility':
                regime_adjustment = 1.3
            
            # Adjust for cluster characteristics
            cluster_adjustment = 1.0
            if characteristics.get('volatility') == 'extreme':
                cluster_adjustment = 0.5
            elif characteristics.get('volatility') == 'high':
                cluster_adjustment = 0.8
            
            risk_reward = base_risk_reward * regime_adjustment * cluster_adjustment
            
            return max(risk_reward, 0.1)  # Minimum 0.1
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk/reward: {e}")
            return 0.5  # Default value
    
    def _calculate_position_size(self, opportunity: Dict[str, Any], final_score: float) -> float:
        """Calculate recommended position size based on opportunity score"""
        try:
            # Base position size (percentage of portfolio)
            base_size = 0.02  # 2% base position
            
            # Adjust by final score
            score_adjustment = min(final_score * 2, 1.0)  # Cap at 100%
            
            # Adjust by cluster multiplier
            cluster_multiplier = self.asset_cluster_manager.get_position_size_multiplier(opportunity['pair'])
            
            # Calculate final position size
            position_size = base_size * score_adjustment * cluster_multiplier
            
            # Apply risk limits
            max_position = 0.05  # Maximum 5% per position
            position_size = min(position_size, max_position)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0.01  # Default 1%
    
    async def _get_recent_data(self, pair: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get recent data for a pair"""
        try:
            # Get 1-minute candles
            data = self.data_collector.get_klines(pair, "1m", limit=limit)
            
            if data is not None and not data.empty:
                # Process data
                processed_data = self.data_processor.clean_data(data, pair)
                return processed_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting data for {pair}: {e}")
            return None
    
    async def _detect_market_regime(self) -> str:
        """Detect current market regime"""
        try:
            # Get BTC data as market indicator
            btc_data = await self._get_recent_data('BTCFDUSD', limit=1000)
            
            if btc_data is not None and not btc_data.empty:
                # Calculate market regime indicators
                volatility = btc_data['close'].pct_change().rolling(20).std().iloc[-1]
                
                if volatility > 0.05:  # High volatility threshold
                    return 'high_volatility'
                elif volatility < 0.02:  # Low volatility threshold
                    return 'low_volatility'
                else:
                    return 'normal'
            
            return 'normal'
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting market regime: {e}")
            return 'normal'
    
    def _log_top_opportunities(self, top_opportunities: List[Dict[str, Any]]):
        """Log top opportunities for monitoring"""
        try:
            self.logger.info("🏆 TOP OPPORTUNITIES:")
            self.logger.info("=" * 80)
            
            for i, opp in enumerate(top_opportunities, 1):
                self.logger.info(
                    f"{i}. {opp['pair']} ({opp['cluster']}) - "
                    f"Score: {opp['final_score']:.3f}, "
                    f"Conviction: {opp['conviction_score']:.3f}, "
                    f"Position: {opp['recommended_position_size']:.1%}"
                )
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"❌ Error logging opportunities: {e}")
    
    def get_top_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top ranked opportunities"""
        return self.ranked_trades[:limit]
    
    def get_opportunity_by_pair(self, pair: str) -> Optional[Dict[str, Any]]:
        """Get opportunity for a specific pair"""
        return self.opportunities.get(pair)
    
    def get_market_regime(self) -> str:
        """Get current market regime"""
        return self.market_regime
    
    def get_scan_stats(self) -> Dict[str, Any]:
        """Get scanning statistics"""
        return {
            'scan_count': self.scan_count,
            'opportunities_found': self.opportunities_found,
            'last_scan_time': self.last_scan_time,
            'market_regime': self.market_regime,
            'total_pairs': len(training_config.get_all_pairs())
        }
    
    def get_cluster_opportunities(self, cluster_name: str) -> List[Dict[str, Any]]:
        """Get opportunities for a specific cluster"""
        return [
            opp for opp in self.ranked_trades 
            if opp['cluster'] == cluster_name
        ]
    
    def export_opportunities(self, filepath: str = None):
        """Export opportunities to JSON file"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"opportunities/opportunities_{timestamp}.json"
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'market_regime': self.market_regime,
                'scan_stats': self.get_scan_stats(),
                'opportunities': self.opportunities,
                'ranked_trades': self.ranked_trades
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Exported opportunities to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting opportunities: {e}") 