"""
🚀 PROJECT HYPERION - DYNAMIC CAPITAL ALLOCATOR
==============================================

Implements Phase 2 from gemini_plan_new.md
Manages global risk budget and allocates capital to highest-ranked opportunities.

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

from core.opportunity_scanner import OpportunityScanner
from core.asset_cluster_manager import AssetClusterManager
from risk.maximum_intelligence_risk import MaximumIntelligenceRisk


class DynamicCapitalAllocator:
    """
    Dynamic Capital Allocation System
    Manages global risk budget and allocates capital to highest-ranked opportunities
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the Dynamic Capital Allocator"""
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
        self.opportunity_scanner = OpportunityScanner(config_path)
        self.asset_cluster_manager = AssetClusterManager(config_path)
        self.risk_manager = MaximumIntelligenceRisk(config=self.config)
        
        # Capital allocation settings
        self.total_portfolio_value = 100000  # Default $100k portfolio
        self.daily_risk_budget = 0.02  # 2% daily risk budget
        self.max_position_size = 0.05  # Maximum 5% per position
        self.max_cluster_exposure = 0.30  # Maximum 30% per cluster
        self.max_total_exposure = 0.80  # Maximum 80% total exposure
        
        # Current allocations
        self.current_positions = {}
        self.cluster_exposures = {}
        self.daily_pnl = 0.0
        self.daily_risk_used = 0.0
        
        # Performance tracking
        self.allocation_history = []
        self.risk_metrics = {}
        
        self.logger.info("🚀 Dynamic Capital Allocator initialized")
    
    def set_portfolio_value(self, value: float):
        """Set total portfolio value"""
        self.total_portfolio_value = value
        self.logger.info(f"💰 Portfolio value set to ${value:,.2f}")
    
    def set_risk_budget(self, daily_risk_percent: float):
        """Set daily risk budget as percentage"""
        self.daily_risk_budget = daily_risk_percent
        self.logger.info(f"🛡️ Daily risk budget set to {daily_risk_percent:.1%}")
    
    def calculate_available_capital(self) -> float:
        """Calculate available capital for new positions"""
        try:
            # Calculate current total exposure
            total_exposure = sum(pos['value'] for pos in self.current_positions.values())
            
            # Calculate maximum allowed exposure
            max_allowed_exposure = self.total_portfolio_value * self.max_total_exposure
            
            # Calculate remaining capital
            remaining_capital = max_allowed_exposure - total_exposure
            
            # Apply daily risk budget constraint
            daily_risk_capital = self.total_portfolio_value * self.daily_risk_budget
            available_risk_capital = daily_risk_capital - self.daily_risk_used
            
            # Return the smaller of the two constraints
            available_capital = min(remaining_capital, available_risk_capital)
            
            return max(available_capital, 0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating available capital: {e}")
            return 0
    
    def allocate_capital(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Allocate capital to highest-ranked opportunities"""
        try:
            available_capital = self.calculate_available_capital()
            
            if available_capital <= 0:
                self.logger.info("💰 No available capital for new positions")
                return {'allocations': [], 'total_allocated': 0}
            
            self.logger.info(f"💰 Allocating ${available_capital:,.2f} to opportunities")
            
            allocations = []
            total_allocated = 0
            
            # Sort opportunities by final score
            sorted_opportunities = sorted(opportunities, key=lambda x: x['final_score'], reverse=True)
            
            for opp in sorted_opportunities:
                if total_allocated >= available_capital:
                    break
                
                # Check cluster exposure limit
                cluster_name = opp['cluster']
                current_cluster_exposure = self.cluster_exposures.get(cluster_name, 0)
                max_cluster_capital = self.total_portfolio_value * self.max_cluster_exposure
                
                if current_cluster_exposure >= max_cluster_capital:
                    self.logger.debug(f"⚠️ Cluster {cluster_name} at exposure limit")
                    continue
                
                # Calculate position size
                position_size = self._calculate_position_size(opp, available_capital - total_allocated)
                
                if position_size > 0:
                    allocation = {
                        'pair': opp['pair'],
                        'cluster': cluster_name,
                        'position_size': position_size,
                        'position_value': position_size * self.total_portfolio_value,
                        'conviction_score': opp['conviction_score'],
                        'final_score': opp['final_score'],
                        'risk_reward_ratio': opp.get('risk_reward_ratio', 0),
                        'timestamp': datetime.now()
                    }
                    
                    allocations.append(allocation)
                    total_allocated += allocation['position_value']
                    
                    # Update cluster exposure
                    self.cluster_exposures[cluster_name] = current_cluster_exposure + allocation['position_value']
            
            # Update daily risk used
            self.daily_risk_used += total_allocated
            
            # Log allocation results
            self._log_allocation_results(allocations, total_allocated)
            
            # Store allocation history
            self.allocation_history.append({
                'timestamp': datetime.now(),
                'allocations': allocations,
                'total_allocated': total_allocated,
                'available_capital': available_capital
            })
            
            return {
                'allocations': allocations,
                'total_allocated': total_allocated,
                'available_capital': available_capital
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error allocating capital: {e}")
            return {'allocations': [], 'total_allocated': 0}
    
    def _calculate_position_size(self, opportunity: Dict[str, Any], available_capital: float) -> float:
        """Calculate optimal position size for an opportunity"""
        try:
            # Base position size from opportunity score
            base_size = opportunity.get('recommended_position_size', 0.01)
            
            # Adjust for available capital
            max_position_value = min(
                self.total_portfolio_value * base_size,
                available_capital,
                self.total_portfolio_value * self.max_position_size
            )
            
            # Adjust for cluster exposure
            cluster_name = opportunity['cluster']
            current_cluster_exposure = self.cluster_exposures.get(cluster_name, 0)
            max_cluster_capital = self.total_portfolio_value * self.max_cluster_exposure
            remaining_cluster_capital = max_cluster_capital - current_cluster_exposure
            
            max_position_value = min(max_position_value, remaining_cluster_capital)
            
            # Convert to percentage
            position_size = max_position_value / self.total_portfolio_value
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0
    
    def update_position(self, pair: str, action: str, value: float, price: float):
        """Update position (buy/sell)"""
        try:
            if action == 'buy':
                if pair in self.current_positions:
                    # Add to existing position
                    current_pos = self.current_positions[pair]
                    current_pos['value'] += value
                    current_pos['quantity'] += value / price
                    current_pos['avg_price'] = current_pos['value'] / current_pos['quantity']
                else:
                    # Create new position
                    self.current_positions[pair] = {
                        'value': value,
                        'quantity': value / price,
                        'avg_price': price,
                        'entry_time': datetime.now()
                    }
                
                self.logger.info(f"📈 Bought {pair}: ${value:,.2f} at ${price:.4f}")
                
            elif action == 'sell':
                if pair in self.current_positions:
                    # Reduce position
                    current_pos = self.current_positions[pair]
                    sold_quantity = value / price
                    
                    if sold_quantity >= current_pos['quantity']:
                        # Close position
                        pnl = (price - current_pos['avg_price']) * current_pos['quantity']
                        self.daily_pnl += pnl
                        
                        # Update cluster exposure
                        cluster_name = self.asset_cluster_manager.get_cluster_for_asset(pair)
                        if cluster_name:
                            self.cluster_exposures[cluster_name] = max(
                                0, self.cluster_exposures.get(cluster_name, 0) - current_pos['value']
                            )
                        
                        del self.current_positions[pair]
                        self.logger.info(f"📉 Sold {pair}: ${value:,.2f} at ${price:.4f} (PnL: ${pnl:,.2f})")
                    else:
                        # Partial sell
                        pnl = (price - current_pos['avg_price']) * sold_quantity
                        self.daily_pnl += pnl
                        
                        current_pos['quantity'] -= sold_quantity
                        current_pos['value'] = current_pos['quantity'] * current_pos['avg_price']
                        
                        self.logger.info(f"📉 Partial sell {pair}: ${value:,.2f} at ${price:.4f} (PnL: ${pnl:,.2f})")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating position: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        try:
            total_positions_value = sum(pos['value'] for pos in self.current_positions.values())
            total_positions_pnl = sum(
                (pos.get('current_price', pos['avg_price']) - pos['avg_price']) * pos['quantity']
                for pos in self.current_positions.values()
            )
            
            # Calculate cluster breakdown
            cluster_breakdown = {}
            for pair, pos in self.current_positions.items():
                cluster_name = self.asset_cluster_manager.get_cluster_for_asset(pair)
                if cluster_name:
                    if cluster_name not in cluster_breakdown:
                        cluster_breakdown[cluster_name] = 0
                    cluster_breakdown[cluster_name] += pos['value']
            
            return {
                'total_portfolio_value': self.total_portfolio_value,
                'total_positions_value': total_positions_value,
                'total_exposure_percent': total_positions_value / self.total_portfolio_value,
                'total_positions_pnl': total_positions_pnl,
                'daily_pnl': self.daily_pnl,
                'daily_risk_used': self.daily_risk_used,
                'daily_risk_budget': self.total_portfolio_value * self.daily_risk_budget,
                'available_capital': self.calculate_available_capital(),
                'cluster_breakdown': cluster_breakdown,
                'position_count': len(self.current_positions)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting portfolio summary: {e}")
            return {}
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        try:
            portfolio_summary = self.get_portfolio_summary()
            
            # Calculate risk metrics
            var_95 = self._calculate_var(0.95)
            max_drawdown = self._calculate_max_drawdown()
            
            risk_metrics = {
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'risk_reward_ratio': portfolio_summary.get('total_positions_pnl', 0) / max(var_95, 0.01),
                'exposure_utilization': portfolio_summary.get('total_exposure_percent', 0) / self.max_total_exposure,
                'daily_risk_utilization': self.daily_risk_used / (self.total_portfolio_value * self.daily_risk_budget)
            }
            
            self.risk_metrics = risk_metrics
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_var(self, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        try:
            # Simple VaR calculation based on position values and volatility
            total_exposure = sum(pos['value'] for pos in self.current_positions.values())
            avg_volatility = 0.02  # Assume 2% average volatility
            
            var = total_exposure * avg_volatility * 1.645  # 95% confidence
            return var
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating VaR: {e}")
            return 0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            # Simple max drawdown calculation
            if not self.allocation_history:
                return 0
            
            portfolio_values = []
            current_value = self.total_portfolio_value
            
            for allocation in self.allocation_history:
                current_value += allocation.get('total_allocated', 0)
                portfolio_values.append(current_value)
            
            if len(portfolio_values) < 2:
                return 0
            
            peak = portfolio_values[0]
            max_dd = 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            
            return max_dd
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating max drawdown: {e}")
            return 0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not self.allocation_history:
                return 0
            
            returns = []
            for i in range(1, len(self.allocation_history)):
                prev_value = self.allocation_history[i-1].get('total_allocated', 0)
                curr_value = self.allocation_history[i].get('total_allocated', 0)
                returns.append((curr_value - prev_value) / max(prev_value, 1))
            
            if not returns:
                return 0
            
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0
            
            sharpe = avg_return / std_return
            return sharpe
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating Sharpe ratio: {e}")
            return 0
    
    def _log_allocation_results(self, allocations: List[Dict[str, Any]], total_allocated: float):
        """Log allocation results"""
        try:
            self.logger.info("💰 CAPITAL ALLOCATION RESULTS:")
            self.logger.info("=" * 80)
            self.logger.info(f"Total allocated: ${total_allocated:,.2f}")
            self.logger.info(f"Number of positions: {len(allocations)}")
            self.logger.info("=" * 80)
            
            for i, alloc in enumerate(allocations, 1):
                self.logger.info(
                    f"{i}. {alloc['pair']} ({alloc['cluster']}) - "
                    f"${alloc['position_value']:,.2f} ({alloc['position_size']:.1%}) - "
                    f"Score: {alloc['final_score']:.3f}"
                )
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"❌ Error logging allocation results: {e}")
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of new day)"""
        self.daily_pnl = 0.0
        self.daily_risk_used = 0.0
        self.logger.info("🔄 Daily metrics reset")
    
    def export_allocation_history(self, filepath: str = None):
        """Export allocation history to JSON"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"allocations/allocation_history_{timestamp}.json"
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_summary': self.get_portfolio_summary(),
                'risk_metrics': self.get_risk_metrics(),
                'allocation_history': self.allocation_history,
                'current_positions': self.current_positions,
                'cluster_exposures': self.cluster_exposures
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Exported allocation history to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting allocation history: {e}") 