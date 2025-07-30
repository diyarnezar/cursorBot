"""
Mean-Variance Portfolio Optimization
Part of Project Hyperion - Ultimate Autonomous Trading Bot

Implements:
- Modern Portfolio Theory (MPT)
- Efficient Frontier calculation
- Risk-return optimization
- Multi-asset portfolio management
- Dynamic rebalancing
- Risk constraints and limits
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
from scipy.optimize import minimize
from scipy.stats import norm
import cvxpy as cp

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer:
    """
    Mean-Variance Portfolio Optimization using Modern Portfolio Theory
    
    Features:
    - Efficient frontier calculation
    - Risk-return optimization
    - Multi-asset portfolio management
    - Dynamic rebalancing
    - Risk constraints and limits
    - Transaction cost optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.portfolio_history = []
        self.optimization_results = {}
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.transaction_costs = config.get('transaction_costs', 0.001)
        
        # Optimization parameters
        self.max_position_size = config.get('max_position_size', 0.3)
        self.min_position_size = config.get('min_position_size', 0.01)
        self.target_volatility = config.get('target_volatility', 0.15)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.20)
        
        logger.info("Mean-Variance Optimizer initialized")

    def calculate_portfolio_metrics(self, returns: pd.DataFrame, 
                                  weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate portfolio risk and return metrics"""
        try:
            if weights is None:
                weights = np.ones(len(returns.columns)) / len(returns.columns)
            
            # Calculate expected returns
            expected_returns = returns.mean()
            portfolio_return = np.sum(expected_returns * weights)
            
            # Calculate covariance matrix
            cov_matrix = returns.cov()
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate Sharpe ratio
            excess_return = portfolio_return - self.risk_free_rate
            sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            portfolio_cumulative = np.sum(cumulative_returns * weights, axis=1)
            running_max = np.maximum.accumulate(portfolio_cumulative)
            drawdown = (portfolio_cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Calculate VaR and CVaR
            portfolio_returns = np.sum(returns * weights, axis=1)
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
            
            return {
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'weights': weights
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}

    def optimize_portfolio_max_sharpe(self, returns: pd.DataFrame, 
                                    current_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize portfolio to maximize Sharpe ratio"""
        try:
            logger.info("Optimizing portfolio to maximize Sharpe ratio")
            
            n_assets = len(returns.columns)
            
            # Initialize weights if not provided
            if current_weights is None:
                current_weights = np.ones(n_assets) / n_assets
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Define objective function (negative Sharpe ratio to minimize)
            def objective_function(weights):
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                excess_return = portfolio_return - self.risk_free_rate
                sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                return -sharpe_ratio  # Negative because we minimize
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Define bounds
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
            
            # Optimize
            result = minimize(
                objective_function,
                current_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimized_weights = result.x
                
                # Calculate optimized portfolio metrics
                portfolio_metrics = self.calculate_portfolio_metrics(returns, optimized_weights)
                
                result_dict = {
                    'optimization_method': 'max_sharpe',
                    'optimized_weights': dict(zip(returns.columns, optimized_weights)),
                    'portfolio_metrics': portfolio_metrics,
                    'optimization_success': True,
                    'optimization_message': result.message
                }
                
                logger.info(f"Max Sharpe optimization completed. Sharpe: {portfolio_metrics['sharpe_ratio']:.4f}")
                return result_dict
            else:
                logger.warning(f"Max Sharpe optimization failed: {result.message}")
                return {
                    'optimization_method': 'max_sharpe',
                    'optimized_weights': dict(zip(returns.columns, current_weights)),
                    'optimization_success': False,
                    'optimization_message': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in max Sharpe optimization: {e}")
            return {'optimization_success': False, 'error': str(e)}

    def optimize_portfolio_min_variance(self, returns: pd.DataFrame,
                                      current_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize portfolio to minimize variance"""
        try:
            logger.info("Optimizing portfolio to minimize variance")
            
            n_assets = len(returns.columns)
            
            # Initialize weights if not provided
            if current_weights is None:
                current_weights = np.ones(n_assets) / n_assets
            
            # Calculate covariance matrix
            cov_matrix = returns.cov()
            
            # Define objective function (portfolio variance)
            def objective_function(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                return portfolio_variance
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Define bounds
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
            
            # Optimize
            result = minimize(
                objective_function,
                current_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimized_weights = result.x
                
                # Calculate optimized portfolio metrics
                portfolio_metrics = self.calculate_portfolio_metrics(returns, optimized_weights)
                
                result_dict = {
                    'optimization_method': 'min_variance',
                    'optimized_weights': dict(zip(returns.columns, optimized_weights)),
                    'portfolio_metrics': portfolio_metrics,
                    'optimization_success': True,
                    'optimization_message': result.message
                }
                
                logger.info(f"Min variance optimization completed. Volatility: {portfolio_metrics['portfolio_volatility']:.4f}")
                return result_dict
            else:
                logger.warning(f"Min variance optimization failed: {result.message}")
                return {
                    'optimization_method': 'min_variance',
                    'optimized_weights': dict(zip(returns.columns, current_weights)),
                    'optimization_success': False,
                    'optimization_message': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in min variance optimization: {e}")
            return {'optimization_success': False, 'error': str(e)}

    def optimize_portfolio_target_return(self, returns: pd.DataFrame,
                                       target_return: float,
                                       current_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize portfolio for target return with minimum variance"""
        try:
            logger.info(f"Optimizing portfolio for target return: {target_return}")
            
            n_assets = len(returns.columns)
            
            # Initialize weights if not provided
            if current_weights is None:
                current_weights = np.ones(n_assets) / n_assets
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Define objective function (portfolio variance)
            def objective_function(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                return portfolio_variance
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) - target_return}  # Target return
            ]
            
            # Define bounds
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
            
            # Optimize
            result = minimize(
                objective_function,
                current_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimized_weights = result.x
                
                # Calculate optimized portfolio metrics
                portfolio_metrics = self.calculate_portfolio_metrics(returns, optimized_weights)
                
                result_dict = {
                    'optimization_method': 'target_return',
                    'target_return': target_return,
                    'optimized_weights': dict(zip(returns.columns, optimized_weights)),
                    'portfolio_metrics': portfolio_metrics,
                    'optimization_success': True,
                    'optimization_message': result.message
                }
                
                logger.info(f"Target return optimization completed. Return: {portfolio_metrics['portfolio_return']:.4f}")
                return result_dict
            else:
                logger.warning(f"Target return optimization failed: {result.message}")
                return {
                    'optimization_method': 'target_return',
                    'target_return': target_return,
                    'optimized_weights': dict(zip(returns.columns, current_weights)),
                    'optimization_success': False,
                    'optimization_message': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in target return optimization: {e}")
            return {'optimization_success': False, 'error': str(e)}

    def calculate_efficient_frontier(self, returns: pd.DataFrame, 
                                   n_points: int = 50) -> Dict[str, Any]:
        """Calculate efficient frontier"""
        try:
            logger.info("Calculating efficient frontier")
            
            expected_returns = returns.mean()
            cov_matrix = returns.cov()
            n_assets = len(returns.columns)
            
            # Calculate minimum and maximum returns
            min_return = expected_returns.min()
            max_return = expected_returns.max()
            
            # Generate target returns
            target_returns = np.linspace(min_return, max_return, n_points)
            
            efficient_portfolios = []
            
            for target_return in target_returns:
                try:
                    # Optimize for each target return
                    result = self.optimize_portfolio_target_return(returns, target_return)
                    
                    if result['optimization_success']:
                        efficient_portfolios.append({
                            'target_return': target_return,
                            'actual_return': result['portfolio_metrics']['portfolio_return'],
                            'volatility': result['portfolio_metrics']['portfolio_volatility'],
                            'sharpe_ratio': result['portfolio_metrics']['sharpe_ratio'],
                            'weights': result['optimized_weights']
                        })
                except Exception as e:
                    logger.warning(f"Failed to optimize for target return {target_return}: {e}")
                    continue
            
            # Sort by volatility
            efficient_portfolios.sort(key=lambda x: x['volatility'])
            
            result_dict = {
                'efficient_frontier': efficient_portfolios,
                'n_points': len(efficient_portfolios),
                'min_volatility': min(p['volatility'] for p in efficient_portfolios) if efficient_portfolios else 0,
                'max_sharpe': max(p['sharpe_ratio'] for p in efficient_portfolios) if efficient_portfolios else 0
            }
            
            logger.info(f"Efficient frontier calculated with {len(efficient_portfolios)} points")
            return result_dict
            
        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return {'efficient_frontier': [], 'error': str(e)}

    def optimize_with_transaction_costs(self, returns: pd.DataFrame,
                                      current_weights: np.ndarray,
                                      target_weights: np.ndarray) -> Dict[str, Any]:
        """Optimize portfolio considering transaction costs"""
        try:
            logger.info("Optimizing portfolio with transaction costs")
            
            n_assets = len(returns.columns)
            expected_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Calculate transaction costs
            weight_changes = np.abs(target_weights - current_weights)
            total_transaction_cost = np.sum(weight_changes) * self.transaction_costs
            
            # Define objective function (net return after transaction costs)
            def objective_function(weights):
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Net return after transaction costs
                net_return = portfolio_return - total_transaction_cost
                sharpe_ratio = (net_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
                
                return -sharpe_ratio  # Negative because we minimize
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Define bounds
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
            
            # Optimize
            result = minimize(
                objective_function,
                target_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimized_weights = result.x
                
                # Calculate metrics
                portfolio_metrics = self.calculate_portfolio_metrics(returns, optimized_weights)
                
                result_dict = {
                    'optimization_method': 'with_transaction_costs',
                    'optimized_weights': dict(zip(returns.columns, optimized_weights)),
                    'portfolio_metrics': portfolio_metrics,
                    'transaction_costs': total_transaction_cost,
                    'weight_changes': dict(zip(returns.columns, weight_changes)),
                    'optimization_success': True
                }
                
                logger.info(f"Transaction cost optimization completed. Net return: {portfolio_metrics['portfolio_return']:.4f}")
                return result_dict
            else:
                logger.warning(f"Transaction cost optimization failed: {result.message}")
                return {
                    'optimization_method': 'with_transaction_costs',
                    'optimized_weights': dict(zip(returns.columns, current_weights)),
                    'optimization_success': False,
                    'optimization_message': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in transaction cost optimization: {e}")
            return {'optimization_success': False, 'error': str(e)}

    def calculate_portfolio_risk_decomposition(self, returns: pd.DataFrame,
                                             weights: np.ndarray) -> Dict[str, Any]:
        """Calculate risk decomposition for portfolio"""
        try:
            logger.info("Calculating portfolio risk decomposition")
            
            cov_matrix = returns.cov()
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate marginal contribution to risk
            marginal_contribution = np.dot(cov_matrix, weights) / portfolio_volatility
            
            # Calculate percentage contribution to risk
            percentage_contribution = (weights * marginal_contribution) / portfolio_volatility
            
            # Calculate component VaR
            component_var = weights * marginal_contribution * norm.ppf(0.05)
            
            risk_decomposition = {
                'portfolio_volatility': portfolio_volatility,
                'marginal_contribution': dict(zip(returns.columns, marginal_contribution)),
                'percentage_contribution': dict(zip(returns.columns, percentage_contribution)),
                'component_var': dict(zip(returns.columns, component_var)),
                'weights': dict(zip(returns.columns, weights))
            }
            
            logger.info("Risk decomposition calculated")
            return risk_decomposition
            
        except Exception as e:
            logger.error(f"Error calculating risk decomposition: {e}")
            return {'error': str(e)}

    def rebalance_portfolio(self, returns: pd.DataFrame,
                          current_weights: np.ndarray,
                          rebalancing_threshold: float = 0.05) -> Dict[str, Any]:
        """Rebalance portfolio if weights deviate significantly"""
        try:
            logger.info("Checking portfolio rebalancing needs")
            
            # Optimize for current market conditions
            optimization_result = self.optimize_portfolio_max_sharpe(returns, current_weights)
            
            if not optimization_result['optimization_success']:
                return {
                    'rebalancing_needed': False,
                    'reason': 'Optimization failed',
                    'current_weights': dict(zip(returns.columns, current_weights))
                }
            
            target_weights = np.array(list(optimization_result['optimized_weights'].values()))
            
            # Calculate weight deviations
            weight_deviations = np.abs(target_weights - current_weights)
            max_deviation = np.max(weight_deviations)
            
            if max_deviation > rebalancing_threshold:
                logger.info(f"Rebalancing needed. Max deviation: {max_deviation:.4f}")
                
                # Optimize with transaction costs
                rebalancing_result = self.optimize_with_transaction_costs(
                    returns, current_weights, target_weights
                )
                
                return {
                    'rebalancing_needed': True,
                    'max_deviation': max_deviation,
                    'rebalancing_threshold': rebalancing_threshold,
                    'current_weights': dict(zip(returns.columns, current_weights)),
                    'target_weights': dict(zip(returns.columns, target_weights)),
                    'rebalancing_result': rebalancing_result
                }
            else:
                logger.info(f"No rebalancing needed. Max deviation: {max_deviation:.4f}")
                return {
                    'rebalancing_needed': False,
                    'max_deviation': max_deviation,
                    'rebalancing_threshold': rebalancing_threshold,
                    'current_weights': dict(zip(returns.columns, current_weights)),
                    'target_weights': dict(zip(returns.columns, target_weights))
                }
                
        except Exception as e:
            logger.error(f"Error in portfolio rebalancing: {e}")
            return {'rebalancing_needed': False, 'error': str(e)}

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio optimization summary"""
        try:
            if not self.portfolio_history:
                return {'total_optimizations': 0}
            
            recent_optimizations = self.portfolio_history[-10:]  # Last 10 optimizations
            
            return {
                'total_optimizations': len(self.portfolio_history),
                'recent_optimizations': len(recent_optimizations),
                'last_optimization': self.portfolio_history[-1]['timestamp'] if self.portfolio_history else None,
                'optimization_methods_used': list(set(opt['method'] for opt in self.portfolio_history)),
                'average_sharpe_ratio': np.mean([opt.get('sharpe_ratio', 0) for opt in recent_optimizations]),
                'average_volatility': np.mean([opt.get('volatility', 0) for opt in recent_optimizations])
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {'total_optimizations': 0}


# Example usage
if __name__ == "__main__":
    config = {
        'risk_free_rate': 0.02,
        'transaction_costs': 0.001,
        'max_position_size': 0.3,
        'min_position_size': 0.01,
        'target_volatility': 0.15,
        'max_drawdown_limit': 0.20
    }
    
    optimizer = MeanVarianceOptimizer(config)
    
    # Generate sample data
    np.random.seed(42)
    n_assets = 5
    n_periods = 1000
    
    returns_data = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * 0.02,
        columns=[f'Asset_{i+1}' for i in range(n_assets)]
    )
    
    # Test different optimization methods
    max_sharpe_result = optimizer.optimize_portfolio_max_sharpe(returns_data)
    min_variance_result = optimizer.optimize_portfolio_min_variance(returns_data)
    efficient_frontier = optimizer.calculate_efficient_frontier(returns_data)
    
    print("Max Sharpe Portfolio:", max_sharpe_result['portfolio_metrics']['sharpe_ratio'])
    print("Min Variance Portfolio:", min_variance_result['portfolio_metrics']['portfolio_volatility'])
    print("Efficient Frontier Points:", len(efficient_frontier['efficient_frontier'])) 