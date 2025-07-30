"""
Risk Parity Portfolio Optimization
Part of Project Hyperion - Ultimate Autonomous Trading Bot

Implements:
- Risk parity portfolio construction
- Equal risk contribution optimization
- Risk budgeting and allocation
- Dynamic risk rebalancing
- Multi-asset risk parity
- Risk factor decomposition
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


class RiskParityOptimizer:
    """
    Risk Parity Portfolio Optimization
    
    Features:
    - Equal risk contribution optimization
    - Risk budgeting and allocation
    - Dynamic risk rebalancing
    - Multi-asset risk parity
    - Risk factor decomposition
    - Volatility targeting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.portfolio_history = []
        self.risk_allocations = {}
        self.optimization_results = {}
        
        # Risk parity parameters
        self.target_volatility = config.get('target_volatility', 0.10)
        self.risk_contribution_tolerance = config.get('risk_contribution_tolerance', 0.01)
        self.max_position_size = config.get('max_position_size', 0.4)
        self.min_position_size = config.get('min_position_size', 0.01)
        
        # Rebalancing parameters
        self.rebalancing_threshold = config.get('rebalancing_threshold', 0.05)
        self.transaction_costs = config.get('transaction_costs', 0.001)
        
        logger.info("Risk Parity Optimizer initialized")

    def calculate_risk_contributions(self, returns: pd.DataFrame, 
                                   weights: np.ndarray) -> Dict[str, Any]:
        """Calculate risk contributions for each asset"""
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov()
            
            # Calculate portfolio variance and volatility
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate marginal contribution to risk
            marginal_contribution = np.dot(cov_matrix, weights) / portfolio_volatility
            
            # Calculate risk contribution for each asset
            risk_contributions = weights * marginal_contribution
            
            # Calculate percentage risk contribution
            percentage_contributions = risk_contributions / portfolio_volatility
            
            # Calculate component VaR
            component_var = risk_contributions * norm.ppf(0.05)
            
            risk_analysis = {
                'portfolio_volatility': portfolio_volatility,
                'risk_contributions': dict(zip(returns.columns, risk_contributions)),
                'percentage_contributions': dict(zip(returns.columns, percentage_contributions)),
                'marginal_contributions': dict(zip(returns.columns, marginal_contribution)),
                'component_var': dict(zip(returns.columns, component_var)),
                'weights': dict(zip(returns.columns, weights))
            }
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error calculating risk contributions: {e}")
            return {'error': str(e)}

    def optimize_risk_parity(self, returns: pd.DataFrame,
                           current_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize portfolio for equal risk contribution"""
        try:
            logger.info("Optimizing risk parity portfolio")
            
            n_assets = len(returns.columns)
            
            # Initialize weights if not provided
            if current_weights is None:
                current_weights = np.ones(n_assets) / n_assets
            
            # Calculate covariance matrix
            cov_matrix = returns.cov()
            
            # Define objective function (risk contribution deviation)
            def objective_function(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Calculate risk contributions
                marginal_contribution = np.dot(cov_matrix, weights) / portfolio_volatility
                risk_contributions = weights * marginal_contribution
                
                # Calculate deviation from equal risk contribution
                target_contribution = portfolio_volatility / n_assets
                deviations = (risk_contributions - target_contribution) ** 2
                
                return np.sum(deviations)
            
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
                options={'maxiter': 2000}
            )
            
            if result.success:
                optimized_weights = result.x
                
                # Calculate risk analysis
                risk_analysis = self.calculate_risk_contributions(returns, optimized_weights)
                
                # Calculate portfolio metrics
                expected_returns = returns.mean()
                portfolio_return = np.sum(expected_returns * optimized_weights)
                sharpe_ratio = portfolio_return / risk_analysis['portfolio_volatility']
                
                result_dict = {
                    'optimization_method': 'risk_parity',
                    'optimized_weights': dict(zip(returns.columns, optimized_weights)),
                    'risk_analysis': risk_analysis,
                    'portfolio_return': portfolio_return,
                    'sharpe_ratio': sharpe_ratio,
                    'optimization_success': True,
                    'optimization_message': result.message
                }
                
                logger.info(f"Risk parity optimization completed. Volatility: {risk_analysis['portfolio_volatility']:.4f}")
                return result_dict
            else:
                logger.warning(f"Risk parity optimization failed: {result.message}")
                return {
                    'optimization_method': 'risk_parity',
                    'optimized_weights': dict(zip(returns.columns, current_weights)),
                    'optimization_success': False,
                    'optimization_message': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return {'optimization_success': False, 'error': str(e)}

    def optimize_risk_parity_target_volatility(self, returns: pd.DataFrame,
                                             target_volatility: float,
                                             current_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize risk parity portfolio with target volatility"""
        try:
            logger.info(f"Optimizing risk parity portfolio with target volatility: {target_volatility}")
            
            n_assets = len(returns.columns)
            
            # Initialize weights if not provided
            if current_weights is None:
                current_weights = np.ones(n_assets) / n_assets
            
            # Calculate covariance matrix
            cov_matrix = returns.cov()
            
            # Define objective function (risk contribution deviation)
            def objective_function(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Calculate risk contributions
                marginal_contribution = np.dot(cov_matrix, weights) / portfolio_volatility
                risk_contributions = weights * marginal_contribution
                
                # Calculate deviation from equal risk contribution
                target_contribution = portfolio_volatility / n_assets
                deviations = (risk_contributions - target_contribution) ** 2
                
                # Add volatility targeting penalty
                volatility_penalty = 1000 * (portfolio_volatility - target_volatility) ** 2
                
                return np.sum(deviations) + volatility_penalty
            
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
                options={'maxiter': 2000}
            )
            
            if result.success:
                optimized_weights = result.x
                
                # Calculate risk analysis
                risk_analysis = self.calculate_risk_contributions(returns, optimized_weights)
                
                # Calculate portfolio metrics
                expected_returns = returns.mean()
                portfolio_return = np.sum(expected_returns * optimized_weights)
                sharpe_ratio = portfolio_return / risk_analysis['portfolio_volatility']
                
                result_dict = {
                    'optimization_method': 'risk_parity_target_volatility',
                    'target_volatility': target_volatility,
                    'optimized_weights': dict(zip(returns.columns, optimized_weights)),
                    'risk_analysis': risk_analysis,
                    'portfolio_return': portfolio_return,
                    'sharpe_ratio': sharpe_ratio,
                    'volatility_deviation': abs(risk_analysis['portfolio_volatility'] - target_volatility),
                    'optimization_success': True,
                    'optimization_message': result.message
                }
                
                logger.info(f"Target volatility risk parity optimization completed. Volatility: {risk_analysis['portfolio_volatility']:.4f}")
                return result_dict
            else:
                logger.warning(f"Target volatility risk parity optimization failed: {result.message}")
                return {
                    'optimization_method': 'risk_parity_target_volatility',
                    'target_volatility': target_volatility,
                    'optimized_weights': dict(zip(returns.columns, current_weights)),
                    'optimization_success': False,
                    'optimization_message': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in target volatility risk parity optimization: {e}")
            return {'optimization_success': False, 'error': str(e)}

    def optimize_custom_risk_budget(self, returns: pd.DataFrame,
                                  risk_budget: Dict[str, float],
                                  current_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize portfolio with custom risk budget"""
        try:
            logger.info("Optimizing portfolio with custom risk budget")
            
            n_assets = len(returns.columns)
            
            # Initialize weights if not provided
            if current_weights is None:
                current_weights = np.ones(n_assets) / n_assets
            
            # Validate risk budget
            total_budget = sum(risk_budget.values())
            if abs(total_budget - 1.0) > 0.01:
                logger.warning(f"Risk budget does not sum to 1.0: {total_budget}")
                # Normalize risk budget
                risk_budget = {k: v / total_budget for k, v in risk_budget.items()}
            
            # Calculate covariance matrix
            cov_matrix = returns.cov()
            
            # Define objective function (risk contribution deviation from budget)
            def objective_function(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Calculate risk contributions
                marginal_contribution = np.dot(cov_matrix, weights) / portfolio_volatility
                risk_contributions = weights * marginal_contribution
                
                # Calculate percentage contributions
                percentage_contributions = risk_contributions / portfolio_volatility
                
                # Calculate deviation from target risk budget
                deviations = 0
                for i, asset in enumerate(returns.columns):
                    target_contribution = risk_budget.get(asset, 1.0 / n_assets)
                    actual_contribution = percentage_contributions[i]
                    deviations += (actual_contribution - target_contribution) ** 2
                
                return deviations
            
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
                options={'maxiter': 2000}
            )
            
            if result.success:
                optimized_weights = result.x
                
                # Calculate risk analysis
                risk_analysis = self.calculate_risk_contributions(returns, optimized_weights)
                
                # Calculate portfolio metrics
                expected_returns = returns.mean()
                portfolio_return = np.sum(expected_returns * optimized_weights)
                sharpe_ratio = portfolio_return / risk_analysis['portfolio_volatility']
                
                # Calculate budget deviation
                budget_deviations = {}
                for asset in returns.columns:
                    target_budget = risk_budget.get(asset, 1.0 / n_assets)
                    actual_budget = risk_analysis['percentage_contributions'][asset]
                    budget_deviations[asset] = actual_budget - target_budget
                
                result_dict = {
                    'optimization_method': 'custom_risk_budget',
                    'risk_budget': risk_budget,
                    'optimized_weights': dict(zip(returns.columns, optimized_weights)),
                    'risk_analysis': risk_analysis,
                    'portfolio_return': portfolio_return,
                    'sharpe_ratio': sharpe_ratio,
                    'budget_deviations': budget_deviations,
                    'optimization_success': True,
                    'optimization_message': result.message
                }
                
                logger.info(f"Custom risk budget optimization completed. Volatility: {risk_analysis['portfolio_volatility']:.4f}")
                return result_dict
            else:
                logger.warning(f"Custom risk budget optimization failed: {result.message}")
                return {
                    'optimization_method': 'custom_risk_budget',
                    'risk_budget': risk_budget,
                    'optimized_weights': dict(zip(returns.columns, current_weights)),
                    'optimization_success': False,
                    'optimization_message': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in custom risk budget optimization: {e}")
            return {'optimization_success': False, 'error': str(e)}

    def calculate_risk_factor_decomposition(self, returns: pd.DataFrame,
                                          weights: np.ndarray,
                                          factor_returns: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate risk factor decomposition"""
        try:
            logger.info("Calculating risk factor decomposition")
            
            # If no factor returns provided, use PCA-based factors
            if factor_returns is None:
                factor_returns = self._calculate_pca_factors(returns)
            
            # Calculate factor loadings
            factor_loadings = self._calculate_factor_loadings(returns, factor_returns)
            
            # Calculate factor contributions to portfolio risk
            portfolio_weights = weights.reshape(-1, 1)
            factor_exposures = np.dot(factor_loadings.T, portfolio_weights)
            
            # Calculate factor risk contributions
            factor_covariance = factor_returns.cov()
            factor_risk_contributions = np.dot(factor_covariance, factor_exposures)
            
            # Calculate total factor risk
            total_factor_risk = np.sqrt(np.dot(factor_exposures.T, np.dot(factor_covariance, factor_exposures)))[0, 0]
            
            # Calculate specific risk (residual)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
            specific_risk = np.sqrt(portfolio_volatility ** 2 - total_factor_risk ** 2)
            
            factor_decomposition = {
                'factor_exposures': dict(zip(factor_returns.columns, factor_exposures.flatten())),
                'factor_risk_contributions': dict(zip(factor_returns.columns, factor_risk_contributions.flatten())),
                'total_factor_risk': total_factor_risk,
                'specific_risk': specific_risk,
                'factor_risk_ratio': total_factor_risk / portfolio_volatility if portfolio_volatility > 0 else 0,
                'factor_loadings': factor_loadings
            }
            
            logger.info("Risk factor decomposition calculated")
            return factor_decomposition
            
        except Exception as e:
            logger.error(f"Error calculating risk factor decomposition: {e}")
            return {'error': str(e)}

    def _calculate_pca_factors(self, returns: pd.DataFrame, n_factors: int = 3) -> pd.DataFrame:
        """Calculate PCA-based factors"""
        try:
            from sklearn.decomposition import PCA
            
            # Standardize returns
            returns_std = (returns - returns.mean()) / returns.std()
            
            # Apply PCA
            pca = PCA(n_components=min(n_factors, len(returns.columns)))
            pca_factors = pca.fit_transform(returns_std)
            
            # Create factor returns DataFrame
            factor_returns = pd.DataFrame(
                pca_factors,
                index=returns.index,
                columns=[f'Factor_{i+1}' for i in range(pca_factors.shape[1])]
            )
            
            return factor_returns
            
        except Exception as e:
            logger.error(f"Error calculating PCA factors: {e}")
            return pd.DataFrame()

    def _calculate_factor_loadings(self, returns: pd.DataFrame, 
                                 factor_returns: pd.DataFrame) -> np.ndarray:
        """Calculate factor loadings using regression"""
        try:
            # Calculate factor loadings using linear regression
            factor_loadings = np.zeros((len(returns.columns), len(factor_returns.columns)))
            
            for i, asset in enumerate(returns.columns):
                # Regress asset returns on factor returns
                asset_returns = returns[asset]
                factor_matrix = factor_returns.values
                
                # Add constant term
                factor_matrix_with_const = np.column_stack([np.ones(len(factor_matrix)), factor_matrix])
                
                # Solve least squares
                loadings = np.linalg.lstsq(factor_matrix_with_const, asset_returns, rcond=None)[0]
                
                # Store factor loadings (excluding constant)
                factor_loadings[i, :] = loadings[1:]
            
            return factor_loadings
            
        except Exception as e:
            logger.error(f"Error calculating factor loadings: {e}")
            return np.zeros((len(returns.columns), len(factor_returns.columns)))

    def rebalance_risk_parity(self, returns: pd.DataFrame,
                            current_weights: np.ndarray,
                            rebalancing_threshold: float = None) -> Dict[str, Any]:
        """Rebalance risk parity portfolio if risk contributions deviate"""
        try:
            logger.info("Checking risk parity rebalancing needs")
            
            if rebalancing_threshold is None:
                rebalancing_threshold = self.rebalancing_threshold
            
            # Calculate current risk contributions
            current_risk_analysis = self.calculate_risk_contributions(returns, current_weights)
            
            # Check if risk contributions are balanced
            percentage_contributions = list(current_risk_analysis['percentage_contributions'].values())
            target_contribution = 1.0 / len(percentage_contributions)
            
            max_deviation = max(abs(contribution - target_contribution) for contribution in percentage_contributions)
            
            if max_deviation > rebalancing_threshold:
                logger.info(f"Risk parity rebalancing needed. Max deviation: {max_deviation:.4f}")
                
                # Optimize risk parity portfolio
                optimization_result = self.optimize_risk_parity(returns, current_weights)
                
                if optimization_result['optimization_success']:
                    # Calculate transaction costs
                    target_weights = np.array(list(optimization_result['optimized_weights'].values()))
                    weight_changes = np.abs(target_weights - current_weights)
                    total_transaction_cost = np.sum(weight_changes) * self.transaction_costs
                    
                    return {
                        'rebalancing_needed': True,
                        'max_deviation': max_deviation,
                        'rebalancing_threshold': rebalancing_threshold,
                        'current_weights': dict(zip(returns.columns, current_weights)),
                        'target_weights': optimization_result['optimized_weights'],
                        'transaction_costs': total_transaction_cost,
                        'optimization_result': optimization_result
                    }
                else:
                    return {
                        'rebalancing_needed': True,
                        'max_deviation': max_deviation,
                        'rebalancing_threshold': rebalancing_threshold,
                        'optimization_failed': True,
                        'current_weights': dict(zip(returns.columns, current_weights))
                    }
            else:
                logger.info(f"No risk parity rebalancing needed. Max deviation: {max_deviation:.4f}")
                return {
                    'rebalancing_needed': False,
                    'max_deviation': max_deviation,
                    'rebalancing_threshold': rebalancing_threshold,
                    'current_weights': dict(zip(returns.columns, current_weights))
                }
                
        except Exception as e:
            logger.error(f"Error in risk parity rebalancing: {e}")
            return {'rebalancing_needed': False, 'error': str(e)}

    def get_risk_parity_summary(self) -> Dict[str, Any]:
        """Get risk parity optimization summary"""
        try:
            if not self.portfolio_history:
                return {'total_optimizations': 0}
            
            recent_optimizations = self.portfolio_history[-10:]  # Last 10 optimizations
            
            return {
                'total_optimizations': len(self.portfolio_history),
                'recent_optimizations': len(recent_optimizations),
                'last_optimization': self.portfolio_history[-1]['timestamp'] if self.portfolio_history else None,
                'average_volatility': np.mean([opt.get('volatility', 0) for opt in recent_optimizations]),
                'average_sharpe_ratio': np.mean([opt.get('sharpe_ratio', 0) for opt in recent_optimizations]),
                'risk_contribution_balance': np.mean([opt.get('risk_balance_score', 0) for opt in recent_optimizations])
            }
            
        except Exception as e:
            logger.error(f"Error getting risk parity summary: {e}")
            return {'total_optimizations': 0}


# Example usage
if __name__ == "__main__":
    config = {
        'target_volatility': 0.10,
        'risk_contribution_tolerance': 0.01,
        'max_position_size': 0.4,
        'min_position_size': 0.01,
        'rebalancing_threshold': 0.05,
        'transaction_costs': 0.001
    }
    
    risk_parity_optimizer = RiskParityOptimizer(config)
    
    # Generate sample data
    np.random.seed(42)
    n_assets = 5
    n_periods = 1000
    
    returns_data = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * 0.02,
        columns=[f'Asset_{i+1}' for i in range(n_assets)]
    )
    
    # Test risk parity optimization
    risk_parity_result = risk_parity_optimizer.optimize_risk_parity(returns_data)
    target_vol_result = risk_parity_optimizer.optimize_risk_parity_target_volatility(returns_data, 0.08)
    
    # Test custom risk budget
    custom_budget = {
        'Asset_1': 0.3,
        'Asset_2': 0.2,
        'Asset_3': 0.2,
        'Asset_4': 0.15,
        'Asset_5': 0.15
    }
    custom_budget_result = risk_parity_optimizer.optimize_custom_risk_budget(returns_data, custom_budget)
    
    print("Risk Parity Portfolio:", risk_parity_result['risk_analysis']['portfolio_volatility'])
    print("Target Volatility Portfolio:", target_vol_result['risk_analysis']['portfolio_volatility'])
    print("Custom Budget Portfolio:", custom_budget_result['risk_analysis']['portfolio_volatility']) 