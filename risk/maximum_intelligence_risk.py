"""
🛡️ Maximum Intelligence Risk Management Module

This module implements maximum intelligence risk management with
advanced risk controls and adaptive risk strategies.

Author: Hyperion Trading System
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports for risk modeling
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

# Configure logging
logger = logging.getLogger(__name__)

class MaximumIntelligenceRisk:
    """
    🛡️ Maximum Intelligence Risk Management System
    
    Implements advanced risk management with maximum intelligence
    controls and adaptive risk strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the maximum intelligence risk management system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.risk_models = {}
        self.risk_metrics = {}
        self.risk_history = []
        self.risk_alerts = []
        self.adaptive_limits = {}
        
        # Risk parameters
        self.risk_params = {
            'max_position_size': 0.1,  # 10% of portfolio
            'max_drawdown': 0.15,  # 15% maximum drawdown
            'var_confidence': 0.95,  # 95% VaR confidence
            'max_correlation': 0.7,  # Maximum correlation between positions
            'volatility_target': 0.2,  # 20% annualized volatility target
            'kelly_fraction': 0.25,  # Kelly criterion fraction
            'risk_budget': 1.0,  # Total risk budget
            'max_leverage': 2.0,  # Maximum leverage
            'stop_loss': 0.05,  # 5% stop loss
            'take_profit': 0.15,  # 15% take profit
        }
        
        # Risk models configuration
        self.risk_model_configs = {
            'var_model': {
                'window': 252,  # 1 year
                'confidence_level': 0.95,
                'method': 'historical'
            },
            'volatility_model': {
                'window': 30,
                'method': 'ewm'
            },
            'correlation_model': {
                'window': 60,
                'method': 'rolling'
            },
            'drawdown_model': {
                'window': 252,
                'method': 'peak'
            }
        }
        
        logger.info("🛡️ Maximum Intelligence Risk Management initialized")
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = None) -> float:
        """Calculate Value at Risk (VaR)."""
        try:
            if confidence_level is None:
                confidence_level = self.risk_params['var_confidence']
            
            # Historical VaR
            var = np.percentile(returns, (1 - confidence_level) * 100)
            
            # Parametric VaR (assuming normal distribution)
            mean_return = returns.mean()
            std_return = returns.std()
            z_score = norm.ppf(1 - confidence_level)
            parametric_var = mean_return - z_score * std_return
            
            # Use the more conservative estimate
            var = min(var, parametric_var)
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate VaR: {e}")
            return 0.05  # Default 5% VaR
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = None) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        try:
            if confidence_level is None:
                confidence_level = self.risk_params['var_confidence']
            
            # Calculate VaR first
            var = self.calculate_var(returns, confidence_level)
            
            # Calculate CVaR (expected loss beyond VaR)
            tail_returns = returns[returns <= -var]
            if len(tail_returns) > 0:
                cvar = abs(tail_returns.mean())
            else:
                cvar = var
            
            return cvar
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate CVaR: {e}")
            return 0.08  # Default 8% CVaR
    
    def calculate_volatility(self, returns: pd.Series, window: int = None) -> float:
        """Calculate rolling volatility."""
        try:
            if window is None:
                window = self.risk_model_configs['volatility_model']['window']
            
            # Exponential weighted volatility
            volatility = returns.ewm(span=window).std().iloc[-1]
            
            # Annualize
            volatility *= np.sqrt(252)
            
            return volatility
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate volatility: {e}")
            return 0.2  # Default 20% volatility
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame, window: int = None) -> pd.DataFrame:
        """Calculate rolling correlation matrix."""
        try:
            if window is None:
                window = self.risk_model_configs['correlation_model']['window']
            
            # Rolling correlation
            correlation_matrix = returns_df.rolling(window=window).corr()
            
            # Get latest correlation matrix
            latest_corr = correlation_matrix.iloc[-len(returns_df.columns):, -len(returns_df.columns):]
            
            return latest_corr
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate correlation matrix: {e}")
            return pd.DataFrame()
    
    def calculate_max_drawdown(self, prices: pd.Series, window: int = None) -> float:
        """Calculate maximum drawdown."""
        try:
            if window is None:
                window = self.risk_model_configs['drawdown_model']['window']
            
            # Calculate rolling maximum
            rolling_max = prices.rolling(window=window).max()
            
            # Calculate drawdown
            drawdown = (prices - rolling_max) / rolling_max
            
            # Get maximum drawdown
            max_drawdown = abs(drawdown.min())
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate max drawdown: {e}")
            return 0.1  # Default 10% max drawdown
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion for position sizing."""
        try:
            if avg_loss == 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety factor
            kelly_fraction *= self.risk_params['kelly_fraction']
            
            # Ensure within bounds
            kelly_fraction = max(0, min(kelly_fraction, self.risk_params['max_position_size']))
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate Kelly criterion: {e}")
            return 0.05  # Default 5% position size
    
    def calculate_risk_parity_weights(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk parity weights."""
        try:
            # Calculate volatility for each asset
            volatilities = {}
            for column in returns_df.columns:
                volatilities[column] = self.calculate_volatility(returns_df[column])
            
            # Calculate inverse volatility weights
            total_inverse_vol = sum(1 / vol for vol in volatilities.values())
            weights = {asset: (1 / vol) / total_inverse_vol for asset, vol in volatilities.items()}
            
            return weights
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate risk parity weights: {e}")
            return {}
    
    def calculate_optimal_position_size(self, signal_strength: float, volatility: float, 
                                      correlation: float = 0.0) -> float:
        """Calculate optimal position size using multiple factors."""
        try:
            # Base position size from signal strength
            base_size = abs(signal_strength) * self.risk_params['max_position_size']
            
            # Volatility adjustment
            vol_adjustment = self.risk_params['volatility_target'] / (volatility + 1e-8)
            vol_adjustment = min(vol_adjustment, 2.0)  # Cap at 2x
            
            # Correlation adjustment
            corr_adjustment = 1 - abs(correlation) * 0.5
            
            # Risk budget adjustment
            risk_budget = self.risk_params['risk_budget']
            
            # Calculate final position size
            position_size = base_size * vol_adjustment * corr_adjustment * risk_budget
            
            # Apply limits
            position_size = min(position_size, self.risk_params['max_position_size'])
            position_size = max(position_size, 0.0)
            
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate optimal position size: {e}")
            return 0.05  # Default 5% position size
    
    def check_risk_limits(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if portfolio violates risk limits."""
        try:
            risk_violations = {}
            
            # Check position size limits
            total_exposure = sum(abs(pos['size']) for pos in portfolio_state.get('positions', []))
            if total_exposure > self.risk_params['max_leverage']:
                risk_violations['leverage_limit'] = {
                    'current': total_exposure,
                    'limit': self.risk_params['max_leverage'],
                    'violation': True
                }
            
            # Check drawdown limit
            current_drawdown = portfolio_state.get('current_drawdown', 0)
            if current_drawdown > self.risk_params['max_drawdown']:
                risk_violations['drawdown_limit'] = {
                    'current': current_drawdown,
                    'limit': self.risk_params['max_drawdown'],
                    'violation': True
                }
            
            # Check VaR limit
            portfolio_var = portfolio_state.get('var', 0)
            var_limit = self.risk_params['max_position_size'] * 0.1  # 1% VaR limit
            if portfolio_var > var_limit:
                risk_violations['var_limit'] = {
                    'current': portfolio_var,
                    'limit': var_limit,
                    'violation': True
                }
            
            # Check correlation limit
            max_correlation = portfolio_state.get('max_correlation', 0)
            if max_correlation > self.risk_params['max_correlation']:
                risk_violations['correlation_limit'] = {
                    'current': max_correlation,
                    'limit': self.risk_params['max_correlation'],
                    'violation': True
                }
            
            return risk_violations
            
        except Exception as e:
            logger.error(f"❌ Failed to check risk limits: {e}")
            return {}
    
    def generate_risk_alerts(self, risk_violations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk alerts based on violations."""
        try:
            alerts = []
            
            for limit_type, violation in risk_violations.items():
                if violation.get('violation', False):
                    alert = {
                        'timestamp': datetime.now(),
                        'type': 'risk_violation',
                        'limit_type': limit_type,
                        'current_value': violation['current'],
                        'limit_value': violation['limit'],
                        'severity': 'high' if violation['current'] > violation['limit'] * 1.5 else 'medium',
                        'message': f"Risk limit violated: {limit_type} - Current: {violation['current']:.4f}, Limit: {violation['limit']:.4f}"
                    }
                    alerts.append(alert)
            
            # Store alerts
            self.risk_alerts.extend(alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Failed to generate risk alerts: {e}")
            return []
    
    def calculate_adaptive_risk_limits(self, market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate adaptive risk limits based on market conditions."""
        try:
            adaptive_limits = {}
            
            # Volatility-based adjustments
            market_volatility = market_conditions.get('volatility', 0.2)
            vol_adjustment = self.risk_params['volatility_target'] / (market_volatility + 1e-8)
            vol_adjustment = min(vol_adjustment, 1.5)  # Cap at 1.5x
            
            # Market regime adjustments
            market_regime = market_conditions.get('regime', 'normal')
            regime_adjustment = {
                'bull': 1.2,
                'bear': 0.7,
                'crisis': 0.5,
                'normal': 1.0
            }.get(market_regime, 1.0)
            
            # Calculate adaptive limits
            adaptive_limits['max_position_size'] = self.risk_params['max_position_size'] * vol_adjustment * regime_adjustment
            adaptive_limits['max_drawdown'] = self.risk_params['max_drawdown'] / regime_adjustment
            adaptive_limits['stop_loss'] = self.risk_params['stop_loss'] * vol_adjustment
            adaptive_limits['take_profit'] = self.risk_params['take_profit'] * regime_adjustment
            
            # Store adaptive limits
            self.adaptive_limits = adaptive_limits
            
            return adaptive_limits
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate adaptive risk limits: {e}")
            return {}
    
    def create_risk_model(self, model_type: str, training_data: pd.DataFrame) -> Any:
        """Create risk prediction models."""
        try:
            if model_type == 'var_prediction':
                # VaR prediction model
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                
                # Prepare features and target
                features = training_data.drop(['var'], axis=1)
                target = training_data['var']
                
                model.fit(features, target)
                self.risk_models['var_prediction'] = model
                
            elif model_type == 'volatility_prediction':
                # Volatility prediction model
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                
                # Prepare features and target
                features = training_data.drop(['volatility'], axis=1)
                target = training_data['volatility']
                
                model.fit(features, target)
                self.risk_models['volatility_prediction'] = model
                
            elif model_type == 'drawdown_prediction':
                # Drawdown prediction model
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42
                )
                
                # Prepare features and target
                features = training_data.drop(['drawdown'], axis=1)
                target = training_data['drawdown']
                
                model.fit(features, target)
                self.risk_models['drawdown_prediction'] = model
            
            logger.info(f"✅ Created {model_type} risk model")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create {model_type} risk model: {e}")
            return None
    
    def predict_risk_metrics(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Predict risk metrics using trained models."""
        try:
            predictions = {}
            
            for model_name, model in self.risk_models.items():
                if model_name == 'var_prediction':
                    pred = model.predict(current_data)
                    predictions['predicted_var'] = pred[0]
                elif model_name == 'volatility_prediction':
                    pred = model.predict(current_data)
                    predictions['predicted_volatility'] = pred[0]
                elif model_name == 'drawdown_prediction':
                    pred = model.predict(current_data)
                    predictions['predicted_drawdown'] = pred[0]
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to predict risk metrics: {e}")
            return {}
    
    def update_risk_metrics(self, portfolio_data: Dict[str, Any], market_data: pd.DataFrame):
        """Update risk metrics with latest data."""
        try:
            # Calculate current risk metrics
            current_metrics = {}
            
            # Calculate portfolio VaR
            if 'returns' in portfolio_data:
                current_metrics['var'] = self.calculate_var(portfolio_data['returns'])
                current_metrics['cvar'] = self.calculate_cvar(portfolio_data['returns'])
            
            # Calculate portfolio volatility
            if 'returns' in portfolio_data:
                current_metrics['volatility'] = self.calculate_volatility(portfolio_data['returns'])
            
            # Calculate maximum drawdown
            if 'prices' in portfolio_data:
                current_metrics['max_drawdown'] = self.calculate_max_drawdown(portfolio_data['prices'])
            
            # Calculate correlation matrix
            if 'returns_matrix' in portfolio_data:
                corr_matrix = self.calculate_correlation_matrix(portfolio_data['returns_matrix'])
                if not corr_matrix.empty:
                    current_metrics['max_correlation'] = corr_matrix.abs().max().max()
            
            # Store metrics
            self.risk_metrics.update(current_metrics)
            
            # Add to history
            self.risk_history.append({
                'timestamp': datetime.now(),
                'metrics': current_metrics.copy()
            })
            
            logger.info("✅ Risk metrics updated")
            
        except Exception as e:
            logger.error(f"❌ Failed to update risk metrics: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get a summary of risk management activities."""
        return {
            'risk_params': self.risk_params,
            'risk_metrics': self.risk_metrics,
            'adaptive_limits': self.adaptive_limits,
            'risk_alerts': len(self.risk_alerts),
            'risk_history_length': len(self.risk_history),
            'risk_models': list(self.risk_models.keys())
        }
    
    def save_risk_state(self, filepath: str):
        """Save risk management state."""
        try:
            import pickle
            
            risk_state = {
                'risk_params': self.risk_params,
                'risk_metrics': self.risk_metrics,
                'adaptive_limits': self.adaptive_limits,
                'risk_alerts': self.risk_alerts,
                'risk_history': self.risk_history,
                'risk_models': self.risk_models
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(risk_state, f)
            
            logger.info(f"💾 Risk state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save risk state: {e}")
    
    def load_risk_state(self, filepath: str):
        """Load risk management state."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                risk_state = pickle.load(f)
            
            self.risk_params = risk_state['risk_params']
            self.risk_metrics = risk_state['risk_metrics']
            self.adaptive_limits = risk_state['adaptive_limits']
            self.risk_alerts = risk_state['risk_alerts']
            self.risk_history = risk_state['risk_history']
            self.risk_models = risk_state['risk_models']
            
            logger.info(f"📂 Risk state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load risk state: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'risk_management_enabled': True,
        'max_drawdown': 0.15,
        'var_confidence': 0.95
    }
    
    # Initialize risk management
    risk_manager = MaximumIntelligenceRisk(config)
    
    # Create sample data
    sample_returns = pd.Series(np.random.normal(0, 0.02, 1000))
    sample_prices = pd.Series(np.cumprod(1 + sample_returns))
    
    # Calculate risk metrics
    var = risk_manager.calculate_var(sample_returns)
    cvar = risk_manager.calculate_cvar(sample_returns)
    volatility = risk_manager.calculate_volatility(sample_returns)
    max_dd = risk_manager.calculate_max_drawdown(sample_prices)
    
    print(f"VaR: {var:.4f}")
    print(f"CVaR: {cvar:.4f}")
    print(f"Volatility: {volatility:.4f}")
    print(f"Max Drawdown: {max_dd:.4f}")
    
    # Get risk summary
    summary = risk_manager.get_risk_summary()
    print(f"Risk management initialized with {len(summary['risk_models'])} models") 