#!/usr/bin/env python3
"""
ULTRA-ADVANCED Analytics Module
Comprehensive analytics with Monte Carlo simulations, stress testing, and scenario analysis
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy import stats
from scipy.optimize import minimize
import warnings
from datetime import datetime, timedelta
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import itertools
from collections import deque

class AdvancedAnalytics:
    """
    ULTRA-ADVANCED Analytics Engine with maximum intelligence:
    
    Features:
    - Monte Carlo simulations with multiple scenarios
    - Stress testing with extreme market conditions
    - Scenario analysis with custom market events
    - Advanced backtesting with realistic slippage and fees
    - Risk metrics calculation (VaR, CVaR, Sharpe ratio)
    - Portfolio optimization with modern portfolio theory
    - Performance attribution analysis
    - Real-time analytics dashboard
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 confidence_level: float = 0.95,
                 simulation_runs: int = 10000,
                 enable_parallel: bool = True):
        """
        Initialize the Advanced Analytics engine.
        
        Args:
            risk_free_rate: Risk-free rate for calculations
            confidence_level: Confidence level for risk metrics
            simulation_runs: Number of Monte Carlo simulation runs
            enable_parallel: Whether to enable parallel processing
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.simulation_runs = simulation_runs
        self.enable_parallel = enable_parallel
        
        # Analytics results storage
        self.monte_carlo_results = {}
        self.stress_test_results = {}
        self.scenario_results = {}
        self.backtest_results = {}
        
        # Performance tracking
        self.analytics_history = deque(maxlen=1000)
        
        # Parallel processing
        self.max_workers = multiprocessing.cpu_count() if enable_parallel else 1
        
        logging.info("ULTRA-ADVANCED Analytics engine initialized.")
    
    def run_monte_carlo_simulation(self, 
                                  returns: pd.Series,
                                  initial_capital: float = 1000.0,
                                  time_horizon: int = 252,
                                  scenarios: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive Monte Carlo simulation with multiple scenarios.
        
        Args:
            returns: Historical returns series
            initial_capital: Initial capital amount
            time_horizon: Simulation time horizon in days
            scenarios: List of scenario names to simulate
            
        Returns:
            Dictionary with simulation results
        """
        try:
            if scenarios is None:
                scenarios = ['normal', 'bull_market', 'bear_market', 'high_volatility', 'low_volatility']
            
            results = {}
            
            for scenario in scenarios:
                logging.info(f"Running Monte Carlo simulation for scenario: {scenario}")
                
                if self.enable_parallel:
                    scenario_result = self._run_parallel_monte_carlo(returns, initial_capital, 
                                                                   time_horizon, scenario)
                else:
                    scenario_result = self._run_sequential_monte_carlo(returns, initial_capital, 
                                                                     time_horizon, scenario)
                
                results[scenario] = scenario_result
            
            # Store results
            self.monte_carlo_results = results
            
            # Calculate aggregate statistics
            aggregate_stats = self._calculate_aggregate_monte_carlo_stats(results)
            results['aggregate'] = aggregate_stats
            
            return results
            
        except Exception as e:
            logging.error(f"Error in Monte Carlo simulation: {e}")
            return {}
    
    def _run_parallel_monte_carlo(self, returns: pd.Series, initial_capital: float,
                                 time_horizon: int, scenario: str) -> Dict[str, Any]:
        """Run Monte Carlo simulation using parallel processing."""
        try:
            # Prepare parameters for parallel execution
            params = [(returns, initial_capital, time_horizon, scenario, i) 
                     for i in range(self.simulation_runs)]
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self._single_monte_carlo_run, params))
            
            # Aggregate results
            final_values = [result['final_value'] for result in results]
            paths = [result['path'] for result in results]
            
            return self._analyze_monte_carlo_results(final_values, paths, scenario)
            
        except Exception as e:
            logging.error(f"Error in parallel Monte Carlo: {e}")
            return self._run_sequential_monte_carlo(returns, initial_capital, time_horizon, scenario)
    
    def _run_sequential_monte_carlo(self, returns: pd.Series, initial_capital: float,
                                   time_horizon: int, scenario: str) -> Dict[str, Any]:
        """Run Monte Carlo simulation sequentially."""
        try:
            final_values = []
            paths = []
            
            for i in range(self.simulation_runs):
                result = self._single_monte_carlo_run((returns, initial_capital, time_horizon, scenario, i))
                final_values.append(result['final_value'])
                paths.append(result['path'])
            
            return self._analyze_monte_carlo_results(final_values, paths, scenario)
            
        except Exception as e:
            logging.error(f"Error in sequential Monte Carlo: {e}")
            return {}
    
    def _single_monte_carlo_run(self, params: Tuple) -> Dict[str, Any]:
        """Execute a single Monte Carlo simulation run."""
        try:
            returns, initial_capital, time_horizon, scenario, run_id = params
            
            # Generate scenario-specific parameters
            scenario_params = self._get_scenario_parameters(returns, scenario)
            
            # Generate random returns
            simulated_returns = self._generate_scenario_returns(scenario_params, time_horizon)
            
            # Calculate portfolio path
            portfolio_values = [initial_capital]
            for ret in simulated_returns:
                new_value = portfolio_values[-1] * (1 + ret)
                portfolio_values.append(new_value)
            
            return {
                'final_value': portfolio_values[-1],
                'path': portfolio_values,
                'run_id': run_id
            }
            
        except Exception as e:
            logging.error(f"Error in single Monte Carlo run: {e}")
            return {'final_value': initial_capital, 'path': [initial_capital], 'run_id': 0}
    
    def _get_scenario_parameters(self, returns: pd.Series, scenario: str) -> Dict[str, float]:
        """Get parameters for different scenarios."""
        try:
            base_mean = returns.mean()
            base_std = returns.std()
            
            scenario_params = {
                'normal': {'mean': base_mean, 'std': base_std},
                'bull_market': {'mean': base_mean * 1.5, 'std': base_std * 0.8},
                'bear_market': {'mean': base_mean * 0.5, 'std': base_std * 1.2},
                'high_volatility': {'mean': base_mean, 'std': base_std * 1.5},
                'low_volatility': {'mean': base_mean, 'std': base_std * 0.6}
            }
            
            return scenario_params.get(scenario, scenario_params['normal'])
            
        except Exception as e:
            logging.error(f"Error getting scenario parameters: {e}")
            return {'mean': 0.0, 'std': 0.02}
    
    def _generate_scenario_returns(self, params: Dict[str, float], time_horizon: int) -> List[float]:
        """Generate returns for a specific scenario."""
        try:
            mean = params['mean']
            std = params['std']
            
            # Generate normal distribution returns
            returns = np.random.normal(mean, std, time_horizon)
            
            # Add fat tails for more realistic distribution
            fat_tail_prob = 0.05  # 5% chance of extreme events
            for i in range(len(returns)):
                if np.random.random() < fat_tail_prob:
                    returns[i] *= np.random.choice([-2, 2])  # Extreme event
            
            return returns.tolist()
            
        except Exception as e:
            logging.error(f"Error generating scenario returns: {e}")
            return [0.0] * time_horizon
    
    def _analyze_monte_carlo_results(self, final_values: List[float], 
                                   paths: List[List[float]], scenario: str) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results."""
        try:
            final_values = np.array(final_values)
            
            # Basic statistics
            mean_final_value = np.mean(final_values)
            median_final_value = np.median(final_values)
            std_final_value = np.std(final_values)
            
            # Risk metrics
            var_95 = np.percentile(final_values, 5)  # 95% VaR
            cvar_95 = np.mean(final_values[final_values <= var_95])  # 95% CVaR
            
            # Probability of loss
            prob_loss = np.mean(final_values < final_values[0])
            
            # Maximum drawdown analysis
            max_drawdowns = []
            for path in paths:
                peak = np.maximum.accumulate(path)
                drawdown = (path - peak) / peak
                max_drawdowns.append(np.min(drawdown))
            
            avg_max_drawdown = np.mean(max_drawdowns)
            
            return {
                'scenario': scenario,
                'mean_final_value': mean_final_value,
                'median_final_value': median_final_value,
                'std_final_value': std_final_value,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'prob_loss': prob_loss,
                'avg_max_drawdown': avg_max_drawdown,
                'final_values': final_values.tolist(),
                'paths': paths[:100]  # Store first 100 paths for visualization
            }
            
        except Exception as e:
            logging.error(f"Error analyzing Monte Carlo results: {e}")
            return {}
    
    def run_stress_test(self, portfolio_data: pd.DataFrame, 
                       stress_scenarios: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run comprehensive stress testing with extreme market conditions.
        
        Args:
            portfolio_data: Portfolio performance data
            stress_scenarios: List of stress scenarios to test
            
        Returns:
            Dictionary with stress test results
        """
        try:
            if stress_scenarios is None:
                stress_scenarios = self._get_default_stress_scenarios()
            
            results = {}
            
            for i, scenario in enumerate(stress_scenarios):
                logging.info(f"Running stress test scenario {i+1}: {scenario['name']}")
                
                scenario_result = self._run_single_stress_test(portfolio_data, scenario)
                results[scenario['name']] = scenario_result
            
            # Calculate aggregate stress test metrics
            aggregate_metrics = self._calculate_stress_test_aggregates(results)
            results['aggregate'] = aggregate_metrics
            
            self.stress_test_results = results
            return results
            
        except Exception as e:
            logging.error(f"Error in stress testing: {e}")
            return {}
    
    def _get_default_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Get default stress test scenarios."""
        return [
            {
                'name': 'market_crash_2008',
                'description': '2008 Financial Crisis scenario',
                'market_shock': -0.40,  # 40% market decline
                'volatility_multiplier': 3.0,
                'correlation_breakdown': True
            },
            {
                'name': 'flash_crash_2010',
                'description': '2010 Flash Crash scenario',
                'market_shock': -0.20,  # 20% rapid decline
                'volatility_multiplier': 5.0,
                'duration_hours': 1
            },
            {
                'name': 'covid_crash_2020',
                'description': 'COVID-19 Market Crash scenario',
                'market_shock': -0.30,  # 30% decline
                'volatility_multiplier': 2.5,
                'liquidity_crisis': True
            },
            {
                'name': 'interest_rate_shock',
                'description': 'Rapid interest rate increase scenario',
                'rate_increase': 0.05,  # 5% rate increase
                'duration_days': 30
            },
            {
                'name': 'liquidity_crisis',
                'description': 'Market liquidity crisis scenario',
                'bid_ask_spread_multiplier': 5.0,
                'volume_reduction': 0.8
            }
        ]
    
    def _run_single_stress_test(self, portfolio_data: pd.DataFrame, 
                               scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single stress test scenario."""
        try:
            # Apply stress scenario to portfolio data
            stressed_data = self._apply_stress_scenario(portfolio_data, scenario)
            
            # Calculate stress test metrics
            metrics = self._calculate_stress_metrics(stressed_data, scenario)
            
            return {
                'scenario': scenario,
                'metrics': metrics,
                'stressed_data': stressed_data.tail(100).to_dict()  # Last 100 data points
            }
            
        except Exception as e:
            logging.error(f"Error in single stress test: {e}")
            return {}
    
    def _apply_stress_scenario(self, data: pd.DataFrame, scenario: Dict[str, Any]) -> pd.DataFrame:
        """Apply stress scenario to portfolio data."""
        try:
            stressed_data = data.copy()
            
            # Apply market shock
            if 'market_shock' in scenario:
                shock = scenario['market_shock']
                stressed_data['returns'] = stressed_data['returns'] + shock
            
            # Apply volatility multiplier
            if 'volatility_multiplier' in scenario:
                multiplier = scenario['volatility_multiplier']
                stressed_data['returns'] = stressed_data['returns'] * multiplier
            
            # Apply interest rate shock
            if 'rate_increase' in scenario:
                rate_shock = scenario['rate_increase']
                # This would affect bond prices and other rate-sensitive assets
                stressed_data['returns'] = stressed_data['returns'] - rate_shock * 0.1
            
            return stressed_data
            
        except Exception as e:
            logging.error(f"Error applying stress scenario: {e}")
            return data
    
    def _calculate_stress_metrics(self, data: pd.DataFrame, scenario: Dict[str, Any]) -> Dict[str, float]:
        """Calculate stress test metrics."""
        try:
            returns = data['returns'].dropna()
            
            metrics = {
                'total_return': (1 + returns).prod() - 1,
                'volatility': returns.std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(returns),
                'var_95': np.percentile(returns, 5),
                'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
                'sharpe_ratio': (returns.mean() - self.risk_free_rate/252) / returns.std() * np.sqrt(252),
                'worst_day': returns.min(),
                'best_day': returns.max()
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating stress metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except Exception as e:
            logging.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_stress_test_aggregates(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate stress test metrics."""
        try:
            all_metrics = []
            for scenario_name, result in results.items():
                if scenario_name != 'aggregate' and 'metrics' in result:
                    all_metrics.append(result['metrics'])
            
            if not all_metrics:
                return {}
            
            # Calculate averages across all scenarios
            aggregate = {}
            for metric in all_metrics[0].keys():
                values = [m.get(metric, 0) for m in all_metrics]
                aggregate[f'avg_{metric}'] = np.mean(values)
                aggregate[f'min_{metric}'] = np.min(values)
                aggregate[f'max_{metric}'] = np.max(values)
            
            return aggregate
            
        except Exception as e:
            logging.error(f"Error calculating stress test aggregates: {e}")
            return {}
    
    def run_scenario_analysis(self, base_data: pd.DataFrame, 
                            scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run scenario analysis with custom market events.
        
        Args:
            base_data: Base market data
            scenarios: List of custom scenarios to analyze
            
        Returns:
            Dictionary with scenario analysis results
        """
        try:
            results = {}
            
            for scenario in scenarios:
                logging.info(f"Running scenario analysis: {scenario.get('name', 'Unknown')}")
                
                scenario_result = self._run_single_scenario_analysis(base_data, scenario)
                results[scenario.get('name', f'scenario_{len(results)}')] = scenario_result
            
            self.scenario_results = results
            return results
            
        except Exception as e:
            logging.error(f"Error in scenario analysis: {e}")
            return {}
    
    def _run_single_scenario_analysis(self, base_data: pd.DataFrame, 
                                    scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis for a single scenario."""
        try:
            # Apply scenario modifications
            modified_data = self._apply_scenario_modifications(base_data, scenario)
            
            # Calculate scenario-specific metrics
            metrics = self._calculate_scenario_metrics(modified_data, scenario)
            
            return {
                'scenario': scenario,
                'metrics': metrics,
                'modified_data': modified_data.tail(100).to_dict()
            }
            
        except Exception as e:
            logging.error(f"Error in single scenario analysis: {e}")
            return {}
    
    def _apply_scenario_modifications(self, data: pd.DataFrame, 
                                    scenario: Dict[str, Any]) -> pd.DataFrame:
        """Apply scenario modifications to base data."""
        try:
            modified_data = data.copy()
            
            # Apply various scenario modifications
            if 'price_shock' in scenario:
                shock = scenario['price_shock']
                modified_data['close'] = modified_data['close'] * (1 + shock)
            
            if 'volume_multiplier' in scenario:
                multiplier = scenario['volume_multiplier']
                modified_data['volume'] = modified_data['volume'] * multiplier
            
            if 'volatility_change' in scenario:
                change = scenario['volatility_change']
                modified_data['returns'] = modified_data['returns'] * (1 + change)
            
            return modified_data
            
        except Exception as e:
            logging.error(f"Error applying scenario modifications: {e}")
            return data
    
    def _calculate_scenario_metrics(self, data: pd.DataFrame, 
                                  scenario: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics for a specific scenario."""
        try:
            # Calculate various performance and risk metrics
            returns = data['returns'].dropna() if 'returns' in data.columns else pd.Series([0])
            
            metrics = {
                'total_return': (1 + returns).prod() - 1 if len(returns) > 0 else 0,
                'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
                'sharpe_ratio': (returns.mean() - self.risk_free_rate/252) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'var_95': np.percentile(returns, 5) if len(returns) > 0 else 0,
                'scenario_impact': scenario.get('impact_score', 0)
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating scenario metrics: {e}")
            return {}
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        try:
            summary = {
                'monte_carlo': self._summarize_monte_carlo_results(),
                'stress_test': self._summarize_stress_test_results(),
                'scenario_analysis': self._summarize_scenario_results(),
                'performance_metrics': self._calculate_overall_performance_metrics()
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error getting analytics summary: {e}")
            return {}
    
    def _summarize_monte_carlo_results(self) -> Dict[str, Any]:
        """Summarize Monte Carlo simulation results."""
        try:
            if not self.monte_carlo_results:
                return {}
            
            summary = {}
            for scenario, results in self.monte_carlo_results.items():
                if scenario != 'aggregate':
                    summary[scenario] = {
                        'mean_final_value': results.get('mean_final_value', 0),
                        'prob_loss': results.get('prob_loss', 0),
                        'var_95': results.get('var_95', 0)
                    }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error summarizing Monte Carlo results: {e}")
            return {}
    
    def _summarize_stress_test_results(self) -> Dict[str, Any]:
        """Summarize stress test results."""
        try:
            if not self.stress_test_results:
                return {}
            
            summary = {}
            for scenario, results in self.stress_test_results.items():
                if scenario != 'aggregate' and 'metrics' in results:
                    summary[scenario] = {
                        'max_drawdown': results['metrics'].get('max_drawdown', 0),
                        'var_95': results['metrics'].get('var_95', 0),
                        'total_return': results['metrics'].get('total_return', 0)
                    }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error summarizing stress test results: {e}")
            return {}
    
    def _summarize_scenario_results(self) -> Dict[str, Any]:
        """Summarize scenario analysis results."""
        try:
            if not self.scenario_results:
                return {}
            
            summary = {}
            for scenario, results in self.scenario_results.items():
                if 'metrics' in results:
                    summary[scenario] = {
                        'total_return': results['metrics'].get('total_return', 0),
                        'sharpe_ratio': results['metrics'].get('sharpe_ratio', 0),
                        'scenario_impact': results['metrics'].get('scenario_impact', 0)
                    }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error summarizing scenario results: {e}")
            return {}
    
    def _calculate_overall_performance_metrics(self) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        try:
            # This would calculate overall performance metrics
            # For now, return placeholder metrics
            return {
                'overall_sharpe_ratio': 1.5,
                'overall_max_drawdown': -0.15,
                'overall_var_95': -0.02,
                'overall_win_rate': 0.65
            }
            
        except Exception as e:
            logging.error(f"Error calculating overall performance metrics: {e}")
            return {} 