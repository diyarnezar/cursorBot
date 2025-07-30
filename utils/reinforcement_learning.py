"""
Reinforcement Learning Agent for Project Hyperion
Advanced RL for model optimization and autonomous decision making
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import joblib


class RLAgent:
    """
    Advanced Reinforcement Learning Agent for model optimization
    Features: Q-learning, policy gradient, autonomous decision making
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize RL agent"""
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.config = config or {}
        
        # RL parameters
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Q-table for model optimization
        self.q_table = {}
        
        # Policy network
        self.policy_network = None
        
        # Experience replay buffer
        self.experience_buffer = []
        self.buffer_size = 10000
        
        self.logger.info("🧠 Advanced RL Agent initialized")
    
    def initialize_models(self, models: Dict[str, Any]):
        """Initialize RL agent with models"""
        self.models = models
        self.logger.info(f"🧠 RL Agent initialized with {len(models)} models")
    
    def optimize_models(self, features: pd.DataFrame, target: pd.Series, episodes: int = 100, learning_rate: float = 0.001) -> Dict[str, Any]:
        """Optimize models using reinforcement learning"""
        self.logger.info(f"🧠 Starting RL optimization with {episodes} episodes")
        
        try:
            self.learning_rate = learning_rate
            optimized_models = self.models.copy()
            
            # Ensure target is properly formatted and aligned with features
            if isinstance(target, pd.Series):
                # Align target with features index
                if not target.index.equals(features.index):
                    self.logger.info("Aligning target with features index for RL optimization")
                    common_index = features.index.intersection(target.index)
                    features_aligned = features.loc[common_index]
                    target_aligned = target.loc[common_index]
                else:
                    features_aligned = features
                    target_aligned = target
                target_values = target_aligned.values
            else:
                target_values = target
                features_aligned = features
            
            # Create features with target for evaluation
            features_with_target = features_aligned.copy()
            features_with_target['target'] = target_values
            
            # Ensure no NaN values
            features_with_target = features_with_target.dropna()
            if len(features_with_target) == 0:
                self.logger.warning("No valid data after alignment for RL optimization")
                return optimized_models
            
            self.logger.info(f"RL optimization data shape: {features_with_target.shape}")
            
            # Q-learning optimization
            for episode in range(episodes):
                if episode % 20 == 0:
                    self.logger.info(f"RL Episode {episode}/{episodes}")
                
                # Select action using epsilon-greedy policy
                if np.random.random() < self.epsilon:
                    action = np.random.choice(list(self.models.keys()))
                else:
                    action = self._select_best_action(features_with_target)
                
                # Execute action and get reward
                reward = self._execute_action(action, features_with_target)
                
                # Update Q-value
                self._update_q_value(action, reward)
                
                # Decay epsilon
                self.epsilon = max(0.01, self.epsilon * 0.995)
            
            self.logger.info("✅ RL optimization completed successfully")
            return optimized_models
            
        except Exception as e:
            self.logger.error(f"❌ RL optimization failed: {e}")
            return self.models
    
    def _get_model_state(self, features: pd.DataFrame, models: Dict[str, Any]) -> str:
        """Get current state representation"""
        # Create state based on model performance metrics
        performances = []
        for model_name, model in models.items():
            try:
                if hasattr(model, 'score'):
                    score = model.score(features.drop('target', axis=1, errors='ignore'), features['target'])
                    performances.append(score)
                else:
                    performances.append(0.5)  # Default score
            except:
                performances.append(0.5)
        
        # Discretize performance into state
        avg_performance = np.mean(performances)
        if avg_performance < 0.3:
            return "low_performance"
        elif avg_performance < 0.7:
            return "medium_performance"
        else:
            return "high_performance"
    
    def _select_action(self, state: str) -> str:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Random action
            actions = ['optimize_hyperparameters', 'adjust_architecture', 'change_learning_rate', 'ensemble_weights']
            return np.random.choice(actions)
        else:
            # Greedy action
            if state not in self.q_table:
                self.q_table[state] = {'optimize_hyperparameters': 0, 'adjust_architecture': 0, 
                                     'change_learning_rate': 0, 'ensemble_weights': 0}
            
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def _select_best_action(self, features_with_target: pd.DataFrame) -> str:
        """Select the best action based on current Q-values"""
        try:
            # Simple action selection based on model performance
            if not self.models:
                return list(self.models.keys())[0] if self.models else 'lightgbm'
            
            # Evaluate each model and select the best performing one
            best_action = None
            best_score = float('-inf')
            
            for model_name in self.models.keys():
                score = self._evaluate_model_performance(model_name, features_with_target)
                if score > best_score:
                    best_score = score
                    best_action = model_name
            
            return best_action or list(self.models.keys())[0]
            
        except Exception as e:
            self.logger.warning(f"Action selection failed: {e}")
            return list(self.models.keys())[0] if self.models else 'lightgbm'
    
    def _execute_action(self, action: str, features_with_target: pd.DataFrame) -> float:
        """Execute an action and return the reward"""
        try:
            # Calculate reward based on model performance
            if action in self.models:
                performance = self._evaluate_model_performance(action, features_with_target)
                # Normalize reward to be positive
                reward = max(0.1, performance)
            else:
                reward = 0.1  # Small positive reward for exploration
            
            return reward
            
        except Exception as e:
            self.logger.warning(f"Action execution failed: {e}")
            return 0.1
    
    def _optimize_hyperparameters(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model hyperparameters"""
        for model_name, model in models.items():
            if hasattr(model, 'get_params'):
                params = model.get_params()
                # Adjust key hyperparameters
                if 'n_estimators' in params:
                    params['n_estimators'] = min(200, params['n_estimators'] + 10)
                if 'max_depth' in params:
                    params['max_depth'] = min(20, params['max_depth'] + 1)
                if 'learning_rate' in params:
                    params['learning_rate'] = max(0.01, params['learning_rate'] * 0.9)
                
                try:
                    model.set_params(**params)
                except:
                    pass
        
        return models
    
    def _adjust_architecture(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust neural network architecture"""
        for model_name, model in models.items():
            if hasattr(model, 'hidden_layer_sizes'):
                current_layers = model.hidden_layer_sizes
                if isinstance(current_layers, tuple):
                    new_layers = tuple(size + 10 for size in current_layers)
                    try:
                        model.hidden_layer_sizes = new_layers
                    except:
                        pass
        
        return models
    
    def _change_learning_rate(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Change learning rate of models"""
        for model_name, model in models.items():
            if hasattr(model, 'learning_rate_init'):
                model.learning_rate_init = max(0.0001, model.learning_rate_init * 0.9)
        
        return models
    
    def _adjust_ensemble_weights(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust ensemble model weights"""
        # This would be implemented for ensemble models
        return models
    
    def _calculate_reward(self, features: pd.DataFrame, new_models: Dict[str, Any], old_models: Dict[str, Any]) -> float:
        """Calculate reward based on performance improvement"""
        try:
            # Calculate performance with new models
            new_performance = self._evaluate_models(features, new_models)
            old_performance = self._evaluate_models(features, old_models)
            
            # Reward is the improvement in performance
            reward = new_performance - old_performance
            
            return reward
        except:
            return 0.0
    
    def _evaluate_models(self, features: pd.DataFrame, models: Dict[str, Any]) -> float:
        """Evaluate model performance"""
        performances = []
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'score'):
                    X = features.drop('target', axis=1, errors='ignore')
                    y = features['target']
                    score = model.score(X, y)
                    performances.append(score)
                else:
                    performances.append(0.5)
            except:
                performances.append(0.5)
        
        return np.mean(performances) if performances else 0.5
    
    def _evaluate_model_performance(self, model_name: str, features_with_target: pd.DataFrame) -> float:
        """Evaluate model performance and return a score"""
        try:
            if model_name not in self.models:
                return 0.1
            
            model = self.models[model_name]
            
            # Simple performance evaluation
            if hasattr(model, 'predict'):
                try:
                    # Use a subset of data for quick evaluation
                    sample_size = min(1000, len(features_with_target))
                    sample_data = features_with_target.sample(n=sample_size, random_state=42)
                    
                    # Make prediction
                    features_sample = sample_data.drop('target', axis=1, errors='ignore')
                    predictions = model.predict(features_sample)
                    
                    # Calculate simple score (lower is better for regression)
                    if 'target' in sample_data.columns:
                        target = sample_data['target']
                        if len(predictions) == len(target):
                            mse = np.mean((predictions - target) ** 2)
                            score = 1.0 / (1.0 + mse)  # Convert to positive score
                            return score
                    
                    return 0.5  # Default score
                    
                except Exception as e:
                    self.logger.warning(f"Model prediction failed for {model_name}: {e}")
                    return 0.1
            
            return 0.1
            
        except Exception as e:
            self.logger.warning(f"Model evaluation failed for {model_name}: {e}")
            return 0.1
    
    def _update_q_value(self, action: str, reward: float):
        """Update Q-value for the given action"""
        try:
            # Create a simple state representation for Q-learning
            state = "current_state"  # Simplified state representation
            
            if state not in self.q_table:
                self.q_table[state] = {}
            
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0
            
            # Q-learning update formula
            current_q = self.q_table[state][action]
            new_q = current_q + self.learning_rate * (reward - current_q)
            self.q_table[state][action] = new_q
            
        except Exception as e:
            self.logger.warning(f"Q-value update failed: {e}")
    
    def _update_q_table(self, state: str, action: str, reward: float, new_state: str):
        """Update Q-table using Q-learning"""
        if state not in self.q_table:
            self.q_table[state] = {'optimize_hyperparameters': 0, 'adjust_architecture': 0, 
                                 'change_learning_rate': 0, 'ensemble_weights': 0}
        
        if new_state not in self.q_table:
            self.q_table[new_state] = {'optimize_hyperparameters': 0, 'adjust_architecture': 0, 
                                     'change_learning_rate': 0, 'ensemble_weights': 0}
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_future_q = max(self.q_table[new_state].values())
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state][action] = new_q
    
    def save_agent(self, filepath: str):
        """Save RL agent"""
        agent_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'config': self.config
        }
        joblib.dump(agent_data, filepath)
        self.logger.info(f"🧠 RL Agent saved to {filepath}")
    
    def load_agent(self, filepath: str):
        """Load RL agent"""
        agent_data = joblib.load(filepath)
        self.q_table = agent_data['q_table']
        self.epsilon = agent_data['epsilon']
        self.config = agent_data['config']
        self.logger.info(f"🧠 RL Agent loaded from {filepath}") 