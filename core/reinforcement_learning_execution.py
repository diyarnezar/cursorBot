"""
🚀 PROJECT HYPERION - REINFORCEMENT LEARNING EXECUTION
====================================================

Reinforcement Learning agent for mastering Intelligent Execution Alchemist.
Learns optimal order placement strategies through trial and error.

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
import pickle
import time
import asyncio

from core.intelligent_execution import IntelligentExecutionAlchemist
from core.historical_data_warehouse import HistoricalDataWarehouse


class RLExecutionAgent:
    """
    Reinforcement Learning agent for execution optimization
    Masters the Intelligent Execution Alchemist through learning
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the RL Execution Agent"""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Initialize components
        self.execution_engine = IntelligentExecutionAlchemist(config_path)
        self.data_warehouse = HistoricalDataWarehouse(config_path)
        
        # RL settings
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # State and action spaces
        self.state_size = 20  # Order book features + market conditions
        self.action_size = 4   # Passive, Aggressive, Wait, Cancel/Reprice
        
        # Neural network model
        self.model = None
        self.target_model = None
        self._build_model()
        
        # Experience replay
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 32
        
        # Training state
        self.training_episodes = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Performance tracking
        self.execution_stats = {
            'total_episodes': 0,
            'avg_reward': 0.0,
            'best_reward': float('-inf'),
            'avg_fill_rate': 0.0,
            'avg_slippage': 0.0,
            'learning_progress': []
        }
        
        self.logger.info("🚀 RL Execution Agent initialized")
    
    def _build_model(self):
        """Build neural network model for Q-learning"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            # Main model
            self.model = Sequential([
                Dense(64, input_dim=self.state_size, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(self.action_size, activation='linear')
            ])
            
            self.model.compile(
                loss='mse',
                optimizer=Adam(learning_rate=self.learning_rate)
            )
            
            # Target model (for stable learning)
            self.target_model = Sequential([
                Dense(64, input_dim=self.state_size, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(self.action_size, activation='linear')
            ])
            
            self.target_model.compile(
                loss='mse',
                optimizer=Adam(learning_rate=self.learning_rate)
            )
            
            # Initialize target model weights
            self._update_target_model()
            
            self.logger.info("🧠 Neural network model built successfully")
            
        except ImportError:
            self.logger.warning("⚠️ TensorFlow not available, using simple Q-table")
            self.model = None
            self.target_model = None
            self.q_table = {}
    
    def _update_target_model(self):
        """Update target model weights"""
        if self.target_model is not None:
            self.target_model.set_weights(self.model.get_weights())
    
    def get_state(self, order_book_data: Dict[str, Any], market_conditions: Dict[str, Any]) -> np.ndarray:
        """Convert order book and market data to state vector"""
        try:
            state = []
            
            # Order book features (10 features)
            if 'metrics' in order_book_data:
                metrics = order_book_data['metrics']
                state.extend([
                    metrics.get('spread_percent', 0),
                    metrics.get('flow_imbalance', 0),
                    metrics.get('liquidity_ratio', 1.0),
                    metrics.get('bid_depth', 0) / 1000,  # Normalize
                    metrics.get('ask_depth', 0) / 1000,  # Normalize
                    metrics.get('bid_vwap', 0) / 100,    # Normalize
                    metrics.get('ask_vwap', 0) / 100,    # Normalize
                    len(order_book_data.get('bids', [])),
                    len(order_book_data.get('asks', [])),
                    metrics.get('best_bid', 0) / 100     # Normalize
                ])
            else:
                state.extend([0] * 10)
            
            # Market conditions (10 features)
            state.extend([
                market_conditions.get('volatility', 0.02),
                market_conditions.get('volume', 0) / 10000,  # Normalize
                market_conditions.get('price_momentum', 0),
                market_conditions.get('market_regime', 0.5),  # 0=low_vol, 0.5=normal, 1=high_vol
                market_conditions.get('time_of_day', 0.5),    # 0-1 normalized
                market_conditions.get('day_of_week', 0.5),    # 0-1 normalized
                market_conditions.get('spread_trend', 0),     # Spread widening/narrowing
                market_conditions.get('volume_trend', 0),     # Volume increasing/decreasing
                market_conditions.get('price_trend', 0),      # Price momentum
                market_conditions.get('liquidity_trend', 0)   # Liquidity changing
            ])
            
            return np.array(state, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"❌ Error creating state vector: {e}")
            return np.zeros(self.state_size, dtype=np.float32)
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        try:
            if training and np.random.random() < self.epsilon:
                # Exploration: random action
                return np.random.randint(0, self.action_size)
            else:
                # Exploitation: best action
                if self.model is not None:
                    q_values = self.model.predict(state.reshape(1, -1), verbose=0)
                    return np.argmax(q_values[0])
                else:
                    # Fallback to Q-table
                    state_key = tuple(state)
                    if state_key in self.q_table:
                        return np.argmax(self.q_table[state_key])
                    else:
                        return np.random.randint(0, self.action_size)
                        
        except Exception as e:
            self.logger.error(f"❌ Error choosing action: {e}")
            return 0
    
    def execute_action(self, action: int, symbol: str, side: str, quantity: float, 
                      confidence: float) -> Dict[str, Any]:
        """Execute the chosen action and get results"""
        try:
            action_map = {
                0: 'passive',
                1: 'aggressive', 
                2: 'wait',
                3: 'cancel_reprice'
            }
            
            action_name = action_map.get(action, 'passive')
            
            # Execute the action
            if action_name == 'passive':
                result = self.execution_engine.place_maker_order(
                    symbol, side, quantity, confidence * 0.8  # Reduce confidence for passive
                )
            elif action_name == 'aggressive':
                result = self.execution_engine.place_maker_order(
                    symbol, side, quantity, confidence * 1.2  # Increase confidence for aggressive
                )
            elif action_name == 'wait':
                result = {'status': 'wait', 'fill_rate': 0.0, 'slippage': 0.0}
            else:  # cancel_reprice
                result = {'status': 'cancel', 'fill_rate': 0.0, 'slippage': 0.0}
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error executing action: {e}")
            return {'status': 'error', 'fill_rate': 0.0, 'slippage': 0.0}
    
    def calculate_reward(self, action_result: Dict[str, Any], 
                        order_book_state: Dict[str, Any]) -> float:
        """Calculate reward based on execution results"""
        try:
            reward = 0.0
            
            # Base reward from fill rate
            fill_rate = action_result.get('fill_rate', 0.0)
            reward += fill_rate * 10  # High fill rate = high reward
            
            # Penalty for slippage
            slippage = action_result.get('slippage', 0.0)
            reward -= slippage * 100  # High slippage = high penalty
            
            # Bonus for efficient execution
            if fill_rate > 0.9 and slippage < 0.001:
                reward += 5  # Bonus for excellent execution
            
            # Penalty for waiting too long
            if action_result.get('status') == 'wait':
                reward -= 1  # Small penalty for waiting
            
            # Penalty for cancellation
            if action_result.get('status') == 'cancel':
                reward -= 2  # Penalty for cancellation
            
            # Market condition adjustments
            spread = order_book_state.get('spread_percent', 0.001)
            if spread > 0.005:  # Wide spread
                reward *= 0.8  # Reduce reward in difficult conditions
            
            return reward
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating reward: {e}")
            return 0.0
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        try:
            experience = (state, action, reward, next_state, done)
            self.memory.append(experience)
            
            # Maintain memory size
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)
                
        except Exception as e:
            self.logger.error(f"❌ Error storing experience: {e}")
    
    def replay(self):
        """Train the model on a batch of experiences"""
        try:
            if len(self.memory) < self.batch_size:
                return
            
            # Sample batch from memory
            batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
            
            states = np.array([self.memory[i][0] for i in batch])
            actions = np.array([self.memory[i][1] for i in batch])
            rewards = np.array([self.memory[i][2] for i in batch])
            next_states = np.array([self.memory[i][3] for i in batch])
            dones = np.array([self.memory[i][4] for i in batch])
            
            if self.model is not None:
                # Get current Q values
                current_q_values = self.model.predict(states, verbose=0)
                
                # Get next Q values from target model
                next_q_values = self.target_model.predict(next_states, verbose=0)
                
                # Calculate target Q values
                target_q_values = current_q_values.copy()
                
                for i in range(self.batch_size):
                    if dones[i]:
                        target_q_values[i][actions[i]] = rewards[i]
                    else:
                        target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * np.max(next_q_values[i])
                
                # Train the model
                self.model.fit(states, target_q_values, epochs=1, verbose=0)
                
            else:
                # Update Q-table
                for i in range(self.batch_size):
                    state_key = tuple(states[i])
                    if state_key not in self.q_table:
                        self.q_table[state_key] = np.zeros(self.action_size)
                    
                    if dones[i]:
                        self.q_table[state_key][actions[i]] = rewards[i]
                    else:
                        next_state_key = tuple(next_states[i])
                        if next_state_key not in self.q_table:
                            self.q_table[next_state_key] = np.zeros(self.action_size)
                        
                        self.q_table[state_key][actions[i]] = rewards[i] + self.discount_factor * np.max(self.q_table[next_state_key])
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
        except Exception as e:
            self.logger.error(f"❌ Error in replay training: {e}")
    
    async def train_episode(self, symbol: str, episode_length: int = 100) -> Dict[str, Any]:
        """Train the agent for one episode"""
        try:
            episode_reward = 0.0
            episode_fill_rate = 0.0
            episode_slippage = 0.0
            actions_taken = 0
            
            # Get initial state
            order_book = self.execution_engine.get_order_book_summary(symbol)
            market_conditions = self.execution_engine.get_market_conditions(symbol)
            
            if not order_book or not market_conditions:
                return {'reward': 0.0, 'success': False}
            
            state = self.get_state(order_book, market_conditions)
            
            for step in range(episode_length):
                try:
                    # Choose action
                    action = self.choose_action(state, training=True)
                    
                    # Execute action
                    side = 'buy' if np.random.random() < 0.5 else 'sell'
                    quantity = np.random.uniform(0.1, 2.0)
                    confidence = np.random.uniform(0.3, 0.9)
                    
                    action_result = self.execute_action(action, symbol, side, quantity, confidence)
                    
                    # Get new state
                    new_order_book = self.execution_engine.get_order_book_summary(symbol)
                    new_market_conditions = self.execution_engine.get_market_conditions(symbol)
                    
                    if new_order_book and new_market_conditions:
                        next_state = self.get_state(new_order_book, new_market_conditions)
                    else:
                        next_state = state
                    
                    # Calculate reward
                    reward = self.calculate_reward(action_result, order_book)
                    episode_reward += reward
                    
                    # Store experience
                    done = (step == episode_length - 1)
                    self.remember(state, action, reward, next_state, done)
                    
                    # Update statistics
                    episode_fill_rate += action_result.get('fill_rate', 0.0)
                    episode_slippage += action_result.get('slippage', 0.0)
                    actions_taken += 1
                    
                    # Update state
                    state = next_state
                    order_book = new_order_book
                    market_conditions = new_market_conditions
                    
                    # Train on batch
                    if len(self.memory) >= self.batch_size:
                        self.replay()
                    
                    # Small delay to simulate real-time execution
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in training step {step}: {e}")
                    continue
            
            # Update episode statistics
            self.training_episodes += 1
            self.total_reward += episode_reward
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Calculate averages
            avg_fill_rate = episode_fill_rate / actions_taken if actions_taken > 0 else 0.0
            avg_slippage = episode_slippage / actions_taken if actions_taken > 0 else 0.0
            
            # Update execution stats
            self.execution_stats['total_episodes'] = self.training_episodes
            self.execution_stats['avg_reward'] = self.total_reward / self.training_episodes
            self.execution_stats['avg_fill_rate'] = avg_fill_rate
            self.execution_stats['avg_slippage'] = avg_slippage
            
            if episode_reward > self.execution_stats['best_reward']:
                self.execution_stats['best_reward'] = episode_reward
            
            # Update target model periodically
            if self.training_episodes % 10 == 0:
                self._update_target_model()
            
            episode_result = {
                'episode': self.training_episodes,
                'reward': episode_reward,
                'avg_fill_rate': avg_fill_rate,
                'avg_slippage': avg_slippage,
                'epsilon': self.epsilon,
                'success': True
            }
            
            self.logger.info(f"🎯 Episode {self.training_episodes}: Reward={episode_reward:.2f}, "
                           f"Fill Rate={avg_fill_rate:.2%}, Slippage={avg_slippage:.4f}")
            
            return episode_result
            
        except Exception as e:
            self.logger.error(f"❌ Error in training episode: {e}")
            return {'reward': 0.0, 'success': False}
    
    async def train(self, symbols: List[str], episodes_per_symbol: int = 50):
        """Train the agent on multiple symbols"""
        try:
            self.logger.info(f"🚀 Starting RL training on {len(symbols)} symbols")
            
            total_episodes = len(symbols) * episodes_per_symbol
            
            for symbol in symbols:
                self.logger.info(f"🎯 Training on {symbol}")
                
                for episode in range(episodes_per_symbol):
                    episode_result = await self.train_episode(symbol)
                    
                    if not episode_result['success']:
                        self.logger.warning(f"⚠️ Episode failed for {symbol}")
                        continue
                    
                    # Log progress
                    if (episode + 1) % 10 == 0:
                        progress = (self.training_episodes / total_episodes) * 100
                        self.logger.info(f"📊 Training progress: {progress:.1f}%")
            
            # Save trained model
            self.save_model()
            
            self.logger.info("✅ RL training completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in training: {e}")
    
    def predict_optimal_action(self, symbol: str) -> Dict[str, Any]:
        """Predict optimal action for current market conditions"""
        try:
            # Get current state
            order_book = self.execution_engine.get_order_book_summary(symbol)
            market_conditions = self.execution_engine.get_market_conditions(symbol)
            
            if not order_book or not market_conditions:
                return {'action': 'passive', 'confidence': 0.5}
            
            state = self.get_state(order_book, market_conditions)
            
            # Choose action (no exploration during prediction)
            action = self.choose_action(state, training=False)
            
            action_map = {
                0: 'passive',
                1: 'aggressive',
                2: 'wait', 
                3: 'cancel_reprice'
            }
            
            # Get Q-values for confidence
            if self.model is not None:
                q_values = self.model.predict(state.reshape(1, -1), verbose=0)
                confidence = np.max(q_values[0]) / 10  # Normalize confidence
            else:
                state_key = tuple(state)
                if state_key in self.q_table:
                    confidence = np.max(self.q_table[state_key]) / 10
                else:
                    confidence = 0.5
            
            return {
                'action': action_map.get(action, 'passive'),
                'confidence': min(confidence, 1.0),
                'q_values': q_values[0].tolist() if self.model is not None else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error predicting optimal action: {e}")
            return {'action': 'passive', 'confidence': 0.5}
    
    def save_model(self, filepath: str = None):
        """Save the trained model"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"models/rl_execution_model_{timestamp}"
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            if self.model is not None:
                # Save neural network model
                self.model.save(f"{filepath}.h5")
                
                # Save training statistics
                training_data = {
                    'execution_stats': self.execution_stats,
                    'episode_rewards': self.episode_rewards,
                    'episode_lengths': self.episode_lengths,
                    'epsilon': self.epsilon,
                    'training_episodes': self.training_episodes
                }
                
                with open(f"{filepath}_stats.pkl", 'wb') as f:
                    pickle.dump(training_data, f)
                    
            else:
                # Save Q-table
                with open(f"{filepath}_qtable.pkl", 'wb') as f:
                    pickle.dump(self.q_table, f)
            
            self.logger.info(f"💾 RL model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            if filepath.endswith('.h5'):
                # Load neural network model
                from tensorflow.keras.models import load_model
                self.model = load_model(filepath)
                self.target_model = load_model(filepath)
                
                # Load training statistics
                stats_file = filepath.replace('.h5', '_stats.pkl')
                if Path(stats_file).exists():
                    with open(stats_file, 'rb') as f:
                        training_data = pickle.load(f)
                        self.execution_stats = training_data['execution_stats']
                        self.episode_rewards = training_data['episode_rewards']
                        self.episode_lengths = training_data['episode_lengths']
                        self.epsilon = training_data['epsilon']
                        self.training_episodes = training_data['training_episodes']
                        
            elif filepath.endswith('_qtable.pkl'):
                # Load Q-table
                with open(filepath, 'rb') as f:
                    self.q_table = pickle.load(f)
            
            self.logger.info(f"📂 RL model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return self.execution_stats.copy()
    
    def get_learning_curve(self) -> List[float]:
        """Get learning curve (episode rewards)"""
        return self.episode_rewards.copy() 