"""
DQN (Deep Q-Network) Agent for Advanced Cryptocurrency Trading
Part of Project Hyperion - Ultimate Autonomous Trading Bot
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


class DQNAgent:
    """
    Deep Q-Network Agent for Cryptocurrency Trading
    
    Features:
    - Experience Replay Buffer
    - Target Network
    - Epsilon-Greedy Exploration
    - Prioritized Experience Replay
    - Double DQN
    - Dueling DQN
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state_size = config.get('state_size', 20)
        self.action_size = config.get('action_size', 3)  # Buy, Sell, Hold
        self.memory = deque(maxlen=config.get('memory_size', 10000))
        self.gamma = config.get('gamma', 0.95)  # Discount factor
        self.epsilon = config.get('epsilon', 1.0)  # Exploration rate
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.update_target_freq = config.get('update_target_freq', 100)
        
        # Networks
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        
        # Training tracking
        self.training_step = 0
        self.episode_rewards = []
        self.episode_losses = []
        
        # DQN variants
        self.use_double_dqn = config.get('use_double_dqn', True)
        self.use_dueling_dqn = config.get('use_dueling_dqn', True)
        self.use_prioritized_replay = config.get('use_prioritized_replay', False)
        
        # Initialize networks
        self._build_networks()
        
        logger.info("DQN Agent initialized")

    def _build_networks(self):
        """Build Q-network and target network"""
        if self.use_dueling_dqn:
            self.q_network = self._build_dueling_network()
            self.target_network = self._build_dueling_network()
        else:
            self.q_network = self._build_standard_network()
            self.target_network = self._build_standard_network()
        
        # Compile Q-network
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.q_network.compile(optimizer=self.optimizer, loss='mse')
        
        # Initialize target network
        self._update_target_network()

    def _build_standard_network(self) -> Model:
        """Build standard DQN network"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        return model

    def _build_dueling_network(self) -> Model:
        """Build dueling DQN network with separate value and advantage streams"""
        inputs = Input(shape=(self.state_size,))
        
        # Shared layers
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Value stream
        value_stream = Dense(32, activation='relu')(x)
        value_stream = Dense(1, activation='linear')(value_stream)
        
        # Advantage stream
        advantage_stream = Dense(32, activation='relu')(x)
        advantage_stream = Dense(self.action_size, activation='linear')(advantage_stream)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        advantage_mean = tf.reduce_mean(advantage_stream, axis=1, keepdims=True)
        q_values = value_stream + (advantage_stream - advantage_mean)
        
        model = Model(inputs=inputs, outputs=q_values)
        
        return model

    def _build_lstm_network(self) -> Model:
        """Build LSTM-based DQN network for temporal dependencies"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.state_size, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        return model

    def get_state(self, data: pd.DataFrame, current_step: int, lookback: int = 20) -> np.ndarray:
        """Get current state representation"""
        if current_step < lookback:
            # Pad with zeros if not enough history
            padding = np.zeros((lookback - current_step, data.shape[1]))
            state_data = np.vstack([padding, data.iloc[:current_step].values])
        else:
            state_data = data.iloc[current_step-lookback:current_step].values
        
        # Flatten the state
        state = state_data.flatten()
        
        # Ensure state size matches
        if len(state) > self.state_size:
            state = state[:self.state_size]
        elif len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)))
        
        return state

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            # Random action
            action = random.randrange(self.action_size)
            confidence = 1.0 / self.action_size
        else:
            # Greedy action
            state_reshaped = state.reshape(1, -1)
            q_values = self.q_network.predict(state_reshaped, verbose=0)
            action = np.argmax(q_values[0])
            confidence = q_values[0][action]
        
        return action, confidence

    def calculate_reward(self, action: int, current_price: float, next_price: float, 
                        current_balance: float, current_position: float, 
                        transaction_cost: float = 0.001) -> float:
        """Calculate reward based on trading action and outcome"""
        price_change = (next_price - current_price) / current_price
        
        if action == 0:  # Hold
            reward = 0
        elif action == 1:  # Buy
            if current_position == 0:  # No position, can buy
                reward = price_change - transaction_cost
            else:  # Already have position
                reward = -transaction_cost  # Penalty for unnecessary action
        elif action == 2:  # Sell
            if current_position > 0:  # Have position, can sell
                reward = -price_change - transaction_cost
            else:  # No position
                reward = -transaction_cost  # Penalty for unnecessary action
        
        return reward

    def update_balance_and_position(self, action: int, current_price: float, 
                                  current_balance: float, current_position: float,
                                  transaction_cost: float = 0.001) -> Tuple[float, float]:
        """Update balance and position based on action"""
        new_balance = current_balance
        new_position = current_position
        
        if action == 1:  # Buy
            if current_position == 0 and current_balance > 0:
                # Buy with all available balance
                new_position = current_balance / current_price
                new_balance = 0
        elif action == 2:  # Sell
            if current_position > 0:
                # Sell all position
                new_balance = current_position * current_price * (1 - transaction_cost)
                new_position = 0
        
        return new_balance, new_position

    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool, priority: float = 1.0):
        """Store experience in replay buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'priority': priority
        }
        
        self.memory.append(experience)

    def replay(self) -> float:
        """Train the agent using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        if self.use_prioritized_replay:
            batch = self._sample_prioritized_batch()
        else:
            batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        
        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q values
        if self.use_double_dqn:
            # Double DQN: use Q-network for action selection, target network for evaluation
            next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
            next_q_values = self.target_network.predict(next_states, verbose=0)
            next_q_values = next_q_values[np.arange(self.batch_size), next_actions]
        else:
            # Standard DQN: use target network for both action selection and evaluation
            next_q_values = np.max(self.target_network.predict(next_states, verbose=0), axis=1)
        
        # Target Q values
        target_q_values = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * next_q_values[i]
        
        # Train the network
        history = self.q_network.fit(states, target_q_values, 
                                   batch_size=self.batch_size, 
                                   epochs=1, verbose=0)
        
        loss = history.history['loss'][0]
        self.episode_losses.append(loss)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.update_target_freq == 0:
            self._update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss

    def _sample_prioritized_batch(self) -> List[Dict[str, Any]]:
        """Sample batch using prioritized experience replay"""
        # Simple implementation - can be enhanced with proper priority sampling
        priorities = np.array([exp['priority'] for exp in self.memory])
        probabilities = priorities / np.sum(priorities)
        
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        batch = [self.memory[i] for i in indices]
        
        return batch

    def _update_target_network(self):
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
        logger.debug("Target network updated")

    def collect_experience(self, data: pd.DataFrame, episodes: int = 10, 
                         initial_balance: float = 10000.0) -> List[Dict[str, Any]]:
        """Collect experience through simulated trading episodes"""
        episode_results = []
        
        for episode in range(episodes):
            current_balance = initial_balance
            current_position = 0.0
            episode_reward = 0.0
            episode_actions = []
            
            for step in range(20, len(data) - 1):  # Start from step 20 to have enough history
                # Get current state
                current_state = self.get_state(data, step)
                
                # Select action
                action, confidence = self.select_action(current_state, training=True)
                
                # Get current and next prices
                current_price = data.iloc[step]['close']
                next_price = data.iloc[step + 1]['close']
                
                # Calculate reward
                reward = self.calculate_reward(action, current_price, next_price, 
                                             current_balance, current_position)
                
                # Update balance and position
                new_balance, new_position = self.update_balance_and_position(
                    action, current_price, current_balance, current_position
                )
                
                # Get next state
                next_state = self.get_state(data, step + 1)
                
                # Check if episode is done
                done = (step == len(data) - 2)
                
                # Store experience
                self.remember(current_state, action, reward, next_state, done)
                
                # Update tracking variables
                current_balance = new_balance
                current_position = new_position
                episode_reward += reward
                episode_actions.append(action)
                
                # Train the agent
                if len(self.memory) >= self.batch_size:
                    self.replay()
            
            # Episode results
            final_value = current_balance + (current_position * data.iloc[-1]['close'])
            episode_results.append({
                'episode': episode,
                'total_reward': episode_reward,
                'final_value': final_value,
                'return': (final_value - initial_balance) / initial_balance,
                'actions': episode_actions
            })
            
            self.episode_rewards.append(episode_reward)
            
            logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                       f"Final Value={final_value:.2f}, Return={((final_value - initial_balance) / initial_balance) * 100:.2f}%")
        
        return episode_results

    def evaluate_agent(self, data: pd.DataFrame, episodes: int = 5, 
                      initial_balance: float = 10000.0) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        evaluation_results = []
        
        for episode in range(episodes):
            current_balance = initial_balance
            current_position = 0.0
            episode_reward = 0.0
            trades = []
            
            for step in range(20, len(data) - 1):
                # Get current state
                current_state = self.get_state(data, step)
                
                # Select action (no exploration during evaluation)
                action, confidence = self.select_action(current_state, training=False)
                
                # Get current and next prices
                current_price = data.iloc[step]['close']
                next_price = data.iloc[step + 1]['close']
                
                # Calculate reward
                reward = self.calculate_reward(action, current_price, next_price, 
                                             current_balance, current_position)
                
                # Update balance and position
                new_balance, new_position = self.update_balance_and_position(
                    action, current_price, current_balance, current_position
                )
                
                # Record trade if action was taken
                if action != 0:  # Not hold
                    trades.append({
                        'step': step,
                        'action': action,
                        'price': current_price,
                        'confidence': confidence
                    })
                
                # Update tracking variables
                current_balance = new_balance
                current_position = new_position
                episode_reward += reward
            
            # Episode results
            final_value = current_balance + (current_position * data.iloc[-1]['close'])
            evaluation_results.append({
                'episode': episode,
                'total_reward': episode_reward,
                'final_value': final_value,
                'return': (final_value - initial_balance) / initial_balance,
                'num_trades': len(trades),
                'trades': trades
            })
        
        # Summary statistics
        returns = [result['return'] for result in evaluation_results]
        rewards = [result['total_reward'] for result in evaluation_results]
        
        summary = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'episode_results': evaluation_results
        }
        
        logger.info(f"Evaluation Summary: Mean Return={summary['mean_return']*100:.2f}%, "
                   f"Sharpe={summary['sharpe_ratio']:.2f}")
        
        return summary

    def save_model(self, filepath: str):
        """Save the trained model"""
        self.q_network.save(filepath)
        logger.info(f"DQN model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        self.q_network = tf.keras.models.load_model(filepath)
        self._update_target_network()
        logger.info(f"DQN model loaded from {filepath}")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'memory_size': len(self.memory),
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_episode_loss': np.mean(self.episode_losses) if self.episode_losses else 0,
            'total_episodes': len(self.episode_rewards)
        } 