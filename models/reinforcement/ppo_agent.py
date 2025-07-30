"""
🤖 PPO Agent Module

This module implements a Proximal Policy Optimization (PPO) reinforcement
learning agent for cryptocurrency trading.

Author: Hyperion Trading System
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Configure logging
logger = logging.getLogger(__name__)

class PPOTrader:
    """
    🤖 PPO Trading Agent
    
    Implements a Proximal Policy Optimization agent for cryptocurrency trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PPO trading agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.actor_model = None
        self.critic_model = None
        self.optimizer = None
        self.memory = []
        self.episode_rewards = []
        self.training_history = []
        
        # PPO hyperparameters
        self.ppo_params = {
            'learning_rate': 0.0003,
            'gamma': 0.99,  # Discount factor
            'gae_lambda': 0.95,  # GAE lambda
            'clip_ratio': 0.2,  # PPO clip ratio
            'value_loss_coef': 0.5,  # Value loss coefficient
            'entropy_coef': 0.01,  # Entropy coefficient
            'max_grad_norm': 0.5,  # Maximum gradient norm
            'batch_size': 64,
            'epochs': 10,
            'buffer_size': 10000
        }
        
        # Trading parameters
        self.trading_params = {
            'initial_balance': 10000,
            'transaction_fee': 0.001,  # 0.1% fee
            'max_position_size': 0.5,  # 50% of balance
            'action_space': ['buy', 'sell', 'hold'],
            'state_features': ['price', 'volume', 'returns', 'volatility', 'momentum']
        }
        
        logger.info("🤖 PPO Trading Agent initialized")
    
    def create_actor_model(self, state_shape: Tuple[int, int]) -> Model:
        """Create the actor (policy) network."""
        try:
            # Input layer
            state_input = Input(shape=state_shape)
            
            # LSTM layer for temporal dependencies
            x = LSTM(64, return_sequences=True)(state_input)
            x = Dropout(0.2)(x)
            x = LSTM(32)(x)
            x = Dropout(0.2)(x)
            
            # Dense layers
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            x = Dense(32, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            # Output layer (action probabilities)
            action_probs = Dense(len(self.trading_params['action_space']), activation='softmax')(x)
            
            actor_model = Model(inputs=state_input, outputs=action_probs)
            
            optimizer = Adam(learning_rate=self.ppo_params['learning_rate'])
            actor_model.compile(optimizer=optimizer, loss='categorical_crossentropy')
            
            self.actor_model = actor_model
            logger.info("✅ Actor model created")
            
            return actor_model
            
        except Exception as e:
            logger.error(f"❌ Failed to create actor model: {e}")
            return None
    
    def create_critic_model(self, state_shape: Tuple[int, int]) -> Model:
        """Create the critic (value) network."""
        try:
            # Input layer
            state_input = Input(shape=state_shape)
            
            # LSTM layer for temporal dependencies
            x = LSTM(64, return_sequences=True)(state_input)
            x = Dropout(0.2)(x)
            x = LSTM(32)(x)
            x = Dropout(0.2)(x)
            
            # Dense layers
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            x = Dense(32, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            # Output layer (state value)
            state_value = Dense(1, activation='linear')(x)
            
            critic_model = Model(inputs=state_input, outputs=state_value)
            
            optimizer = Adam(learning_rate=self.ppo_params['learning_rate'])
            critic_model.compile(optimizer=optimizer, loss='mse')
            
            self.critic_model = critic_model
            logger.info("✅ Critic model created")
            
            return critic_model
            
        except Exception as e:
            logger.error(f"❌ Failed to create critic model: {e}")
            return None
    
    def create_models(self, state_shape: Tuple[int, int]):
        """Create both actor and critic models."""
        try:
            self.create_actor_model(state_shape)
            self.create_critic_model(state_shape)
            
            logger.info("✅ PPO models created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create PPO models: {e}")
    
    def get_state(self, data: pd.DataFrame, current_step: int, lookback: int = 20) -> np.ndarray:
        """Get the current state representation."""
        try:
            if current_step < lookback:
                # Pad with zeros if not enough history
                padding = np.zeros((lookback - current_step, len(self.trading_params['state_features'])))
                recent_data = data.iloc[:current_step + 1][self.trading_params['state_features']].values
                state = np.vstack([padding, recent_data])
            else:
                # Get recent data
                state = data.iloc[current_step - lookback + 1:current_step + 1][self.trading_params['state_features']].values
            
            # Normalize state
            state = (state - np.mean(state, axis=0)) / (np.std(state, axis=0) + 1e-8)
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Failed to get state: {e}")
            return np.zeros((lookback, len(self.trading_params['state_features'])))
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Select an action using the current policy."""
        try:
            # Get action probabilities
            state_batch = np.expand_dims(state, axis=0)
            action_probs = self.actor_model.predict(state_batch, verbose=0)[0]
            
            if training:
                # Sample action from probability distribution
                action = np.random.choice(len(action_probs), p=action_probs)
            else:
                # Select best action
                action = np.argmax(action_probs)
            
            # Get log probability
            log_prob = np.log(action_probs[action] + 1e-8)
            
            return action, log_prob
            
        except Exception as e:
            logger.error(f"❌ Failed to select action: {e}")
            return 0, 0.0
    
    def calculate_reward(self, action: int, current_price: float, next_price: float, 
                        current_balance: float, current_position: float) -> float:
        """Calculate the reward for the current action."""
        try:
            # Calculate price change
            price_change = (next_price - current_price) / current_price
            
            # Calculate reward based on action and outcome
            if action == 0:  # Buy
                if current_position == 0:  # No position
                    reward = price_change * current_balance * self.trading_params['max_position_size']
                else:
                    reward = 0  # Already have position
            elif action == 1:  # Sell
                if current_position > 0:  # Have position
                    reward = -price_change * current_position
                else:
                    reward = 0  # No position to sell
            else:  # Hold
                if current_position > 0:  # Have position
                    reward = price_change * current_position
                else:
                    reward = 0  # No position
            
            # Add transaction cost penalty
            if action in [0, 1]:  # Buy or sell
                reward -= self.trading_params['transaction_fee'] * current_balance * 0.1
            
            return reward
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate reward: {e}")
            return 0.0
    
    def update_balance_and_position(self, action: int, current_price: float, 
                                   current_balance: float, current_position: float) -> Tuple[float, float]:
        """Update balance and position based on action."""
        try:
            new_balance = current_balance
            new_position = current_position
            
            if action == 0:  # Buy
                if current_position == 0:  # No position
                    buy_amount = current_balance * self.trading_params['max_position_size']
                    new_position = buy_amount / current_price
                    new_balance -= buy_amount * (1 + self.trading_params['transaction_fee'])
            elif action == 1:  # Sell
                if current_position > 0:  # Have position
                    sell_amount = current_position * current_price
                    new_balance += sell_amount * (1 - self.trading_params['transaction_fee'])
                    new_position = 0
            
            return new_balance, new_position
            
        except Exception as e:
            logger.error(f"❌ Failed to update balance and position: {e}")
            return current_balance, current_position
    
    def collect_experience(self, data: pd.DataFrame, episodes: int = 10) -> List[Dict[str, Any]]:
        """Collect experience by running episodes."""
        try:
            all_experiences = []
            
            for episode in range(episodes):
                logger.info(f"🔄 Collecting experience for episode {episode + 1}/{episodes}")
                
                episode_experiences = []
                current_balance = self.trading_params['initial_balance']
                current_position = 0
                
                for step in range(len(data) - 1):
                    # Get current state
                    state = self.get_state(data, step)
                    
                    # Select action
                    action, log_prob = self.select_action(state, training=True)
                    
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
                    
                    # Store experience
                    experience = {
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'log_prob': log_prob,
                        'balance': current_balance,
                        'position': current_position,
                        'price': current_price
                    }
                    
                    episode_experiences.append(experience)
                    
                    # Update for next step
                    current_balance = new_balance
                    current_position = new_position
                
                # Calculate episode reward
                episode_reward = sum([exp['reward'] for exp in episode_experiences])
                self.episode_rewards.append(episode_reward)
                
                all_experiences.extend(episode_experiences)
                
                logger.info(f"✅ Episode {episode + 1} completed - Total reward: {episode_reward:.2f}")
            
            return all_experiences
            
        except Exception as e:
            logger.error(f"❌ Failed to collect experience: {e}")
            return []
    
    def calculate_advantages(self, experiences: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate advantages using GAE (Generalized Advantage Estimation)."""
        try:
            advantages = []
            returns = []
            
            # Calculate returns and advantages
            for i, exp in enumerate(experiences):
                # Calculate return (discounted sum of future rewards)
                future_rewards = [exp['reward'] for exp in experiences[i:]]
                return_val = sum([reward * (self.ppo_params['gamma'] ** j) 
                                for j, reward in enumerate(future_rewards)])
                returns.append(return_val)
                
                # Calculate advantage using GAE
                if i < len(experiences) - 1:
                    # Get next state value
                    next_state = experiences[i + 1]['state']
                    next_state_batch = np.expand_dims(next_state, axis=0)
                    next_value = self.critic_model.predict(next_state_batch, verbose=0)[0][0]
                else:
                    next_value = 0
                
                # Get current state value
                current_state_batch = np.expand_dims(exp['state'], axis=0)
                current_value = self.critic_model.predict(current_state_batch, verbose=0)[0][0]
                
                # Calculate TD error
                td_error = exp['reward'] + self.ppo_params['gamma'] * next_value - current_value
                
                # Calculate advantage using GAE
                if i == len(experiences) - 1:
                    advantage = td_error
                else:
                    advantage = td_error + self.ppo_params['gamma'] * self.ppo_params['gae_lambda'] * advantages[-1]
                
                advantages.append(advantage)
            
            # Normalize advantages
            advantages = np.array(advantages)
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            return advantages, np.array(returns)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate advantages: {e}")
            return np.array([]), np.array([])
    
    def train_ppo(self, experiences: List[Dict[str, Any]], epochs: int = None) -> Dict[str, Any]:
        """Train the PPO agent."""
        try:
            if epochs is None:
                epochs = self.ppo_params['epochs']
            
            # Calculate advantages and returns
            advantages, returns = self.calculate_advantages(experiences)
            
            if len(advantages) == 0:
                logger.error("❌ No advantages calculated")
                return {}
            
            # Prepare training data
            states = np.array([exp['state'] for exp in experiences])
            actions = np.array([exp['action'] for exp in experiences])
            old_log_probs = np.array([exp['log_prob'] for exp in experiences])
            
            # Convert actions to one-hot encoding
            action_one_hot = tf.keras.utils.to_categorical(actions, num_classes=len(self.trading_params['action_space']))
            
            # Training history
            training_history = {
                'actor_loss': [],
                'critic_loss': [],
                'total_loss': []
            }
            
            for epoch in range(epochs):
                logger.info(f"🏋️ Training epoch {epoch + 1}/{epochs}")
                
                # Train actor (policy)
                with tf.GradientTape() as tape:
                    # Get new action probabilities
                    new_log_probs = self.actor_model(states, training=True)
                    
                    # Calculate ratio
                    ratio = tf.exp(new_log_probs - old_log_probs)
                    
                    # Calculate clipped ratio
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.ppo_params['clip_ratio'], 
                                                   1 + self.ppo_params['clip_ratio'])
                    
                    # Calculate policy loss
                    policy_loss1 = ratio * advantages
                    policy_loss2 = clipped_ratio * advantages
                    policy_loss = -tf.reduce_mean(tf.minimum(policy_loss1, policy_loss2))
                    
                    # Add entropy bonus
                    entropy = -tf.reduce_mean(new_log_probs * tf.math.log(new_log_probs + 1e-8))
                    policy_loss -= self.ppo_params['entropy_coef'] * entropy
                
                # Apply gradients to actor
                actor_gradients = tape.gradient(policy_loss, self.actor_model.trainable_variables)
                tf.keras.optimizers.Adam(learning_rate=self.ppo_params['learning_rate']).apply_gradients(
                    zip(actor_gradients, self.actor_model.trainable_variables)
                )
                
                # Train critic (value)
                with tf.GradientTape() as tape:
                    value_pred = self.critic_model(states, training=True)
                    value_loss = tf.reduce_mean(tf.square(returns - value_pred))
                
                # Apply gradients to critic
                critic_gradients = tape.gradient(value_loss, self.critic_model.trainable_variables)
                tf.keras.optimizers.Adam(learning_rate=self.ppo_params['learning_rate']).apply_gradients(
                    zip(critic_gradients, self.critic_model.trainable_variables)
                )
                
                # Store training history
                total_loss = policy_loss + self.ppo_params['value_loss_coef'] * value_loss
                training_history['actor_loss'].append(float(policy_loss))
                training_history['critic_loss'].append(float(value_loss))
                training_history['total_loss'].append(float(total_loss))
                
                logger.info(f"✅ Epoch {epoch + 1} - Policy Loss: {float(policy_loss):.4f}, "
                          f"Value Loss: {float(value_loss):.4f}")
            
            self.training_history.append(training_history)
            logger.info("✅ PPO training completed")
            
            return training_history
            
        except Exception as e:
            logger.error(f"❌ Failed to train PPO: {e}")
            return {}
    
    def evaluate_agent(self, data: pd.DataFrame, episodes: int = 5) -> Dict[str, Any]:
        """Evaluate the trained agent."""
        try:
            evaluation_results = []
            
            for episode in range(episodes):
                current_balance = self.trading_params['initial_balance']
                current_position = 0
                episode_reward = 0
                trades = []
                
                for step in range(len(data) - 1):
                    # Get current state
                    state = self.get_state(data, step)
                    
                    # Select action (no exploration during evaluation)
                    action, _ = self.select_action(state, training=False)
                    
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
                    if action in [0, 1]:  # Buy or sell
                        trades.append({
                            'step': step,
                            'action': self.trading_params['action_space'][action],
                            'price': current_price,
                            'balance': current_balance,
                            'position': current_position
                        })
                    
                    episode_reward += reward
                    current_balance = new_balance
                    current_position = new_position
                
                # Calculate final portfolio value
                final_value = current_balance + current_position * data.iloc[-1]['close']
                total_return = (final_value - self.trading_params['initial_balance']) / self.trading_params['initial_balance']
                
                evaluation_results.append({
                    'episode': episode + 1,
                    'total_reward': episode_reward,
                    'final_balance': current_balance,
                    'final_position': current_position,
                    'final_value': final_value,
                    'total_return': total_return,
                    'num_trades': len(trades),
                    'trades': trades
                })
            
            # Calculate average metrics
            avg_reward = np.mean([result['total_reward'] for result in evaluation_results])
            avg_return = np.mean([result['total_return'] for result in evaluation_results])
            avg_trades = np.mean([result['num_trades'] for result in evaluation_results])
            
            evaluation_summary = {
                'avg_reward': avg_reward,
                'avg_return': avg_return,
                'avg_trades': avg_trades,
                'episode_results': evaluation_results
            }
            
            logger.info(f"✅ Agent evaluation completed - Avg Reward: {avg_reward:.2f}, "
                      f"Avg Return: {avg_return:.2%}, Avg Trades: {avg_trades:.1f}")
            
            return evaluation_summary
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate agent: {e}")
            return {}
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get a summary of the PPO agent."""
        return {
            'ppo_params': self.ppo_params,
            'trading_params': self.trading_params,
            'episode_rewards': self.episode_rewards,
            'training_history': self.training_history,
            'actor_model_created': self.actor_model is not None,
            'critic_model_created': self.critic_model is not None
        }
    
    def save_agent(self, filepath: str):
        """Save the PPO agent."""
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save models
            if self.actor_model:
                self.actor_model.save(f"{filepath}_actor.h5")
            if self.critic_model:
                self.critic_model.save(f"{filepath}_critic.h5")
            
            # Save agent data
            agent_data = {
                'ppo_params': self.ppo_params,
                'trading_params': self.trading_params,
                'episode_rewards': self.episode_rewards,
                'training_history': self.training_history
            }
            
            import pickle
            with open(f"{filepath}_data.pkl", 'wb') as f:
                pickle.dump(agent_data, f)
            
            logger.info(f"💾 PPO agent saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save PPO agent: {e}")
    
    def load_agent(self, filepath: str):
        """Load the PPO agent."""
        try:
            # Load models
            if os.path.exists(f"{filepath}_actor.h5"):
                self.actor_model = tf.keras.models.load_model(f"{filepath}_actor.h5")
            if os.path.exists(f"{filepath}_critic.h5"):
                self.critic_model = tf.keras.models.load_model(f"{filepath}_critic.h5")
            
            # Load agent data
            import pickle
            with open(f"{filepath}_data.pkl", 'rb') as f:
                agent_data = pickle.load(f)
            
            self.ppo_params = agent_data['ppo_params']
            self.trading_params = agent_data['trading_params']
            self.episode_rewards = agent_data['episode_rewards']
            self.training_history = agent_data['training_history']
            
            logger.info(f"📂 PPO agent loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load PPO agent: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'ppo_enabled': True,
        'training_episodes': 10,
        'evaluation_episodes': 5
    }
    
    # Initialize PPO agent
    ppo_agent = PPOTrader(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'close': np.random.uniform(1000, 2000, 1000),
        'volume': np.random.uniform(1000, 10000, 1000),
        'returns': np.random.normal(0, 0.02, 1000),
        'volatility': np.random.uniform(0.01, 0.05, 1000),
        'momentum': np.random.normal(0, 0.01, 1000)
    })
    
    # Create models
    state_shape = (20, 5)  # 20 timesteps, 5 features
    ppo_agent.create_models(state_shape)
    
    # Collect experience
    experiences = ppo_agent.collect_experience(sample_data, episodes=5)
    
    if len(experiences) > 0:
        # Train agent
        training_history = ppo_agent.train_ppo(experiences)
        
        # Evaluate agent
        evaluation_results = ppo_agent.evaluate_agent(sample_data, episodes=3)
        
        print(f"Training completed. Evaluation results: {evaluation_results}")
    
    # Get agent summary
    summary = ppo_agent.get_agent_summary()
    print(f"PPO agent created with {len(summary['episode_rewards'])} training episodes") 