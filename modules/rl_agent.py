import numpy as np
import os
import logging
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.envs import DummyVecEnv
    STABLE_BASELINES = True
except ImportError:
    STABLE_BASELINES = False
    logging.warning("stable-baselines3 not available, using custom PPO skeleton.")

class TradingEnv:
    """Custom trading environment for RL agent (compatible with stable-baselines3)."""
    def __init__(self, state_shape, action_space=3):
        self.state_shape = state_shape
        self.action_space = action_space
        self.current_step = 0
        self.state = np.zeros(state_shape)
        self.done = False
    def reset(self):
        self.current_step = 0
        self.state = np.zeros(self.state_shape)
        self.done = False
        return self.state
    def step(self, action):
        # Placeholder: reward = 0, done = False
        reward = 0.0
        self.current_step += 1
        if self.current_step > 1000:
            self.done = True
        return self.state, reward, self.done, {}
    def render(self, mode='human'):
        pass

class ReplayBuffer:
    """Replay buffer for RL experience tuples."""
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size=64):
        idx = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    def __len__(self):
        return len(self.buffer)

class RLAgent:
    """
    Advanced RL agent for trading with whale intelligence.
    Uses PPO (Proximal Policy Optimization) if available, otherwise a custom skeleton.
    Supports online and batch learning, state persistence, and whale feature integration.
    """
    def __init__(self, state_shape=20, action_space=3, model_path='models/rl_agent.zip', replay_buffer=None, reward_function=None, regime_policy=None):
        self.state_shape = state_shape
        self.action_space = action_space
        self.model_path = model_path
        self.model = None
        self.replay_buffer = replay_buffer or ReplayBuffer(max_size=10000)
        self.reward_function = reward_function  # Custom reward shaping function
        self.regime_policy = regime_policy or {}  # Dict of regime: RLAgent or reward_function
        if STABLE_BASELINES:
            self.env = DummyVecEnv([lambda: TradingEnv(state_shape, action_space)])
            if os.path.exists(model_path):
                self.model = PPO.load(model_path, env=self.env)
                logging.info("RL agent loaded from saved policy.")
            else:
                self.model = PPO('MlpPolicy', self.env, verbose=0)
                logging.info("RL agent initialized with new PPO policy.")
        else:
            self.model = None  # Custom PPO skeleton (not implemented)
    def get_action(self, state, predictions, market_analysis, whale_features):
        """
        Given the current state, model predictions, market analysis, and whale features,
        return an action: 0=hold, 1=buy, 2=sell, and optionally a position size adjustment.
        """
        # Flatten and concatenate all features for the RL state
        state_vec = self._build_state_vector(state, predictions, market_analysis, whale_features)
        if STABLE_BASELINES and self.model is not None:
            action, _ = self.model.predict(state_vec, deterministic=True)
            return int(action)
        else:
            # Fallback: simple rule-based action
            if whale_features.get('large_buy_volume', 0) > whale_features.get('large_sell_volume', 0):
                return 1  # buy
            elif whale_features.get('large_sell_volume', 0) > whale_features.get('large_buy_volume', 0):
                return 2  # sell
            else:
                return 0  # hold
    def train_on_batch(self, experiences=None):
        """
        Train the RL agent on a batch of experiences from the replay buffer.
        """
        if experiences is None:
            if len(self.replay_buffer) < 64:
                return  # Not enough data
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(64)
        else:
            states, actions, rewards, next_states, dones = experiences
        if STABLE_BASELINES and self.model is not None:
            # Not implemented: would require a custom callback or replay buffer integration
            pass
        else:
            pass  # Custom PPO skeleton (not implemented)
    def save(self):
        if STABLE_BASELINES and self.model is not None:
            self.model.save(self.model_path)
            logging.info(f"RL agent policy saved to {self.model_path}")
    def load(self):
        if STABLE_BASELINES and os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path, env=self.env)
            logging.info(f"RL agent policy loaded from {self.model_path}")
    def _build_state_vector(self, state, predictions, market_analysis, whale_features):
        """
        Build a flat state vector from all available features for RL input.
        """
        vec = []
        # Add last N prices/volumes if available
        if isinstance(state, dict):
            for k in sorted(state.keys()):
                try:
                    vec.append(float(state[k][-1]))
                except Exception:
                    vec.append(0.0)
        # Add predictions
        for tf in ['1m', '5m', '15m']:
            vec.append(float(predictions.get(tf, 0.0)))
        # Add market analysis
        for k in ['trend', 'volatility_value', 'rsi', 'adx']:
            v = market_analysis.get(k, 0.0)
            if isinstance(v, (int, float)):
                vec.append(float(v))
            else:
                vec.append(0.0)
        # Add whale features
        for k in [
            'large_trade_count', 'large_trade_volume', 'large_buy_count', 'large_sell_count',
            'large_buy_volume', 'large_sell_volume', 'whale_alert_count', 'whale_alert_flag',
            'order_book_imbalance', 'onchain_whale_inflow', 'onchain_whale_outflow']:
            vec.append(float(whale_features.get(k, 0.0)))
        # Pad or trim to state_shape
        if len(vec) < self.state_shape:
            vec += [0.0] * (self.state_shape - len(vec))
        elif len(vec) > self.state_shape:
            vec = vec[:self.state_shape]
        return np.array(vec).reshape(1, -1)
    def meta_learn(self, performance_history, regime=None):
        """
        Meta-learning: adjust learning rate or reward shaping based on performance or market regime.
        """
        if STABLE_BASELINES and self.model is not None:
            # Example: adjust learning rate if performance drops
            if len(performance_history) > 10:
                recent_perf = np.mean(performance_history[-10:])
                if recent_perf < 0.5:
                    new_lr = 1e-5
                    self.model.learning_rate = new_lr
                    logging.info(f"[Meta-Learning] Lowered RL agent learning rate to {new_lr}")
                else:
                    new_lr = 3e-4
                    self.model.learning_rate = new_lr
                    logging.info(f"[Meta-Learning] Restored RL agent learning rate to {new_lr}")
        else:
            logging.info("[Meta-Learning] Custom meta-learning not implemented.")
    def adversarial_train(self, adversarial_data=None):
        """
        Adversarial training: train RL agent against adversarial scenarios or regime shifts.
        """
        if STABLE_BASELINES and self.model is not None:
            # Example: train on adversarial data (not implemented)
            logging.info("[Adversarial] Adversarial training hook called (not implemented)")
        else:
            logging.info("[Adversarial] Custom adversarial training not implemented.")
    def compute_reward(self, state, action, reward, next_state, done, regime=None):
        """
        Compute custom reward using reward_function or regime-specific logic.
        """
        if regime and regime in self.regime_policy:
            # Use regime-specific reward function or RL agent
            policy = self.regime_policy[regime]
            if callable(policy):
                return policy(state, action, reward, next_state, done)
            # If policy is an RLAgent, could delegate (not implemented)
        if self.reward_function:
            return self.reward_function(state, action, reward, next_state, done)
        # Default: use raw reward
        return reward 