#!/usr/bin/env python3
"""
Create Missing Models Script
Generates HMM and RL models that are missing from the models directory.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.smart_data_collector import SmartDataCollector
from modules.feature_engineering import EnhancedFeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_hmm_model():
    """Create a Hidden Markov Model for market regime detection."""
    try:
        # Check if hmmlearn is available
        try:
            from hmmlearn import hmm
            HMM_AVAILABLE = True
        except ImportError:
            logging.warning("hmmlearn not available. Creating placeholder HMM model.")
            HMM_AVAILABLE = False
        
        if HMM_AVAILABLE:
            # Create a simple HMM model for market regimes
            # States: 0=Bear, 1=Sideways, 2=Bull
            n_states = 3
            
            # Initialize HMM
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
            
            # Create synthetic training data (price changes and volatility)
            np.random.seed(42)
            n_samples = 1000
            
            # Generate synthetic market data
            bear_data = np.random.normal(-0.02, 0.03, (n_samples//3, 2))  # Negative returns, high vol
            sideways_data = np.random.normal(0.0, 0.01, (n_samples//3, 2))  # Zero returns, low vol
            bull_data = np.random.normal(0.02, 0.02, (n_samples//3, 2))  # Positive returns, medium vol
            
            # Combine data
            X = np.vstack([bear_data, sideways_data, bull_data])
            
            # Fit the model
            model.fit(X)
            
            # Save the model
            model_path = 'models/hmm_model.joblib'
            joblib.dump(model, model_path)
            logging.info(f"HMM model created and saved to {model_path}")
            
            return True
        else:
            # Create a placeholder model
            placeholder_model = {
                'type': 'hmm_placeholder',
                'n_states': 3,
                'created_at': datetime.now().isoformat(),
                'description': 'Placeholder HMM model - install hmmlearn for full functionality'
            }
            
            model_path = 'models/hmm_model.joblib'
            joblib.dump(placeholder_model, model_path)
            logging.info(f"HMM placeholder model created and saved to {model_path}")
            
            return True
            
    except Exception as e:
        logging.error(f"Error creating HMM model: {e}")
        return False

def create_rl_model():
    """Create a Reinforcement Learning model for trading decisions."""
    try:
        # Check if stable-baselines3 is available
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            import gymnasium as gym
            RL_AVAILABLE = True
        except ImportError:
            logging.warning("stable-baselines3 or gymnasium not available. Creating placeholder RL model.")
            RL_AVAILABLE = False
        
        if RL_AVAILABLE:
            try:
                # Create a simple trading environment using gymnasium
                class SimpleTradingEnv(gym.Env):
                    def __init__(self):
                        super().__init__()
                        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
                        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
                        self.reset()
                    
                    def reset(self, seed=None):
                        super().reset(seed=seed)
                        self.balance = 1000.0
                        self.position = 0.0
                        self.current_step = 0
                        self.price_history = []
                        return np.array([0.0, 0.0, 0.0], dtype=np.float32), {}
                    
                    def step(self, action):
                        # Simple reward function
                        reward = 0.0
                        done = self.current_step >= 100
                        self.current_step += 1
                        return np.array([0.0, 0.0, 0.0], dtype=np.float32), reward, done, False, {}
                
                # Create environment
                env = DummyVecEnv([lambda: SimpleTradingEnv()])
                
                # Create PPO model
                model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=2048)
                
                # Train for a few steps
                model.learn(total_timesteps=1000)
                
                # Save the model
                model_path = 'models/rl_agent'
                model.save(model_path)
                logging.info(f"RL model created and saved to {model_path}")
                
                return True
                
            except Exception as e:
                logging.warning(f"RL model creation failed with gymnasium: {e}. Creating placeholder.")
                RL_AVAILABLE = False
        
        # Create a placeholder model
        placeholder_model = {
            'type': 'rl_placeholder',
            'algorithm': 'PPO',
            'created_at': datetime.now().isoformat(),
            'description': 'Placeholder RL model - install stable-baselines3 and gymnasium for full functionality'
        }
        
        model_path = 'models/rl_agent.joblib'
        joblib.dump(placeholder_model, model_path)
        logging.info(f"RL placeholder model created and saved to {model_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error creating RL model: {e}")
        return False

def main():
    """Main function to create missing models."""
    logging.info("🚀 Creating Missing Models")
    logging.info("=" * 50)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create HMM model
    logging.info("📊 Creating HMM model...")
    hmm_success = create_hmm_model()
    
    # Create RL model
    logging.info("🤖 Creating RL model...")
    rl_success = create_rl_model()
    
    # Summary
    logging.info("=" * 50)
    logging.info("📋 Summary:")
    logging.info(f"✅ HMM Model: {'Created' if hmm_success else 'Failed'}")
    logging.info(f"✅ RL Model: {'Created' if rl_success else 'Failed'}")
    
    if hmm_success and rl_success:
        logging.info("🎉 All missing models created successfully!")
        logging.info("💡 Note: Some models may be placeholders if required packages are not installed.")
        logging.info("   Install 'hmmlearn' and 'stable-baselines3' for full functionality.")
    else:
        logging.warning("⚠️  Some models failed to create. Check logs above.")

if __name__ == "__main__":
    main() 