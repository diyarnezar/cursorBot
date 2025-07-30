#!/usr/bin/env python3
"""
Test script to demonstrate pause/resume during optimization
"""

import time
import logging
import optuna
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import pause/resume controller
from modules.pause_resume_controller import setup_pause_resume, optimize_with_pause_support

def test_pause_during_optimization():
    """Test pause/resume functionality during optimization"""
    
    # Setup pause/resume controller
    controller = setup_pause_resume()
    controller.set_callbacks(
        on_pause=lambda: logger.info("⏸️ Training paused callback"),
        on_resume=lambda: logger.info("▶️ Training resumed callback"),
        on_checkpoint=lambda data: logger.info(f"💾 Checkpoint saved: {data}")
    )
    
    # Start monitoring
    controller.start_monitoring()
    
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randn(100))
    
    def objective(trial):
        """Objective function for optimization"""
        # Add a small delay to make trials more visible
        time.sleep(0.1)
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state': 42
        }
        
        try:
            model = RandomForestRegressor(**params)
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            return -scores.mean() if len(scores) > 0 else float('inf')
        except Exception:
            return float('inf')
    
    # Create study
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    
    logger.info("🚀 Starting optimization with pause support...")
    logger.info("💡 Try pressing Ctrl+P to pause during optimization!")
    logger.info("💡 Then press Ctrl+R to resume")
    
    # Use the new optimize_with_pause_support function
    start_time = time.time()
    study = optimize_with_pause_support(study, objective, n_trials=15, timeout=300)
    end_time = time.time()
    
    logger.info(f"🏁 Optimization completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Cleanup
    controller.cleanup()

if __name__ == "__main__":
    test_pause_during_optimization() 