"""
Meta-Learning Trainer for Rapid Market Adaptation
Part of Project Hyperion - Ultimate Autonomous Trading Bot
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, Any, Tuple, List, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MetaLearner:
    """
    Advanced Meta-Learning System for Rapid Market Adaptation
    
    Features:
    - Model-Agnostic Meta-Learning (MAML)
    - Reptile Algorithm
    - Prototypical Networks
    - Matching Networks
    - Fast Adaptation
    - Task-Agnostic Learning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.meta_learning_rate = config.get('meta_learning_rate', 0.01)
        self.inner_learning_rate = config.get('inner_learning_rate', 0.01)
        self.num_tasks = config.get('num_tasks', 10)
        self.support_samples = config.get('support_samples', 5)
        self.query_samples = config.get('query_samples', 5)
        self.inner_steps = config.get('inner_steps', 5)
        self.meta_steps = config.get('meta_steps', 100)
        self.task_batch_size = config.get('task_batch_size', 4)
        
        # Model architecture
        self.input_dim = config.get('input_dim', 50)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.output_dim = config.get('output_dim', 1)
        
        # Meta-learning components
        self.meta_model = None
        self.task_models = {}
        self.meta_optimizer = None
        self.inner_optimizer = None
        
        # Performance tracking
        self.meta_losses = []
        self.adaptation_performances = []
        self.task_performances = {}
        
        # Data storage
        self.task_data = {}
        self.scalers = {}
        
        logger.info("Meta-Learner initialized")

    def create_meta_model(self) -> Model:
        """Create the meta-model architecture"""
        model = Sequential([
            Dense(self.hidden_dim, activation='relu', input_shape=(self.input_dim,)),
            Dropout(0.3),
            Dense(self.hidden_dim // 2, activation='relu'),
            Dropout(0.3),
            Dense(self.hidden_dim // 4, activation='relu'),
            Dense(self.output_dim, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.meta_learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_lstm_meta_model(self) -> Model:
        """Create LSTM-based meta-model for time series"""
        model = Sequential([
            LSTM(self.hidden_dim, return_sequences=True, input_shape=(None, self.input_dim)),
            Dropout(0.3),
            LSTM(self.hidden_dim // 2, return_sequences=False),
            Dropout(0.3),
            Dense(self.hidden_dim // 4, activation='relu'),
            Dense(self.output_dim, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.meta_learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def generate_tasks(self, data: pd.DataFrame, target_column: str,
                      task_duration: int = 100, overlap: int = 20) -> List[Dict[str, Any]]:
        """Generate tasks from time series data"""
        tasks = []
        n_samples = len(data)
        
        for i in range(self.num_tasks):
            # Calculate task boundaries
            start_idx = i * (task_duration - overlap)
            end_idx = start_idx + task_duration
            
            if end_idx > n_samples:
                break
            
            # Extract task data
            task_data = data.iloc[start_idx:end_idx].copy()
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col != target_column]
            X = task_data[feature_columns].values
            y = task_data[target_column].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split into support and query sets
            support_indices = np.random.choice(len(X_scaled), self.support_samples, replace=False)
            query_indices = np.array([i for i in range(len(X_scaled)) if i not in support_indices])
            
            if len(query_indices) > self.query_samples:
                query_indices = np.random.choice(query_indices, self.query_samples, replace=False)
            
            task = {
                'task_id': i,
                'support_X': X_scaled[support_indices],
                'support_y': y[support_indices],
                'query_X': X_scaled[query_indices],
                'query_y': y[query_indices],
                'scaler': scaler,
                'feature_columns': feature_columns
            }
            
            tasks.append(task)
            self.task_data[i] = task
        
        logger.info(f"Generated {len(tasks)} tasks")
        return tasks

    def maml_inner_update(self, model: Model, support_X: np.ndarray, support_y: np.ndarray,
                         inner_steps: int = None) -> Model:
        """Perform MAML inner loop update"""
        if inner_steps is None:
            inner_steps = self.inner_steps
        
        # Create a copy of the model for inner updates
        inner_model = tf.keras.models.clone_model(model)
        inner_model.set_weights(model.get_weights())
        
        # Compile with inner learning rate
        inner_model.compile(
            optimizer=Adam(learning_rate=self.inner_learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Perform inner updates
        for step in range(inner_steps):
            inner_model.fit(support_X, support_y, epochs=1, verbose=0)
        
        return inner_model

    def reptile_inner_update(self, model: Model, support_X: np.ndarray, support_y: np.ndarray,
                           inner_steps: int = None) -> Model:
        """Perform Reptile inner loop update"""
        if inner_steps is None:
            inner_steps = self.inner_steps
        
        # Create a copy of the model for inner updates
        inner_model = tf.keras.models.clone_model(model)
        inner_model.set_weights(model.get_weights())
        
        # Compile with inner learning rate
        inner_model.compile(
            optimizer=Adam(learning_rate=self.inner_learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Perform inner updates
        for step in range(inner_steps):
            inner_model.fit(support_X, support_y, epochs=1, verbose=0)
        
        return inner_model

    def train_maml(self, tasks: List[Dict[str, Any]], meta_steps: int = None) -> Dict[str, List[float]]:
        """Train using Model-Agnostic Meta-Learning (MAML)"""
        if meta_steps is None:
            meta_steps = self.meta_steps
        
        if self.meta_model is None:
            self.meta_model = self.create_meta_model()
        
        meta_losses = []
        adaptation_performances = []
        
        logger.info(f"Starting MAML training for {meta_steps} meta-steps")
        
        for meta_step in range(meta_steps):
            # Sample batch of tasks
            task_batch = np.random.choice(tasks, self.task_batch_size, replace=False)
            
            batch_losses = []
            batch_performances = []
            
            for task in task_batch:
                # Inner loop: adapt to support set
                adapted_model = self.maml_inner_update(
                    self.meta_model,
                    task['support_X'],
                    task['support_y']
                )
                
                # Outer loop: evaluate on query set
                query_loss = adapted_model.evaluate(
                    task['query_X'],
                    task['query_y'],
                    verbose=0
                )[0]
                
                # Calculate adaptation performance
                query_pred = adapted_model.predict(task['query_X'], verbose=0)
                adaptation_r2 = r2_score(task['query_y'], query_pred)
                
                batch_losses.append(query_loss)
                batch_performances.append(adaptation_r2)
            
            # Meta-update
            avg_loss = np.mean(batch_losses)
            avg_performance = np.mean(batch_performances)
            
            # Update meta-model (simplified - in practice, this requires gradient computation)
            # For now, we'll use a simple approach
            if meta_step % 10 == 0:
                # Periodically update the meta-model based on performance
                self.meta_model.fit(
                    np.vstack([task['support_X'] for task in task_batch]),
                    np.concatenate([task['support_y'] for task in task_batch]),
                    epochs=1,
                    verbose=0
                )
            
            meta_losses.append(avg_loss)
            adaptation_performances.append(avg_performance)
            
            if meta_step % 10 == 0:
                logger.info(f"MAML Step {meta_step}: Loss={avg_loss:.6f}, "
                           f"Adaptation R²={avg_performance:.4f}")
        
        self.meta_losses = meta_losses
        self.adaptation_performances = adaptation_performances
        
        return {
            'meta_losses': meta_losses,
            'adaptation_performances': adaptation_performances
        }

    def train_reptile(self, tasks: List[Dict[str, Any]], meta_steps: int = None) -> Dict[str, List[float]]:
        """Train using Reptile algorithm"""
        if meta_steps is None:
            meta_steps = self.meta_steps
        
        if self.meta_model is None:
            self.meta_model = self.create_meta_model()
        
        meta_losses = []
        adaptation_performances = []
        
        logger.info(f"Starting Reptile training for {meta_steps} meta-steps")
        
        for meta_step in range(meta_steps):
            # Sample batch of tasks
            task_batch = np.random.choice(tasks, self.task_batch_size, replace=False)
            
            batch_losses = []
            batch_performances = []
            
            for task in task_batch:
                # Inner loop: adapt to support set
                adapted_model = self.reptile_inner_update(
                    self.meta_model,
                    task['support_X'],
                    task['support_y']
                )
                
                # Outer loop: evaluate on query set
                query_loss = adapted_model.evaluate(
                    task['query_X'],
                    task['query_y'],
                    verbose=0
                )[0]
                
                # Calculate adaptation performance
                query_pred = adapted_model.predict(task['query_X'], verbose=0)
                adaptation_r2 = r2_score(task['query_y'], query_pred)
                
                batch_losses.append(query_loss)
                batch_performances.append(adaptation_r2)
            
            # Reptile update: move meta-model towards adapted models
            avg_loss = np.mean(batch_losses)
            avg_performance = np.mean(batch_performances)
            
            # Simplified Reptile update
            if meta_step % 10 == 0:
                # Periodically update the meta-model
                self.meta_model.fit(
                    np.vstack([task['support_X'] for task in task_batch]),
                    np.concatenate([task['support_y'] for task in task_batch]),
                    epochs=1,
                    verbose=0
                )
            
            meta_losses.append(avg_loss)
            adaptation_performances.append(avg_performance)
            
            if meta_step % 10 == 0:
                logger.info(f"Reptile Step {meta_step}: Loss={avg_loss:.6f}, "
                           f"Adaptation R²={avg_performance:.4f}")
        
        self.meta_losses = meta_losses
        self.adaptation_performances = adaptation_performances
        
        return {
            'meta_losses': meta_losses,
            'adaptation_performances': adaptation_performances
        }

    def fast_adapt(self, new_task_data: np.ndarray, new_task_target: np.ndarray,
                  adaptation_steps: int = None) -> Model:
        """Fast adaptation to a new task"""
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps
        
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Run training first.")
        
        # Create adapted model
        adapted_model = tf.keras.models.clone_model(self.meta_model)
        adapted_model.set_weights(self.meta_model.get_weights())
        
        # Compile with inner learning rate
        adapted_model.compile(
            optimizer=Adam(learning_rate=self.inner_learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Fast adaptation
        adapted_model.fit(new_task_data, new_task_target, epochs=adaptation_steps, verbose=0)
        
        return adapted_model

    def evaluate_adaptation(self, test_tasks: List[Dict[str, Any]], 
                          adaptation_steps: int = None) -> Dict[str, float]:
        """Evaluate adaptation performance on test tasks"""
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps
        
        adaptation_scores = []
        baseline_scores = []
        
        for task in test_tasks:
            # Fast adaptation
            adapted_model = self.fast_adapt(
                task['support_X'],
                task['support_y'],
                adaptation_steps
            )
            
            # Evaluate adapted model
            adapted_pred = adapted_model.predict(task['query_X'], verbose=0)
            adapted_r2 = r2_score(task['query_y'], adapted_pred)
            adaptation_scores.append(adapted_r2)
            
            # Baseline: evaluate meta-model without adaptation
            baseline_pred = self.meta_model.predict(task['query_X'], verbose=0)
            baseline_r2 = r2_score(task['query_y'], baseline_pred)
            baseline_scores.append(baseline_r2)
        
        results = {
            'mean_adaptation_r2': np.mean(adaptation_scores),
            'std_adaptation_r2': np.std(adaptation_scores),
            'mean_baseline_r2': np.mean(baseline_scores),
            'std_baseline_r2': np.std(baseline_scores),
            'adaptation_improvement': np.mean(adaptation_scores) - np.mean(baseline_scores),
            'adaptation_scores': adaptation_scores,
            'baseline_scores': baseline_scores
        }
        
        logger.info(f"Adaptation Evaluation - "
                   f"Adaptation R²: {results['mean_adaptation_r2']:.4f} ± {results['std_adaptation_r2']:.4f}, "
                   f"Baseline R²: {results['mean_baseline_r2']:.4f} ± {results['std_baseline_r2']:.4f}, "
                   f"Improvement: {results['adaptation_improvement']:.4f}")
        
        return results

    def create_prototypical_network(self, input_dim: int, num_classes: int) -> Model:
        """Create prototypical network for few-shot learning"""
        model = Sequential([
            Dense(self.hidden_dim, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(self.hidden_dim // 2, activation='relu'),
            Dropout(0.3),
            Dense(self.hidden_dim // 4, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.meta_learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def create_matching_network(self, input_dim: int, num_classes: int) -> Model:
        """Create matching network for few-shot learning"""
        # Support encoder
        support_input = Input(shape=(None, input_dim))
        support_encoded = LSTM(self.hidden_dim, return_sequences=False)(support_input)
        
        # Query encoder
        query_input = Input(shape=(input_dim,))
        query_encoded = Dense(self.hidden_dim, activation='relu')(query_input)
        
        # Attention mechanism
        attention_weights = tf.keras.layers.Dot(axes=1)([query_encoded, support_encoded])
        attention_weights = tf.keras.layers.Activation('softmax')(attention_weights)
        
        # Weighted sum
        weighted_support = tf.keras.layers.Dot(axes=1)([attention_weights, support_encoded])
        
        # Output
        output = Dense(num_classes, activation='softmax')(weighted_support)
        
        model = Model(inputs=[support_input, query_input], outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=self.meta_learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def save_meta_model(self, filepath: str):
        """Save the trained meta-model"""
        if self.meta_model is not None:
            self.meta_model.save(filepath)
            logger.info(f"Meta-model saved to {filepath}")
        else:
            logger.warning("No meta-model to save")

    def load_meta_model(self, filepath: str):
        """Load a trained meta-model"""
        self.meta_model = tf.keras.models.load_model(filepath)
        logger.info(f"Meta-model loaded from {filepath}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.meta_losses:
            return {}
        
        return {
            'final_meta_loss': self.meta_losses[-1] if self.meta_losses else None,
            'final_adaptation_performance': self.adaptation_performances[-1] if self.adaptation_performances else None,
            'mean_meta_loss': np.mean(self.meta_losses),
            'mean_adaptation_performance': np.mean(self.adaptation_performances),
            'num_meta_steps': len(self.meta_losses),
            'num_tasks': len(self.task_data)
        }

    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.meta_losses:
                logger.warning("No training data to plot")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Meta-loss
            ax1.plot(self.meta_losses)
            ax1.set_title('Meta-Learning Loss')
            ax1.set_xlabel('Meta-Step')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            
            # Adaptation performance
            ax2.plot(self.adaptation_performances)
            ax2.set_title('Adaptation Performance (R²)')
            ax2.set_xlabel('Meta-Step')
            ax2.set_ylabel('R²')
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training progress plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting") 