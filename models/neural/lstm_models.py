"""
🧠 LSTM Models Module

This module implements advanced LSTM models for time series prediction
in cryptocurrency trading with multiple architectures and configurations.

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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Input, Concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configure logging
logger = logging.getLogger(__name__)

class LSTMModels:
    """
    🧠 Advanced LSTM Models Generator
    
    Implements multiple LSTM architectures for time series prediction
    in cryptocurrency trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LSTM models generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.scalers = {}
        self.model_configs = {}
        self.performance_metrics = {}
        
        # LSTM configurations
        self.lstm_configs = {
            'simple_lstm': {
                'layers': [50, 25],
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'optimizer': 'adam',
                'learning_rate': 0.001
            },
            'deep_lstm': {
                'layers': [100, 75, 50, 25],
                'dropout': 0.3,
                'recurrent_dropout': 0.3,
                'optimizer': 'adam',
                'learning_rate': 0.0005
            },
            'bidirectional_lstm': {
                'layers': [64, 32],
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'optimizer': 'adam',
                'learning_rate': 0.001
            },
            'stacked_lstm': {
                'layers': [128, 64, 32],
                'dropout': 0.25,
                'recurrent_dropout': 0.25,
                'optimizer': 'rmsprop',
                'learning_rate': 0.001
            },
            'attention_lstm': {
                'layers': [80, 40],
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'optimizer': 'adam',
                'learning_rate': 0.001
            }
        }
        
        logger.info("🧠 LSTM Models initialized")
    
    def create_simple_lstm(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a simple LSTM model."""
        try:
            config = self.lstm_configs['simple_lstm']
            
            model = Sequential([
                LSTM(config['layers'][0], 
                     return_sequences=True, 
                     input_shape=input_shape,
                     dropout=config['dropout'],
                     recurrent_dropout=config['recurrent_dropout']),
                LSTM(config['layers'][1], 
                     dropout=config['dropout'],
                     recurrent_dropout=config['recurrent_dropout']),
                Dense(output_size)
            ])
            
            optimizer = Adam(learning_rate=config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create simple LSTM: {e}")
            return None
    
    def create_deep_lstm(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a deep LSTM model."""
        try:
            config = self.lstm_configs['deep_lstm']
            
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(config['layers'][0], 
                          return_sequences=True, 
                          input_shape=input_shape,
                          dropout=config['dropout'],
                          recurrent_dropout=config['recurrent_dropout']))
            
            # Middle LSTM layers
            for i in range(1, len(config['layers']) - 1):
                model.add(LSTM(config['layers'][i], 
                              return_sequences=True,
                              dropout=config['dropout'],
                              recurrent_dropout=config['recurrent_dropout']))
            
            # Final LSTM layer
            model.add(LSTM(config['layers'][-1], 
                          dropout=config['dropout'],
                          recurrent_dropout=config['recurrent_dropout']))
            
            # Output layer
            model.add(Dense(output_size))
            
            optimizer = Adam(learning_rate=config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create deep LSTM: {e}")
            return None
    
    def create_bidirectional_lstm(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a bidirectional LSTM model."""
        try:
            config = self.lstm_configs['bidirectional_lstm']
            
            model = Sequential([
                Bidirectional(LSTM(config['layers'][0], 
                                  return_sequences=True,
                                  dropout=config['dropout'],
                                  recurrent_dropout=config['recurrent_dropout']),
                             input_shape=input_shape),
                Bidirectional(LSTM(config['layers'][1],
                                  dropout=config['dropout'],
                                  recurrent_dropout=config['recurrent_dropout'])),
                Dense(output_size)
            ])
            
            optimizer = Adam(learning_rate=config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create bidirectional LSTM: {e}")
            return None
    
    def create_stacked_lstm(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a stacked LSTM model."""
        try:
            config = self.lstm_configs['stacked_lstm']
            
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(config['layers'][0], 
                          return_sequences=True, 
                          input_shape=input_shape,
                          dropout=config['dropout'],
                          recurrent_dropout=config['recurrent_dropout']))
            
            # Stacked LSTM layers
            for i in range(1, len(config['layers'])):
                model.add(LSTM(config['layers'][i], 
                              return_sequences=(i < len(config['layers']) - 1),
                              dropout=config['dropout'],
                              recurrent_dropout=config['recurrent_dropout']))
            
            # Output layer
            model.add(Dense(output_size))
            
            optimizer = RMSprop(learning_rate=config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create stacked LSTM: {e}")
            return None
    
    def create_attention_lstm(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create an LSTM model with attention mechanism."""
        try:
            config = self.lstm_configs['attention_lstm']
            
            # Input layer
            inputs = Input(shape=input_shape)
            
            # LSTM layers
            lstm1 = LSTM(config['layers'][0], 
                        return_sequences=True,
                        dropout=config['dropout'],
                        recurrent_dropout=config['recurrent_dropout'])(inputs)
            
            lstm2 = LSTM(config['layers'][1],
                        return_sequences=True,
                        dropout=config['dropout'],
                        recurrent_dropout=config['recurrent_dropout'])(lstm1)
            
            # Simple attention mechanism
            attention = Dense(1, activation='tanh')(lstm2)
            attention = Dense(1, activation='softmax')(attention)
            
            # Apply attention
            context = tf.multiply(lstm2, attention)
            context = tf.reduce_sum(context, axis=1)
            
            # Output layer
            outputs = Dense(output_size)(context)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            optimizer = Adam(learning_rate=config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create attention LSTM: {e}")
            return None
    
    def prepare_data(self, data: pd.DataFrame, target_column: str, 
                    sequence_length: int = 60, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        try:
            # Select features and target
            feature_columns = [col for col in data.columns if col != target_column]
            features = data[feature_columns].values
            target = data[target_column].values
            
            # Scale the data
            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()
            
            features_scaled = feature_scaler.fit_transform(features)
            target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(features_scaled)):
                X.append(features_scaled[i-sequence_length:i])
                y.append(target_scaled[i])
            
            X = np.array(X)
            y = np.array(y)
            
            # Split into train and test
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Store scalers
            self.scalers['features'] = feature_scaler
            self.scalers['target'] = target_scaler
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare data: {e}")
            return None, None, None, None
    
    def train_model(self, model: Model, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Train an LSTM model."""
        try:
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
                ModelCheckpoint(f'models/lstm_{model_name}_best.h5', 
                              monitor='val_loss', save_best_only=True)
            ]
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Store model and history
            self.models[model_name] = model
            self.model_configs[model_name] = {
                'history': history.history,
                'epochs_trained': len(history.history['loss'])
            }
            
            logger.info(f"✅ {model_name} trained successfully")
            return history.history
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model_name}: {e}")
            return {}
    
    def evaluate_model(self, model: Model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """Evaluate an LSTM model."""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store metrics
            metrics = {
                'mse': mse,
                'mae': mae,
                'r2_score': r2,
                'rmse': np.sqrt(mse)
            }
            
            self.performance_metrics[model_name] = metrics
            
            logger.info(f"✅ {model_name} evaluation completed")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate {model_name}: {e}")
            return {}
    
    def predict(self, model: Model, X: np.ndarray, model_name: str) -> np.ndarray:
        """Make predictions with an LSTM model."""
        try:
            predictions = model.predict(X)
            
            # Inverse transform if scaler is available
            if 'target' in self.scalers:
                predictions = self.scalers['target'].inverse_transform(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to make predictions with {model_name}: {e}")
            return np.array([])
    
    def create_all_models(self, input_shape: Tuple[int, int], output_size: int = 1) -> Dict[str, Model]:
        """Create all LSTM model variants."""
        try:
            models = {}
            
            # Create all model types
            models['simple_lstm'] = self.create_simple_lstm(input_shape, output_size)
            models['deep_lstm'] = self.create_deep_lstm(input_shape, output_size)
            models['bidirectional_lstm'] = self.create_bidirectional_lstm(input_shape, output_size)
            models['stacked_lstm'] = self.create_stacked_lstm(input_shape, output_size)
            models['attention_lstm'] = self.create_attention_lstm(input_shape, output_size)
            
            # Store models
            self.models.update(models)
            
            logger.info(f"✅ Created {len(models)} LSTM models")
            return models
            
        except Exception as e:
            logger.error(f"❌ Failed to create all models: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of all LSTM models."""
        return {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'configurations': self.lstm_configs,
            'performance_metrics': self.performance_metrics,
            'model_configs': self.model_configs
        }
    
    def save_models(self, base_path: str = 'models/lstm/'):
        """Save all trained models."""
        try:
            import os
            os.makedirs(base_path, exist_ok=True)
            
            for model_name, model in self.models.items():
                model_path = os.path.join(base_path, f'{model_name}.h5')
                model.save(model_path)
                logger.info(f"💾 Saved {model_name} to {model_path}")
            
            # Save scalers
            scaler_path = os.path.join(base_path, 'scalers.pkl')
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers, f)
            
            logger.info(f"💾 Saved scalers to {scaler_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save models: {e}")
    
    def load_models(self, base_path: str = 'models/lstm/'):
        """Load trained models."""
        try:
            import os
            import pickle
            
            # Load models
            for model_name in self.lstm_configs.keys():
                model_path = os.path.join(base_path, f'{model_name}.h5')
                if os.path.exists(model_path):
                    self.models[model_name] = tf.keras.models.load_model(model_path)
                    logger.info(f"📂 Loaded {model_name} from {model_path}")
            
            # Load scalers
            scaler_path = os.path.join(base_path, 'scalers.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info(f"📂 Loaded scalers from {scaler_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'lstm_models_enabled': True,
        'sequence_length': 60,
        'test_size': 0.2
    }
    
    # Initialize LSTM models
    lstm_models = LSTMModels(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'target': np.random.randn(1000)
    })
    
    # Prepare data
    X_train, X_test, y_train, y_test = lstm_models.prepare_data(
        sample_data, 'target', sequence_length=60
    )
    
    if X_train is not None:
        # Create all models
        models = lstm_models.create_all_models((60, 3), 1)
        
        # Train and evaluate each model
        for model_name, model in models.items():
            if model is not None:
                # Train model
                history = lstm_models.train_model(model, X_train, y_train, X_test, y_test, model_name)
                
                # Evaluate model
                metrics = lstm_models.evaluate_model(model, X_test, y_test, model_name)
                print(f"{model_name}: {metrics}")
    
    # Get model summary
    summary = lstm_models.get_model_summary()
    print(f"Created {summary['total_models']} LSTM models") 