"""
GRU (Gated Recurrent Unit) Models for Advanced Time Series Prediction
Part of Project Hyperion - Ultimate Autonomous Trading Bot
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, LSTM, Attention, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


class GRUModels:
    """
    Advanced GRU Models for Time Series Prediction
    
    Features:
    - Simple GRU
    - Deep GRU
    - Bidirectional GRU
    - Stacked GRU
    - Attention GRU
    - Hybrid GRU-LSTM
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.model_configs = {}
        self.performance_metrics = {}
        
        # GRU model configurations
        self.gru_configs = {
            'simple_gru': {
                'units': 50,
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'deep_gru': {
                'units': [64, 32, 16],
                'dropout': 0.3,
                'recurrent_dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 150
            },
            'bidirectional_gru': {
                'units': 50,
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'stacked_gru': {
                'units': [64, 32],
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 120
            },
            'attention_gru': {
                'units': 50,
                'attention_units': 25,
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'hybrid_gru_lstm': {
                'gru_units': 50,
                'lstm_units': 30,
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 120
            }
        }
        
        logger.info("GRU Models initialized with configurations")

    def create_simple_gru(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a simple GRU model"""
        config = self.gru_configs['simple_gru']
        
        model = Sequential([
            GRU(units=config['units'], 
                return_sequences=False,
                dropout=config['dropout'],
                recurrent_dropout=config['recurrent_dropout'],
                input_shape=input_shape),
            Dropout(config['dropout']),
            Dense(output_size)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_deep_gru(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a deep GRU model with multiple layers"""
        config = self.gru_configs['deep_gru']
        
        model = Sequential()
        
        # First GRU layer
        model.add(GRU(units=config['units'][0], 
                     return_sequences=True,
                     dropout=config['dropout'],
                     recurrent_dropout=config['recurrent_dropout'],
                     input_shape=input_shape))
        model.add(Dropout(config['dropout']))
        
        # Middle GRU layers
        for units in config['units'][1:-1]:
            model.add(GRU(units=units, 
                         return_sequences=True,
                         dropout=config['dropout'],
                         recurrent_dropout=config['recurrent_dropout']))
            model.add(Dropout(config['dropout']))
        
        # Final GRU layer
        model.add(GRU(units=config['units'][-1], 
                     return_sequences=False,
                     dropout=config['dropout'],
                     recurrent_dropout=config['recurrent_dropout']))
        model.add(Dropout(config['dropout']))
        
        # Output layer
        model.add(Dense(output_size))
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_bidirectional_gru(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a bidirectional GRU model"""
        config = self.gru_configs['bidirectional_gru']
        
        model = Sequential([
            Bidirectional(GRU(units=config['units'], 
                             dropout=config['dropout'],
                             recurrent_dropout=config['recurrent_dropout']),
                         input_shape=input_shape),
            Dropout(config['dropout']),
            Dense(output_size)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_stacked_gru(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a stacked GRU model"""
        config = self.gru_configs['stacked_gru']
        
        model = Sequential()
        
        # First GRU layer
        model.add(GRU(units=config['units'][0], 
                     return_sequences=True,
                     dropout=config['dropout'],
                     recurrent_dropout=config['recurrent_dropout'],
                     input_shape=input_shape))
        model.add(Dropout(config['dropout']))
        
        # Second GRU layer
        model.add(GRU(units=config['units'][1], 
                     return_sequences=False,
                     dropout=config['dropout'],
                     recurrent_dropout=config['recurrent_dropout']))
        model.add(Dropout(config['dropout']))
        
        # Output layer
        model.add(Dense(output_size))
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_attention_gru(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a GRU model with attention mechanism"""
        config = self.gru_configs['attention_gru']
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # GRU layer with return sequences for attention
        gru_out = GRU(units=config['units'], 
                     return_sequences=True,
                     dropout=config['dropout'],
                     recurrent_dropout=config['recurrent_dropout'])(inputs)
        
        # Attention mechanism
        attention = Dense(config['attention_units'], activation='tanh')(gru_out)
        attention = Dense(1, activation='softmax')(attention)
        attention = tf.keras.layers.Multiply()([gru_out, attention])
        
        # Global average pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
        
        # Dropout and output
        dropout = Dropout(config['dropout'])(pooled)
        outputs = Dense(output_size)(dropout)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_hybrid_gru_lstm(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a hybrid GRU-LSTM model"""
        config = self.gru_configs['hybrid_gru_lstm']
        
        model = Sequential([
            GRU(units=config['gru_units'], 
                return_sequences=True,
                dropout=config['dropout'],
                recurrent_dropout=config['recurrent_dropout'],
                input_shape=input_shape),
            Dropout(config['dropout']),
            LSTM(units=config['lstm_units'], 
                 return_sequences=False,
                 dropout=config['dropout'],
                 recurrent_dropout=config['recurrent_dropout']),
            Dropout(config['dropout']),
            Dense(output_size)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def prepare_data(self, data: pd.DataFrame, target_column: str, 
                    sequence_length: int = 60, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for GRU models"""
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, data.columns.get_loc(target_column)])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split the data
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        return X_train, X_test, y_train, y_test, scaler

    def train_model(self, model: Model, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Train a GRU model"""
        config = self.gru_configs.get(model_name, self.gru_configs['simple_gru'])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint(f'models/gru_{model_name}_best.h5', 
                          monitor='val_loss', save_best_only=True)
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'history': history.history,
            'model': model,
            'config': config
        }

    def evaluate_model(self, model: Model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """Evaluate a GRU model"""
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        directional_accuracy = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred)))
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'rmse': np.sqrt(mse)
        }
        
        self.performance_metrics[model_name] = metrics
        
        logger.info(f"GRU Model {model_name} - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        return metrics

    def predict(self, model: Model, X: np.ndarray, model_name: str) -> np.ndarray:
        """Make predictions with a GRU model"""
        return model.predict(X)

    def create_all_models(self, input_shape: Tuple[int, int], output_size: int = 1) -> Dict[str, Model]:
        """Create all GRU model architectures"""
        models = {
            'simple_gru': self.create_simple_gru(input_shape, output_size),
            'deep_gru': self.create_deep_gru(input_shape, output_size),
            'bidirectional_gru': self.create_bidirectional_gru(input_shape, output_size),
            'stacked_gru': self.create_stacked_gru(input_shape, output_size),
            'attention_gru': self.create_attention_gru(input_shape, output_size),
            'hybrid_gru_lstm': self.create_hybrid_gru_lstm(input_shape, output_size)
        }
        
        self.models.update(models)
        
        logger.info(f"Created {len(models)} GRU model architectures")
        
        return models

    def get_model_summary(self, model_name: str) -> str:
        """Get model summary as string"""
        if model_name in self.models:
            model = self.models[model_name]
            
            # Capture model summary
            summary_list = []
            model.summary(print_fn=lambda x: summary_list.append(x))
            
            return '\n'.join(summary_list)
        
        return f"Model {model_name} not found"

    def save_model(self, model: Model, model_name: str, filepath: str):
        """Save a trained model"""
        model.save(filepath)
        logger.info(f"Saved GRU model {model_name} to {filepath}")

    def load_model(self, model_name: str, filepath: str) -> Model:
        """Load a trained model"""
        model = tf.keras.models.load_model(filepath)
        self.models[model_name] = model
        logger.info(f"Loaded GRU model {model_name} from {filepath}")
        return model 