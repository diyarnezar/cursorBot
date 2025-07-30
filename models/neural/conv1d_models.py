"""
Conv1D (1D Convolutional) Models for Advanced Time Series Prediction
Part of Project Hyperion - Ultimate Autonomous Trading Bot
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Flatten, GlobalAveragePooling1D, Input, BatchNormalization, Add, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


class Conv1DModels:
    """
    Advanced Conv1D Models for Time Series Prediction
    
    Features:
    - Simple Conv1D
    - Deep Conv1D
    - Residual Conv1D
    - Multi-scale Conv1D
    - Attention Conv1D
    - Hybrid Conv1D-LSTM
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.model_configs = {}
        self.performance_metrics = {}
        
        # Conv1D model configurations
        self.conv1d_configs = {
            'simple_conv1d': {
                'filters': [32, 64],
                'kernel_sizes': [3, 3],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'deep_conv1d': {
                'filters': [32, 64, 128, 256],
                'kernel_sizes': [3, 3, 3, 3],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 150
            },
            'residual_conv1d': {
                'filters': [32, 64, 128],
                'kernel_sizes': [3, 3, 3],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 120
            },
            'multiscale_conv1d': {
                'filters': [32, 64, 128],
                'kernel_sizes': [3, 5, 7],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 120
            },
            'attention_conv1d': {
                'filters': [32, 64, 128],
                'kernel_sizes': [3, 3, 3],
                'attention_units': 64,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 120
            },
            'hybrid_conv1d_lstm': {
                'conv_filters': [32, 64],
                'conv_kernel_sizes': [3, 3],
                'lstm_units': 50,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 120
            }
        }
        
        logger.info("Conv1D Models initialized with configurations")

    def create_simple_conv1d(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a simple Conv1D model"""
        config = self.conv1d_configs['simple_conv1d']
        
        model = Sequential([
            Conv1D(filters=config['filters'][0], 
                   kernel_size=config['kernel_sizes'][0],
                   activation='relu',
                   input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(config['dropout']),
            
            Conv1D(filters=config['filters'][1], 
                   kernel_size=config['kernel_sizes'][1],
                   activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(config['dropout']),
            
            GlobalAveragePooling1D(),
            Dense(50, activation='relu'),
            Dropout(config['dropout']),
            Dense(output_size)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_deep_conv1d(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a deep Conv1D model with multiple layers"""
        config = self.conv1d_configs['deep_conv1d']
        
        model = Sequential()
        
        # Input layer
        model.add(Conv1D(filters=config['filters'][0], 
                        kernel_size=config['kernel_sizes'][0],
                        activation='relu',
                        input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(config['dropout']))
        
        # Middle layers
        for i in range(1, len(config['filters']) - 1):
            model.add(Conv1D(filters=config['filters'][i], 
                           kernel_size=config['kernel_sizes'][i],
                           activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(config['dropout']))
        
        # Final conv layer
        model.add(Conv1D(filters=config['filters'][-1], 
                        kernel_size=config['kernel_sizes'][-1],
                        activation='relu'))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(config['dropout']))
        
        # Dense layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(config['dropout']))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(config['dropout']))
        model.add(Dense(output_size))
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_residual_conv1d(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a residual Conv1D model"""
        config = self.conv1d_configs['residual_conv1d']
        
        inputs = Input(shape=input_shape)
        
        # Initial conv layer
        x = Conv1D(filters=config['filters'][0], 
                  kernel_size=config['kernel_sizes'][0],
                  activation='relu',
                  padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # Residual blocks
        for i in range(1, len(config['filters'])):
            # Residual connection
            residual = x
            
            # Conv layers
            x = Conv1D(filters=config['filters'][i], 
                      kernel_size=config['kernel_sizes'][i],
                      activation='relu',
                      padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv1D(filters=config['filters'][i], 
                      kernel_size=config['kernel_sizes'][i],
                      activation='relu',
                      padding='same')(x)
            x = BatchNormalization()(x)
            
            # Add residual connection if dimensions match
            if x.shape[-1] == residual.shape[-1]:
                x = Add()([x, residual])
            
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(config['dropout'])(x)
        
        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(config['dropout'])(x)
        outputs = Dense(output_size)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_multiscale_conv1d(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a multi-scale Conv1D model"""
        config = self.conv1d_configs['multiscale_conv1d']
        
        inputs = Input(shape=input_shape)
        
        # Multi-scale conv branches
        conv_outputs = []
        for i, (filters, kernel_size) in enumerate(zip(config['filters'], config['kernel_sizes'])):
            conv_branch = Conv1D(filters=filters, 
                               kernel_size=kernel_size,
                               activation='relu',
                               padding='same')(inputs)
            conv_branch = BatchNormalization()(conv_branch)
            conv_branch = MaxPooling1D(pool_size=2)(conv_branch)
            conv_branch = Dropout(config['dropout'])(conv_branch)
            conv_outputs.append(conv_branch)
        
        # Concatenate multi-scale features
        if len(conv_outputs) > 1:
            x = Concatenate()(conv_outputs)
        else:
            x = conv_outputs[0]
        
        # Additional processing
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(config['dropout'])(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(config['dropout'])(x)
        outputs = Dense(output_size)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_attention_conv1d(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a Conv1D model with attention mechanism"""
        config = self.conv1d_configs['attention_conv1d']
        
        inputs = Input(shape=input_shape)
        
        # Conv layers
        x = Conv1D(filters=config['filters'][0], 
                  kernel_size=config['kernel_sizes'][0],
                  activation='relu',
                  padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        for i in range(1, len(config['filters'])):
            x = Conv1D(filters=config['filters'][i], 
                      kernel_size=config['kernel_sizes'][i],
                      activation='relu',
                      padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
        
        # Attention mechanism
        attention = Dense(config['attention_units'], activation='tanh')(x)
        attention = Dense(1, activation='softmax')(attention)
        attention = tf.keras.layers.Multiply()([x, attention])
        
        # Global pooling
        x = GlobalAveragePooling1D()(attention)
        x = Dropout(config['dropout'])(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(config['dropout'])(x)
        outputs = Dense(output_size)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_hybrid_conv1d_lstm(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a hybrid Conv1D-LSTM model"""
        config = self.conv1d_configs['hybrid_conv1d_lstm']
        
        model = Sequential([
            # Conv1D layers
            Conv1D(filters=config['conv_filters'][0], 
                   kernel_size=config['conv_kernel_sizes'][0],
                   activation='relu',
                   input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(config['dropout']),
            
            Conv1D(filters=config['conv_filters'][1], 
                   kernel_size=config['conv_kernel_sizes'][1],
                   activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(config['dropout']),
            
            # LSTM layer
            tf.keras.layers.LSTM(units=config['lstm_units'], 
                                return_sequences=False,
                                dropout=config['dropout'],
                                recurrent_dropout=config['dropout']),
            Dropout(config['dropout']),
            
            # Dense layers
            Dense(50, activation='relu'),
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
        """Prepare data for Conv1D models"""
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
        """Train a Conv1D model"""
        config = self.conv1d_configs.get(model_name, self.conv1d_configs['simple_conv1d'])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint(f'models/conv1d_{model_name}_best.h5', 
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
        """Evaluate a Conv1D model"""
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
        
        logger.info(f"Conv1D Model {model_name} - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        return metrics

    def predict(self, model: Model, X: np.ndarray, model_name: str) -> np.ndarray:
        """Make predictions with a Conv1D model"""
        return model.predict(X)

    def create_all_models(self, input_shape: Tuple[int, int], output_size: int = 1) -> Dict[str, Model]:
        """Create all Conv1D model architectures"""
        models = {
            'simple_conv1d': self.create_simple_conv1d(input_shape, output_size),
            'deep_conv1d': self.create_deep_conv1d(input_shape, output_size),
            'residual_conv1d': self.create_residual_conv1d(input_shape, output_size),
            'multiscale_conv1d': self.create_multiscale_conv1d(input_shape, output_size),
            'attention_conv1d': self.create_attention_conv1d(input_shape, output_size),
            'hybrid_conv1d_lstm': self.create_hybrid_conv1d_lstm(input_shape, output_size)
        }
        
        self.models.update(models)
        
        logger.info(f"Created {len(models)} Conv1D model architectures")
        
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
        logger.info(f"Saved Conv1D model {model_name} to {filepath}")

    def load_model(self, model_name: str, filepath: str) -> Model:
        """Load a trained model"""
        model = tf.keras.models.load_model(filepath)
        self.models[model_name] = model
        logger.info(f"Loaded Conv1D model {model_name} from {filepath}")
        return model 