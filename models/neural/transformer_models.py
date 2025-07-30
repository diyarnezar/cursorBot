"""
🧠 Transformer Models Module

This module implements advanced Transformer models for time series prediction
in cryptocurrency trading with attention mechanisms.

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
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configure logging
logger = logging.getLogger(__name__)

class TransformerBlock(tf.keras.layers.Layer):
    """Custom Transformer block with self-attention."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """Token and position embedding layer."""
    
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerModels:
    """
    🧠 Advanced Transformer Models Generator
    
    Implements multiple Transformer architectures for time series prediction
    in cryptocurrency trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Transformer models generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.scalers = {}
        self.model_configs = {}
        self.performance_metrics = {}
        
        # Transformer configurations
        self.transformer_configs = {
            'simple_transformer': {
                'embed_dim': 32,
                'num_heads': 4,
                'ff_dim': 64,
                'num_blocks': 2,
                'dropout': 0.1,
                'learning_rate': 0.001
            },
            'deep_transformer': {
                'embed_dim': 64,
                'num_heads': 8,
                'ff_dim': 128,
                'num_blocks': 4,
                'dropout': 0.2,
                'learning_rate': 0.0005
            },
            'wide_transformer': {
                'embed_dim': 128,
                'num_heads': 16,
                'ff_dim': 256,
                'num_blocks': 3,
                'dropout': 0.15,
                'learning_rate': 0.001
            },
            'attention_transformer': {
                'embed_dim': 48,
                'num_heads': 6,
                'ff_dim': 96,
                'num_blocks': 3,
                'dropout': 0.1,
                'learning_rate': 0.001
            }
        }
        
        logger.info("🧠 Transformer Models initialized")
    
    def create_simple_transformer(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a simple Transformer model."""
        try:
            config = self.transformer_configs['simple_transformer']
            
            # Input layer
            inputs = Input(shape=input_shape)
            
            # Dense projection to embed_dim
            x = Dense(config['embed_dim'])(inputs)
            
            # Transformer blocks
            for _ in range(config['num_blocks']):
                x = TransformerBlock(
                    config['embed_dim'], 
                    config['num_heads'], 
                    config['ff_dim'], 
                    config['dropout']
                )(x)
            
            # Global average pooling
            x = GlobalAveragePooling1D()(x)
            
            # Output layer
            outputs = Dense(output_size)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            optimizer = Adam(learning_rate=config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create simple transformer: {e}")
            return None
    
    def create_deep_transformer(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a deep Transformer model."""
        try:
            config = self.transformer_configs['deep_transformer']
            
            # Input layer
            inputs = Input(shape=input_shape)
            
            # Dense projection to embed_dim
            x = Dense(config['embed_dim'])(inputs)
            
            # Multiple transformer blocks
            for i in range(config['num_blocks']):
                x = TransformerBlock(
                    config['embed_dim'], 
                    config['num_heads'], 
                    config['ff_dim'], 
                    config['dropout']
                )(x)
                
                # Add residual connection and normalization
                if i > 0:
                    x = LayerNormalization(epsilon=1e-6)(x)
            
            # Global average pooling
            x = GlobalAveragePooling1D()(x)
            
            # Dense layers for final prediction
            x = Dense(config['ff_dim'], activation='relu')(x)
            x = Dropout(config['dropout'])(x)
            outputs = Dense(output_size)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            optimizer = Adam(learning_rate=config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create deep transformer: {e}")
            return None
    
    def create_wide_transformer(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a wide Transformer model with more attention heads."""
        try:
            config = self.transformer_configs['wide_transformer']
            
            # Input layer
            inputs = Input(shape=input_shape)
            
            # Dense projection to embed_dim
            x = Dense(config['embed_dim'])(inputs)
            
            # Transformer blocks with wide attention
            for _ in range(config['num_blocks']):
                x = TransformerBlock(
                    config['embed_dim'], 
                    config['num_heads'], 
                    config['ff_dim'], 
                    config['dropout']
                )(x)
            
            # Global average pooling
            x = GlobalAveragePooling1D()(x)
            
            # Wide dense layers
            x = Dense(config['ff_dim'], activation='relu')(x)
            x = Dropout(config['dropout'])(x)
            x = Dense(config['ff_dim'] // 2, activation='relu')(x)
            x = Dropout(config['dropout'])(x)
            outputs = Dense(output_size)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            optimizer = Adam(learning_rate=config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create wide transformer: {e}")
            return None
    
    def create_attention_transformer(self, input_shape: Tuple[int, int], output_size: int = 1) -> Model:
        """Create a Transformer model with enhanced attention mechanisms."""
        try:
            config = self.transformer_configs['attention_transformer']
            
            # Input layer
            inputs = Input(shape=input_shape)
            
            # Dense projection to embed_dim
            x = Dense(config['embed_dim'])(inputs)
            
            # Multiple transformer blocks with different attention patterns
            for i in range(config['num_blocks']):
                # Self-attention block
                x = TransformerBlock(
                    config['embed_dim'], 
                    config['num_heads'], 
                    config['ff_dim'], 
                    config['dropout']
                )(x)
                
                # Additional attention layer for enhanced focus
                if i < config['num_blocks'] - 1:
                    attention_output = MultiHeadAttention(
                        num_heads=config['num_heads'], 
                        key_dim=config['embed_dim']
                    )(x, x)
                    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Global average pooling
            x = GlobalAveragePooling1D()(x)
            
            # Output layers
            x = Dense(config['ff_dim'], activation='relu')(x)
            x = Dropout(config['dropout'])(x)
            outputs = Dense(output_size)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            optimizer = Adam(learning_rate=config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create attention transformer: {e}")
            return None
    
    def prepare_data(self, data: pd.DataFrame, target_column: str, 
                    sequence_length: int = 60, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for Transformer training."""
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
        """Train a Transformer model."""
        try:
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
                ModelCheckpoint(f'models/transformer_{model_name}_best.h5', 
                              monitor='val_loss', save_best_only=True)
            ]
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=150,
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
        """Evaluate a Transformer model."""
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
        """Make predictions with a Transformer model."""
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
        """Create all Transformer model variants."""
        try:
            models = {}
            
            # Create all model types
            models['simple_transformer'] = self.create_simple_transformer(input_shape, output_size)
            models['deep_transformer'] = self.create_deep_transformer(input_shape, output_size)
            models['wide_transformer'] = self.create_wide_transformer(input_shape, output_size)
            models['attention_transformer'] = self.create_attention_transformer(input_shape, output_size)
            
            # Store models
            self.models.update(models)
            
            logger.info(f"✅ Created {len(models)} Transformer models")
            return models
            
        except Exception as e:
            logger.error(f"❌ Failed to create all models: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of all Transformer models."""
        return {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'configurations': self.transformer_configs,
            'performance_metrics': self.performance_metrics,
            'model_configs': self.model_configs
        }
    
    def save_models(self, base_path: str = 'models/transformer/'):
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
    
    def load_models(self, base_path: str = 'models/transformer/'):
        """Load trained models."""
        try:
            import os
            import pickle
            
            # Load models
            for model_name in self.transformer_configs.keys():
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
        'transformer_models_enabled': True,
        'sequence_length': 60,
        'test_size': 0.2
    }
    
    # Initialize Transformer models
    transformer_models = TransformerModels(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'target': np.random.randn(1000)
    })
    
    # Prepare data
    X_train, X_test, y_train, y_test = transformer_models.prepare_data(
        sample_data, 'target', sequence_length=60
    )
    
    if X_train is not None:
        # Create all models
        models = transformer_models.create_all_models((60, 3), 1)
        
        # Train and evaluate each model
        for model_name, model in models.items():
            if model is not None:
                # Train model
                history = transformer_models.train_model(model, X_train, y_train, X_test, y_test, model_name)
                
                # Evaluate model
                metrics = transformer_models.evaluate_model(model, X_test, y_test, model_name)
                print(f"{model_name}: {metrics}")
    
    # Get model summary
    summary = transformer_models.get_model_summary()
    print(f"Created {summary['total_models']} Transformer models") 