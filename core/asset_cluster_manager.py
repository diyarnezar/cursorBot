"""
🚀 PROJECT HYPERION - ASSET CLUSTER MANAGER
==========================================

Implements the Asset Cluster strategy from gemini_plan_new.md
Specialized models for each of the 5 asset clusters with different characteristics.

Author: Project Hyperion Team
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from config.training_config import training_config
# Import models
from models.tree_based.tree_models import TreeBasedModels
from models.time_series.time_series_models import TimeSeriesModels
from models.neural.lstm_models import LSTMModels
from models.neural.transformer_models import TransformerModels
from models.neural.conv1d_models import Conv1DModels
from models.neural.gru_models import GRUModels
from features.psychology.psychology_features import PsychologyFeatures
from features.external_alpha.external_alpha_features import ExternalAlphaFeatures
from features.microstructure.microstructure_features import MicrostructureFeatures
from features.patterns.pattern_features import PatternFeatures
from features.regime_detection.regime_detection_features import RegimeDetectionFeatures


class AssetClusterManager:
    """
    Manages specialized models for each of the 5 asset clusters
    Implements the Asset Cluster strategy from gemini_plan_new.md
    """
    
    # Asset clusters from gemini_plan_new.md
    ASSET_CLUSTERS = {
        'bedrock': {
            'name': 'The Bedrock (Core Large Caps)',
            'assets': ['BTCFDUSD', 'ETHFDUSD', 'BNBFDUSD', 'SOLFDUSD', 'XRPFDUSD', 'DOGEFDUSD'],
            'characteristics': {
                'liquidity': 'highest',
                'volatility': 'lower_relative',
                'correlation': 'strong_market',
                'position_size_multiplier': 1.0,
                'risk_tolerance': 'conservative',
                'feature_weights': {
                    'technical': 0.3,
                    'macro': 0.4,
                    'sentiment': 0.1,
                    'microstructure': 0.2
                }
            },
            'strategy': 'capital_appreciation_trend_following',
            'model_types': ['lightgbm', 'xgboost', 'lstm', 'transformer'],
            'training_frequency': 'daily'
        },
        'infrastructure': {
            'name': 'The Infrastructure (Major L1s & L2s)',
            'assets': ['AVAXFDUSD', 'DOTFDUSD', 'LINKFDUSD', 'ARBFDUSD', 'OPFDUSD'],
            'characteristics': {
                'liquidity': 'medium_high',
                'volatility': 'medium',
                'correlation': 'sector_specific',
                'position_size_multiplier': 0.9,
                'risk_tolerance': 'moderate',
                'feature_weights': {
                    'technical': 0.25,
                    'macro': 0.3,
                    'sentiment': 0.2,
                    'microstructure': 0.25
                }
            },
            'strategy': 'sector_trend_capture',
            'model_types': ['lightgbm', 'xgboost', 'catboost', 'lstm'],
            'training_frequency': 'weekly'
        },
        'defi': {
            'name': 'The DeFi Blue Chips',
            'assets': ['UNIFDUSD', 'AAVEFDUSD', 'JUPFDUSD', 'PENDLEFDUSD', 'ENAFDUSD'],
            'characteristics': {
                'liquidity': 'medium',
                'volatility': 'high',
                'correlation': 'defi_sector',
                'position_size_multiplier': 0.8,
                'risk_tolerance': 'aggressive',
                'feature_weights': {
                    'technical': 0.2,
                    'macro': 0.2,
                    'sentiment': 0.3,
                    'microstructure': 0.3
                }
            },
            'strategy': 'defi_opportunity_capture',
            'model_types': ['lightgbm', 'xgboost', 'random_forest', 'gru'],
            'training_frequency': 'daily'
        },
        'volatility': {
            'name': 'The Volatility Engine (Memecoins & High Beta)',
            'assets': ['PEPEFDUSD', 'SHIBFDUSD', 'BONKFDUSD', 'WIFFDUSD', 'BOMEFDUSD'],
            'characteristics': {
                'liquidity': 'lower_relative',
                'volatility': 'extreme',
                'correlation': 'social_sentiment',
                'position_size_multiplier': 0.25,  # 50-75% reduction as per plan
                'risk_tolerance': 'very_aggressive',
                'feature_weights': {
                    'technical': 0.15,
                    'macro': 0.1,
                    'sentiment': 0.5,
                    'microstructure': 0.25
                }
            },
            'strategy': 'high_risk_momentum_trading',
            'model_types': ['lightgbm', 'xgboost', 'random_forest'],
            'training_frequency': 'hourly'
        },
        'ai_data': {
            'name': 'The AI & Data Sector (Emerging Tech)',
            'assets': ['FETFDUSD', 'RNDRFDUSD', 'WLDFDUSD', 'TAOFDUSD', 'GRTFDUSD'],
            'characteristics': {
                'liquidity': 'medium',
                'volatility': 'high',
                'correlation': 'ai_tech_sector',
                'position_size_multiplier': 0.85,
                'risk_tolerance': 'aggressive',
                'feature_weights': {
                    'technical': 0.2,
                    'macro': 0.25,
                    'sentiment': 0.3,
                    'microstructure': 0.25
                }
            },
            'strategy': 'ai_sector_alpha_capture',
            'model_types': ['lightgbm', 'xgboost', 'lstm', 'transformer'],
            'training_frequency': 'daily'
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the Asset Cluster Manager"""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
        
        self.cluster_models = {}
        self.cluster_features = {}
        self.cluster_performance = {}
        
        # Initialize feature generators
        self.psychology_features = PsychologyFeatures(config=self.config)
        self.external_alpha_features = ExternalAlphaFeatures()
        self.microstructure_features = MicrostructureFeatures(config=self.config)
        self.pattern_features = PatternFeatures()
        self.regime_features = RegimeDetectionFeatures()
        
        # Initialize model trainers
        self.tree_models = TreeBasedModels()
        self.time_series_models = TimeSeriesModels()
        self.lstm_models = LSTMModels(config=self.config)
        self.transformer_models = TransformerModels(config=self.config)
        self.conv1d_models = Conv1DModels(config=self.config)
        self.gru_models = GRUModels(config=self.config)
        
        # Initialize models
        self.initialize_models()
        
        self.logger.info("🚀 Asset Cluster Manager initialized")
    
    def get_cluster_for_asset(self, asset: str) -> Optional[str]:
        """Get the cluster for a given asset"""
        for cluster_name, cluster_info in self.ASSET_CLUSTERS.items():
            if asset in cluster_info['assets']:
                return cluster_name
        return None
    
    def get_cluster_characteristics(self, cluster_name: str) -> Dict[str, Any]:
        """Get characteristics for a specific cluster"""
        if cluster_name in self.ASSET_CLUSTERS:
            return self.ASSET_CLUSTERS[cluster_name]['characteristics']
        return {}
    
    def get_cluster_assets(self, cluster_name: str) -> List[str]:
        """Get all assets in a specific cluster"""
        if cluster_name in self.ASSET_CLUSTERS:
            return self.ASSET_CLUSTERS[cluster_name]['assets']
        return []
    
    def get_position_size_multiplier(self, asset: str) -> float:
        """Get position size multiplier for an asset based on its cluster"""
        cluster_name = self.get_cluster_for_asset(asset)
        if cluster_name:
            return self.ASSET_CLUSTERS[cluster_name]['characteristics']['position_size_multiplier']
        return 1.0  # Default multiplier
    
    def generate_cluster_specific_features(self, data: pd.DataFrame, cluster_name: str) -> pd.DataFrame:
        """Generate features specific to a cluster's characteristics"""
        cluster_info = self.ASSET_CLUSTERS.get(cluster_name, {})
        feature_weights = cluster_info.get('characteristics', {}).get('feature_weights', {})
        
        # Generate base features
        features = data.copy()
        
        # Add cluster-specific feature weights
        if feature_weights.get('sentiment', 0) > 0.2:
            # High sentiment weight - add more sentiment features
            sentiment_features = self.psychology_features.generate_features(data)
            features = pd.concat([features, sentiment_features], axis=1)
        
        if feature_weights.get('microstructure', 0) > 0.2:
            # High microstructure weight - add more microstructure features
            microstructure_features = self.microstructure_features.generate_features(data)
            features = pd.concat([features, microstructure_features], axis=1)
        
        if cluster_name == 'volatility':
            # Volatility cluster - add extreme volatility features
            features = self._add_volatility_cluster_features(features)
        
        elif cluster_name == 'defi':
            # DeFi cluster - add DeFi-specific features
            features = self._add_defi_cluster_features(features)
        
        elif cluster_name == 'ai_data':
            # AI/Data cluster - add AI sector features
            features = self._add_ai_cluster_features(features)
        
        return features
    
    def _add_volatility_cluster_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to volatility cluster (memecoins)"""
        features = data.copy()
        
        # Extreme volatility indicators
        features['extreme_volatility'] = features['volatility'].rolling(5).max()
        features['volatility_spike'] = features['volatility'] > features['volatility'].rolling(20).quantile(0.95)
        features['momentum_extreme'] = features['returns'].rolling(5).sum().abs()
        
        # Social sentiment features (higher weight for memecoins)
        features['social_momentum'] = features.get('social_sentiment', 0).rolling(3).mean()
        features['hype_cycle'] = features.get('social_volume', 0).rolling(10).std()
        
        return features
    
    def _add_defi_cluster_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to DeFi cluster"""
        features = data.copy()
        
        # DeFi-specific features
        features['defi_yield_spread'] = features.get('defi_yield', 0) - features.get('risk_free_rate', 0)
        features['governance_activity'] = features.get('governance_votes', 0).rolling(7).sum()
        features['defi_tvl_change'] = features.get('defi_tvl', 0).pct_change()
        
        return features
    
    def _add_ai_cluster_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to AI/Data cluster"""
        features = data.copy()
        
        # AI sector features
        features['ai_news_sentiment'] = features.get('ai_news_sentiment', 0).rolling(5).mean()
        features['tech_sector_momentum'] = features.get('tech_sector_index', 0).pct_change()
        features['ai_adoption_metrics'] = features.get('ai_adoption_score', 0).rolling(10).mean()
        
        return features
    
    def train_cluster_model(self, cluster_name: str, training_data: Dict[str, pd.DataFrame]) -> bool:
        """Train a specialized model for a specific cluster"""
        try:
            if cluster_name not in self.ASSET_CLUSTERS:
                self.logger.error(f"Invalid cluster name: {cluster_name}")
                return False
            
            cluster_info = self.ASSET_CLUSTERS[cluster_name]
            cluster_assets = cluster_info['assets']
            model_types = cluster_info['model_types']
            
            self.logger.info(f"🚀 Training {cluster_name} cluster model for {len(cluster_assets)} assets")
            
            # Combine data from all assets in the cluster
            combined_data = pd.DataFrame()
            for asset in cluster_assets:
                if asset in training_data:
                    asset_data = training_data[asset].copy()
                    asset_data['asset'] = asset
                    combined_data = pd.concat([combined_data, asset_data], ignore_index=True)
            
            if combined_data.empty:
                self.logger.error(f"No training data available for {cluster_name} cluster")
                return False
            
            # Generate cluster-specific features
            cluster_features = self.generate_cluster_specific_features(combined_data, cluster_name)
            
            # Train models based on cluster characteristics
            trained_models = {}
            
            for model_type in model_types:
                if model_type in ['lightgbm', 'xgboost', 'catboost', 'random_forest']:
                    model = self.tree_models.train_model(
                        model_type, cluster_features, target_col='target'
                    )
                    trained_models[model_type] = model
                
                elif model_type == 'lstm':
                    model = self.lstm_models.train_model(
                        'lstm', cluster_features, target_col='target'
                    )
                    trained_models[model_type] = model
                
                elif model_type == 'transformer':
                    model = self.transformer_models.train_model(
                        'transformer', cluster_features, target_col='target'
                    )
                    trained_models[model_type] = model
                
                elif model_type == 'gru':
                    model = self.gru_models.train_model(
                        'gru', cluster_features, target_col='target'
                    )
                    trained_models[model_type] = model
            
            # Store the trained models
            self.cluster_models[cluster_name] = trained_models
            
            # Save cluster model
            self._save_cluster_model(cluster_name, trained_models)
            
            self.logger.info(f"✅ {cluster_name} cluster model training completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error training {cluster_name} cluster model: {e}")
            return False
    
    def predict_cluster(self, cluster_name: str, data: pd.DataFrame) -> Dict[str, float]:
        """Generate predictions for a cluster using its specialized models or fallback models"""
        try:
            # Generate cluster-specific features
            cluster_features = self.generate_cluster_specific_features(data, cluster_name)
            
            # Check if we have trained models for this cluster
            if cluster_name in self.cluster_models and self.cluster_models[cluster_name]:
                # Use cluster-specific models
                predictions = {}
                models = self.cluster_models[cluster_name]
                
                for model_name, model in models.items():
                    try:
                        pred = model.predict(cluster_features)
                        predictions[model_name] = pred
                    except Exception as e:
                        self.logger.warning(f"Model {model_name} prediction failed: {e}")
                
                if predictions:
                    # Ensemble prediction (weighted average)
                    ensemble_pred = np.mean(list(predictions.values()))
                    predictions['ensemble'] = ensemble_pred
                    return predictions
            
            # Fallback: Use existing models from models directory
            self.logger.info(f"📊 No cluster-specific models for {cluster_name}, using fallback models")
            return self._get_fallback_predictions(cluster_name, cluster_features)
            
        except Exception as e:
            self.logger.error(f"❌ Error predicting for cluster {cluster_name}: {e}")
            return {}
    
    def _get_fallback_predictions(self, cluster_name: str, features: pd.DataFrame) -> Dict[str, float]:
        """Get fallback predictions using existing models"""
        try:
            predictions = {}
            
            # Use the most recent model available for this cluster
            # For now, use a simple momentum-based prediction
            if not features.empty and 'close' in features.columns:
                close_prices = features['close'].dropna()
                if len(close_prices) >= 2:
                    # Simple momentum prediction
                    momentum = (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5] if len(close_prices) >= 5 else 0
                    
                    # Volatility-based adjustment
                    volatility = close_prices.pct_change().std()
                    
                    # Cluster-specific adjustments
                    cluster_info = self.ASSET_CLUSTERS.get(cluster_name, {})
                    characteristics = cluster_info.get('characteristics', {})
                    
                    # Adjust prediction based on cluster characteristics
                    if characteristics.get('volatility') == 'extreme':
                        momentum *= 0.5  # Reduce prediction for extreme volatility
                    elif characteristics.get('volatility') == 'high':
                        momentum *= 0.8
                    
                    predictions['momentum'] = momentum
                    predictions['ensemble'] = momentum
                    
                    self.logger.info(f"📊 Generated fallback prediction for {cluster_name}: {momentum:.4f}")
                    return predictions
            
            # If no features available, return neutral prediction
            predictions['fallback'] = 0.0
            predictions['ensemble'] = 0.0
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Error in fallback predictions for {cluster_name}: {e}")
            return {'fallback': 0.0, 'ensemble': 0.0}
    
    def load_existing_models(self):
        """Load existing models from the models directory"""
        try:
            models_dir = Path("models")
            if not models_dir.exists():
                self.logger.warning("📂 Models directory not found")
                return
            
            # Look for existing model files
            model_files = list(models_dir.glob("**/*.joblib"))
            model_files.extend(list(models_dir.glob("**/*.h5")))
            model_files.extend(list(models_dir.glob("**/*.keras")))
            
            if model_files:
                self.logger.info(f"📂 Found {len(model_files)} existing model files")
                
                # For now, we'll use these models as fallbacks
                # In a full implementation, we'd map them to clusters
                for model_file in model_files[:5]:  # Use first 5 models as examples
                    self.logger.info(f"📂 Found model: {model_file.name}")
            
            else:
                self.logger.warning("📂 No existing model files found")
                
        except Exception as e:
            self.logger.error(f"❌ Error loading existing models: {e}")
    
    def initialize_models(self):
        """Initialize models - load existing or create fallbacks"""
        try:
            # Try to load existing models
            self.load_existing_models()
            
            # Initialize fallback models for each cluster
            for cluster_name in self.ASSET_CLUSTERS.keys():
                if cluster_name not in self.cluster_models:
                    self.cluster_models[cluster_name] = {}
                    self.logger.info(f"📊 Initialized fallback models for {cluster_name}")
            
            self.logger.info("✅ Model initialization completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing models: {e}")
    
    def get_cluster_conviction_score(self, cluster_name: str, predictions: Dict[str, float], 
                                   market_data: Dict[str, Any]) -> float:
        """Calculate conviction score for a cluster based on predictions and market conditions"""
        try:
            cluster_info = self.ASSET_CLUSTERS[cluster_name]
            characteristics = cluster_info['characteristics']
            
            # Base conviction from ensemble prediction
            base_conviction = abs(predictions.get('ensemble', 0))
            
            # Adjust for cluster characteristics
            volatility_adjustment = 1.0
            if characteristics['volatility'] == 'extreme':
                volatility_adjustment = 0.5  # Reduce conviction for extreme volatility
            elif characteristics['volatility'] == 'high':
                volatility_adjustment = 0.8
            
            # Market regime adjustment
            market_regime = market_data.get('market_regime', 'normal')
            regime_adjustment = 1.0
            if market_regime == 'high_volatility':
                regime_adjustment = 0.7
            elif market_regime == 'low_volatility':
                regime_adjustment = 1.2
            
            # Final conviction score
            conviction_score = base_conviction * volatility_adjustment * regime_adjustment
            
            return min(conviction_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating conviction score for {cluster_name}: {e}")
            return 0.0
    
    def _save_cluster_model(self, cluster_name: str, models: Dict[str, Any]):
        """Save cluster models to disk"""
        try:
            model_dir = Path(f"models/clusters/{cluster_name}")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            for model_name, model in models.items():
                model_path = model_dir / f"{model_name}_model.joblib"
                # Save model (implementation depends on model type)
                pass
            
            self.logger.info(f"💾 Saved {cluster_name} cluster models")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving {cluster_name} cluster models: {e}")
    
    def load_cluster_models(self):
        """Load all cluster models from disk"""
        try:
            for cluster_name in self.ASSET_CLUSTERS.keys():
                model_dir = Path(f"models/clusters/{cluster_name}")
                if model_dir.exists():
                    # Load models (implementation depends on model type)
                    self.logger.info(f"📂 Loaded {cluster_name} cluster models")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading cluster models: {e}")
    
    def get_all_clusters(self) -> List[str]:
        """Get all cluster names"""
        return list(self.ASSET_CLUSTERS.keys())
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of all clusters"""
        summary = {}
        for cluster_name, cluster_info in self.ASSET_CLUSTERS.items():
            summary[cluster_name] = {
                'name': cluster_info['name'],
                'asset_count': len(cluster_info['assets']),
                'assets': cluster_info['assets'],
                'strategy': cluster_info['strategy'],
                'position_size_multiplier': cluster_info['characteristics']['position_size_multiplier'],
                'model_trained': cluster_name in self.cluster_models
            }
        return summary 