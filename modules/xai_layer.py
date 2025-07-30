"""
Advanced Explainable AI (XAI) Layer
Provides comprehensive explanations for trading decisions and model predictions
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import os
from dataclasses import dataclass, asdict
import shap
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ExplanationResult:
    """Structured explanation result"""
    action: str
    confidence: float
    feature_importance: Dict[str, float]
    reasoning: List[str]
    model_contributions: Dict[str, float]
    risk_factors: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]

class AdvancedExplainabilitySystem:
    """
    Advanced explainability system for trading decisions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize explainers
        self.explainers = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        
        # Explanation history
        self.explanation_history = []
        self.max_history_size = 1000
        
        # Performance tracking
        self.explanation_times = []
        
        # Initialize with dummy data to avoid 0 features error
        self._initialize_explainers()
        
        self.logger.info("🔍 Advanced Explainability System initialized")

    def _initialize_explainers(self):
        """Initialize SHAP explainers with dummy data"""
        try:
            # Create dummy data with at least 1 feature
            dummy_features = np.random.randn(100, 5)  # 100 samples, 5 features
            dummy_feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
            
            # Fit scaler
            self.scaler.fit(dummy_features)
            self.feature_names = dummy_feature_names
            
            # Initialize explainers for different model types
            self.explainers = {
                'lightgbm': shap.TreeExplainer(None),  # Will be updated with actual model
                'xgboost': shap.TreeExplainer(None),
                'neural_network': shap.DeepExplainer(None, None),
                'linear': shap.LinearExplainer(None, None)
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing explainers: {e}")
            # Fallback: create minimal explainers
            self.explainers = {}

    def update_models(self, models: Dict[str, Any], feature_names: List[str]):
        """Update models and feature names for explanations"""
        try:
            self.feature_names = feature_names
            
            # Update explainers with actual models
            for model_name, model in models.items():
                if model is not None:
                    if model_name in ['lightgbm', 'xgboost']:
                        self.explainers[model_name] = shap.TreeExplainer(model)
                    elif model_name == 'neural_network':
                        # For neural networks, we need background data
                        background_data = np.random.randn(50, len(feature_names))
                        self.explainers[model_name] = shap.DeepExplainer(model, background_data)
                    elif model_name == 'linear':
                        background_data = np.random.randn(50, len(feature_names))
                        self.explainers[model_name] = shap.LinearExplainer(model, background_data)
                        
        except Exception as e:
            self.logger.error(f"Error updating models: {e}")

    def explain_prediction(self, features: pd.DataFrame, prediction: Dict[str, Any], 
                          models: Dict[str, Any] = None) -> ExplanationResult:
        """Generate comprehensive explanation for a prediction"""
        start_time = datetime.now()
        
        try:
            # Extract prediction details
            action = prediction.get('direction', 'neutral')
            confidence = prediction.get('confidence', 0.0)
            price_prediction = prediction.get('price_prediction', 0.0)
            
            # Generate feature importance
            feature_importance = self._calculate_feature_importance(features, models)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(features, prediction, feature_importance)
            
            # Calculate model contributions
            model_contributions = self._calculate_model_contributions(prediction, models)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(features, prediction)
            
            # Create explanation result
            explanation = ExplanationResult(
                action=action,
                confidence=confidence,
                feature_importance=feature_importance,
                reasoning=reasoning,
                model_contributions=model_contributions,
                risk_factors=risk_factors,
                timestamp=datetime.now(),
                metadata={
                    'price_prediction': price_prediction,
                    'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
                }
            )
            
            # Store in history
            self._store_explanation(explanation)
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            # Return fallback explanation
            return self._create_fallback_explanation(prediction)

    def _calculate_feature_importance(self, features: pd.DataFrame, 
                                    models: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate feature importance using SHAP values"""
        try:
            if features.empty or len(features.columns) == 0:
                return {}
            
            # Use the first available explainer
            explainer = next(iter(self.explainers.values()), None)
            if explainer is None:
                return self._calculate_simple_importance(features)
            
            # Calculate SHAP values
            feature_array = features.values
            if len(feature_array.shape) == 1:
                feature_array = feature_array.reshape(1, -1)
            
            # Ensure we have the right number of features
            if feature_array.shape[1] != len(self.feature_names):
                # Use available features
                available_features = features.columns.tolist()
                return {feat: abs(features[feat].iloc[0]) for feat in available_features}
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(feature_array)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Get feature importance
            importance = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_names[:len(importance)]):
                feature_importance[feature] = float(importance[i])
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return self._calculate_simple_importance(features)

    def _calculate_simple_importance(self, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate simple feature importance using correlation"""
        try:
            importance = {}
            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    # Use absolute value as simple importance
                    importance[col] = abs(features[col].iloc[0])
                else:
                    importance[col] = 0.0
            return importance
        except Exception as e:
            self.logger.error(f"Error in simple importance calculation: {e}")
            return {}

    def _generate_reasoning(self, features: pd.DataFrame, prediction: Dict[str, Any], 
                           feature_importance: Dict[str, float]) -> List[str]:
        """Generate human-readable reasoning for the prediction"""
        reasoning = []
        
        try:
            # Get top features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Generate reasoning based on top features
            for feature, importance in top_features:
                if importance > 0.1:  # Only include significant features
                    feature_value = features[feature].iloc[0] if feature in features.columns else 0
                    
                    # Generate feature-specific reasoning
                    if 'rsi' in feature.lower():
                        if feature_value > 70:
                            reasoning.append(f"RSI is overbought ({feature_value:.2f})")
                        elif feature_value < 30:
                            reasoning.append(f"RSI is oversold ({feature_value:.2f})")
                    elif 'macd' in feature.lower():
                        if feature_value > 0:
                            reasoning.append(f"MACD is bullish ({feature_value:.4f})")
                        else:
                            reasoning.append(f"MACD is bearish ({feature_value:.4f})")
                    elif 'volume' in feature.lower():
                        if feature_value > 1.5:
                            reasoning.append(f"High volume activity ({feature_value:.2f}x average)")
                        elif feature_value < 0.5:
                            reasoning.append(f"Low volume activity ({feature_value:.2f}x average)")
                    elif 'volatility' in feature.lower():
                        if feature_value > 0.05:
                            reasoning.append(f"High volatility detected ({feature_value:.4f})")
                        else:
                            reasoning.append(f"Low volatility environment ({feature_value:.4f})")
                    else:
                        reasoning.append(f"{feature}: {feature_value:.4f} (importance: {importance:.3f})")
            
            # Add confidence-based reasoning
            confidence = prediction.get('confidence', 0)
            if confidence > 0.8:
                reasoning.append(f"High confidence prediction ({confidence:.1%})")
            elif confidence > 0.6:
                reasoning.append(f"Moderate confidence prediction ({confidence:.1%})")
            else:
                reasoning.append(f"Low confidence prediction ({confidence:.1%})")
            
            # Add direction reasoning
            direction = prediction.get('direction', 'neutral')
            if direction == 'buy':
                reasoning.append("Market conditions favor buying")
            elif direction == 'sell':
                reasoning.append("Market conditions favor selling")
            else:
                reasoning.append("Market conditions are neutral")
                
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {e}")
            reasoning.append("Unable to generate detailed reasoning")
        
        return reasoning

    def _calculate_model_contributions(self, prediction: Dict[str, Any], 
                                     models: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate contribution of each model to the final prediction"""
        try:
            contributions = {}
            
            # Extract ensemble weights if available
            ensemble_weights = prediction.get('ensemble_weights', {})
            if ensemble_weights:
                for model_name, weight in ensemble_weights.items():
                    contributions[model_name] = float(weight)
            else:
                # Default equal contributions
                model_names = ['lightgbm', 'xgboost', 'neural_network', 'rl_agent']
                for model_name in model_names:
                    contributions[model_name] = 0.25
                    
            return contributions
            
        except Exception as e:
            self.logger.error(f"Error calculating model contributions: {e}")
            return {'ensemble': 1.0}

    def _identify_risk_factors(self, features: pd.DataFrame, 
                              prediction: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []
        
        try:
            # Check for high volatility
            volatility_features = [col for col in features.columns if 'volatility' in col.lower()]
            for vol_feat in volatility_features:
                vol_value = features[vol_feat].iloc[0]
                if vol_value > 0.1:
                    risk_factors.append(f"High volatility: {vol_feat} = {vol_value:.4f}")
            
            # Check for low volume
            volume_features = [col for col in features.columns if 'volume' in col.lower()]
            for vol_feat in volume_features:
                vol_value = features[vol_feat].iloc[0]
                if vol_value < 0.3:
                    risk_factors.append(f"Low volume: {vol_feat} = {vol_value:.2f}")
            
            # Check for extreme RSI
            rsi_features = [col for col in features.columns if 'rsi' in col.lower()]
            for rsi_feat in rsi_features:
                rsi_value = features[rsi_feat].iloc[0]
                if rsi_value > 80 or rsi_value < 20:
                    risk_factors.append(f"Extreme RSI: {rsi_feat} = {rsi_value:.2f}")
            
            # Check confidence
            confidence = prediction.get('confidence', 0)
            if confidence < 0.5:
                risk_factors.append(f"Low prediction confidence: {confidence:.1%}")
                
        except Exception as e:
            self.logger.error(f"Error identifying risk factors: {e}")
            risk_factors.append("Unable to assess risk factors")
        
        return risk_factors

    def _create_fallback_explanation(self, prediction: Dict[str, Any]) -> ExplanationResult:
        """Create a fallback explanation when detailed analysis fails"""
        return ExplanationResult(
            action=prediction.get('direction', 'neutral'),
            confidence=prediction.get('confidence', 0.0),
            feature_importance={},
            reasoning=["Fallback explanation - detailed analysis unavailable"],
            model_contributions={'ensemble': 1.0},
            risk_factors=["Risk assessment unavailable"],
            timestamp=datetime.now(),
            metadata={'fallback': True}
        )

    def _store_explanation(self, explanation: ExplanationResult):
        """Store explanation in history"""
        try:
            self.explanation_history.append(asdict(explanation))
            
            # Limit history size
            if len(self.explanation_history) > self.max_history_size:
                self.explanation_history = self.explanation_history[-self.max_history_size:]
                
        except Exception as e:
            self.logger.error(f"Error storing explanation: {e}")

    def get_explanation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent explanations"""
        try:
            cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
            
            recent_explanations = [
                exp for exp in self.explanation_history
                if pd.to_datetime(exp['timestamp']) > cutoff_time
            ]
            
            if not recent_explanations:
                return {"message": "No recent explanations available"}
            
            # Calculate statistics
            actions = [exp['action'] for exp in recent_explanations]
            confidences = [exp['confidence'] for exp in recent_explanations]
            
            # Aggregate feature importance
            feature_importance_agg = {}
            for exp in recent_explanations:
                for feature, importance in exp['feature_importance'].items():
                    if feature not in feature_importance_agg:
                        feature_importance_agg[feature] = []
                    feature_importance_agg[feature].append(importance)
            
            # Calculate average importance
            avg_feature_importance = {
                feature: np.mean(values) 
                for feature, values in feature_importance_agg.items()
            }
            
            return {
                'total_explanations': len(recent_explanations),
                'action_distribution': pd.Series(actions).value_counts().to_dict(),
                'average_confidence': np.mean(confidences),
                'top_features': dict(sorted(avg_feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True)[:10]),
                'time_period_hours': hours
            }
            
        except Exception as e:
            self.logger.error(f"Error generating explanation summary: {e}")
            return {"error": str(e)}

    def save_explanations(self, filepath: str):
        """Save explanation history to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.explanation_history, f, indent=2, default=str)
            self.logger.info(f"Explanations saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving explanations: {e}")

    def load_explanations(self, filepath: str):
        """Load explanation history from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.explanation_history = json.load(f)
                self.logger.info(f"Explanations loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading explanations: {e}")

# Legacy XaiLayer class for backward compatibility
class XaiLayer:
    """
    Legacy XAI Layer for backward compatibility
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.advanced_system = AdvancedExplainabilitySystem(config or {})

    def generate_explanation(self, action: str, reasons: Dict[str, str]) -> str:
        """Legacy method for backward compatibility"""
        return self.advanced_system._create_fallback_explanation({
            'direction': action,
            'confidence': 0.5
        }).reasoning[0]
