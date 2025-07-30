"""
🔍 SHAP Analyzer Module

This module implements SHAP (SHapley Additive exPlanations) analysis
for model explainability and feature importance analysis.

Author: Hyperion Trading System
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
import shap
from shap import TreeExplainer, DeepExplainer, KernelExplainer, LinearExplainer
from shap.plots import waterfall, force, beeswarm, bar, heatmap, scatter

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    """
    🔍 SHAP Analysis System
    
    Implements comprehensive SHAP analysis for model explainability
    and feature importance analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SHAP analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.explainers = {}
        self.shap_values = {}
        self.feature_importance = {}
        self.analysis_results = {}
        
        # SHAP analysis parameters
        self.shap_params = {
            'background_samples': 100,
            'nsamples': 100,
            'l1_reg': 'auto',
            'feature_names': None,
            'max_display': 20,
            'plot_type': 'bar'
        }
        
        # Analysis types
        self.analysis_types = {
            'global': ['summary', 'bar', 'heatmap'],
            'local': ['waterfall', 'force', 'scatter'],
            'interaction': ['dependence', 'interaction'],
            'custom': ['decision', 'partial']
        }
        
        logger.info("🔍 SHAP Analyzer initialized")
    
    def create_explainer(self, model: Any, model_type: str, 
                        background_data: np.ndarray = None) -> Any:
        """Create appropriate SHAP explainer for the model."""
        try:
            explainer = None
            
            if model_type == 'tree':
                # Tree-based models (Random Forest, XGBoost, LightGBM)
                explainer = TreeExplainer(model, background_data)
                
            elif model_type == 'linear':
                # Linear models
                explainer = LinearExplainer(model, background_data)
                
            elif model_type == 'neural':
                # Neural network models
                explainer = DeepExplainer(model, background_data)
                
            elif model_type == 'kernel':
                # Generic kernel explainer for any model
                explainer = KernelExplainer(
                    model.predict, 
                    background_data,
                    nsamples=self.shap_params['nsamples']
                )
                
            else:
                # Default to kernel explainer
                explainer = KernelExplainer(
                    model.predict, 
                    background_data,
                    nsamples=self.shap_params['nsamples']
                )
            
            self.explainers[model_type] = explainer
            logger.info(f"✅ Created {model_type} explainer")
            
            return explainer
            
        except Exception as e:
            logger.error(f"❌ Failed to create {model_type} explainer: {e}")
            return None
    
    def calculate_shap_values(self, model_type: str, data: np.ndarray, 
                            sample_indices: List[int] = None) -> np.ndarray:
        """Calculate SHAP values for the given data."""
        try:
            explainer = self.explainers.get(model_type)
            if explainer is None:
                logger.error(f"❌ No explainer found for {model_type}")
                return None
            
            # Sample data if specified
            if sample_indices is not None:
                data = data[sample_indices]
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(data)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first output for regression
            
            # Store SHAP values
            self.shap_values[model_type] = shap_values
            
            logger.info(f"✅ Calculated SHAP values for {len(data)} samples")
            return shap_values
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate SHAP values: {e}")
            return None
    
    def analyze_global_importance(self, model_type: str, feature_names: List[str] = None) -> Dict[str, Any]:
        """Analyze global feature importance using SHAP values."""
        try:
            shap_values = self.shap_values.get(model_type)
            if shap_values is None:
                logger.error(f"❌ No SHAP values found for {model_type}")
                return {}
            
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature importance dataframe
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(shap_values.shape[1])]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap,
                'std': np.std(np.abs(shap_values), axis=0)
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Store results
            self.feature_importance[model_type] = importance_df
            
            # Calculate additional metrics
            total_importance = importance_df['importance'].sum()
            cumulative_importance = importance_df['importance'].cumsum() / total_importance
            
            analysis_results = {
                'feature_importance': importance_df.to_dict('records'),
                'total_importance': total_importance,
                'top_features': importance_df.head(10).to_dict('records'),
                'cumulative_importance': cumulative_importance.tolist(),
                'importance_thresholds': {
                    'top_10_percent': cumulative_importance[cumulative_importance <= 0.1].index.tolist(),
                    'top_25_percent': cumulative_importance[cumulative_importance <= 0.25].index.tolist(),
                    'top_50_percent': cumulative_importance[cumulative_importance <= 0.5].index.tolist()
                }
            }
            
            self.analysis_results[f'{model_type}_global'] = analysis_results
            
            logger.info(f"✅ Global importance analysis completed for {model_type}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze global importance: {e}")
            return {}
    
    def analyze_local_importance(self, model_type: str, sample_index: int, 
                               feature_names: List[str] = None) -> Dict[str, Any]:
        """Analyze local feature importance for a specific sample."""
        try:
            shap_values = self.shap_values.get(model_type)
            if shap_values is None:
                logger.error(f"❌ No SHAP values found for {model_type}")
                return {}
            
            if sample_index >= len(shap_values):
                logger.error(f"❌ Sample index {sample_index} out of range")
                return {}
            
            # Get SHAP values for the specific sample
            sample_shap = shap_values[sample_index]
            
            # Create feature contribution dataframe
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(sample_shap))]
            
            contribution_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': sample_shap,
                'abs_shap': np.abs(sample_shap)
            })
            
            # Sort by absolute SHAP value
            contribution_df = contribution_df.sort_values('abs_shap', ascending=False)
            
            # Calculate prediction components
            base_value = self.explainers[model_type].expected_value
            prediction = base_value + sample_shap.sum()
            
            analysis_results = {
                'sample_index': sample_index,
                'base_value': base_value,
                'prediction': prediction,
                'feature_contributions': contribution_df.to_dict('records'),
                'top_contributors': contribution_df.head(10).to_dict('records'),
                'positive_contributors': contribution_df[contribution_df['shap_value'] > 0].to_dict('records'),
                'negative_contributors': contribution_df[contribution_df['shap_value'] < 0].to_dict('records')
            }
            
            self.analysis_results[f'{model_type}_local_{sample_index}'] = analysis_results
            
            logger.info(f"✅ Local importance analysis completed for sample {sample_index}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze local importance: {e}")
            return {}
    
    def analyze_feature_interactions(self, model_type: str, feature_names: List[str] = None) -> Dict[str, Any]:
        """Analyze feature interactions using SHAP."""
        try:
            explainer = self.explainers.get(model_type)
            if explainer is None:
                logger.error(f"❌ No explainer found for {model_type}")
                return {}
            
            # Get background data
            background_data = explainer.background_data
            
            # Calculate interaction values (for tree-based models)
            if hasattr(explainer, 'shap_interaction_values'):
                interaction_values = explainer.shap_interaction_values(background_data[:100])
                
                # Calculate interaction importance
                interaction_importance = np.mean(np.abs(interaction_values), axis=0)
                
                # Create interaction matrix
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(interaction_importance.shape[0])]
                
                interaction_df = pd.DataFrame(
                    interaction_importance,
                    index=feature_names,
                    columns=feature_names
                )
                
                # Find top interactions
                interactions = []
                for i in range(len(feature_names)):
                    for j in range(i + 1, len(feature_names)):
                        interactions.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'interaction_strength': interaction_importance[i, j]
                        })
                
                # Sort by interaction strength
                interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
                
                analysis_results = {
                    'interaction_matrix': interaction_df.to_dict(),
                    'top_interactions': interactions[:20],
                    'feature_names': feature_names
                }
                
                self.analysis_results[f'{model_type}_interactions'] = analysis_results
                
                logger.info(f"✅ Feature interaction analysis completed for {model_type}")
                return analysis_results
            
            else:
                logger.warning(f"⚠️ Interaction analysis not available for {model_type}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to analyze feature interactions: {e}")
            return {}
    
    def create_waterfall_plot(self, model_type: str, sample_index: int, 
                            feature_names: List[str] = None, save_path: str = None):
        """Create waterfall plot for a specific sample."""
        try:
            shap_values = self.shap_values.get(model_type)
            if shap_values is None:
                logger.error(f"❌ No SHAP values found for {model_type}")
                return
            
            if sample_index >= len(shap_values):
                logger.error(f"❌ Sample index {sample_index} out of range")
                return
            
            # Get SHAP values for the sample
            sample_shap = shap_values[sample_index]
            
            # Create waterfall plot
            plt.figure(figsize=(10, 8))
            waterfall(sample_shap, max_display=self.shap_params['max_display'])
            plt.title(f'SHAP Waterfall Plot - Sample {sample_index}')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"💾 Waterfall plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Failed to create waterfall plot: {e}")
    
    def create_summary_plot(self, model_type: str, feature_names: List[str] = None, 
                          save_path: str = None):
        """Create summary plot for global feature importance."""
        try:
            shap_values = self.shap_values.get(model_type)
            if shap_values is None:
                logger.error(f"❌ No SHAP values found for {model_type}")
                return
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            beeswarm(shap_values, feature_names=feature_names, max_display=self.shap_params['max_display'])
            plt.title(f'SHAP Summary Plot - {model_type}')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"💾 Summary plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Failed to create summary plot: {e}")
    
    def create_force_plot(self, model_type: str, sample_index: int, 
                         feature_names: List[str] = None, save_path: str = None):
        """Create force plot for a specific sample."""
        try:
            shap_values = self.shap_values.get(model_type)
            if shap_values is None:
                logger.error(f"❌ No SHAP values found for {model_type}")
                return
            
            if sample_index >= len(shap_values):
                logger.error(f"❌ Sample index {sample_index} out of range")
                return
            
            # Get SHAP values for the sample
            sample_shap = shap_values[sample_index]
            
            # Create force plot
            plt.figure(figsize=(12, 6))
            force(sample_shap, feature_names=feature_names)
            plt.title(f'SHAP Force Plot - Sample {sample_index}')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"💾 Force plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Failed to create force plot: {e}")
    
    def create_bar_plot(self, model_type: str, save_path: str = None):
        """Create bar plot of feature importance."""
        try:
            importance_df = self.feature_importance.get(model_type)
            if importance_df is None:
                logger.error(f"❌ No feature importance found for {model_type}")
                return
            
            # Create bar plot
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('SHAP Importance')
            plt.title(f'Feature Importance - {model_type}')
            plt.gca().invert_yaxis()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"💾 Bar plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Failed to create bar plot: {e}")
    
    def generate_explainability_report(self, model_type: str, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive explainability report."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'analysis_summary': {},
                'feature_importance': {},
                'sample_analyses': {},
                'recommendations': []
            }
            
            # Global importance analysis
            global_analysis = self.analyze_global_importance(model_type)
            report['analysis_summary']['global_importance'] = global_analysis
            
            # Feature importance summary
            if model_type in self.feature_importance:
                importance_df = self.feature_importance[model_type]
                report['feature_importance'] = {
                    'top_features': importance_df.head(10).to_dict('records'),
                    'total_features': len(importance_df),
                    'importance_distribution': {
                        'mean': importance_df['importance'].mean(),
                        'std': importance_df['importance'].std(),
                        'max': importance_df['importance'].max(),
                        'min': importance_df['importance'].min()
                    }
                }
            
            # Sample analyses (top 5 samples)
            shap_values = self.shap_values.get(model_type)
            if shap_values is not None:
                for i in range(min(5, len(shap_values))):
                    sample_analysis = self.analyze_local_importance(model_type, i)
                    report['sample_analyses'][f'sample_{i}'] = sample_analysis
            
            # Generate recommendations
            recommendations = self._generate_recommendations(model_type)
            report['recommendations'] = recommendations
            
            # Save report
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"💾 Explainability report saved to {output_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate explainability report: {e}")
            return {}
    
    def _generate_recommendations(self, model_type: str) -> List[str]:
        """Generate recommendations based on SHAP analysis."""
        try:
            recommendations = []
            
            # Get feature importance
            importance_df = self.feature_importance.get(model_type)
            if importance_df is None:
                return recommendations
            
            # Analyze feature importance distribution
            top_features = importance_df.head(5)
            low_importance_features = importance_df[importance_df['importance'] < importance_df['importance'].quantile(0.1)]
            
            # Generate recommendations
            if len(top_features) > 0:
                recommendations.append(f"Focus on top features: {', '.join(top_features['feature'].tolist())}")
            
            if len(low_importance_features) > 0:
                recommendations.append(f"Consider removing low-importance features: {len(low_importance_features)} features")
            
            # Feature engineering recommendations
            high_std_features = importance_df[importance_df['std'] > importance_df['std'].quantile(0.9)]
            if len(high_std_features) > 0:
                recommendations.append(f"High variance features detected: {len(high_std_features)} features may need stabilization")
            
            # Model-specific recommendations
            if model_type == 'tree':
                recommendations.append("Tree-based model detected - consider feature interactions")
            elif model_type == 'linear':
                recommendations.append("Linear model detected - check for multicollinearity")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate recommendations: {e}")
            return []
    
    def get_analyzer_summary(self) -> Dict[str, Any]:
        """Get a summary of SHAP analysis activities."""
        return {
            'total_models': len(self.explainers),
            'model_types': list(self.explainers.keys()),
            'total_analyses': len(self.analysis_results),
            'feature_importance_models': list(self.feature_importance.keys()),
            'shap_params': self.shap_params,
            'analysis_types': self.analysis_types
        }
    
    def save_analysis_state(self, filepath: str):
        """Save analysis state to file."""
        try:
            import pickle
            
            analysis_state = {
                'explainers': self.explainers,
                'shap_values': self.shap_values,
                'feature_importance': self.feature_importance,
                'analysis_results': self.analysis_results,
                'shap_params': self.shap_params
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(analysis_state, f)
            
            logger.info(f"💾 Analysis state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save analysis state: {e}")
    
    def load_analysis_state(self, filepath: str):
        """Load analysis state from file."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                analysis_state = pickle.load(f)
            
            self.explainers = analysis_state['explainers']
            self.shap_values = analysis_state['shap_values']
            self.feature_importance = analysis_state['feature_importance']
            self.analysis_results = analysis_state['analysis_results']
            self.shap_params = analysis_state['shap_params']
            
            logger.info(f"📂 Analysis state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load analysis state: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'shap_analysis_enabled': True,
        'background_samples': 100,
        'nsamples': 100
    }
    
    # Initialize SHAP analyzer
    analyzer = SHAPAnalyzer(config)
    
    # Create sample model and data
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    
    # Train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = analyzer.create_explainer(model, 'tree', X[:100])
    
    # Calculate SHAP values
    shap_values = analyzer.calculate_shap_values('tree', X[:100])
    
    if shap_values is not None:
        # Analyze global importance
        global_analysis = analyzer.analyze_global_importance('tree')
        
        # Analyze local importance
        local_analysis = analyzer.analyze_local_importance('tree', 0)
        
        # Generate report
        report = analyzer.generate_explainability_report('tree')
        
        print(f"SHAP analysis completed. Top features: {global_analysis.get('top_features', [])}")
    
    # Get analyzer summary
    summary = analyzer.get_analyzer_summary()
    print(f"SHAP analyzer initialized with {summary['total_models']} models") 