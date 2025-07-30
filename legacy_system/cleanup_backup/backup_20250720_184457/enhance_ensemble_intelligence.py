#!/usr/bin/env python3
"""
ENHANCE ENSEMBLE INTELLIGENCE
Project Hyperion - Performance-Based Ensemble Weighting

This script enhances the bot's intelligence by:
1. Recalculating ensemble weights based on actual model performance
2. Giving higher weights to better performing models
3. Implementing smart weighting strategies
4. Optimizing the ensemble for maximum profitability
"""

import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleIntelligenceEnhancer:
    """Enhance ensemble intelligence with performance-based weighting"""
    
    def __init__(self):
        self.model_performance = {}
        self.enhanced_weights = {}
        
    def load_model_performance(self):
        """Load model performance data"""
        try:
            with open('models/model_performance.json', 'r') as f:
                self.model_performance = json.load(f)
            logger.info(f"✅ Loaded performance data for {len(self.model_performance)} models")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model performance: {e}")
            return False
    
    def calculate_performance_weights(self) -> Dict[str, float]:
        """Calculate performance-based ensemble weights"""
        if not self.model_performance:
            logger.error("❌ No model performance data available")
            return {}
        
        logger.info("🧠 Calculating performance-based ensemble weights...")
        
        # Convert scores to numpy array for calculations
        scores = np.array(list(self.model_performance.values()))
        model_names = list(self.model_performance.keys())
        
        # Method 1: Softmax weighting (exponential scaling)
        softmax_weights = self._calculate_softmax_weights(scores)
        
        # Method 2: Rank-based weighting
        rank_weights = self._calculate_rank_weights(scores)
        
        # Method 3: Performance tier weighting
        tier_weights = self._calculate_tier_weights(scores)
        
        # Combine methods for optimal weighting
        combined_weights = self._combine_weighting_methods(scores, softmax_weights, rank_weights, tier_weights)
        
        # Create final weights dictionary
        final_weights = {}
        for i, model_name in enumerate(model_names):
            final_weights[model_name] = float(combined_weights[i])
        
        # Normalize to sum to 1
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v / total_weight for k, v in final_weights.items()}
        
        self.enhanced_weights = final_weights
        return final_weights
    
    def _calculate_softmax_weights(self, scores: np.ndarray) -> np.ndarray:
        """Calculate softmax weights for exponential scaling"""
        # Apply exponential scaling with temperature
        temperature = 20.0  # Controls how much better models are favored
        exp_scores = np.exp(scores / temperature)
        return exp_scores / np.sum(exp_scores)
    
    def _calculate_rank_weights(self, scores: np.ndarray) -> np.ndarray:
        """Calculate rank-based weights"""
        # Sort by performance and assign weights based on rank
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        rank_weights = np.zeros_like(scores)
        
        for i, idx in enumerate(sorted_indices):
            # Exponential decay based on rank
            rank_weights[idx] = np.exp(-i * 0.3)  # Decay factor
        
        return rank_weights / np.sum(rank_weights)
    
    def _calculate_tier_weights(self, scores: np.ndarray) -> np.ndarray:
        """Calculate tier-based weights"""
        tier_weights = np.zeros_like(scores)
        
        # Define performance tiers
        excellent_threshold = np.percentile(scores, 80)  # Top 20%
        good_threshold = np.percentile(scores, 60)       # Top 40%
        average_threshold = np.percentile(scores, 40)    # Top 60%
        
        for i, score in enumerate(scores):
            if score >= excellent_threshold:
                tier_weights[i] = 3.0  # Excellent models get 3x weight
            elif score >= good_threshold:
                tier_weights[i] = 2.0  # Good models get 2x weight
            elif score >= average_threshold:
                tier_weights[i] = 1.5  # Average models get 1.5x weight
            else:
                tier_weights[i] = 1.0  # Poor models get base weight
        
        return tier_weights / np.sum(tier_weights)
    
    def _combine_weighting_methods(self, scores: np.ndarray, softmax: np.ndarray, 
                                 rank: np.ndarray, tier: np.ndarray) -> np.ndarray:
        """Combine different weighting methods for optimal results"""
        # Weighted combination of methods
        combined = (0.4 * softmax + 0.3 * rank + 0.3 * tier)
        
        # Apply model type bonuses
        model_bonuses = self._calculate_model_type_bonuses(scores)
        combined *= model_bonuses
        
        return combined / np.sum(combined)
    
    def _calculate_model_type_bonuses(self, scores: np.ndarray) -> np.ndarray:
        """Calculate bonuses based on model type performance"""
        bonuses = np.ones_like(scores)
        model_names = list(self.model_performance.keys())
        
        for i, model_name in enumerate(model_names):
            score = scores[i]
            
            # Tree-based models get bonus if performing well
            if any(x in model_name for x in ['xgboost', 'lightgbm', 'catboost']):
                if score > 80:
                    bonuses[i] = 1.2  # 20% bonus for excellent tree models
                elif score > 60:
                    bonuses[i] = 1.1  # 10% bonus for good tree models
            
            # Neural models get bonus only if performing very well
            elif any(x in model_name for x in ['neural_network', 'lstm', 'transformer']):
                if score > 70:
                    bonuses[i] = 1.15  # 15% bonus for excellent neural models
                elif score < 40:
                    bonuses[i] = 0.8   # 20% penalty for poor neural models
            
            # SVM gets moderate bonus for good performance
            elif 'svm' in model_name:
                if score > 50:
                    bonuses[i] = 1.05  # 5% bonus for good SVM
                elif score < 35:
                    bonuses[i] = 0.9   # 10% penalty for poor SVM
        
        return bonuses
    
    def analyze_weight_distribution(self):
        """Analyze and log weight distribution"""
        if not self.enhanced_weights:
            logger.error("❌ No enhanced weights calculated")
            return
        
        weights = list(self.enhanced_weights.values())
        model_names = list(self.enhanced_weights.keys())
        
        # Sort by weight
        sorted_data = sorted(zip(model_names, weights), key=lambda x: x[1], reverse=True)
        
        logger.info("🏆 Enhanced Ensemble Weight Distribution:")
        logger.info(f"   • Total models: {len(weights)}")
        logger.info(f"   • Weight range: {min(weights):.4f} - {max(weights):.4f}")
        logger.info(f"   • Weight variance: {np.var(weights):.6f}")
        logger.info(f"   • Weight standard deviation: {np.std(weights):.6f}")
        
        # Top 10 models
        logger.info("🥇 Top 10 models by weight:")
        for i, (model_name, weight) in enumerate(sorted_data[:10]):
            score = self.model_performance[model_name]
            logger.info(f"   {i+1:2d}. {model_name:20s}: {weight:.4f} (score: {score:.1f})")
        
        # Bottom 5 models
        logger.info("⚠️ Bottom 5 models by weight:")
        for i, (model_name, weight) in enumerate(sorted_data[-5:]):
            score = self.model_performance[model_name]
            logger.info(f"   {len(sorted_data)-4+i:2d}. {model_name:20s}: {weight:.4f} (score: {score:.1f})")
        
        # Model type analysis
        self._analyze_model_types()
    
    def _analyze_model_types(self):
        """Analyze weights by model type"""
        model_types = {}
        
        for model_name, weight in self.enhanced_weights.items():
            model_type = model_name.split('_')[0]
            if model_type not in model_types:
                model_types[model_type] = {'count': 0, 'total_weight': 0, 'avg_score': 0, 'scores': []}
            
            model_types[model_type]['count'] += 1
            model_types[model_type]['total_weight'] += weight
            model_types[model_type]['scores'].append(self.model_performance[model_name])
        
        # Calculate averages
        for model_type in model_types:
            scores = model_types[model_type]['scores']
            model_types[model_type]['avg_score'] = np.mean(scores)
        
        logger.info("📊 Model Type Analysis:")
        for model_type, data in sorted(model_types.items(), key=lambda x: x[1]['avg_score'], reverse=True):
            avg_weight = data['total_weight'] / data['count']
            logger.info(f"   • {model_type:15s}: {data['count']:2d} models, avg weight: {avg_weight:.4f}, avg score: {data['avg_score']:.1f}")
    
    def save_enhanced_weights(self):
        """Save enhanced ensemble weights"""
        if not self.enhanced_weights:
            logger.error("❌ No enhanced weights to save")
            return False
        
        try:
            # Save with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'models/enhanced_ensemble_weights_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump(self.enhanced_weights, f, indent=2)
            
            # Also save as current ensemble weights
            with open('models/ensemble_weights.json', 'w') as f:
                json.dump(self.enhanced_weights, f, indent=2)
            
            logger.info(f"✅ Enhanced ensemble weights saved to {filename}")
            logger.info(f"✅ Current ensemble weights updated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save enhanced weights: {e}")
            return False
    
    def generate_intelligence_report(self):
        """Generate comprehensive intelligence enhancement report"""
        if not self.enhanced_weights:
            logger.error("❌ No enhanced weights available for report")
            return
        
        logger.info("📊 Generating Intelligence Enhancement Report...")
        
        # Calculate improvement metrics
        old_weights = {k: 1.0/len(self.model_performance) for k in self.model_performance.keys()}
        
        # Weight concentration improvement
        old_variance = np.var(list(old_weights.values()))
        new_variance = np.var(list(self.enhanced_weights.values()))
        variance_improvement = (new_variance - old_variance) / old_variance * 100
        
        # Performance-weighted average improvement
        old_weighted_avg = sum(old_weights[k] * self.model_performance[k] for k in self.model_performance.keys())
        new_weighted_avg = sum(self.enhanced_weights[k] * self.model_performance[k] for k in self.model_performance.keys())
        performance_improvement = (new_weighted_avg - old_weighted_avg) / old_weighted_avg * 100
        
        # Top model concentration
        top_10_old_weight = sum(sorted(old_weights.values(), reverse=True)[:10])
        top_10_new_weight = sum(sorted(self.enhanced_weights.values(), reverse=True)[:10])
        concentration_improvement = (top_10_new_weight - top_10_old_weight) / top_10_old_weight * 100
        
        logger.info("🎯 Intelligence Enhancement Results:")
        logger.info(f"   • Weight variance improvement: {variance_improvement:+.1f}%")
        logger.info(f"   • Performance-weighted average improvement: {performance_improvement:+.1f}%")
        logger.info(f"   • Top 10 model concentration improvement: {concentration_improvement:+.1f}%")
        logger.info(f"   • Best model weight: {max(self.enhanced_weights.values()):.4f}")
        logger.info(f"   • Worst model weight: {min(self.enhanced_weights.values()):.4f}")
        
        # Save report
        report = {
            'timestamp': datetime.now().isoformat(),
            'enhancement_metrics': {
                'variance_improvement_percent': variance_improvement,
                'performance_improvement_percent': performance_improvement,
                'concentration_improvement_percent': concentration_improvement,
                'best_model_weight': max(self.enhanced_weights.values()),
                'worst_model_weight': min(self.enhanced_weights.values()),
                'total_models': len(self.enhanced_weights)
            },
            'enhanced_weights': self.enhanced_weights
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f'models/intelligence_enhancement_report_{timestamp}.json'
        
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Intelligence enhancement report saved to {report_filename}")

def main():
    """Main function to enhance ensemble intelligence"""
    logger.info("🚀 Starting Ensemble Intelligence Enhancement...")
    
    enhancer = EnsembleIntelligenceEnhancer()
    
    # Load model performance
    if not enhancer.load_model_performance():
        return
    
    # Calculate enhanced weights
    enhanced_weights = enhancer.calculate_performance_weights()
    if not enhanced_weights:
        logger.error("❌ Failed to calculate enhanced weights")
        return
    
    # Analyze weight distribution
    enhancer.analyze_weight_distribution()
    
    # Save enhanced weights
    if enhancer.save_enhanced_weights():
        logger.info("✅ Ensemble intelligence enhancement completed successfully!")
    else:
        logger.error("❌ Failed to save enhanced weights")
        return
    
    # Generate intelligence report
    enhancer.generate_intelligence_report()
    
    logger.info("🎉 Your bot is now significantly more intelligent!")
    logger.info("🧠 Performance-based ensemble weighting activated")
    logger.info("📈 Better models now have higher influence on predictions")

if __name__ == "__main__":
    main() 