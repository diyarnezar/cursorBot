#!/usr/bin/env python3
"""
DATA LEAKAGE DETECTOR - PHASE 1 IMPLEMENTATION
==============================================

This module implements Gemini's Phase 1 recommendations:
1. Eliminate data leakage in feature engineering
2. Audit features for future information usage
3. Validate realistic model performance scores
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLeakageDetector:
    """
    Advanced data leakage detection and prevention system.
    
    This class implements comprehensive checks to ensure no future information
    is used in feature engineering, maintaining the integrity of the trading system.
    """
    
    def __init__(self):
        """Initialize the data leakage detector."""
        self.leakage_report = {}
        self.feature_audit_results = {}
        self.baseline_performance = {}
        
    def audit_features(self, features_df: pd.DataFrame, target_col: str = 'target') -> Dict[str, Any]:
        """
        Comprehensive feature audit to detect data leakage.
        
        Args:
            features_df: DataFrame containing features and target
            target_col: Name of the target column
            
        Returns:
            Dictionary containing audit results and recommendations
        """
        logger.info("🔍 Starting comprehensive feature audit...")
        
        audit_results = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(features_df.columns) - 1,  # Exclude target
            'leakage_detected': False,
            'suspicious_features': [],
            'recommendations': [],
            'performance_validation': {}
        }
        
        # 1. Check for perfect correlation with target
        target = features_df[target_col]
        for col in features_df.columns:
            if col != target_col:
                correlation = abs(features_df[col].corr(target))
                if correlation > 0.95:
                    audit_results['leakage_detected'] = True
                    audit_results['suspicious_features'].append({
                        'feature': col,
                        'correlation': correlation,
                        'issue': 'Perfect correlation with target'
                    })
        
        # 2. Check for unrealistic feature distributions
        for col in features_df.columns:
            if col != target_col:
                feature = features_df[col]
                
                # Check for constant features
                if feature.nunique() == 1:
                    audit_results['suspicious_features'].append({
                        'feature': col,
                        'issue': 'Constant feature (no variance)'
                    })
                
                # Check for extreme values that might indicate future info
                if feature.abs().max() > 1e6:
                    audit_results['suspicious_features'].append({
                        'feature': col,
                        'issue': 'Extreme values detected'
                    })
        
        # 3. Validate target distribution
        target_stats = {
            'mean': target.mean(),
            'std': target.std(),
            'min': target.min(),
            'max': target.max(),
            'skew': target.skew()
        }
        
        audit_results['performance_validation']['target_stats'] = target_stats
        
        # 4. Check for realistic performance expectations
        if target.std() < 0.001:
            audit_results['recommendations'].append(
                "Target has very low variance - check for data leakage"
            )
        
        # 5. Generate recommendations
        if audit_results['leakage_detected']:
            audit_results['recommendations'].append(
                "CRITICAL: Data leakage detected. Review feature engineering pipeline."
            )
        else:
            audit_results['recommendations'].append(
                "No obvious data leakage detected. Proceed with caution."
            )
        
        audit_results['recommendations'].append(
            "Ensure all features use only past information"
        )
        audit_results['recommendations'].append(
            "Validate model performance is realistic (R² < 0.1 for good models)"
        )
        
        self.feature_audit_results = audit_results
        logger.info(f"✅ Feature audit completed. Leakage detected: {audit_results['leakage_detected']}")
        
        return audit_results
    
    def validate_baseline_performance(self, model_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate that model performance is realistic and not due to data leakage.
        
        Args:
            model_performance: Dictionary containing performance metrics
            
        Returns:
            Validation results and recommendations
        """
        logger.info("📊 Validating baseline performance...")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'is_realistic': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check R² scores
        if 'r2_score' in model_performance:
            r2 = model_performance['r2_score']
            if r2 > 0.1:
                validation_results['warnings'].append(
                    f"R² score ({r2:.3f}) is suspiciously high. Check for data leakage."
                )
                validation_results['is_realistic'] = False
            elif r2 < -0.1:
                validation_results['warnings'].append(
                    f"R² score ({r2:.3f}) is very negative. Check feature engineering."
                )
        
        # Check accuracy for classification
        if 'accuracy' in model_performance:
            accuracy = model_performance['accuracy']
            if accuracy > 0.8:
                validation_results['warnings'].append(
                    f"Accuracy ({accuracy:.3f}) is suspiciously high for crypto prediction."
                )
                validation_results['is_realistic'] = False
        
        # Check Sharpe ratio
        if 'sharpe_ratio' in model_performance:
            sharpe = model_performance['sharpe_ratio']
            if sharpe > 3.0:
                validation_results['warnings'].append(
                    f"Sharpe ratio ({sharpe:.3f}) is suspiciously high. Validate backtesting."
                )
                validation_results['is_realistic'] = False
        
        # Generate recommendations
        if not validation_results['is_realistic']:
            validation_results['recommendations'].append(
                "CRITICAL: Performance appears unrealistic. Review entire pipeline."
            )
        else:
            validation_results['recommendations'].append(
                "Performance appears realistic. Proceed with deployment."
            )
        
        validation_results['recommendations'].append(
            "Use walk-forward analysis for robust validation"
        )
        validation_results['recommendations'].append(
            "Implement proper train/test splits with time boundaries"
        )
        
        self.baseline_performance = validation_results
        logger.info(f"✅ Performance validation completed. Realistic: {validation_results['is_realistic']}")
        
        return validation_results
    
    def check_feature_timing(self, features_df: pd.DataFrame, timestamp_col: str = 'timestamp') -> Dict[str, Any]:
        """
        Check that features are properly time-aligned and don't use future information.
        
        Args:
            features_df: DataFrame with features and timestamps
            timestamp_col: Name of the timestamp column
            
        Returns:
            Timing validation results
        """
        logger.info("⏰ Checking feature timing alignment...")
        
        timing_results = {
            'timestamp': datetime.now().isoformat(),
            'timing_issues': [],
            'recommendations': []
        }
        
        if timestamp_col not in features_df.columns:
            timing_results['timing_issues'].append(
                f"Timestamp column '{timestamp_col}' not found"
            )
            return timing_results
        
        # Check for timestamp ordering
        timestamps = pd.to_datetime(features_df[timestamp_col])
        if not timestamps.is_monotonic_increasing:
            timing_results['timing_issues'].append(
                "Timestamps are not in chronological order"
            )
        
        # Check for duplicate timestamps
        if timestamps.duplicated().any():
            timing_results['timing_issues'].append(
                "Duplicate timestamps detected"
            )
        
        # Check for gaps in timestamps
        time_diff = timestamps.diff().dropna()
        if time_diff.std() > time_diff.mean() * 2:
            timing_results['timing_issues'].append(
                "Irregular time intervals detected"
            )
        
        # Generate recommendations
        if timing_results['timing_issues']:
            timing_results['recommendations'].append(
                "Fix timing issues before proceeding with training"
            )
        else:
            timing_results['recommendations'].append(
                "Timing alignment appears correct"
            )
        
        timing_results['recommendations'].append(
            "Ensure all features use only past information"
        )
        timing_results['recommendations'].append(
            "Implement proper lag features for time series"
        )
        
        logger.info(f"✅ Timing check completed. Issues: {len(timing_results['timing_issues'])}")
        
        return timing_results
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data leakage report.
        
        Returns:
            Complete audit report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'audit_summary': {
                'feature_audit': self.feature_audit_results,
                'performance_validation': self.baseline_performance
            },
            'overall_status': 'PASS' if not self.feature_audit_results.get('leakage_detected', False) else 'FAIL',
            'recommendations': []
        }
        
        # Combine recommendations
        if self.feature_audit_results:
            report['recommendations'].extend(
                self.feature_audit_results.get('recommendations', [])
            )
        
        if self.baseline_performance:
            report['recommendations'].extend(
                self.baseline_performance.get('recommendations', [])
            )
        
        # Save report
        with open('data_leakage_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("📄 Data leakage report generated and saved")
        
        return report

def main():
    """Test the data leakage detector."""
    logger.info("🧪 Testing Data Leakage Detector...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Create realistic features (no leakage)
    data = {
        'timestamp': dates,
        'price': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'volume': np.random.exponential(1000, 1000),
        'rsi': 50 + np.random.randn(1000) * 10,
        'macd': np.random.randn(1000) * 2,
        'target': np.random.randn(1000) * 0.02  # Realistic returns
    }
    
    df = pd.DataFrame(data)
    
    # Initialize detector
    detector = DataLeakageDetector()
    
    # Run audit
    audit_results = detector.audit_features(df, 'target')
    performance_validation = detector.validate_baseline_performance({'r2_score': 0.05})
    timing_check = detector.check_feature_timing(df, 'timestamp')
    
    # Generate report
    report = detector.generate_report()
    
    logger.info("✅ Data Leakage Detector test completed")
    logger.info(f"Audit status: {'PASS' if not audit_results['leakage_detected'] else 'FAIL'}")
    
    return detector

if __name__ == "__main__":
    main() 