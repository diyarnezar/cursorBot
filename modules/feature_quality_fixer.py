#!/usr/bin/env python3
"""
FEATURE QUALITY FIXER
=====================

This module fixes common feature quality issues:
1. NaN values and infinite values
2. Excessive zeros
3. Low variance features
4. Highly correlated features
5. Feature scaling issues
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureQualityFixer:
    """Comprehensive feature quality improvement"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = None
        self.feature_importance = {}
        self.removed_features = []
        self.fixed_features = {}
        
    def fix_all_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all quality fixes to the dataframe"""
        self.logger.info(f"🔧 Starting comprehensive feature quality fix...")
        self.logger.info(f"   Initial shape: {df.shape}")
        
        original_shape = df.shape
        
        # Step 1: Fix NaN and infinite values
        df = self._fix_nan_and_infinite(df)
        
        # Step 2: Fix excessive zeros
        df = self._fix_excessive_zeros(df)
        
        # Step 3: Remove low variance features
        df = self._remove_low_variance_features(df)
        
        # Step 4: Fix highly correlated features
        df = self._fix_highly_correlated_features(df)
        
        # Step 5: Improve feature scaling
        df = self._improve_feature_scaling(df)
        
        # Step 6: Add feature quality indicators
        df = self._add_quality_indicators(df)
        
        self.logger.info(f"✅ Feature quality fix completed")
        self.logger.info(f"   Final shape: {df.shape}")
        self.logger.info(f"   Removed features: {len(self.removed_features)}")
        self.logger.info(f"   Fixed features: {len(self.fixed_features)}")
        
        return df
    
    def _fix_nan_and_infinite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix NaN and infinite values with intelligent imputation"""
        self.logger.info(f"   🔧 Fixing NaN and infinite values...")
        
        # Count issues before fixing
        nan_count = df.isna().sum().sum()
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        if nan_count > 0 or inf_count > 0:
            self.logger.info(f"   Found {nan_count} NaN values and {inf_count} infinite values")
        
        # Create a copy to avoid modifying original
        df_fixed = df.copy()
        
        # Handle numeric columns
        numeric_cols = df_fixed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_fixed[col].isna().any() or np.isinf(df_fixed[col]).any():
                # Get the column data
                col_data = df_fixed[col].copy()
                
                # Replace infinite values with NaN first
                col_data = col_data.replace([np.inf, -np.inf], np.nan)
                
                # Calculate statistics for imputation
                mean_val = col_data.mean()
                median_val = col_data.median()
                std_val = col_data.std()
                
                # Use different imputation strategies based on data characteristics
                if std_val == 0 or pd.isna(std_val):
                    # Constant column, use the constant value
                    fill_value = mean_val if not pd.isna(mean_val) else 0
                elif abs(col_data.skew()) > 1:
                    # Skewed data, use median
                    fill_value = median_val if not pd.isna(median_val) else mean_val
                else:
                    # Normal distribution, use mean
                    fill_value = mean_val if not pd.isna(mean_val) else 0
                
                # Fill NaN values
                col_data = col_data.fillna(fill_value)
                
                # Update the dataframe
                df_fixed[col] = col_data
                
                # Track what we fixed
                self.fixed_features[col] = {
                    'nan_count': df[col].isna().sum(),
                    'inf_count': np.isinf(df[col]).sum(),
                    'imputation_method': 'median' if abs(df[col].skew()) > 1 else 'mean',
                    'fill_value': fill_value
                }
        
        # Handle non-numeric columns (categorical)
        categorical_cols = df_fixed.select_dtypes(exclude=[np.number]).columns
        
        for col in categorical_cols:
            if df_fixed[col].isna().any():
                # For categorical, use mode (most frequent value)
                mode_val = df_fixed[col].mode()
                fill_value = mode_val.iloc[0] if len(mode_val) > 0 else 'unknown'
                df_fixed[col] = df_fixed[col].fillna(fill_value)
                
                self.fixed_features[col] = {
                    'nan_count': df[col].isna().sum(),
                    'imputation_method': 'mode',
                    'fill_value': fill_value
                }
        
        return df_fixed
    
    def _fix_excessive_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix features with excessive zeros by adding small noise or removing them"""
        self.logger.info(f"   🔧 Fixing excessive zeros...")
        
        df_fixed = df.copy()
        numeric_cols = df_fixed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            zero_ratio = (df_fixed[col] == 0).mean()
            
            if zero_ratio > 0.8:  # More than 80% zeros
                self.logger.info(f"   Feature '{col}' has {zero_ratio:.1%} zeros")
                
                # Check if this is a meaningful feature
                if col in ['hour_of_day', 'day_of_week', 'maker_fee_advantage', 'optimal_maker_timing']:
                    # These are categorical features encoded as numbers, keep them
                    continue
                
                # For other features with excessive zeros, add small noise
                if zero_ratio > 0.95:  # More than 95% zeros, remove the feature
                    self.logger.info(f"   Removing feature '{col}' (too many zeros)")
                    df_fixed = df_fixed.drop(columns=[col])
                    self.removed_features.append(col)
                else:
                    # Add small noise to break up the zeros
                    noise = np.random.normal(0, 1e-6, len(df_fixed))
                    df_fixed[col] = df_fixed[col] + noise
                    
                    self.fixed_features[col] = {
                        'zero_ratio': zero_ratio,
                        'action': 'added_noise'
                    }
        
        return df_fixed
    
    def _remove_low_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with very low variance"""
        self.logger.info(f"   🔧 Removing low variance features...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            variance = df[col].var()
            
            if variance < 1e-8:  # Very low variance
                self.logger.info(f"   Removing feature '{col}' (variance: {variance:.2e})")
                df = df.drop(columns=[col])
                self.removed_features.append(col)
        
        return df
    
    def _fix_highly_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features to reduce redundancy"""
        self.logger.info(f"   🔧 Fixing highly correlated features...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return df
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:  # Very high correlation
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        
        for col1, col2, corr in high_corr_pairs:
            # Keep the feature with higher variance (more information)
            var1 = df[col1].var()
            var2 = df[col2].var()
            
            if var1 >= var2:
                features_to_remove.add(col2)
            else:
                features_to_remove.add(col1)
        
        # Remove the selected features
        for col in features_to_remove:
            if col in df.columns:
                self.logger.info(f"   Removing highly correlated feature '{col}'")
                df = df.drop(columns=[col])
                self.removed_features.append(col)
        
        return df
    
    def _improve_feature_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Improve feature scaling for better model performance"""
        self.logger.info(f"   🔧 Improving feature scaling...")
        
        df_scaled = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Skip timestamp and target columns
        exclude_cols = ['timestamp', 'target', 'target_1m', 'target_5m', 'target_15m', 
                       'target_30m', 'target_1h', 'target_4h', 'target_1d']
        
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(scale_cols) > 0:
            # Use robust scaling to handle outliers
            scaler = RobustScaler()
            df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
            self.scaler = scaler
            
            self.logger.info(f"   Applied robust scaling to {len(scale_cols)} features")
        
        return df_scaled
    
    def _add_quality_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality indicators to track feature health"""
        self.logger.info(f"   🔧 Adding quality indicators...")
        
        df_quality = df.copy()
        
        # Add feature quality score
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Calculate quality metrics
            variance = df[col].var()
            zero_ratio = (df[col] == 0).mean()
            unique_ratio = df[col].nunique() / len(df[col])
            
            # Create quality score (0-1, higher is better)
            quality_score = (
                min(1.0, variance * 100) * 0.4 +  # Variance component
                (1 - zero_ratio) * 0.3 +          # Non-zero component
                unique_ratio * 0.3                 # Uniqueness component
            )
            
            # Add quality indicator
            df_quality[f'{col}_quality'] = quality_score
        
        # Add overall dataset quality score
        overall_quality = df_quality[[col for col in df_quality.columns if col.endswith('_quality')]].mean(axis=1)
        df_quality['dataset_quality'] = overall_quality
        
        self.logger.info(f"   Added quality indicators for {len(numeric_cols)} features")
        
        return df_quality
    
    def get_quality_report(self) -> Dict:
        """Generate a comprehensive quality report"""
        return {
            'removed_features': self.removed_features,
            'fixed_features': self.fixed_features,
            'total_removed': len(self.removed_features),
            'total_fixed': len(self.fixed_features),
            'scaler_applied': self.scaler is not None
        }
    
    def apply_to_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same fixes to new data (for consistency)"""
        self.logger.info(f"🔧 Applying quality fixes to new data...")
        
        # Apply the same transformations
        df = self._fix_nan_and_infinite(df)
        df = self._fix_excessive_zeros(df)
        df = self._remove_low_variance_features(df)
        df = self._fix_highly_correlated_features(df)
        
        # Apply scaling if we have a fitted scaler
        if self.scaler is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['timestamp', 'target', 'target_1m', 'target_5m', 'target_15m', 
                           'target_30m', 'target_1h', 'target_4h', 'target_1d']
            scale_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if len(scale_cols) > 0:
                df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        return df

# Global instance
feature_fixer = FeatureQualityFixer()

def fix_feature_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Apply comprehensive feature quality fixes"""
    return feature_fixer.fix_all_quality_issues(df)

def get_quality_report() -> Dict:
    """Get the quality improvement report"""
    return feature_fixer.get_quality_report()

def apply_quality_fixes_to_new_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality fixes to new data consistently"""
    return feature_fixer.apply_to_new_data(df) 