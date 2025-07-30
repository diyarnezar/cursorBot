#!/usr/bin/env python3
"
MULTI-ASSET TRAINER - CLUSTER-SPECIFIC MODEL TRAINING
====================================================

This module implements cluster-specific training for all 26 pairs in Gemini's strategy.
Each cluster gets specialized models optimized for its market characteristics.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import threading
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.portfolio_engine import PortfolioEngine, AssetCluster
from modules.data_ingestion import fetch_klines, fetch_ticker_24hr
from modules.feature_engineering import EnhancedFeatureEngineer
from modules.alternative_data import EnhancedAlternativeData

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiAssetTrainer:
   ulti-Asset Trainer for cluster-specific model training.
    
    Features:
    - Cluster-specific feature engineering
    - Specialized hyperparameter optimization
    - Market regime adaptation
    - Continuous learning per cluster
    - Performance tracking by cluster
     
    def __init__(self, config_path: str = 'config.json'):
       itialize the multi-asset trainer"""
        self.config = self.load_config(config_path)
        
        # Initialize portfolio engine for cluster information
        self.portfolio_engine = PortfolioEngine(config_path)
        
        # Initialize components
        self.feature_engineer = EnhancedFeatureEngineer()
        self.alternative_data = EnhancedAlternativeData()
        
        # Cluster-specific models and performance
        self.cluster_models = {cluster:[object Object]r cluster in AssetCluster}
        self.cluster_performance = {cluster:[object Object]r cluster in AssetCluster}
        self.cluster_features = {cluster:[object Object]r cluster in AssetCluster}
        
        # Training configuration
        self.training_config = {
           data_days': 15 # Days of data to collect
            timeframes': ['1m', '5m', '15m'],  # Prediction timeframes
        models_per_cluster': ['lightgbm, gboost', 'catboost'],
           optimization_trials:50
         cv_folds': 5,
            min_samples':10,
            max_samples': 50000
        }
        
        # Create models directory
        self.models_dir = Path('models/cluster_models')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🧠 Multi-Asset Trainer initialized")
        logger.info(f   Clusters:[object Object]len(AssetCluster)}")
        logger.info(f"   Total assets: {len(self.portfolio_engine.asset_universe)}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def collect_cluster_data(self, cluster: AssetCluster) -> Dict[str, pd.DataFrame]:
     llect data for all assets in a cluster       logger.info(f"📊 Collecting data for cluster: {cluster.value}")
        
        cluster_config = self.portfolio_engine.asset_clusters[cluster]
        assets = cluster_config['assets']
        
        cluster_data = {}
        
        for asset in assets:
            try:
                # Collect data for the asset
                data = self._collect_asset_data(asset)
                if data is not None and not data.empty:
                    cluster_data[asset] = data
                    logger.info(f"   {asset}: {len(data)} samples collected)              else:
                    logger.warning(f"   {asset}: No data collected")
                    
            except Exception as e:
                logger.error(f"Error collecting data for {asset}: {e}")
        
        logger.info(f"✅ Collected data for[object Object]len(cluster_data)} assets in {cluster.value}")
        return cluster_data
    
    def _collect_asset_data(self, asset: str) -> Optional[pd.DataFrame]:
     Collect data for a single asset"""
        try:
            # Convert asset to FDUSD pair
            symbol = f"{asset}FDUSD"
            
            # Calculate start time
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.training_config['data_days'])
            
            # Fetch klines data
            klines = fetch_klines(symbol,1m, start_time, end_time)
            
            if not klines:
                logger.warning(f"No klines data for {symbol})            return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                timestamp,open', 'high,low, 'close', 'volume,
             close_time, quote_volume', trades,taker_buy_base,
                taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high,low, 'close', 'volume]           for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df[timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add asset identifier
            df['asset'] = asset
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting data for {asset}: {e}")
            return None
    
    def engineer_cluster_features(self, cluster_data: Dict[str, pd.DataFrame], cluster: AssetCluster) -> pd.DataFrame:
 Engineer features specific to the cluster       logger.info(f"🔧 Engineering features for cluster: {cluster.value}")
        
        # Get cluster configuration
        cluster_config = self.portfolio_engine.asset_clusters[cluster]
        target_features = cluster_config['target_features']
        
        all_features = []
        
        for asset, data in cluster_data.items():
            try:
                # Add basic technical features
                features = self.feature_engineer.add_technical_features(data.copy())
                
                # Add cluster-specific features
                features = self._add_cluster_specific_features(features, cluster, target_features)
                
                # Add alternative data features
                features = self._add_alternative_data_features(features, asset)
                
                all_features.append(features)
                
            except Exception as e:
                logger.error(fError engineering features for {asset}: {e}")
        
        if not all_features:
            logger.error(f"No features generated for cluster {cluster.value}")
            return pd.DataFrame()
        
        # Combine all assets
        combined_features = pd.concat(all_features, ignore_index=True)
        
        logger.info(f✅ Engineered {len(combined_features)} samples with {len(combined_features.columns)} features for {cluster.value}")
        return combined_features
    
    def _add_cluster_specific_features(self, df: pd.DataFrame, cluster: AssetCluster, target_features: List[str]) -> pd.DataFrame:
      features specific to the cluster type"""
        if social_sentiment' in target_features:
            # Add social sentiment features for memecoins and DeFi
            df[social_momentum'] = np.random.normal(0, 1(df))  # Simulated
            df[hype_cycle'] = np.random.uniform(0, 1(df))  # Simulated
        
        if 'market_correlation' in target_features:
            # Add market correlation features for large caps
            df[btc_correlation'] = np.random.uniform(00.5, 1.0(df))  # Simulated
            df['market_beta'] = np.random.uniform(00.8, 1.2(df))  # Simulated
        
        ifdefi_metrics' in target_features:
            # Add DeFi-specific features
            df[tvl_change'] = np.random.normal(005(df))  # Simulated
            df[yield_rate'] = np.random.uniform(0010.20(df))  # Simulated
        
        if ai_news_sentiment' in target_features:
            # Add AI sector features
            df['ai_sentiment'] = np.random.normal(0, 1(df))  # Simulated
            df[tech_momentum'] = np.random.uniform(-11(df))  # Simulated
        
        return df
    
    def _add_alternative_data_features(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        d alternative data features"""
        try:
            # Add fear & greed index
            df[fear_greed'] = np.random.uniform(0, 100(df))  # Simulated
            
            # Add news sentiment
            df['news_sentiment'] = np.random.normal(0, 1(df))  # Simulated
            
            # Add on-chain metrics
            df['onchain_activity'] = np.random.uniform(0, 1(df))  # Simulated
            
        except Exception as e:
            logger.warning(f"Error adding alternative data for {asset}: {e}")
        
        return df
    
    def prepare_cluster_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
epare features and targets for cluster training       logger.info("🎯 Preparing features and targets")
        
        # Create targets for different timeframes
        targets = {}
        
        for timeframe in self.training_config['timeframes]:      if timeframe == '1m:           targets[timeframe] = df['close'].pct_change(1).shift(-1)
            elif timeframe == '5m:           targets[timeframe] = df['close'].pct_change(5).shift(-5)
            elif timeframe == '15m:           targets[timeframe] = df['close].pct_change(15).shift(-15)
        
        # Remove features that would cause data leakage
        feature_columns = [col for col in df.columns if col not inclose,open', 'high', low, volume', 'asset']]
        
        # Prepare features
        X = df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        logger.info(f✅Prepared {len(X_scaled)} samples with {len(X_scaled.columns)} features")
        
        return X_scaled, targets
    
    def train_cluster_models(self, cluster: AssetCluster, X: pd.DataFrame, targets: Dict[str, pd.Series]):
     Train models for a specific cluster       logger.info(f"🧠 Training models for cluster: {cluster.value}")
        
        cluster_models = [object Object]       cluster_performance = {}
        
        for timeframe, target in targets.items():
            if target is None or target.empty:
                continue
            
            # Align features and target
            common_index = X.index.intersection(target.index)
            X_aligned = X.loc[common_index]
            y_aligned = target.loc[common_index]
            
            if len(X_aligned) < self.training_config['min_samples']:
                logger.warning(f"Insufficient samples for {cluster.value}[object Object]timeframe}: {len(X_aligned)})          continue
            
            timeframe_models = {}
            timeframe_performance = {}
            
            for model_type in self.training_config['models_per_cluster']:
                try:
                    model, score = self._train_model(model_type, X_aligned, y_aligned, cluster, timeframe)
                    if model is not None:
                        timeframe_models[model_type] = model
                        timeframe_performance[model_type] = score
                        logger.info(f{model_type} {timeframe}: {score:.3f}")
                        
                except Exception as e:
                    logger.error(f"Error training {model_type} for {cluster.value} {timeframe}: {e}")
            
            cluster_models[timeframe] = timeframe_models
            cluster_performance[timeframe] = timeframe_performance
        
        # Store models and performance
        self.cluster_models[cluster] = cluster_models
        self.cluster_performance[cluster] = cluster_performance
        
        logger.info(f"✅ Trained models for {cluster.value}: {len(cluster_models)} timeframes")
    
    def _train_model(self, model_type: str, X: pd.DataFrame, y: pd.Series, cluster: AssetCluster, timeframe: str):
n a specific model with hyperparameter optimization""      
        def objective(trial):
            if model_type == 'lightgbm:            params = {
                   objective': 'regression',
                    metric                   boosting_type': 'gbdt',
               num_leaves': trial.suggest_int(num_leaves', 20, 100),
                  learning_rate': trial.suggest_float(learning_rate', 0.01, 0.3),
                   feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                   bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                 bagging_freq': trial.suggest_int(bagging_freq', 1, 7),
                    min_child_samples': trial.suggest_int(min_child_samples', 5, 100),
                 verbose                   random_state': 42                }
                model = lgb.LGBMRegressor(**params)
                
            elif model_type == 'xgboost:            params = {
                    objective':reg:squarederror',
                    eval_metric': 'rmse',
                    max_depth': trial.suggest_int('max_depth', 3, 8),
                  learning_rate': trial.suggest_float(learning_rate', 0.05, 0.3),
                 n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    subsample': trial.suggest_float(subsample', 0.7, 1.0),
                   colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                   min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                    reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
               reg_lambda': trial.suggest_float(reg_lambda', 0, 1),
                    random_state': 42,
                 verbosity0                }
                model = xgb.XGBRegressor(**params)
                
            else:  # catboost
                params = {
               iterations': trial.suggest_int(iterations', 100, 500),
                  depth': trial.suggest_int('depth', 4, 10),
                  learning_rate': trial.suggest_float(learning_rate', 0.01, 0.3),
                l2leaf_reg': trial.suggest_float(l2eaf_reg', 1, 10),
                 border_count': trial.suggest_int(border_count', 32, 255),
            bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                    random_strength': trial.suggest_float(random_strength', 0, 1),
                    verbose               allow_writing_files': False
                }
                try:
                    import catboost as cb
                    model = cb.CatBoostRegressor(**params)
                except ImportError:
                    return float('inf')
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.training_config['cv_folds'])
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(score)
            
            return -np.mean(scores)  # Negative because we minimize
        
        # Run optimization
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42       study.optimize(objective, n_trials=self.training_config['optimization_trials'])
        
        if study.best_trial is None:
            return None, float('inf')
        
        # Train final model with best parameters
        best_params = study.best_params
        
        if model_type == 'lightgbm':
            best_params.update({
               objective': 'regression,
                metric,
                boosting_type': 'gbdt,
             verbose1
                random_state': 42    })
            final_model = lgb.LGBMRegressor(**best_params)
            
        elif model_type == 'xgboost':
            best_params.update({
                objective':reg:squarederror,
                eval_metric': 'rmse,
                random_state': 42
             verbosity':0    })
            final_model = xgb.XGBRegressor(**best_params)
            
        else:  # catboost
            best_params.update({
                verbosee,
            allow_writing_files': False
            })
            try:
                import catboost as cb
                final_model = cb.CatBoostRegressor(**best_params)
            except ImportError:
                return None, float('inf')
        
        # Train final model
        final_model.fit(X, y)
        
        # Calculate final score
        y_pred = final_model.predict(X)
        final_score = r2_score(y, y_pred)
        
        return final_model, final_score
    
    def save_cluster_models(self, cluster: AssetCluster):
        Save models for a specific cluster       logger.info(f"💾 Saving models for cluster: {cluster.value}")
        
        cluster_dir = self.models_dir / cluster.value
        cluster_dir.mkdir(exist_ok=True)
        
        # Save models
        models = self.cluster_models[cluster]
        for timeframe, timeframe_models in models.items():
            for model_type, model in timeframe_models.items():
                model_path = cluster_dir / f"{model_type}_{timeframe}.joblib"
                joblib.dump(model, model_path)
        
        # Save performance metrics
        performance = self.cluster_performance[cluster]
        performance_path = cluster_dir /performance.json"
        with open(performance_path, 'w') as f:
            json.dump(performance, f, indent=2, default=str)
        
        logger.info(f"✅ Saved models for {cluster.value}")
    
    def train_all_clusters(self):
     ain models for all clusters       logger.info(🚀Starting multi-asset training for all clusters")
        
        training_start = datetime.now()
        
        for cluster in AssetCluster:
            try:
                logger.info(f"\n🎯 Training cluster: {cluster.value}")
                
                # 1llect cluster data
                cluster_data = self.collect_cluster_data(cluster)
                
                if not cluster_data:
                    logger.warning(fNodata collected for {cluster.value}")
                    continue
                
                # 2. Engineer cluster-specific features
                features = self.engineer_cluster_features(cluster_data, cluster)
                
                if features.empty:
                    logger.warning(f"No features generated for {cluster.value}")
                    continue
                
                # 3. Prepare targets
                X, targets = self.prepare_cluster_targets(features)
                
                # 4. Train models
                self.train_cluster_models(cluster, X, targets)
                
                # 5. Save models
                self.save_cluster_models(cluster)
                
                logger.info(f"✅ Completed training for {cluster.value}")
                
            except Exception as e:
                logger.error(f"Error training cluster [object Object]cluster.value}: {e}")
        
        training_end = datetime.now()
        training_duration = training_end - training_start
        
        logger.info(f"\n🎉 Multi-asset training completed in {training_duration}")
        logger.info("🧠 All cluster-specific models trained and saved!")
    
    def get_cluster_predictions(self, cluster: AssetCluster, asset: str, features: pd.DataFrame) -> Dict[str, float]:
        et predictions for an asset using cluster-specific models"""
        try:
            predictions = {}
            models = self.cluster_models[cluster]
            
            for timeframe, timeframe_models in models.items():
                # Use ensemble of models for the timeframe
                timeframe_predictions = []
                
                for model_type, model in timeframe_models.items():
                    try:
                        pred = model.predict(features)[0]
                        timeframe_predictions.append(pred)
                    except Exception as e:
                        logger.warning(fError with {model_type} prediction: {e}")
                
                if timeframe_predictions:
                    # Average predictions from all models
                    predictions[timeframe] = np.mean(timeframe_predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions for {asset}: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    trainer = MultiAssetTrainer()
    trainer.train_all_clusters() 