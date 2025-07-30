#!/usr/bin/env python3
"""
COMPREHENSIVE CPU OPTIMIZER
===========================

This module ensures ALL ML libraries use maximum CPU cores:
- LightGBM, XGBoost, CatBoost
- Scikit-learn, TensorFlow, PyTorch
- NumPy, Pandas, Joblib
- All parallel processing operations
"""

import os
import multiprocessing as mp
import psutil
import logging
from typing import Dict, Any

class CPUOptimizer:
    """Comprehensive CPU optimization for all ML libraries"""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.optimal_cores = self._calculate_optimal_cores()
        self.logger = logging.getLogger(__name__)
        
    def _calculate_optimal_cores(self) -> int:
        """Calculate optimal number of cores based on system specs"""
        if self.memory_gb >= 32:
            # High-end system: use all cores
            return self.cpu_count
        elif self.memory_gb >= 16:
            # Mid-range system: use all but 1 core
            return max(self.cpu_count - 1, self.cpu_count // 2)
        elif self.memory_gb >= 8:
            # Lower-end system: use 75% of cores
            return max(self.cpu_count - 2, int(self.cpu_count * 0.75))
        else:
            # Low memory: use 50% of cores
            return max(2, self.cpu_count // 2)
    
    def optimize_globally(self) -> int:
        """Apply comprehensive CPU optimization globally"""
        self.logger.info(f"🚀 COMPREHENSIVE CPU OPTIMIZATION")
        self.logger.info(f"   System: {self.cpu_count} cores, {self.memory_gb:.1f}GB RAM")
        self.logger.info(f"   Target: {self.optimal_cores} cores ({self.optimal_cores/self.cpu_count*100:.1f}%)")
        
        # === ENVIRONMENT VARIABLES ===
        env_vars = {
            # Core ML libraries
            'OMP_NUM_THREADS': str(self.optimal_cores),
            'MKL_NUM_THREADS': str(self.optimal_cores),
            'OPENBLAS_NUM_THREADS': str(self.optimal_cores),
            'VECLIB_MAXIMUM_THREADS': str(self.optimal_cores),
            'NUMEXPR_NUM_THREADS': str(self.optimal_cores),
            
            # Additional optimization
            'BLAS_NUM_THREADS': str(self.optimal_cores),
            'LAPACK_NUM_THREADS': str(self.optimal_cores),
            'ATLAS_NUM_THREADS': str(self.optimal_cores),
            
            # Joblib parallel processing
            'JOBLIB_NUM_THREADS': str(self.optimal_cores),
            
            # Pandas optimization
            'PANDAS_NUM_THREADS': str(self.optimal_cores),
            
            # NumPy optimization
            'NPY_NUM_THREADS': str(self.optimal_cores),
            
            # Additional libraries
            'OPENMP_NUM_THREADS': str(self.optimal_cores),
            'MKL_DYNAMIC': 'FALSE',  # Prevent dynamic thread adjustment
            'OMP_DYNAMIC': 'FALSE',  # Prevent dynamic thread adjustment
            
            # TensorFlow optimization
            'TF_NUM_THREADS': str(self.optimal_cores),
            'TF_CPP_MIN_LOG_LEVEL': '2',  # Reduce TensorFlow logging
            
            # PyTorch optimization
            'OMP_NUM_THREADS': str(self.optimal_cores),
            'MKL_NUM_THREADS': str(self.optimal_cores),
            
            # CUDA optimization (if available)
            'CUDA_VISIBLE_DEVICES': '0',  # Use first GPU if available
            
            # Memory optimization
            'MALLOC_TRIM_THRESHOLD_': '131072',  # 128KB
            'MALLOC_MMAP_THRESHOLD_': '131072',  # 128KB
        }
        
        # Apply environment variables
        for var, value in env_vars.items():
            os.environ[var] = value
            self.logger.info(f"   Set {var} = {value}")
        
        # === GLOBAL CONFIGURATION ===
        self._configure_global_libraries()
        
        self.logger.info(f"✅ CPU optimization applied: {self.optimal_cores}/{self.cpu_count} cores")
        return self.optimal_cores
    
    def _configure_global_libraries(self):
        """Configure global library settings for maximum parallelism"""
        try:
            # Configure NumPy
            import numpy as np
            np.set_num_threads(self.optimal_cores)
            self.logger.info(f"   NumPy threads: {np.get_num_threads()}")
            
            # Configure Pandas
            import pandas as pd
            pd.options.mode.chained_assignment = None  # Reduce warnings
            
            # Configure Joblib
            from joblib import parallel_backend
            parallel_backend('threading', n_jobs=self.optimal_cores)
            
            # Configure Scikit-learn
            from sklearn import set_config
            set_config(n_jobs=self.optimal_cores)
            
            # Configure TensorFlow
            try:
                import tensorflow as tf
                tf.config.threading.set_intra_op_parallelism_threads(self.optimal_cores)
                tf.config.threading.set_inter_op_parallelism_threads(self.optimal_cores)
                self.logger.info(f"   TensorFlow threads configured")
            except ImportError:
                pass
            
            # Configure PyTorch
            try:
                import torch
                torch.set_num_threads(self.optimal_cores)
                self.logger.info(f"   PyTorch threads: {torch.get_num_threads()}")
            except ImportError:
                pass
            
        except Exception as e:
            self.logger.warning(f"   Some library configurations failed: {e}")
    
    def get_optimal_cores(self) -> int:
        """Get the optimal number of cores for this system"""
        return self.optimal_cores
    
    def get_parallel_params(self) -> Dict[str, Any]:
        """Get parallel processing parameters for all ML libraries"""
        return {
            # LightGBM
            'lightgbm': {
                'n_jobs': self.optimal_cores,
                'num_threads': self.optimal_cores,
                'verbose': -1
            },
            
            # XGBoost
            'xgboost': {
                'n_jobs': self.optimal_cores,
                'nthread': self.optimal_cores,
                'verbosity': 0
            },
            
            # CatBoost
            'catboost': {
                'thread_count': self.optimal_cores,
                'verbose': False
            },
            
            # Scikit-learn
            'sklearn': {
                'n_jobs': self.optimal_cores
            },
            
            # Random Forest
            'random_forest': {
                'n_jobs': self.optimal_cores
            },
            
            # SVM
            'svm': {
                'n_jobs': self.optimal_cores
            },
            
            # Cross validation
            'cv': {
                'n_jobs': self.optimal_cores
            },
            
            # Feature selection
            'feature_selection': {
                'n_jobs': self.optimal_cores
            },
            
            # Grid search
            'grid_search': {
                'n_jobs': self.optimal_cores
            },
            
            # Optuna
            'optuna': {
                'n_jobs': self.optimal_cores
            }
        }
    
    def verify_optimization(self) -> bool:
        """Verify that CPU optimization is working"""
        try:
            import numpy as np
            import pandas as pd
            
            # Check NumPy
            numpy_threads = np.get_num_threads()
            
            # Check Pandas
            pandas_threads = pd.get_option('display.max_rows')
            
            # Check environment variables
            env_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
            
            self.logger.info(f"🔍 CPU Optimization Verification:")
            self.logger.info(f"   Target cores: {self.optimal_cores}")
            self.logger.info(f"   NumPy threads: {numpy_threads}")
            self.logger.info(f"   Environment OMP_NUM_THREADS: {env_threads}")
            
            # Check if optimization is working
            if numpy_threads >= self.optimal_cores * 0.8:  # Allow 20% tolerance
                self.logger.info(f"✅ CPU optimization verified successfully")
                return True
            else:
                self.logger.warning(f"⚠️ CPU optimization may not be fully applied")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to verify CPU optimization: {e}")
            return False

# Global instance
cpu_optimizer = CPUOptimizer()
OPTIMAL_CORES = cpu_optimizer.optimize_globally()
PARALLEL_PARAMS = cpu_optimizer.get_parallel_params()

def get_optimal_cores() -> int:
    """Get optimal number of cores"""
    return OPTIMAL_CORES

def get_parallel_params() -> Dict[str, Any]:
    """Get parallel processing parameters"""
    return PARALLEL_PARAMS

def verify_cpu_optimization() -> bool:
    """Verify CPU optimization is working"""
    return cpu_optimizer.verify_optimization()

if __name__ == "__main__":
    # Test the optimizer
    print("🧪 Testing CPU Optimizer...")
    cores = get_optimal_cores()
    params = get_parallel_params()
    verified = verify_cpu_optimization()
    
    print(f"✅ Optimal cores: {cores}")
    print(f"✅ Parallel params: {params}")
    print(f"✅ Verification: {verified}") 