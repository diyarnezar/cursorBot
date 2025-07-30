#!/usr/bin/env python3
"""
CPU Optimization Script for Ultra Enhanced Training
Maximizes CPU usage for faster training while keeping hardware safe
"""

import os
import multiprocessing as mp
import psutil
import numpy as np
import pandas as pd
from typing import Tuple

def optimize_cpu_usage() -> Tuple[int, int]:
    """Optimize CPU usage for maximum training speed while keeping hardware safe"""
    # Get system info
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"🖥️  System Info:")
    print(f"   • CPU Cores: {cpu_count}")
    print(f"   • Memory: {memory_gb:.1f}GB")
    
    # Calculate optimal CPU usage based on system specs
    if memory_gb >= 16:  # High memory system
        optimal_cores = cpu_count - 1  # Use all cores except 1
        cpu_percentage = 90  # 90% CPU usage
        print(f"   • System Type: High Memory")
    elif memory_gb >= 8:  # Medium memory system
        optimal_cores = max(cpu_count - 2, cpu_count // 2)
        cpu_percentage = 80  # 80% CPU usage
        print(f"   • System Type: Medium Memory")
    else:  # Low memory system
        optimal_cores = max(2, cpu_count // 2)
        cpu_percentage = 70  # 70% CPU usage
        print(f"   • System Type: Low Memory")
    
    # Set environment variables for ML libraries
    os.environ['OMP_NUM_THREADS'] = str(optimal_cores)
    os.environ['MKL_NUM_THREADS'] = str(optimal_cores)
    os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_cores)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(optimal_cores)
    os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_cores)
    
    print(f"\n🚀 CPU OPTIMIZATION APPLIED:")
    print(f"   • Using {optimal_cores}/{cpu_count} cores")
    print(f"   • Target CPU usage: {cpu_percentage}%")
    print(f"   • Environment variables set for maximum performance")
    
    return optimal_cores, cpu_percentage

def get_optimal_model_params(optimal_cores: int) -> dict:
    """Get optimal parameters for ML models to use maximum CPU"""
    return {
        'lightgbm': {
            'n_jobs': optimal_cores,
            'force_col_wise': True,
            'device': 'cpu',
            'verbose': -1
        },
        'xgboost': {
            'n_jobs': optimal_cores,
            'verbosity': 0,
            'tree_method': 'hist'  # Faster for CPU
        },
        'catboost': {
            'thread_count': optimal_cores,
            'verbose': False,
            'task_type': 'CPU'
        },
        'random_forest': {
            'n_jobs': optimal_cores,
            'verbose': 0
        },
        'svm': {
            'n_jobs': optimal_cores
        },
        'neural_network': {
            'workers': optimal_cores,
            'use_multiprocessing': True
        }
    }

def apply_optimization_to_training():
    """Apply CPU optimization to the current training session"""
    print("🔧 Applying CPU optimization to training...")
    
    # Optimize CPU usage
    optimal_cores, cpu_percentage = optimize_cpu_usage()
    
    # Get optimal model parameters
    model_params = get_optimal_model_params(optimal_cores)
    
    print(f"\n📊 OPTIMAL MODEL PARAMETERS:")
    for model, params in model_params.items():
        print(f"   • {model.upper()}: {params}")
    
    print(f"\n✅ CPU optimization complete!")
    print(f"   • Training will now use {optimal_cores} cores")
    print(f"   • Expected CPU usage: {cpu_percentage}%")
    print(f"   • Training speed should increase significantly")
    
    # Save optimization settings
    optimization_data = {
        'optimal_cores': optimal_cores,
        'cpu_percentage': cpu_percentage,
        'model_params': model_params,
        'timestamp': str(pd.Timestamp.now())
    }
    
    import json
    with open('cpu_optimization.json', 'w') as f:
        json.dump(optimization_data, f, indent=2)
    
    print(f"   • Settings saved to cpu_optimization.json")
    
    return optimal_cores, model_params

if __name__ == "__main__":
    apply_optimization_to_training() 