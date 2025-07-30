#!/usr/bin/env python3
"""
Project Hyperion - Training Entry Point
Professional Training Interface

This provides a clean interface for running training operations
with the new modular architecture.
"""

import sys
import os
from pathlib import Path
from typing import Optional, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.logging.logger import setup_logger, log_training_start, log_training_complete
from config.training_config import training_config, TrainingMode
from training.modes.month_trainer import MonthTrainer

def display_training_menu():
    """Display available training options"""
    print("\n" + "="*60)
    print("🚀 PROJECT HYPERION - TRAINING MODES")
    print("="*60)
    
    modes = training_config.get_all_modes()
    
    for i, mode in enumerate(modes, 1):
        config = training_config.get_mode_config(mode)
        description = config.get('description', 'Unknown')
        time_estimate = config.get('time_estimate', 'Unknown')
        weight_estimate = config.get('weight_estimate', 'Unknown')
        
        # Add indicators for new modes
        if mode in ['month', 'quarter', 'half_year', 'year', 'two_year']:
            description += " 🆕"
        
        print(f"{i:2d}. {description}")
        print(f"    ⏱️  Time: {time_estimate} | 📊 Weight: {weight_estimate}")
    
    print("\n" + "="*60)
    print("💡 NEW: Extended timeframes (30d, 3m, 6m, 1y, 2y)")
    print("🛡️  All modes are rate-limit compliant")
    print("🧠 All modes include 10X intelligence features")
    print("="*60)

def get_training_mode() -> Optional[str]:
    """Get user's training mode choice"""
    try:
        choice = input("\nSelect training mode (1-11): ").strip()
        choice_num = int(choice)
        
        modes = training_config.get_all_modes()
        if 1 <= choice_num <= len(modes):
            return modes[choice_num - 1]
        else:
            print("❌ Invalid choice. Please select 1-11.")
            return None
    except ValueError:
        print("❌ Please enter a valid number.")
        return None
    except KeyboardInterrupt:
        print("\n\n👋 Training cancelled.")
        return None

def run_training(mode: str):
    """Run training for the selected mode"""
    logger = setup_logger(f"hyperion.training.{mode}")
    
    try:
        # For now, we'll implement the 30-day trainer as an example
        if mode == 'month':
            trainer = MonthTrainer()
            log_training_start(logger, mode, trainer.symbols)
            
            # Run training
            results = trainer.run_training()
            
            # Log completion
            duration = results.get('training_duration', 0)
            models_trained = results.get('models_trained', 0)
            metrics = results.get('metrics', {})
            
            log_training_complete(logger, duration, models_trained, metrics)
            
            return results
        else:
            logger.info(f"Training mode '{mode}' will be implemented in the next phase")
            logger.info("Currently demonstrating the 30-day trainer structure")
            return None
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None

def main():
    """Main training function"""
    
    # Setup logging
    logger = setup_logger("hyperion.train")
    
    logger.info("🚀 Project Hyperion - Training Interface")
    logger.info("📊 Professional training with modular architecture")
    
    # Display menu
    display_training_menu()
    
    # Get user choice
    mode = get_training_mode()
    if not mode:
        return False
    
    # Confirm choice
    config = training_config.get_mode_config(mode)
    description = config.get('description', 'Unknown')
    
    print(f"\n🎯 Selected: {description}")
    print(f"⏱️  Estimated time: {config.get('time_estimate', 'Unknown')}")
    print(f"📊 Estimated weight: {config.get('weight_estimate', 'Unknown')}")
    
    confirm = input("\nProceed with training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("👋 Training cancelled.")
        return False
    
    # Run training
    print(f"\n🚀 Starting {description}...")
    results = run_training(mode)
    
    if results and results.get('success'):
        print(f"\n✅ {description} completed successfully!")
        print(f"⏱️  Duration: {results.get('training_duration', 0):.2f} seconds")
        print(f"🧠 Models trained: {results.get('models_trained', 0)}")
        
        metrics = results.get('metrics', {})
        if metrics:
            avg_metric = sum(metrics.values()) / len(metrics)
            print(f"📊 Average performance: {avg_metric:.4f}")
    else:
        print(f"\n❌ {description} failed or not yet implemented")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Training cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1) 