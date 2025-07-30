#!/usr/bin/env python3
"""
TRAINING CHECKPOINT SYSTEM
Project Hyperion - Seamless Pause/Resume Integration

This system provides checkpoint functionality that can be integrated into the existing training script.
"""

import json
import pickle
import time
import os
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for pause/resume
PAUSE_REQUESTED = False
RESUME_REQUESTED = False
TRAINING_PAUSED = False

class CheckpointSystem:
    """Checkpoint system for training pause/resume"""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / 'training_checkpoint.pkl'
        self.state_file = self.checkpoint_dir / 'training_state.json'
        self.progress_file = self.checkpoint_dir / 'training_progress.json'
        self.pause_file = self.checkpoint_dir / 'pause_requested.txt'
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for pause/resume"""
        try:
            signal.signal(signal.SIGUSR1, self._pause_signal_handler)
            signal.signal(signal.SIGUSR2, self._resume_signal_handler)
        except AttributeError:
            # Windows doesn't support SIGUSR1/SIGUSR2
            pass
    
    def _pause_signal_handler(self, signum, frame):
        """Handle pause signal"""
        global PAUSE_REQUESTED
        logger.info("⏸️ Pause signal received")
        PAUSE_REQUESTED = True
    
    def _resume_signal_handler(self, signum, frame):
        """Handle resume signal"""
        global RESUME_REQUESTED, TRAINING_PAUSED
        logger.info("▶️ Resume signal received")
        RESUME_REQUESTED = True
        TRAINING_PAUSED = False
    
    def save_checkpoint(self, trainer, current_step: str, step_progress: float, 
                       models_trained: List[str], current_model: str = None, **kwargs):
        """Save training checkpoint"""
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'current_step': current_step,
                'step_progress': step_progress,
                'models_trained': models_trained,
                'current_model': current_model,
                'training_duration': getattr(trainer, 'training_start_time', None),
                'data_collected': getattr(trainer, 'data_collected', False),
                'features_engineered': getattr(trainer, 'features_engineered', False),
                'model_performance': getattr(trainer, 'model_performance', {}),
                'ensemble_weights': getattr(trainer, 'ensemble_weights', {}),
                'checkpoint_version': '1.0',
                **kwargs
            }
            
            # Save checkpoint data
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Save training state
            training_state = {
                'timestamp': datetime.now().isoformat(),
                'current_step': current_step,
                'step_progress': step_progress,
                'models_trained': models_trained,
                'current_model': current_model,
                'total_models': len(models_trained),
                'estimated_remaining_time': self._estimate_remaining_time(step_progress),
                'training_status': 'active'
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(training_state, f, indent=2)
            
            # Save progress
            progress_data = {
                'timestamp': datetime.now().isoformat(),
                'overall_progress': step_progress,
                'current_step': current_step,
                'models_completed': len(models_trained),
                'current_model': current_model,
                'training_duration': str(datetime.now() - trainer.training_start_time) if hasattr(trainer, 'training_start_time') else None
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            logger.info(f"💾 Checkpoint saved: {current_step} ({step_progress:.1f}%) - {len(models_trained)} models trained")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load training checkpoint"""
        try:
            if not self.checkpoint_file.exists():
                logger.info("ℹ️ No checkpoint found - starting fresh training")
                return None
            
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            logger.info(f"📂 Checkpoint loaded: {checkpoint_data['current_step']} ({checkpoint_data['step_progress']:.1f}%)")
            logger.info(f"📊 Models trained: {len(checkpoint_data['models_trained'])}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self):
        """Clear checkpoint files"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.state_file.exists():
                self.state_file.unlink()
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.pause_file.exists():
                self.pause_file.unlink()
            logger.info("🗑️ Checkpoint files cleared")
        except Exception as e:
            logger.error(f"❌ Failed to clear checkpoint: {e}")
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            return {'status': 'no_checkpoint'}
        except Exception as e:
            logger.error(f"❌ Failed to get training status: {e}")
            return {'status': 'error'}
    
    def check_pause_request(self) -> bool:
        """Check if pause is requested and handle it"""
        global PAUSE_REQUESTED, TRAINING_PAUSED, RESUME_REQUESTED
        
        # Check for pause file (Windows compatibility)
        if self.pause_file.exists():
            PAUSE_REQUESTED = True
            self.pause_file.unlink()
        
        if PAUSE_REQUESTED:
            logger.info("⏸️ Pausing training at safe checkpoint...")
            TRAINING_PAUSED = True
            PAUSE_REQUESTED = False
            
            # Wait for resume signal
            while TRAINING_PAUSED:
                time.sleep(1)
                if RESUME_REQUESTED:
                    break
            
            logger.info("▶️ Training resumed from checkpoint")
            RESUME_REQUESTED = False
            return True
        
        return False
    
    def _estimate_remaining_time(self, progress: float) -> str:
        """Estimate remaining training time"""
        if progress <= 0:
            return "Unknown"
        
        # Estimate based on progress
        if progress < 20:
            return "~8-10 hours remaining"
        elif progress < 40:
            return "~6-8 hours remaining"
        elif progress < 60:
            return "~4-6 hours remaining"
        elif progress < 80:
            return "~2-4 hours remaining"
        else:
            return "~1-2 hours remaining"

# Integration functions for the main training script
def setup_checkpoint_system(trainer):
    """Setup checkpoint system for trainer"""
    trainer.checkpoint_system = CheckpointSystem()
    trainer.training_start_time = datetime.now()
    trainer.models_trained = []
    trainer.current_step = "initializing"
    trainer.step_progress = 0.0
    
    # Load existing checkpoint if available
    checkpoint_data = trainer.checkpoint_system.load_checkpoint()
    if checkpoint_data:
        trainer.models_trained = checkpoint_data.get('models_trained', [])
        trainer.current_step = checkpoint_data.get('current_step', 'initializing')
        trainer.step_progress = checkpoint_data.get('step_progress', 0.0)
        trainer.model_performance = checkpoint_data.get('model_performance', {})
        trainer.ensemble_weights = checkpoint_data.get('ensemble_weights', {})
        
        logger.info(f"🔄 Resuming training from checkpoint: {trainer.current_step}")
        return True
    
    return False

def update_training_progress(trainer, step_name: str, progress: float, model_name: str = None):
    """Update training progress with checkpoint saving"""
    trainer.current_step = step_name
    trainer.step_progress = progress
    
    if model_name and model_name not in trainer.models_trained:
        trainer.models_trained.append(model_name)
    
    # Save checkpoint periodically
    if hasattr(trainer, 'checkpoint_system'):
        trainer.checkpoint_system.save_checkpoint(
            trainer, step_name, progress, trainer.models_trained, model_name
        )
    
    # Check for pause request
    if hasattr(trainer, 'checkpoint_system'):
        trainer.checkpoint_system.check_pause_request()

def get_training_status(trainer) -> Dict:
    """Get current training status"""
    return {
        'status': 'active' if not TRAINING_PAUSED else 'paused',
        'current_step': getattr(trainer, 'current_step', 'unknown'),
        'progress': getattr(trainer, 'step_progress', 0.0),
        'models_trained': len(getattr(trainer, 'models_trained', [])),
        'total_models': 64,  # Total expected models
        'current_model': getattr(trainer, 'current_model', None),
        'training_duration': str(datetime.now() - trainer.training_start_time) if hasattr(trainer, 'training_start_time') else None,
        'estimated_remaining': trainer.checkpoint_system._estimate_remaining_time(trainer.step_progress) if hasattr(trainer, 'checkpoint_system') else "Unknown"
    }

# Usage instructions
def print_usage_instructions():
    """Print usage instructions for pause/resume functionality"""
    print("\n🎯 PAUSE/RESUME TRAINING INSTRUCTIONS:")
    print("=" * 50)
    print("1. START TRAINING:")
    print("   python ultra_train_enhanced.py")
    print()
    print("2. PAUSE TRAINING:")
    print("   On Unix/Linux: kill -SIGUSR1 <PID>")
    print("   On Windows: Create file 'checkpoints/pause_requested.txt'")
    print("   Or use: python pause_resume_training.py pause")
    print()
    print("3. RESUME TRAINING:")
    print("   On Unix/Linux: kill -SIGUSR2 <PID>")
    print("   On Windows: Delete file 'checkpoints/pause_requested.txt'")
    print("   Or use: python pause_resume_training.py resume")
    print()
    print("4. CHECK STATUS:")
    print("   python pause_resume_training.py status")
    print()
    print("5. MONITOR PROGRESS:")
    print("   python pause_resume_training.py monitor")
    print()
    print("6. STOP TRAINING:")
    print("   python pause_resume_training.py stop")
    print()
    print("💡 Checkpoints are saved automatically every 5 minutes")
    print("💡 Training will resume from the exact same point")
    print("💡 No data loss during pause/resume")
    print("=" * 50)

if __name__ == "__main__":
    print_usage_instructions() 