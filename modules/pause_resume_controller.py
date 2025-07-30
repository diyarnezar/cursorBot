#!/usr/bin/env python3
"""
Pause/Resume Controller Module
Provides keyboard-based pause/resume functionality for long-running training processes.
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import signal

# Try to import keyboard for hotkey support
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("WARNING: 'keyboard' package not installed. Installing now...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard"])
        import keyboard
        KEYBOARD_AVAILABLE = True
        print("✓ keyboard package installed successfully")
    except Exception as e:
        print(f"WARNING: Could not install keyboard package: {e}")
        print("Pause/resume functionality will be limited to Ctrl+C only")
        KEYBOARD_AVAILABLE = False

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class PauseResumeController:
    """
    Controller for pause/resume functionality with keyboard shortcuts and checkpointing.
    
    Features:
    - Ctrl+P to pause training
    - Ctrl+R to resume training
    - Automatic checkpoint saving every 5 minutes
    - Manual checkpoint saving on pause
    - Resume from checkpoint functionality
    - Status monitoring and logging
    - Custom Optuna optimization with pause support
    """
    
    def __init__(self, checkpoint_file: str = 'training_checkpoint.json', 
                 checkpoint_interval: int = 300):
        """
        Initialize the pause/resume controller.
        
        Args:
            checkpoint_file: Path to save/load checkpoints
            checkpoint_interval: Seconds between automatic checkpoints
        """
        self.checkpoint_file = checkpoint_file
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint_time = None
        
        # Pause/resume state
        self.training_paused = False
        self.training_resumed = False
        self.pause_requested = False
        self.resume_requested = False
        
        # Callback functions
        self.on_pause_callback = None
        self.on_resume_callback = None
        self.on_checkpoint_callback = None
        
        # Threading
        self.monitor_thread = None
        self._stop_monitoring = False
        
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Setup keyboard shortcuts if available
        self._setup_keyboard_shortcuts()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("✅ Pause/Resume Controller initialized")
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for pause/resume"""
        if KEYBOARD_AVAILABLE:
            try:
                # Clear any existing hotkeys first
                keyboard.unhook_all()
                
                # Setup new hotkeys
                keyboard.add_hotkey('ctrl+p', self.pause_training, suppress=True)
                keyboard.add_hotkey('ctrl+r', self.resume_training, suppress=True)
                
                # Test that the hotkeys are working
                print("✅ Keyboard shortcuts enabled: Ctrl+P (pause), Ctrl+R (resume)")
                self.logger.info("✅ Keyboard shortcuts enabled: Ctrl+P (pause), Ctrl+R (resume)")
                
            except Exception as e:
                print(f"❌ Could not setup keyboard shortcuts: {e}")
                self.logger.warning(f"Could not setup keyboard shortcuts: {e}")
        else:
            print("⚠️  Keyboard shortcuts disabled - use Ctrl+C to interrupt")
            self.logger.info("⚠️  Keyboard shortcuts disabled - use Ctrl+C to interrupt")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        self.logger.info(f"Received signal {signum}, saving checkpoint and exiting...")
        self.save_checkpoint()
        sys.exit(0)
    
    def set_callbacks(self, on_pause: Optional[Callable] = None, 
                     on_resume: Optional[Callable] = None,
                     on_checkpoint: Optional[Callable] = None):
        """
        Set callback functions for pause/resume events.
        
        Args:
            on_pause: Function to call when training is paused
            on_resume: Function to call when training is resumed
            on_checkpoint: Function to call when checkpoint is saved
        """
        self.on_pause_callback = on_pause
        self.on_resume_callback = on_resume
        self.on_checkpoint_callback = on_checkpoint
    
    def pause_training(self):
        """Pause training and save checkpoint"""
        if not self.training_paused:
            self.training_paused = True
            self.pause_requested = True
            self.logger.info("⏸️ Training PAUSED by user (Ctrl+P)")
            
            # Save checkpoint
            self.save_checkpoint()
            
            # Call pause callback if set
            if self.on_pause_callback:
                try:
                    self.on_pause_callback()
                except Exception as e:
                    self.logger.error(f"Error in pause callback: {e}")
            
            print("\n" + "="*50)
            print("⏸️ TRAINING PAUSED")
            print("Press Ctrl+R to resume or Ctrl+C to exit")
            print("="*50)
    
    def resume_training(self):
        """Resume training from checkpoint"""
        if self.training_paused:
            self.training_paused = False
            self.training_resumed = True
            self.resume_requested = True
            self.logger.info("▶️ Training RESUMED by user (Ctrl+R)")
            
            # Call resume callback if set
            if self.on_resume_callback:
                try:
                    self.on_resume_callback()
                except Exception as e:
                    self.logger.error(f"Error in resume callback: {e}")
            
            print("\n" + "="*50)
            print("▶️ TRAINING RESUMED")
            print("="*50)
    
    def save_checkpoint(self, data: Optional[Dict[str, Any]] = None):
        """
        Save training checkpoint.
        
        Args:
            data: Additional data to save in checkpoint
        """
        try:
            checkpoint = {
                'timestamp': datetime.now().isoformat(),
                'training_paused': self.training_paused,
                'training_resumed': self.training_resumed,
                'checkpoint_interval': self.checkpoint_interval,
                'data': data or {}
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, cls=NumpyEncoder)
            
            self.last_checkpoint_time = time.time()
            self.logger.info(f"💾 Checkpoint saved: {self.checkpoint_file}")
            
            # Call checkpoint callback if set
            if self.on_checkpoint_callback:
                try:
                    self.on_checkpoint_callback(checkpoint)
                except Exception as e:
                    self.logger.error(f"Error in checkpoint callback: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load training checkpoint if available.
        
        Returns:
            Checkpoint data if available, None otherwise
        """
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                self.logger.info(f"📂 Checkpoint loaded from: {self.checkpoint_file}")
                self.logger.info(f"📅 Checkpoint timestamp: {checkpoint.get('timestamp', 'unknown')}")
                
                # Restore state
                self.training_paused = checkpoint.get('training_paused', False)
                self.training_resumed = checkpoint.get('training_resumed', False)
                self.checkpoint_interval = checkpoint.get('checkpoint_interval', 300)
                
                return checkpoint.get('data', {})
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def should_save_checkpoint(self) -> bool:
        """Check if it's time to save a checkpoint"""
        if self.last_checkpoint_time is None:
            return True
        return (time.time() - self.last_checkpoint_time) >= self.checkpoint_interval
    
    def is_paused(self) -> bool:
        """Check if training is currently paused"""
        return self.training_paused
    
    def is_resumed(self) -> bool:
        """Check if training was resumed"""
        return self.training_resumed
    
    def wait_if_paused(self):
        """Wait if training is paused, checking for resume"""
        while self.training_paused and not self._stop_monitoring:
            time.sleep(0.1)  # Small delay to prevent high CPU usage
    
    def optimize_with_pause_support(self, study, objective, n_trials, timeout=None, **kwargs):
        """
        Custom optimization method that supports pausing between trials.
        
        Args:
            study: Optuna study object
            objective: Objective function
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            **kwargs: Additional arguments for study.optimize
        
        Returns:
            The study object after optimization
        """
        self.logger.info(f"🚀 Starting optimization with pause support: {n_trials} trials")
        
        # Track start time for timeout
        start_time = time.time()
        completed_trials = 0
        
        while completed_trials < n_trials:
            # Check for timeout
            if timeout and (time.time() - start_time) > timeout:
                self.logger.info(f"⏰ Optimization timeout reached after {timeout} seconds")
                break
            
            # Check if paused
            if self.training_paused:
                self.logger.info("⏸️ Optimization paused, waiting for resume...")
                self.wait_if_paused()
                if self._stop_monitoring:
                    self.logger.info("🛑 Optimization stopped by user")
                    break
                self.logger.info("▶️ Optimization resumed")
            
            # Run one trial at a time
            try:
                # Run single trial directly on the original study
                study.optimize(objective, n_trials=1, **kwargs)
                completed_trials += 1
                
                self.logger.info(f"✅ Trial {completed_trials}/{n_trials} completed")
                
                # Save checkpoint periodically
                if completed_trials % 5 == 0:
                    self.save_checkpoint({
                        'completed_trials': completed_trials,
                        'total_trials': n_trials,
                        'best_value': study.best_value if study.trials else None
                    })
                
            except Exception as e:
                self.logger.error(f"Error in trial {completed_trials + 1}: {e}")
                completed_trials += 1  # Count failed trials too
        
        self.logger.info(f"🏁 Optimization completed: {completed_trials} trials")
        return study
    
    def start_monitoring(self):
        """Start the monitoring thread for automatic checkpoints"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self._stop_monitoring = False
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("🔍 Started checkpoint monitoring thread")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self._stop_monitoring = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
        self.logger.info("🛑 Stopped checkpoint monitoring thread")
    
    def _monitor_loop(self):
        """Monitoring loop for automatic checkpoints"""
        while not self._stop_monitoring:
            try:
                if self.should_save_checkpoint():
                    self.save_checkpoint()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the controller"""
        return {
            'training_paused': self.training_paused,
            'training_resumed': self.training_resumed,
            'pause_requested': self.pause_requested,
            'resume_requested': self.resume_requested,
            'checkpoint_file': self.checkpoint_file,
            'last_checkpoint_time': self.last_checkpoint_time,
            'keyboard_available': KEYBOARD_AVAILABLE,
            'monitor_thread_alive': self.monitor_thread.is_alive() if self.monitor_thread else False
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        if KEYBOARD_AVAILABLE:
            try:
                keyboard.unhook_all()
            except Exception as e:
                self.logger.warning(f"Error unhooking keyboard: {e}")
        self.logger.info("🧹 Pause/Resume Controller cleaned up")

# Global instance for easy access
_controller = None

def get_controller() -> PauseResumeController:
    """Get the global controller instance"""
    global _controller
    if _controller is None:
        _controller = PauseResumeController()
    return _controller

def setup_pause_resume(checkpoint_file: str = 'training_checkpoint.json',
                      checkpoint_interval: int = 300) -> PauseResumeController:
    """
    Setup pause/resume functionality.
    
    Args:
        checkpoint_file: Path to save/load checkpoints
        checkpoint_interval: Seconds between automatic checkpoints
    
    Returns:
        PauseResumeController instance
    """
    global _controller
    _controller = PauseResumeController(checkpoint_file, checkpoint_interval)
    return _controller

def pause_training():
    """Pause training (global function)"""
    controller = get_controller()
    controller.pause_training()

def resume_training():
    """Resume training (global function)"""
    controller = get_controller()
    controller.resume_training()

def save_checkpoint(data: Optional[Dict[str, Any]] = None):
    """Save checkpoint (global function)"""
    controller = get_controller()
    controller.save_checkpoint(data)

def load_checkpoint() -> Optional[Dict[str, Any]]:
    """Load checkpoint (global function)"""
    controller = get_controller()
    return controller.load_checkpoint()

def is_paused() -> bool:
    """Check if training is paused (global function)"""
    controller = get_controller()
    return controller.is_paused()

def wait_if_paused():
    """Wait if training is paused (global function)"""
    controller = get_controller()
    controller.wait_if_paused()

def optimize_with_pause_support(study, objective, n_trials, timeout=None, **kwargs):
    """Optimize with pause support (global function)"""
    controller = get_controller()
    return controller.optimize_with_pause_support(study, objective, n_trials, timeout, **kwargs) 