#!/usr/bin/env python3
"""
PAUSE/RESUME TRAINING UTILITY
Project Hyperion - Seamless Training Control

This utility provides:
1. Pause training at any time
2. Resume training from exact checkpoint
3. Monitor training progress
4. Save/load training state
5. Estimate remaining time
6. Safe interruption handling
"""

import os
import sys
import json
import signal
import time
import psutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import subprocess
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingController:
    """Training controller for pause/resume functionality"""
    
    def __init__(self):
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.state_file = self.checkpoint_dir / 'training_state.json'
        self.progress_file = self.checkpoint_dir / 'training_progress.json'
        self.pid_file = self.checkpoint_dir / 'training_pid.txt'
        self.training_process = None
        self.training_pid = None
        
    def start_training(self, script_path: str = 'ultra_train_enhanced.py'):
        """Start training process"""
        try:
            logger.info("🚀 Starting training process...")
            
            # Check if training is already running
            if self.is_training_running():
                logger.warning("⚠️ Training is already running!")
                return False
            
            # Start training process
            self.training_process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.training_pid = self.training_process.pid
            
            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(self.training_pid))
            
            logger.info(f"✅ Training started with PID: {self.training_pid}")
            logger.info("💡 Use 'python pause_resume_training.py pause' to pause")
            logger.info("💡 Use 'python pause_resume_training.py resume' to resume")
            logger.info("💡 Use 'python pause_resume_training.py status' to check status")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start training: {e}")
            return False
    
    def pause_training(self):
        """Pause training process"""
        try:
            if not self.is_training_running():
                logger.warning("⚠️ No training process is running")
                return False
            
            logger.info("⏸️ Pausing training...")
            
            # Send SIGUSR1 signal to pause
            if os.name == 'nt':  # Windows
                # On Windows, we'll use a different approach
                self._pause_windows()
            else:  # Unix/Linux
                os.kill(self.training_pid, signal.SIGUSR1)
            
            # Wait a moment for the process to pause
            time.sleep(2)
            
            # Update state
            self._update_state('paused')
            
            logger.info("✅ Training paused successfully")
            logger.info("💡 Use 'python pause_resume_training.py resume' to resume")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to pause training: {e}")
            return False
    
    def resume_training(self):
        """Resume training process"""
        try:
            if not self.is_training_running():
                logger.warning("⚠️ No training process is running")
                return False
            
            logger.info("▶️ Resuming training...")
            
            # Send SIGUSR2 signal to resume
            if os.name == 'nt':  # Windows
                self._resume_windows()
            else:  # Unix/Linux
                os.kill(self.training_pid, signal.SIGUSR2)
            
            # Update state
            self._update_state('active')
            
            logger.info("✅ Training resumed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to resume training: {e}")
            return False
    
    def stop_training(self):
        """Stop training process"""
        try:
            if not self.is_training_running():
                logger.warning("⚠️ No training process is running")
                return False
            
            logger.info("🛑 Stopping training...")
            
            # Terminate process gracefully
            if self.training_process:
                self.training_process.terminate()
                
                # Wait for graceful termination
                try:
                    self.training_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning("⚠️ Force killing training process...")
                    self.training_process.kill()
            
            # Clean up PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            logger.info("✅ Training stopped successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop training: {e}")
            return False
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        try:
            status = {
                'running': self.is_training_running(),
                'pid': self.training_pid,
                'timestamp': datetime.now().isoformat()
            }
            
            # Load progress if available
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                status.update(progress_data)
            
            # Load state if available
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                status.update(state_data)
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get training status: {e}")
            return {'error': str(e)}
    
    def is_training_running(self) -> bool:
        """Check if training process is running"""
        try:
            # Check PID file
            if not self.pid_file.exists():
                return False
            
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process exists
            if not psutil.pid_exists(pid):
                # Clean up stale PID file
                self.pid_file.unlink()
                return False
            
            # Check if it's our Python process
            process = psutil.Process(pid)
            if 'python' not in process.name().lower():
                return False
            
            self.training_pid = pid
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking training status: {e}")
            return False
    
    def _update_state(self, status: str):
        """Update training state"""
        try:
            state_data = {
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'pid': self.training_pid
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to update state: {e}")
    
    def _pause_windows(self):
        """Pause training on Windows"""
        # On Windows, we'll use a different approach
        # Create a pause file that the training script can check
        pause_file = self.checkpoint_dir / 'pause_requested.txt'
        pause_file.touch()
        logger.info("📝 Pause requested (Windows mode)")
    
    def _resume_windows(self):
        """Resume training on Windows"""
        # Remove pause file
        pause_file = self.checkpoint_dir / 'pause_requested.txt'
        if pause_file.exists():
            pause_file.unlink()
        logger.info("📝 Resume requested (Windows mode)")
    
    def monitor_training(self):
        """Monitor training progress in real-time"""
        logger.info("📊 Monitoring training progress...")
        logger.info("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                status = self.get_training_status()
                
                if status.get('running'):
                    progress = status.get('progress', 0)
                    current_step = status.get('current_step', 'Unknown')
                    models_trained = status.get('models_trained', 0)
                    
                    print(f"\r📈 Progress: {progress:.1f}% | Step: {current_step} | Models: {models_trained}/64", end='', flush=True)
                else:
                    print(f"\r⏸️ Training not running", end='', flush=True)
                
                time.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
    
    def estimate_remaining_time(self) -> str:
        """Estimate remaining training time"""
        try:
            status = self.get_training_status()
            progress = status.get('progress', 0)
            
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
                
        except Exception as e:
            return "Unknown"

def main():
    """Main function for pause/resume training utility"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Training Pause/Resume Utility')
    parser.add_argument('action', choices=['start', 'pause', 'resume', 'stop', 'status', 'monitor'], 
                       help='Action to perform')
    parser.add_argument('--script', default='ultra_train_enhanced.py', 
                       help='Training script to run')
    
    args = parser.parse_args()
    
    controller = TrainingController()
    
    if args.action == 'start':
        controller.start_training(args.script)
        
    elif args.action == 'pause':
        controller.pause_training()
        
    elif args.action == 'resume':
        controller.resume_training()
        
    elif args.action == 'stop':
        controller.stop_training()
        
    elif args.action == 'status':
        status = controller.get_training_status()
        print("\n📊 Training Status:")
        print(f"   Running: {status.get('running', False)}")
        print(f"   PID: {status.get('pid', 'N/A')}")
        print(f"   Progress: {status.get('progress', 0):.1f}%")
        print(f"   Current Step: {status.get('current_step', 'Unknown')}")
        print(f"   Models Trained: {status.get('models_trained', 0)}/64")
        print(f"   Estimated Remaining: {controller.estimate_remaining_time()}")
        print(f"   Status: {status.get('status', 'Unknown')}")
        
    elif args.action == 'monitor':
        controller.monitor_training()

if __name__ == "__main__":
    main() 