#!/usr/bin/env python3
"""
AUTONOMOUS MANAGER - CONTINUOUS LEARNING & IMPROVEMENT
=====================================================

This is the main autonomous system that runs continuously to:
1. Monitor bot performance
2. Trigger retraining when needed
3. Optimize all parameters automatically
4. Send notifications about improvements
5. Handle all learning and enhancement autonomously

USAGE:
- Start: python autonomous_manager.py --start
- Stop: python autonomous_manager.py --stop
- Status: python autonomous_manager.py --status
- Monitor: python autonomous_manager.py --monitor
"""

import os
import sys
import json
import time
import logging
import threading
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import signal
import atexit
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultra_train_enhanced import UltraEnhancedTrainer
from modules.telegram_bot import TelegramNotifier

class AutonomousManager:
    """
    Autonomous Manager for continuous learning and improvement
    """
    
    def __init__(self):
        self.running = False
        self.trainer = None
        self.telegram_notifier = None
        self.monitor_thread = None
        self.last_status_check = None
        self.performance_history = []
        self.retrain_count = 0
        self.improvement_count = 0
        
        # Load config
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Setup Telegram notifications
        api_keys = self.config.get('api_keys', {})
        telegram_token = api_keys.get('telegram_bot_token')
        telegram_chat_id = api_keys.get('telegram_chat_id')
        
        if telegram_token and telegram_chat_id:
            self.telegram_notifier = TelegramNotifier(telegram_token, telegram_chat_id)
        
        # Setup logging
        self.setup_logging()
        
        # Autonomous configuration
        self.autonomous_config = {
            'check_interval_minutes': 30,  # Check every 30 minutes
            'retrain_interval_hours': 12,  # Retrain every 12 hours
            'performance_threshold': 0.7,  # Retrain if performance drops below 70%
            'data_freshness_hours': 3,     # Use data from last 3 hours
            'max_retrain_attempts': 3,     # Max retrain attempts per day
            'notification_interval_hours': 6,  # Send status every 6 hours
            'emergency_retrain_threshold': 0.5  # Emergency retrain if performance < 50%
        }
        
        self.logger.info("🤖 Autonomous Manager initialized")
    
    def setup_logging(self):
        """Setup logging for autonomous manager"""
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Create timestamp for log files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Configure logger
        self.logger = logging.getLogger('AutonomousManager')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(f'logs/autonomous_manager_{timestamp}.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def send_notification(self, message: str):
        """Send notification via Telegram"""
        if self.telegram_notifier:
            try:
                self.telegram_notifier.send_message(message)
                self.logger.info(f"📱 Notification sent: {message}")
            except Exception as e:
                self.logger.error(f"Failed to send notification: {e}")
        else:
            self.logger.info(f"📱 Notification (no Telegram): {message}")
    
    def initialize_trainer(self):
        """Initialize the trainer with autonomous capabilities"""
        try:
            self.logger.info("🧠 Initializing Ultra Enhanced Trainer...")
            self.trainer = UltraEnhancedTrainer()
            self.trainer.setup_autonomous_training()
            
            # Enable all autonomous features
            self.trainer.online_learning_enabled = True
            self.trainer.meta_learning_enabled = True
            self.trainer.self_feature_engineering = True
            self.trainer.self_repair_enabled = True
            
            self.logger.info("✅ Trainer initialized with all autonomous features")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trainer: {e}")
            return False
    
    def check_performance(self) -> float:
        """Check current bot performance"""
        try:
            if not self.trainer:
                return 0.0
            
            # Get autonomous status
            status = self.trainer.get_autonomous_status()
            current_performance = status.get('best_performance', 0.0)
            
            # Add to history
            self.performance_history.append(current_performance)
            if len(self.performance_history) > 100:
                self.performance_history.pop(0)
            
            self.logger.info(f"📊 Current performance: {current_performance:.3f}")
            return current_performance
            
        except Exception as e:
            self.logger.error(f"Error checking performance: {e}")
            return 0.0
    
    def should_retrain(self) -> bool:
        """Determine if retraining is needed"""
        try:
            current_time = datetime.now()
            
            # Check if we've exceeded max retrain attempts today
            today = current_time.date()
            today_retrains = sum(1 for p in self.performance_history 
                               if datetime.fromtimestamp(p).date() == today)
            
            if today_retrains >= self.autonomous_config['max_retrain_attempts']:
                self.logger.info("📅 Max retrain attempts reached for today")
                return False
            
            # Check performance-based retraining
            if len(self.performance_history) > 5:
                recent_performance = np.mean(self.performance_history[-5:])
                
                # Emergency retrain if performance is very low
                if recent_performance < self.autonomous_config['emergency_retrain_threshold']:
                    self.logger.warning(f"🚨 EMERGENCY RETRAIN: Performance {recent_performance:.3f} below threshold")
                    return True
                
                # Regular retrain if performance is below threshold
                if recent_performance < self.autonomous_config['performance_threshold']:
                    self.logger.info(f"📉 Performance-based retrain: {recent_performance:.3f} below threshold")
                    return True
            
            # Check time-based retraining
            if self.last_status_check:
                time_since_last = current_time - self.last_status_check
                if time_since_last.total_seconds() > self.autonomous_config['retrain_interval_hours'] * 3600:
                    self.logger.info("⏰ Time-based retraining triggered")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking retrain conditions: {e}")
            return False
    
    def perform_autonomous_retraining(self):
        """Perform autonomous retraining with all enhancements"""
        try:
            self.logger.info("🚀 Starting autonomous retraining...")
            self.send_notification("🤖 Starting autonomous retraining...")
            
            # Collect fresh data
            fresh_data = self.trainer.collect_enhanced_training_data(days=7)
            if fresh_data.empty:
                self.logger.warning("No fresh data available for retraining")
                return False
            
            # Prepare features
            X, y_1m, y_5m, y_15m = self.trainer.prepare_features(fresh_data)
            
            # Train models with all enhancements
            self.trainer.train_10x_intelligence_models(X, y_1m, y_5m, y_15m)
            
            # Perform all autonomous optimizations
            self.trainer.perform_meta_learning()
            self.trainer.perform_self_feature_engineering(X, y_1m)
            self.trainer.perform_self_repair()
            
            # Optimize for maximum profits
            if hasattr(self.trainer, '_optimize_for_maximum_profits'):
                self.trainer._optimize_for_maximum_profits()
            
            # Evaluate new performance
            new_performance = self.trainer._evaluate_autonomous_performance()
            
            # Check if performance improved
            if new_performance > max(self.performance_history[-10:], default=0):
                self.improvement_count += 1
                self.send_notification(f"🎉 PERFORMANCE IMPROVED! New score: {new_performance:.3f}")
                self.logger.info(f"🎉 Performance improved to {new_performance:.3f}")
            else:
                self.logger.info(f"📊 Performance: {new_performance:.3f}")
            
            # Save models
            self.trainer.save_10x_intelligence_models()
            
            self.retrain_count += 1
            self.last_status_check = datetime.now()
            
            self.logger.info("✅ Autonomous retraining completed")
            self.send_notification("✅ Autonomous retraining completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in autonomous retraining: {e}")
            self.send_notification(f"❌ Autonomous retraining failed: {str(e)}")
            return False
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.logger.info("🔍 Performing autonomous monitoring check...")
                
                # Check performance
                current_performance = self.check_performance()
                
                # Check if retraining is needed
                if self.should_retrain():
                    self.perform_autonomous_retraining()
                
                # Send periodic status updates
                if (not self.last_status_check or 
                    (datetime.now() - self.last_status_check).total_seconds() > 
                    self.autonomous_config['notification_interval_hours'] * 3600):
                    
                    status_message = (
                        f"🤖 AUTONOMOUS STATUS UPDATE\n"
                        f"📊 Current Performance: {current_performance:.3f}\n"
                        f"🔄 Retrain Count: {self.retrain_count}\n"
                        f"📈 Improvements: {self.improvement_count}\n"
                        f"⏰ Next Check: {self.autonomous_config['check_interval_minutes']} minutes"
                    )
                    self.send_notification(status_message)
                    self.last_status_check = datetime.now()
                
                # Sleep until next check
                sleep_seconds = self.autonomous_config['check_interval_minutes'] * 60
                self.logger.info(f"💤 Sleeping for {self.autonomous_config['check_interval_minutes']} minutes...")
                time.sleep(sleep_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                time.sleep(300)  # Sleep 5 minutes on error
    
    def start(self):
        """Start the autonomous manager"""
        if self.running:
            self.logger.warning("Autonomous manager is already running")
            return False
        
        self.logger.info("🚀 Starting Autonomous Manager...")
        self.send_notification("🚀 Autonomous Manager starting...")
        
        # Initialize trainer
        if not self.initialize_trainer():
            return False
        
        # Start monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("✅ Autonomous Manager started successfully")
        self.send_notification("✅ Autonomous Manager is now running and monitoring")
        
        return True
    
    def stop(self):
        """Stop the autonomous manager"""
        if not self.running:
            self.logger.warning("Autonomous manager is not running")
            return False
        
        self.logger.info("🛑 Stopping Autonomous Manager...")
        self.send_notification("🛑 Autonomous Manager stopping...")
        
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        if self.trainer:
            self.trainer.stop_autonomous_training()
        
        self.logger.info("✅ Autonomous Manager stopped")
        self.send_notification("✅ Autonomous Manager stopped")
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        status = {
            'running': self.running,
            'retrain_count': self.retrain_count,
            'improvement_count': self.improvement_count,
            'current_performance': self.performance_history[-1] if self.performance_history else 0.0,
            'performance_history': self.performance_history[-10:],
            'last_status_check': self.last_status_check.isoformat() if self.last_status_check else None,
            'autonomous_config': self.autonomous_config
        }
        
        if self.trainer:
            trainer_status = self.trainer.get_autonomous_status()
            status['trainer_status'] = trainer_status
        
        return status
    
    def monitor(self):
        """Run in monitor mode (continuous output)"""
        if not self.running:
            print("❌ Autonomous Manager is not running. Start it first with --start")
            return
        
        print("🔍 Monitoring Autonomous Manager (Press Ctrl+C to stop)...")
        print("=" * 60)
        
        try:
            while self.running:
                status = self.get_status()
                
                print(f"\n📊 STATUS UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"🔄 Running: {status['running']}")
                print(f"📈 Retrain Count: {status['retrain_count']}")
                print(f"🎉 Improvements: {status['improvement_count']}")
                print(f"📊 Current Performance: {status['current_performance']:.3f}")
                
                if status.get('trainer_status'):
                    trainer_status = status['trainer_status']
                    print(f"🤖 Trainer Running: {trainer_status.get('autonomous_running', False)}")
                    print(f"⏰ Next Retrain: {trainer_status.get('next_retrain_hours', 0):.1f} hours")
                
                print("-" * 60)
                time.sleep(60)  # Update every minute
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Autonomous Manager for Trading Bot')
    parser.add_argument('--start', action='store_true', help='Start autonomous manager')
    parser.add_argument('--stop', action='store_true', help='Stop autonomous manager')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--monitor', action='store_true', help='Monitor in real-time')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon (background)')
    
    args = parser.parse_args()
    
    # Create manager
    manager = AutonomousManager()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, stopping...")
        manager.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup
    atexit.register(manager.stop)
    
    if args.start:
        print("🚀 Starting Autonomous Manager...")
        if manager.start():
            if args.daemon:
                print("✅ Autonomous Manager started in background")
                print("Use --status to check status, --stop to stop")
                # Keep running in background
                try:
                    while manager.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    manager.stop()
            else:
                print("✅ Autonomous Manager started")
                print("Press Ctrl+C to stop")
                try:
                    while manager.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    manager.stop()
        else:
            print("❌ Failed to start Autonomous Manager")
            sys.exit(1)
    
    elif args.stop:
        print("🛑 Stopping Autonomous Manager...")
        if manager.stop():
            print("✅ Autonomous Manager stopped")
        else:
            print("❌ Failed to stop Autonomous Manager")
            sys.exit(1)
    
    elif args.status:
        status = manager.get_status()
        print("🤖 AUTONOMOUS MANAGER STATUS")
        print("=" * 40)
        print(f"Running: {status['running']}")
        print(f"Retrain Count: {status['retrain_count']}")
        print(f"Improvement Count: {status['improvement_count']}")
        print(f"Current Performance: {status['current_performance']:.3f}")
        print(f"Last Status Check: {status['last_status_check']}")
        
        if status.get('trainer_status'):
            trainer_status = status['trainer_status']
            print(f"\n🤖 TRAINER STATUS:")
            print(f"Autonomous Running: {trainer_status.get('autonomous_running', False)}")
            print(f"Best Performance: {trainer_status.get('best_performance', 0):.3f}")
            print(f"Next Retrain: {trainer_status.get('next_retrain_hours', 0):.1f} hours")
    
    elif args.monitor:
        manager.monitor()
    
    else:
        print("🤖 AUTONOMOUS MANAGER")
        print("=" * 30)
        print("Usage:")
        print("  --start     : Start autonomous manager")
        print("  --stop      : Stop autonomous manager")
        print("  --status    : Show current status")
        print("  --monitor   : Monitor in real-time")
        print("  --daemon    : Run in background mode")
        print("\nExample:")
        print("  python autonomous_manager.py --start --daemon")
        print("  python autonomous_manager.py --status")
        print("  python autonomous_manager.py --monitor")

if __name__ == "__main__":
    main() 