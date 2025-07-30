#!/usr/bin/env python3
"""
Test script to verify pause/resume functionality
"""

import time
import sys
from modules.pause_resume_controller import setup_pause_resume

def test_pause_callback():
    print("⏸️ PAUSE CALLBACK TRIGGERED!")
    print("   - Models would be saved here")
    print("   - Training state would be preserved")

def test_resume_callback():
    print("▶️ RESUME CALLBACK TRIGGERED!")
    print("   - Training would continue from checkpoint")

def test_checkpoint_callback(checkpoint_data):
    print("💾 CHECKPOINT CALLBACK TRIGGERED!")
    print(f"   - Checkpoint data: {checkpoint_data}")

def main():
    print("🧪 Testing Pause/Resume Functionality")
    print("=" * 50)
    
    # Setup pause/resume controller
    controller = setup_pause_resume(
        checkpoint_file='test_checkpoint.json',
        checkpoint_interval=30  # 30 seconds for testing
    )
    
    # Set up callbacks
    controller.set_callbacks(
        on_pause=test_pause_callback,
        on_resume=test_resume_callback,
        on_checkpoint=test_checkpoint_callback
    )
    
    # Start monitoring
    controller.start_monitoring()
    
    print("\n✅ Pause/Resume Controller initialized!")
    print("🎹 Keyboard shortcuts should be active:")
    print("   - Ctrl+P: Pause training")
    print("   - Ctrl+R: Resume training")
    print("   - Ctrl+C: Exit test")
    print("\n⏰ Automatic checkpoints every 30 seconds")
    print("📁 Checkpoint file: test_checkpoint.json")
    
    print("\n" + "=" * 50)
    print("🔍 TESTING IN PROGRESS...")
    print("Try pressing Ctrl+P or Ctrl+R now!")
    print("=" * 50)
    
    # Simulate some work
    counter = 0
    try:
        while True:
            counter += 1
            print(f"⏱️  Working... {counter} (Press Ctrl+P to pause, Ctrl+R to resume)")
            
            # Check if paused
            if controller.is_paused():
                print("⏸️  PAUSED - waiting for resume...")
                controller.wait_if_paused()
                print("▶️  RESUMED - continuing...")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        controller.cleanup()
        print("✅ Test completed successfully!")

if __name__ == "__main__":
    main() 