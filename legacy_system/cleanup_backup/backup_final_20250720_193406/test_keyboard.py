#!/usr/bin/env python3
"""
Simple test script to check if keyboard shortcuts are working
"""

import time
import sys

try:
    import keyboard
    print("✅ Keyboard module imported successfully")
    
    def test_pause():
        print("⏸️ PAUSE triggered!")
    
    def test_resume():
        print("▶️ RESUME triggered!")
    
    # Setup keyboard shortcuts
    keyboard.add_hotkey('ctrl+p', test_pause)
    keyboard.add_hotkey('ctrl+r', test_resume)
    
    print("✅ Keyboard shortcuts set up:")
    print("   - Ctrl+P: Pause")
    print("   - Ctrl+R: Resume")
    print("   - Ctrl+C: Exit")
    print("\nPress Ctrl+P or Ctrl+R to test, or Ctrl+C to exit...")
    
    # Keep the script running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
        
except ImportError as e:
    print(f"❌ Could not import keyboard: {e}")
except Exception as e:
    print(f"❌ Error setting up keyboard: {e}") 