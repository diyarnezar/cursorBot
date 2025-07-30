#!/usr/bin/env python3
"""
ULTRA MULTI-PAIR TRAINING SCRIPT
================================

Advanced training system for all 26 FDUSD pairs at ETH/FDUSD level sophistication.
This script provides the same training modes as ultra_train_enhanced.py but for all pairs.
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.multi_pair_trainer import MultiPairTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/multipair_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def show_multipair_menu():
    """Show the multi-pair training menu"""
    print("\n" + "=" * 70)
    print("MULTI-PAIR TRAINING - ALL 26 FDUSD PAIRS")
    print("ETH/FDUSD Level Sophistication for All Pairs")
    print("=" * 70)
    print("\nChoose Multi-Pair Training Mode:")
    print("0. Ultra-Short Test (30 minutes) - All 26 pairs")
    print("1. Ultra-Fast Testing (2 hours) - All 26 pairs")
    print("2. Quick Training (1 day) - All 26 pairs")
    print("3. Full Training (7 days) - All 26 pairs")
    print("4. Extended Training (15 days) - All 26 pairs")
    print("5. Full Historical Data - All 26 pairs")
    print("6. Custom Days - Specify number of days")
    print("7. Exit")

def get_training_days(choice):
    """Get training days based on choice"""
    if choice == "0":
        return 0.021  # 30 minutes
    elif choice == "1":
        return 0.083  # 2 hours
    elif choice == "2":
        return 1.0    # 1 day
    elif choice == "3":
        return 7.0    # 7 days
    elif choice == "4":
        return 15.0   # 15 days
    elif choice == "5":
        return 30.0   # Full historical
    else:
        return 15.0   # Default

def main():
    """Main function for multi-pair training"""
    print("PROJECT HYPERION - MULTI-PAIR TRAINING")
    print("=" * 50)
    print("Training all 26 FDUSD pairs at ETH/FDUSD level")
    print("=" * 50)

    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("\nMulti-Pair Training Modes:")
            print("  --mode fast        : Ultra-Short Test (30 minutes) - All 26 pairs")
            print("  --mode ultra-fast  : Ultra-Fast Testing (2 hours) - All 26 pairs")
            print("  --mode 1day        : Quick Training (1 day) - All 26 pairs")
            print("  --mode full        : Full Training (7 days) - All 26 pairs")
            print("  --mode 15days      : Extended Training (15 days) - All 26 pairs")
            print("  --mode historical  : Full Historical Data - All 26 pairs")
            print("  --mode custom      : Custom days (specify with --days)")
            print("\nExamples:")
            print("  python ultra_train_multipair.py --mode fast")
            print("  python ultra_train_multipair.py --mode custom --days 10")
            return
        
        if sys.argv[1] == "--mode" and len(sys.argv) > 2:
            mode_arg = sys.argv[2].lower()
            days = get_training_days(mode_arg)
            
            # Handle custom days
            if mode_arg == "custom" and len(sys.argv) > 3 and sys.argv[3] == "--days":
                if len(sys.argv) > 4:
                    try:
                        days = float(sys.argv[4])
                    except ValueError:
                        print("❌ Invalid days value")
                        return
                else:
                    print("❌ --days requires a value")
                    return
            
            print(f"\nStarting Multi-Pair Training: {mode_arg} mode ({days} days)")
            run_multipair_training(days)
            return

    # Interactive mode
    while True:
        show_multipair_menu()
        
        try:
            choice = input("\nEnter your choice (0-7): ").strip()
        except KeyboardInterrupt:
            print("\nTraining cancelled by user.")
            return

        if choice == "7":
            print("Exiting multi-pair training.")
            return
        elif choice == "6":
            try:
                custom_days = float(input("Enter number of days for training: "))
                print(f"\nStarting Custom Multi-Pair Training ({custom_days} days)")
                run_multipair_training(custom_days)
            except ValueError:
                print("❌ Invalid number of days")
                continue
        elif choice in ["0", "1", "2", "3", "4", "5"]:
            days = get_training_days(choice)
            mode_names = {
                "0": "Ultra-Short Test",
                "1": "Ultra-Fast Testing", 
                "2": "Quick Training",
                "3": "Full Training",
                "4": "Extended Training",
                "5": "Full Historical Data"
            }
            print(f"\nStarting {mode_names[choice]} Multi-Pair Training ({days} days)")
            run_multipair_training(days)
        else:
            print("❌ Invalid choice. Please enter 0-7.")

def run_multipair_training(days):
    """Run multi-pair training with specified days"""
    try:
        logger.info(f"🚀 Starting Multi-Pair Training for {days} days...")
        
        # Initialize multi-pair trainer
        multi_trainer = MultiPairTrainer()
        
        # Train all pairs
        results = multi_trainer.train_all_pairs(days=days)
        
        # Save all models
        multi_trainer.save_all_models()
        
        # Show results
        successful = sum(results.values())
        total = len(results)
        
        print("\n" + "=" * 70)
        print("MULTI-PAIR TRAINING COMPLETED!")
        print("=" * 70)
        print(f"✅ Successful: {successful}/{total} pairs")
        print(f"❌ Failed: {total - successful}/{total} pairs")
        
        if successful > 0:
            print(f"\n🎉 Successfully trained {successful} pairs!")
            print("📁 Models saved in: models/multi_pair/")
            print("📊 Training results saved in: models/multi_pair/training_results.json")
            
            # Show successful pairs
            successful_pairs = [pair for pair, success in results.items() if success]
            print(f"\n✅ Successfully trained pairs: {', '.join(successful_pairs)}")
            
            # Show failed pairs
            failed_pairs = [pair for pair, success in results.items() if not success]
            if failed_pairs:
                print(f"\n❌ Failed pairs: {', '.join(failed_pairs)}")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Your bot is now trained on all successful pairs")
        print("2. Models are ready for multi-pair trading")
        print("3. You can now run the main bot with multi-pair support")
        
        logger.info("Multi-Pair Training completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in multi-pair training: {e}")
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main() 