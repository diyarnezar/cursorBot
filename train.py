#!/usr/bin/env python3
"""
🚀 PROJECT HYPERION - MAIN TRAINING SCRIPT
==========================================

Professional, lightweight training entry point that integrates with all existing modules.
Uses the modular architecture for maximum organization and maintainability.

Author: Project Hyperion Team
Date: 2025
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the professional training orchestrator
from training.orchestrator import TrainingOrchestrator
from config.training_config import training_config
from utils.logging.logger import start_logging_session, get_session_logger, end_logging_session, log_system_info, log_error

def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(
        description="🚀 Project Hyperion - Professional Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  test         - 15 minutes (fast test)
  ultra_short  - 30 minutes (ultra-fast test)
  ultra_fast   - 2 hours (fast testing)
  quick        - 1 day (quick training)
  month        - 30 days (monthly training)
  quarter      - 3 months (quarterly training)
  half_year    - 6 months (semi-annual training)
  year         - 1 year (annual training)
  two_year     - 2 years (extended training)

Examples:
  python train.py --mode test
  python train.py --mode month
  python train.py --mode quarter --interactive
  python train.py --list-modes
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=['test', 'ultra_short', 'ultra_fast', 'quick', 'month', 'quarter', 'half_year', 'year', 'two_year'],
        help="Training mode"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["ETHFDUSD"],
        help="Trading symbols (default: ETHFDUSD)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with menu"
    )
    
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List all available training modes"
    )
    
    parser.add_argument(
        "--config",
        default="config.json",
        help="Configuration file path (default: config.json)"
    )
    
    args = parser.parse_args()
    
    # Start session-based logging
    session_name = f"main_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_id = start_logging_session(session_name)
    
    # Get main logger
    logger = get_session_logger("main")
    
    try:
        # Log system info
        log_system_info(logger)
        
        # Create training orchestrator
        orchestrator = TrainingOrchestrator(args.config)
        
        # List modes if requested
        if args.list_modes:
            orchestrator.list_training_modes()
            end_logging_session()
            return
        
        # Interactive mode (default when no mode specified)
        if args.interactive or not args.mode:
            orchestrator.run_interactive()
            end_logging_session()
            return
        
        # Run training
        success = orchestrator.train_mode(args.mode, args.symbols)
        
        if success:
            print(f"\n🎉 {args.mode.upper()} training completed successfully!")
            logger.info(f"{args.mode.upper()} training completed successfully")
        else:
            print(f"\n❌ {args.mode.upper()} training failed!")
            logger.error(f"{args.mode.upper()} training failed")
            end_logging_session()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user")
        logger.warning("Training interrupted by user")
    except Exception as e:
        log_error(logger, e, "main training")
        print(f"\n❌ Training error: {e}")
        end_logging_session()
        sys.exit(1)
    
    # End session
    end_logging_session()


if __name__ == "__main__":
    main() 