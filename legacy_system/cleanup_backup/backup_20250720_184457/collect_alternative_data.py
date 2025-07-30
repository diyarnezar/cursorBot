#!/usr/bin/env python3
"""
PROJECT HYPERION - SAFE ALTERNATIVE DATA COLLECTION
Collects alternative data with proper rate limiting and safety measures.
Run this separately from training to avoid rate limit issues.
"""

import json
import logging
import time
import argparse
from datetime import datetime
import os
import sys

# Import project modules
from modules.alternative_data import EnhancedAlternativeData
from modules.telegram_bot import TelegramNotifier

def setup_logging():
    """Setup logging for alternative data collection."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('alternative_data.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str = 'config.json'):
    """Load configuration file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)

def collect_alternative_data(config_path: str = 'config.json', safe_mode: bool = True):
    """
    Collect alternative data with safety measures.
    
    Args:
        config_path: Path to configuration file
        safe_mode: If True, uses conservative rate limits and skips problematic APIs
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Initialize components
        api_keys = config.get('api_keys', {})
        alternative_data = EnhancedAlternativeData(
            api_keys=api_keys,
            data_dir=config['app_settings']['data_dir'],
            cache_expiry=config['cache_settings']['alternative_data_expiry_minutes'] * 60
        )
        
        telegram_notifier = TelegramNotifier(
            api_keys.get('telegram_bot_token', ''),
            api_keys.get('telegram_chat_id', '')
        )
        
        logging.info("Starting alternative data collection...")
        
        if telegram_notifier.enabled:
            telegram_notifier.send_message("ALTERNATIVE DATA COLLECTION STARTED\nCollecting external data with rate limit protection.")
        
        # Collect data with safety measures
        if safe_mode:
            logging.info("Running in SAFE MODE - using conservative rate limits")
            
            # Collect only essential data with longer delays
            data = {}
            
            # Social sentiment (with delays)
            logging.info("Collecting social sentiment data...")
            try:
                sentiment_data = alternative_data.get_social_sentiment()
                data.update(sentiment_data)
                time.sleep(30)  # 30 second delay between API calls
            except Exception as e:
                logging.warning(f"Failed to collect social sentiment: {e}")
            
            # News impact (with delays)
            logging.info("Collecting news impact data...")
            try:
                news_data = alternative_data.get_news_impact()
                data.update(news_data)
                time.sleep(30)  # 30 second delay
            except Exception as e:
                logging.warning(f"Failed to collect news data: {e}")
            
            # On-chain metrics (with delays)
            logging.info("Collecting on-chain metrics...")
            try:
                onchain_data = alternative_data.get_onchain_metrics()
                data.update(onchain_data)
                time.sleep(30)  # 30 second delay
            except Exception as e:
                logging.warning(f"Failed to collect on-chain data: {e}")
            
            # Exchange data (with delays)
            logging.info("Collecting exchange data...")
            try:
                exchange_data = alternative_data.get_exchange_data()
                data.update(exchange_data)
                time.sleep(30)  # 30 second delay
            except Exception as e:
                logging.warning(f"Failed to collect exchange data: {e}")
            
            # Whale activity (with delays)
            logging.info("Collecting whale activity data...")
            try:
                whale_data = alternative_data.get_whale_activity()
                data.update(whale_data)
                time.sleep(30)  # 30 second delay
            except Exception as e:
                logging.warning(f"Failed to collect whale data: {e}")
            
            # Network metrics (with delays)
            logging.info("Collecting network metrics...")
            try:
                network_data = alternative_data.get_network_metrics()
                data.update(network_data)
                time.sleep(30)  # 30 second delay
            except Exception as e:
                logging.warning(f"Failed to collect network data: {e}")
            
        else:
            # Full data collection (use with caution)
            logging.info("Running in FULL MODE - collecting all available data")
            data = alternative_data.get_all_data()
        
        # Save collected data
        data_dir = config['app_settings']['data_dir']
        os.makedirs(data_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = os.path.join(data_dir, f"alternative_data_{timestamp}.json")
        
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logging.info(f"Alternative data collection completed. Saved to {data_file}")
        logging.info(f"Collected {len(data)} data points")
        
        # Send completion notification
        if telegram_notifier.enabled:
            message = f"""
ALTERNATIVE DATA COLLECTION COMPLETE

Data Points Collected: {len(data)}
File Saved: {os.path.basename(data_file)}
Mode: {'SAFE' if safe_mode else 'FULL'}

Data Types:
• Social Sentiment: {'OK' if 'sentiment_score' in data else 'FAILED'}
• News Impact: {'OK' if 'news_impact' in data else 'FAILED'}
• On-chain Metrics: {'OK' if 'network_value' in data else 'FAILED'}
• Exchange Data: {'OK' if 'funding_rate' in data else 'FAILED'}
• Whale Activity: {'OK' if 'whale_activity' in data else 'FAILED'}
• Network Metrics: {'OK' if 'hash_rate' in data else 'FAILED'}
"""
            telegram_notifier.send_message(message)
        
        return True
        
    except Exception as e:
        logging.error(f"Error in alternative data collection: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Safe Alternative Data Collection')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--safe', action='store_true', default=True, help='Use safe mode with conservative rate limits')
    parser.add_argument('--full', action='store_true', help='Use full mode (use with caution)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Determine mode
    safe_mode = args.safe and not args.full
    
    try:
        success = collect_alternative_data(args.config, safe_mode)
        
        if success:
            print("✅ Alternative data collection completed successfully!")
            sys.exit(0)
        else:
            print("❌ Alternative data collection failed!")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 