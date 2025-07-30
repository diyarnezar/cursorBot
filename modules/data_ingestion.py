import pandas as pd
import logging
import sqlite3
from binance.client import Client
from datetime import datetime, timedelta
import time
import json
from tqdm import tqdm
import random

# Import the new Binance rate limiter
from modules.binance_rate_limiter import binance_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(path: str = 'config.json'):
    with open(path, 'r') as f:
        return json.load(f)

def fetch_klines(symbol: str, interval: str, start_time: datetime, end_time: datetime) -> list:
    """
    Fetch klines data from Binance with enhanced reliability, retry logic, and rate limiting.
    
    Args:
        symbol: Trading symbol (e.g., 'ETHFDUSD')
        interval: Time interval (e.g., '1m', '5m', '15m')
        start_time: Start time
        end_time: End time
        
    Returns:
        List of klines data
    """
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Initialize client (without API keys for public data)
            client = Client()
            
            # Convert interval to Binance format
            interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }
            
            binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_1MINUTE)
            
            # Convert times to milliseconds
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            # Calculate expected data points
            time_diff = end_time - start_time
            if interval == '1m':
                expected_points = int(time_diff.total_seconds() / 60)
            elif interval == '5m':
                expected_points = int(time_diff.total_seconds() / 300)
            elif interval == '15m':
                expected_points = int(time_diff.total_seconds() / 900)
            elif interval == '1h':
                expected_points = int(time_diff.total_seconds() / 3600)
            else:
                expected_points = 100
            
            logging.info(f"Attempting to fetch {expected_points} expected data points for {symbol} {interval}")
            
            # Apply rate limiting before making the request
            # The Binance client will make requests to: https://api.binance.com/api/v3/klines
            binance_client.rate_limiter.wait_if_needed('https://api.binance.com/api/v3/klines')
            
            # Fetch klines with timeout
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=binance_interval,
                start_str=start_ms,
                end_str=end_ms
            )
            
            # Record successful request for rate limiting
            binance_client.rate_limiter.record_request('https://api.binance.com/api/v3/klines')
            
            # Validate the data
            if klines and len(klines) > 0:
                # Check if we got reasonable amount of data
                min_expected = max(10, expected_points * 0.5)  # At least 50% of expected
                if len(klines) >= min_expected:
                    logging.info(f"✅ Successfully fetched {len(klines)} klines for {symbol} {interval} (attempt {attempt + 1})")
                    return klines
                else:
                    logging.warning(f"⚠️ Got only {len(klines)} klines, expected at least {min_expected} (attempt {attempt + 1})")
            else:
                logging.warning(f"⚠️ No klines data received (attempt {attempt + 1})")
            
            # If we get here, the data wasn't sufficient
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logging.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            
        except Exception as e:
            logging.error(f"Error fetching klines (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logging.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
    
    logging.error(f"❌ Failed to fetch klines after {max_retries} attempts")
    return []

def fetch_ticker_24hr(symbol: str) -> dict:
    """
    Fetch 24-hour ticker data from Binance with rate limiting.
    
    Args:
        symbol: Trading symbol (e.g., 'ETHFDUSD')
        
    Returns:
        Dictionary with ticker data
    """
    try:
        # Apply rate limiting before making the request
        binance_client.rate_limiter.wait_if_needed('https://api.binance.com/api/v3/ticker/24hr')
        
        client = Client()
        ticker = client.get_ticker(symbol=symbol)
        
        # Record successful request for rate limiting
        binance_client.rate_limiter.record_request('https://api.binance.com/api/v3/ticker/24hr')
        
        return ticker
        
    except Exception as e:
        logging.error(f"Error fetching ticker: {e}")
        return {}

def fetch_order_book(symbol: str, limit: int = 10) -> dict:
    """
    Fetch order book data from Binance with rate limiting.
    
    Args:
        symbol: Trading symbol (e.g., 'ETHFDUSD')
        limit: Number of orders to fetch
        
    Returns:
        Dictionary with order book data
    """
    try:
        # Apply rate limiting before making the request
        binance_client.rate_limiter.wait_if_needed('https://api.binance.com/api/v3/depth')
        
        client = Client()
        order_book = client.get_order_book(symbol=symbol, limit=limit)
        
        # Record successful request for rate limiting
        binance_client.rate_limiter.record_request('https://api.binance.com/api/v3/depth')
        
        return order_book
        
    except Exception as e:
        logging.error(f"Error fetching order book: {e}")
        return {}

def get_full_historical_data(client, trading_pair):
    """
    Fetches the entire 1-minute K-line data history for a pair from Binance.
    """
    logging.info(f"Starting FULL historical data download for {trading_pair}...")
    
    start_date_str = "2019-01-01"
    
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades', 
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    all_data_df = pd.DataFrame()
    start_date_ms = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)
    
    # --- THIS IS THE ENHANCED PROGRESS BAR ---
    pbar = tqdm(desc=f"Fetching {trading_pair} History", 
                unit=" candles", 
                mininterval=0.1, # Update at least every 0.1s
                smoothing=0.05,  # Make speed estimate more responsive
                bar_format='{l_bar}{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    while True:
        try:
            klines = client.get_historical_klines(trading_pair, Client.KLINE_INTERVAL_1MINUTE, start_date_ms, limit=1000)
            
            if not klines:
                break 
                
            temp_df = pd.DataFrame(klines, columns=columns)
            all_data_df = pd.concat([all_data_df, temp_df], ignore_index=True)
            
            last_timestamp = int(all_data_df.iloc[-1]['timestamp'])
            start_date_ms = last_timestamp + 60000 
            
            pbar.update(len(klines))
            time.sleep(0.1)

        except Exception as e:
            logging.error(f"An error occurred during download: {e}. Pausing for 60 seconds.")
            time.sleep(60)

    pbar.close()
    
    logging.info("Processing downloaded data...")
    all_data_df['timestamp'] = pd.to_datetime(all_data_df['timestamp'], unit='ms')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades']
    all_data_df[numeric_cols] = all_data_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    all_data_df.drop_duplicates(subset=['timestamp'], inplace=True)
    all_data_df.set_index('timestamp', inplace=True)
    
    logging.info(f"Successfully downloaded and processed {len(all_data_df)} total 1-minute candles.")
    return all_data_df

def save_to_db(df, table_name, db_path):
    logging.info(f"Saving data to database at {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=True)
        conn.close()
        logging.info(f"Successfully saved {len(df)} rows to table '{table_name}'.")
    except Exception as e:
        logging.error(f"Failed to save data to database: {e}")

if __name__ == "__main__":
    config = load_config()
    client = Client(config['binance_credentials']['api_key'], config['binance_credentials']['api_secret'])
    trading_pair = config['trading_parameters']['trading_pair']
    db_path = config['app_settings']['database_path']
    table_name = f"klines_{trading_pair.lower()}"
    
    historical_df = get_full_historical_data(client, trading_pair)
    
    if not historical_df.empty:
        save_to_db(historical_df, table_name, db_path)
