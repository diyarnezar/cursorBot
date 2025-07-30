#!/usr/bin/env python3
"""
Safe Full Historical Data Processor
Implements batch processing to respect Binance API limits (1,200 weight/minute)
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SafeFullHistoricalProcessor:
    """
    Safe processor for full historical data that respects Binance API limits
    """
    
    def __init__(self):
        # Binance API limits
        self.WEIGHT_LIMIT_PER_MINUTE = 1200
        self.WEIGHT_PER_KLINE_CALL = 2
        self.MAX_KLINES_PER_CALL = 1000
        
        # Safety settings (use 90% of limit to be safe)
        self.SAFE_WEIGHT_LIMIT = int(self.WEIGHT_LIMIT_PER_MINUTE * 0.9)  # 1080 weight
        self.MAX_CALLS_PER_BATCH = self.SAFE_WEIGHT_LIMIT // self.WEIGHT_PER_KLINE_CALL  # 540 calls
        
        # Batch processing settings
        self.BATCH_DELAY_SECONDS = 60  # Wait 1 minute between batches
        self.SYMBOL_DELAY_SECONDS = 1   # Wait 1 second between symbols
        self.CALL_DELAY_SECONDS = 0.1   # Wait 0.1 seconds between calls
        
        # Tracking
        self.weight_history = deque(maxlen=60)  # Track last 60 seconds
        self.current_batch_weight = 0
        self.total_weight_used = 0
        self.total_calls_made = 0
        
        logger.info("🛡️ Safe Full Historical Processor initialized")
        logger.info(f"   Weight limit: {self.WEIGHT_LIMIT_PER_MINUTE}/min")
        logger.info(f"   Safe limit: {self.SAFE_WEIGHT_LIMIT}/min")
        logger.info(f"   Max calls per batch: {self.MAX_CALLS_PER_BATCH}")
    
    def calculate_full_historical_requirements(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Calculate the requirements for full historical data collection"""
        
        total_days = (end_date - start_date).days
        total_minutes = total_days * 1440  # 24 hours * 60 minutes
        
        calls_per_symbol = (total_minutes // self.MAX_KLINES_PER_CALL) + 1
        weight_per_symbol = calls_per_symbol * self.WEIGHT_PER_KLINE_CALL
        total_weight = weight_per_symbol * len(symbols)
        
        # Calculate batch requirements
        batches_needed = (total_weight // self.SAFE_WEIGHT_LIMIT) + 1
        symbols_per_batch = min(len(symbols), self.MAX_CALLS_PER_BATCH // calls_per_symbol)
        batches_for_symbols = (len(symbols) // symbols_per_batch) + (1 if len(symbols) % symbols_per_batch else 0)
        
        total_batches = batches_needed * batches_for_symbols
        estimated_time_minutes = total_batches * (self.BATCH_DELAY_SECONDS / 60)
        
        return {
            'total_days': total_days,
            'total_minutes': total_minutes,
            'calls_per_symbol': calls_per_symbol,
            'weight_per_symbol': weight_per_symbol,
            'total_weight': total_weight,
            'batches_needed': batches_needed,
            'symbols_per_batch': symbols_per_batch,
            'total_batches': total_batches,
            'estimated_time_minutes': estimated_time_minutes,
            'is_feasible': total_weight <= (self.SAFE_WEIGHT_LIMIT * 24 * 60)  # Max 24 hours of continuous processing
        }
    
    def process_full_historical_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Process full historical data for all symbols safely
        Returns: Dict[symbol, DataFrame]
        """
        
        logger.info("🚀 Starting Safe Full Historical Data Processing")
        
        # Calculate requirements
        requirements = self.calculate_full_historical_requirements(symbols, start_date, end_date)
        
        logger.info(f"📊 Processing Requirements:")
        logger.info(f"   - Symbols: {len(symbols)}")
        logger.info(f"   - Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"   - Total days: {requirements['total_days']}")
        logger.info(f"   - Total weight needed: {requirements['total_weight']:,}")
        logger.info(f"   - Batches needed: {requirements['total_batches']}")
        logger.info(f"   - Estimated time: {requirements['estimated_time_minutes']:.1f} minutes")
        
        if not requirements['is_feasible']:
            logger.error("❌ Full historical data collection is not feasible!")
            logger.error("   Consider reducing date range or number of symbols")
            return {}
        
        # Process in batches
        all_data = {}
        batch_count = 0
        
        for i in range(0, len(symbols), requirements['symbols_per_batch']):
            batch_symbols = symbols[i:i + requirements['symbols_per_batch']]
            batch_count += 1
            
            logger.info(f"📦 Processing batch {batch_count}/{requirements['total_batches']}: {len(batch_symbols)} symbols")
            
            # Process this batch
            batch_data = self._process_batch(batch_symbols, start_date, end_date, batch_count)
            all_data.update(batch_data)
            
            # Wait between batches (except for the last batch)
            if i + requirements['symbols_per_batch'] < len(symbols):
                logger.info(f"⏳ Waiting {self.BATCH_DELAY_SECONDS} seconds before next batch...")
                time.sleep(self.BATCH_DELAY_SECONDS)
        
        logger.info(f"✅ Full historical data processing completed!")
        logger.info(f"   - Total weight used: {self.total_weight_used:,}")
        logger.info(f"   - Total calls made: {self.total_calls_made:,}")
        logger.info(f"   - Symbols processed: {len(all_data)}")
        
        return all_data
    
    def _process_batch(self, symbols: List[str], start_date: datetime, end_date: datetime, batch_num: int) -> Dict[str, pd.DataFrame]:
        """Process a batch of symbols"""
        
        batch_data = {}
        current_weight = 0
        
        for symbol in symbols:
            logger.info(f"📈 Processing {symbol} (batch {batch_num})")
            
            try:
                # Process single symbol
                symbol_data = self._process_single_symbol(symbol, start_date, end_date)
                
                if not symbol_data.empty:
                    batch_data[symbol] = symbol_data
                    logger.info(f"✅ {symbol}: {len(symbol_data)} rows collected")
                else:
                    logger.warning(f"⚠️ {symbol}: No data collected")
                
                # Update weight tracking
                symbol_weight = self._calculate_symbol_weight(start_date, end_date)
                current_weight += symbol_weight
                self.total_weight_used += symbol_weight
                
                # Check if we're approaching the limit
                if current_weight >= self.SAFE_WEIGHT_LIMIT * 0.8:  # 80% threshold
                    logger.warning(f"⚠️ Batch {batch_num} approaching weight limit ({current_weight}/{self.SAFE_WEIGHT_LIMIT})")
                    break
                
                # Wait between symbols
                time.sleep(self.SYMBOL_DELAY_SECONDS)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue
        
        return batch_data
    
    def _process_single_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Process a single symbol's full historical data"""
        
        all_data = []
        current_start = start_date
        
        while current_start < end_date:
            # Calculate end time for this chunk
            current_end = min(current_start + timedelta(minutes=self.MAX_KLINES_PER_CALL), end_date)
            
            try:
                # Fetch klines for this chunk
                klines = self._fetch_klines_safe(symbol, current_start, current_end)
                
                if klines:
                    # Convert to DataFrame
                    chunk_df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert to numeric
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                                     'quote_asset_volume', 'number_of_trades',
                                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
                    
                    for col in numeric_columns:
                        chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
                    
                    all_data.append(chunk_df)
                
                # Move to next chunk
                current_start = current_end
                
                # Wait between calls
                time.sleep(self.CALL_DELAY_SECONDS)
                
            except Exception as e:
                logger.error(f"❌ Error fetching data for {symbol} at {current_start}: {e}")
                current_start = current_end
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all chunks
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp')
        
        return df
    
    def _fetch_klines_safe(self, symbol: str, start_time: datetime, end_time: datetime) -> List:
        """Safely fetch klines with rate limiting"""
        
        # This is a placeholder - in real implementation, you'd use your actual API client
        # with proper rate limiting integration
        
        # Simulate API call
        self.total_calls_made += 1
        
        # For now, return empty list (placeholder)
        # In real implementation, this would call your actual kline fetching function
        return []
    
    def _calculate_symbol_weight(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate weight needed for a symbol"""
        total_minutes = (end_date - start_date).total_seconds() / 60
        calls_needed = (total_minutes // self.MAX_KLINES_PER_CALL) + 1
        return calls_needed * self.WEIGHT_PER_KLINE_CALL

def test_safe_full_historical():
    """Test the safe full historical processor"""
    
    print("🧪 Testing Safe Full Historical Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = SafeFullHistoricalProcessor()
    
    # Test symbols
    test_symbols = [
        'BTCFDUSD', 'ETHFDUSD', 'BNBFDUSD', 'ADAFDUSD', 'SOLFDUSD',
        'DOTFDUSD', 'MATICFDUSD', 'LINKFDUSD', 'UNIFDUSD', 'AVAXFDUSD',
        'ATOMFDUSD', 'LTCFDUSD', 'BCHFDUSD', 'XRPFDUSD', 'DOGEFDUSD',
        'SHIBFDUSD', 'TRXFDUSD', 'ETCFDUSD', 'FILFDUSD', 'NEARFDUSD',
        'APTFDUSD', 'OPFDUSD', 'ARBFDUSD', 'SUIFDUSD', 'SEIFDUSD', 'JUPFDUSD'
    ]
    
    # Test date range (last 30 days as example)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Calculate requirements
    requirements = processor.calculate_full_historical_requirements(test_symbols, start_date, end_date)
    
    print(f"📊 Test Results (30 days, 26 symbols):")
    print(f"   - Total weight needed: {requirements['total_weight']:,}")
    print(f"   - Batches needed: {requirements['total_batches']}")
    print(f"   - Estimated time: {requirements['estimated_time_minutes']:.1f} minutes")
    print(f"   - Feasible: {'✅ YES' if requirements['is_feasible'] else '❌ NO'}")
    
    # Test with full historical range
    full_start_date = datetime(2023, 12, 1)  # ETH/FDUSD listing
    full_requirements = processor.calculate_full_historical_requirements(test_symbols, full_start_date, end_date)
    
    print(f"\n📊 Full Historical Results (since Dec 2023, 26 symbols):")
    print(f"   - Total weight needed: {full_requirements['total_weight']:,}")
    print(f"   - Batches needed: {full_requirements['total_batches']}")
    print(f"   - Estimated time: {full_requirements['estimated_time_minutes']:.1f} minutes")
    print(f"   - Feasible: {'✅ YES' if full_requirements['is_feasible'] else '❌ NO'}")

if __name__ == "__main__":
    test_safe_full_historical() 