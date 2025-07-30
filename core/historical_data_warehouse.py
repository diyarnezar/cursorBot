"""
🚀 PROJECT HYPERION - HISTORICAL DATA WAREHOUSE
==============================================

Centralized data warehouse for all historical data sources.
Stores raw, alternative, and processed data with proper timestamp alignment.

Author: Project Hyperion Team
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
import pickle
import gzip
import asyncio
import aiohttp
import time

from config.api_config import APIConfig
from data.collectors.binance_collector import BinanceDataCollector
from data.processors.data_processor import DataProcessor


class HistoricalDataWarehouse:
    """
    Comprehensive historical data warehouse
    Centralizes all data ingestion and storage
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the Historical Data Warehouse"""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
        
        # Initialize components
        self.api_config = APIConfig()
        self.data_collector = BinanceDataCollector()
        self.data_processor = DataProcessor(config=self.config)
        
        # Warehouse settings
        self.warehouse_path = Path("data/warehouse")
        self.warehouse_path.mkdir(parents=True, exist_ok=True)
        
        # Database connection
        self.db_path = self.warehouse_path / "warehouse.db"
        self._init_database()
        
        # Data sources configuration
        self.data_sources = {
            'price_data': {
                'enabled': True,
                'update_frequency': '1m',
                'retention_days': 730,  # 2 years
                'compression': 'gzip'
            },
            'sentiment_data': {
                'enabled': True,
                'update_frequency': '5m',
                'retention_days': 365,
                'compression': 'gzip'
            },
            'onchain_data': {
                'enabled': True,
                'update_frequency': '15m',
                'retention_days': 365,
                'compression': 'gzip'
            },
            'news_data': {
                'enabled': True,
                'update_frequency': '1h',
                'retention_days': 180,
                'compression': 'gzip'
            },
            'social_data': {
                'enabled': True,
                'update_frequency': '5m',
                'retention_days': 90,
                'compression': 'gzip'
            },
            'macro_data': {
                'enabled': True,
                'update_frequency': '1d',
                'retention_days': 1095,  # 3 years
                'compression': 'gzip'
            }
        }
        
        # Data ingestion state
        self.ingestion_status = {}
        self.last_update = {}
        
        # Performance tracking
        self.ingestion_stats = {
            'total_records': 0,
            'total_size_gb': 0.0,
            'last_cleanup': None,
            'compression_ratio': 0.0
        }
        
        self.logger.info("🚀 Historical Data Warehouse initialized")
    
    def _init_database(self):
        """Initialize the warehouse database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create metadata tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_sources (
                    source_name TEXT PRIMARY KEY,
                    enabled BOOLEAN,
                    update_frequency TEXT,
                    retention_days INTEGER,
                    compression TEXT,
                    last_update TIMESTAMP,
                    record_count INTEGER,
                    data_size_mb REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT,
                    timestamp TIMESTAMP,
                    records_ingested INTEGER,
                    data_size_mb REAL,
                    status TEXT,
                    error_message TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_quality (
                    source_name TEXT,
                    timestamp TIMESTAMP,
                    completeness REAL,
                    accuracy REAL,
                    consistency REAL,
                    timeliness REAL,
                    PRIMARY KEY (source_name, timestamp)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("💾 Warehouse database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing database: {e}")
    
    async def start_data_ingestion(self, symbols: List[str]):
        """Start continuous data ingestion for all sources"""
        try:
            self.logger.info(f"🚀 Starting data ingestion for {len(symbols)} symbols")
            
            # Start ingestion tasks for each data source
            tasks = []
            
            for source_name, config in self.data_sources.items():
                if config['enabled']:
                    task = asyncio.create_task(
                        self._ingest_data_source(source_name, symbols, config)
                    )
                    tasks.append(task)
            
            # Wait for all ingestion tasks
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"❌ Error in data ingestion: {e}")
    
    async def _ingest_data_source(self, source_name: str, symbols: List[str], config: Dict[str, Any]):
        """Ingest data for a specific source"""
        try:
            while True:
                start_time = time.time()
                
                self.logger.info(f"📥 Ingesting {source_name} data...")
                
                # Ingest data for each symbol
                total_records = 0
                total_size = 0
                
                for symbol in symbols:
                    try:
                        records, size = await self._ingest_symbol_data(source_name, symbol, config)
                        total_records += records
                        total_size += size
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error ingesting {source_name} for {symbol}: {e}")
                        continue
                
                # Update ingestion status
                self.ingestion_status[source_name] = {
                    'last_update': datetime.now(),
                    'records_ingested': total_records,
                    'data_size_mb': total_size,
                    'status': 'success'
                }
                
                # Log ingestion
                self._log_ingestion(source_name, total_records, total_size, 'success')
                
                # Update database
                self._update_source_metadata(source_name, total_records, total_size)
                
                ingestion_time = time.time() - start_time
                self.logger.info(f"✅ {source_name} ingestion completed: {total_records} records, {total_size:.2f}MB in {ingestion_time:.2f}s")
                
                # Wait for next update cycle
                await self._wait_for_next_update(config['update_frequency'])
                
        except Exception as e:
            self.logger.error(f"❌ Error in {source_name} ingestion: {e}")
            self.ingestion_status[source_name] = {
                'last_update': datetime.now(),
                'status': 'error',
                'error_message': str(e)
            }
    
    async def _ingest_symbol_data(self, source_name: str, symbol: str, config: Dict[str, Any]) -> Tuple[int, float]:
        """Ingest data for a specific symbol and source"""
        try:
            data = None
            
            if source_name == 'price_data':
                data = await self._ingest_price_data(symbol)
            elif source_name == 'sentiment_data':
                data = await self._ingest_sentiment_data(symbol)
            elif source_name == 'onchain_data':
                data = await self._ingest_onchain_data(symbol)
            elif source_name == 'news_data':
                data = await self._ingest_news_data(symbol)
            elif source_name == 'social_data':
                data = await self._ingest_social_data(symbol)
            elif source_name == 'macro_data':
                data = await self._ingest_macro_data(symbol)
            
            if data is not None and not data.empty:
                # Store data
                records, size = await self._store_data(source_name, symbol, data, config)
                return records, size
            
            return 0, 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error ingesting {source_name} for {symbol}: {e}")
            return 0, 0.0
    
    async def _ingest_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Ingest price data from Binance"""
        try:
            # Get recent klines data
            data = self.data_collector.get_klines(symbol, "1m", limit=1000)
            
            if data is not None and not data.empty:
                # Process and clean data
                processed_data = self.data_processor.clean_data(data, symbol)
                return processed_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error ingesting price data for {symbol}: {e}")
            return None
    
    async def _ingest_sentiment_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Ingest sentiment data from external sources"""
        try:
            # Simulate sentiment data collection
            # In production, this would connect to sentiment APIs
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=1), 
                                end=datetime.now(), freq='5min')
            
            sentiment_data = pd.DataFrame({
                'timestamp': dates,
                'symbol': symbol,
                'sentiment_score': np.random.normal(0, 0.1, len(dates)),
                'sentiment_volume': np.random.lognormal(5, 1, len(dates)),
                'fear_greed_index': np.random.uniform(0, 100, len(dates)),
                'social_volume': np.random.lognormal(8, 1, len(dates))
            })
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"❌ Error ingesting sentiment data for {symbol}: {e}")
            return None
    
    async def _ingest_onchain_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Ingest on-chain data"""
        try:
            # Simulate on-chain data collection
            # In production, this would connect to blockchain APIs
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=1), 
                                end=datetime.now(), freq='15min')
            
            onchain_data = pd.DataFrame({
                'timestamp': dates,
                'symbol': symbol,
                'transaction_count': np.random.poisson(1000, len(dates)),
                'active_addresses': np.random.poisson(5000, len(dates)),
                'network_hashrate': np.random.lognormal(12, 0.5, len(dates)),
                'defi_tvl': np.random.lognormal(15, 0.3, len(dates)),
                'whale_transactions': np.random.poisson(10, len(dates))
            })
            
            return onchain_data
            
        except Exception as e:
            self.logger.error(f"❌ Error ingesting on-chain data for {symbol}: {e}")
            return None
    
    async def _ingest_news_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Ingest news data"""
        try:
            # Simulate news data collection
            # In production, this would connect to news APIs
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=1), 
                                end=datetime.now(), freq='1h')
            
            news_data = pd.DataFrame({
                'timestamp': dates,
                'symbol': symbol,
                'news_count': np.random.poisson(5, len(dates)),
                'news_sentiment': np.random.normal(0, 0.2, len(dates)),
                'headline_impact': np.random.uniform(0, 1, len(dates)),
                'source_credibility': np.random.uniform(0.5, 1.0, len(dates))
            })
            
            return news_data
            
        except Exception as e:
            self.logger.error(f"❌ Error ingesting news data for {symbol}: {e}")
            return None
    
    async def _ingest_social_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Ingest social media data"""
        try:
            # Simulate social data collection
            # In production, this would connect to social media APIs
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=1), 
                                end=datetime.now(), freq='5min')
            
            social_data = pd.DataFrame({
                'timestamp': dates,
                'symbol': symbol,
                'tweet_volume': np.random.poisson(100, len(dates)),
                'tweet_sentiment': np.random.normal(0, 0.15, len(dates)),
                'reddit_mentions': np.random.poisson(50, len(dates)),
                'reddit_sentiment': np.random.normal(0, 0.2, len(dates)),
                'telegram_volume': np.random.poisson(200, len(dates))
            })
            
            return social_data
            
        except Exception as e:
            self.logger.error(f"❌ Error ingesting social data for {symbol}: {e}")
            return None
    
    async def _ingest_macro_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Ingest macroeconomic data"""
        try:
            # Simulate macro data collection
            # In production, this would connect to economic data APIs
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                end=datetime.now(), freq='1d')
            
            macro_data = pd.DataFrame({
                'timestamp': dates,
                'symbol': symbol,
                'dollar_index': np.random.normal(100, 2, len(dates)),
                'vix_index': np.random.lognormal(2, 0.3, len(dates)),
                'treasury_yield': np.random.normal(2.5, 0.5, len(dates)),
                'gold_price': np.random.lognormal(7, 0.1, len(dates)),
                'oil_price': np.random.lognormal(4, 0.2, len(dates))
            })
            
            return macro_data
            
        except Exception as e:
            self.logger.error(f"❌ Error ingesting macro data for {symbol}: {e}")
            return None
    
    async def _store_data(self, source_name: str, symbol: str, data: pd.DataFrame, 
                         config: Dict[str, Any]) -> Tuple[int, float]:
        """Store data in the warehouse"""
        try:
            # Create source directory
            source_dir = self.warehouse_path / source_name / symbol
            source_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timestamp}.pkl"
            filepath = source_dir / filename
            
            # Compress and store data
            if config['compression'] == 'gzip':
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            
            # Calculate size
            file_size = filepath.stat().st_size / (1024 * 1024)  # MB
            
            # Update statistics
            self.ingestion_stats['total_records'] += len(data)
            self.ingestion_stats['total_size_gb'] += file_size / 1024
            
            return len(data), file_size
            
        except Exception as e:
            self.logger.error(f"❌ Error storing {source_name} data for {symbol}: {e}")
            return 0, 0.0
    
    async def _wait_for_next_update(self, frequency: str):
        """Wait for next update cycle"""
        try:
            # Parse frequency
            if frequency.endswith('m'):
                minutes = int(frequency[:-1])
                await asyncio.sleep(minutes * 60)
            elif frequency.endswith('h'):
                hours = int(frequency[:-1])
                await asyncio.sleep(hours * 3600)
            elif frequency.endswith('d'):
                days = int(frequency[:-1])
                await asyncio.sleep(days * 86400)
            else:
                await asyncio.sleep(60)  # Default 1 minute
                
        except Exception as e:
            self.logger.error(f"❌ Error in update wait: {e}")
            await asyncio.sleep(60)
    
    def query_data(self, source_name: str, symbol: str, start_time: datetime, 
                   end_time: datetime) -> Optional[pd.DataFrame]:
        """Query data from warehouse"""
        try:
            source_dir = self.warehouse_path / source_name / symbol
            
            if not source_dir.exists():
                return None
            
            # Collect all data files in time range
            all_data = []
            
            for filepath in source_dir.glob("*.pkl"):
                try:
                    # Extract timestamp from filename
                    filename = filepath.stem
                    file_timestamp = datetime.strptime(filename.split('_')[-1], '%Y%m%d_%H%M%S')
                    
                    if start_time <= file_timestamp <= end_time:
                        # Load data
                        if filepath.suffix == '.gz':
                            with gzip.open(filepath, 'rb') as f:
                                data = pickle.load(f)
                        else:
                            with open(filepath, 'rb') as f:
                                data = pickle.load(f)
                        
                        if isinstance(data, pd.DataFrame) and not data.empty:
                            all_data.append(data)
                            
                except Exception as e:
                    self.logger.error(f"❌ Error loading {filepath}: {e}")
                    continue
            
            if all_data:
                # Combine and sort by timestamp
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
                
                # Filter by time range
                mask = (combined_data['timestamp'] >= start_time) & (combined_data['timestamp'] <= end_time)
                filtered_data = combined_data[mask]
                
                return filtered_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error querying data: {e}")
            return None
    
    def merge_data_by_timestamp(self, data_sources: List[str], symbol: str, 
                               start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Merge multiple data sources by timestamp"""
        try:
            merged_data = None
            
            for source_name in data_sources:
                source_data = self.query_data(source_name, symbol, start_time, end_time)
                
                if source_data is not None and not source_data.empty:
                    if merged_data is None:
                        merged_data = source_data
                    else:
                        # Merge on timestamp
                        merged_data = pd.merge(merged_data, source_data, on='timestamp', how='outer')
            
            if merged_data is not None:
                # Sort by timestamp and forward fill missing values
                merged_data = merged_data.sort_values('timestamp').reset_index(drop=True)
                merged_data = merged_data.fillna(method='ffill')
                
                return merged_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error merging data: {e}")
            return None
    
    def _log_ingestion(self, source_name: str, records: int, size_mb: float, status: str):
        """Log ingestion activity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ingestion_log (source_name, timestamp, records_ingested, data_size_mb, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (source_name, datetime.now(), records, size_mb, status))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error logging ingestion: {e}")
    
    def _update_source_metadata(self, source_name: str, records: int, size_mb: float):
        """Update source metadata in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO data_sources 
                (source_name, enabled, update_frequency, retention_days, compression, 
                 last_update, record_count, data_size_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                source_name,
                self.data_sources[source_name]['enabled'],
                self.data_sources[source_name]['update_frequency'],
                self.data_sources[source_name]['retention_days'],
                self.data_sources[source_name]['compression'],
                datetime.now(),
                records,
                size_mb
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating metadata: {e}")
    
    def cleanup_old_data(self):
        """Clean up old data based on retention policies"""
        try:
            self.logger.info("🧹 Starting data cleanup...")
            
            for source_name, config in self.data_sources.items():
                retention_days = config['retention_days']
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                source_dir = self.warehouse_path / source_name
                if not source_dir.exists():
                    continue
                
                files_removed = 0
                space_freed = 0
                
                for symbol_dir in source_dir.iterdir():
                    if symbol_dir.is_dir():
                        for filepath in symbol_dir.glob("*.pkl"):
                            try:
                                # Extract timestamp from filename
                                filename = filepath.stem
                                file_timestamp = datetime.strptime(filename.split('_')[-1], '%Y%m%d_%H%M%S')
                                
                                if file_timestamp < cutoff_date:
                                    file_size = filepath.stat().st_size
                                    filepath.unlink()
                                    files_removed += 1
                                    space_freed += file_size
                                    
                            except Exception as e:
                                self.logger.error(f"❌ Error cleaning up {filepath}: {e}")
                
                if files_removed > 0:
                    self.logger.info(f"🧹 Cleaned up {source_name}: {files_removed} files, {space_freed / (1024*1024):.2f}MB freed")
            
            self.ingestion_stats['last_cleanup'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"❌ Error in data cleanup: {e}")
    
    def get_warehouse_stats(self) -> Dict[str, Any]:
        """Get warehouse statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get source statistics
            cursor.execute('SELECT * FROM data_sources')
            sources = cursor.fetchall()
            
            # Get recent ingestion activity
            cursor.execute('''
                SELECT source_name, COUNT(*), SUM(records_ingested), SUM(data_size_mb)
                FROM ingestion_log 
                WHERE timestamp > datetime('now', '-1 day')
                GROUP BY source_name
            ''')
            daily_activity = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_sources': len(sources),
                'active_sources': len([s for s in sources if s[1]]),  # enabled sources
                'total_records': self.ingestion_stats['total_records'],
                'total_size_gb': self.ingestion_stats['total_size_gb'],
                'last_cleanup': self.ingestion_stats['last_cleanup'],
                'daily_activity': daily_activity,
                'ingestion_status': self.ingestion_status
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting warehouse stats: {e}")
            return {}
    
    def export_warehouse_report(self, filepath: str = None):
        """Export warehouse report"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"reports/warehouse_report_{timestamp}.json"
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'warehouse_stats': self.get_warehouse_stats(),
                'data_sources': self.data_sources,
                'ingestion_status': self.ingestion_status
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"💾 Warehouse report exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting warehouse report: {e}") 