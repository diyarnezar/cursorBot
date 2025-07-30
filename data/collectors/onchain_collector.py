"""
On-Chain Data Collector for Blockchain Analytics
Part of Project Hyperion - Ultimate Autonomous Trading Bot

Collects on-chain data from:
- Ethereum blockchain
- Bitcoin blockchain
- DeFi protocols
- Network metrics
- Wallet analytics
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class OnChainConfig:
    """Configuration for on-chain data collection"""
    # API keys (free tiers)
    etherscan_api_key: str = ""
    blockchain_info_api_key: str = ""
    
    # Rate limiting
    rate_limit_per_second: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    
    # Collection parameters
    collection_interval: int = 300  # 5 minutes
    data_window_hours: int = 24
    
    # Networks to monitor
    networks: List[str] = None
    
    # DeFi protocols to track
    defi_protocols: List[str] = None


class OnChainCollector:
    """
    Advanced On-Chain Data Collector
    
    Features:
    - Multi-blockchain data collection
    - DeFi protocol analytics
    - Network metrics monitoring
    - Wallet movement tracking
    - Gas fee analysis
    - Transaction volume analysis
    """
    
    def __init__(self, config: OnChainConfig):
        self.config = config
        self.session = requests.Session()
        self.onchain_data = {}
        self.data_history = []
        
        # Initialize default values
        if self.config.networks is None:
            self.config.networks = ['ethereum', 'bitcoin']
        
        if self.config.defi_protocols is None:
            self.config.defi_protocols = [
                'uniswap', 'sushiswap', 'aave', 'compound', 'curve', 'makerdao'
            ]
        
        logger.info("On-Chain Data Collector initialized")

    async def collect_all_onchain_data(self) -> Dict[str, Any]:
        """Collect on-chain data from all sources"""
        try:
            onchain_data = {
                'ethereum_data': await self.collect_ethereum_data(),
                'bitcoin_data': await self.collect_bitcoin_data(),
                'defi_data': await self.collect_defi_data(),
                'network_metrics': await self.collect_network_metrics(),
                'wallet_analytics': await self.collect_wallet_analytics(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store on-chain data
            self._store_onchain_data(onchain_data)
            
            logger.info(f"Collected on-chain data from {len(onchain_data) - 1} sources")
            return onchain_data
            
        except Exception as e:
            logger.error(f"Error collecting on-chain data: {e}")
            return {'timestamp': datetime.now().isoformat()}

    async def collect_ethereum_data(self) -> Dict[str, Any]:
        """Collect Ethereum blockchain data"""
        try:
            ethereum_data = {
                'gas_metrics': await self._collect_ethereum_gas_data(),
                'transaction_metrics': await self._collect_ethereum_transaction_data(),
                'network_metrics': await self._collect_ethereum_network_data(),
                'defi_metrics': await self._collect_ethereum_defi_data(),
                'timestamp': datetime.now().isoformat()
            }
            
            return ethereum_data
            
        except Exception as e:
            logger.error(f"Error collecting Ethereum data: {e}")
            return {'timestamp': datetime.now().isoformat()}

    async def _collect_ethereum_gas_data(self) -> Dict[str, Any]:
        """Collect Ethereum gas metrics"""
        try:
            # Simulate gas data collection
            # In production, use Etherscan API or similar
            
            gas_price = np.random.uniform(10, 100)  # Gwei
            gas_used = np.random.uniform(1000000, 5000000)  # Gas used
            gas_limit = np.random.uniform(15000000, 20000000)  # Gas limit
            
            return {
                'gas_price_gwei': gas_price,
                'gas_used': gas_used,
                'gas_limit': gas_limit,
                'gas_utilization': (gas_used / gas_limit) * 100,
                'gas_price_usd': gas_price * 0.000000001 * 2000,  # Approximate USD value
                'gas_trend': 'stable' if gas_price < 50 else 'high'
            }
            
        except Exception as e:
            logger.error(f"Error collecting Ethereum gas data: {e}")
            return {'gas_price_gwei': 20.0, 'gas_used': 2000000}

    async def _collect_ethereum_transaction_data(self) -> Dict[str, Any]:
        """Collect Ethereum transaction metrics"""
        try:
            # Simulate transaction data
            tx_count = np.random.randint(100000, 500000)
            tx_volume_eth = np.random.uniform(100000, 500000)
            tx_volume_usd = tx_volume_eth * 2000  # Approximate USD value
            
            return {
                'transaction_count': tx_count,
                'transaction_volume_eth': tx_volume_eth,
                'transaction_volume_usd': tx_volume_usd,
                'average_transaction_size_eth': tx_volume_eth / tx_count if tx_count > 0 else 0,
                'transaction_fee_total_eth': tx_count * 0.001,  # Approximate
                'transaction_fee_total_usd': tx_count * 0.001 * 2000
            }
            
        except Exception as e:
            logger.error(f"Error collecting Ethereum transaction data: {e}")
            return {'transaction_count': 200000, 'transaction_volume_eth': 300000}

    async def _collect_ethereum_network_data(self) -> Dict[str, Any]:
        """Collect Ethereum network metrics"""
        try:
            # Simulate network data
            block_number = np.random.randint(18000000, 19000000)
            block_time = np.random.uniform(12, 15)  # seconds
            difficulty = np.random.uniform(1000000000000, 2000000000000)
            
            return {
                'block_number': block_number,
                'block_time_seconds': block_time,
                'difficulty': difficulty,
                'hashrate_th': difficulty / (block_time * 1000000000000),
                'network_health': 'healthy' if block_time < 14 else 'congested'
            }
            
        except Exception as e:
            logger.error(f"Error collecting Ethereum network data: {e}")
            return {'block_number': 18500000, 'block_time_seconds': 13.5}

    async def _collect_ethereum_defi_data(self) -> Dict[str, Any]:
        """Collect Ethereum DeFi metrics"""
        try:
            # Simulate DeFi data
            defi_tvl = np.random.uniform(40000, 60000)  # Millions USD
            defi_volume_24h = np.random.uniform(2000, 8000)  # Millions USD
            
            return {
                'defi_tvl_millions': defi_tvl,
                'defi_volume_24h_millions': defi_volume_24h,
                'defi_dominance': (defi_tvl / 100000) * 100,  # Percentage of total crypto market
                'defi_activity_score': (defi_volume_24h / defi_tvl) * 100
            }
            
        except Exception as e:
            logger.error(f"Error collecting Ethereum DeFi data: {e}")
            return {'defi_tvl_millions': 50000, 'defi_volume_24h_millions': 5000}

    async def collect_bitcoin_data(self) -> Dict[str, Any]:
        """Collect Bitcoin blockchain data"""
        try:
            bitcoin_data = {
                'transaction_metrics': await self._collect_bitcoin_transaction_data(),
                'network_metrics': await self._collect_bitcoin_network_data(),
                'mining_metrics': await self._collect_bitcoin_mining_data(),
                'timestamp': datetime.now().isoformat()
            }
            
            return bitcoin_data
            
        except Exception as e:
            logger.error(f"Error collecting Bitcoin data: {e}")
            return {'timestamp': datetime.now().isoformat()}

    async def _collect_bitcoin_transaction_data(self) -> Dict[str, Any]:
        """Collect Bitcoin transaction metrics"""
        try:
            # Simulate Bitcoin transaction data
            tx_count = np.random.randint(200000, 400000)
            tx_volume_btc = np.random.uniform(50000, 150000)
            tx_volume_usd = tx_volume_btc * 40000  # Approximate USD value
            
            return {
                'transaction_count': tx_count,
                'transaction_volume_btc': tx_volume_btc,
                'transaction_volume_usd': tx_volume_usd,
                'average_transaction_size_btc': tx_volume_btc / tx_count if tx_count > 0 else 0,
                'transaction_fee_total_btc': tx_count * 0.0001,  # Approximate
                'transaction_fee_total_usd': tx_count * 0.0001 * 40000
            }
            
        except Exception as e:
            logger.error(f"Error collecting Bitcoin transaction data: {e}")
            return {'transaction_count': 300000, 'transaction_volume_btc': 100000}

    async def _collect_bitcoin_network_data(self) -> Dict[str, Any]:
        """Collect Bitcoin network metrics"""
        try:
            # Simulate Bitcoin network data
            block_height = np.random.randint(800000, 850000)
            block_time = np.random.uniform(8, 12)  # minutes
            difficulty = np.random.uniform(50000000000000, 100000000000000)
            
            return {
                'block_height': block_height,
                'block_time_minutes': block_time,
                'difficulty': difficulty,
                'hashrate_eh': difficulty / (block_time * 60 * 1000000000000000000),
                'network_health': 'healthy' if block_time < 10 else 'congested'
            }
            
        except Exception as e:
            logger.error(f"Error collecting Bitcoin network data: {e}")
            return {'block_height': 825000, 'block_time_minutes': 10.0}

    async def _collect_bitcoin_mining_data(self) -> Dict[str, Any]:
        """Collect Bitcoin mining metrics"""
        try:
            # Simulate mining data
            block_reward = 6.25  # BTC
            mining_revenue_btc = np.random.uniform(800, 1200)  # Daily
            mining_revenue_usd = mining_revenue_btc * 40000
            
            return {
                'block_reward_btc': block_reward,
                'mining_revenue_btc_daily': mining_revenue_btc,
                'mining_revenue_usd_daily': mining_revenue_usd,
                'mining_difficulty_change': np.random.uniform(-5, 5),  # Percentage
                'mining_pool_distribution': {
                    'antpool': np.random.uniform(10, 20),
                    'f2pool': np.random.uniform(8, 15),
                    'binance_pool': np.random.uniform(5, 12),
                    'other': np.random.uniform(50, 70)
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting Bitcoin mining data: {e}")
            return {'block_reward_btc': 6.25, 'mining_revenue_btc_daily': 1000}

    async def collect_defi_data(self) -> Dict[str, Any]:
        """Collect DeFi protocol data"""
        try:
            defi_data = {
                'protocol_metrics': {},
                'total_tvl': 0.0,
                'total_volume_24h': 0.0,
                'defi_dominance': 0.0
            }
            
            total_tvl = 0.0
            total_volume = 0.0
            
            for protocol in self.config.defi_protocols:
                protocol_data = await self._collect_protocol_data(protocol)
                defi_data['protocol_metrics'][protocol] = protocol_data
                
                total_tvl += protocol_data.get('tvl_millions', 0.0)
                total_volume += protocol_data.get('volume_24h_millions', 0.0)
            
            defi_data['total_tvl'] = total_tvl
            defi_data['total_volume_24h'] = total_volume
            defi_data['defi_dominance'] = (total_tvl / 100000) * 100  # Percentage of total crypto market
            
            return defi_data
            
        except Exception as e:
            logger.error(f"Error collecting DeFi data: {e}")
            return {'total_tvl': 0.0, 'total_volume_24h': 0.0}

    async def _collect_protocol_data(self, protocol: str) -> Dict[str, Any]:
        """Collect data for a specific DeFi protocol"""
        try:
            # Simulate protocol data
            tvl = np.random.uniform(100, 10000)  # Millions USD
            volume_24h = np.random.uniform(10, 1000)  # Millions USD
            user_count = np.random.randint(10000, 1000000)
            
            return {
                'tvl_millions': tvl,
                'volume_24h_millions': volume_24h,
                'user_count': user_count,
                'protocol_health': 'healthy' if tvl > 1000 else 'moderate',
                'activity_score': (volume_24h / tvl) * 100 if tvl > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error collecting protocol data for {protocol}: {e}")
            return {'tvl_millions': 0.0, 'volume_24h_millions': 0.0}

    async def collect_network_metrics(self) -> Dict[str, Any]:
        """Collect overall network metrics"""
        try:
            network_metrics = {
                'ethereum_metrics': await self._get_ethereum_network_summary(),
                'bitcoin_metrics': await self._get_bitcoin_network_summary(),
                'cross_chain_metrics': await self._get_cross_chain_metrics(),
                'timestamp': datetime.now().isoformat()
            }
            
            return network_metrics
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
            return {'timestamp': datetime.now().isoformat()}

    async def _get_ethereum_network_summary(self) -> Dict[str, Any]:
        """Get Ethereum network summary"""
        try:
            return {
                'network_utilization': np.random.uniform(60, 90),  # Percentage
                'transaction_success_rate': np.random.uniform(95, 99.5),  # Percentage
                'average_confirmation_time': np.random.uniform(10, 30),  # Seconds
                'network_congestion': 'low' if np.random.random() > 0.3 else 'high'
            }
        except Exception as e:
            logger.error(f"Error getting Ethereum network summary: {e}")
            return {'network_utilization': 75.0}

    async def _get_bitcoin_network_summary(self) -> Dict[str, Any]:
        """Get Bitcoin network summary"""
        try:
            return {
                'network_utilization': np.random.uniform(70, 95),  # Percentage
                'transaction_success_rate': np.random.uniform(98, 99.9),  # Percentage
                'average_confirmation_time': np.random.uniform(10, 60),  # Minutes
                'network_congestion': 'low' if np.random.random() > 0.2 else 'high'
            }
        except Exception as e:
            logger.error(f"Error getting Bitcoin network summary: {e}")
            return {'network_utilization': 85.0}

    async def _get_cross_chain_metrics(self) -> Dict[str, Any]:
        """Get cross-chain metrics"""
        try:
            return {
                'bridge_volume_24h': np.random.uniform(100, 1000),  # Millions USD
                'bridge_transactions_24h': np.random.randint(10000, 100000),
                'most_active_bridge': 'ethereum_bitcoin',
                'bridge_health': 'healthy'
            }
        except Exception as e:
            logger.error(f"Error getting cross-chain metrics: {e}")
            return {'bridge_volume_24h': 500.0}

    async def collect_wallet_analytics(self) -> Dict[str, Any]:
        """Collect wallet movement analytics"""
        try:
            wallet_analytics = {
                'whale_movements': await self._collect_whale_movements(),
                'exchange_flows': await self._collect_exchange_flows(),
                'wallet_distribution': await self._collect_wallet_distribution(),
                'timestamp': datetime.now().isoformat()
            }
            
            return wallet_analytics
            
        except Exception as e:
            logger.error(f"Error collecting wallet analytics: {e}")
            return {'timestamp': datetime.now().isoformat()}

    async def _collect_whale_movements(self) -> Dict[str, Any]:
        """Collect whale wallet movements"""
        try:
            # Simulate whale movement data
            whale_transactions = np.random.randint(10, 100)
            whale_volume_eth = np.random.uniform(10000, 100000)
            whale_volume_btc = np.random.uniform(100, 1000)
            
            return {
                'whale_transactions_24h': whale_transactions,
                'whale_volume_eth': whale_volume_eth,
                'whale_volume_btc': whale_volume_btc,
                'whale_sentiment': 'bullish' if np.random.random() > 0.5 else 'bearish',
                'largest_whale_transaction_eth': np.random.uniform(1000, 10000)
            }
            
        except Exception as e:
            logger.error(f"Error collecting whale movements: {e}")
            return {'whale_transactions_24h': 50, 'whale_volume_eth': 50000}

    async def _collect_exchange_flows(self) -> Dict[str, Any]:
        """Collect exchange inflow/outflow data"""
        try:
            # Simulate exchange flow data
            exchange_inflow_eth = np.random.uniform(50000, 200000)
            exchange_outflow_eth = np.random.uniform(40000, 180000)
            exchange_inflow_btc = np.random.uniform(5000, 20000)
            exchange_outflow_btc = np.random.uniform(4000, 18000)
            
            return {
                'exchange_inflow_eth': exchange_inflow_eth,
                'exchange_outflow_eth': exchange_outflow_eth,
                'exchange_net_flow_eth': exchange_inflow_eth - exchange_outflow_eth,
                'exchange_inflow_btc': exchange_inflow_btc,
                'exchange_outflow_btc': exchange_outflow_btc,
                'exchange_net_flow_btc': exchange_inflow_btc - exchange_outflow_btc,
                'flow_sentiment': 'bullish' if (exchange_inflow_eth - exchange_outflow_eth) < 0 else 'bearish'
            }
            
        except Exception as e:
            logger.error(f"Error collecting exchange flows: {e}")
            return {'exchange_inflow_eth': 100000, 'exchange_outflow_eth': 90000}

    async def _collect_wallet_distribution(self) -> Dict[str, Any]:
        """Collect wallet distribution data"""
        try:
            # Simulate wallet distribution
            return {
                'wallets_1_10_eth': np.random.randint(1000000, 5000000),
                'wallets_10_100_eth': np.random.randint(100000, 500000),
                'wallets_100_1000_eth': np.random.randint(10000, 50000),
                'wallets_1000_plus_eth': np.random.randint(1000, 5000),
                'total_unique_wallets': np.random.randint(10000000, 50000000),
                'wallet_growth_rate': np.random.uniform(1, 5)  # Percentage daily
            }
            
        except Exception as e:
            logger.error(f"Error collecting wallet distribution: {e}")
            return {'total_unique_wallets': 25000000}

    def _store_onchain_data(self, onchain_data: Dict[str, Any]):
        """Store on-chain data"""
        try:
            # Store in memory
            self.data_history.append(onchain_data)
            
            # Keep only recent data (last 7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            self.data_history = [
                data for data in self.data_history
                if datetime.fromisoformat(data.get('timestamp', '')) >= cutoff_time
            ]
            
            # Store to file
            timestamp = datetime.now()
            filename = f"onchain_data_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = f"data/onchain/{filename}"
            
            with open(filepath, 'w') as f:
                json.dump(onchain_data, f, indent=2)
            
            logger.info(f"Stored on-chain data: {filepath}")
            
        except Exception as e:
            logger.error(f"Error storing on-chain data: {e}")

    def get_onchain_summary(self) -> Dict[str, Any]:
        """Get on-chain data summary"""
        try:
            if not self.data_history:
                return {'total_records': 0}
            
            latest_data = self.data_history[-1]
            
            return {
                'total_records': len(self.data_history),
                'last_updated': latest_data.get('timestamp', ''),
                'ethereum_gas_price': latest_data.get('ethereum_data', {}).get('gas_metrics', {}).get('gas_price_gwei', 0),
                'bitcoin_transaction_count': latest_data.get('bitcoin_data', {}).get('transaction_metrics', {}).get('transaction_count', 0),
                'defi_tvl': latest_data.get('defi_data', {}).get('total_tvl', 0),
                'whale_activity': latest_data.get('wallet_analytics', {}).get('whale_movements', {}).get('whale_transactions_24h', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting on-chain summary: {e}")
            return {'total_records': 0}

    def get_network_health(self) -> Dict[str, str]:
        """Get network health status"""
        try:
            if not self.data_history:
                return {'ethereum': 'unknown', 'bitcoin': 'unknown'}
            
            latest_data = self.data_history[-1]
            
            ethereum_health = latest_data.get('ethereum_data', {}).get('network_metrics', {}).get('network_health', 'unknown')
            bitcoin_health = latest_data.get('bitcoin_data', {}).get('network_metrics', {}).get('network_health', 'unknown')
            
            return {
                'ethereum': ethereum_health,
                'bitcoin': bitcoin_health
            }
            
        except Exception as e:
            logger.error(f"Error getting network health: {e}")
            return {'ethereum': 'unknown', 'bitcoin': 'unknown'}


# Example usage
if __name__ == "__main__":
    config = OnChainConfig()
    collector = OnChainCollector(config)
    
    # Run on-chain data collection
    asyncio.run(collector.collect_all_onchain_data()) 