# 🚀 Project Hyperion - API Limits Compliance Guide

## 📊 Free APIs with Rate Limits

### 🔑 APIs Requiring API Keys (Free Tier)

| API Service | Rate Limits | Free Tier | Key Required | Safety Margin |
|-------------|-------------|-----------|--------------|---------------|
| **CoinMarketCap** | 30/min, 10,000/month | ✅ Free | ✅ Required | 0.8 |
| **CoinRanking** | 10,000/month (~0.12/sec) | ✅ Free | ✅ Required | 0.8 |
| **Nomics** | Unlimited | ✅ Free | ✅ Required | 0.8 |
| **Messari** | Anon: 20/min, 1,000/day<br>Key: 30/min, 2,000/day | ✅ Free | ✅ Required | 0.8 |
| **Alpha Vantage** | 5/min, ~500/day | ✅ Free | ✅ Required | 0.8 |
| **Etherscan** | 5/sec, 100,000/day | ✅ Free | ✅ Required | 0.8 |
| **NewsData.io** | 300 credits/15 min (~5/sec), 2,000/day | ✅ Free | ✅ Required | 0.8 |
| **NewsAPI.org** | 100/day | ✅ Free | ✅ Required | 0.8 |
| **Mediastack** | ~500/month | ✅ Free | ✅ Required | 0.8 |
| **Gnews.io** | 100/day | ✅ Free | ✅ Required | 0.8 |

### 🔓 APIs Without API Keys (Free Tier)

| API Service | Rate Limits | Free Tier | Key Required | Safety Margin |
|-------------|-------------|-----------|--------------|---------------|
| **Binance** | 1,200/min, ~100,000/day | ✅ Free | ❌ Not Required | 0.8 |
| **CoinGecko** | 5-15/min<br>Demo key: 30/min, 10,000/month | ✅ Free | ❌ Not Required | 0.8 |
| **0x API** | 10/sec | ✅ Free | ❌ Not Required | 0.8 |
| **Infura** | ~1-2/sec, 100,000/day | ✅ Free | ❌ Not Required | 0.8 |
| **CoinLib** | 120/hr | ✅ Free | ❌ Not Required | 0.8 |
| **CryptoCompare** | ~8,000/hr (~133/min) | ✅ Free | ❌ Not Required | 0.8 |
| **Coinpaprika** | ~100/min | ✅ Free | ❌ Not Required | 0.8 |
| **CoinLore** | ~60/min | ✅ Free | ❌ Not Required | 0.8 |
| **CoinDesk News** | ~200/day | ✅ Free | ❌ Not Required | 0.8 |

## 🛡️ Rate Limiting Implementation

### Smart Rate Limiter Features

1. **Per-API Tracking**: Each API has its own rate limit counter
2. **Automatic Backoff**: Exponential backoff when limits are approached
3. **Request Queuing**: Requests are queued and spaced out automatically
4. **Cache Integration**: Reduces API calls through intelligent caching
5. **Fallback Mechanisms**: Multiple data sources for redundancy

### Rate Limit Safety Margins

The bot uses **80% of the actual limits** to ensure safety:

- If limit is 30/min → Bot uses max 24/min
- If limit is 10,000/month → Bot uses max 8,000/month
- If limit is 5/sec → Bot uses max 4/sec

## 🔧 Configuration

### API Keys Setup

Add your API keys to `config.json`:

```json
{
  "api_keys": {
    "coinmarketcap_api_key": "YOUR_CMC_API_KEY",
    "coinranking_api_key": "YOUR_COINRANKING_API_KEY",
    "nomics_api_key": "YOUR_NOMICS_API_KEY",
    "messari_api_key": "YOUR_MESSARI_API_KEY",
    "alpha_vantage_api_key": "YOUR_ALPHA_VANTAGE_API_KEY",
    "etherscan_api_key": "YOUR_ETHERSCAN_API_KEY",
    "newsdata_api_key": "YOUR_NEWSDATA_API_KEY",
    "newsapi_api_key": "YOUR_NEWSAPI_API_KEY",
    "mediastack_api_key": "YOUR_MEDIASTACK_API_KEY",
    "gnews_api_key": "YOUR_GNEWS_API_KEY"
  }
}
```

### Rate Limit Configuration

Rate limits are automatically configured in `config.json`:

```json
{
  "cache_settings": {
    "rate_limiting": {
      "enabled": true,
      "safety_margin": 0.8,
      "api_specific_limits": {
        "coinmarketcap": {
          "requires_key": true,
          "per_minute": 30,
          "per_month": 10000,
          "safety_margin": 0.8
        },
        "binance": {
          "requires_key": false,
          "per_minute": 1200,
          "per_day": 100000,
          "safety_margin": 0.8
        }
      }
    }
  }
}
```

## 🚀 Getting API Keys

### Free API Key Registration Links

1. **CoinMarketCap**: https://coinmarketcap.com/api/
2. **CoinRanking**: https://coinranking.com/api
3. **Nomics**: https://nomics.com/docs/
4. **Messari**: https://messari.io/api
5. **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
6. **Etherscan**: https://etherscan.io/apis
7. **NewsData.io**: https://newsdata.io/
8. **NewsAPI.org**: https://newsapi.org/
9. **Mediastack**: https://mediastack.com/
10. **Gnews.io**: https://gnews.io/

## 📈 Data Collection Strategy

### Intelligent Data Prioritization

1. **Primary Sources**: Binance, CoinGecko (no limits)
2. **Secondary Sources**: Messari, CryptoCompare (high limits)
3. **Tertiary Sources**: Coinpaprika, CoinLore (moderate limits)
4. **Specialized Sources**: Etherscan, Infura (for blockchain data)

### Cache Strategy

- **Market Data**: 5 minutes cache
- **Alternative Data**: 15 minutes cache
- **News Data**: 30 minutes cache
- **On-chain Data**: 1 hour cache

## 🔍 Monitoring & Alerts

The bot includes built-in monitoring:

- **Rate Limit Warnings**: When approaching 80% of limits
- **API Error Tracking**: Automatic fallback to alternative sources
- **Performance Metrics**: Response times and success rates
- **Usage Analytics**: Daily API call statistics

## ✅ Compliance Checklist

- [x] All APIs respect rate limits
- [x] 80% safety margin implemented
- [x] Automatic backoff and retry logic
- [x] Intelligent caching system
- [x] Multiple data source fallbacks
- [x] Real-time monitoring and alerts
- [x] Configurable limits per API
- [x] Request queuing and spacing

---

**Note**: This configuration ensures your bot will never hit API limits while maximizing data collection for optimal trading performance! 🎯 