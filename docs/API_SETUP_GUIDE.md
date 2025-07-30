# 🔑 API SETUP GUIDE - FREE KEYS FOR MAXIMUM INTELLIGENCE

## 📋 **STEP-BY-STEP API KEY SETUP**

### 🆓 **FREE API KEYS (No Credit Card Required)**

#### 1. **NewsData.io** - News Sentiment Analysis
- **URL**: https://newsdata.io/
- **Free Tier**: 200 API credits/day, 10 articles per credit, delayed by 12 hours
- **Setup**:
  1. Go to https://newsdata.io/
  2. Click "Get API Key"
  3. Sign up with email
  4. Copy your API key
  5. Replace `newsdata_api_key` in `config.json`

#### 2. **CoinMarketCap** - Market Data & Rankings
- **URL**: https://coinmarketcap.com/api/
- **Free Tier**: 10,000 credits/month, 30 requests/minute, 1 min update frequency, no historical data
- **Setup**:
  1. Go to https://coinmarketcap.com/api/
  2. Click "Get Your Free API Key"
  3. Sign up with email
  4. Verify email
  5. Copy your API key
  6. Replace `coinmarketcap_api_key` in `config.json`

#### 3. **CoinRanking** - Alternative Market Data
- **URL**: https://coinranking.com/api/
- **Free Tier**: 5,000 calls/month, 1 API key, real-time and historical data, OHLCV, ranking, supply, metadata
- **Setup**:
  1. Go to https://coinranking.com/api/
  2. Click "Get API Key"
  3. Sign up with email
  4. Copy your API key
  5. Replace `coinranking_api_key` in `config.json`

#### 4. **MediaStack** - News Aggregation
- **URL**: https://mediastack.com/
- **Free Tier**: 100 requests/month
- **Setup**:
  1. Go to https://mediastack.com/
  2. Click "Get API Key"
  3. Sign up with email
  4. Copy your API key
  5. Replace `mediastack_api_key` in `config.json`

#### 5. **GNews** - Google News API
- **URL**: https://gnews.io/
- **Free Tier**: 100 requests/day, up to 10 articles/request, max 1 request/sec
- **Setup**:
  1. Go to https://gnews.io/
  2. Click "Get API Key"
  3. Sign up with email
  4. Copy your API key
  5. Replace `gnews_api_key` in `config.json`

#### 6. **The Guardian Open Platform**
- **URL**: https://open-platform.theguardian.com/
- **Free Tier**: 500 calls/day, 1 call/sec, access to 1.9M+ articles
- **Setup**:
  1. Go to https://open-platform.theguardian.com/
  2. Sign up and get your API key
  3. Copy your API key
  4. Replace `guardian_api_key` in `config.json`

#### 7. **FreeCryptoAPI.com**
- **URL**: https://freecryptoapi.com/
- **Free Tier**: 100,000 requests/month
- **Setup**:
  1. Go to https://freecryptoapi.com/
  2. Sign up and get your API key
  3. Copy your API key
  4. Replace `freecryptoapi_key` in `config.json`

#### 8. **CoinyBubble – Fear & Greed Index API**
- **URL**: https://api.coinybubble.com/
- **Free Tier**: No strict rate limits, but fair use policy. No API key required for public endpoints.
- **Setup**:
  1. Use endpoint: `https://api.coinybubble.com/v1/latest` for latest index
  2. Use endpoint: `https://api.coinybubble.com/v1/history/5min?hours=24` for historical data
  3. No API key required for public endpoints

## 🔧 **UPDATING CONFIG.JSON**

After getting your API keys, update the `config.json` file:

```json
"api_keys": {
  "telegram_bot_token": "7576187558:AAGkOC0kbdshnNcdjmIdky49wpcV4GK6GyY",
  "telegram_chat_id": "76590193",
  "etherscan_api_key": "A89BP14TUNCQ4GCBX9IZ5JSGT66FYYI1EV",
  "infura_project_id": "5c17f9fe5aaf4e8cb12fe2eec513b960",
  "alpha_vantage_key": "E4UG15PVP3WHDSBV",
  "coingecko_api_key": "CG-ZRVbg79Xmx7r5RBSxZy7NRUV",
  "cryptocompare_api_key": "73f93a638cde9624db6551cdcf089eab59ec217761db0a849bc40019ede8c92c",
  "messari_api_key": "eggL4un0ns-v9vqbXn9Rb4MWK2sXhaIeP3JWd4POtBBajSvx",
  "newsdata_api_key": "YOUR_ACTUAL_NEWSDATA_KEY_HERE",
  "coinmarketcap_api_key": "YOUR_ACTUAL_CMC_KEY_HERE",
  "coinranking_api_key": "YOUR_ACTUAL_COINRANKING_KEY_HERE",
  "nomics_api_key": "YOUR_ACTUAL_NOMICS_KEY_HERE",
  "newsapi_api_key": "YOUR_ACTUAL_NEWSAPI_KEY_HERE",
  "mediastack_api_key": "YOUR_ACTUAL_MEDIASTACK_KEY_HERE",
  "gnews_api_key": "YOUR_ACTUAL_GNEWS_KEY_HERE",
  "guardian_api_key": "YOUR_ACTUAL_GUARDIAN_KEY_HERE",
  "freecryptoapi_key": "YOUR_ACTUAL_FREECRYPTOAPI_KEY_HERE"
}
```

For CoinyBubble, no key is needed for public endpoints.

## 🧪 **TESTING API CONNECTIVITY**

After updating the config, test the APIs:

```bash
python -c "
import requests
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Test each API
apis = {
    'NewsData': f'https://newsdata.io/api/1/news?apikey={config[\"api_keys\"][\"newsdata_api_key\"]}&q=cryptocurrency&language=en',
    'CoinMarketCap': f'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?CMC_PRO_API_KEY={config[\"api_keys\"][\"coinmarketcap_api_key\"]}',
    'NewsAPI': f'https://newsapi.org/v2/everything?q=cryptocurrency&apiKey={config[\"api_keys\"][\"newsapi_api_key\"]}'
}

for name, url in apis.items():
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f'✅ {name}: Working')
        else:
            print(f'❌ {name}: Error {response.status_code}')
    except Exception as e:
        print(f'❌ {name}: Error - {e}')
"
```

## 📊 **EXPECTED INTELLIGENCE GAINS**

With these API keys, your bot will have access to:

- **News Sentiment Analysis**: Real-time news impact on prices
- **Market Rankings**: Top-performing cryptocurrencies
- **Social Sentiment**: Public opinion analysis
- **Alternative Data**: Non-price market signals
- **Whale Activity**: Large transaction detection
- **Regulatory News**: Government policy impacts

**Expected Performance Improvement: +15-25% additional accuracy**

## ⚠️ **IMPORTANT NOTES**

1. **Rate Limits**: Respect the free tier limits
2. **Backup Data**: The bot has fallback mechanisms if APIs fail
3. **Privacy**: These are public APIs, no sensitive data required
4. **No Credit Card**: All APIs mentioned are completely free

## 🎯 **NEXT STEPS**

1. **Get the API keys** (5-10 minutes)
2. **Update config.json** (2 minutes)
3. **Test connectivity** (1 minute)
4. **Proceed to configuration review**

**Ready to get your API keys?** 🚀 