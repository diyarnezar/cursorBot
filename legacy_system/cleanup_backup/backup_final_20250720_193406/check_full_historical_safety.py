#!/usr/bin/env python3
"""
Check Full Historical Data Mode Safety
"""

from datetime import datetime

def check_full_historical_safety():
    """Check if full historical data mode is safe under rate limits"""
    
    print("🔍 Full Historical Data Mode Safety Check")
    print("=" * 50)
    
    # Calculate days since ETH/FDUSD listing
    listing_date = datetime(2023, 12, 1)  # Safe starting point
    current_date = datetime.now()
    total_days = (current_date - listing_date).days
    
    print(f"📅 ETH/FDUSD listing date: {listing_date.strftime('%Y-%m-%d')}")
    print(f"📅 Current date: {current_date.strftime('%Y-%m-%d')}")
    print(f"📅 Total days: {total_days}")
    
    # Calculate API usage
    minutes_per_day = 1440  # 24 hours * 60 minutes
    total_minutes = total_days * minutes_per_day
    calls_per_symbol = (total_minutes // 1000) + 1  # Binance API limit is 1000 per call
    weight_per_symbol = calls_per_symbol * 2  # Each kline call costs 2 weight
    weight_for_26_symbols = weight_per_symbol * 26
    
    print(f"\n📊 API Usage Calculation:")
    print(f"   - Minutes per day: {minutes_per_day}")
    print(f"   - Total minutes: {total_minutes:,}")
    print(f"   - Calls per symbol: {calls_per_symbol}")
    print(f"   - Weight per symbol: {weight_per_symbol}")
    print(f"   - Weight for 26 symbols: {weight_for_26_symbols}")
    
    # Check safety
    binance_limit = 1200
    is_safe = weight_for_26_symbols <= binance_limit
    usage_percent = (weight_for_26_symbols / binance_limit) * 100
    
    print(f"\n🛡️ Safety Check:")
    print(f"   - Binance limit: {binance_limit} weight")
    print(f"   - Full historical usage: {weight_for_26_symbols} weight")
    print(f"   - Usage percentage: {usage_percent:.1f}%")
    print(f"   - Safe under limit: {'✅ YES' if is_safe else '❌ NO'}")
    
    if is_safe:
        print(f"\n🎉 Full Historical Data Mode is SAFE!")
        print(f"   - {weight_for_26_symbols} weight is safely under {binance_limit} limit")
        print(f"   - {usage_percent:.1f}% usage leaves room for other operations")
    else:
        print(f"\n⚠️ Full Historical Data Mode EXCEEDS limits!")
        print(f"   - {weight_for_26_symbols} weight exceeds {binance_limit} limit")
        print(f"   - Would need to reduce data collection or use smaller batches")
    
    return is_safe

if __name__ == "__main__":
    check_full_historical_safety() 