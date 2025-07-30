#!/usr/bin/env python3
"""
Test 1-Minute Intervals
Demonstrate the corrected calculations for 1-minute intervals
"""

from modules.historical_kline_fetcher import kline_fetcher

def test_1minute_intervals():
    """Test the corrected 1-minute interval calculations"""
    
    print("🔧 Testing 1-Minute Interval Calculations")
    print("=" * 60)
    print("CORRECTED: Now properly handles 1-minute intervals for 15 days")
    print()
    
    # Test symbols (all 26 FDUSD pairs)
    test_symbols = [
        'ETHFDUSD', 'BTCFDUSD', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT',
        'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
        'LINKUSDT', 'UNIUSDT', 'LTCUSDT', 'BCHUSDT', 'ATOMUSDT',
        'ETCUSDT', 'FILUSDT', 'NEARUSDT', 'APTUSDT', 'OPUSDT',
        'ARBUSDT', 'MKRUSDT', 'AAVEUSDT', 'SNXUSDT', 'COMPUSDT',
        'SUSHIUSDT'
    ]
    
    # Test 1: Show corrected calculations
    print("📊 Test 1: Corrected calculations for 1-minute intervals")
    
    # Manual calculation
    days = 15
    minutes_per_day = 24 * 60
    total_minutes = days * minutes_per_day
    calls_per_symbol = (total_minutes + 1000 - 1) // 1000  # 1000 klines per call
    weight_per_call = 2
    total_calls = len(test_symbols) * calls_per_symbol
    total_weight = total_calls * weight_per_call
    
    print("Manual calculation:")
    print(f"  Days: {days}")
    print(f"  Minutes per day: {minutes_per_day}")
    print(f"  Total minutes: {total_minutes:,}")
    print(f"  Calls per symbol: {calls_per_symbol}")
    print(f"  Weight per call: {weight_per_call}")
    print(f"  Total calls: {total_calls:,}")
    print(f"  Total weight: {total_weight:,}")
    print(f"  Weight usage: {(total_weight/1200)*100:.1f}%")
    
    # Test 2: Strategy validation
    print("\n🔍 Test 2: Strategy validation")
    
    is_valid = kline_fetcher.validate_strategy(test_symbols)
    print(f"Strategy validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    
    # Test 3: Get optimized strategy
    print("\n⚡ Test 3: Optimized strategy")
    
    strategy = kline_fetcher.get_optimized_strategy(test_symbols)
    
    print("Strategy analysis:")
    for key, value in strategy['current_strategy'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nOptimization needed: {strategy['optimization_needed']}")
    print(f"Recommendation: {strategy['recommendation']}")
    
    # Test 4: Show the analysis
    print("\n🚨 Test 4: Analysis")
    
    if strategy['optimization_needed']:
        print("❌ PROBLEM: Current strategy exceeds rate limits!")
        print(f"   Required weight: {strategy['current_strategy']['total_weight']:,}")
        print(f"   Available weight: 1,200")
        print(f"   Excess: {strategy['current_strategy']['total_weight'] - 1200:,}")
        
        print("\n💡 SOLUTIONS:")
        print("1. Extended delays:")
        print(f"   Use {strategy['safe_inter_call_delay']:.3f}s delay between calls")
        print(f"   Total time: {strategy['required_minutes']:.1f} minutes")
        
        print("\n2. Split into sessions:")
        print(f"   Split into {strategy['required_minutes']:.0f} separate 1-minute sessions")
        print(f"   Each session: {1200 // len(test_symbols)} calls per symbol")
        
        print("\n3. Reduce data span:")
        safe_days = (1200 / (len(test_symbols) * weight_per_call)) * (1000 / minutes_per_day)
        print(f"   Safe days: {safe_days:.1f} days (instead of {days})")
        
        print("\n4. Use higher timeframes:")
        print("   Consider 5-minute or 15-minute intervals for longer periods")
        
    else:
        print("✅ Current strategy is SAFE!")
        print(f"   Weight usage: {strategy['current_strategy']['weight_usage_percent']:.1f}%")
        print(f"   Time estimate: {strategy['current_strategy']['sequential_time_estimate']:.0f}s ({strategy['current_strategy']['sequential_time_estimate']/60:.1f}min)")
    
    # Test 5: Practical recommendations
    print("\n🎯 Test 5: Practical recommendations")
    
    if strategy['optimization_needed']:
        print("For 15 days of 1-minute data on 26 pairs:")
        print("1. Use extended delays (recommended):")
        print(f"   - Delay: {strategy['safe_inter_call_delay']:.3f}s between calls")
        print(f"   - Time: {strategy['required_minutes']:.1f} minutes total")
        
        print("\n2. Alternative: Split into sessions:")
        sessions = int(strategy['required_minutes']) + 1
        calls_per_session = 1200 // len(test_symbols)
        print(f"   - {sessions} sessions of 1 minute each")
        print(f"   - {calls_per_session} calls per symbol per session")
        
        print("\n3. Alternative: Reduce timeframe:")
        print("   - Use 5-minute intervals for 15 days")
        print("   - Use 1-minute intervals for 3 days")
    else:
        print("✅ Current strategy works perfectly!")
        print("For 15 days of 1-minute data on 26 pairs:")
        print(f"1. Sequential approach:")
        print(f"   - {strategy['current_strategy']['total_calls']} total calls")
        print(f"   - {strategy['current_strategy']['weight_usage_percent']:.1f}% weight usage")
        print(f"   - {strategy['current_strategy']['sequential_time_estimate']:.0f}s total time")
        
        print(f"\n2. Rate limiting:")
        print(f"   - {strategy['current_strategy']['inter_call_delay']}s delay between calls")
        print(f"   - {strategy['current_strategy']['symbol_delay']}s delay between symbols")
        print(f"   - Safe under 1,200 weight/minute limit")
        
        print(f"\n3. Data volume:")
        print(f"   - {strategy['current_strategy']['total_minutes']:,} total minutes")
        print(f"   - {strategy['current_strategy']['total_minutes'] * len(test_symbols):,} total data points")
        print(f"   - {strategy['current_strategy']['total_minutes'] * len(test_symbols) * 1000:,} total klines")

if __name__ == "__main__":
    test_1minute_intervals() 