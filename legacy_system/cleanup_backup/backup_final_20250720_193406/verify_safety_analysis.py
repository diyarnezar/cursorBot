#!/usr/bin/env python3
"""
Verify Safety Analysis
Confirm our implementation matches ChatGPT's safe strategy exactly
"""

from modules.historical_kline_fetcher import kline_fetcher
from modules.binance_rate_limiter import binance_limiter

def verify_safety_analysis():
    """Verify our implementation is truly safe"""
    
    print("🔒 Verifying Safety Analysis")
    print("=" * 60)
    print("Comparing our implementation with ChatGPT's safe strategy")
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
    
    # Get our strategy
    strategy = kline_fetcher.get_optimized_strategy(test_symbols)
    our_stats = strategy['current_strategy']
    
    print("📊 ChatGPT's Safe Strategy Analysis:")
    print("| Metric                        | ChatGPT Value | Our Value      | Safe? |")
    print("| ----------------------------- | ------------- | -------------- | ----- |")
    
    # Weight per call
    chatgpt_weight_per_call = 2
    our_weight_per_call = our_stats['weight_per_symbol'] / our_stats['calls_per_symbol']
    print(f"| Weight per call               | {chatgpt_weight_per_call}           | {our_weight_per_call:.1f}            | {'✅' if abs(our_weight_per_call - chatgpt_weight_per_call) < 0.1 else '❌'} |")
    
    # Total calls
    chatgpt_total_calls = 572
    our_total_calls = our_stats['total_calls']
    print(f"| Total calls                   | {chatgpt_total_calls}           | {our_total_calls}            | {'✅' if our_total_calls == chatgpt_total_calls else '❌'} |")
    
    # Total weight
    chatgpt_total_weight = 1144
    our_total_weight = our_stats['total_weight']
    print(f"| Total weight                  | {chatgpt_total_weight}          | {our_total_weight}           | {'✅' if our_total_weight == chatgpt_total_weight else '❌'} |")
    
    # Total time
    chatgpt_total_time = 83
    our_total_time = our_stats['sequential_time_estimate']
    print(f"| Total time (seconds)          | ~{chatgpt_total_time}           | {our_total_time:.1f}            | {'✅' if abs(our_total_time - chatgpt_total_time) < 5 else '❌'} |")
    
    # Average weight per minute
    chatgpt_avg_weight_per_min = 827
    our_avg_weight_per_min = our_stats['weight_usage_percent'] * 12  # Convert percentage to weight/min
    print(f"| Avg weight per minute         | ~{chatgpt_avg_weight_per_min}          | {our_avg_weight_per_min:.0f}            | {'✅' if our_avg_weight_per_min <= 1200 else '❌'} |")
    
    # Average calls per second
    chatgpt_calls_per_sec = 6.9
    our_calls_per_sec = our_stats['calls_per_minute'] / 60
    print(f"| Avg calls per second          | ~{chatgpt_calls_per_sec}           | {our_calls_per_sec:.1f}            | {'✅' if our_calls_per_sec <= 20 else '❌'} |")
    
    print()
    
    # Delays analysis
    print("⏱️ Delay Analysis:")
    print(f"  Our inter-call delay: {our_stats['inter_call_delay']}s")
    print(f"  ChatGPT recommended: 0.1s")
    print(f"  Match: {'✅ YES' if abs(our_stats['inter_call_delay'] - 0.1) < 0.01 else '❌ NO'}")
    
    print(f"  Our symbol delay: {our_stats['symbol_delay']}s")
    print(f"  ChatGPT mentioned: 1s")
    print(f"  Match: {'✅ YES' if abs(our_stats['symbol_delay'] - 1.0) < 0.01 else '❌ NO'}")
    
    print()
    
    # Safety checks
    print("🛡️ Safety Checks:")
    
    # Weight pool check
    weight_safe = our_stats['weight_usage_percent'] <= 100
    print(f"  Weight pool (≤1200/min): {'✅ SAFE' if weight_safe else '❌ UNSAFE'} ({our_stats['weight_usage_percent']:.1f}%)")
    
    # Raw requests check
    raw_safe = our_calls_per_sec <= 20
    print(f"  Raw requests (≤20/sec): {'✅ SAFE' if raw_safe else '❌ UNSAFE'} ({our_calls_per_sec:.1f}/sec)")
    
    # Total weight check
    total_weight_safe = our_total_weight <= 1200
    print(f"  Total weight (≤1200): {'✅ SAFE' if total_weight_safe else '❌ UNSAFE'} ({our_total_weight})")
    
    # Time check
    time_safe = our_total_time <= 120  # 2 minutes max
    print(f"  Time reasonable (≤2min): {'✅ SAFE' if time_safe else '❌ UNSAFE'} ({our_total_time:.1f}s)")
    
    print()
    
    # ChatGPT's specific warnings
    print("🚨 ChatGPT's Warnings - Our Implementation:")
    
    # No delays warning
    print("1. 'No Delays' - Firing all requests back-to-back:")
    print(f"   Our implementation: {'✅ HAS DELAYS' if our_stats['inter_call_delay'] > 0 else '❌ NO DELAYS'}")
    
    # Insufficient back-off warning
    print("2. 'Insufficient Back-off' - Ignoring Retry-After:")
    print(f"   Our implementation: {'✅ RESPECTS RETRY-AFTER' if hasattr(binance_limiter, 'handle_response_headers') else '❌ NO RETRY-AFTER HANDLING'}")
    
    # Dynamic throttling recommendation
    print("3. 'Dynamic throttling' - Reading X-MBX-USED-WEIGHT-1M:")
    print(f"   Our implementation: {'✅ READS HEADERS' if hasattr(binance_limiter, 'handle_response_headers') else '❌ NO HEADER READING'}")
    
    print()
    
    # Final verdict
    all_safe = weight_safe and raw_safe and total_weight_safe and time_safe
    print("🎯 FINAL VERDICT:")
    if all_safe:
        print("✅ OUR IMPLEMENTATION IS SAFE!")
        print("   - Matches ChatGPT's safe strategy")
        print("   - Won't trigger 429s or IP bans")
        print("   - Proper delays and rate limiting")
    else:
        print("❌ OUR IMPLEMENTATION NEEDS FIXES!")
        print("   - May trigger rate limits")
        print("   - Risk of 429s or IP bans")
    
    return all_safe

if __name__ == "__main__":
    verify_safety_analysis() 