import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Project Hyperion Dashboard", layout="wide")
st.title("🐋 Project Hyperion - Real-Time Monitoring Dashboard")

st.markdown("""
- **Live Trades & Signals**
- **Whale Activity**
- **RL Agent Actions**
- **Performance Metrics**
- **Backtest Results**
""")

# Helper to load logs or JSON
@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def load_log(path, n=200):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return lines[-n:]
    return []

# Live trades
st.header("Live Trades & Signals")
trade_log = load_log('hyperion.log')
if trade_log:
    st.code(''.join(trade_log), language='text')
else:
    st.info("No live trade log found.")

# Whale activity
st.header("Whale Activity")
whale_data = load_json('models/autonomous_training_info.json')
if whale_data:
    st.json(whale_data)
else:
    st.info("No whale activity data found.")

# RL agent actions
st.header("RL Agent Actions")
rl_log = load_log('ultra_enhanced_training.log')
if rl_log:
    st.code(''.join([l for l in rl_log if '[RL]' in l or '[Meta-Learning]' in l or '[Adversarial]' in l]), language='text')
else:
    st.info("No RL agent log found.")

# Performance metrics
st.header("Performance Metrics")
perf_data = load_json('models/autonomous_training_info.json')
if perf_data:
    st.write(perf_data.get('model_scores', {}))
    st.write(perf_data.get('ensemble_weights', {}))
    st.write(f"Best performance: {perf_data.get('performance', 0):.3f}")
else:
    st.info("No performance data found.")

# Backtest results
st.header("Backtest Results")
backtest_log = load_log('ultra_enhanced_training.log')
if backtest_log:
    st.code(''.join([l for l in backtest_log if '[Backtest]' in l]), language='text')
else:
    st.info("No backtest log found.")

st.markdown("---")
st.markdown("**How to run:**  ")
st.code("streamlit run modules/dashboard.py", language="bash")
st.info("Dashboard auto-refreshes when you refresh the page. For real-time, run the bot and dashboard in parallel.") 