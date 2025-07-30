# 🚀 PROJECT HYPERION - USAGE GUIDE

## 🎯 **Which File Should I Run?**

### **🚀 start_hyperion.py** (THE MAIN ENTRY POINT)
**Use this for everything!** This is the complete system with all features.

```bash
# Autonomous trading (recommended)
python start_hyperion.py --mode autonomous

# Training mode
python start_hyperion.py --mode training

# Backtesting
python start_hyperion.py --mode backtest

# Paper trading
python start_hyperion.py --mode paper_trading

# With specific pairs
python start_hyperion.py --mode autonomous --pairs ETHFDUSD BTCFDUSD

# Show system status
python start_hyperion.py --status

# Export system report
python start_hyperion.py --export-report
```

### **🎓 train.py** (TRAINING ONLY)
Use this for interactive training sessions.

```bash
# Interactive training menu
python train.py --interactive

# Specific training mode
python train.py --mode month --symbols ETHFDUSD

# Train all pairs
python train.py --mode quarter --symbols all
```

## ✅ **start_hyperion.py Has Everything!**

The `start_hyperion.py` includes **ALL** core features plus advanced components:

### **✅ Core System Features**
- ✅ **HyperionStartup class** with all methods
- ✅ **startup()** - autonomous trading, backtest, paper trading
- ✅ **stop()** - graceful shutdown with signal handling
- ✅ **get_status()** - system status
- ✅ **print_status()** - status display
- ✅ **export_report()** - report export
- ✅ **Signal handling** for graceful shutdown (SIGINT, SIGTERM)
- ✅ **All command line arguments** (--pairs, --mode, --config, --status, --export-report)

### **✅ Advanced Features**
- ✅ **Complete system validation** before startup
- ✅ **All 4 phases** from gemini_plan_new.md
- ✅ **26 FDUSD pairs** with 5 asset clusters
- ✅ **300+ advanced features**
- ✅ **RL execution agent**
- ✅ **Automated strategy discovery**
- ✅ **High-fidelity backtesting**
- ✅ **Risk management system**

## 📋 **What Each Mode Does**

### **autonomous** (Default)
- Starts the complete autonomous trading system
- Uses all 26 FDUSD pairs
- Runs all 4 phases from gemini_plan_new.md
- Includes RL execution, strategy discovery, risk management
- **Recommended for production use**

### **training**
- Trains all components (asset clusters, RL agent, strategy discovery)
- Prepares the system for autonomous trading
- **Use this first time or when you want to retrain**

### **backtest**
- Runs high-fidelity backtesting
- Tests strategies without real money
- **Use this to validate strategies**

### **paper_trading**
- Simulates trading without real execution
- Tests the complete system safely
- **Use this to test before going live**

## 🎯 **Quick Start**

1. **First time setup**:
   ```bash
   python start_hyperion.py --mode training
   ```

2. **Start autonomous trading**:
   ```bash
   python start_hyperion.py --mode autonomous
   ```

3. **Monitor system**:
   ```bash
   python start_hyperion.py --status
   ```

4. **Export report**:
   ```bash
   python start_hyperion.py --export-report
   ```

## 📊 **System Features**

The complete system includes:
- ✅ **26 FDUSD pairs** organized into 5 asset clusters
- ✅ **300+ advanced features** (quantum, AI, microstructure, psychology, etc.)
- ✅ **5 specialized models** for each asset cluster
- ✅ **Reinforcement Learning** for optimal execution
- ✅ **Automated Strategy Discovery** for continuous improvement
- ✅ **Risk Management** with portfolio-level controls
- ✅ **High-Fidelity Backtesting** with realistic simulation
- ✅ **Real-time Order Book Analysis** via WebSocket
- ✅ **Emergency Circuit Breakers** for market stress
- ✅ **Graceful shutdown** with signal handling
- ✅ **System status monitoring** and reporting

## 🚀 **Ready to Start?**

**Just run:**
```bash
python start_hyperion.py --mode autonomous
```

**That's it! The system will handle everything else automatically.**

**All features are now consolidated into the single, complete start_hyperion.py! 🚀** 