import numpy as np
import pandas as pd
import logging

class RollingBacktester:
    """
    Rolling backtesting engine for Project Hyperion.
    Simulates trading using the current strategy (including RL agent and whale logic).
    Logs and returns performance metrics (PnL, Sharpe, win rate, drawdown, etc.).
    """
    def __init__(self, strategy, rl_agent=None, initial_capital=1000.0):
        self.strategy = strategy  # Should be a function: (row, prev_state) -> action
        self.rl_agent = rl_agent
        self.initial_capital = initial_capital
    def run(self, df: pd.DataFrame, verbose=False):
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        trade_history = []
        returns = []
        max_capital = capital
        min_capital = capital
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            # Build state for RL agent
            state = prev_row.values
            whale_features = {k: row[k] for k in [
                'large_trade_count', 'large_trade_volume', 'large_buy_count', 'large_sell_count',
                'large_buy_volume', 'large_sell_volume', 'whale_alert_count', 'whale_alert_flag',
                'order_book_imbalance', 'onchain_whale_inflow', 'onchain_whale_outflow'] if k in row}
            predictions = {'1m': row.get('pred_1m', 0.0), '5m': row.get('pred_5m', 0.0), '15m': row.get('pred_15m', 0.0)}
            market_analysis = {'trend': row.get('trend', 'neutral'), 'volatility_value': row.get('volatility_value', 0.0), 'rsi': row.get('rsi', 50.0), 'adx': row.get('adx', 20.0)}
            # Use RL agent or strategy
            if self.rl_agent is not None:
                action = self.rl_agent.get_action(state, predictions, market_analysis, whale_features)
            else:
                action = self.strategy(row, prev_row)
            price = row['close'] if 'close' in row else 0.0
            # Simulate trade
            if action == 1 and position == 0:  # Buy
                position = capital / price
                entry_price = price
                capital = 0.0
                trade_history.append({'type': 'buy', 'price': price, 'step': i})
            elif action == 2 and position > 0:  # Sell
                capital = position * price
                returns.append((capital - self.initial_capital) / self.initial_capital)
                max_capital = max(max_capital, capital)
                min_capital = min(min_capital, capital)
                position = 0.0
                trade_history.append({'type': 'sell', 'price': price, 'step': i})
            # Hold: do nothing
        # Finalize
        if position > 0:
            capital = position * price
            returns.append((capital - self.initial_capital) / self.initial_capital)
        pnl = capital - self.initial_capital
        win_trades = [t for t in trade_history if t['type'] == 'sell' and t['price'] > entry_price]
        win_rate = len(win_trades) / max(1, len([t for t in trade_history if t['type'] == 'sell'])) * 100
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252) if returns else 0.0
        drawdown = (min_capital - max_capital) / max_capital * 100 if max_capital > 0 else 0.0
        metrics = {
            'final_capital': capital,
            'pnl': pnl,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'drawdown': drawdown,
            'num_trades': len(trade_history)
        }
        if verbose:
            logging.info(f"[Backtest] Metrics: {metrics}")
        return metrics, trade_history 