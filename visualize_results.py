"""
Visualize CTA strategy test results.
"""
import sys
sys.path.append('src')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from data import get_stock_data, get_crypto_data
from strategies import CTAStrategy
from indicators import sma, rsi

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_strategy_signals(symbol, data_fetcher, save_path=None):
    """Plot strategy signals on price chart."""
    # Get data
    data = data_fetcher(symbol, start_date='2023-06-01', end_date='2024-01-31')
    if data.empty:
        print(f"No data for {symbol}")
        return
    
    # Initialize strategy
    strategy = CTAStrategy()
    signals = strategy.generate_signals(data, symbol)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'CTA Strategy Analysis - {symbol}', fontsize=16, fontweight='bold')
    
    # Plot 1: Price and Moving Averages
    ax1.plot(data.index, data['close'], label='Close Price', linewidth=2, alpha=0.8)
    ax1.plot(signals.index, signals['fast_ma'], label='Fast MA (10)', linewidth=1.5)
    ax1.plot(signals.index, signals['slow_ma'], label='Slow MA (30)', linewidth=1.5)
    
    # Mark entry points
    long_entries = signals[signals['entry_long'] == True]
    short_entries = signals[signals['entry_short'] == True]
    
    if not long_entries.empty:
        ax1.scatter(long_entries.index, long_entries['close'], 
                   color='green', marker='^', s=100, label='Long Entry', zorder=5)
    
    if not short_entries.empty:
        ax1.scatter(short_entries.index, short_entries['close'], 
                   color='red', marker='v', s=100, label='Short Entry', zorder=5)
    
    ax1.set_title('Price Action and Moving Averages')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RSI
    ax2.plot(signals.index, signals['rsi'], label='RSI', color='purple', linewidth=2)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.fill_between(signals.index, 70, 100, alpha=0.2, color='red')
    ax2.fill_between(signals.index, 0, 30, alpha=0.2, color='green')
    
    ax2.set_title('RSI Oscillator')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Signals
    signal_values = signals['signal'].fillna(0)
    colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in signal_values]
    
    ax3.bar(signals.index, signal_values, color=colors, alpha=0.7, width=1)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_title('Trading Signals')
    ax3.set_ylabel('Signal')
    ax3.set_xlabel('Date')
    ax3.set_ylim(-1.5, 1.5)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    
    plt.show()

def plot_performance_comparison():
    """Plot performance comparison across assets."""
    # Test results data (from our comprehensive test)
    stock_results = {
        'AAPL': {'pnl': 59.42, 'win_rate': 33.3, 'trades': 6, 'volatility': 1.21},
        'TSLA': {'pnl': 1571.80, 'win_rate': 71.4, 'trades': 7, 'volatility': 3.07},
        'SPY': {'pnl': 628.35, 'win_rate': 100.0, 'trades': 2, 'volatility': 0.72},
        'GOOGL': {'pnl': -577.78, 'win_rate': 14.3, 'trades': 7, 'volatility': 1.74},
        'MSFT': {'pnl': -162.74, 'win_rate': 28.6, 'trades': 7, 'volatility': 1.33}
    }
    
    crypto_results = {
        'BTC/USDT': {'pnl': -446.59, 'win_rate': 0.0, 'trades': 3, 'volatility': 2.18},
        'ETH/USDT': {'pnl': -69.07, 'win_rate': 20.0, 'trades': 5, 'volatility': 2.39},
        'ADA/USDT': {'pnl': -221.85, 'win_rate': 16.7, 'trades': 6, 'volatility': 3.85},
        'SOL/USDT': {'pnl': -268.28, 'win_rate': 25.0, 'trades': 4, 'volatility': 4.99}
    }
    
    # Create DataFrames
    stock_df = pd.DataFrame(stock_results).T
    stock_df['asset_type'] = 'Stock'
    
    crypto_df = pd.DataFrame(crypto_results).T
    crypto_df['asset_type'] = 'Crypto'
    
    combined_df = pd.concat([stock_df, crypto_df])
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CTA Strategy Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: PnL by Asset
    colors = ['green' if x > 0 else 'red' for x in combined_df['pnl']]
    bars1 = ax1.bar(range(len(combined_df)), combined_df['pnl'], color=colors, alpha=0.7)
    ax1.set_title('Profit & Loss by Asset')
    ax1.set_ylabel('PnL ($)')
    ax1.set_xticks(range(len(combined_df)))
    ax1.set_xticklabels(combined_df.index, rotation=45)
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, combined_df['pnl']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (50 if height > 0 else -50),
                f'${value:.0f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # Plot 2: Win Rate by Asset
    bars2 = ax2.bar(range(len(combined_df)), combined_df['win_rate'], 
                   color=['blue' if x == 'Stock' else 'orange' for x in combined_df['asset_type']], 
                   alpha=0.7)
    ax2.set_title('Win Rate by Asset')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_xticks(range(len(combined_df)))
    ax2.set_xticklabels(combined_df.index, rotation=45)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, combined_df['win_rate']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Plot 3: PnL vs Volatility
    stock_mask = combined_df['asset_type'] == 'Stock'
    crypto_mask = combined_df['asset_type'] == 'Crypto'
    
    ax3.scatter(combined_df[stock_mask]['volatility'], combined_df[stock_mask]['pnl'], 
               color='blue', s=100, alpha=0.7, label='Stocks')
    ax3.scatter(combined_df[crypto_mask]['volatility'], combined_df[crypto_mask]['pnl'], 
               color='orange', s=100, alpha=0.7, label='Crypto')
    
    # Add labels for each point
    for idx, row in combined_df.iterrows():
        ax3.annotate(idx, (row['volatility'], row['pnl']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_title('PnL vs Volatility')
    ax3.set_xlabel('Daily Volatility (%)')
    ax3.set_ylabel('PnL ($)')
    ax3.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Asset Type Comparison
    asset_summary = combined_df.groupby('asset_type').agg({
        'pnl': 'mean',
        'win_rate': 'mean',
        'trades': 'mean'
    })
    
    x = range(len(asset_summary))
    width = 0.25
    
    bars1 = ax4.bar([i - width for i in x], asset_summary['pnl'], width, 
                   label='Avg PnL ($)', alpha=0.7)
    bars2 = ax4.bar(x, asset_summary['win_rate']*10, width, 
                   label='Avg Win Rate (×10)', alpha=0.7)
    bars3 = ax4.bar([i + width for i in x], asset_summary['trades']*100, width, 
                   label='Avg Trades (×100)', alpha=0.7)
    
    ax4.set_title('Asset Type Performance Summary')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels(asset_summary.index)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/cta_strategy_performance.png', dpi=300, bbox_inches='tight')
    print("Performance chart saved to results/cta_strategy_performance.png")
    plt.show()

def plot_parameter_sensitivity():
    """Plot parameter sensitivity analysis."""
    param_results = {
        'Conservative': {'pnl': 151.78, 'trades': 2, 'win_rate': 50.0},
        'Balanced': {'pnl': 254.40, 'trades': 5, 'win_rate': 40.0},
        'Trend Following': {'pnl': -16.33, 'trades': 7, 'win_rate': 28.6},
        'Aggressive': {'pnl': -749.23, 'trades': 4, 'win_rate': 0.0}
    }
    
    df = pd.DataFrame(param_results).T
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: PnL by Parameter Set
    colors = ['green' if x > 0 else 'red' for x in df['pnl']]
    bars = ax1.bar(df.index, df['pnl'], color=colors, alpha=0.7)
    ax1.set_title('PnL by Parameter Configuration')
    ax1.set_ylabel('PnL ($)')
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, df['pnl']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (20 if height > 0 else -20),
                f'${value:.0f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # Plot 2: Trades vs Win Rate
    scatter = ax2.scatter(df['trades'], df['win_rate'], 
                         s=[abs(x)*2 for x in df['pnl']], 
                         c=df['pnl'], cmap='RdYlGn', alpha=0.7)
    
    # Add labels
    for idx, row in df.iterrows():
        ax2.annotate(idx, (row['trades'], row['win_rate']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax2.set_title('Trades vs Win Rate (bubble size = |PnL|)')
    ax2.set_xlabel('Number of Trades')
    ax2.set_ylabel('Win Rate (%)')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('PnL ($)')
    
    plt.tight_layout()
    plt.savefig('results/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    print("Parameter sensitivity chart saved to results/parameter_sensitivity.png")
    plt.show()

def main():
    """Generate all visualizations."""
    print("🎨 Generating CTA Strategy Visualizations...")
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Plot 1: Strategy signals for TSLA (best performer)
    print("\n1. Plotting TSLA strategy signals...")
    plot_strategy_signals('TSLA', get_stock_data, 'results/tsla_strategy_signals.png')
    
    # Plot 2: Performance comparison
    print("\n2. Plotting performance comparison...")
    plot_performance_comparison()
    
    # Plot 3: Parameter sensitivity
    print("\n3. Plotting parameter sensitivity...")
    plot_parameter_sensitivity()
    
    print("\n✅ All visualizations completed!")
    print("Charts saved in the 'results' directory.")

if __name__ == "__main__":
    main()
