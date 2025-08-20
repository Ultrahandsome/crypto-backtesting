"""
Multi-Asset Portfolio Example

This example demonstrates how to:
1. Create a multi-asset portfolio
2. Run portfolio-level backtesting
3. Analyze portfolio performance
4. Compare individual vs portfolio performance
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import get_stock_data
from strategies import CTAStrategy
from backtesting import StrategyBacktester
from analytics import PerformanceAnalyzer, PerformanceReporter
from risk import PortfolioRiskMonitor
import pandas as pd
import numpy as np

def main():
    """Run multi-asset portfolio example."""
    print("🌍 Multi-Asset Portfolio Example")
    print("="*50)
    
    # 1. Define portfolio assets
    print("\n📊 Step 1: Setting up multi-asset portfolio...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    start_date = '2023-01-01'
    end_date = '2024-01-31'
    
    print(f"Portfolio assets: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    
    # 2. Fetch data for all assets
    print("\n📈 Step 2: Fetching data for all assets...")
    data = {}
    
    for symbol in symbols:
        try:
            asset_data = get_stock_data(symbol, start_date=start_date, end_date=end_date)
            if not asset_data.empty:
                data[symbol] = asset_data
                print(f"✅ {symbol}: {len(asset_data)} days")
            else:
                print(f"❌ {symbol}: No data available")
        except Exception as e:
            print(f"❌ {symbol}: Error fetching data - {e}")
    
    if len(data) < 2:
        print("❌ Need at least 2 assets for portfolio analysis")
        return
    
    print(f"\n✅ Successfully loaded data for {len(data)} assets")
    
    # 3. Create portfolio strategy
    print("\n🎯 Step 3: Creating portfolio strategy...")
    strategy = CTAStrategy(
        fast_ma_period=10,
        slow_ma_period=30,
        rsi_period=14,
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
        position_size_pct=0.05,  # Smaller position size for diversification
        initial_capital=100000
    )
    
    print("✅ Portfolio strategy created:")
    print(f"   Position size per asset: {strategy.position_size_pct:.1%}")
    print(f"   Maximum portfolio exposure: {len(data) * strategy.position_size_pct:.1%}")
    
    # 4. Run portfolio backtest
    print("\n🔬 Step 4: Running portfolio backtest...")
    backtester = StrategyBacktester(
        strategy=strategy,
        initial_capital=100000,
        commission_rate=0.001
    )
    
    portfolio_results = backtester.run_backtest(data)
    
    print("✅ Portfolio backtest completed:")
    print(f"   Final portfolio value: ${portfolio_results['summary']['final_portfolio_value']:,.2f}")
    print(f"   Total return: {portfolio_results['summary']['total_return']:.2%}")
    print(f"   Total trades: {len(portfolio_results['trades'])}")
    
    # 5. Run individual asset backtests for comparison
    print("\n📊 Step 5: Running individual asset backtests...")
    individual_results = {}
    
    for symbol in data.keys():
        try:
            individual_strategy = CTAStrategy(
                fast_ma_period=10,
                slow_ma_period=30,
                rsi_period=14,
                stop_loss_pct=0.02,
                take_profit_pct=0.06,
                position_size_pct=0.2,  # Higher position size for individual assets
                initial_capital=100000
            )
            
            individual_backtester = StrategyBacktester(
                strategy=individual_strategy,
                initial_capital=100000,
                commission_rate=0.001
            )
            
            result = individual_backtester.run_backtest({symbol: data[symbol]})
            individual_results[symbol] = result
            
            print(f"✅ {symbol}: {result['summary']['total_return']:.2%} return")
            
        except Exception as e:
            print(f"❌ {symbol}: Error in backtest - {e}")
    
    # 6. Analyze portfolio performance
    print("\n📈 Step 6: Analyzing portfolio performance...")
    
    if not portfolio_results['equity_curve'].empty:
        # Portfolio metrics
        portfolio_equity = portfolio_results['equity_curve']['portfolio_value']
        portfolio_returns = portfolio_equity.pct_change().dropna()
        
        analyzer = PerformanceAnalyzer()
        portfolio_metrics = analyzer.calculate_metrics(
            portfolio_returns, 
            trades=portfolio_results['trades']
        )
        
        print("✅ Portfolio Performance Metrics:")
        print(f"   Total Return: {portfolio_metrics.total_return:.2%}")
        print(f"   Sharpe Ratio: {portfolio_metrics.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio: {portfolio_metrics.sortino_ratio:.2f}")
        print(f"   Maximum Drawdown: {portfolio_metrics.max_drawdown:.2%}")
        print(f"   Volatility: {portfolio_metrics.volatility:.2%}")
        print(f"   Win Rate: {portfolio_metrics.win_rate:.1f}%")
        
        # Individual asset metrics
        print(f"\n📊 Individual Asset Performance:")
        individual_metrics = {}
        
        for symbol, result in individual_results.items():
            if not result['equity_curve'].empty:
                equity = result['equity_curve']['portfolio_value']
                returns = equity.pct_change().dropna()
                metrics = analyzer.calculate_metrics(returns, trades=result['trades'])
                individual_metrics[symbol] = metrics
                
                print(f"   {symbol}:")
                print(f"     Return: {metrics.total_return:.2%}")
                print(f"     Sharpe: {metrics.sharpe_ratio:.2f}")
                print(f"     Max DD: {metrics.max_drawdown:.2%}")
        
        # 7. Portfolio risk analysis
        print(f"\n🛡️ Step 7: Portfolio risk analysis...")
        
        # Calculate asset returns for correlation analysis
        asset_returns = pd.DataFrame()
        for symbol in data.keys():
            returns = data[symbol]['close'].pct_change().dropna()
            asset_returns[symbol] = returns
        
        # Align returns
        asset_returns = asset_returns.dropna()
        
        if len(asset_returns) > 0:
            # Correlation matrix
            correlation_matrix = asset_returns.corr()
            print(f"✅ Asset Correlation Matrix:")
            print(correlation_matrix.round(3))
            
            # Portfolio risk metrics
            risk_monitor = PortfolioRiskMonitor(
                max_portfolio_risk=0.02,
                max_position_size=0.1,
                max_correlation=0.7
            )
            
            # Calculate portfolio risk (simplified)
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            portfolio_var_95 = portfolio_returns.quantile(0.05)
            
            print(f"\n📊 Portfolio Risk Metrics:")
            print(f"   Portfolio Volatility: {portfolio_volatility:.2%}")
            print(f"   Portfolio VaR (95%): {portfolio_var_95:.2%}")
            print(f"   Average Correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.3f}")
        
        # 8. Performance comparison
        print(f"\n🏆 Step 8: Performance comparison...")
        
        # Create comparison table
        comparison_data = []
        
        # Portfolio
        comparison_data.append({
            'Strategy': 'Portfolio',
            'Return': f"{portfolio_metrics.total_return:.2%}",
            'Sharpe': f"{portfolio_metrics.sharpe_ratio:.2f}",
            'Max DD': f"{portfolio_metrics.max_drawdown:.2%}",
            'Volatility': f"{portfolio_metrics.volatility:.2%}",
            'Trades': len(portfolio_results['trades'])
        })
        
        # Individual assets
        for symbol, metrics in individual_metrics.items():
            comparison_data.append({
                'Strategy': symbol,
                'Return': f"{metrics.total_return:.2%}",
                'Sharpe': f"{metrics.sharpe_ratio:.2f}",
                'Max DD': f"{metrics.max_drawdown:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Trades': len(individual_results[symbol]['trades'])
            })
        
        # Display comparison
        print("✅ Performance Comparison:")
        print(f"{'Strategy':<12} {'Return':<8} {'Sharpe':<7} {'Max DD':<8} {'Vol':<8} {'Trades':<7}")
        print("-" * 60)
        for row in comparison_data:
            print(f"{row['Strategy']:<12} {row['Return']:<8} {row['Sharpe']:<7} {row['Max DD']:<8} {row['Volatility']:<8} {row['Trades']:<7}")
        
        # 9. Generate reports
        print(f"\n📄 Step 9: Generating reports...")
        
        os.makedirs('results', exist_ok=True)
        
        # Portfolio report
        reporter = PerformanceReporter()
        portfolio_report = reporter.generate_summary_report(
            returns=portfolio_returns,
            trades=portfolio_results['trades'],
            strategy_name="Multi-Asset Portfolio"
        )
        
        reporter.export_report(portfolio_report, 'results/portfolio_report.json', 'json')
        
        # Save comparison data
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('results/performance_comparison.csv', index=False)
        
        print("✅ Reports generated:")
        print("   - results/portfolio_report.json")
        print("   - results/performance_comparison.csv")
        
        # 10. Summary and insights
        print(f"\n🎯 Step 10: Summary and insights")
        print("="*50)
        
        # Best performing individual asset
        best_individual = max(individual_metrics.items(), key=lambda x: x[1].total_return)
        worst_individual = min(individual_metrics.items(), key=lambda x: x[1].total_return)
        
        print(f"Portfolio vs Individual Performance:")
        print(f"  🏆 Best individual asset: {best_individual[0]} ({best_individual[1].total_return:.2%})")
        print(f"  📉 Worst individual asset: {worst_individual[0]} ({worst_individual[1].total_return:.2%})")
        print(f"  📊 Portfolio performance: {portfolio_metrics.total_return:.2%}")
        
        # Diversification benefit
        avg_individual_return = np.mean([m.total_return for m in individual_metrics.values()])
        diversification_benefit = portfolio_metrics.total_return - avg_individual_return
        
        print(f"\nDiversification Analysis:")
        print(f"  📈 Average individual return: {avg_individual_return:.2%}")
        print(f"  🌍 Portfolio return: {portfolio_metrics.total_return:.2%}")
        print(f"  ✨ Diversification benefit: {diversification_benefit:.2%}")
        
        # Risk analysis
        avg_individual_volatility = np.mean([m.volatility for m in individual_metrics.values()])
        risk_reduction = avg_individual_volatility - portfolio_metrics.volatility
        
        print(f"\nRisk Reduction:")
        print(f"  📊 Average individual volatility: {avg_individual_volatility:.2%}")
        print(f"  🛡️ Portfolio volatility: {portfolio_metrics.volatility:.2%}")
        print(f"  📉 Risk reduction: {risk_reduction:.2%}")
        
        print(f"\n💡 Key Insights:")
        if diversification_benefit > 0:
            print(f"  ✅ Portfolio diversification provided positive benefit")
        else:
            print(f"  ⚠️ Portfolio diversification reduced returns")
        
        if risk_reduction > 0:
            print(f"  ✅ Portfolio reduced risk through diversification")
        else:
            print(f"  ⚠️ Portfolio did not reduce risk effectively")
        
        if portfolio_metrics.sharpe_ratio > np.mean([m.sharpe_ratio for m in individual_metrics.values()]):
            print(f"  ✅ Portfolio achieved better risk-adjusted returns")
        else:
            print(f"  ⚠️ Individual assets had better risk-adjusted returns")
    
    else:
        print("❌ No portfolio equity curve data available")
    
    print("\n🎉 Multi-asset portfolio example completed!")

if __name__ == "__main__":
    main()
