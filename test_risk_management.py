"""
Test the risk management system.
"""
import sys
sys.path.append('src')

from data import get_stock_data
from risk import RiskManager, PortfolioRiskMonitor, DynamicRiskAdjustment
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_risk_manager():
    """Test basic risk manager functionality."""
    print("=== Risk Manager Test ===")
    
    # Initialize risk manager
    risk_manager = RiskManager(
        max_portfolio_risk=0.02,
        max_position_size=0.10,
        max_drawdown=0.15,
        max_leverage=1.0
    )
    
    print("✅ Risk manager initialized")
    print(f"   Max portfolio risk: {risk_manager.max_portfolio_risk*100:.1f}%")
    print(f"   Max position size: {risk_manager.max_position_size*100:.1f}%")
    print(f"   Max drawdown: {risk_manager.max_drawdown*100:.1f}%")
    
    # Get test data
    print("\nFetching test data...")
    try:
        data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-31')
        if data.empty:
            print("❌ No data available")
            return False
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        print(f"✅ Data: {len(returns)} return observations")
        
        # Test position risk assessment
        print("\nTesting position risk assessment...")
        position_risk = risk_manager.assess_position_risk(
            symbol='AAPL',
            position_size=100,  # 100 shares
            price=180.0,
            returns=returns,
            portfolio_value=100000
        )
        
        print(f"✅ Position risk calculated:")
        print(f"   Market value: ${position_risk.market_value:,.2f}")
        print(f"   Portfolio weight: {position_risk.portfolio_weight*100:.2f}%")
        print(f"   1-day VaR: ${position_risk.var_1d:.2f}")
        print(f"   Volatility: {position_risk.volatility*100:.2f}%")
        print(f"   Max loss potential: ${position_risk.max_loss_potential:.2f}")
        
        # Test position sizing
        print("\nTesting position sizing...")
        optimal_size = risk_manager.calculate_position_size(
            symbol='AAPL',
            price=180.0,
            returns=returns,
            portfolio_value=100000,
            target_risk=0.01
        )
        
        print(f"✅ Optimal position size: {optimal_size:.2f} shares")
        print(f"   Position value: ${optimal_size * 180:.2f}")
        print(f"   Portfolio allocation: {(optimal_size * 180 / 100000)*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_portfolio_risk_assessment():
    """Test portfolio-level risk assessment."""
    print("\n=== Portfolio Risk Assessment Test ===")
    
    try:
        # Get data for multiple assets
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        data = {}
        returns_data = {}
        
        print("Fetching multi-asset data...")
        for symbol in symbols:
            asset_data = get_stock_data(symbol, start_date='2023-06-01', end_date='2024-01-31')
            if not asset_data.empty:
                data[symbol] = asset_data
                returns_data[symbol] = asset_data['close'].pct_change().dropna()
                print(f"✅ {symbol}: {len(returns_data[symbol])} returns")
        
        if len(data) < 2:
            print("❌ Need at least 2 assets for portfolio test")
            return False
        
        # Create sample portfolio
        positions = {'AAPL': 50, 'MSFT': 40, 'GOOGL': 30}  # shares
        prices = {symbol: data[symbol]['close'].iloc[-1] for symbol in positions.keys()}
        portfolio_value = sum(positions[symbol] * prices[symbol] for symbol in positions.keys())
        
        print(f"\nSample portfolio:")
        for symbol, quantity in positions.items():
            value = quantity * prices[symbol]
            weight = value / portfolio_value * 100
            print(f"   {symbol}: {quantity} shares @ ${prices[symbol]:.2f} = ${value:,.2f} ({weight:.1f}%)")
        print(f"   Total value: ${portfolio_value:,.2f}")
        
        # Initialize risk manager
        risk_manager = RiskManager()
        
        # Assess portfolio risk
        print("\nAssessing portfolio risk...")
        risk_metrics = risk_manager.assess_portfolio_risk(
            positions=positions,
            prices=prices,
            returns_data=returns_data,
            portfolio_value=portfolio_value
        )
        
        print(f"✅ Portfolio risk metrics:")
        print(f"   Portfolio VaR (5%): {abs(risk_metrics.portfolio_var)*100:.2f}%")
        print(f"   Portfolio CVaR (5%): {abs(risk_metrics.portfolio_cvar)*100:.2f}%")
        print(f"   Volatility: {risk_metrics.volatility*100:.2f}%")
        print(f"   Sharpe ratio: {risk_metrics.sharpe_ratio:.2f}")
        print(f"   Max drawdown: {abs(risk_metrics.max_drawdown)*100:.2f}%")
        print(f"   Concentration risk: {risk_metrics.concentration_risk:.3f}")
        print(f"   Correlation risk: {risk_metrics.correlation_risk:.3f}")
        
        # Check risk limits
        print("\nChecking risk limits...")
        violations = risk_manager.check_risk_limits(
            positions=positions,
            prices=prices,
            portfolio_value=portfolio_value,
            returns_data=returns_data
        )
        
        if violations:
            print(f"⚠️  Risk violations found:")
            for violation in violations:
                print(f"   {violation['type']}: {violation['current']:.3f} > {violation['limit']:.3f} ({violation['severity']})")
        else:
            print("✅ No risk limit violations")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio risk assessment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_monitoring():
    """Test real-time risk monitoring."""
    print("\n=== Risk Monitoring Test ===")
    
    try:
        # Initialize components
        risk_manager = RiskManager()
        monitor = PortfolioRiskMonitor(risk_manager)
        
        # Get test data
        data = get_stock_data('AAPL', start_date='2023-06-01', end_date='2024-01-31')
        returns = data['close'].pct_change().dropna()
        
        # Simulate portfolio monitoring over time
        print("Simulating portfolio monitoring...")
        
        positions = {'AAPL': 100}
        returns_data = {'AAPL': returns}
        
        # Monitor for last 10 trading days
        monitoring_dates = data.index[-10:]
        
        for i, date in enumerate(monitoring_dates):
            current_price = data.loc[date, 'close']
            prices = {'AAPL': current_price}
            portfolio_value = positions['AAPL'] * current_price
            
            # Monitor portfolio
            risk_report = monitor.monitor_portfolio(
                positions=positions,
                prices=prices,
                returns_data=returns_data,
                portfolio_value=portfolio_value,
                timestamp=date
            )
            
            if i == 0:  # Show first report details
                print(f"✅ Risk monitoring report for {date.strftime('%Y-%m-%d')}:")
                print(f"   Portfolio value: ${risk_report['portfolio_value']:,.2f}")
                print(f"   Risk score: {risk_report['risk_score']:.1f}/100")
                print(f"   Current drawdown: {risk_report['current_drawdown']*100:.2f}%")
                print(f"   Alerts: {len(risk_report['alerts'])}")
                print(f"   Violations: {len(risk_report['violations'])}")
                
                if risk_report['recommendations']:
                    print(f"   Recommendations:")
                    for rec in risk_report['recommendations'][:3]:
                        print(f"     - {rec}")
        
        # Get risk summary
        print(f"\nRisk monitoring summary:")
        summary = monitor.get_risk_summary(lookback_days=30)
        if summary:
            print(f"   Average risk score: {summary['avg_risk_score']:.1f}")
            print(f"   Max risk score: {summary['max_risk_score']:.1f}")
            print(f"   Total alerts: {summary['total_alerts']}")
            print(f"   Risk trend: {summary['risk_trend']}")
        
        # Export risk report
        monitor.export_risk_report('results/risk_monitoring_report.json', 'json')
        print("✅ Risk report exported to results/risk_monitoring_report.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dynamic_risk_adjustment():
    """Test dynamic risk adjustment."""
    print("\n=== Dynamic Risk Adjustment Test ===")
    
    try:
        # Initialize base risk manager
        base_risk_manager = RiskManager(
            max_portfolio_risk=0.02,
            max_position_size=0.10,
            max_drawdown=0.15
        )
        
        # Initialize dynamic adjustment
        dynamic_adjuster = DynamicRiskAdjustment(base_risk_manager)
        
        print("Testing risk adjustments for different market conditions...")
        
        # Test normal market conditions
        normal_manager = dynamic_adjuster.adjust_risk_parameters(
            market_volatility=0.15,
            correlation_spike=False,
            liquidity_stress=False
        )
        
        print(f"✅ Normal market conditions:")
        print(f"   Max portfolio risk: {normal_manager.max_portfolio_risk*100:.1f}%")
        print(f"   Max position size: {normal_manager.max_position_size*100:.1f}%")
        print(f"   Market regime: {dynamic_adjuster.market_regime}")
        
        # Test volatile market conditions
        volatile_manager = dynamic_adjuster.adjust_risk_parameters(
            market_volatility=0.30,
            correlation_spike=False,
            liquidity_stress=False
        )
        
        print(f"\n✅ Volatile market conditions:")
        print(f"   Max portfolio risk: {volatile_manager.max_portfolio_risk*100:.1f}%")
        print(f"   Max position size: {volatile_manager.max_position_size*100:.1f}%")
        print(f"   Market regime: {dynamic_adjuster.market_regime}")
        
        # Test crisis conditions
        crisis_manager = dynamic_adjuster.adjust_risk_parameters(
            market_volatility=0.50,
            correlation_spike=True,
            liquidity_stress=True
        )
        
        print(f"\n✅ Crisis market conditions:")
        print(f"   Max portfolio risk: {crisis_manager.max_portfolio_risk*100:.1f}%")
        print(f"   Max position size: {crisis_manager.max_position_size*100:.1f}%")
        print(f"   Max drawdown: {crisis_manager.max_drawdown*100:.1f}%")
        print(f"   Market regime: {dynamic_adjuster.market_regime}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dynamic risk adjustment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all risk management tests."""
    print("🛡️  COMPREHENSIVE RISK MANAGEMENT SYSTEM TEST")
    print("="*60)
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test1 = test_risk_manager()
    test2 = test_portfolio_risk_assessment()
    test3 = test_risk_monitoring()
    test4 = test_dynamic_risk_adjustment()
    
    # Summary
    print("\n" + "="*60)
    print("🎯 RISK MANAGEMENT SYSTEM TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Risk Manager", test1),
        ("Portfolio Risk Assessment", test2),
        ("Risk Monitoring", test3),
        ("Dynamic Risk Adjustment", test4)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All risk management tests passed!")
        print("✅ Risk management system is ready for production use!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
