"""
Comprehensive test suite for the CTA Strategy Framework.

This script runs all major components and examples to verify the system works end-to-end.
"""
import sys
import os
import subprocess
import time
from datetime import datetime

def run_test(test_name, test_file, timeout=300):
    """Run a test file and return the result."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED ({duration:.1f}s)")
            if result.stdout:
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 5:
                    print("   Last few lines of output:")
                    for line in lines[-5:]:
                        print(f"   {line}")
                else:
                    print(f"   Output: {result.stdout.strip()}")
            return True, duration, result.stdout
        else:
            print(f"❌ {test_name} FAILED ({duration:.1f}s)")
            print(f"   Return code: {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            if result.stdout:
                print(f"   Output: {result.stdout}")
            return False, duration, result.stderr
    
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} TIMEOUT (>{timeout}s)")
        return False, timeout, "Test timed out"
    
    except Exception as e:
        print(f"💥 {test_name} EXCEPTION: {e}")
        return False, 0, str(e)

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'requests', 'yfinance'
    ]
    
    optional_packages = [
        'plotly', 'streamlit', 'optuna', 'scipy', 'seaborn'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"   ❌ {package} (required)")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"   ⚠️ {package} (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {missing_required}")
        print("   Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️ Missing optional packages: {missing_optional}")
        print("   Some features may not work. Install with: pip install " + " ".join(missing_optional))
    
    print("✅ Dependency check completed")
    return True

def create_test_summary(test_results):
    """Create a comprehensive test summary."""
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result['passed'])
    total_time = sum(result['duration'] for result in test_results)
    
    summary = f"""
# 🧪 CTA Strategy Framework Test Summary

**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Results
- **Total Tests:** {total_tests}
- **Passed:** {passed_tests}
- **Failed:** {total_tests - passed_tests}
- **Success Rate:** {passed_tests/total_tests*100:.1f}%
- **Total Time:** {total_time:.1f} seconds

## Individual Test Results

| Test | Status | Duration | Notes |
|------|--------|----------|-------|
"""
    
    for result in test_results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        notes = "Success" if result['passed'] else "Check logs"
        summary += f"| {result['name']} | {status} | {result['duration']:.1f}s | {notes} |\n"
    
    summary += f"""
## System Status

"""
    
    if passed_tests == total_tests:
        summary += """✅ **All tests passed!** The CTA Strategy Framework is fully functional.

### What's Working:
- Data fetching and management
- Strategy development and signal generation
- Backtesting engine with realistic execution
- Performance analytics and metrics
- Risk management and monitoring
- Visualization and charting
- Parameter optimization
- Interactive dashboard capabilities

### Next Steps:
1. Explore the examples in the `examples/` directory
2. Run the interactive dashboard: `streamlit run src/visualization/dashboard.py`
3. Try optimizing strategies with different parameters
4. Test on different assets and time periods
5. Implement your own custom strategies
"""
    else:
        summary += f"""⚠️ **{total_tests - passed_tests} test(s) failed.** Some components may not be working correctly.

### Troubleshooting:
1. Check that all required dependencies are installed
2. Ensure you have internet connection for data fetching
3. Review error messages in the test output
4. Check the individual test files for specific issues

### Common Issues:
- **Data fetching failures**: Check internet connection and API limits
- **Import errors**: Install missing dependencies
- **Visualization errors**: Install plotly and matplotlib
- **Optimization errors**: Install optuna for Bayesian optimization
"""
    
    summary += f"""
## Framework Capabilities

### 📊 Data Management
- Multi-source data fetching (stocks, crypto)
- Intelligent caching system
- Data validation and cleaning

### 🎯 Strategy Development  
- Modular strategy architecture
- Built-in technical indicators
- Signal generation and position management

### 🔬 Backtesting Engine
- Event-driven backtesting
- Realistic order execution
- Commission and slippage modeling

### 📈 Performance Analytics
- 20+ performance metrics
- Risk analysis (VaR, CVaR, drawdown)
- Benchmark comparison

### 🛡️ Risk Management
- Position sizing algorithms
- Stop-loss and take-profit management
- Portfolio risk monitoring

### 🎨 Visualization
- Interactive charts with Plotly
- Strategy performance dashboards
- Risk metrics visualization

### ⚙️ Optimization
- Parameter optimization (Grid, Random, Bayesian)
- Walk-forward analysis
- Sensitivity analysis

## Usage Examples

Check the `examples/` directory for:
- `basic_strategy_example.py` - Simple strategy implementation
- `multi_asset_portfolio_example.py` - Portfolio-level analysis
- `optimization_example.py` - Parameter optimization

## Documentation

- README.md - Main documentation
- Individual module documentation in `src/` directories
- Generated reports in `results/` directory

---

**Happy Trading! 📈**
"""
    
    return summary

def main():
    """Run comprehensive test suite."""
    print("🚀 CTA STRATEGY FRAMEWORK COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install required packages.")
        return
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Define tests to run
    tests = [
        ("Data Fetching", "test_data_fetching.py"),
        ("Technical Indicators", "test_indicators.py"),
        ("Strategy Development", "test_strategies.py"),
        ("Backtesting Engine", "test_backtesting.py"),
        ("Risk Management", "test_risk_management.py"),
        ("Performance Analytics", "test_performance_analytics.py"),
        ("Visualization System", "test_visualization.py"),
        ("Optimization System", "test_optimization_simple.py"),
        ("Basic Strategy Example", "examples/basic_strategy_example.py"),
        ("Multi-Asset Portfolio Example", "examples/multi_asset_portfolio_example.py"),
        ("Optimization Example", "examples/optimization_example.py")
    ]
    
    # Run tests
    test_results = []
    start_time = time.time()
    
    for test_name, test_file in tests:
        if os.path.exists(test_file):
            passed, duration, output = run_test(test_name, test_file)
            test_results.append({
                'name': test_name,
                'file': test_file,
                'passed': passed,
                'duration': duration,
                'output': output
            })
        else:
            print(f"\n⚠️ Test file not found: {test_file}")
            test_results.append({
                'name': test_name,
                'file': test_file,
                'passed': False,
                'duration': 0,
                'output': "Test file not found"
            })
    
    total_time = time.time() - start_time
    
    # Generate summary
    print(f"\n{'='*70}")
    print("🎯 COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*70}")
    
    passed_tests = sum(1 for result in test_results if result['passed'])
    total_tests = len(test_results)
    
    print(f"Total tests run: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    print(f"Total time: {total_time:.1f} seconds")
    
    print(f"\n📊 Individual Test Results:")
    for result in test_results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"   {result['name']:<35} {status} ({result['duration']:5.1f}s)")
    
    # Create detailed summary report
    summary_report = create_test_summary(test_results)
    
    with open('results/test_summary.md', 'w') as f:
        f.write(summary_report)
    
    print(f"\n📄 Detailed test summary saved to: results/test_summary.md")
    
    # Final assessment
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ The CTA Strategy Framework is fully functional and ready for use!")
        print(f"\n🚀 Next steps:")
        print(f"   1. Explore examples in the examples/ directory")
        print(f"   2. Run the interactive dashboard: streamlit run src/visualization/dashboard.py")
        print(f"   3. Start developing your own strategies!")
    
    elif passed_tests >= total_tests * 0.8:
        print(f"\n✅ MOSTLY SUCCESSFUL!")
        print(f"🎯 {passed_tests}/{total_tests} tests passed - the framework is largely functional")
        print(f"⚠️ Some optional features may not be working")
        print(f"💡 Check the test summary for details on failed tests")
    
    else:
        print(f"\n⚠️ SOME ISSUES DETECTED")
        print(f"❌ {total_tests - passed_tests}/{total_tests} tests failed")
        print(f"🔧 Please review the error messages and fix issues before proceeding")
        print(f"📋 Check results/test_summary.md for detailed information")
    
    print(f"\n📁 Generated files:")
    print(f"   - results/test_summary.md (detailed test report)")
    if os.path.exists('results'):
        result_files = [f for f in os.listdir('results') if f.endswith(('.png', '.json', '.csv', '.html'))]
        for file in result_files[:10]:  # Show first 10 files
            print(f"   - results/{file}")
        if len(result_files) > 10:
            print(f"   - ... and {len(result_files) - 10} more files")
    
    print(f"\n🙏 Thank you for testing the CTA Strategy Framework!")

if __name__ == "__main__":
    main()
