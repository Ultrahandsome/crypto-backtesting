# 🚀 CTA Strategy Dashboard Instructions

## Running the Interactive Dashboard

To run the interactive Streamlit dashboard:

1. **Install Streamlit** (if not already installed):
	```bash
	pip install streamlit plotly
	```

2. **Run the dashboard**:
	```bash
	streamlit run src/visualization/dashboard.py
	```

3. **Access the dashboard**:
	- Open your browser to http://localhost:8501
	- The dashboard will load automatically

## Dashboard Features

### 📊 **Strategy Configuration**
- Select asset type (Stock or Cryptocurrency)
- Choose from popular symbols
- Set date range for analysis
- Adjust strategy parameters (MA periods, RSI, stop loss, etc.)
- Configure risk management settings

### 📈 **Interactive Visualizations**
- Strategy overview with price, signals, and indicators
- Equity curve with drawdown analysis
- Returns distribution and analysis
- Trade analysis and statistics

### 🛡️ **Risk Analysis**
- Real-time risk metrics
- Portfolio risk monitoring
- Performance vs benchmark comparison

### 📋 **Performance Reports**
- Comprehensive performance metrics
- Trade statistics and analysis
- Risk-adjusted returns
- Detailed analytics

## Usage Tips

1. **Start Simple**: Begin with default parameters and a familiar stock like AAPL
2. **Experiment**: Try different parameter combinations to see their impact
3. **Compare Assets**: Test the same strategy on different assets
4. **Analyze Results**: Use the detailed metrics to understand performance drivers
5. **Export Data**: Charts and data can be downloaded for further analysis

## Troubleshooting

- **Data Issues**: Ensure you have internet connection for data fetching
- **Performance**: Large date ranges may take longer to process
- **Browser**: Use Chrome or Firefox for best compatibility

Enjoy exploring your CTA strategy performance! 🎯
