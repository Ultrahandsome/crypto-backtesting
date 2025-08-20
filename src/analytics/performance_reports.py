"""
Performance reporting and analysis tools.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from .performance_metrics import PerformanceAnalyzer, PerformanceMetrics
import json

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """
    Generate comprehensive performance reports.
    """
    
    def __init__(self, analyzer: Optional[PerformanceAnalyzer] = None):
        """
        Initialize performance reporter.
        
        Args:
            analyzer: Performance analyzer instance
        """
        self.analyzer = analyzer or PerformanceAnalyzer()
    
    def generate_summary_report(self,
                              returns: pd.Series,
                              benchmark_returns: Optional[pd.Series] = None,
                              trades: Optional[List[Dict]] = None,
                              strategy_name: str = "Strategy") -> Dict[str, Any]:
        """
        Generate comprehensive summary report.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            trades: Trade data
            strategy_name: Name of the strategy
            
        Returns:
            Summary report dictionary
        """
        # Calculate metrics
        metrics = self.analyzer.calculate_metrics(returns, benchmark_returns, trades)
        
        # Create report
        report = {
            'strategy_name': strategy_name,
            'report_date': datetime.now().isoformat(),
            'period': {
                'start_date': str(returns.index[0]) if len(returns) > 0 else None,
                'end_date': str(returns.index[-1]) if len(returns) > 0 else None,
                'total_days': len(returns),
                'trading_days': len(returns[returns != 0]) if len(returns) > 0 else 0
            },
            'returns': {
                'total_return': f"{metrics.total_return:.2%}",
                'annualized_return': f"{metrics.annualized_return:.2%}",
                'cagr': f"{metrics.cagr:.2%}",
                'best_day': f"{returns.max():.2%}" if len(returns) > 0 else "0.00%",
                'worst_day': f"{returns.min():.2%}" if len(returns) > 0 else "0.00%",
                'positive_days': len(returns[returns > 0]) if len(returns) > 0 else 0,
                'negative_days': len(returns[returns < 0]) if len(returns) > 0 else 0
            },
            'risk': {
                'volatility': f"{metrics.volatility:.2%}",
                'downside_deviation': f"{metrics.downside_deviation:.2%}",
                'max_drawdown': f"{metrics.max_drawdown:.2%}",
                'max_drawdown_duration': f"{metrics.max_drawdown_duration} days",
                'var_95': f"{metrics.var_95:.2%}",
                'cvar_95': f"{metrics.cvar_95:.2%}",
                'ulcer_index': f"{metrics.ulcer_index:.2f}"
            },
            'risk_adjusted': {
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                'calmar_ratio': f"{metrics.calmar_ratio:.2f}",
                'omega_ratio': f"{metrics.omega_ratio:.2f}",
                'sterling_ratio': f"{metrics.sterling_ratio:.2f}",
                'recovery_factor': f"{metrics.recovery_factor:.2f}"
            },
            'distribution': {
                'skewness': f"{metrics.skewness:.2f}",
                'kurtosis': f"{metrics.kurtosis:.2f}",
                'skewness_interpretation': self._interpret_skewness(metrics.skewness),
                'kurtosis_interpretation': self._interpret_kurtosis(metrics.kurtosis)
            },
            'trading': {
                'total_trades': metrics.total_trades,
                'win_rate': f"{metrics.win_rate:.1f}%",
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'avg_win': f"${metrics.avg_win:.2f}",
                'avg_loss': f"${metrics.avg_loss:.2f}",
                'largest_win': f"${metrics.largest_win:.2f}",
                'largest_loss': f"${metrics.largest_loss:.2f}"
            }
        }
        
        # Add benchmark comparison if available
        if benchmark_returns is not None:
            benchmark_metrics = self.analyzer.calculate_metrics(benchmark_returns)
            report['benchmark_comparison'] = {
                'alpha': f"{metrics.alpha:.2%}",
                'beta': f"{metrics.beta:.2f}",
                'information_ratio': f"{metrics.information_ratio:.2f}",
                'tracking_error': f"{metrics.tracking_error:.2%}",
                'benchmark_return': f"{benchmark_metrics.total_return:.2%}",
                'excess_return': f"{metrics.total_return - benchmark_metrics.total_return:.2%}",
                'outperformance': metrics.total_return > benchmark_metrics.total_return
            }
        
        # Add performance rating
        report['performance_rating'] = self._calculate_performance_rating(metrics)
        
        return report
    
    def generate_monthly_returns_table(self, returns: pd.Series) -> pd.DataFrame:
        """
        Generate monthly returns table.
        
        Args:
            returns: Daily returns series
            
        Returns:
            Monthly returns DataFrame
        """
        if len(returns) == 0:
            return pd.DataFrame()
        
        # Resample to monthly returns
        monthly_returns = (1 + returns).resample('M').prod() - 1
        
        # Create pivot table
        monthly_data = []
        for date, ret in monthly_returns.items():
            monthly_data.append({
                'Year': date.year,
                'Month': date.strftime('%b'),
                'Return': ret
            })
        
        if not monthly_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(monthly_data)
        pivot_table = df.pivot(index='Year', columns='Month', values='Return')
        
        # Reorder columns to calendar order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Only include months that exist in the data
        available_months = [month for month in month_order if month in pivot_table.columns]
        pivot_table = pivot_table[available_months]
        
        # Add annual returns
        annual_returns = (1 + returns).resample('Y').prod() - 1
        pivot_table['Annual'] = annual_returns.values
        
        # Format as percentages
        pivot_table = pivot_table.round(4)
        
        return pivot_table
    
    def generate_drawdown_analysis(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Generate detailed drawdown analysis.
        
        Args:
            returns: Returns series
            
        Returns:
            Drawdown analysis dictionary
        """
        if len(returns) == 0:
            return {}
        
        # Calculate drawdown series
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        start_idx = None
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                end_idx = i - 1
                period_dd = drawdown.iloc[start_idx:end_idx + 1]
                max_dd = period_dd.min()
                duration = end_idx - start_idx + 1
                
                drawdown_periods.append({
                    'start_date': returns.index[start_idx],
                    'end_date': returns.index[end_idx],
                    'duration_days': duration,
                    'max_drawdown': max_dd,
                    'recovery_date': returns.index[i] if i < len(returns) else None
                })
                start_idx = None
        
        # Handle ongoing drawdown
        if start_idx is not None:
            period_dd = drawdown.iloc[start_idx:]
            max_dd = period_dd.min()
            duration = len(period_dd)
            
            drawdown_periods.append({
                'start_date': returns.index[start_idx],
                'end_date': returns.index[-1],
                'duration_days': duration,
                'max_drawdown': max_dd,
                'recovery_date': None  # Ongoing
            })
        
        # Sort by magnitude
        drawdown_periods.sort(key=lambda x: x['max_drawdown'])
        
        # Calculate statistics
        if drawdown_periods:
            avg_drawdown = np.mean([dd['max_drawdown'] for dd in drawdown_periods])
            avg_duration = np.mean([dd['duration_days'] for dd in drawdown_periods])
            max_duration = max([dd['duration_days'] for dd in drawdown_periods])
        else:
            avg_drawdown = avg_duration = max_duration = 0
        
        return {
            'max_drawdown': drawdown.min(),
            'current_drawdown': drawdown.iloc[-1],
            'avg_drawdown': avg_drawdown,
            'avg_duration': avg_duration,
            'max_duration': max_duration,
            'total_drawdown_periods': len(drawdown_periods),
            'top_5_drawdowns': drawdown_periods[:5],
            'time_in_drawdown': (in_drawdown.sum() / len(in_drawdown)) * 100
        }
    
    def generate_rolling_performance(self,
                                   returns: pd.Series,
                                   window_months: int = 12) -> pd.DataFrame:
        """
        Generate rolling performance analysis.
        
        Args:
            returns: Returns series
            window_months: Rolling window in months
            
        Returns:
            Rolling performance DataFrame
        """
        if len(returns) == 0:
            return pd.DataFrame()
        
        window_days = window_months * 21  # Approximate trading days per month
        
        if len(returns) < window_days:
            logger.warning(f"Insufficient data for {window_months}-month rolling analysis")
            return pd.DataFrame()
        
        rolling_data = []
        
        for i in range(window_days, len(returns) + 1):
            window_returns = returns.iloc[i - window_days:i]
            window_metrics = self.analyzer.calculate_metrics(window_returns)
            
            rolling_data.append({
                'date': returns.index[i - 1],
                'return': window_metrics.total_return,
                'volatility': window_metrics.volatility,
                'sharpe_ratio': window_metrics.sharpe_ratio,
                'max_drawdown': window_metrics.max_drawdown,
                'sortino_ratio': window_metrics.sortino_ratio
            })
        
        return pd.DataFrame(rolling_data).set_index('date')
    
    def export_report(self,
                     report: Dict[str, Any],
                     filepath: str,
                     format: str = 'json') -> None:
        """
        Export performance report to file.
        
        Args:
            report: Report dictionary
            filepath: Output file path
            format: Export format ('json', 'html', 'txt')
        """
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format == 'html':
            self._export_html_report(report, filepath)
        
        elif format == 'txt':
            self._export_text_report(report, filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Performance report exported to {filepath}")
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value."""
        if skewness > 0.5:
            return "Positively skewed (right tail)"
        elif skewness < -0.5:
            return "Negatively skewed (left tail)"
        else:
            return "Approximately symmetric"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value."""
        if kurtosis > 1:
            return "Leptokurtic (fat tails)"
        elif kurtosis < -1:
            return "Platykurtic (thin tails)"
        else:
            return "Approximately normal"
    
    def _calculate_performance_rating(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Calculate overall performance rating."""
        score = 0
        max_score = 100
        
        # Sharpe ratio component (30 points)
        if metrics.sharpe_ratio >= 2.0:
            score += 30
        elif metrics.sharpe_ratio >= 1.5:
            score += 25
        elif metrics.sharpe_ratio >= 1.0:
            score += 20
        elif metrics.sharpe_ratio >= 0.5:
            score += 15
        elif metrics.sharpe_ratio >= 0:
            score += 10
        
        # Return component (25 points)
        if metrics.annualized_return >= 0.20:
            score += 25
        elif metrics.annualized_return >= 0.15:
            score += 20
        elif metrics.annualized_return >= 0.10:
            score += 15
        elif metrics.annualized_return >= 0.05:
            score += 10
        elif metrics.annualized_return >= 0:
            score += 5
        
        # Drawdown component (25 points)
        if abs(metrics.max_drawdown) <= 0.05:
            score += 25
        elif abs(metrics.max_drawdown) <= 0.10:
            score += 20
        elif abs(metrics.max_drawdown) <= 0.15:
            score += 15
        elif abs(metrics.max_drawdown) <= 0.20:
            score += 10
        elif abs(metrics.max_drawdown) <= 0.30:
            score += 5
        
        # Win rate component (20 points)
        if metrics.win_rate >= 70:
            score += 20
        elif metrics.win_rate >= 60:
            score += 15
        elif metrics.win_rate >= 50:
            score += 10
        elif metrics.win_rate >= 40:
            score += 5
        
        # Determine rating
        percentage = (score / max_score) * 100
        
        if percentage >= 90:
            rating = "Excellent"
        elif percentage >= 80:
            rating = "Very Good"
        elif percentage >= 70:
            rating = "Good"
        elif percentage >= 60:
            rating = "Fair"
        elif percentage >= 50:
            rating = "Below Average"
        else:
            rating = "Poor"
        
        return {
            'score': score,
            'max_score': max_score,
            'percentage': percentage,
            'rating': rating
        }
    
    def _export_html_report(self, report: Dict[str, Any], filepath: str) -> None:
        """Export report as HTML."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report - {report['strategy_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Report: {report['strategy_name']}</h1>
                <p>Generated on: {report['report_date']}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metric">Total Return: {report['returns']['total_return']}</div>
                <div class="metric">Annualized Return: {report['returns']['annualized_return']}</div>
                <div class="metric">Sharpe Ratio: {report['risk_adjusted']['sharpe_ratio']}</div>
                <div class="metric">Max Drawdown: {report['risk']['max_drawdown']}</div>
            </div>
            
            <!-- Add more sections as needed -->
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
    
    def _export_text_report(self, report: Dict[str, Any], filepath: str) -> None:
        """Export report as text."""
        with open(filepath, 'w') as f:
            f.write(f"PERFORMANCE REPORT: {report['strategy_name']}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("RETURNS:\n")
            for key, value in report['returns'].items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nRISK METRICS:\n")
            for key, value in report['risk'].items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nRISK-ADJUSTED RETURNS:\n")
            for key, value in report['risk_adjusted'].items():
                f.write(f"  {key}: {value}\n")
            
            # Add more sections as needed
