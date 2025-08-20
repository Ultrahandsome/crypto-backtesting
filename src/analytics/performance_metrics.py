"""
Comprehensive performance analytics and metrics calculation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    cagr: float
    
    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    
    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Benchmark comparison
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float
    
    # Additional metrics
    recovery_factor: float
    ulcer_index: float
    sterling_ratio: float


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis engine.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(self,
                         returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         trades: Optional[List[Dict]] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Strategy returns series
            benchmark_returns: Benchmark returns series
            trades: List of trade dictionaries
            
        Returns:
            PerformanceMetrics object
        """
        if len(returns) == 0:
            return self._empty_metrics()
        
        # Calculate return metrics
        total_return = self._calculate_total_return(returns)
        annualized_return = self._calculate_annualized_return(returns)
        cagr = self._calculate_cagr(returns)
        
        # Calculate risk metrics
        volatility = self._calculate_volatility(returns)
        downside_deviation = self._calculate_downside_deviation(returns)
        max_drawdown, max_dd_duration = self._calculate_max_drawdown(returns)
        
        # Calculate risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
        omega_ratio = self._calculate_omega_ratio(returns)
        
        # Calculate distribution metrics
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        var_95 = self._calculate_var(returns, 0.05)
        cvar_95 = self._calculate_cvar(returns, 0.05)
        
        # Calculate trade metrics
        trade_metrics = self._calculate_trade_metrics(trades) if trades else {}
        
        # Calculate benchmark comparison metrics
        benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns)
        
        # Calculate additional metrics
        recovery_factor = self._calculate_recovery_factor(returns, max_drawdown)
        ulcer_index = self._calculate_ulcer_index(returns)
        sterling_ratio = self._calculate_sterling_ratio(returns)
        
        return PerformanceMetrics(
            # Return metrics
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            
            # Risk metrics
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            
            # Risk-adjusted returns
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            
            # Distribution metrics
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            
            # Trade metrics
            total_trades=trade_metrics.get('total_trades', 0),
            win_rate=trade_metrics.get('win_rate', 0),
            profit_factor=trade_metrics.get('profit_factor', 0),
            avg_win=trade_metrics.get('avg_win', 0),
            avg_loss=trade_metrics.get('avg_loss', 0),
            largest_win=trade_metrics.get('largest_win', 0),
            largest_loss=trade_metrics.get('largest_loss', 0),
            
            # Benchmark comparison
            alpha=benchmark_metrics.get('alpha', 0),
            beta=benchmark_metrics.get('beta', 1),
            information_ratio=benchmark_metrics.get('information_ratio', 0),
            tracking_error=benchmark_metrics.get('tracking_error', 0),
            
            # Additional metrics
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            sterling_ratio=sterling_ratio
        )
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate total return."""
        return (1 + returns).prod() - 1
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
        
        total_return = self._calculate_total_return(returns)
        years = len(returns) / 252  # Assuming daily returns
        
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_cagr(self, returns: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        return self._calculate_annualized_return(returns)
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) <= 1:
            return 0.0
        return returns.std() * np.sqrt(252)
    
    def _calculate_downside_deviation(self, returns: pd.Series, target: float = 0.0) -> float:
        """Calculate downside deviation."""
        downside_returns = returns[returns < target]
        if len(downside_returns) <= 1:
            return 0.0
        return downside_returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if len(returns) == 0:
            return 0.0, 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        
        # Calculate drawdown duration
        in_drawdown = drawdown < 0
        if not in_drawdown.any():
            return max_dd, 0
        
        # Find longest consecutive drawdown period
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return max_dd, max_dd_duration
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        volatility = self._calculate_volatility(returns)
        
        return excess_returns / volatility if volatility > 0 else 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        downside_deviation = self._calculate_downside_deviation(returns)
        
        return excess_returns / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        annualized_return = self._calculate_annualized_return(returns)
        return annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        if len(returns) == 0:
            return 0.0
        
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        
        return gains / losses if losses > 0 else float('inf')
    
    def _calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate skewness."""
        if len(returns) <= 2:
            return 0.0
        return returns.skew()
    
    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate excess kurtosis."""
        if len(returns) <= 3:
            return 0.0
        return returns.kurtosis()
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        return returns.quantile(confidence)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk."""
        if len(returns) == 0:
            return 0.0
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate trade-based metrics."""
        if not trades:
            return {}
        
        pnls = [trade.get('pnl', 0) for trade in trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        largest_win = max(pnls) if pnls else 0
        largest_loss = min(pnls) if pnls else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def _calculate_benchmark_metrics(self,
                                   returns: pd.Series,
                                   benchmark_returns: Optional[pd.Series]) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return {'alpha': 0, 'beta': 1, 'information_ratio': 0, 'tracking_error': 0}
        
        # Align returns
        aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        if len(aligned_data) <= 1:
            return {'alpha': 0, 'beta': 1, 'information_ratio': 0, 'tracking_error': 0}
        
        strategy_returns = aligned_data.iloc[:, 0]
        bench_returns = aligned_data.iloc[:, 1]
        
        # Calculate beta
        covariance = strategy_returns.cov(bench_returns)
        benchmark_variance = bench_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # Calculate alpha
        strategy_mean = strategy_returns.mean() * 252
        benchmark_mean = bench_returns.mean() * 252
        alpha = strategy_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
        
        # Calculate tracking error and information ratio
        excess_returns = strategy_returns - bench_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        return {
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error
        }
    
    def _calculate_recovery_factor(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate recovery factor."""
        total_return = self._calculate_total_return(returns)
        return total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    def _calculate_ulcer_index(self, returns: pd.Series) -> float:
        """Calculate Ulcer Index."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        return np.sqrt((drawdown ** 2).mean())
    
    def _calculate_sterling_ratio(self, returns: pd.Series) -> float:
        """Calculate Sterling ratio."""
        annualized_return = self._calculate_annualized_return(returns)
        ulcer_index = self._calculate_ulcer_index(returns)
        
        return annualized_return / (ulcer_index / 100) if ulcer_index > 0 else 0.0
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics object."""
        return PerformanceMetrics(
            total_return=0, annualized_return=0, cagr=0,
            volatility=0, downside_deviation=0, max_drawdown=0, max_drawdown_duration=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, omega_ratio=0,
            skewness=0, kurtosis=0, var_95=0, cvar_95=0,
            total_trades=0, win_rate=0, profit_factor=0, avg_win=0, avg_loss=0,
            largest_win=0, largest_loss=0,
            alpha=0, beta=1, information_ratio=0, tracking_error=0,
            recovery_factor=0, ulcer_index=0, sterling_ratio=0
        )


class RollingPerformanceAnalyzer:
    """
    Rolling performance analysis for time-varying metrics.
    """
    
    def __init__(self, window_size: int = 252):
        """
        Initialize rolling analyzer.
        
        Args:
            window_size: Rolling window size in periods (default: 252 for 1 year)
        """
        self.window_size = window_size
        self.analyzer = PerformanceAnalyzer()
    
    def calculate_rolling_metrics(self,
                                returns: pd.Series,
                                metrics: List[str] = None) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Returns series
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with rolling metrics
        """
        if metrics is None:
            metrics = ['sharpe_ratio', 'volatility', 'max_drawdown', 'sortino_ratio']
        
        if len(returns) < self.window_size:
            logger.warning(f"Insufficient data for rolling analysis: {len(returns)} < {self.window_size}")
            return pd.DataFrame()
        
        rolling_data = []
        
        for i in range(self.window_size, len(returns) + 1):
            window_returns = returns.iloc[i - self.window_size:i]
            window_metrics = self.analyzer.calculate_metrics(window_returns)
            
            metric_values = {'date': returns.index[i - 1]}
            for metric in metrics:
                metric_values[metric] = getattr(window_metrics, metric, 0)
            
            rolling_data.append(metric_values)
        
        return pd.DataFrame(rolling_data).set_index('date')
    
    def calculate_rolling_correlation(self,
                                    returns1: pd.Series,
                                    returns2: pd.Series) -> pd.Series:
        """Calculate rolling correlation between two return series."""
        aligned_data = pd.concat([returns1, returns2], axis=1, join='inner')
        return aligned_data.iloc[:, 0].rolling(window=self.window_size).corr(aligned_data.iloc[:, 1])


class PerformanceAttribution:
    """
    Performance attribution analysis.
    """
    
    def __init__(self):
        """Initialize performance attribution analyzer."""
        pass
    
    def sector_attribution(self,
                          portfolio_returns: pd.Series,
                          sector_weights: Dict[str, float],
                          sector_returns: Dict[str, pd.Series],
                          benchmark_weights: Dict[str, float],
                          benchmark_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Calculate sector-based performance attribution.
        
        Args:
            portfolio_returns: Portfolio returns
            sector_weights: Portfolio sector weights
            sector_returns: Sector return series
            benchmark_weights: Benchmark sector weights
            benchmark_returns: Benchmark sector returns
            
        Returns:
            Attribution breakdown
        """
        attribution = {}
        
        for sector in sector_weights.keys():
            if sector in benchmark_weights and sector in sector_returns:
                # Allocation effect
                weight_diff = sector_weights[sector] - benchmark_weights[sector]
                sector_return = sector_returns[sector].mean() * 252
                benchmark_sector_return = benchmark_returns[sector].mean() * 252
                
                allocation_effect = weight_diff * benchmark_sector_return
                
                # Selection effect
                return_diff = sector_return - benchmark_sector_return
                selection_effect = benchmark_weights[sector] * return_diff
                
                # Interaction effect
                interaction_effect = weight_diff * return_diff
                
                attribution[sector] = {
                    'allocation_effect': allocation_effect,
                    'selection_effect': selection_effect,
                    'interaction_effect': interaction_effect,
                    'total_effect': allocation_effect + selection_effect + interaction_effect
                }
        
        return attribution
    
    def factor_attribution(self,
                          returns: pd.Series,
                          factor_exposures: Dict[str, pd.Series],
                          factor_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Calculate factor-based performance attribution.
        
        Args:
            returns: Strategy returns
            factor_exposures: Factor exposure series
            factor_returns: Factor return series
            
        Returns:
            Factor attribution breakdown
        """
        attribution = {}
        
        for factor_name, exposures in factor_exposures.items():
            if factor_name in factor_returns:
                # Align data
                aligned_data = pd.concat([
                    returns, exposures, factor_returns[factor_name]
                ], axis=1, join='inner')
                
                if len(aligned_data) > 1:
                    factor_contribution = (aligned_data.iloc[:, 1] * aligned_data.iloc[:, 2]).mean() * 252
                    attribution[factor_name] = factor_contribution
        
        return attribution
