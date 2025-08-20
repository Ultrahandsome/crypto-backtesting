"""
Comprehensive risk management system for trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    leverage_ratio: float


@dataclass
class PositionRisk:
    """Risk metrics for individual positions."""
    symbol: str
    position_size: float
    market_value: float
    portfolio_weight: float
    var_1d: float
    volatility: float
    beta: float
    correlation: float
    max_loss_potential: float


class RiskManager:
    """
    Comprehensive risk management system.
    """
    
    def __init__(self,
                 max_portfolio_risk: float = 0.02,
                 max_position_size: float = 0.10,
                 max_drawdown: float = 0.15,
                 max_leverage: float = 1.0,
                 var_confidence: float = 0.05,
                 lookback_period: int = 252):
        """
        Initialize risk manager.
        
        Args:
            max_portfolio_risk: Maximum daily portfolio VaR (2%)
            max_position_size: Maximum position size as % of portfolio (10%)
            max_drawdown: Maximum allowed drawdown (15%)
            max_leverage: Maximum leverage ratio (1.0 = no leverage)
            var_confidence: VaR confidence level (5% = 95% VaR)
            lookback_period: Lookback period for risk calculations (252 days)
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.var_confidence = var_confidence
        self.lookback_period = lookback_period
        
        # Risk state tracking
        self.current_drawdown = 0.0
        self.peak_portfolio_value = 0.0
        self.risk_alerts = []
        
    def assess_portfolio_risk(self,
                            positions: Dict[str, float],
                            prices: Dict[str, float],
                            returns_data: Dict[str, pd.Series],
                            portfolio_value: float,
                            benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """
        Assess comprehensive portfolio risk.
        
        Args:
            positions: Current positions {symbol: quantity}
            prices: Current prices {symbol: price}
            returns_data: Historical returns {symbol: returns series}
            portfolio_value: Current portfolio value
            benchmark_returns: Benchmark returns for beta calculation
            
        Returns:
            RiskMetrics object
        """
        # Calculate portfolio weights
        weights = self._calculate_portfolio_weights(positions, prices, portfolio_value)
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(weights, returns_data)
        
        # Calculate VaR and CVaR
        portfolio_var = self._calculate_var(portfolio_returns, self.var_confidence)
        portfolio_cvar = self._calculate_cvar(portfolio_returns, self.var_confidence)
        
        # Calculate drawdown
        max_drawdown = self._calculate_max_drawdown(portfolio_returns, portfolio_value)
        
        # Calculate volatility metrics
        volatility = portfolio_returns.std() * np.sqrt(252) if len(portfolio_returns) > 1 else 0
        
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        
        # Calculate Sortino ratio
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        
        # Calculate beta
        beta = self._calculate_beta(portfolio_returns, benchmark_returns) if benchmark_returns is not None else 1.0
        
        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk(weights, returns_data)
        
        # Calculate concentration risk
        concentration_risk = self._calculate_concentration_risk(weights)
        
        # Calculate leverage ratio
        leverage_ratio = self._calculate_leverage_ratio(positions, prices, portfolio_value)
        
        return RiskMetrics(
            portfolio_var=portfolio_var,
            portfolio_cvar=portfolio_cvar,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            beta=beta,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            leverage_ratio=leverage_ratio
        )
    
    def assess_position_risk(self,
                           symbol: str,
                           position_size: float,
                           price: float,
                           returns: pd.Series,
                           portfolio_value: float,
                           benchmark_returns: Optional[pd.Series] = None) -> PositionRisk:
        """
        Assess risk for individual position.
        
        Args:
            symbol: Asset symbol
            position_size: Position size (quantity)
            price: Current price
            returns: Historical returns
            portfolio_value: Current portfolio value
            benchmark_returns: Benchmark returns for beta calculation
            
        Returns:
            PositionRisk object
        """
        market_value = abs(position_size * price)
        portfolio_weight = market_value / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate position VaR
        var_1d = self._calculate_var(returns, self.var_confidence) * market_value
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Calculate beta
        beta = self._calculate_beta(returns, benchmark_returns) if benchmark_returns is not None else 1.0
        
        # Calculate correlation (placeholder - would need other asset returns)
        correlation = 0.0
        
        # Calculate maximum loss potential (based on historical worst case)
        max_loss_potential = returns.min() * market_value if len(returns) > 0 else 0
        
        return PositionRisk(
            symbol=symbol,
            position_size=position_size,
            market_value=market_value,
            portfolio_weight=portfolio_weight,
            var_1d=var_1d,
            volatility=volatility,
            beta=beta,
            correlation=correlation,
            max_loss_potential=max_loss_potential
        )
    
    def check_risk_limits(self,
                         positions: Dict[str, float],
                         prices: Dict[str, float],
                         portfolio_value: float,
                         returns_data: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """
        Check if current portfolio violates risk limits.
        
        Args:
            positions: Current positions
            prices: Current prices
            portfolio_value: Current portfolio value
            returns_data: Historical returns data
            
        Returns:
            List of risk violations
        """
        violations = []
        
        # Check position size limits
        for symbol, quantity in positions.items():
            if symbol in prices and quantity != 0:
                market_value = abs(quantity * prices[symbol])
                weight = market_value / portfolio_value if portfolio_value > 0 else 0
                
                if weight > self.max_position_size:
                    violations.append({
                        'type': 'position_size',
                        'symbol': symbol,
                        'current': weight,
                        'limit': self.max_position_size,
                        'severity': 'high' if weight > self.max_position_size * 1.5 else 'medium'
                    })
        
        # Check portfolio VaR
        weights = self._calculate_portfolio_weights(positions, prices, portfolio_value)
        portfolio_returns = self._calculate_portfolio_returns(weights, returns_data)
        
        if len(portfolio_returns) > 0:
            portfolio_var = abs(self._calculate_var(portfolio_returns, self.var_confidence))
            
            if portfolio_var > self.max_portfolio_risk:
                violations.append({
                    'type': 'portfolio_var',
                    'current': portfolio_var,
                    'limit': self.max_portfolio_risk,
                    'severity': 'high' if portfolio_var > self.max_portfolio_risk * 1.5 else 'medium'
                })
        
        # Check drawdown
        if self.current_drawdown > self.max_drawdown:
            violations.append({
                'type': 'max_drawdown',
                'current': self.current_drawdown,
                'limit': self.max_drawdown,
                'severity': 'extreme'
            })
        
        # Check leverage
        leverage = self._calculate_leverage_ratio(positions, prices, portfolio_value)
        if leverage > self.max_leverage:
            violations.append({
                'type': 'leverage',
                'current': leverage,
                'limit': self.max_leverage,
                'severity': 'high'
            })
        
        return violations
    
    def calculate_position_size(self,
                              symbol: str,
                              price: float,
                              returns: pd.Series,
                              portfolio_value: float,
                              target_risk: float = 0.01) -> float:
        """
        Calculate optimal position size based on risk.
        
        Args:
            symbol: Asset symbol
            price: Current price
            returns: Historical returns
            portfolio_value: Current portfolio value
            target_risk: Target risk as % of portfolio
            
        Returns:
            Optimal position size (quantity)
        """
        if len(returns) < 20:  # Need minimum data
            return 0.0
        
        # Calculate volatility
        volatility = returns.std()
        
        if volatility == 0:
            return 0.0
        
        # Kelly criterion with safety factor
        mean_return = returns.mean()
        kelly_fraction = mean_return / (volatility ** 2) if volatility > 0 else 0
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Risk-based sizing
        var_based_size = (target_risk * portfolio_value) / (abs(self._calculate_var(returns, 0.05)) * price)
        
        # Volatility-based sizing
        vol_based_size = (target_risk * portfolio_value) / (volatility * price * np.sqrt(252))
        
        # Take the minimum for conservative sizing
        position_value = min(
            kelly_fraction * portfolio_value,
            var_based_size * price,
            vol_based_size * price,
            self.max_position_size * portfolio_value
        )
        
        return position_value / price if price > 0 else 0.0
    
    def update_drawdown(self, current_value: float) -> float:
        """
        Update current drawdown tracking.
        
        Args:
            current_value: Current portfolio value
            
        Returns:
            Current drawdown percentage
        """
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        if self.peak_portfolio_value > 0:
            self.current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        return self.current_drawdown
    
    def _calculate_portfolio_weights(self,
                                   positions: Dict[str, float],
                                   prices: Dict[str, float],
                                   portfolio_value: float) -> Dict[str, float]:
        """Calculate portfolio weights."""
        weights = {}
        
        for symbol, quantity in positions.items():
            if symbol in prices and quantity != 0:
                market_value = abs(quantity * prices[symbol])
                weights[symbol] = market_value / portfolio_value if portfolio_value > 0 else 0
        
        return weights
    
    def _calculate_portfolio_returns(self,
                                   weights: Dict[str, float],
                                   returns_data: Dict[str, pd.Series]) -> pd.Series:
        """Calculate portfolio returns."""
        if not weights or not returns_data:
            return pd.Series(dtype=float)
        
        # Align all return series
        common_dates = None
        for symbol in weights.keys():
            if symbol in returns_data:
                if common_dates is None:
                    common_dates = returns_data[symbol].index
                else:
                    common_dates = common_dates.intersection(returns_data[symbol].index)
        
        if common_dates is None or len(common_dates) == 0:
            return pd.Series(dtype=float)
        
        # Calculate weighted returns
        portfolio_returns = pd.Series(0.0, index=common_dates)
        
        for symbol, weight in weights.items():
            if symbol in returns_data and weight > 0:
                aligned_returns = returns_data[symbol].reindex(common_dates, fill_value=0)
                portfolio_returns += weight * aligned_returns
        
        return portfolio_returns
    
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
    
    def _calculate_max_drawdown(self, returns: pd.Series, initial_value: float) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod() * initial_value
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        
        return excess_returns / volatility if volatility > 0 else 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        
        return excess_returns / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta relative to benchmark."""
        if len(returns) <= 1 or benchmark_returns is None or len(benchmark_returns) <= 1:
            return 1.0
        
        # Align returns
        aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        if len(aligned_data) <= 1:
            return 1.0
        
        covariance = aligned_data.cov().iloc[0, 1]
        benchmark_variance = aligned_data.iloc[:, 1].var()
        
        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0
    
    def _calculate_correlation_risk(self, weights: Dict[str, float], returns_data: Dict[str, pd.Series]) -> float:
        """Calculate correlation risk (average correlation)."""
        if len(weights) < 2:
            return 0.0
        
        symbols = list(weights.keys())
        correlations = []
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                if symbol1 in returns_data and symbol2 in returns_data:
                    corr = returns_data[symbol1].corr(returns_data[symbol2])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_concentration_risk(self, weights: Dict[str, float]) -> float:
        """Calculate concentration risk (Herfindahl index)."""
        if not weights:
            return 0.0
        
        return sum(w**2 for w in weights.values())
    
    def _calculate_leverage_ratio(self,
                                positions: Dict[str, float],
                                prices: Dict[str, float],
                                portfolio_value: float) -> float:
        """Calculate leverage ratio."""
        total_exposure = 0.0
        
        for symbol, quantity in positions.items():
            if symbol in prices:
                total_exposure += abs(quantity * prices[symbol])
        
        return total_exposure / portfolio_value if portfolio_value > 0 else 0.0
