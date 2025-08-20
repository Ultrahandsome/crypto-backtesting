"""
Portfolio-level risk management and monitoring.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from .risk_manager import RiskManager, RiskMetrics, PositionRisk

logger = logging.getLogger(__name__)


class PortfolioRiskMonitor:
    """
    Real-time portfolio risk monitoring system.
    """
    
    def __init__(self,
                 risk_manager: RiskManager,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize portfolio risk monitor.
        
        Args:
            risk_manager: Risk manager instance
            alert_thresholds: Custom alert thresholds
        """
        self.risk_manager = risk_manager
        self.alert_thresholds = alert_thresholds or {
            'var_warning': 0.015,  # 1.5% daily VaR warning
            'var_critical': 0.025,  # 2.5% daily VaR critical
            'drawdown_warning': 0.10,  # 10% drawdown warning
            'drawdown_critical': 0.15,  # 15% drawdown critical
            'concentration_warning': 0.30,  # 30% in single position warning
            'correlation_warning': 0.70,  # 70% average correlation warning
            'volatility_warning': 0.25,  # 25% annualized volatility warning
        }
        
        # Risk history tracking
        self.risk_history = []
        self.alert_history = []
        
    def monitor_portfolio(self,
                         positions: Dict[str, float],
                         prices: Dict[str, float],
                         returns_data: Dict[str, pd.Series],
                         portfolio_value: float,
                         timestamp: datetime,
                         benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Monitor portfolio risk in real-time.
        
        Args:
            positions: Current positions
            prices: Current prices
            returns_data: Historical returns data
            portfolio_value: Current portfolio value
            timestamp: Current timestamp
            benchmark_returns: Benchmark returns
            
        Returns:
            Risk monitoring report
        """
        # Assess current risk
        risk_metrics = self.risk_manager.assess_portfolio_risk(
            positions, prices, returns_data, portfolio_value, benchmark_returns
        )
        
        # Check risk limits
        violations = self.risk_manager.check_risk_limits(
            positions, prices, portfolio_value, returns_data
        )
        
        # Generate alerts
        alerts = self._generate_alerts(risk_metrics, violations)
        
        # Update drawdown tracking
        current_drawdown = self.risk_manager.update_drawdown(portfolio_value)
        
        # Create risk report
        risk_report = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'risk_metrics': risk_metrics,
            'violations': violations,
            'alerts': alerts,
            'current_drawdown': current_drawdown,
            'position_risks': self._assess_position_risks(
                positions, prices, returns_data, portfolio_value, benchmark_returns
            ),
            'risk_score': self._calculate_risk_score(risk_metrics),
            'recommendations': self._generate_recommendations(risk_metrics, violations)
        }
        
        # Store in history
        self.risk_history.append(risk_report)
        if alerts:
            self.alert_history.extend(alerts)
        
        # Log critical alerts
        for alert in alerts:
            if alert['severity'] in ['critical', 'extreme']:
                logger.warning(f"Risk Alert: {alert['message']}")
        
        return risk_report
    
    def _assess_position_risks(self,
                             positions: Dict[str, float],
                             prices: Dict[str, float],
                             returns_data: Dict[str, pd.Series],
                             portfolio_value: float,
                             benchmark_returns: Optional[pd.Series]) -> List[PositionRisk]:
        """Assess risk for all positions."""
        position_risks = []
        
        for symbol, quantity in positions.items():
            if symbol in prices and symbol in returns_data and quantity != 0:
                position_risk = self.risk_manager.assess_position_risk(
                    symbol, quantity, prices[symbol], returns_data[symbol],
                    portfolio_value, benchmark_returns
                )
                position_risks.append(position_risk)
        
        return position_risks
    
    def _generate_alerts(self,
                        risk_metrics: RiskMetrics,
                        violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate risk alerts based on thresholds."""
        alerts = []
        
        # VaR alerts
        if abs(risk_metrics.portfolio_var) > self.alert_thresholds['var_critical']:
            alerts.append({
                'type': 'var',
                'severity': 'critical',
                'message': f"Portfolio VaR ({abs(risk_metrics.portfolio_var)*100:.2f}%) exceeds critical threshold",
                'value': abs(risk_metrics.portfolio_var),
                'threshold': self.alert_thresholds['var_critical']
            })
        elif abs(risk_metrics.portfolio_var) > self.alert_thresholds['var_warning']:
            alerts.append({
                'type': 'var',
                'severity': 'warning',
                'message': f"Portfolio VaR ({abs(risk_metrics.portfolio_var)*100:.2f}%) exceeds warning threshold",
                'value': abs(risk_metrics.portfolio_var),
                'threshold': self.alert_thresholds['var_warning']
            })
        
        # Drawdown alerts
        if abs(risk_metrics.max_drawdown) > self.alert_thresholds['drawdown_critical']:
            alerts.append({
                'type': 'drawdown',
                'severity': 'critical',
                'message': f"Drawdown ({abs(risk_metrics.max_drawdown)*100:.2f}%) exceeds critical threshold",
                'value': abs(risk_metrics.max_drawdown),
                'threshold': self.alert_thresholds['drawdown_critical']
            })
        elif abs(risk_metrics.max_drawdown) > self.alert_thresholds['drawdown_warning']:
            alerts.append({
                'type': 'drawdown',
                'severity': 'warning',
                'message': f"Drawdown ({abs(risk_metrics.max_drawdown)*100:.2f}%) exceeds warning threshold",
                'value': abs(risk_metrics.max_drawdown),
                'threshold': self.alert_thresholds['drawdown_warning']
            })
        
        # Concentration alerts
        if risk_metrics.concentration_risk > self.alert_thresholds['concentration_warning']:
            alerts.append({
                'type': 'concentration',
                'severity': 'warning',
                'message': f"Portfolio concentration ({risk_metrics.concentration_risk:.2f}) is high",
                'value': risk_metrics.concentration_risk,
                'threshold': self.alert_thresholds['concentration_warning']
            })
        
        # Correlation alerts
        if risk_metrics.correlation_risk > self.alert_thresholds['correlation_warning']:
            alerts.append({
                'type': 'correlation',
                'severity': 'warning',
                'message': f"Average correlation ({risk_metrics.correlation_risk:.2f}) is high",
                'value': risk_metrics.correlation_risk,
                'threshold': self.alert_thresholds['correlation_warning']
            })
        
        # Volatility alerts
        if risk_metrics.volatility > self.alert_thresholds['volatility_warning']:
            alerts.append({
                'type': 'volatility',
                'severity': 'warning',
                'message': f"Portfolio volatility ({risk_metrics.volatility*100:.1f}%) is high",
                'value': risk_metrics.volatility,
                'threshold': self.alert_thresholds['volatility_warning']
            })
        
        # Add violation-based alerts
        for violation in violations:
            severity = violation.get('severity', 'medium')
            alerts.append({
                'type': violation['type'],
                'severity': severity,
                'message': f"Risk limit violation: {violation['type']}",
                'value': violation['current'],
                'threshold': violation['limit']
            })
        
        return alerts
    
    def _calculate_risk_score(self, risk_metrics: RiskMetrics) -> float:
        """
        Calculate overall risk score (0-100, higher = riskier).
        
        Args:
            risk_metrics: Risk metrics
            
        Returns:
            Risk score
        """
        score = 0.0
        
        # VaR component (0-30 points)
        var_score = min(30, abs(risk_metrics.portfolio_var) / 0.05 * 30)
        score += var_score
        
        # Volatility component (0-25 points)
        vol_score = min(25, risk_metrics.volatility / 0.5 * 25)
        score += vol_score
        
        # Drawdown component (0-25 points)
        dd_score = min(25, abs(risk_metrics.max_drawdown) / 0.3 * 25)
        score += dd_score
        
        # Concentration component (0-10 points)
        conc_score = min(10, risk_metrics.concentration_risk / 0.5 * 10)
        score += conc_score
        
        # Correlation component (0-10 points)
        corr_score = min(10, risk_metrics.correlation_risk / 1.0 * 10)
        score += corr_score
        
        return min(100, score)
    
    def _generate_recommendations(self,
                                risk_metrics: RiskMetrics,
                                violations: List[Dict[str, Any]]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        # High VaR recommendations
        if abs(risk_metrics.portfolio_var) > 0.02:
            recommendations.append("Consider reducing position sizes to lower portfolio VaR")
        
        # High concentration recommendations
        if risk_metrics.concentration_risk > 0.3:
            recommendations.append("Diversify portfolio to reduce concentration risk")
        
        # High correlation recommendations
        if risk_metrics.correlation_risk > 0.7:
            recommendations.append("Add uncorrelated assets to reduce correlation risk")
        
        # High volatility recommendations
        if risk_metrics.volatility > 0.25:
            recommendations.append("Consider adding lower volatility assets")
        
        # Drawdown recommendations
        if abs(risk_metrics.max_drawdown) > 0.1:
            recommendations.append("Review stop-loss levels and risk management rules")
        
        # Low Sharpe ratio recommendations
        if risk_metrics.sharpe_ratio < 0.5:
            recommendations.append("Review strategy performance and consider adjustments")
        
        # Violation-specific recommendations
        for violation in violations:
            if violation['type'] == 'position_size':
                recommendations.append(f"Reduce position size in {violation.get('symbol', 'asset')}")
            elif violation['type'] == 'leverage':
                recommendations.append("Reduce leverage to comply with risk limits")
            elif violation['type'] == 'max_drawdown':
                recommendations.append("Consider emergency risk reduction measures")
        
        return recommendations
    
    def get_risk_summary(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Get risk summary for the last N days.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Risk summary
        """
        if not self.risk_history:
            return {}
        
        # Filter recent history
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_history = []

        for report in self.risk_history:
            report_time = report['timestamp']
            # Handle timezone-aware timestamps
            if hasattr(report_time, 'tz_localize'):
                if report_time.tz is not None:
                    cutoff_date = cutoff_date.replace(tzinfo=report_time.tz)
                else:
                    report_time = report_time.tz_localize(None)

            if report_time >= cutoff_date:
                recent_history.append(report)
        
        if not recent_history:
            return {}
        
        # Calculate summary statistics
        risk_scores = [report['risk_score'] for report in recent_history]
        drawdowns = [report['current_drawdown'] for report in recent_history]
        
        # Count alerts by severity
        alert_counts = {'warning': 0, 'critical': 0, 'extreme': 0}
        for report in recent_history:
            for alert in report['alerts']:
                severity = alert['severity']
                if severity in alert_counts:
                    alert_counts[severity] += 1
        
        return {
            'period_days': lookback_days,
            'avg_risk_score': np.mean(risk_scores),
            'max_risk_score': np.max(risk_scores),
            'avg_drawdown': np.mean(drawdowns),
            'max_drawdown': np.max(drawdowns),
            'alert_counts': alert_counts,
            'total_alerts': sum(alert_counts.values()),
            'latest_risk_score': risk_scores[-1] if risk_scores else 0,
            'risk_trend': 'increasing' if len(risk_scores) > 1 and risk_scores[-1] > risk_scores[0] else 'stable'
        }
    
    def export_risk_report(self, filepath: str, format: str = 'json'):
        """
        Export risk monitoring data.
        
        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            import json
            
            # Prepare data for JSON export (handle non-serializable objects)
            export_data = {
                'risk_history': [],
                'alert_history': self.alert_history
            }
            
            for report in self.risk_history:
                export_report = {
                    'timestamp': report['timestamp'].isoformat(),
                    'portfolio_value': report['portfolio_value'],
                    'risk_score': report['risk_score'],
                    'current_drawdown': report['current_drawdown'],
                    'alerts_count': len(report['alerts']),
                    'violations_count': len(report['violations'])
                }
                export_data['risk_history'].append(export_report)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        elif format == 'csv':
            # Export as CSV
            df_data = []
            for report in self.risk_history:
                df_data.append({
                    'timestamp': report['timestamp'],
                    'portfolio_value': report['portfolio_value'],
                    'risk_score': report['risk_score'],
                    'current_drawdown': report['current_drawdown'],
                    'alerts_count': len(report['alerts']),
                    'violations_count': len(report['violations'])
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False)
        
        logger.info(f"Risk report exported to {filepath}")


class DynamicRiskAdjustment:
    """
    Dynamic risk adjustment based on market conditions.
    """
    
    def __init__(self, base_risk_manager: RiskManager):
        """
        Initialize dynamic risk adjustment.
        
        Args:
            base_risk_manager: Base risk manager
        """
        self.base_risk_manager = base_risk_manager
        self.market_regime = 'normal'  # normal, volatile, crisis
        
    def adjust_risk_parameters(self,
                             market_volatility: float,
                             correlation_spike: bool = False,
                             liquidity_stress: bool = False) -> RiskManager:
        """
        Adjust risk parameters based on market conditions.
        
        Args:
            market_volatility: Current market volatility
            correlation_spike: Whether correlations have spiked
            liquidity_stress: Whether there's liquidity stress
            
        Returns:
            Adjusted risk manager
        """
        # Determine market regime
        if market_volatility > 0.4 or correlation_spike or liquidity_stress:
            self.market_regime = 'crisis'
        elif market_volatility > 0.25:
            self.market_regime = 'volatile'
        else:
            self.market_regime = 'normal'
        
        # Adjust parameters based on regime
        adjusted_manager = RiskManager(
            max_portfolio_risk=self._adjust_portfolio_risk(),
            max_position_size=self._adjust_position_size(),
            max_drawdown=self._adjust_max_drawdown(),
            max_leverage=self._adjust_leverage(),
            var_confidence=self.base_risk_manager.var_confidence,
            lookback_period=self._adjust_lookback_period()
        )
        
        return adjusted_manager
    
    def _adjust_portfolio_risk(self) -> float:
        """Adjust portfolio risk limit based on regime."""
        base_risk = self.base_risk_manager.max_portfolio_risk
        
        if self.market_regime == 'crisis':
            return base_risk * 0.5  # Halve risk in crisis
        elif self.market_regime == 'volatile':
            return base_risk * 0.75  # Reduce risk in volatile markets
        else:
            return base_risk
    
    def _adjust_position_size(self) -> float:
        """Adjust position size limit based on regime."""
        base_size = self.base_risk_manager.max_position_size
        
        if self.market_regime == 'crisis':
            return base_size * 0.6  # Smaller positions in crisis
        elif self.market_regime == 'volatile':
            return base_size * 0.8  # Smaller positions in volatile markets
        else:
            return base_size
    
    def _adjust_max_drawdown(self) -> float:
        """Adjust max drawdown based on regime."""
        base_dd = self.base_risk_manager.max_drawdown
        
        if self.market_regime == 'crisis':
            return base_dd * 0.7  # Tighter drawdown control in crisis
        else:
            return base_dd
    
    def _adjust_leverage(self) -> float:
        """Adjust leverage based on regime."""
        base_leverage = self.base_risk_manager.max_leverage
        
        if self.market_regime == 'crisis':
            return min(base_leverage, 0.5)  # Reduce leverage in crisis
        elif self.market_regime == 'volatile':
            return min(base_leverage, 0.8)  # Reduce leverage in volatile markets
        else:
            return base_leverage
    
    def _adjust_lookback_period(self) -> int:
        """Adjust lookback period based on regime."""
        base_period = self.base_risk_manager.lookback_period
        
        if self.market_regime == 'crisis':
            return min(base_period, 60)  # Shorter lookback in crisis
        elif self.market_regime == 'volatile':
            return min(base_period, 120)  # Shorter lookback in volatile markets
        else:
            return base_period
