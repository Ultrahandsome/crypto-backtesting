"""
Parameter optimization and strategy tuning tools.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import logging
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_time: float
    method: str
    objective_function: str


class ParameterOptimizer:
    """
    Comprehensive parameter optimization engine.
    """
    
    def __init__(self,
                 objective_function: str = 'sharpe_ratio',
                 n_jobs: int = -1,
                 random_state: int = 42):
        """
        Initialize parameter optimizer.
        
        Args:
            objective_function: Objective to optimize ('sharpe_ratio', 'calmar_ratio', 'total_return', etc.)
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random state for reproducibility
        """
        self.objective_function = objective_function
        self.n_jobs = n_jobs if n_jobs != -1 else None
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        
    def grid_search(self,
                   strategy_class,
                   data: Dict[str, pd.DataFrame],
                   param_grid: Dict[str, List],
                   backtest_func: Callable,
                   cv_splits: int = 1) -> OptimizationResult:
        """
        Perform grid search optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            data: Market data
            param_grid: Parameter grid to search
            backtest_func: Backtesting function
            cv_splits: Number of cross-validation splits
            
        Returns:
            OptimizationResult
        """
        start_time = datetime.now()
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        logger.info(f"Starting grid search with {len(param_combinations)} combinations")
        
        results = []
        
        if self.n_jobs == 1:
            # Sequential execution
            for i, params in enumerate(param_combinations):
                param_dict = dict(zip(param_names, params))
                result = self._evaluate_parameters(
                    strategy_class, data, param_dict, backtest_func, cv_splits
                )
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(param_combinations)} combinations")
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                
                for params in param_combinations:
                    param_dict = dict(zip(param_names, params))
                    future = executor.submit(
                        self._evaluate_parameters,
                        strategy_class, data, param_dict, backtest_func, cv_splits
                    )
                    futures.append(future)
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"Completed {i + 1}/{len(param_combinations)} combinations")
                    except Exception as e:
                        logger.error(f"Error in parameter evaluation: {e}")
        
        # Find best result
        valid_results = [r for r in results if r['score'] is not None and not np.isnan(r['score'])]
        
        if not valid_results:
            raise ValueError("No valid optimization results found")
        
        best_result = max(valid_results, key=lambda x: x['score'])
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=results,
            optimization_time=optimization_time,
            method='grid_search',
            objective_function=self.objective_function
        )
    
    def random_search(self,
                     strategy_class,
                     data: Dict[str, pd.DataFrame],
                     param_distributions: Dict[str, Tuple],
                     n_iter: int = 100,
                     backtest_func: Callable = None,
                     cv_splits: int = 1) -> OptimizationResult:
        """
        Perform random search optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            data: Market data
            param_distributions: Parameter distributions (name: (min, max) or (min, max, type))
            n_iter: Number of iterations
            backtest_func: Backtesting function
            cv_splits: Number of cross-validation splits
            
        Returns:
            OptimizationResult
        """
        start_time = datetime.now()
        
        logger.info(f"Starting random search with {n_iter} iterations")
        
        results = []
        
        for i in range(n_iter):
            # Sample random parameters
            params = {}
            for param_name, distribution in param_distributions.items():
                if len(distribution) == 2:
                    # Continuous parameter
                    min_val, max_val = distribution
                    params[param_name] = np.random.uniform(min_val, max_val)
                elif len(distribution) == 3:
                    # Discrete parameter
                    min_val, max_val, param_type = distribution
                    if param_type == 'int':
                        params[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        params[param_name] = np.random.uniform(min_val, max_val)
            
            result = self._evaluate_parameters(
                strategy_class, data, params, backtest_func, cv_splits
            )
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{n_iter} iterations")
        
        # Find best result
        valid_results = [r for r in results if r['score'] is not None and not np.isnan(r['score'])]
        
        if not valid_results:
            raise ValueError("No valid optimization results found")
        
        best_result = max(valid_results, key=lambda x: x['score'])
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=results,
            optimization_time=optimization_time,
            method='random_search',
            objective_function=self.objective_function
        )
    
    def bayesian_optimization(self,
                            strategy_class,
                            data: Dict[str, pd.DataFrame],
                            param_space: Dict[str, Tuple],
                            n_trials: int = 100,
                            backtest_func: Callable = None,
                            cv_splits: int = 1) -> OptimizationResult:
        """
        Perform Bayesian optimization using Optuna.

        Args:
            strategy_class: Strategy class to optimize
            data: Market data
            param_space: Parameter space (name: (min, max, type))
            n_trials: Number of trials
            backtest_func: Backtesting function
            cv_splits: Number of cross-validation splits

        Returns:
            OptimizationResult
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for Bayesian optimization. Install with: pip install optuna")

        start_time = datetime.now()

        logger.info(f"Starting Bayesian optimization with {n_trials} trials")

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, space_def in param_space.items():
                if len(space_def) == 3:
                    min_val, max_val, param_type = space_def
                    if param_type == 'int':
                        params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                    elif param_type == 'float':
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                    elif param_type == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, space_def[2])
                else:
                    # Assume float
                    min_val, max_val = space_def
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            
            result = self._evaluate_parameters(
                strategy_class, data, params, backtest_func, cv_splits
            )
            
            return result['score'] if result['score'] is not None else -np.inf
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials)
        
        # Collect all results
        all_results = []
        for trial in study.trials:
            result = {
                'params': trial.params,
                'score': trial.value if trial.value is not None else np.nan,
                'trial_number': trial.number
            }
            all_results.append(result)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            all_results=all_results,
            optimization_time=optimization_time,
            method='bayesian_optimization',
            objective_function=self.objective_function
        )
    
    def _evaluate_parameters(self,
                           strategy_class,
                           data: Dict[str, pd.DataFrame],
                           params: Dict[str, Any],
                           backtest_func: Callable,
                           cv_splits: int = 1) -> Dict[str, Any]:
        """
        Evaluate a single parameter combination.
        
        Args:
            strategy_class: Strategy class
            data: Market data
            params: Parameters to evaluate
            backtest_func: Backtesting function
            cv_splits: Number of cross-validation splits
            
        Returns:
            Evaluation result
        """
        try:
            if cv_splits == 1:
                # Single evaluation
                score = self._single_evaluation(strategy_class, data, params, backtest_func)
            else:
                # Cross-validation
                score = self._cross_validation_evaluation(
                    strategy_class, data, params, backtest_func, cv_splits
                )
            
            return {
                'params': params.copy(),
                'score': score,
                'cv_splits': cv_splits
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating parameters {params}: {e}")
            return {
                'params': params.copy(),
                'score': None,
                'error': str(e)
            }
    
    def _single_evaluation(self,
                          strategy_class,
                          data: Dict[str, pd.DataFrame],
                          params: Dict[str, Any],
                          backtest_func: Callable) -> float:
        """Perform single parameter evaluation."""
        # Initialize strategy with parameters
        strategy = strategy_class(**params)
        
        # Run backtest
        if backtest_func:
            results = backtest_func(strategy, data)
        else:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from backtesting import StrategyBacktester
            backtester = StrategyBacktester(strategy)
            results = backtester.run_backtest(data)
        
        # Extract objective value
        return self._extract_objective_value(results)
    
    def _cross_validation_evaluation(self,
                                   strategy_class,
                                   data: Dict[str, pd.DataFrame],
                                   params: Dict[str, Any],
                                   backtest_func: Callable,
                                   cv_splits: int) -> float:
        """Perform cross-validation evaluation."""
        scores = []
        
        # Split data for cross-validation
        data_splits = self._create_cv_splits(data, cv_splits)
        
        for train_data, test_data in data_splits:
            try:
                # Train on training data and test on test data
                strategy = strategy_class(**params)
                
                if backtest_func:
                    results = backtest_func(strategy, test_data)
                else:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                    from backtesting import StrategyBacktester
                    backtester = StrategyBacktester(strategy)
                    results = backtester.run_backtest(test_data)
                
                score = self._extract_objective_value(results)
                if score is not None and not np.isnan(score):
                    scores.append(score)
                    
            except Exception as e:
                logger.warning(f"Error in CV fold: {e}")
                continue
        
        return np.mean(scores) if scores else None
    
    def _extract_objective_value(self, results: Dict[str, Any]) -> Optional[float]:
        """Extract objective value from backtest results."""
        try:
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                
                if self.objective_function == 'sharpe_ratio':
                    return getattr(metrics, 'sharpe_ratio', None)
                elif self.objective_function == 'calmar_ratio':
                    return getattr(metrics, 'calmar_ratio', None)
                elif self.objective_function == 'sortino_ratio':
                    return getattr(metrics, 'sortino_ratio', None)
                elif self.objective_function == 'total_return':
                    return getattr(metrics, 'total_return', None)
                elif self.objective_function == 'win_rate':
                    return getattr(metrics, 'win_rate', None) / 100  # Convert to decimal
                else:
                    return getattr(metrics, self.objective_function, None)
            
            # Fallback to summary metrics
            elif 'summary' in results:
                summary = results['summary']
                
                if self.objective_function == 'total_return':
                    return summary.get('total_return', None)
                elif self.objective_function == 'win_rate':
                    return summary.get('win_rate', None)
                else:
                    return summary.get(self.objective_function, None)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting objective value: {e}")
            return None
    
    def _create_cv_splits(self,
                         data: Dict[str, pd.DataFrame],
                         cv_splits: int) -> List[Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]]:
        """Create cross-validation splits."""
        splits = []
        
        # Get common date range across all assets
        all_dates = None
        for symbol, df in data.items():
            if all_dates is None:
                all_dates = df.index
            else:
                all_dates = all_dates.intersection(df.index)
        
        if all_dates is None or len(all_dates) == 0:
            raise ValueError("No common dates found across assets")
        
        # Create time-based splits
        split_size = len(all_dates) // cv_splits
        
        for i in range(cv_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < cv_splits - 1 else len(all_dates)
            
            test_dates = all_dates[start_idx:end_idx]
            train_dates = all_dates[~all_dates.isin(test_dates)]
            
            train_data = {}
            test_data = {}
            
            for symbol, df in data.items():
                train_data[symbol] = df.loc[df.index.isin(train_dates)]
                test_data[symbol] = df.loc[df.index.isin(test_dates)]
            
            splits.append((train_data, test_data))
        
        return splits


class WalkForwardOptimizer:
    """
    Walk-forward optimization for robust parameter testing.
    """
    
    def __init__(self,
                 training_window: int = 252,
                 testing_window: int = 63,
                 step_size: int = 21):
        """
        Initialize walk-forward optimizer.
        
        Args:
            training_window: Training window size in days
            testing_window: Testing window size in days
            step_size: Step size for walk-forward
        """
        self.training_window = training_window
        self.testing_window = testing_window
        self.step_size = step_size
    
    def optimize(self,
                strategy_class,
                data: Dict[str, pd.DataFrame],
                param_space: Dict[str, List],
                optimizer: ParameterOptimizer) -> Dict[str, Any]:
        """
        Perform walk-forward optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            data: Market data
            param_space: Parameter space to search
            optimizer: Parameter optimizer instance
            
        Returns:
            Walk-forward optimization results
        """
        # Get common date range
        all_dates = None
        for symbol, df in data.items():
            if all_dates is None:
                all_dates = df.index
            else:
                all_dates = all_dates.intersection(df.index)
        
        all_dates = sorted(all_dates)
        
        results = []
        current_start = 0
        
        while current_start + self.training_window + self.testing_window <= len(all_dates):
            # Define windows
            train_end = current_start + self.training_window
            test_start = train_end
            test_end = test_start + self.testing_window
            
            train_dates = all_dates[current_start:train_end]
            test_dates = all_dates[test_start:test_end]
            
            # Create data splits
            train_data = {}
            test_data = {}
            
            for symbol, df in data.items():
                train_data[symbol] = df.loc[df.index.isin(train_dates)]
                test_data[symbol] = df.loc[df.index.isin(test_dates)]
            
            # Optimize on training data
            optimization_result = optimizer.grid_search(
                strategy_class, train_data, param_space, None
            )
            
            # Test on out-of-sample data
            best_strategy = strategy_class(**optimization_result.best_params)

            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from backtesting import StrategyBacktester
            backtester = StrategyBacktester(best_strategy)
            test_results = backtester.run_backtest(test_data)
            
            # Store results
            period_result = {
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'best_params': optimization_result.best_params,
                'in_sample_score': optimization_result.best_score,
                'out_of_sample_score': optimizer._extract_objective_value(test_results),
                'test_results': test_results
            }
            
            results.append(period_result)
            
            logger.info(f"Completed walk-forward period {len(results)}: "
                       f"{train_dates[0]} to {test_dates[-1]}")
            
            # Move to next period
            current_start += self.step_size
        
        return {
            'periods': results,
            'summary': self._calculate_walk_forward_summary(results)
        }
    
    def _calculate_walk_forward_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate walk-forward summary statistics."""
        in_sample_scores = [r['in_sample_score'] for r in results if r['in_sample_score'] is not None]
        out_of_sample_scores = [r['out_of_sample_score'] for r in results if r['out_of_sample_score'] is not None]
        
        return {
            'total_periods': len(results),
            'avg_in_sample_score': np.mean(in_sample_scores) if in_sample_scores else None,
            'avg_out_of_sample_score': np.mean(out_of_sample_scores) if out_of_sample_scores else None,
            'in_sample_std': np.std(in_sample_scores) if len(in_sample_scores) > 1 else None,
            'out_of_sample_std': np.std(out_of_sample_scores) if len(out_of_sample_scores) > 1 else None,
            'degradation': (np.mean(in_sample_scores) - np.mean(out_of_sample_scores)) if in_sample_scores and out_of_sample_scores else None
        }
