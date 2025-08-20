"""
Strategy tuning and optimization utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from .parameter_optimizer import ParameterOptimizer, OptimizationResult, WalkForwardOptimizer
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class StrategyTuner:
    """
    High-level strategy tuning interface.
    """
    
    def __init__(self,
                 strategy_class,
                 data: Dict[str, pd.DataFrame],
                 objective_function: str = 'sharpe_ratio'):
        """
        Initialize strategy tuner.
        
        Args:
            strategy_class: Strategy class to tune
            data: Market data
            objective_function: Objective function to optimize
        """
        self.strategy_class = strategy_class
        self.data = data
        self.objective_function = objective_function
        self.optimizer = ParameterOptimizer(objective_function=objective_function)
        
        # Results storage
        self.optimization_results = {}
        self.best_params = None
        self.best_score = None
    
    def quick_tune(self,
                  param_ranges: Dict[str, Tuple[int, int]],
                  method: str = 'grid',
                  n_trials: int = 50) -> OptimizationResult:
        """
        Quick parameter tuning with reasonable defaults.
        
        Args:
            param_ranges: Parameter ranges to search
            method: Optimization method ('grid', 'random', 'bayesian')
            n_trials: Number of trials for random/bayesian methods
            
        Returns:
            OptimizationResult
        """
        logger.info(f"Starting quick tune with {method} method")
        
        if method == 'grid':
            # Create grid with 3-5 points per parameter
            param_grid = {}
            for param, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer parameter
                    n_points = min(5, max_val - min_val + 1)
                    param_grid[param] = list(np.linspace(min_val, max_val, n_points, dtype=int))
                else:
                    # Float parameter
                    param_grid[param] = list(np.linspace(min_val, max_val, 4))
            
            result = self.optimizer.grid_search(
                self.strategy_class, self.data, param_grid, None
            )
            
        elif method == 'random':
            # Convert to distributions
            param_distributions = {}
            for param, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    param_distributions[param] = (min_val, max_val, 'int')
                else:
                    param_distributions[param] = (min_val, max_val)
            
            result = self.optimizer.random_search(
                self.strategy_class, self.data, param_distributions, n_trials, None
            )
            
        elif method == 'bayesian':
            # Convert to space
            param_space = {}
            for param, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    param_space[param] = (min_val, max_val, 'int')
                else:
                    param_space[param] = (min_val, max_val, 'float')
            
            result = self.optimizer.bayesian_optimization(
                self.strategy_class, self.data, param_space, n_trials, None
            )
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Store results
        self.optimization_results[method] = result
        self.best_params = result.best_params
        self.best_score = result.best_score
        
        logger.info(f"Quick tune completed. Best score: {result.best_score:.4f}")
        logger.info(f"Best parameters: {result.best_params}")
        
        return result
    
    def comprehensive_tune(self,
                          param_ranges: Dict[str, Tuple[int, int]],
                          methods: List[str] = ['grid', 'random', 'bayesian'],
                          n_trials: int = 100) -> Dict[str, OptimizationResult]:
        """
        Comprehensive tuning using multiple methods.
        
        Args:
            param_ranges: Parameter ranges to search
            methods: List of optimization methods to use
            n_trials: Number of trials for random/bayesian methods
            
        Returns:
            Dictionary of optimization results
        """
        logger.info("Starting comprehensive tuning")
        
        results = {}
        
        for method in methods:
            logger.info(f"Running {method} optimization...")
            
            try:
                if method == 'grid':
                    # Smaller grid for comprehensive search
                    param_grid = {}
                    for param, (min_val, max_val) in param_ranges.items():
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            n_points = min(3, max_val - min_val + 1)
                            param_grid[param] = list(np.linspace(min_val, max_val, n_points, dtype=int))
                        else:
                            param_grid[param] = list(np.linspace(min_val, max_val, 3))
                    
                    result = self.optimizer.grid_search(
                        self.strategy_class, self.data, param_grid, None
                    )
                    
                elif method == 'random':
                    param_distributions = {}
                    for param, (min_val, max_val) in param_ranges.items():
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            param_distributions[param] = (min_val, max_val, 'int')
                        else:
                            param_distributions[param] = (min_val, max_val)
                    
                    result = self.optimizer.random_search(
                        self.strategy_class, self.data, param_distributions, n_trials, None
                    )
                    
                elif method == 'bayesian':
                    param_space = {}
                    for param, (min_val, max_val) in param_ranges.items():
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            param_space[param] = (min_val, max_val, 'int')
                        else:
                            param_space[param] = (min_val, max_val, 'float')
                    
                    result = self.optimizer.bayesian_optimization(
                        self.strategy_class, self.data, param_space, n_trials, None
                    )
                
                results[method] = result
                self.optimization_results[method] = result
                
                logger.info(f"{method} optimization completed. Score: {result.best_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error in {method} optimization: {e}")
                continue
        
        # Find overall best
        if results:
            best_method = max(results.keys(), key=lambda k: results[k].best_score)
            self.best_params = results[best_method].best_params
            self.best_score = results[best_method].best_score
            
            logger.info(f"Comprehensive tuning completed. Best method: {best_method}")
            logger.info(f"Best score: {self.best_score:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
        
        return results
    
    def walk_forward_optimization(self,
                                param_ranges: Dict[str, Tuple[int, int]],
                                training_window: int = 252,
                                testing_window: int = 63,
                                step_size: int = 21) -> Dict[str, Any]:
        """
        Perform walk-forward optimization.
        
        Args:
            param_ranges: Parameter ranges to search
            training_window: Training window size in days
            testing_window: Testing window size in days
            step_size: Step size for walk-forward
            
        Returns:
            Walk-forward optimization results
        """
        logger.info("Starting walk-forward optimization")
        
        # Create parameter grid
        param_grid = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                n_points = min(3, max_val - min_val + 1)
                param_grid[param] = list(np.linspace(min_val, max_val, n_points, dtype=int))
            else:
                param_grid[param] = list(np.linspace(min_val, max_val, 3))
        
        # Initialize walk-forward optimizer
        wf_optimizer = WalkForwardOptimizer(
            training_window=training_window,
            testing_window=testing_window,
            step_size=step_size
        )
        
        # Run walk-forward optimization
        results = wf_optimizer.optimize(
            self.strategy_class, self.data, param_grid, self.optimizer
        )
        
        logger.info(f"Walk-forward optimization completed. {results['summary']['total_periods']} periods")
        logger.info(f"Average out-of-sample score: {results['summary']['avg_out_of_sample_score']:.4f}")
        
        return results
    
    def sensitivity_analysis(self,
                           base_params: Dict[str, Any],
                           param_ranges: Dict[str, Tuple[float, float]],
                           n_points: int = 10) -> Dict[str, Any]:
        """
        Perform parameter sensitivity analysis.
        
        Args:
            base_params: Base parameter set
            param_ranges: Ranges for sensitivity analysis
            n_points: Number of points to test for each parameter
            
        Returns:
            Sensitivity analysis results
        """
        logger.info("Starting sensitivity analysis")
        
        sensitivity_results = {}
        
        for param_name, (min_val, max_val) in param_ranges.items():
            if param_name not in base_params:
                logger.warning(f"Parameter {param_name} not in base params, skipping")
                continue
            
            param_values = np.linspace(min_val, max_val, n_points)
            scores = []
            
            for value in param_values:
                # Create parameter set with modified value
                test_params = base_params.copy()
                test_params[param_name] = value
                
                # Evaluate
                try:
                    result = self.optimizer._evaluate_parameters(
                        self.strategy_class, self.data, test_params, None
                    )
                    scores.append(result['score'])
                except Exception as e:
                    logger.warning(f"Error evaluating {param_name}={value}: {e}")
                    scores.append(None)
            
            sensitivity_results[param_name] = {
                'values': param_values.tolist(),
                'scores': scores,
                'base_value': base_params[param_name]
            }
        
        logger.info("Sensitivity analysis completed")
        return sensitivity_results
    
    def plot_optimization_results(self,
                                results: OptimizationResult,
                                save_path: Optional[str] = None) -> None:
        """
        Plot optimization results.
        
        Args:
            results: Optimization results
            save_path: Path to save plot
        """
        if not results.all_results:
            logger.warning("No results to plot")
            return
        
        # Extract data
        scores = [r['score'] for r in results.all_results if r['score'] is not None]
        
        if not scores:
            logger.warning("No valid scores to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Optimization Results - {results.method}', fontsize=16)
        
        # Score distribution
        axes[0, 0].hist(scores, bins=30, alpha=0.7, color='blue')
        axes[0, 0].axvline(results.best_score, color='red', linestyle='--', 
                          label=f'Best: {results.best_score:.4f}')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].set_xlabel(f'{self.objective_function}')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Score over iterations
        axes[0, 1].plot(scores, alpha=0.7, color='blue')
        axes[0, 1].axhline(results.best_score, color='red', linestyle='--', 
                          label=f'Best: {results.best_score:.4f}')
        axes[0, 1].set_title('Score Over Iterations')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel(f'{self.objective_function}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Parameter correlation (if enough parameters)
        param_names = list(results.best_params.keys())
        if len(param_names) >= 2:
            # Create parameter matrix
            param_data = []
            for result in results.all_results:
                if result['score'] is not None:
                    row = [result['params'].get(param, 0) for param in param_names]
                    row.append(result['score'])
                    param_data.append(row)
            
            if param_data:
                param_df = pd.DataFrame(param_data, columns=param_names + ['score'])
                corr_matrix = param_df.corr()
                
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           ax=axes[1, 0], cbar_kws={'label': 'Correlation'})
                axes[1, 0].set_title('Parameter Correlation Matrix')
        else:
            axes[1, 0].text(0.5, 0.5, 'Not enough parameters\nfor correlation analysis',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Parameter Correlation Matrix')
        
        # Best parameters
        axes[1, 1].axis('off')
        best_params_text = "Best Parameters:\n\n"
        for param, value in results.best_params.items():
            if isinstance(value, float):
                best_params_text += f"{param}: {value:.4f}\n"
            else:
                best_params_text += f"{param}: {value}\n"
        
        best_params_text += f"\nBest Score: {results.best_score:.4f}"
        best_params_text += f"\nOptimization Time: {results.optimization_time:.1f}s"
        best_params_text += f"\nTotal Evaluations: {len(results.all_results)}"
        
        axes[1, 1].text(0.1, 0.9, best_params_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Optimization Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization plot saved to {save_path}")
        
        plt.show()
    
    def plot_sensitivity_analysis(self,
                                sensitivity_results: Dict[str, Any],
                                save_path: Optional[str] = None) -> None:
        """
        Plot sensitivity analysis results.
        
        Args:
            sensitivity_results: Sensitivity analysis results
            save_path: Path to save plot
        """
        n_params = len(sensitivity_results)
        if n_params == 0:
            logger.warning("No sensitivity results to plot")
            return
        
        # Calculate subplot layout
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle('Parameter Sensitivity Analysis', fontsize=16)
        
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_params == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (param_name, data) in enumerate(sensitivity_results.items()):
            ax = axes[i] if n_params > 1 else axes[0]
            
            values = data['values']
            scores = [s for s in data['scores'] if s is not None]
            valid_values = [v for v, s in zip(values, data['scores']) if s is not None]
            
            if scores and valid_values:
                ax.plot(valid_values, scores, 'b-o', alpha=0.7)
                ax.axvline(data['base_value'], color='red', linestyle='--', 
                          label=f"Base: {data['base_value']}")
                ax.set_title(f'{param_name} Sensitivity')
                ax.set_xlabel(param_name)
                ax.set_ylabel(self.objective_function)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{param_name} Sensitivity')
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sensitivity analysis plot saved to {save_path}")
        
        plt.show()
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results."""
        if not self.optimization_results:
            return {"message": "No optimization results available"}
        
        summary = {
            "best_overall_score": self.best_score,
            "best_overall_params": self.best_params,
            "methods_tested": list(self.optimization_results.keys()),
            "method_comparison": {}
        }
        
        for method, result in self.optimization_results.items():
            summary["method_comparison"][method] = {
                "best_score": result.best_score,
                "optimization_time": result.optimization_time,
                "total_evaluations": len(result.all_results)
            }
        
        return summary
