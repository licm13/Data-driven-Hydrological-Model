"""
Learning curve evaluation for comparing model performance across different training data sizes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm

try:
    from ..utils.metrics import evaluate_model
except ImportError:
    from utils.metrics import evaluate_model


class LearningCurveEvaluator:
    """
    Evaluate and compare learning curves of different models
    """
    
    def __init__(self, train_sizes: List[int] = None):
        """
        Initialize evaluator
        
        Args:
            train_sizes: List of training data sizes to evaluate
        """
        if train_sizes is None:
            # Default training sizes
            self.train_sizes = [30, 60, 90, 120, 180, 240, 365, 
                               2*365, 3*365, 5*365]
        else:
            self.train_sizes = train_sizes
        
        self.results = {}
    
    def evaluate_model(self, model, model_name: str, X: np.ndarray, y: np.ndarray,
                      val_size: int = 365) -> pd.DataFrame:
        """
        Evaluate a single model across different training sizes
        
        Args:
            model: Model instance with fit() and predict() methods
            model_name: Name of the model
            X: Full feature dataset
            y: Full target dataset
            val_size: Size of validation set
            
        Returns:
            DataFrame with results for different training sizes
        """
        results = []
        
        for train_size in tqdm(self.train_sizes, desc=f"Evaluating {model_name}"):
            # Skip if not enough data
            if train_size + val_size > len(X):
                continue
            
            # Split data
            X_train = X[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_train = y[:train_size]
            y_val = y[train_size:train_size + val_size]
            
            try:
                # Prefer calibrate() for process-driven models, fit() for data-driven
                if hasattr(model, 'calibrate'):
                    # Process-driven model
                    model.calibrate(X_train, y_train, n_iterations=50)
                elif hasattr(model, 'fit'):
                    # Data-driven model
                    model.fit(X_train, y_train)
                else:
                    raise AttributeError("Model must have either fit() or calibrate() method")
                
                # Predict on validation set
                y_pred = model.predict(X_val)
                
                # Evaluate
                metrics = evaluate_model(y_val, y_pred)
                
                # Store results
                result = {
                    'model': model_name,
                    'train_size': train_size,
                    'val_size': val_size,
                    **metrics
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating {model_name} with train_size={train_size}: {e}")
                continue
        
        df_results = pd.DataFrame(results)
        self.results[model_name] = df_results
        
        return df_results
    
    def compare_models(self, models: Dict, X: np.ndarray, y: np.ndarray,
                      val_size: int = 365) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of {model_name: model_instance}
            X: Full feature dataset
            y: Full target dataset
            val_size: Size of validation set
            
        Returns:
            Combined DataFrame with all results
        """
        all_results = []
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name}")
            print(f"{'='*60}")
            
            df = self.evaluate_model(model, model_name, X, y, val_size)
            all_results.append(df)
        
        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_summary(self, metric: str = 'NSE') -> pd.DataFrame:
        """
        Get summary statistics for a specific metric
        
        Args:
            metric: Metric to summarize (default: NSE)
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for model_name, df in self.results.items():
            summary = {
                'model': model_name,
                f'{metric}_mean': df[metric].mean(),
                f'{metric}_std': df[metric].std(),
                f'{metric}_min': df[metric].min(),
                f'{metric}_max': df[metric].max(),
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def identify_crossover_point(self, model1_name: str, model2_name: str,
                                metric: str = 'NSE') -> Dict:
        """
        Identify the crossover point where one model starts outperforming another
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model
            metric: Metric to compare
            
        Returns:
            Dictionary with crossover information
        """
        if model1_name not in self.results or model2_name not in self.results:
            return None
        
        df1 = self.results[model1_name].sort_values('train_size')
        df2 = self.results[model2_name].sort_values('train_size')
        
        # Find common training sizes
        common_sizes = set(df1['train_size']).intersection(set(df2['train_size']))
        common_sizes = sorted(list(common_sizes))
        
        crossover_point = None
        
        for i, size in enumerate(common_sizes):
            val1 = df1[df1['train_size'] == size][metric].values[0]
            val2 = df2[df2['train_size'] == size][metric].values[0]
            
            if i > 0:
                prev_size = common_sizes[i-1]
                prev_val1 = df1[df1['train_size'] == prev_size][metric].values[0]
                prev_val2 = df2[df2['train_size'] == prev_size][metric].values[0]
                
                # Check if there's a crossover
                if (prev_val1 > prev_val2 and val1 < val2) or \
                   (prev_val1 < prev_val2 and val1 > val2):
                    crossover_point = {
                        'crossover_size': size,
                        f'{model1_name}_{metric}': val1,
                        f'{model2_name}_{metric}': val2,
                    }
                    break
        
        return crossover_point
    
    def analyze_learning_efficiency(self) -> pd.DataFrame:
        """
        Analyze learning efficiency: rate of improvement per additional training sample
        
        Returns:
            DataFrame with learning efficiency metrics
        """
        efficiency_data = []
        
        for model_name, df in self.results.items():
            # Skip if no results
            if len(df) == 0:
                continue
                
            df_sorted = df.sort_values('train_size')
            
            if len(df_sorted) < 2:
                continue
            
            # Calculate improvement rate
            train_sizes = df_sorted['train_size'].values
            nse_values = df_sorted['NSE'].values
            
            # Linear regression of NSE vs log(train_size)
            log_sizes = np.log(train_sizes)
            
            # Calculate slope (learning rate)
            slope = np.polyfit(log_sizes, nse_values, 1)[0]
            
            # Calculate efficiency: NSE gain per 100 samples
            if len(train_sizes) > 1:
                avg_improvement = (nse_values[-1] - nse_values[0]) / (train_sizes[-1] - train_sizes[0]) * 100
            else:
                avg_improvement = 0
            
            efficiency_data.append({
                'model': model_name,
                'learning_rate': slope,
                'avg_improvement_per_100_samples': avg_improvement,
                'initial_performance': nse_values[0],
                'final_performance': nse_values[-1],
                'total_improvement': nse_values[-1] - nse_values[0]
            })
        
        return pd.DataFrame(efficiency_data)
