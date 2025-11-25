"""
Base experiment class defining the common interface for all experiments.

All experiments in the dMG framework inherit from BaseExperiment and implement
a standardized workflow with four phases:
    1. Setup: Initialize data, models, and resources
    2. Execute: Run the core experiment logic
    3. Evaluate: Compute performance metrics
    4. Report: Save results and generate visualizations
"""

import logging
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """
    Abstract base class for all experiments.

    This class defines the standard workflow and common utilities for
    hydrological modeling experiments. All concrete experiments must
    implement the abstract methods.

    Parameters
    ----------
    config : DictConfig
        Experiment configuration from Hydra

    Attributes
    ----------
    config : DictConfig
        Complete experiment configuration
    output_dir : Path
        Directory for saving results
    results : dict
        Storage for experiment results
    metrics : dict
        Storage for computed metrics
    start_time : float
        Experiment start timestamp
    """

    def __init__(self, config: DictConfig):
        """
        Initialize base experiment.

        Parameters
        ----------
        config : DictConfig
            Hydra configuration for the experiment
        """
        self.config = config
        self.output_dir = Path(config.get('output', {}).get('base_dir', './results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.results: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}

        # Timing
        self.start_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None

        log.info(f"Initialized experiment: {self.__class__.__name__}")
        log.info(f"Output directory: {self.output_dir}")

    # ========================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # ========================================================================

    @abstractmethod
    def setup(self) -> None:
        """
        Prepare experiment environment.

        This method should:
            - Load and preprocess data
            - Initialize models
            - Set up any required resources (GPU, parallel workers, etc.)
            - Validate configuration

        Raises
        ------
        ValueError
            If configuration is invalid or resources unavailable
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute core experiment logic.

        This is the main computational phase where models are trained,
        evaluated, and predictions are generated.

        Returns
        -------
        Dict[str, Any]
            Raw experimental results

        Examples
        --------
        For learning curves experiment, might return:
        {
            'HBV': {
                50: [replicate_results_1, replicate_results_2, ...],
                100: [...],
                ...
            },
            'LSTM': {...}
        }
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate experiment results.

        Compute aggregate statistics and performance metrics from raw results.

        Returns
        -------
        Dict[str, Any]
            Computed metrics and statistics

        Examples
        --------
        {
            'HBV': {
                'KGE': {'median': 0.75, 'q25': 0.70, 'q75': 0.80},
                'H_conditional': {...}
            },
            'LSTM': {...}
        }
        """
        raise NotImplementedError

    @abstractmethod
    def report(self) -> None:
        """
        Generate reports and visualizations.

        This method should:
            - Save results to disk (pickle, CSV, NetCDF)
            - Generate plots and figures
            - Create summary tables
            - Optionally generate LaTeX/HTML reports

        The output format and location are specified in self.config['output'].
        """
        raise NotImplementedError

    # ========================================================================
    # Main Execution Method
    # ========================================================================

    def run(self) -> Dict[str, Any]:
        """
        Execute complete experiment workflow.

        Runs all four phases in sequence:
        1. Setup
        2. Execute
        3. Evaluate
        4. Report

        Returns
        -------
        Dict[str, Any]
            Complete experiment results including:
                - 'results': Raw results from execute()
                - 'metrics': Computed metrics from evaluate()
                - 'elapsed_time': Total execution time in seconds
                - 'config': Experiment configuration

        Examples
        --------
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.load('conf/experiments/learning_curves.yaml')
        >>> experiment = LearningCurveExperiment(config)
        >>> results = experiment.run()
        >>> print(f"Experiment completed in {results['elapsed_time']:.2f} seconds")
        """
        log.info("="*70)
        log.info(f"STARTING EXPERIMENT: {self.__class__.__name__}")
        log.info("="*70)

        self.start_time = time.time()

        try:
            # Phase 1: Setup
            log.info("\n[1/4] SETUP: Initializing experiment...")
            self.setup()
            log.info("✓ Setup complete")

            # Phase 2: Execute
            log.info("\n[2/4] EXECUTE: Running experiment...")
            self.results = self.execute()
            log.info("✓ Execution complete")

            # Phase 3: Evaluate
            log.info("\n[3/4] EVALUATE: Computing metrics...")
            self.metrics = self.evaluate()
            log.info("✓ Evaluation complete")

            # Phase 4: Report
            log.info("\n[4/4] REPORT: Generating outputs...")
            self.report()
            log.info("✓ Reporting complete")

        except KeyboardInterrupt:
            log.warning("\n⚠ Experiment interrupted by user")
            self._save_checkpoint()
            raise

        except Exception as e:
            log.error(f"\n✗ Experiment failed: {e}")
            self._save_checkpoint()
            raise

        finally:
            self.elapsed_time = time.time() - self.start_time

        # Summary
        log.info("\n" + "="*70)
        log.info(f"EXPERIMENT COMPLETE: {self.__class__.__name__}")
        log.info(f"Total time: {self.elapsed_time / 60:.2f} minutes")
        log.info(f"Results saved to: {self.output_dir}")
        log.info("="*70)

        return {
            'results': self.results,
            'metrics': self.metrics,
            'elapsed_time': self.elapsed_time,
            'config': OmegaConf.to_container(self.config, resolve=True),
        }

    # ========================================================================
    # Common Utility Methods
    # ========================================================================

    def save_results(
        self,
        results: Dict[str, Any],
        filename: str,
        format: str = 'pickle',
    ) -> None:
        """
        Save results to disk.

        Parameters
        ----------
        results : Dict[str, Any]
            Results dictionary to save
        filename : str
            Output filename (without extension)
        format : str, optional
            Output format: 'pickle', 'json', or 'csv' (default: 'pickle')
        """
        output_path = self.output_dir / filename

        if format == 'pickle':
            output_path = output_path.with_suffix('.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info(f"Saved results to {output_path}")

        elif format == 'json':
            output_path = output_path.with_suffix('.json')
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            log.info(f"Saved results to {output_path}")

        elif format == 'csv':
            output_path = output_path.with_suffix('.csv')
            # Flatten results to DataFrame if possible
            try:
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
                log.info(f"Saved results to {output_path}")
            except Exception as e:
                log.warning(f"Could not save as CSV: {e}. Using pickle instead.")
                self.save_results(results, filename, format='pickle')

        else:
            raise ValueError(f"Unknown format: {format}")

    def load_results(
        self,
        filename: str,
        format: str = 'pickle',
    ) -> Dict[str, Any]:
        """
        Load previously saved results.

        Parameters
        ----------
        filename : str
            Filename to load (without extension)
        format : str, optional
            Format: 'pickle', 'json', or 'csv' (default: 'pickle')

        Returns
        -------
        Dict[str, Any]
            Loaded results
        """
        output_path = self.output_dir / filename

        if format == 'pickle':
            output_path = output_path.with_suffix('.pkl')
            with open(output_path, 'rb') as f:
                results = pickle.load(f)

        elif format == 'json':
            output_path = output_path.with_suffix('.json')
            import json
            with open(output_path, 'r') as f:
                results = json.load(f)

        elif format == 'csv':
            output_path = output_path.with_suffix('.csv')
            df = pd.read_csv(output_path)
            results = df.to_dict(orient='records')

        else:
            raise ValueError(f"Unknown format: {format}")

        log.info(f"Loaded results from {output_path}")
        return results

    def compute_replicate_statistics(
        self,
        replicate_values: List[float],
    ) -> Dict[str, float]:
        """
        Compute statistics across replicates.

        Parameters
        ----------
        replicate_values : List[float]
            List of metric values from different replicates

        Returns
        -------
        Dict[str, float]
            Statistics: median, mean, std, q25, q75, min, max
        """
        values = np.array(replicate_values)

        # Remove NaN/Inf
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return {
                'median': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'q25': np.nan,
                'q75': np.nan,
                'min': np.nan,
                'max': np.nan,
                'n_valid': 0,
            }

        return {
            'median': float(np.median(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'n_valid': int(len(values)),
        }

    def log_progress(
        self,
        current: int,
        total: int,
        message: str = "",
    ) -> None:
        """
        Log progress of long-running operations.

        Parameters
        ----------
        current : int
            Current iteration
        total : int
            Total iterations
        message : str, optional
            Additional message to log
        """
        percent = (current / total) * 100
        log.info(f"Progress: [{current}/{total}] ({percent:.1f}%) {message}")

    def _save_checkpoint(self) -> None:
        """Save checkpoint for interrupted experiments."""
        checkpoint_path = self.output_dir / 'checkpoint.pkl'
        checkpoint = {
            'results': self.results,
            'metrics': self.metrics,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'config': OmegaConf.to_container(self.config, resolve=True),
        }

        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            log.info(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            log.error(f"Failed to save checkpoint: {e}")

    def validate_config(self, required_keys: List[str]) -> None:
        """
        Validate that required configuration keys are present.

        Parameters
        ----------
        required_keys : List[str]
            List of required configuration keys (dot-separated paths)

        Raises
        ------
        ValueError
            If any required key is missing

        Examples
        --------
        >>> self.validate_config([
        >>>     'experiment.sample_sizes',
        >>>     'experiment.n_replicates',
        >>>     'models',
        >>> ])
        """
        missing_keys = []

        for key_path in required_keys:
            keys = key_path.split('.')
            current = self.config

            for key in keys:
                if not OmegaConf.select(current, key):
                    missing_keys.append(key_path)
                    break
                current = current[key]

        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {missing_keys}"
            )

    def get_device(self) -> torch.device:
        """
        Get PyTorch device based on configuration.

        Returns
        -------
        torch.device
            Device for computation ('cuda' or 'cpu')
        """
        device_name = self.config.get('compute', {}).get('device', 'cpu')

        if device_name == 'cuda' and not torch.cuda.is_available():
            log.warning("CUDA not available. Falling back to CPU.")
            device_name = 'cpu'

        device = torch.device(device_name)
        log.info(f"Using device: {device}")

        return device

    def seed_everything(self, seed: Optional[int] = None) -> None:
        """
        Set random seeds for reproducibility.

        Parameters
        ----------
        seed : int, optional
            Random seed. If None, uses config['random_seed']
        """
        if seed is None:
            seed = self.config.get('random_seed', 42)

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        log.info(f"Random seed set to: {seed}")


if __name__ == '__main__':
    # Example: Create a minimal concrete experiment for testing
    class MinimalExperiment(BaseExperiment):
        def setup(self):
            log.info("Setting up minimal experiment")

        def execute(self):
            log.info("Executing minimal experiment")
            return {'result': 42}

        def evaluate(self):
            log.info("Evaluating minimal experiment")
            return {'metric': 1.0}

        def report(self):
            log.info("Reporting minimal experiment")
            self.save_results(self.results, 'test_results')

    # Test
    from omegaconf import OmegaConf

    config = OmegaConf.create({
        'output': {'base_dir': './test_output'},
        'random_seed': 42,
    })

    experiment = MinimalExperiment(config)
    results = experiment.run()

    print("\n✓ BaseExperiment test passed")
    print(f"Results: {results}")
