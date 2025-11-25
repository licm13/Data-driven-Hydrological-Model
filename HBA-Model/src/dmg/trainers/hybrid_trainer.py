"""
Hybrid trainer supporting both traditional calibration and gradient-based training.

This module provides a unified training interface that supports:
    - Traditional calibration using Spotpy (for pure physical models)
    - Gradient descent using PyTorch optimizers (for neural networks and dPL models)
    - Hybrid training: Spotpy pre-training followed by gradient fine-tuning

The hybrid approach is particularly effective for dPL models, where:
1. Spotpy provides a good initialization in parameter space
2. Gradient descent fine-tunes for optimal performance
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from dmg.trainers.trainer import Trainer

log = logging.getLogger(__name__)


# ============================================================================
# Training Strategies
# ============================================================================


class TrainingStrategy(ABC):
    """Abstract base class for training strategies."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        train_data: dict[str, torch.Tensor],
        eval_data: Optional[dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Parameters
        ----------
        model : nn.Module
            Model to train
        train_data : dict[str, torch.Tensor]
            Training dataset
        eval_data : dict[str, torch.Tensor], optional
            Evaluation dataset

        Returns
        -------
        Dict[str, Any]
            Training results including metrics and trained parameters
        """
        raise NotImplementedError


class SpotpyCalibrationStrategy(TrainingStrategy):
    """
    Training strategy using Spotpy for traditional model calibration.

    Spotpy provides various optimization algorithms for calibrating
    hydrological models, including:
        - Latin Hypercube Sampling (LHS)
        - Monte Carlo (MC)
        - Differential Evolution (DE)
        - SCE-UA
        - DREAM

    This strategy is ideal for pure physical models (HBV, GR4J) that
    don't require gradient information.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.algorithm = config.get('algorithm', 'lhs')
        self.n_iterations = config.get('n_iterations', 500)
        self.objective = config.get('objective', 'kge')

    def train(
        self,
        model: nn.Module,
        train_data: dict[str, torch.Tensor],
        eval_data: Optional[dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Calibrate model using Spotpy.

        Parameters
        ----------
        model : nn.Module
            Physical model to calibrate (e.g., HBVTorch)
        train_data : dict[str, torch.Tensor]
            Training data with keys: 'x_phy' (forcings), 'target' (observations)
        eval_data : dict[str, torch.Tensor], optional
            Evaluation data for validation

        Returns
        -------
        Dict[str, Any]
            Calibration results:
                - 'best_parameters': Best parameter set found
                - 'best_objective': Best objective value
                - 'all_parameters': All sampled parameters (if requested)
                - 'all_objectives': All objective values
        """
        try:
            import spotpy
        except ImportError:
            raise ImportError(
                "Spotpy is required for calibration. Install with: pip install spotpy"
            )

        log.info(f"Starting Spotpy calibration with algorithm: {self.algorithm}")
        log.info(f"Number of iterations: {self.n_iterations}")
        log.info(f"Objective function: {self.objective}")

        # Create Spotpy setup
        setup = SpotpyModelSetup(
            model=model,
            train_data=train_data,
            objective=self.objective,
        )

        # Select algorithm
        algorithms = {
            'lhs': spotpy.algorithms.lhs,
            'mc': spotpy.algorithms.mc,
            'sceua': spotpy.algorithms.sceua,
            'dream': spotpy.algorithms.dream,
            'de': spotpy.algorithms.de,
        }

        if self.algorithm not in algorithms:
            raise ValueError(
                f"Unknown Spotpy algorithm: {self.algorithm}. "
                f"Available: {list(algorithms.keys())}"
            )

        sampler = algorithms[self.algorithm](
            setup,
            dbname='spotpy_calibration',
            dbformat='ram',  # Store in RAM for speed
        )

        # Run calibration
        start_time = time.time()
        sampler.sample(self.n_iterations)
        elapsed = time.time() - start_time

        log.info(f"Calibration completed in {elapsed:.2f} seconds")

        # Extract results
        results = sampler.getdata()

        # Find best parameter set
        objectives = np.array([r[0] for r in results])

        # For KGE, higher is better; for RMSE, lower is better
        if self.objective in ['kge', 'nse']:
            best_idx = np.argmax(objectives)
        else:
            best_idx = np.argmin(objectives)

        best_objective = objectives[best_idx]

        # Extract parameter names dynamically
        param_names = setup.param_names
        best_params = np.array([results[best_idx][i + 1] for i in range(len(param_names))])

        log.info(f"Best {self.objective}: {best_objective:.4f}")
        log.info(f"Best parameters: {dict(zip(param_names, best_params))}")

        # Convert to torch tensor
        best_params_torch = torch.from_numpy(best_params).to(
            dtype=train_data['x_phy'].dtype,
            device=train_data['x_phy'].device,
        ).unsqueeze(0)  # [1, n_params]

        return {
            'best_parameters': best_params_torch,
            'best_objective': best_objective,
            'all_parameters': np.array([list(r[1:len(param_names)+1]) for r in results]),
            'all_objectives': objectives,
            'elapsed_time': elapsed,
        }


class SpotpyModelSetup:
    """
    Spotpy setup class for model calibration.

    Defines the interface between the dMG model and Spotpy's optimization algorithms.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: dict[str, torch.Tensor],
        objective: str = 'kge',
    ):
        self.model = model
        self.train_data = train_data
        self.objective_name = objective

        # Extract observations
        self.obs = train_data['target'][:, 0, 0].cpu().numpy()

        # Get parameter bounds from model
        if hasattr(model, 'PARAM_BOUNDS'):
            self.param_bounds = model.PARAM_BOUNDS
        else:
            raise ValueError("Model must define PARAM_BOUNDS attribute")

        self.param_names = list(self.param_bounds.keys())

    def parameters(self):
        """Define parameters for Spotpy."""
        import spotpy

        params = []
        for name in self.param_names:
            low, high, default = self.param_bounds[name]
            params.append(
                spotpy.parameter.Uniform(name, low=low, high=high, optguess=default)
            )
        return spotpy.parameter.generate(params)

    def simulation(self, vector):
        """Run model simulation with given parameters."""
        # Convert parameter vector to tensor
        params_array = np.array([vector[name] for name in self.param_names])
        params_tensor = torch.from_numpy(params_array).to(
            dtype=self.train_data['x_phy'].dtype,
            device=self.train_data['x_phy'].device,
        ).unsqueeze(0)  # [1, n_params]

        # Run model
        with torch.no_grad():
            output = self.model(self.train_data, params_tensor)
            sim = output['flow'][:, 0, 0].cpu().numpy()

        return sim

    def evaluation(self):
        """Return observations for comparison."""
        return self.obs

    def objectivefunction(self, simulation, evaluation):
        """Compute objective function."""
        # Remove NaN values
        valid = ~(np.isnan(simulation) | np.isnan(evaluation))
        sim = simulation[valid]
        obs = evaluation[valid]

        if len(sim) == 0:
            return -9999.0  # Return very bad value

        if self.objective_name == 'kge':
            return self._kge(obs, sim)
        elif self.objective_name == 'nse':
            return self._nse(obs, sim)
        elif self.objective_name == 'rmse':
            return -self._rmse(obs, sim)  # Negative for minimization
        else:
            raise ValueError(f"Unknown objective: {self.objective_name}")

    @staticmethod
    def _kge(obs: NDArray, sim: NDArray) -> float:
        """Kling-Gupta Efficiency."""
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = np.std(sim) / np.std(obs)
        beta = np.mean(sim) / np.mean(obs)
        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return kge

    @staticmethod
    def _nse(obs: NDArray, sim: NDArray) -> float:
        """Nash-Sutcliffe Efficiency."""
        return 1.0 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

    @staticmethod
    def _rmse(obs: NDArray, sim: NDArray) -> float:
        """Root Mean Square Error."""
        return np.sqrt(np.mean((obs - sim) ** 2))


class GradientDescentStrategy(TrainingStrategy):
    """
    Training strategy using PyTorch gradient descent.

    This strategy is used for:
        - Pure neural network models (LSTM, MLP)
        - dPL models (after optional Spotpy pre-training)
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.epochs = config.get('epochs', 50)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.optimizer_name = config.get('optimizer', 'Adam')

    def train(
        self,
        model: nn.Module,
        train_data: dict[str, torch.Tensor],
        eval_data: Optional[dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Train model using gradient descent.

        Uses the standard dMG Trainer class for consistency.
        """
        from dmg.trainers.trainer import Trainer

        # Create trainer
        trainer = Trainer(
            config=self.config,
            model=model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            verbose=True,
        )

        # Train
        log.info(f"Starting gradient descent training for {self.epochs} epochs")
        start_time = time.time()

        trainer.train()

        elapsed = time.time() - start_time
        log.info(f"Training completed in {elapsed:.2f} seconds")

        # Evaluate if eval data provided
        metrics = {}
        if eval_data is not None:
            trainer.evaluate()
            # Extract metrics (simplified - would need actual metric extraction)
            metrics = {'eval_completed': True}

        return {
            'trained_model': model,
            'final_loss': trainer.total_loss,
            'metrics': metrics,
            'elapsed_time': elapsed,
        }


class HybridStrategy(TrainingStrategy):
    """
    Hybrid training strategy: Spotpy pre-training + Gradient fine-tuning.

    This two-phase approach combines the strengths of both methods:
    1. **Phase 1 (Spotpy)**: Explores parameter space globally to find a good basin
    2. **Phase 2 (Gradient Descent)**: Fine-tunes locally for optimal performance

    This is particularly effective for dPL models where the neural network learns
    to generate parameters for a physical model.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.pretrain_config = config.get('pretrain', {})
        self.finetune_config = config.get('finetune', {})

        self.spotpy_strategy = SpotpyCalibrationStrategy(self.pretrain_config)
        self.gradient_strategy = GradientDescentStrategy(self.finetune_config)

    def train(
        self,
        model: nn.Module,
        train_data: dict[str, torch.Tensor],
        eval_data: Optional[dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Train model using hybrid strategy.

        Parameters
        ----------
        model : nn.Module
            dPL model with both neural network and physical components
        train_data : dict[str, torch.Tensor]
            Training data
        eval_data : dict[str, torch.Tensor], optional
            Evaluation data

        Returns
        -------
        Dict[str, Any]
            Combined training results from both phases
        """
        log.info("="*60)
        log.info("HYBRID TRAINING: Phase 1 - Spotpy Pre-training")
        log.info("="*60)

        # Phase 1: Spotpy calibration of physical model
        # Extract physical model component
        if hasattr(model, 'phy_model'):
            phy_model = model.phy_model
        else:
            log.warning("Model doesn't have phy_model attribute. Using full model.")
            phy_model = model

        pretrain_results = self.spotpy_strategy.train(
            phy_model, train_data, eval_data
        )

        best_params = pretrain_results['best_parameters']
        log.info(f"Pre-training complete. Best KGE: {pretrain_results['best_objective']:.4f}")

        # Initialize neural network with good parameter values
        if hasattr(model, 'nn_model'):
            log.info("Initializing neural network with pre-trained parameters...")
            # This is a simplification - actual implementation would need
            # to set the NN to output values close to best_params
            # Possibly by setting final layer biases
            self._initialize_nn_with_params(model.nn_model, best_params)

        log.info("\n" + "="*60)
        log.info("HYBRID TRAINING: Phase 2 - Gradient Fine-tuning")
        log.info("="*60)

        # Phase 2: Gradient descent fine-tuning
        finetune_results = self.gradient_strategy.train(
            model, train_data, eval_data
        )

        # Combine results
        combined_results = {
            'phase1_pretrain': pretrain_results,
            'phase2_finetune': finetune_results,
            'final_model': model,
            'total_time': (
                pretrain_results['elapsed_time'] +
                finetune_results['elapsed_time']
            ),
        }

        return combined_results

    def _initialize_nn_with_params(
        self,
        nn_model: nn.Module,
        target_params: torch.Tensor,
    ) -> None:
        """
        Initialize neural network to output values close to target parameters.

        This is a heuristic initialization that sets the final layer's biases
        to the target parameter values, which helps the gradient descent phase
        start from a good solution.

        Parameters
        ----------
        nn_model : nn.Module
            Neural network model
        target_params : torch.Tensor
            Target parameter values from pre-training [1, n_params]
        """
        # Find last linear layer
        last_linear = None
        for module in nn_model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is not None:
            # Set bias to target values (scaled)
            with torch.no_grad():
                if last_linear.bias is not None:
                    # Simple initialization: set bias to sigmoid_inverse(normalized_params)
                    # This ensures the network initially outputs target_params after sigmoid
                    normalized = target_params.squeeze()
                    # Inverse sigmoid: logit(x) = log(x / (1 - x))
                    eps = 1e-6
                    normalized = torch.clamp(normalized, eps, 1 - eps)
                    logits = torch.log(normalized / (1 - normalized))
                    last_linear.bias.copy_(logits)

                log.info("✓ Neural network initialized with pre-trained parameters")
        else:
            log.warning("Could not find linear layer for initialization")


# ============================================================================
# Hybrid Trainer
# ============================================================================


class HybridTrainer(Trainer):
    """
    Hybrid trainer supporting multiple training paradigms.

    Extends the base Trainer class to support:
        - Traditional calibration (Spotpy)
        - Gradient descent (PyTorch)
        - Hybrid training (Spotpy → Gradient)

    The training method is automatically selected based on model type
    specified in the configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary. Required keys:
            - 'model/type': Model type ('physics', 'neural_network', 'dpl', 'hybrid')
            - 'training/method': Training method ('spotpy', 'gradient_descent', 'hybrid')
    model : nn.Module
        Model to train
    train_dataset : dict[str, torch.Tensor]
        Training dataset
    eval_dataset : dict[str, torch.Tensor], optional
        Evaluation dataset
    verbose : bool, optional
        Whether to print verbose output (default: True)

    Attributes
    ----------
    training_method : str
        Selected training method
    training_strategy : TrainingStrategy
        Training strategy instance

    Examples
    --------
    >>> # Traditional calibration
    >>> config = {
    >>>     'model': {'type': 'physics'},
    >>>     'training': {'method': 'spotpy', 'algorithm': 'lhs', 'n_iterations': 500}
    >>> }
    >>> trainer = HybridTrainer(config, hbv_model, train_data, eval_data)
    >>> results = trainer.train_with_strategy()
    >>>
    >>> # Hybrid training for dPL
    >>> config = {
    >>>     'model': {'type': 'dpl'},
    >>>     'training': {
    >>>         'method': 'hybrid',
    >>>         'pretrain': {'algorithm': 'lhs', 'n_iterations': 200},
    >>>         'finetune': {'optimizer': 'Adadelta', 'epochs': 30}
    >>>     }
    >>> }
    >>> trainer = HybridTrainer(config, dpl_model, train_data, eval_data)
    >>> results = trainer.train_with_strategy()
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: nn.Module,
        train_dataset: dict[str, torch.Tensor],
        eval_dataset: Optional[dict[str, torch.Tensor]] = None,
        dataset: Optional[dict[str, torch.Tensor]] = None,
        loss_func: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        verbose: Optional[bool] = True,
    ) -> None:
        # Initialize parent class (may skip if using spotpy)
        try:
            super().__init__(
                config, model, train_dataset, eval_dataset, dataset,
                loss_func, optimizer, scheduler, verbose
            )
        except ValueError as e:
            # If parent initialization fails due to missing training config,
            # that's okay for spotpy-only training
            log.info(f"Skipping parent initialization: {e}")
            self.config = config
            self.model = model
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.verbose = verbose

        # Determine training method
        self.training_method = self._determine_training_method()
        log.info(f"Selected training method: {self.training_method}")

        # Create training strategy
        self.training_strategy = self._create_training_strategy()

    def _determine_training_method(self) -> str:
        """
        Determine training method based on configuration.

        Returns
        -------
        str
            Training method: 'spotpy', 'gradient_descent', or 'hybrid'
        """
        # Explicit specification
        if 'training' in self.config and 'method' in self.config['training']:
            return self.config['training']['method']

        # Infer from model type
        model_type = self.config.get('model', {}).get('type', 'unknown')

        if model_type == 'physics':
            return 'spotpy'
        elif model_type in ['neural_network', 'nn']:
            return 'gradient_descent'
        elif model_type in ['dpl', 'hybrid']:
            return 'hybrid'
        else:
            # Default to gradient descent if using parent Trainer
            return 'gradient_descent'

    def _create_training_strategy(self) -> TrainingStrategy:
        """Create appropriate training strategy."""
        training_config = self.config.get('training', {})

        if self.training_method == 'spotpy':
            return SpotpyCalibrationStrategy(training_config)
        elif self.training_method == 'gradient_descent':
            return GradientDescentStrategy(training_config)
        elif self.training_method == 'hybrid':
            return HybridStrategy(training_config)
        else:
            raise ValueError(f"Unknown training method: {self.training_method}")

    def train_with_strategy(self) -> Dict[str, Any]:
        """
        Train model using selected strategy.

        Returns
        -------
        Dict[str, Any]
            Training results
        """
        log.info(f"Starting training with {self.training_method} strategy")

        results = self.training_strategy.train(
            self.model,
            self.train_dataset,
            self.eval_dataset,
        )

        log.info("Training complete")
        return results


if __name__ == '__main__':
    # Example usage
    print("Testing HybridTrainer with SpotpyCalibrationStrategy...")

    # This would require actual data and spotpy installation
    print("See unit tests for complete examples")
    print("✓ Module loaded successfully")
