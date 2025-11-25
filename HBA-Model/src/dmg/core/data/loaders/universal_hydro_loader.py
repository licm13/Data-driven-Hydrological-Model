"""
Universal hydrological data loader supporting multiple data formats and sampling strategies.

This module extends the base HydroLoader to support:
    - CAMELS dataset (NetCDF format)
    - CSV/TSV files (legacy format)
    - IMPRO ASCII format
    - Various sampling strategies (random, Douglas-Peucker, stratified)

The loader seamlessly integrates with the dMG framework while maintaining backward
compatibility with legacy experimental code.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from dmg.core.data.loaders.base import BaseLoader
from dmg.core.data.loaders.hydro_loader import HydroLoader

log = logging.getLogger(__name__)


# ============================================================================
# Sampling Strategies
# ============================================================================


class SamplingStrategy:
    """Base class for sampling strategies."""

    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_samples(
        self,
        n_total: int,
        sample_size: int,
        n_replicates: int,
    ) -> List[NDArray[np.int_]]:
        """
        Generate sampling indices.

        Parameters
        ----------
        n_total : int
            Total number of available samples
        sample_size : int
            Number of samples to draw
        n_replicates : int
            Number of independent replicates

        Returns
        -------
        List[NDArray[np.int_]]
            List of index arrays, one per replicate
        """
        raise NotImplementedError


class ConsecutiveRandomSampling(SamplingStrategy):
    """
    Consecutive random sampling strategy.

    Selects a random starting point and takes consecutive samples,
    maintaining temporal continuity. This is important for hydrological
    models where temporal dependencies matter.
    """

    def generate_samples(
        self,
        n_total: int,
        sample_size: int,
        n_replicates: int,
    ) -> List[NDArray[np.int_]]:
        """Generate consecutive random samples."""
        if sample_size > n_total:
            raise ValueError(
                f"sample_size ({sample_size}) cannot exceed n_total ({n_total})"
            )

        samples = []
        max_start = n_total - sample_size

        for _ in range(n_replicates):
            start_idx = self.rng.randint(0, max_start + 1)
            indices = np.arange(start_idx, start_idx + sample_size, dtype=np.int_)
            samples.append(indices)

        return samples


class DouglasP euckerSampling(SamplingStrategy):
    """
    Douglas-Peucker algorithm for information-driven sampling.

    Iteratively selects the most informative points by maximizing the
    perpendicular distance from the line segment connecting neighboring points.
    This preserves the overall shape of the time series while reducing sample size.

    References
    ----------
    Douglas, D. H., & Peucker, T. K. (1973). Algorithms for the reduction of the
    number of points required to represent a digitized line or its caricature.
    Cartographica: the international journal for geographic information and
    geovisualization, 10(2), 112-122.
    """

    def __init__(
        self,
        feature: str = 'discharge',
        normalize: bool = True,
        seed: Optional[int] = 42,
    ):
        super().__init__(seed)
        self.feature = feature
        self.normalize = normalize

    def generate_samples(
        self,
        n_total: int,
        sample_size: int,
        n_replicates: int,
        time_series: Optional[NDArray[np.float64]] = None,
    ) -> List[NDArray[np.int_]]:
        """
        Generate samples using Douglas-Peucker algorithm.

        Parameters
        ----------
        n_total : int
            Total number of points
        sample_size : int
            Target sample size
        n_replicates : int
            Number of replicates (Douglas-Peucker is deterministic, so replicates
            will be identical unless time_series varies)
        time_series : NDArray[np.float64], optional
            Time series data to use for sampling. If None, generates random data.

        Returns
        -------
        List[NDArray[np.int_]]
            List of sampled indices
        """
        if time_series is None:
            log.warning(
                "Douglas-Peucker requires time series data. "
                "Falling back to random sampling."
            )
            fallback = ConsecutiveRandomSampling(self.seed)
            return fallback.generate_samples(n_total, sample_size, n_replicates)

        # Normalize if requested
        data = time_series.copy()
        if self.normalize:
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)

        # Perform Douglas-Peucker simplification
        selected_indices = self._douglas_peucker(data, sample_size)

        # Replicate (deterministic algorithm produces same result)
        samples = [selected_indices for _ in range(n_replicates)]

        return samples

    def _douglas_peucker(
        self,
        data: NDArray[np.float64],
        target_size: int,
    ) -> NDArray[np.int_]:
        """
        Core Douglas-Peucker algorithm.

        Parameters
        ----------
        data : NDArray[np.float64]
            1D time series data
        target_size : int
            Target number of points to retain

        Returns
        -------
        NDArray[np.int_]
            Indices of selected points
        """
        n_points = len(data)
        if target_size >= n_points:
            return np.arange(n_points, dtype=np.int_)

        # Always keep first and last points
        selected = {0, n_points - 1}

        # Iteratively add points with maximum perpendicular distance
        while len(selected) < target_size:
            max_distance = -1
            max_idx = -1

            # Sort selected indices for segment iteration
            selected_sorted = sorted(selected)

            # Check all segments between consecutive selected points
            for i in range(len(selected_sorted) - 1):
                start_idx = selected_sorted[i]
                end_idx = selected_sorted[i + 1]

                # Skip if consecutive
                if end_idx - start_idx <= 1:
                    continue

                # Find point with maximum distance in this segment
                for idx in range(start_idx + 1, end_idx):
                    if idx in selected:
                        continue

                    distance = self._perpendicular_distance(
                        idx, start_idx, end_idx, data
                    )

                    if distance > max_distance:
                        max_distance = distance
                        max_idx = idx

            # Add point with maximum distance
            if max_idx != -1:
                selected.add(max_idx)
            else:
                break  # No more points to add

        return np.array(sorted(selected), dtype=np.int_)

    def _perpendicular_distance(
        self,
        point_idx: int,
        start_idx: int,
        end_idx: int,
        data: NDArray[np.float64],
    ) -> float:
        """
        Compute perpendicular distance from point to line segment.

        Parameters
        ----------
        point_idx : int
            Index of point to measure
        start_idx : int
            Index of segment start
        end_idx : int
            Index of segment end
        data : NDArray[np.float64]
            Time series data

        Returns
        -------
        float
            Perpendicular distance
        """
        # Point coordinates (time is normalized to [0, 1])
        x = (point_idx - start_idx) / (end_idx - start_idx)
        y = data[point_idx]

        # Line segment endpoints
        x1, y1 = 0.0, data[start_idx]
        x2, y2 = 1.0, data[end_idx]

        # Perpendicular distance formula
        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator


class StratifiedSampling(SamplingStrategy):
    """
    Stratified sampling based on discharge quantiles.

    Ensures representation across the full range of hydrological conditions
    by sampling proportionally from different flow regimes (low, medium, high).
    """

    def __init__(
        self,
        n_strata: int = 5,
        seed: Optional[int] = 42,
    ):
        super().__init__(seed)
        self.n_strata = n_strata

    def generate_samples(
        self,
        n_total: int,
        sample_size: int,
        n_replicates: int,
        time_series: Optional[NDArray[np.float64]] = None,
    ) -> List[NDArray[np.int_]]:
        """Generate stratified samples."""
        if time_series is None:
            log.warning("Stratified sampling requires time series data.")
            fallback = ConsecutiveRandomSampling(self.seed)
            return fallback.generate_samples(n_total, sample_size, n_replicates)

        # Create quantile-based strata
        quantiles = np.linspace(0, 1, self.n_strata + 1)
        thresholds = np.quantile(time_series, quantiles)

        # Assign each point to a stratum
        strata = np.digitize(time_series, thresholds) - 1
        strata = np.clip(strata, 0, self.n_strata - 1)

        samples = []
        for _ in range(n_replicates):
            selected_indices = []

            # Sample proportionally from each stratum
            for stratum_id in range(self.n_strata):
                stratum_indices = np.where(strata == stratum_id)[0]
                n_from_stratum = int(np.ceil(len(stratum_indices) / n_total * sample_size))
                n_from_stratum = min(n_from_stratum, len(stratum_indices))

                if n_from_stratum > 0:
                    sampled = self.rng.choice(
                        stratum_indices, size=n_from_stratum, replace=False
                    )
                    selected_indices.extend(sampled)

            # Ensure exact sample size
            selected_indices = np.array(selected_indices, dtype=np.int_)
            if len(selected_indices) > sample_size:
                selected_indices = self.rng.choice(
                    selected_indices, size=sample_size, replace=False
                )
            elif len(selected_indices) < sample_size:
                # Add random samples to reach target
                remaining = sample_size - len(selected_indices)
                available = np.setdiff1d(np.arange(n_total), selected_indices)
                additional = self.rng.choice(available, size=remaining, replace=False)
                selected_indices = np.concatenate([selected_indices, additional])

            selected_indices = np.sort(selected_indices)
            samples.append(selected_indices)

        return samples


# ============================================================================
# CSV/ASCII Loader Strategy
# ============================================================================


class CSVLoaderStrategy:
    """
    Strategy for loading CSV and ASCII formatted hydrological data.

    Integrates the legacy data loading utilities to work seamlessly
    with the dMG framework.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.data_dir = Path(config['observations'].get('data_dir', './data'))

    def load_data(
        self,
        catchment_names: List[str],
        period_start: str,
        period_end: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load CSV/ASCII data for specified catchments and time period.

        Parameters
        ----------
        catchment_names : List[str]
            List of catchment names to load
        period_start : str
            Start date (ISO format: 'YYYY-MM-DD')
        period_end : str
            End date (ISO format: 'YYYY-MM-DD')

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping catchment names to DataFrames with columns:
            ['date', 'precip', 'temp', 'pet', 'discharge']
        """
        # Import legacy loader
        import sys
        from pathlib import Path

        # Add legacy src to path
        legacy_src = Path(__file__).parents[6] / 'src'
        if str(legacy_src) not in sys.path:
            sys.path.insert(0, str(legacy_src))

        from utils.data_loader import load_catchment_from_csv

        catchment_data = {}

        for catchment_name in catchment_names:
            try:
                # Load using legacy loader
                data = load_catchment_from_csv(
                    str(self.data_dir), catchment_name
                )

                # Convert to DataFrame
                df = data.to_dataframe()

                # Filter by date range
                df = df.loc[period_start:period_end]

                catchment_data[catchment_name] = df

                log.info(
                    f"Loaded {len(df)} days of data for {catchment_name} "
                    f"({period_start} to {period_end})"
                )

            except Exception as e:
                log.error(f"Failed to load {catchment_name}: {e}")
                continue

        return catchment_data


# ============================================================================
# Universal Hydro Loader
# ============================================================================


class UniversalHydroLoader(BaseLoader):
    """
    Universal data loader supporting multiple formats and sampling strategies.

    This loader extends the capabilities of the base HydroLoader to support:
        - Multiple data formats (CAMELS, CSV, IMPRO)
        - Various sampling strategies (random, Douglas-Peucker, stratified)
        - Seamless integration with legacy experimental code
        - Configuration-driven data loading

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with keys:
            - observations/format: Data format ('camels', 'csv', 'impro')
            - observations/catchments: List of catchment names
            - observations/data_dir: Data directory path
            - sampling/strategy: Sampling strategy name
            - sampling/sample_sizes: List of sample sizes for experiments
            - sampling/n_replicates: Number of replicates per sample size
    test_split : bool, optional
        Whether to create train/test split (default: False)
    overwrite : bool, optional
        Whether to overwrite cached normalization statistics (default: False)
    holdout_index : int, optional
        Index for spatial holdout testing (default: None)

    Attributes
    ----------
    data_format : str
        Data format being used
    sampler : SamplingStrategy
        Sampling strategy instance

    Examples
    --------
    >>> config = {
    >>>     'observations': {
    >>>         'format': 'csv',
    >>>         'catchments': ['Iller', 'Saale'],
    >>>         'data_dir': './data/IMPRO',
    >>>     },
    >>>     'sampling': {
    >>>         'strategy': 'consecutive_random',
    >>>         'sample_sizes': [50, 100, 500],
    >>>         'n_replicates': 30,
    >>>         'seed': 42,
    >>>     },
    >>>     'train': {'target': ['discharge']},
    >>>     'device': 'cuda',
    >>>     'dtype': torch.float32,
    >>> }
    >>> loader = UniversalHydroLoader(config, test_split=True)
    >>> # Access training data
    >>> train_data = loader.train_dataset
    """

    def __init__(
        self,
        config: dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
        holdout_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite
        self.holdout_index = holdout_index

        # Extract configuration
        obs_config = config.get('observations', {})
        self.data_format = obs_config.get('format', 'camels')
        self.catchments = obs_config.get('catchments', [])
        self.data_dir = obs_config.get('data_dir', './data')

        # Sampling configuration
        sampling_config = config.get('sampling', {})
        self.sampling_strategy_name = sampling_config.get('strategy', 'consecutive_random')
        self.sample_sizes = sampling_config.get('sample_sizes', [])
        self.n_replicates = sampling_config.get('n_replicates', 30)
        self.seed = sampling_config.get('seed', 42)

        # Model configuration
        self.target = config['train']['target']
        self.device = config.get('device', 'cpu')
        self.dtype = config.get('dtype', torch.float32)

        # Initialize sampler
        self.sampler = self._create_sampler()

        # Storage
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.raw_data = {}  # Store raw data for sampling strategies

        # Load data
        self.load_dataset()

    def _create_sampler(self) -> SamplingStrategy:
        """Create sampling strategy based on configuration."""
        samplers = {
            'consecutive_random': ConsecutiveRandomSampling,
            'douglas_peucker': DouglasP euckerSampling,
            'stratified': StratifiedSampling,
        }

        sampler_cls = samplers.get(self.sampling_strategy_name)
        if sampler_cls is None:
            log.warning(
                f"Unknown sampling strategy '{self.sampling_strategy_name}'. "
                f"Falling back to 'consecutive_random'."
            )
            sampler_cls = ConsecutiveRandomSampling

        return sampler_cls(seed=self.seed)

    def load_dataset(self) -> None:
        """Load dataset based on format specified in config."""
        if self.data_format == 'camels':
            # Use parent class CAMELS loading
            self._load_camels_data()
        elif self.data_format in ['csv', 'impro']:
            # Use CSV/ASCII loading
            self._load_csv_data()
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")

    def _load_camels_data(self) -> None:
        """Load CAMELS data using parent HydroLoader."""
        # Temporarily instantiate parent loader
        parent_loader = HydroLoader(
            self.config,
            test_split=self.test_split,
            overwrite=self.overwrite,
            holdout_index=self.holdout_index,
        )

        # Copy datasets
        self.train_dataset = parent_loader.train_dataset
        self.eval_dataset = parent_loader.eval_dataset
        self.dataset = parent_loader.dataset
        self.norm_stats = parent_loader.norm_stats

    def _load_csv_data(self) -> None:
        """Load CSV/ASCII data."""
        loader_strategy = CSVLoaderStrategy(self.config)

        # Get time periods from config
        periods = self.config.get('periods', {})
        train_period = periods.get('train', {})
        test_period = periods.get('test', {})

        # Load training data
        if self.test_split or self.config.get('mode') == 'train':
            train_data = loader_strategy.load_data(
                self.catchments,
                train_period.get('start', '2001-01-01'),
                train_period.get('end', '2010-12-31'),
            )
            self.train_dataset = self._prepare_tensors(train_data, scope='train')
            self.raw_data['train'] = train_data

        # Load testing data
        if self.test_split or self.config.get('mode') == 'test':
            test_data = loader_strategy.load_data(
                self.catchments,
                test_period.get('start', '2012-01-01'),
                test_period.get('end', '2015-12-31'),
            )
            self.eval_dataset = self._prepare_tensors(test_data, scope='test')
            self.raw_data['test'] = test_data

    def _prepare_tensors(
        self,
        data_dict: Dict[str, pd.DataFrame],
        scope: str,
    ) -> dict[str, torch.Tensor]:
        """
        Convert DataFrame data to PyTorch tensors.

        Parameters
        ----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary of catchment DataFrames
        scope : str
            Data scope ('train' or 'test')

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of tensors compatible with dMG framework
        """
        # Concatenate all catchments
        all_dfs = []
        for catchment_name, df in data_dict.items():
            df_copy = df.copy()
            df_copy['catchment'] = catchment_name
            all_dfs.append(df_copy)

        combined_df = pd.concat(all_dfs, axis=0)

        # Extract arrays
        precip = combined_df['precip'].values
        temp = combined_df['temp'].values
        pet = combined_df['pet'].values
        discharge = combined_df['discharge'].values

        # Stack forcings [n_timesteps, n_basins=1, n_features=3]
        forcings = np.stack([precip, temp, pet], axis=-1)
        forcings = np.expand_dims(forcings, axis=1)  # Add basin dimension

        # Target [n_timesteps, n_basins=1, 1]
        target = np.expand_dims(discharge, axis=(1, 2))

        # Convert to tensors
        x_phy = torch.from_numpy(forcings).to(dtype=self.dtype, device=self.device)
        target_tensor = torch.from_numpy(target).to(dtype=self.dtype, device=self.device)

        # For simplicity, use same data for nn inputs (can be customized)
        x_nn = x_phy
        c_nn = torch.zeros(1, 1, dtype=self.dtype, device=self.device)  # Placeholder
        c_phy = torch.zeros(1, 1, dtype=self.dtype, device=self.device)  # Placeholder

        # Create normalized version (simple min-max for now)
        xc_nn_norm = (x_nn - x_nn.min()) / (x_nn.max() - x_nn.min() + 1e-8)

        dataset = {
            'x_phy': x_phy,
            'c_phy': c_phy,
            'x_nn': x_nn,
            'c_nn': c_nn,
            'xc_nn_norm': xc_nn_norm,
            'target': target_tensor,
        }

        return dataset

    def generate_learning_curve_samples(
        self,
        sample_size: int,
    ) -> List[NDArray[np.int_]]:
        """
        Generate sampling indices for learning curve experiments.

        Parameters
        ----------
        sample_size : int
            Number of samples to draw

        Returns
        -------
        List[NDArray[np.int_]]
            List of index arrays, one per replicate
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset not loaded")

        n_total = self.train_dataset['x_phy'].shape[0]

        # Get discharge time series for advanced sampling strategies
        discharge = self.train_dataset['target'][:, 0, 0].cpu().numpy()

        # Generate samples
        if isinstance(self.sampler, DouglasP euckerSampling):
            samples = self.sampler.generate_samples(
                n_total, sample_size, self.n_replicates, time_series=discharge
            )
        elif isinstance(self.sampler, StratifiedSampling):
            samples = self.sampler.generate_samples(
                n_total, sample_size, self.n_replicates, time_series=discharge
            )
        else:
            samples = self.sampler.generate_samples(
                n_total, sample_size, self.n_replicates
            )

        return samples

    def get_sampled_dataset(
        self,
        indices: NDArray[np.int_],
        dataset_type: str = 'train',
    ) -> dict[str, torch.Tensor]:
        """
        Extract a subset of data based on indices.

        Parameters
        ----------
        indices : NDArray[np.int_]
            Array of indices to extract
        dataset_type : str, optional
            Which dataset to sample from ('train' or 'test')

        Returns
        -------
        dict[str, torch.Tensor]
            Sampled dataset dictionary
        """
        source = self.train_dataset if dataset_type == 'train' else self.eval_dataset

        if source is None:
            raise ValueError(f"{dataset_type} dataset not loaded")

        sampled = {}
        for key, tensor in source.items():
            if tensor.dim() >= 1:
                sampled[key] = tensor[indices]
            else:
                sampled[key] = tensor

        return sampled


if __name__ == '__main__':
    # Example usage
    print("Testing UniversalHydroLoader...")

    config = {
        'observations': {
            'format': 'csv',
            'catchments': ['Iller'],
            'data_dir': './Dataset/IMPRO_catchment_data_infotheo',
        },
        'periods': {
            'train': {'start': '2001-01-01', 'end': '2010-12-31'},
            'test': {'start': '2012-01-01', 'end': '2015-12-31'},
        },
        'sampling': {
            'strategy': 'consecutive_random',
            'sample_sizes': [50, 100, 500],
            'n_replicates': 3,
            'seed': 42,
        },
        'train': {'target': ['discharge']},
        'device': 'cpu',
        'dtype': torch.float32,
        'mode': 'train_test',
    }

    try:
        loader = UniversalHydroLoader(config, test_split=True)
        print(f"✓ Loader initialized")
        print(f"  Data format: {loader.data_format}")
        print(f"  Sampling strategy: {loader.sampling_strategy_name}")

        if loader.train_dataset:
            print(f"  Training data shape: {loader.train_dataset['x_phy'].shape}")

        # Test sampling
        samples = loader.generate_learning_curve_samples(sample_size=50)
        print(f"✓ Generated {len(samples)} sample replicates of size 50")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
