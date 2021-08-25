from typing import List, Dict

import torch

from yacht.data.datasets import DayMultiFrequencyDataset


def unflatten_observations(
        observations: torch.Tensor,
        intervals: List[str],
        num_env_features: int,
        num_assets: int
) -> Dict[str, torch.Tensor]:
    """
        @param observations: flatten observation from dict to a single tensor
        @param intervals: frequency intervals that are used. It is essential to be in the same order as the intervals
                used when the data was flattened
        @param num_env_features: the number of env_features that are at the 'feature' level
        @param num_assets: the number of assets that observed for every frequency.

        returns: unflattened data in the form of a dictionary: [key]: [batch, window, feature, bars)
    """
    # Flattened observations have the current data:
    # (batch, window, bar_1d + bar_i + bar_i+1 + ... + bar_n + env_features, bar_features)
    # env_features are tiled along the others dimensions to be concatenated, but they have a global value so it is
    # save to be taken only once from a random window and features

    unflattened_observation = dict()
    current_index = 0
    for interval in intervals:
        bar_units = DayMultiFrequencyDataset.get_day_bar_units_for(interval)
        unflattened_observation[interval] = \
            observations[..., current_index:current_index + bar_units, :-num_env_features]

        # Add assets dimension.
        batch_shape, window_shape, bar_shape, features_shape = unflattened_observation[interval].shape
        unflattened_observation[interval] = unflattened_observation[interval].reshape(
            batch_shape,
            window_shape,
            bar_shape,
            num_assets,
            features_shape // num_assets
        )

        current_index += bar_units

    # Env features are the same for every bar. So take only the ones from index 0.
    unflattened_observation['env_features'] = observations[..., 0, -num_env_features:]

    return unflattened_observation
