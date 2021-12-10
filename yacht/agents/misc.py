from typing import List, Dict

import torch

from yacht.data.datasets import DayMultiFrequencyDataset


def unflatten_observations(
        observations: torch.Tensor,
        intervals: List[str],
        num_env_features: int,
        num_assets: int,
        include_weekends: bool
) -> Dict[str, torch.Tensor]:
    """
        @param observations: flatten observation from dict to a single tensor
        @param intervals: frequency intervals that are used. It is essential to be in the same order as the intervals
                used when the data was flattened
        @param num_env_features: the number of env_features that are at the 'feature' level
        @param num_assets: the number of assets that observed for every frequency.
        @param include_weekends: if include_weekends=False the data is within the trading ours,
                otherwise is traded all the time

        returns: unflattened data in the form of a dictionary: [key]: [batch, window, bars, assets, features]
    """
    # Flattened observations have the current data:
    # (batch, window, bars_1d + bars_1h + ..., price features + env features)
    # env_features are tiled along the bar dimension to be concatenated, but they have a global value at the
    # window level, so it is safe to be taken only once at any bar within the window.

    # Add +1 because of the padding meta information.
    num_env_features += 1

    unflattened_observation = dict()
    current_index = 0
    for interval in intervals:
        bar_units = DayMultiFrequencyDataset.get_day_bar_units_for(interval, include_weekends=include_weekends)
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
    # Remove extra padding.
    padding_size = int(unflattened_observation['env_features'][0, 0, -1])
    if padding_size > 0:
        unflattened_observation['env_features'] = unflattened_observation['env_features'][:, :-padding_size, :-1]
    else:
        unflattened_observation['env_features'] = unflattened_observation['env_features'][..., :-1]

    return unflattened_observation
