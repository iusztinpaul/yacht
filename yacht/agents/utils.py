from typing import List, Dict

import torch

from yacht.data.datasets import DayForecastDataset


def unflatten_observations(observations: torch.Tensor, intervals: List[str]) -> Dict[str, torch.Tensor]:
    """
        @param observations: flatten observation from dict to a single tensor
        @param intervals: frequency intervals that are used. It is essential to be in the same order as the intervals
                used when the data was flattened

        returns: unflattened data in the form of a dictionary: [key]: [batch, window, feature, bars)
    """
    # Flattened observations have the current data:
    # (batch, window, bar_1d + bar_i + bar_i+1 + ... + bar_n + env_features, bar_features)
    # env_features are tiled along the others dimensions to be concatenated, but they have a global value so it is
    # save to be taken only once from a random window and features

    batch_size = observations.shape[0]
    window_size = observations.shape[1]
    features_size = observations.shape[3]

    unflattened_observation = dict()
    current_index = 0
    for interval in intervals:
        bar_units = DayForecastDataset.get_day_bar_units_for(interval)
        unflattened_observation[interval] = observations[:, :, current_index:current_index + bar_units, :]
        unflattened_observation[interval] = unflattened_observation[interval].reshape(
            batch_size, window_size, -1, features_size
        )
        unflattened_observation[interval] = torch.transpose(unflattened_observation[interval], 2, 3)

        current_index += bar_units

    unflattened_observation['env_features'] = observations[..., 0, current_index:, 0]

    return unflattened_observation
