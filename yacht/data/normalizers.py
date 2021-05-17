from typing import Tuple

import numpy as np


class Normalizer:
    def __call__(self, prices: np.array):
        raise NotImplementedError()


class LastClosingPriceNormalizer(Normalizer):
    def __call__(self, prices: np.array):
        # window_size x concatenated_day_units x features
        # concatenated_day_units[0] = 1d unit
        # features[0] = closing price
        last_closing_price = prices[-1][0][0]
        prices = prices / last_closing_price

        return prices


class ZeroCenteredNormalizer(Normalizer):
    def __call__(self, other_features: np.array):
        _, features_size = other_features.shape
        mean = np.mean(other_features, axis=0).reshape(1, features_size)
        std = np.std(other_features, axis=0).reshape(1, features_size) + 10e-27
        other_features = (other_features - mean) / std

        return other_features


normalizer_registry = {
    'LastClosingPriceNormalizer': LastClosingPriceNormalizer,
    'ZeroCenteredNormalizer': ZeroCenteredNormalizer
}


def build_normalizer(normalizer_class_name: str) -> Normalizer:
    normalizer_class = normalizer_registry[normalizer_class_name]

    return normalizer_class()
