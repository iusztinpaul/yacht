import numpy as np


class Normalizer:
    def __call__(self, prices: np.array):
        raise NotImplementedError()


class LastClosingPriceNormalizer:
    def __call__(self, prices: np.array):
        # The first element in the second dimension is the closing price for the `1d` interval.
        last_closing_price = prices[-1][0]
        prices = prices / last_closing_price

        return prices


normalizer_registry = {
    'LastClosingPriceNormalizer': LastClosingPriceNormalizer
}


def build_normalizer(input_config) -> Normalizer:
    normalizer_class = normalizer_registry[input_config.normalizer]

    return normalizer_class()
