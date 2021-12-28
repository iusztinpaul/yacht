import numpy as np


class MetaLabelingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.labels = self.compute_labels()

    def compute_labels(self) -> np.ndarray:
        # TODO: Adapt for sell execution.
        mean_price = self.compute_mean_price(
            start=self.start,
            end=self.end
        ).item()
        decision_prices = self.get_decision_prices()

        labels = decision_prices < mean_price
        labels = labels.astype(np.int32)

        return labels
