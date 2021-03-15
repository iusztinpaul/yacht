import torch

from torch import nn, Tensor

from agents.networks import layers
from agents.networks.losses import mean_log_of_pv_vector


class EIIENetwork(nn.Module):
    def __init__(
            self,
            num_features: int,
            num_assets: int,
            window_size: int,
            commission: float,
            name='EIIENetwork'
    ):
        super().__init__()
        self.name = name

        self.commission = commission

        self.eiie_layer = layers.EIIECNN(
            num_features=num_features,
            num_assets=num_assets,
            window_size=window_size
        )

    def forward(self, X, y, previous_w):
        new_w = self.eiie_layer(X, previous_w)

        if self.training:
            return self.compute_loss(new_w, y)
        else:
            return new_w

    def compute_loss(self, predicted_new_w, y):
        batch_num = y.shape[0]

        relative_price = torch.cat(
            [
                torch.ones(size=(batch_num, 1)).to(y.device),
                y[:, 0, :]
            ],
            dim=1
        )

        weights_movement = relative_price * predicted_new_w
        weights_movement_sum = torch.sum(weights_movement, dim=1)
        market_new_w = weights_movement / weights_movement_sum.reshape(batch_num, 1)

        pure_pc = self._compute_pute_pc(predicted_new_w, market_new_w)
        portfolio_value_vector = \
            weights_movement_sum * \
            torch.cat([torch.ones(size=(1, )).to(y.device), pure_pc], dim=0)

        self.compute_metrics(
            weights_movement_sum=weights_movement_sum,
            portfolio_value_vector=portfolio_value_vector
        )

        loss_function = self.build_loss_function()

        return loss_function(portfolio_value_vector)

    def compute_metrics(
            self,
            weights_movement_sum,
            portfolio_value_vector
    ):
        log_mean_free = torch.mean(torch.log(weights_movement_sum))
        portfolio_value = torch.prod(portfolio_value_vector)
        mean = torch.mean(portfolio_value)
        log_mean = torch.mean(torch.log(portfolio_value))
        standard_deviation = torch.sqrt(
            torch.mean(
                (portfolio_value_vector - mean) ** 2
            )
        )
        sharp_ratio = (mean - 1) / standard_deviation

        return {
            'log_mean_free': log_mean_free,
            'portfolio_value': portfolio_value,
            'mean': mean,
            'log_mean': log_mean,
            'standard_deviation': standard_deviation,
            'sharp_ratio': sharp_ratio
        }

    def _compute_pute_pc(self, predicted_new_w: Tensor, market_new_w: Tensor):
        # TODO: Why we slice the batch dim here ?
        market_new_w = market_new_w[:-1, ...]
        predicted_new_w = predicted_new_w[1:, ...]

        mu = 1 - torch.sum(torch.abs(predicted_new_w[:, 1:] - market_new_w[:, 1:]), dim=1) * self.commission

        return mu

    def build_loss_function(self):
        # TODO: Add configurable loss logic.
        return mean_log_of_pv_vector


if __name__ == '__main__':
    BATCH_SIZE = 100
    NUM_FEATURES = 3
    NUM_ASSETS = 11
    WINDOW_SIZE = 31

    eiie_network = EIIENetwork(
        num_features=NUM_FEATURES,
        num_assets=NUM_ASSETS,
        window_size=WINDOW_SIZE,
        commission=0.25 * 10**-2
    ).to('cuda')

    result = eiie_network(
        torch.rand((BATCH_SIZE, NUM_FEATURES, NUM_ASSETS, WINDOW_SIZE)).to('cuda'),
        torch.rand((BATCH_SIZE, NUM_FEATURES, NUM_ASSETS)).to('cuda'),
        torch.rand((BATCH_SIZE, NUM_ASSETS)).to('cuda')
    )
    print(result)
