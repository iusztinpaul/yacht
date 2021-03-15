import torch

from torch import nn
import torch.nn.functional as F


class EIIEDense(nn.Module):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            kernel_width: int,
            name='EIIEDense'
    ):
        super(EIIEDense, self).__init__()
        self.name = name

        self.conv_2d = nn.Conv2d(
            input_channels,
            output_channels,
            stride=(1, 1),
            kernel_size=(1, kernel_width),
            padding=(0, 0),
        )

    def forward(self, input_tensor):
        return F.relu(self.conv_2d(input_tensor))


class EIIEOutputWithW(nn.Module):
    def __init__(
            self,
            num_assets: int,
            name='EIIEOutputWithW'
    ):
        super().__init__()
        self.name = name

        self.conv_2d = nn.Conv2d(
            num_assets,
            1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )

    def forward(self, input_tensor, previous_w):
        batch, features, num_assets, width = input_tensor.shape

        input_tensor = input_tensor.reshape(batch, width * features, num_assets, 1)
        previous_w = previous_w.reshape(batch, 1, num_assets, 1)

        tensor = torch.cat([input_tensor, previous_w], dim=1)
        tensor = self.conv_2d(tensor)
        tensor = tensor.reshape(batch, num_assets)

        # FIXME: Is this really ok ?
        btc_bias = torch.zeros(size=(batch, 1)).to(tensor.device)
        tensor = torch.cat([btc_bias, tensor], dim=1)

        tensor = F.softmax(tensor, dim=1)

        return tensor


class EIIECNN(nn.Module):
    def __init__(
            self,
            num_features: int,
            num_assets: int,
            window_size: int,
            kernel_size=(1, 2),
            name='EIIE_CNN'
    ):
        super().__init__()
        self.name = name

        self.conv_2d = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(0, 0)
        )
        self.eiie_dense = EIIEDense(
            input_channels=num_features,
            output_channels=num_assets - 1,
            kernel_width=window_size - kernel_size[1] + 1
        )
        self.eiie_output_with_w = EIIEOutputWithW(
            num_assets=num_assets
        )

    def forward(self, input_tensor, previous_w):
        batch, features, assets, window = input_tensor.shape

        last_window_assets_closing_price = input_tensor[:, 0, :, -1]
        last_window_assets_closing_price = last_window_assets_closing_price.reshape(batch, 1, assets, 1)

        tensor = input_tensor / last_window_assets_closing_price

        tensor = self.conv_2d(tensor)
        tensor = self.eiie_dense(tensor)
        tensor = self.eiie_output_with_w(tensor, previous_w)

        return tensor


if __name__ == '__main__':
    BATCH_SIZE = 100
    NUM_FEATURES = 3
    NUM_ASSETS = 11
    WINDOW_SIZE = 31

    eiie_module = EIIECNN(
        num_features=NUM_FEATURES,
        num_assets=NUM_ASSETS,
        window_size=WINDOW_SIZE,
    ).to('cuda')

    result = eiie_module(
        torch.rand((BATCH_SIZE, NUM_FEATURES, NUM_ASSETS, WINDOW_SIZE)).to('cuda'),
        torch.rand((BATCH_SIZE, NUM_ASSETS)).to('cuda')
    )
    print(result)
