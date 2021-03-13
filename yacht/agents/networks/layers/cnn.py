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
            input_channels,
            name='EIIEOutputWithW'
    ):
        super().__init__()
        self.name = name

        self.conv_2d = nn.Conv2d(
            input_channels,
            1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )

    def forward(self, input_tensor, previous_w):
        batch, features, num_assets, width = input_tensor.shape

        input_tensor = input_tensor.reshape(batch, num_assets, 1, width * features)
        previous_w = previous_w.reshape(batch, num_assets, 1, 1)

        tensor = torch.cat([input_tensor, previous_w], dim=3)
        tensor = self.conv_2d(tensor)
        tensor = tensor.reshape(batch, num_assets)

        # FIXME: Is this really ok ?
        btc_bias = torch.zeros(size=(batch, 1)).to(tensor.device)
        tensor = torch.cat([btc_bias, tensor], dim=1)

        tensor = F.softmax(tensor)

        return tensor
