import torch


def mean_log_of_pv_vector(portfolio_value_vector: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.log(portfolio_value_vector))
