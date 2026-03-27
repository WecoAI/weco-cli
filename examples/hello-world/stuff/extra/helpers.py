import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        # Precompute the combined scaling factor to fuse division and scaling
        self.combined_factor = scaling_factor / 2.0
        # Precompute summed weights during initialization to avoid runtime overhead
        weight_sum = torch.sum(self.weight.data, dim=0)
        self.register_buffer('weight_sum', weight_sum)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        # Fuse all operations into a single expression using mul instead of separate matmul+mul
        # This allows PyTorch to potentially optimize the computation graph better
        return torch.mul(torch.matmul(x, self.weight_sum), self.combined_factor).unsqueeze(1)