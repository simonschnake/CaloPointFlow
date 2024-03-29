import torch
from torch import LongTensor, FloatTensor, nn, Size
import numpy as np


from calopointflow.freezables import Freezable
from calopointflow.utils import logistic_cdf, logistic_quantile, normal_cdf, normal_quantile

class Dequantization(Freezable):
    """Dequantize the data by:
        - adding uniform noise to each element of the tensor.
        - min-max scaling the tensor to the range [0, 1].
        - logit transform the tensor.
    The inverse of this function is:
        - sigmoid transform the tensor.
        - min-max scale the tensor back to the range [min, max].
        - round the tensor to the nearest integer.
    """

    def __init__(self, size: Size | tuple[int, ...] | int, eps: float = 1e-6) -> None:
        super().__init__()

        self.register_buffer("size", torch.tensor(size))
        self.eps = eps

    def forward(self, x : LongTensor) -> FloatTensor:
        """
        Dequantize the input tensor.

        Parameters:
            x (LongTensor): The input tensor.

        Returns:
            FloatTensor: The output tensor after applying the function.
        """

        x = x.float()
        x += torch.rand_like(x)
        x = x / self.size 
        return x

    def inverse(self, x : FloatTensor) -> LongTensor:
        """
        Quantize the input tensor.

        Args:
            x (FloatTensor): The input tensor.

        Returns:
            LongTensor: The inverse of the input tensor, where each element is 
                        rounded down to the nearest integer.
        """
        x = x * self.size
        x = torch.floor(x)
        return x.int()

    def _update(self, x: LongTensor) -> None:
        """
        Update the maximum and minimum values with the input tensor.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            None
        """
        x_max = torch.max(x, dim=0).values
        x_min = torch.min(x, dim=0).values

        self.max = torch.max(self.max, x_max)
        self.min = torch.min(self.min, x_min)


class CDFDequantization(Freezable):
    """Dequantize the data.
    Adds uniform noise to each element of the tensor.
    """
    def __init__(self, size: Size | tuple[int, ...] | int, eps: float = 1e-6) -> None:
        super().__init__()
        self.size = size
        self.eps = eps
        
        for i, s in enumerate(self.size):
            self.register_buffer(f'pdf_{i}', torch.zeros(s))
            self.register_buffer(f'cdf_{i}', torch.zeros(s))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: LongTensor) -> FloatTensor:
        """
        Normalize the input tensor.

        Args:
            x (Tensor): The input tensor to normalize.

        Returns:
            Tensor: The normalized input tensor.
        """
        if not self.frozen:
            # Update the mean and variance with the input tensor
            self._update(x)

        y = torch.empty(x.size(), dtype=torch.float32, device=x.device)

        for i in range(len(self.size)):
            y[:, i] = self.cdf[i][x[:, i]] 
            y[:, i] += self.pdf[i][x[:, i]] * torch.rand_like(y[:, i])

        return y 
        
    def inverse(self, y : FloatTensor) -> LongTensor:
        x = torch.empty(y.size(), dtype=torch.long, device=y.device)
        for i in range(len(self.size)):
            x[:, i] = torch.searchsorted(self.cdf[i], y[:, i].contiguous()) - 1
        torch.clamp_min_(x, 0) # there is a extremely small chance that y is less than 0
        return x

    @property
    def pdf(self):
        return [getattr(self, f"pdf_{i}") for i in range(len(self.size))]

    @property
    def cdf(self):
        return [getattr(self, f"cdf_{i}") for i in range(len(self.size))]

    def _update(self, x: LongTensor) -> None:
        """
        Update the pdf and cdf.
        """
        self._update_pdf(x)
        self._calculate_cdf()
    
    def _update_pdf(self, x: LongTensor) -> None:
        """
        Update the pdf.
        """
        new_count = x.size(0)
        for i, s in enumerate(self.size):
            hist = torch.bincount(x[:, i], minlength=s)
            self.pdf[i] *= (self.count / (self.count + new_count))
            self.pdf[i] += hist / (new_count + self.count)
        
        self.count += new_count
    
    def _calculate_cdf(self) -> None:
        for i in range(len(self.size)):
            cspdf = torch.cumsum(self.pdf[i], dim=0)
            self.cdf[i][1:] = cspdf[:-1]

