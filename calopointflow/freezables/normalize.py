
import torch
from torch import nn, Tensor, Size

from calopointflow.freezables import Freezable

class Normalize(Freezable):
    """Normalize tensors.
    
    This class implements an online normalizer that can be used to normalize
    data in a streaming fashion.

    Args:
        size (Size | tuple[int, ...] | int): The size of the input tensor.
    """

    def __init__(self, size: Size | tuple[int, ...] | int) ->  None:
        super().__init__()
        self.size = size
        self.register_buffer('mean', torch.zeros(self.size))
        self.register_buffer('var', torch.ones(self.size))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
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
        
        return (x - self.mean) / torch.sqrt(self.var)

    def inverse(self, y: Tensor) -> Tensor:
        """
        Inverse the normalization. 

        Args:
            y (Tensor): The input tensor to unnormalize.
        
        Returns:
            Tensor: The unnormalized input tensor.
        """
        return y * torch.sqrt(self.var) + self.mean
    
    def _update(self, x: Tensor) -> None:
        """
        Update the mean and variance.

        To update the mean and variance, the following formulas are used:

        :math:`\mu_{n+k} = \frac{n}{n+k}\mu_m + \frac{k}{n+k}\mu_k`

        where :math:`\mu_{n+k}` is the updated mean, :math:`\mu_m` is the mean 
        of the current batch, :math:`\mu_k` is the mean of the previous batches,

        :math:`\sigma^2_{n+k} = \frac{n}{n+k}\sigma^2_n + \frac{k}{n+k}\sigma^2_k + 
        \frac{nk}{(n+k)^2}(\mu_m - \mu_k)^2`

        where :math:`\sigma^2_{n+k}` is the updated variance, :math:`\sigma^2_m` is the 
        variance of the current batch, :math:`\sigma^2_k` is the variance of the
        previous batches.

        Args:
            x (Tensor): The input tensor to update the mean and variance with.
        """

        n = self.count
        k = x.size(0)

        # update mean
        mu_n = self.mean
        mu_k = x.mean(dim=0)
        self.mean = (n * mu_n + k * mu_k) / (n + k)

        # update variance
        var_n = self.var
        var_k = x.var(dim=0)
        self.var = (n * var_n + k * var_k + 
                    n * k * (mu_n - mu_k) ** 2 / (n + k)) / (n + k)

        self.count += k


        