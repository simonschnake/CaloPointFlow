import torch
from torch import Tensor

def shrink(x: Tensor, eps: float = 1e-6):
    """
    Shrinks the range of the input tensor to [eps, 1-eps].

    Arguments:
        x (Tensor): The input tensor.
        eps (float, optional): A small value that shrinks the range of the input
                               tensor to [eps, 1-eps]. Default is 1e-6.

    Returns:
        Tensor: The shrunk tensor.
    """
    torch.clamp_(x, min=0, max=1)
    return x * (1 - 2 * eps) + eps

def expand(x: Tensor, eps: float = 1e-6):
    """
    Expands the range of the input tensor to [0, 1].

    Arguments:
        x (Tensor): The input tensor.
        eps (float, optional): A small value that shrinks the range of the input
                               tensor to [eps, 1-eps]. Default is 1e-6.

    Returns:
        Tensor: The expanded tensor.
    """
    x = (x - eps) / (1 - 2 * eps)
    return torch.clamp_(x, min=0, max=1)


def normal_quantile(x: Tensor, eps=1e-6):
    """
    Applies the quantile function of the normal distribution (mu=0, sigma=1) to the
    input tensor.
    q(x): [0, 1] -> [-inf, inf]

    The quantile function is the inverse of the CDF.

    Arguments:
        x (Tensor): The input tensor.
        eps (float, optional): A small value that shrinks the range of the input
                               tensor to [eps, 1-eps]. Default is 1e-6.

    Returns:
        Tensor: The normal quantile-transformed tensor.
    """
    x =  shrink(x, eps)
    return torch.special.ndtri(x)


def normal_cdf(x: Tensor, eps=1e-6):
    """
    Applies the CDF of the normal distribution (mu=0, sigma=1) to the input tensor. 
    cdf(x): [-inf, inf] -> [0, 1] 

    The CDF is the inverse of the quantile function.

    Arguments:
        x (Tensor): The input tensor.
        eps (float, optional): A small value that shrinks the range of the input
                            tensor to [eps, 1-eps]. Default is 1e-6.

    Returns:
        Tensor: The normal CDF-transformed tensor.
    """
    x = torch.special.ndtr(x)
    return expand(x, eps)


def logistic_quantile(x: Tensor, eps=1e-6):
    """
    Applies the quantile function of the logistic distribution (mu=0, s=1) to the
    input tensor.
    q(x): [0, 1] -> [-inf, inf]

    The quantile function is the inverse of the CDF.

    Arguments:
        x (Tensor): The input tensor.
        eps (float, optional): A small value that shrinks the range of the input
                               tensor to [eps, 1-eps]. Default is 1e-6.

    Returns:
        Tensor: The logistic quantile-transformed tensor.
    """
    x = shrink(x, eps)
    return torch.logit(x)

def logistic_cdf(x: Tensor, eps=1e-6):
    """
    Applies the CDF of the logistic distribution (mu=0, s=1) to the input tensor. 
    cdf(x): [-inf, inf] -> [0, 1] 

    The CDF is the inverse of the quantile function.

    Arguments:
        x (Tensor): The input tensor.
        eps (float, optional): A small value that shrinks the range of the input
                            tensor to [eps, 1-eps]. Default is 1e-6.

    Returns:
        Tensor: The logistic CDF-transformed tensor.
    """
    x = torch.sigmoid(x)
    return expand(x, eps)