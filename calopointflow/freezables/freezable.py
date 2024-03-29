from abc import ABC, abstractmethod
from torch import Tensor, nn
import torch


class Freezable(nn.Module, ABC):
    def __init__(self):
        super(Freezable, self).__init__()

        self.register_buffer('_frozen', torch.tensor(False))

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def inverse(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def _update(self, x: Tensor) -> None:
        raise NotImplementedError

    @property
    def frozen(self) -> bool:
        return self._frozen.item()

    def freeze(self) -> None:
        self._frozen = torch.tensor(True)

    