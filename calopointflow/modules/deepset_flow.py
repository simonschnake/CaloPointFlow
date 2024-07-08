import torch
from torch import Tensor, Size, LongTensor
from typing import Sequence, Callable
from functools import partial

from zuko.flows import GeneralCouplingTransform, Unconditional, LazyDistribution, LazyComposedTransform
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.transforms import Transform, CouplingTransform, \
    MonotonicRQSTransform, MonotonicAffineTransform, ComposedTransform
from zuko.nn import MLP
from zuko.utils import broadcast

from .deep_sets import DeepSetModule

class LazyComposedSetTransform(LazyComposedTransform):
    def forward(self, idx: LongTensor, nnz: LongTensor, c: Tensor) -> Transform:
        return ComposedTransform(*(t(idx, nnz, c) for t in self.transforms))

class DeepSetTransform(GeneralCouplingTransform):
    def __init__(
            self, 
            features: int,
            context: int = 0,
            mask: torch.Tensor = None,
            univariate: Callable[..., Transform] = MonotonicAffineTransform,
            shapes: Sequence[Size] = ((), ()),
            point_net_dims : Sequence[int] = (10, 32, 64, 128),
            reduce_net_dims : Sequence[int] = (64, 32, 16, 8),
            reduce: str = "mean",
            **kwargs) -> None:

        super().__init__(
            features=features,
            context=context + reduce_net_dims[-1],
            mask=mask,
            univariate=univariate,
            shapes=shapes,
            **kwargs,
        )

        features_a = self.mask.sum().item()
        self.deep_set = DeepSetModule(
            MLP(
                in_features=features_a, # + context,
                out_features=point_net_dims[-1],
                hidden_features=point_net_dims[:-1]
            ),
            MLP(
                in_features=point_net_dims[-1],
                out_features=reduce_net_dims[-1],
                hidden_features=reduce_net_dims[:-1]
            ),
            reduce=reduce
        )

    def meta(self, idx: LongTensor, nnz: LongTensor, c: Tensor, x: Tensor) -> Tensor:

        xc = self.deep_set(x, idx)
        xc = torch.repeat_interleave(xc, nnz, dim=-2)
        c = torch.cat(broadcast(xc, c, ignore=1), dim=-1)  

        return super().meta(c, x)

    def forward(self, idx: LongTensor, nnz: LongTensor, c: Tensor) -> Tensor:
        return CouplingTransform(partial(self.meta, idx, nnz, c), self.mask)


class DeepSetFlow(LazyDistribution):
    def __init__(
            self, 
            features: int,
            context: int = 0,
            bins: int = 8,
            transforms: int = 3,
            randmask: bool = False,
            **kwargs) -> None:

        super().__init__()
        temp = []

        for i in range(transforms):
            if randmask:
                mask = torch.randperm(features) % 2 == i % 2
            else:
                mask = torch.arange(features) % 2 == i % 2

            temp.append(
                DeepSetTransform(
                    features=features,
                    context=context,
                    mask=mask,
                    univariate=MonotonicRQSTransform,
                    shapes=[(bins,), (bins,), (bins -1,)],
                    **kwargs,
                )
            )

        self.base = Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        self.transform = LazyComposedSetTransform(*temp)
       

    def forward(self, idx: LongTensor, nnz : LongTensor, 
                c: Tensor = None) -> NormalizingFlow:
        transform = self.transform(idx, nnz, c)

        if c is None:
            base = self.base(c)
        else:
            base = self.base(c).expand(c.shape[:-1]) 

        return NormalizingFlow(transform, base)