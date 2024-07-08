from torch import nn, Tensor, LongTensor
from torch_scatter import scatter


class DeepSetModule(nn.Module):
    def __init__(
            self,
            point_net: nn.Module,
            reduced_net: nn.Module,
            reduce: str = "sum",
    ):
        super().__init__()

        self.point_net = point_net
        self.reduce_net = reduced_net
        self.reduce = reduce

    def forward(self, x: Tensor, idx: LongTensor) -> Tensor:
        x = self.point_net(x)
        x = scatter(x, idx, dim=-2, reduce=self.reduce)
        x = self.reduce_net(x)
        return x