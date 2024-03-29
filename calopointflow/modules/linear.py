from torch import nn, Tensor

class Linear(nn.Module):
    """
    A basic layer
    """
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            activation_fn: str = None,
            normalization: str = None,
            **kwargs ) -> None:
        super().__init__()

        self.layer = nn.Linear(in_features, out_features)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the layer with the Glorot initialization.
        """
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer(x)
        return x

