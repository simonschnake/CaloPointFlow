from zuko.flows import NICE
from zuko.transforms import MonotonicRQSTransform

class NeuralSplineCouplingFlow(NICE):
    r"""Creates a neural spline coupling flow.

    References:
        | Neural Spline Flows (Durkan et al., 2019)
        | https://arxiv.org/abs/1906.04032

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of coupling transformations.
        randmask: Whether random coupling masks are used or not. If :py:`False`,
            use alternating checkered masks.
        bins: The number of bins used for the spline.
        kwargs: Keyword arguments passed to :class:`GeneralCouplingTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randmask: bool = False,
        bins: int = 8,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            transforms=transforms,
            randmask=randmask,
            univariate=MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )