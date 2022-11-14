"The base flow class."

from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.flows import base
from nflows.nn.nets import ResidualNet
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import \
    PiecewiseRationalQuadraticCouplingTransform
from nflows.utils.torchutils import create_random_binary_mask
from torch import nn


class Flow(base.Flow):
    def __init__(
        self,
        features: int,
        n_transforms: int,
        tail_bound: float,
        num_bins: int,
        context_features: int,
        context_hidden_features: int,
        hidden_features: int,
        num_blocks: int,
        ) -> None:
        
        self.hidden_features= hidden_features
        self.context_features= context_features
        self.context_hidden_features = context_hidden_features
        self.num_blocks= num_blocks


        transforms = []

        for _ in range(n_transforms):
            mask = create_random_binary_mask(features)
            transforms.append(
                PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=self.create_resnet,
                    num_bins=num_bins,
                    tails='linear'
                ))
            
            transforms.append(
                PiecewiseRationalQuadraticCouplingTransform(
                    mask=(1 - mask),
                    transform_net_create_fn=self.create_resnet,
                    num_bins=num_bins,
                    tails='linear',
                    tail_bound=tail_bound
                ))

        transform = CompositeTransform(transforms)

        distribution_context_encoding = nn.Sequential(
            nn.Linear(context_features, context_hidden_features),
            nn.LeakyReLU(inplace=True),
            nn.Linear(context_hidden_features, 2 * features))
        
        distribution=ConditionalDiagonalNormal(
            shape=[features], context_encoder=distribution_context_encoding)
        
        super().__init__(transform, distribution)

    def create_resnet(self, in_features, out_features):
        '''This is the network that outputs the parameters of the invertible transformation
        The only arguments can be the in dimension and the out dimenson, the structure
        of the network is defined over the config which is a class attribute
        Context Features: Amount of features used to condition the flow - in our case 
        this is usually the mass
        num_blocks: How many Resnet blocks should be used, one res net block is are 1 input+ 2 layers
        and an additive skip connection from the first to the third'''
        return ResidualNet(
            in_features,
            out_features,
            hidden_features=self.hidden_features,
            context_features=self.context_features,
            num_blocks=self.num_blocks,
        )