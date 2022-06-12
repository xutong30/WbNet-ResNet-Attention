import torch
import torch.nn as nn

from functools import partial


# self attention mechanism
class SelfAttentionBlock(nn.Module):
    """
        Implementation of Self attention Block according to paper
        Self-Attention Generative Adversarial Networks (https://arxiv.org/abs/1805.08318)
    """

    def __init__(self, in_feature_maps):
        super(SelfAttentionBlock, self).__init__()

        self.in_feature_maps = in_feature_maps
        # get query(f), key(g), value(h) weights matrices
        self.conv_f = nn.Conv1d(in_channels=in_feature_maps, out_channels=in_feature_maps // 8, kernel_size=1,
                                padding=0)
        self.conv_g = nn.Conv1d(in_channels=in_feature_maps, out_channels=in_feature_maps // 8, kernel_size=1,
                                padding=0)
        self.conv_h = nn.Conv1d(in_channels=in_feature_maps, out_channels=in_feature_maps, kernel_size=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, C, width, height = input.size()
        N = width * height
        x = input.view(batch_size, -1, N)

        # get query(f), key(g), value(h) matrices
        f = self.conv_f(x)
        g = self.conv_g(x)
        h = self.conv_h(x)

        # calculates the attention score
        s = torch.bmm(f.permute(0, 2, 1), g)
        beta = self.softmax(s)
        o = torch.bmm(h, beta)
        o = o.view((batch_size, C, width, height))

        return self.gamma * o + input


"""
    Modified from Implementing ResNet in Pytorch
    Author: Francesco Saverio Zuppichini
    License: https://github.com/FrancescoSaverioZuppichini/ResNet/blob/master/LICENSE
    Code URL: https://github.com/FrancescoSaverioZuppichini/ResNet/blob/master/ResNet.ipynb
"""


class Conv2dAuto(nn.Conv2d):
    # basic convolutional layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


class ResidualBlock(nn.Module):
    # build a residual block that has basic information of shortcut and convolutional layer
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    # build ResNet Residual Block based on basic ResidualBlock function with assigning shortcut features - a convolutional layer followed by batch normalizaiton
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)

            })) if self.should_apply_shortcut else None

    # expand dimensions
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    # to check whether the dimension changes between input and output so to decide whether applying expanded_channels
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


from collections import OrderedDict


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    # batch normalization
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm2d(out_channels)}))


class ResNetBasicBlock(ResNetResidualBlock):
    # Implemented ResNet Blocks that has 2 convolutional layers with the shortcut
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetLayer(nn.Module):
    # ResNet Layer that stacks several residual blocks followed by a self-attention mechanism
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )
        #         add attention
        self.self_attention_block = SelfAttentionBlock(in_feature_maps=out_channels)

    def forward(self, x):
        x = self.blocks(x)
        #   add self-attention mechanism here
        x = self.self_attention_block(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    # Implement the encoder part, which is the feature extraction part in the paper.
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2, 2, 2, 2],
                 activation=nn.ReLU, block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        # At the beginning, connect the input features into the model
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # get each block's block information - input/output dimensions
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))

        # stack all ResNet Layer to gain the structure of Feature Extractor
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    # This is the classification part of my model
    def __init__(self, in_features, n_classes):
        super().__init__()
        # average pooling at first
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # set drop out and softmax function. also limit the output channel to 6, as the same as total species that we are classifying
        self.decoder = nn.Sequential(nn.Linear(in_features, 128),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(128, n_classes),
                                     nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResAtt(nn.Module):
    # The ResNet-Attention (WbNet) model combines both encoder part (Feature Extractor) and decoder part (Classification).
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# define our ResNet-18 based model with self-attention mechanism - WbNet
def resnet18_attention(in_channels, n_classes):
    return ResAtt(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])