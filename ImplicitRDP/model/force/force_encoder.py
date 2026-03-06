"""
Adapted from Adaptive Compliance Policy: https://github.com/yifan-hou/adaptive_compliance_policy
"""

import torch.nn as nn

def init_weights(modules):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class CausalConv1D(nn.Conv1d):
    """A causal 1D convolution.
  """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True
    ):
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        res = super().forward(x)
        if self.__padding != 0:
            return res[:, :, : -self.__padding]
        return res

class ForceEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[32, 64, 128]):
        """
        Force encoder taken from selfsupervised code
        """
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(CausalConv1D(input_dim, h_dim, kernel_size=2, stride=2))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            input_dim = h_dim
        layers.append(CausalConv1D(input_dim, output_dim, kernel_size=2, stride=2))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.frc_encoder = nn.Sequential(*layers)

        init_weights(self.modules())

    def forward(self, force):
        assert len(force.shape) == 4, f"Expected force to be of shape (B, N, T, D), but got {force.shape}"
        original_shape = force.shape
        force = force.reshape(-1, force.shape[2], force.shape[3]).transpose(1, 2)
        force_emb = self.frc_encoder(force)
        force_emb = force_emb.reshape(original_shape[0], original_shape[1], -1)
        return force_emb