import numpy as np
import torch.nn as nn


class BiLinearDeconvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(BiLinearDeconvolution, self).__init__()

        self.weight = self.make_bilinear_weight(in_channels, out_channels, kernel_size, groups)
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, input):
        self.weight = self.weight.to(input.device)
        output = nn.functional.conv_transpose2d(input, self.weight, stride=self.stride, padding=self.padding,
                                                groups=self.groups)
        return output

    @staticmethod
    def make_bilinear_weight(in_channels, out_channels, kernel_size, groups):
        factor = (kernel_size + 1) // 2

        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        grid = np.ogrid[:kernel_size, :kernel_size]
        filter = (1 - abs(grid[0] - center) / factor) * \
                 (1 - abs(grid[1] - center) / factor)

        filter = torch.from_numpy(filter.astype(np.float32))
        weight = filter.repeat(in_channels, out_channels // groups, 1, 1)

        return weight
