import torch
import torch.nn as nn
import torch.nn.functional as F


''' swish '''
# No official version for pytorch yet
# Official version for tensorflow: tf.nn.swish()
#
# used as function:
def swish(x):
    return x * F.sigmoid(x)

# used as class:
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


''' Mish '''
# Official version: https://github.com/digantamisra98/Mish
#
# Simple version
# used as function:
def mish(x):
    return x * torch.tanh(F.softplus(x))

# used as class:
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

