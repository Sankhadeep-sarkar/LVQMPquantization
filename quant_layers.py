"""
quant_layers.py
----------------
Shared quantization-ready building blocks used by every architecture in
this project (resnet20.py, mobilenetv2.py, inceptionv3.py). Extracted out
of resnet20.py so new architectures don't have to import from a file named
after a different model.

QuantConv2d / QuantLinear are drop-in replacements for nn.Conv2d /
nn.Linear whose weight quantization scheme (full precision / uniform LSQ /
LVQ codebook) and bitwidth can be changed at runtime via
`set_weight_bits(bits, use_lvq=...)` -- exactly what the mixed-precision
pipeline needs after the sensitivity + bitwidth search step.
"""
import torch.nn as nn
import torch.nn.functional as F
 
from quantizers import LSQQuantizer, QuantAct
from lvq import LVQVectorQuantizer

__all__ = ["QuantConv2d", "QuantLinear", "QuantAct", "conv3x3", "conv1x1"]


class _QuantWeightMixin:
    """Shared bitwidth / quantization-scheme plumbing for conv & linear."""

    def _setup_quant(self, weight_bits, lvq_vec_dim):
        self.weight_bits = weight_bits
        self.use_lvq = False
        self.lsq_w = LSQQuantizer(bits=weight_bits, signed=True)
        # weight_bits defaults to 32 (== "full precision, don't quantize"
        # sentinel, see _quantized_weight below). The LVQ codebook is only
        # ever meaningfully sized once the mixed-precision search assigns
        # a real bitwidth via set_weight_bits(), so the *initial*
        # placeholder codebook must NOT use 2**weight_bits directly here
        # -- 2**32 rows would try to allocate ~64GB and crash. Cap the
        # placeholder at 8 bits (256 levels); it gets resized to the
        # correct size later in set_weight_bits().
        init_bits = min(weight_bits, 8)
        self.lvq_w = LVQVectorQuantizer(num_levels=max(2 ** init_bits, 2), vec_dim=lvq_vec_dim)

    def set_weight_bits(self, bits: int, use_lvq: bool = False):
        self.weight_bits = int(bits)
        self.use_lvq = use_lvq
        self.lsq_w.set_bits(bits)
        if use_lvq:
            # Defensive cap: LVQ codebooks are only sensible for the small
            # bitwidths mixed precision actually searches over. Guard
            # against ever calling this with the 32-bit "full precision"
            # sentinel (or any other oversized value), which would try to
            # allocate 2**bits codebook rows.
            safe_bits = min(int(bits), 16)
            self.lvq_w.set_num_levels(max(2 ** safe_bits, 2))

    def _quantized_weight(self):
        if self.weight_bits >= 32:
            return self.weight
        if self.use_lvq:
            return self.lvq_w(self.weight)
        return self.lsq_w(self.weight)


class QuantConv2d(nn.Conv2d, _QuantWeightMixin):
    def __init__(self, *args, weight_bits=32, lvq_vec_dim=4, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_quant(weight_bits, lvq_vec_dim)

    def forward(self, x):
        w = self._quantized_weight()
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear, _QuantWeightMixin):
    def __init__(self, *args, weight_bits=32, lvq_vec_dim=4, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_quant(weight_bits, lvq_vec_dim)

    def forward(self, x):
        w = self._quantized_weight()
        return F.linear(x, w, self.bias)


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    return QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, groups=groups, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
