"""
quantizers.py
-------------
Uniform, learnable-step fake quantization (LSQ-style: Esser et al. 2020,
"Learned Step Size Quantization"). This is the "default" per-layer
quantizer used for both weights and activations. It is deliberately kept
separate from the LVQ vector quantizer in lvq.py -- a layer can use either
scheme (or LVQ for weights + LSQ for activations, which is the setup used
in pipeline.py).

Bitwidth is a *runtime-settable integer* on each quantizer, so the mixed
precision search (bitwidth_search.py) can assign a different bitwidth to
every layer after the fact, without rebuilding the model.
"""
import torch
import torch.nn as nn
 

def _grad_scale(x, scale):
    # Gradient-scaling trick from the LSQ paper so that the step-size
    # parameter learns at a well-conditioned rate regardless of tensor size.
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def _round_ste(x):
    # Straight-through estimator for rounding.
    return (x.round() - x).detach() + x


class LSQQuantizer(nn.Module):
    """Learned-step-size uniform fake quantizer.

    Supports both signed (weights) and unsigned (post-ReLU activations)
    ranges. `bits` can be changed at any time (e.g. after the mixed
    precision search) via `set_bits`.
    """

    def __init__(self, bits: int = 8, signed: bool = True, per_channel: bool = False):
        super().__init__()
        self.signed = signed
        self.per_channel = per_channel
        self.bits = bits
        self.step = nn.Parameter(torch.tensor(1.0))
        self._initialized = False

    def set_bits(self, bits: int):
        assert bits >= 1
        self.bits = int(bits)

    def _qrange(self):
        if self.signed:
            qn = -(2 ** (self.bits - 1))
            qp = 2 ** (self.bits - 1) - 1
        else:
            qn = 0
            qp = 2 ** self.bits - 1
        return qn, qp

    @torch.no_grad()
    def _init_step(self, x):
        qn, qp = self._qrange()
        # Standard LSQ initialization: 2*mean(|x|)/sqrt(qp)
        init_val = 2.0 * x.detach().abs().mean() / (qp ** 0.5 + 1e-8)
        self.step.data.fill_(max(init_val.item(), 1e-6))
        self._initialized = True

    def forward(self, x):
        if self.bits >= 32:  # treat as "full precision", passthrough
            return x
        if not self._initialized:
            self._init_step(x)

        qn, qp = self._qrange()
        n_elems = x[0].numel() if x.dim() > 1 else x.numel()
        grad_scale_factor = 1.0 / ((qp * max(n_elems, 1)) ** 0.5)

        step = _grad_scale(self.step.abs().clamp(min=1e-6), grad_scale_factor)
        x_scaled = x / step
        x_clamped = torch.clamp(x_scaled, qn, qp)
        x_quant = _round_ste(x_clamped)
        return x_quant * step


class QuantAct(nn.Module):
    """Wraps activation quantization (unsigned, since it sits after ReLU)."""

    def __init__(self, bits: int = 8):
        super().__init__()
        self.quantizer = LSQQuantizer(bits=bits, signed=False)

    def set_bits(self, bits: int):
        self.quantizer.set_bits(bits)

    def forward(self, x):
        return self.quantizer(x)
