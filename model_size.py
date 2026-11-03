"""
model_size.py
--------------
Exact model storage size, used as the "model size" comparison metric
requested alongside accuracy. This is pure arithmetic over parameter
counts and assigned bitwidths -- no training or forward pass required, so
these numbers are exact regardless of what hardware runs the rest of the
pipeline.

Two things get counted separately, because they're quantized differently:
  - Quantizable layers (QuantConv2d / QuantLinear weights): stored at
    each layer's *assigned* bitwidth (from the mixed precision search),
    plus a small fixed per-layer overhead for the LVQ codebook itself
    (num_levels * vec_dim floats -- negligible next to the weight tensor
    it replaces, but included for exactness) or the LSQ step-size scalar.
  - Everything else (BatchNorm affine params + running stats, conv/linear
    biases if present): kept at fp32, since these are typically NOT
    quantized in mixed-precision pipelines (they're a tiny fraction of
    total parameters and quantizing them buys negligible extra
    compression for real accuracy risk).
"""
from dataclasses import dataclass, field
from typing import Dict

import torch.nn as nn


@dataclass
class ModelSizeReport:
    quant_layer_bits: Dict[str, int]
    quantized_weight_bytes: int
    codebook_overhead_bytes: int
    other_param_bytes: int      # BN, biases, etc. -- kept fp32
    fp32_equivalent_bytes: int  # what this exact architecture would cost at fp32 throughout

    @property
    def total_bytes(self) -> int:
        return self.quantized_weight_bytes + self.codebook_overhead_bytes + self.other_param_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)

    @property
    def fp32_equivalent_mb(self) -> float:
        return self.fp32_equivalent_bytes / (1024 ** 2)

    @property
    def compression_ratio(self) -> float:
        return self.fp32_equivalent_bytes / max(self.total_bytes, 1)

    @property
    def avg_weight_bits(self) -> float:
        n_layers = len(self.quant_layer_bits)
        return sum(self.quant_layer_bits.values()) / n_layers if n_layers else 32.0


def compute_model_size(model: nn.Module, use_lvq: bool = True) -> ModelSizeReport:
    """Walks the model, sums storage cost using each QuantConv2d/
    QuantLinear's *current* `weight_bits` (i.e. call this AFTER
    apply_mixed_precision has set the searched bitwidths)."""
    from quant_layers import QuantConv2d, QuantLinear  # local import avoids a cycle at module load time

    quant_layer_bits = {}
    quantized_weight_bytes = 0
    codebook_overhead_bytes = 0
    fp32_equivalent_bytes = 0
    accounted_param_ids = set()

    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            bits = int(m.weight_bits)
            n_weights = m.weight.numel()
            quant_layer_bits[name] = bits

            quantized_weight_bytes += (n_weights * bits + 7) // 8  # round up to whole bytes
            fp32_equivalent_bytes += n_weights * 4
            accounted_param_ids.add(id(m.weight))

            if use_lvq and bits < 32:
                # codebook: num_levels * vec_dim float32 entries
                codebook_overhead_bytes += m.lvq_w.codebook.numel() * 4
            elif bits < 32:
                codebook_overhead_bytes += 4  # single fp32 LSQ step-size scalar

            if m.bias is not None:
                fp32_equivalent_bytes += m.bias.numel() * 4
                accounted_param_ids.add(id(m.bias))

    other_param_bytes = 0
    for p in model.parameters():
        if id(p) not in accounted_param_ids:
            other_param_bytes += p.numel() * 4
            fp32_equivalent_bytes += p.numel() * 4
    for b in model.buffers():  # BatchNorm running_mean / running_var
        other_param_bytes += b.numel() * 4
        fp32_equivalent_bytes += b.numel() * 4

    return ModelSizeReport(
        quant_layer_bits=quant_layer_bits,
        quantized_weight_bytes=quantized_weight_bytes,
        codebook_overhead_bytes=codebook_overhead_bytes,
        other_param_bytes=other_param_bytes,
        fp32_equivalent_bytes=fp32_equivalent_bytes,
    )


def print_size_report(name: str, report: ModelSizeReport):
    print(f"[{name}] model size:")
    print(f"    fp32 baseline size:        {report.fp32_equivalent_mb:.3f} MB")
    print(f"    mixed-precision+LVQ size:  {report.total_mb:.3f} MB "
          f"(weights {report.quantized_weight_bytes/1024**2:.3f} MB + "
          f"codebooks {report.codebook_overhead_bytes/1024**2:.4f} MB + "
          f"BN/bias {report.other_param_bytes/1024**2:.3f} MB)")
    print(f"    compression ratio:         {report.compression_ratio:.2f}x")
    print(f"    average weight bitwidth:   {report.avg_weight_bits:.2f} bits")
