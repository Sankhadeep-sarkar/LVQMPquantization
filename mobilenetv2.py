"""
mobilenetv2.py
--------------
CIFAR-10-adapted MobileNetV2 (Sandler et al., "MobileNetV2: Inverted
Residuals and Linear Bottlenecks"), built from the same QuantConv2d /
QuantLinear / QuantAct primitives as resnet20.py, so it plugs into the
exact same sensitivity -> bitwidth-search -> mixed-precision + LVQ
pipeline unchanged.

CIFAR adaptation (standard practice, matches the common "MobileNetV2 for
CIFAR" variants used in the quantization/pruning literature): the
ImageNet stem uses stride 2 (224x224 -> 112x112 immediately); for 32x32
CIFAR inputs that would throw away too much spatial resolution before the
network has done any work, so the stem stride is changed to 1, and the
first inverted-residual stage's stride is also kept at 1. Everything else
(expansion ratios, channel widths, inverted-residual block structure,
depthwise separable convs) follows the original paper's config table.

Depthwise convs are QuantConv2d with groups == in_channels == out_channels
(mixed precision search treats them as their own layer, same as any other
conv -- depthwise layers tend to be the most sensitive layers in
MobileNet-family nets, which the Hessian-trace sensitivity analysis will
surface on its own).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_layers import QuantConv2d, QuantLinear, QuantAct, conv1x1

# (expansion_ratio t, output_channels c, num_blocks n, stride s)
# -- identical to the original MobileNetV2 paper's Table 2, except the
# first stage's stride is 1 instead of 2 to suit 32x32 inputs (see module
# docstring).
_INVERTED_RESIDUAL_CFG = [
    (1, 16, 1, 1),
    (6, 24, 2, 1),   # stride 2 -> 1: keep spatial res for CIFAR
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


def quant_dwconv3x3(channels, stride=1):
    return QuantConv2d(channels, channels, kernel_size=3, stride=stride,
                        padding=1, groups=channels, bias=False)


def quant_conv3x3(in_c, out_c, stride=1):
    return QuantConv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()
        assert stride in (1, 2)
        hidden_c = int(round(in_c * expand_ratio))
        self.use_res_connect = stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            layers += [conv1x1(in_c, hidden_c), nn.BatchNorm2d(hidden_c)]
        self.expand_conv = nn.Sequential(*layers) if layers else None
        self.act_expand = QuantAct(bits=32) if layers else None

        self.dw_conv = quant_dwconv3x3(hidden_c, stride)
        self.dw_bn = nn.BatchNorm2d(hidden_c)
        self.act_dw = QuantAct(bits=32)

        self.pw_conv = conv1x1(hidden_c, out_c)
        self.pw_bn = nn.BatchNorm2d(out_c)
        # NOTE: no activation quantizer after the final pointwise conv --
        # MobileNetV2's "linear bottleneck" deliberately has no
        # nonlinearity here, and we don't fake-quantize the bottleneck
        # output either, matching the paper's design rationale (a ReLU
        # there would destroy information in the low-dimensional
        # bottleneck; the same reasoning argues against clipping it into
        # a coarse quantization grid).

    def forward(self, x):
        out = x
        if self.expand_conv is not None:
            out = F.relu6(self.expand_conv(out))
            out = self.act_expand(out)
        out = F.relu6(self.dw_bn(self.dw_conv(out)))
        out = self.act_dw(out)
        out = self.pw_bn(self.pw_conv(out))
        if self.use_res_connect:
            out = out + x
        return out

    def quant_convs(self):
        """Every QuantConv2d inside this block (handy for per-block
        introspection; named_modules() already finds these too)."""
        convs = []
        if self.expand_conv is not None:
            convs.append(self.expand_conv[0])
        convs.append(self.dw_conv)
        convs.append(self.pw_conv)
        return convs


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        input_c = int(32 * width_mult)
        last_c = int(1280 * max(1.0, width_mult))

        self.stem = quant_conv3x3(3, input_c, stride=1)  # stride 1: CIFAR adaptation
        self.stem_bn = nn.BatchNorm2d(input_c)
        self.act_stem = QuantAct(bits=32)

        blocks = []
        in_c = input_c
        for t, c, n, s in _INVERTED_RESIDUAL_CFG:
            out_c = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(in_c, out_c, stride, t))
                in_c = out_c
        self.blocks = nn.Sequential(*blocks)

        self.head_conv = conv1x1(in_c, last_c)
        self.head_bn = nn.BatchNorm2d(last_c)
        self.act_head = QuantAct(bits=32)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = QuantLinear(last_c, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu6(self.stem_bn(self.stem(x)))
        out = self.act_stem(out)
        out = self.blocks(out)
        out = F.relu6(self.head_bn(self.head_conv(out)))
        out = self.act_head(out)
        out = self.pool(out).flatten(1)
        out = self.dropout(out)
        return self.fc(out)

    def quantizable_layers(self):
        layers = {}
        for name, m in self.named_modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                layers[name] = m
        return layers

    def activation_pairs(self):
        pairs = [(self.stem, self.act_stem), (self.head_conv, self.act_head)]
        for block in self.modules():
            if isinstance(block, InvertedResidual):
                if block.expand_conv is not None:
                    pairs.append((block.expand_conv[0], block.act_expand))
                pairs.append((block.dw_conv, block.act_dw))
                # pw_conv (the linear bottleneck output) intentionally has
                # no activation quantizer -- see InvertedResidual docstring.
        return pairs


def mobilenetv2(num_classes=10, width_mult=1.0):
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult)


if __name__ == "__main__":
    model = mobilenetv2()
    layers = model.quantizable_layers()
    print(f"quantizable layers: {len(layers)}")
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("output shape:", y.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params:,}")
