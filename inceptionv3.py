"""
inceptionv3.py
--------------
CIFAR-10-adapted Inception-v3-style network with memory optimizations
for quantized training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

from quant_layers import QuantConv2d, QuantLinear, QuantAct


class BasicConv(nn.Module):
    """conv -> BN -> ReLU -> activation-quantizer with memory optimization"""

    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = QuantConv2d(in_c, out_c, kernel_size=kernel_size,
                                 stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = QuantAct(bits=32)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.act(out)
        return out


class InceptionA(nn.Module):
    """1x1 branch + 5x5-factorized-as-two-3x3 branch + 3x3 branch + pool"""

    def __init__(self, in_c, pool_c):
        super().__init__()
        self.b1 = BasicConv(in_c, 48, kernel_size=1)

        self.b2_1 = BasicConv(in_c, 32, kernel_size=1)
        self.b2_2 = BasicConv(32, 48, kernel_size=3, padding=1)

        self.b3_1 = BasicConv(in_c, 48, kernel_size=1)
        self.b3_2 = BasicConv(48, 64, kernel_size=3, padding=1)
        self.b3_3 = BasicConv(64, 64, kernel_size=3, padding=1)

        self.pool_proj = BasicConv(in_c, pool_c, kernel_size=1)

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_3(self.b3_2(self.b3_1(x)))
        bp = self.pool_proj(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b2, b3, bp], dim=1)


class ReductionA(nn.Module):
    """Grid-size reduction: 32x32 -> 16x16"""

    def __init__(self, in_c):
        super().__init__()
        self.b1 = BasicConv(in_c, 96, kernel_size=3, stride=2, padding=1)

        self.b2_1 = BasicConv(in_c, 48, kernel_size=1)
        self.b2_2 = BasicConv(48, 64, kernel_size=3, padding=1)
        self.b2_3 = BasicConv(64, 64, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2_3(self.b2_2(self.b2_1(x)))
        bp = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return torch.cat([b1, b2, bp], dim=1)


class InceptionB(nn.Module):
    """Asymmetric-factorized module with 1xn and nx1 convolutions"""

    def __init__(self, in_c, mid_c, out_1x1=96, out_branch=96, pool_c=96):
        super().__init__()
        self.b1 = BasicConv(in_c, out_1x1, kernel_size=1)

        self.b2_1 = BasicConv(in_c, mid_c, kernel_size=1)
        self.b2_2 = BasicConv(mid_c, mid_c, kernel_size=(1, 7), padding=(0, 3))
        self.b2_3 = BasicConv(mid_c, out_branch, kernel_size=(7, 1), padding=(3, 0))

        self.b3_1 = BasicConv(in_c, mid_c, kernel_size=1)
        self.b3_2 = BasicConv(mid_c, mid_c, kernel_size=(7, 1), padding=(3, 0))
        self.b3_3 = BasicConv(mid_c, mid_c, kernel_size=(1, 7), padding=(0, 3))
        self.b3_4 = BasicConv(mid_c, mid_c, kernel_size=(7, 1), padding=(3, 0))
        self.b3_5 = BasicConv(mid_c, out_branch, kernel_size=(1, 7), padding=(0, 3))

        self.pool_proj = BasicConv(in_c, pool_c, kernel_size=1)

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2_3(self.b2_2(self.b2_1(x)))
        b3 = self.b3_5(self.b3_4(self.b3_3(self.b3_2(self.b3_1(x)))))
        bp = self.pool_proj(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b2, b3, bp], dim=1)


class ReductionB(nn.Module):
    """Grid-size reduction: 16x16 -> 8x8"""

    def __init__(self, in_c):
        super().__init__()
        self.b1_1 = BasicConv(in_c, 64, kernel_size=1)
        self.b1_2 = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)

        self.b2_1 = BasicConv(in_c, 64, kernel_size=1)
        self.b2_2 = BasicConv(64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.b2_3 = BasicConv(64, 64, kernel_size=(7, 1), padding=(3, 0))
        self.b2_4 = BasicConv(64, 64, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        b1 = self.b1_2(self.b1_1(x))
        b2 = self.b2_4(self.b2_3(self.b2_2(self.b2_1(x))))
        bp = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return torch.cat([b1, b2, bp], dim=1)


class InceptionC(nn.Module):
    """Expanded-filter-bank module with parallel 1x3/3x1 branches"""

    def __init__(self, in_c, pool_c=96):
        super().__init__()
        self.b1 = BasicConv(in_c, 96, kernel_size=1)

        self.b2_1 = BasicConv(in_c, 128, kernel_size=1)
        self.b2_2a = BasicConv(128, 96, kernel_size=(1, 3), padding=(0, 1))
        self.b2_2b = BasicConv(128, 96, kernel_size=(3, 1), padding=(1, 0))

        self.b3_1 = BasicConv(in_c, 128, kernel_size=1)
        self.b3_2 = BasicConv(128, 144, kernel_size=3, padding=1)
        self.b3_3a = BasicConv(144, 96, kernel_size=(1, 3), padding=(0, 1))
        self.b3_3b = BasicConv(144, 96, kernel_size=(3, 1), padding=(1, 0))

        self.pool_proj = BasicConv(in_c, pool_c, kernel_size=1)

    def forward(self, x):
        b1 = self.b1(x)

        b2_mid = self.b2_1(x)
        b2 = torch.cat([self.b2_2a(b2_mid), self.b2_2b(b2_mid)], dim=1)

        b3_mid = self.b3_2(self.b3_1(x))
        b3 = torch.cat([self.b3_3a(b3_mid), self.b3_3b(b3_mid)], dim=1)

        bp = self.pool_proj(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b2, b3, bp], dim=1)


class InceptionV3(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Mild stem: keeps full 32x32 resolution
        self.stem = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=3, padding=1),
            BasicConv(32, 64, kernel_size=3, padding=1),
        )

        self.inception_a1 = InceptionA(64, pool_c=32)     # -> 48+48+64+32 = 192
        self.inception_a2 = InceptionA(192, pool_c=64)    # -> 48+48+64+64 = 224
        self.reduction_a = ReductionA(224)                # -> 96+64+224 = 384, 32x32 -> 16x16

        self.inception_b1 = InceptionB(384, mid_c=96)     # -> 96+96+96+96 = 384
        self.inception_b2 = InceptionB(384, mid_c=96)     # -> 384
        self.reduction_b = ReductionB(384)                # -> 96+64+384 = 544, 16x16 -> 8x8

        self.inception_c1 = InceptionC(544)               # -> 96+192+192+96 = 576
        self.inception_c2 = InceptionC(576)               # -> 576

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = QuantLinear(576, num_classes)

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
        out = self.stem(x)
        out = self.inception_a1(out)
        out = self.inception_a2(out)
        out = self.reduction_a(out)
        
        # Use gradient checkpointing for memory-intensive blocks
        if self.training:
            out = torch.utils.checkpoint.checkpoint(self.inception_b1, out, use_reentrant=False)
            out = torch.utils.checkpoint.checkpoint(self.inception_b2, out, use_reentrant=False)
            out = self.reduction_b(out)
            out = torch.utils.checkpoint.checkpoint(self.inception_c1, out, use_reentrant=False)
            out = torch.utils.checkpoint.checkpoint(self.inception_c2, out, use_reentrant=False)
        else:
            out = self.inception_b1(out)
            out = self.inception_b2(out)
            out = self.reduction_b(out)
            out = self.inception_c1(out)
            out = self.inception_c2(out)
        
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
        return [(m.conv, m.act) for m in self.modules() if isinstance(m, BasicConv)]


def inceptionv3(num_classes=10):
    return InceptionV3(num_classes=num_classes)


if __name__ == "__main__":
    model = inceptionv3()
    layers = model.quantizable_layers()
    print(f"quantizable layers: {len(layers)}")
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("output shape:", y.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params:,}")