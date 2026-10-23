from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg import Hypergraph
from layers import Concentration
class HGNNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.0,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        # 这里只用 LayerNorm 代替 BN，更适合元学习
        self.bn = nn.LayerNorm(out_channels) if use_bn else None
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.drop = nn.Dropout(drop_rate)
        self.act = nn.ReLU(inplace=False)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.drop(self.act(X))
        return X

    def functional_forward(self, X: torch.Tensor, hg: Hypergraph, params: dict) -> torch.Tensor:
        # 1) Linear
        W = params["theta.weight"]
        b = params.get("theta.bias", None)
        X = F.linear(X, W, b)

        # 2) LayerNorm (如果启用)
        if self.bn is not None:
            w_bn = params.get("bn.weight", self.bn.weight)
            b_bn = params.get("bn.bias", self.bn.bias)
            X = F.layer_norm(
                X,
                self.bn.normalized_shape,
                weight=w_bn,
                bias=b_bn,
                eps=self.bn.eps,
            )

        # 3) Hypergraph smoothing
        X = hg.smoothing_with_HGNN(X)

        # 4) 非最后一层加 ReLU + Dropout
        if not self.is_last:
            X = F.relu(X, inplace=False)
            X = F.dropout(X, p=self.drop.p, training=True)

        return X


class MHGNano(nn.Module):
    def __init__(
        self,
        in_channels: int,   # 320
        hid_channels: int,  # 256
        num_classes: int,   # 128
        num_hyperedges: int,
        use_bn: bool = False,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        # 这里直接用 LayerNorm + Dropout = True
        self.layer1 = HGNNConv(in_channels, hid_channels, use_bn=True, drop_rate=0.2)
        self.layer2 = HGNNConv(hid_channels, num_classes, use_bn=True, drop_rate=0.2, is_last=True)

        # 注意：你原来是 Concentration(hid_channels, hid_channels)，这里不改维度假设，
        # 仅保持与你现有的 Concentration 定义兼容。
        self.concentration = Concentration(hid_channels, hid_channels)

        # 分类器：输入 512（假设 Concentration 输出 512 维，这里沿用你的写法）
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def encode(self, x, hg):
        hidden1 = self.layer1(x, hg)
        return self.layer2(hidden1, hg)

    def forward(self, X, hg, GP_info, phase: str = "train"):
        X = self.encode(X, hg)
        concen_Z = self.concentration(X, GP_info)
        out = self.classifier(concen_Z)
        return out  # logits

    def functional_forward(self, X, hg, hg_index, params: Optional[Dict[str, torch.Tensor]] = None):
        if params is None:
            return self.forward(X, hg, hg_index)

        def p(name):
            return params[name]

        # layer1
        if hasattr(self.layer1, "functional_forward"):
            x = self.layer1.functional_forward(
                X,
                hg,
                {
                    "theta.weight": p("layer1.theta.weight"),
                    "theta.bias": p("layer1.theta.bias"),
                    "bn.weight": p("layer1.bn.weight"),
                    "bn.bias": p("layer1.bn.bias"),
                },
            )
        else:
            x = self.layer1(X, hg)

        # layer2
        if hasattr(self.layer2, "functional_forward"):
            x = self.layer2.functional_forward(
                x,
                hg,
                {
                    "theta.weight": p("layer2.theta.weight"),
                    "theta.bias": p("layer2.theta.bias"),
                    "bn.weight": p("layer2.bn.weight"),
                    "bn.bias": p("layer2.bn.bias"),
                },
            )
        else:
            x = self.layer2(x, hg)

        # Concentration
        if hasattr(self.concentration, "functional_forward"):
            z = self.concentration.functional_forward(x, hg_index)
        else:
            z = self.concentration(x, hg_index)

        # Classifier: 和 forward 的 nn.Sequential 完全对齐
        h = F.linear(z, p("classifier.0.weight"), p("classifier.0.bias"))
        h = F.relu(h)
        h = F.linear(h, p("classifier.2.weight"), p("classifier.2.bias"))
        h = F.relu(h)
        out = F.linear(h, p("classifier.4.weight"), p("classifier.4.bias"))
        return out

