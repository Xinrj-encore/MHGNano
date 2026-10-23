import torch.nn as nn
from .meta_train import MetaTrainingMixin
from .evaluation import MetaEvaluationMixin
from .model import MHGNano


class MetaLearner(nn.Module, MetaTrainingMixin, MetaEvaluationMixin):
    def __init__(
        self,
        base_model: MHGNano,
        inner_lr: float = 1e-2,
        inner_steps: int = 5,
        meta_lr: float = 1e-4,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        pairwise_lambda: float = 0.2,
        pairwise_margin: float = 0.0,
        pairwise_max_pairs: int = 4096,
    ):
        super().__init__()
        self.model = base_model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.pairwise_lambda = pairwise_lambda
        self.pairwise_margin = pairwise_margin
        self.pairwise_max_pairs = pairwise_max_pairs
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        # 这里只做占位，真正 loss 我们自己用 BCEWithLogits + 动态 pos_weight
        self.criterion = nn.BCEWithLogitsLoss()
