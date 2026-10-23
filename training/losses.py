import torch
import torch.nn.functional as F
def binary_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    labels = labels.float().view(-1)
    logits = logits.view(-1)
    pos = (labels > 0.5).float().sum().clamp_min(1.0)
    neg = labels.numel() - pos
    pos_weight = (neg / pos).detach()

    if loss_type in {"bce", "bce_pairwise"}:
        return F.binary_cross_entropy_with_logits(logits, labels)
    if loss_type == "weighted_bce":
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    if loss_type in {"focal", "weighted_focal"}:
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * labels + (1.0 - probs) * (1.0 - labels)
        focal_weight = (1.0 - pt).pow(focal_gamma)
        if loss_type == "weighted_focal":
            class_weight = torch.where(labels > 0.5, pos_weight, torch.ones_like(labels))
            focal_weight = focal_weight * class_weight
        return (focal_weight * bce).mean()
    if loss_type == "pairwise_auc":
        return logits.new_tensor(0.0)
    raise ValueError(f"Unknown loss type: {loss_type}")


def pairwise_auc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.0,
    max_pairs: int = 4096,
) -> torch.Tensor:
    labels = labels.float().view(-1)
    logits = logits.view(-1)
    pos_logits = logits[labels > 0.5]
    neg_logits = logits[labels <= 0.5]
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return logits.new_tensor(0.0)

    n_pairs = pos_logits.numel() * neg_logits.numel()
    if n_pairs <= max_pairs:
        diff = pos_logits[:, None] - neg_logits[None, :]
    else:
        pos_idx = torch.randint(pos_logits.numel(), (max_pairs,), device=logits.device)
        neg_idx = torch.randint(neg_logits.numel(), (max_pairs,), device=logits.device)
        diff = pos_logits[pos_idx] - neg_logits[neg_idx]
    return F.softplus(margin - diff).mean()

