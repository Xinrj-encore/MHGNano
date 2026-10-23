import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg.structure.hypergraphs import Hypergraph
_LAYER_UIDS = {}
from utils import try_gpu


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class HGNNConv(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:

        # Args:
        #     X (``torch.Tensor``):
        #     hg (``dhg.Hypergraph``)
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.drop(self.act(X))

        return X

    def functional_forward(self, X: torch.Tensor, hg: Hypergraph, params: dict) -> torch.Tensor:
        W = params['theta.weight']
        b = params.get('theta.bias', None)
        X = F.linear(X, W, b)

        if self.bn is not None:
            X = F.batch_norm(
                X,
                running_mean=self.bn.running_mean,
                running_var=self.bn.running_var,
                weight=params.get('bn.weight', self.bn.weight),
                bias=params.get('bn.bias', self.bn.bias),
                training=False
            )

        X = hg.smoothing_with_HGNN(X)

        if not self.is_last:
            X = F.relu(X, inplace=True)
            X = F.dropout(X, p=self.drop.p, training=self.training)

        return X

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, drop_rate, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.drop_rate = drop_rate
        self.act = act

    def forward(self, X, Z):
        X = F.dropout(X, self.drop_rate, training=self.training)
        Z = F.dropout(Z, self.drop_rate, training=self.training)
        H = self.act(torch.mm(X, Z.T))
        return H

class projection(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            hidden_size: int,
    ):
        super(projection, self).__init__()
        self.linear1 = nn.Linear(hidden_channels, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        return x

class Attention(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            hidden_size: int,
            num_hyperedges: int,
    ):
        super(Attention, self).__init__()
        self.projection = projection(hidden_channels, hidden_size)

    def forward(self, X, hg):
        global Z
        he = hg.state_dict['raw_groups']['main']
        edges = list(he.keys())
        edge_w = []
        for i in range(len(edges)):
            index = torch.tensor(list(edges[i][0]))
            index = index.to(device=try_gpu())
            x_he = torch.index_select(X, 0, index)
            w = self.projection(x_he)
            beta = torch.softmax(w, dim=0)
            edge_w.append(beta)
            z_batch = (beta * x_he).sum(0)
            z_batch = F.leaky_relu(z_batch)
            z_batch = z_batch.unsqueeze(0)
            if (i > 0):
                Z = torch.cat([Z, z_batch], 0)
            else:
                Z = z_batch
        return torch.tanh(Z), edge_w

# def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
#     cost = norm * F.binary_cross_entropy(preds, labels)
#
#     KLD = -0.5 / n_nodes * torch.mean(torch.sum(
#         1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
#     print(f"BCE = {cost:.4f}, KL = {KLD:.4f}")
#
#     return cost + KLD

# def loss_function(
#     preds, labels,
#     mu, logvar,
#     n_nodes, norm,
#     cls_logits=None, cls_labels=None,
#     pos_weight=None,alpha=1.0, beta=1.0
# ):
#
#     if pos_weight is not None:
#         cost = norm * F.binary_cross_entropy(preds, labels)
#     else:
#         cost = norm * F.binary_cross_entropy(preds, labels)
#
#
#     KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#
#     if cls_logits is not None and cls_labels is not None:
#         cls_loss = F.binary_cross_entropy(cls_logits.squeeze().cpu(), cls_labels.float())
#     else:
#         cls_loss = torch.tensor(0.0, device=preds.device)
#
#     total_loss = cost + beta * KLD + alpha * cls_loss
#     return total_loss, cost, cls_loss, KLD

def loss_function(cls_logits, cls_labels,
                  model,
                  use_focal=False,
                  focal_gamma=2.0):


    # loss_kl = -0.5 / n_nodes * torch.mean(torch.sum(
    #     1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), dim=1))

    cls_labels = cls_labels.view(-1)
    cls_logits = cls_logits.view(-1)

    num_pos = (cls_labels == 1).sum().float()
    num_neg = (cls_labels == 0).sum().float()
    pos_weight = num_neg / (num_pos + 1e-8)
    pos_weight = torch.clamp(pos_weight, max=100.0).to(cls_logits.device)

    # loss_struct = norm * F.binary_cross_entropy(preds, labels)

    if use_focal:
        prob = torch.sigmoid(cls_logits)
        bce = F.binary_cross_entropy(prob, cls_labels, reduction='none')
        pt = torch.where(cls_labels == 1, prob, 1 - prob)
        focal = (1 - pt) ** focal_gamma * bce
        loss_cls = focal.mean()
    else:
        loss_cls = F.binary_cross_entropy_with_logits(
            cls_logits.squeeze().cpu(), cls_labels)

    # loss_total = 0*loss_struct \
    #            + torch.exp(-model.log_sigma_kl) * loss_kl + model.log_sigma_kl*0 \
    #            + torch.exp(-model.log_sigma_cls) * loss_cls + model.log_sigma_cls

    return loss_cls

class Attention_PC(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            hidden_size: int,
    ):
        super(Attention_PC, self).__init__()
        self.projections = projection(hidden_channels, hidden_size)

    def forward(self, x1, x2):
        edge_w = []
        for i in range(x1.shape[0]):
            x_he1 = x1[i, :]
            x_he2 = x2[i, :]
            x_he1 = x_he1.unsqueeze(0)
            x_he2 = x_he2.unsqueeze(0)
            x_he = torch.cat([x_he1, x_he2], 0)
            w = self.projections(x_he)
            beta = torch.softmax(w, dim=0)
            edge_w.append(beta)
            z_batch = (beta * x_he)
            z_batch = z_batch.reshape(1, -1)
            z_batch = z_batch.unsqueeze(0)
            if (i > 0):
                Z = torch.cat([Z, z_batch], 0)
            else:
                Z = z_batch
        return torch.tanh(Z), edge_w

class Concentration(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            hidden_size: int,
    ):
        super(Concentration, self).__init__()

    def forward(self, X, GP_info):
        # edges = GP_info
        # Z_bath = torch.tensor(())
        # Z_bath = Z_bath.to(device=try_gpu())
        GP_info = torch.tensor(GP_info).to(X.device)
        idx1 = GP_info[:,0]
        idx2 = GP_info[:, 1]
        Z1 = X.index_select(0, idx1)
        Z2 = X.index_select(0, idx2)
        diff = torch.abs(Z1 - Z2)
        prob = Z1 * Z2
        Z_bath = torch.cat([Z1, Z2, diff, prob], dim=1)

        return Z_bath

    # def forward(self, X, GP_info):
    #     edges = GP_info
    #     Z_bath = torch.tensor(())
    #     Z_bath = Z_bath.to(device=try_gpu())
    #     for i in range(len(edges)):
    #         Z1 = X[edges[i][0]].unsqueeze(0)
    #         Z2 = X[edges[i][1]].unsqueeze(0)
    #         diff = torch.abs(Z1 - Z2)
    #         prob = Z1 * Z2
    #         # Z = torch.cat((Z1,Z2), dim=-1)
    #         Z = torch.cat((Z1, Z2, diff, prob), dim=1)
    #         Z_bath = torch.cat((Z_bath, Z), 0)
    #
    #     return Z_bath

    def functional_forward(self, X, GP_info):
        return self.forward(X, GP_info)

class Classifier_DNN(nn.Module):
    def __init__(
            self,
            in_features,
            out_features
    ):
        super(Classifier_DNN, self).__init__()
        in_features=200
        self.fc1 = nn.Linear(in_features, int(in_features / 1))
        self.bn1 = nn.BatchNorm1d(int(in_features / 1))

        self.fc2 = nn.Linear(int(in_features / 1), int(in_features / 2))
        self.bn2 = nn.BatchNorm1d(int(in_features / 2))

        self.fc3 = nn.Linear(int(in_features / 2), int(in_features / 4))
        self.bn3 = nn.BatchNorm1d(int(in_features / 4))

        self.fc5 = nn.Linear(int(in_features / 4), out_features)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.sigmoid(self.fc5(x))

        return x
