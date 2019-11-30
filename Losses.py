from torch import distributions as d
from torch.nn import functional as F


def reg_loss(x, y):
    target = y.float().unsqueeze(-1).to('cuda')
    return F.mse_loss(x, target)


def prob_loss(x, y):
    n = d.Normal(x[0][0], x[0][1])
    return -1 * n.log_prob(y)
