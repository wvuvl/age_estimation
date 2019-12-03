from torch import distributions as d
from torch.nn import functional as F
import torch as t
import math

def reg_loss(x, y):
    target = y.float().unsqueeze(-1).to('cuda')
    return F.mse_loss(x, target)


def prob_loss(x, y):
    m=x[:,0]
    s=x[:,1]
    s=t.sqrt(s.float()**2)
    inside_sqrt=t.tensor(2*math.pi)
    sqrt=t.sqrt(inside_sqrt)
    bottom=(2 *s.float()*sqrt)
    constant = (1 / bottom)
    term1=t.log(constant)
    exponent=-((m.float()- y.float())**2)/(2*s**2)
    term2=exponent
    full_term=term1+term2
    return -(full_term.mean())


# x=t.tensor([[-0.0772,  0.3068]])
# y=t.tensor([[25]])
#
# print(prob_loss(x,y))