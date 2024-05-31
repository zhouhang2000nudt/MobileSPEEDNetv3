from torch.distributions import Categorical
# from scipy.stats import entropy
import torch
from torch import Tensor

def entropy(p: Tensor, q: Tensor, base: float = torch.tensor(torch.e)):
    return torch.sum(p * torch.log(p / (q) + 1e-9) / torch.log(base), dim=1)

a = torch.tensor([[0.0, 1.0]])
b = torch.tensor([[1.0, 0]])

print(entropy(a, b))