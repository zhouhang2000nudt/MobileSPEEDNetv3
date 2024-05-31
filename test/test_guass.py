from rich import print
import numpy as np

def rho_sc(x, s, c):
    return np.exp(-(x - c)**2 / (2 * s**2))

s = 1
c = 9.5
tou = 3
x = np.arange(int(np.ceil(c - tou * s)), int(np.floor(c + tou * s)) + 1, 1)
# x = np.array([_ for _ in range(int(np.floor(c - tou * s)), int(np.ceil(c + tou * s)) + 1, 0.5)])
print(x)
p = rho_sc(x, s, c)
print(p)
print(p / p.sum())
print((p / p.sum()).sum())
print((p / p.sum() * x).sum())
